"""
Model explainability module using SHAP for understanding predictions.
"""
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import os

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import EquipmentFailurePredictor
from config import MODEL_DIR

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Provides model explainability using SHAP values."""
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or str(MODEL_DIR)
        self.predictor = EquipmentFailurePredictor(self.model_dir)
        self.explainer = None
        self.is_loaded = False
        
    def load_models(self) -> bool:
        """Load models and initialize SHAP explainer."""
        if not self.predictor.load_models():
            logger.error("Failed to load prediction models")
            return False
        
        try:
            # Initialize SHAP explainer for Random Forest
            if 'random_forest' in self.predictor.models:
                self.explainer = shap.TreeExplainer(self.predictor.models['random_forest'])
                self.is_loaded = True
                logger.info("SHAP explainer initialized for Random Forest")
                return True
            else:
                logger.error("Random Forest model not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            return False
    
    def explain_prediction(self, df: pd.DataFrame, sample_idx: int = -1) -> Dict:
        """
        Explain a specific prediction using SHAP values.
        
        Args:
            df: Raw sensor data
            sample_idx: Index of sample to explain (-1 for latest)
            
        Returns:
            Dictionary with SHAP explanation
        """
        if not self.is_loaded:
            logger.error("Models not loaded. Call load_models() first.")
            return {}
        
        try:
            # Prepare features
            X = self.predictor.prepare_features(df)
            if X.empty:
                return {}
            
            # Get the specific sample
            if sample_idx == -1:
                sample_idx = len(X) - 1
            
            sample = X.iloc[[sample_idx]]
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(sample)
            
            # Get feature names
            feature_names = self.predictor.feature_columns
            
            # Prepare explanation
            explanation = {
                'sample_index': sample_idx,
                'feature_names': feature_names,
                'feature_values': sample.iloc[0].values.tolist(),
                'shap_values': shap_values[1].tolist() if len(shap_values) == 2 else shap_values.tolist(),
                'base_value': self.explainer.expected_value[1] if len(self.explainer.expected_value) == 2 else self.explainer.expected_value,
                'prediction': self.predictor.models['random_forest'].predict_proba(
                    self.predictor.models['scaler'].transform(sample)
                )[0].tolist()
            }
            
            logger.info(f"Generated SHAP explanation for sample {sample_idx}")
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanation: {e}")
            return {}
    
    def get_feature_importance(self, df: pd.DataFrame, top_k: int = 10) -> Dict:
        """
        Get feature importance for the latest prediction.
        
        Args:
            df: Raw sensor data
            top_k: Number of top features to return
            
        Returns:
            Dictionary with feature importance
        """
        explanation = self.explain_prediction(df, sample_idx=-1)
        
        if not explanation:
            return {}
        
        # Combine feature names, values, and SHAP values
        features = []
        for i, (name, value, shap_val) in enumerate(zip(
            explanation['feature_names'],
            explanation['feature_values'],
            explanation['shap_values']
        )):
            features.append({
                'feature': name,
                'value': value,
                'shap_value': shap_val,
                'abs_shap_value': abs(shap_val)
            })
        
        # Sort by absolute SHAP value
        features.sort(key=lambda x: x['abs_shap_value'], reverse=True)
        
        return {
            'top_features': features[:top_k],
            'total_features': len(features),
            'base_value': explanation['base_value'],
            'prediction': explanation['prediction']
        }
    
    def create_waterfall_plot(self, df: pd.DataFrame, save_path: str = None) -> str:
        """
        Create SHAP waterfall plot for the latest prediction.
        
        Args:
            df: Raw sensor data
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        if not self.is_loaded:
            logger.error("Models not loaded")
            return ""
        
        try:
            # Prepare features
            X = self.predictor.prepare_features(df)
            if X.empty:
                return ""
            
            # Get latest sample
            sample = X.iloc[[-1]]
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(sample)
            
            # Create waterfall plot
            plt.figure(figsize=(10, 8))
            
            # Use SHAP's built-in waterfall plot
            shap.waterfall_plot(
                self.explainer.expected_value[1] if len(self.explainer.expected_value) == 2 else self.explainer.expected_value,
                shap_values[1] if len(shap_values) == 2 else shap_values,
                sample.iloc[0],
                feature_names=self.predictor.feature_columns,
                max_display=15
            )
            
            plt.title("SHAP Waterfall Plot - Equipment Failure Prediction", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = os.path.join(self.model_dir, "shap_waterfall.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Waterfall plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create waterfall plot: {e}")
            return ""
    
    def create_summary_plot(self, df: pd.DataFrame, save_path: str = None) -> str:
        """
        Create SHAP summary plot for multiple samples.
        
        Args:
            df: Raw sensor data (multiple samples)
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        if not self.is_loaded:
            logger.error("Models not loaded")
            return ""
        
        try:
            # Prepare features
            X = self.predictor.prepare_features(df)
            if X.empty:
                return ""
            
            # Get SHAP values for all samples
            shap_values = self.explainer.shap_values(X)
            
            # Create summary plot
            plt.figure(figsize=(12, 8))
            
            shap.summary_plot(
                shap_values[1] if len(shap_values) == 2 else shap_values,
                X,
                feature_names=self.predictor.feature_columns,
                max_display=15,
                show=False
            )
            
            plt.title("SHAP Summary Plot - Feature Importance", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = os.path.join(self.model_dir, "shap_summary.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Summary plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create summary plot: {e}")
            return ""
    
    def get_explanation_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get a comprehensive explanation summary.
        
        Args:
            df: Raw sensor data
            
        Returns:
            Dictionary with explanation summary
        """
        if not self.is_loaded:
            return {}
        
        # Get feature importance
        importance = self.get_feature_importance(df, top_k=10)
        
        # Get latest prediction
        prediction = self.predictor.get_latest_prediction(df)
        
        # Create summary
        summary = {
            'prediction': prediction,
            'feature_importance': importance,
            'explanation_available': True,
            'model_loaded': self.is_loaded
        }
        
        return summary
    
    def explain_batch(self, df: pd.DataFrame) -> Dict:
        """
        Explain predictions for a batch of data.
        
        Args:
            df: Raw sensor data (multiple samples)
            
        Returns:
            Dictionary with batch explanations
        """
        if not self.is_loaded:
            return {}
        
        try:
            # Prepare features
            X = self.predictor.prepare_features(df)
            if X.empty:
                return {}
            
            # Get SHAP values for all samples
            shap_values = self.explainer.shap_values(X)
            
            # Get predictions
            predictions = self.predictor.predict_all(df)
            
            # Create batch explanation
            batch_explanation = {
                'n_samples': len(X),
                'feature_names': self.predictor.feature_columns,
                'shap_values': shap_values[1].tolist() if len(shap_values) == 2 else shap_values.tolist(),
                'predictions': predictions,
                'base_value': self.explainer.expected_value[1] if len(self.explainer.expected_value) == 2 else self.explainer.expected_value
            }
            
            logger.info(f"Generated batch explanation for {len(X)} samples")
            return batch_explanation
            
        except Exception as e:
            logger.error(f"Failed to generate batch explanation: {e}")
            return {}

def main():
    """Test the explainability functionality."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create explainer
    explainer = ModelExplainer()
    
    # Load models
    if not explainer.load_models():
        print("Failed to load models. Make sure to run train_model.py first.")
        return
    
    # Create sample data
    sample_data = {
        'timestamp': [pd.Timestamp.now().isoformat()],
        'sensor_id': ['sensor_001'],
        'temperature': [75.5],  # High temperature
        'vibration': [4.2],     # High vibration
        'pressure': [2.1],     # Low pressure
        'label': ['failure']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Get explanation
    print("Getting SHAP explanation for sample data:")
    print(df.to_string())
    
    explanation = explainer.get_explanation_summary(df)
    
    print("\nExplanation Summary:")
    print(f"Prediction: {explanation['prediction']}")
    print(f"Feature Importance:")
    
    if 'feature_importance' in explanation and 'top_features' in explanation['feature_importance']:
        for i, feature in enumerate(explanation['feature_importance']['top_features'][:5]):
            print(f"  {i+1}. {feature['feature']}: {feature['shap_value']:.4f}")

if __name__ == "__main__":
    main()
