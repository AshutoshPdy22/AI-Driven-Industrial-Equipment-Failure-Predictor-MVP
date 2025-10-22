"""
Prediction module for equipment failure prediction using trained models.
"""
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Tuple, Optional
import logging

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.clean_data import SensorDataCleaner
from features.feature_engineering import SensorFeatureEngineer
from config import ML_CONFIG, MODEL_DIR

logger = logging.getLogger(__name__)

class EquipmentFailurePredictor:
    """Predicts equipment failure using trained ML models."""
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or str(MODEL_DIR)
        self.models = {}
        self.feature_columns = []
        self.is_loaded = False
        
    def load_models(self) -> bool:
        """
        Load trained models from disk.
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            # Load models
            model_files = {
                'isolation_forest': 'isolation_forest.pkl',
                'random_forest': 'random_forest.pkl',
                'sgd_classifier': 'sgd_classifier.pkl',
                'scaler': 'scaler.pkl',
                'label_encoder': 'label_encoder.pkl'
            }
            
            for model_name, filename in model_files.items():
                filepath = os.path.join(self.model_dir, filename)
                if os.path.exists(filepath):
                    self.models[model_name] = joblib.load(filepath)
                    logger.info(f"Loaded {model_name} from {filepath}")
                else:
                    logger.error(f"Model file not found: {filepath}")
                    return False
            
            # Load feature columns
            feature_file = os.path.join(self.model_dir, 'feature_columns.pkl')
            if os.path.exists(feature_file):
                self.feature_columns = joblib.load(feature_file)
                logger.info(f"Loaded {len(self.feature_columns)} feature columns")
            else:
                logger.error(f"Feature columns file not found: {feature_file}")
                return False
            
            self.is_loaded = True
            logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction.
        
        Args:
            df: Raw sensor data
            
        Returns:
            DataFrame with prepared features
        """
        if not self.is_loaded:
            logger.error("Models not loaded. Call load_models() first.")
            return pd.DataFrame()
        
        # Clean the data
        cleaner = SensorDataCleaner()
        cleaned_df = cleaner.clean_data(df)
        
        # Engineer features
        engineer = SensorFeatureEngineer()
        features_df = engineer.extract_all_features(cleaned_df)
        
        # Select only the features used in training
        available_features = [col for col in self.feature_columns if col in features_df.columns]
        missing_features = [col for col in self.feature_columns if col not in features_df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Fill missing features with zeros
            for col in missing_features:
                features_df[col] = 0
        
        # Select and order features
        X = features_df[available_features].fillna(0)
        
        logger.info(f"Prepared features: {X.shape}")
        return X
    
    def predict_anomaly(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies using Isolation Forest.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (anomaly_scores, anomaly_predictions)
        """
        if not self.is_loaded:
            logger.error("Models not loaded")
            return np.array([]), np.array([])
        
        try:
            # Get anomaly scores (higher is more normal, lower is more anomalous)
            anomaly_scores = self.models['isolation_forest'].decision_function(X)
            
            # Get anomaly predictions (-1 for anomaly, 1 for normal)
            anomaly_predictions = self.models['isolation_forest'].predict(X)
            
            logger.info(f"Anomaly prediction completed for {len(X)} samples")
            return anomaly_scores, anomaly_predictions
            
        except Exception as e:
            logger.error(f"Anomaly prediction failed: {e}")
            return np.array([]), np.array([])
    
    def predict_failure(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict failure probability using Random Forest.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (failure_probabilities, failure_predictions)
        """
        if not self.is_loaded:
            logger.error("Models not loaded")
            return np.array([]), np.array([])
        
        try:
            # Scale features
            X_scaled = self.models['scaler'].transform(X)
            
            # Get failure probabilities
            failure_probabilities = self.models['random_forest'].predict_proba(X_scaled)[:, 1]
            
            # Get failure predictions
            failure_predictions = self.models['random_forest'].predict(X_scaled)
            
            logger.info(f"Failure prediction completed for {len(X)} samples")
            return failure_probabilities, failure_predictions
            
        except Exception as e:
            logger.error(f"Failure prediction failed: {e}")
            return np.array([]), np.array([])
    
    def predict_online(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using SGD Classifier (for online learning).
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (online_probabilities, online_predictions)
        """
        if not self.is_loaded:
            logger.error("Models not loaded")
            return np.array([]), np.array([])
        
        try:
            # Scale features
            X_scaled = self.models['scaler'].transform(X)
            
            # Get online learning predictions
            online_probabilities = self.models['sgd_classifier'].predict_proba(X_scaled)[:, 1]
            online_predictions = self.models['sgd_classifier'].predict(X_scaled)
            
            logger.info(f"Online prediction completed for {len(X)} samples")
            return online_probabilities, online_predictions
            
        except Exception as e:
            logger.error(f"Online prediction failed: {e}")
            return np.array([]), np.array([])
    
    def predict_all(self, df: pd.DataFrame) -> Dict:
        """
        Make predictions using all models.
        
        Args:
            df: Raw sensor data
            
        Returns:
            Dictionary with all predictions and risk scores
        """
        if not self.is_loaded:
            logger.error("Models not loaded. Call load_models() first.")
            return {}
        
        # Prepare features
        X = self.prepare_features(df)
        if X.empty:
            return {}
        
        # Get predictions from all models
        anomaly_scores, anomaly_preds = self.predict_anomaly(X)
        failure_probs, failure_preds = self.predict_failure(X)
        online_probs, online_preds = self.predict_online(X)
        
        # Calculate combined risk score
        risk_scores = self._calculate_risk_score(
            anomaly_scores, failure_probs, online_probs
        )
        
        # Prepare results
        results = {
            'anomaly_scores': anomaly_scores.tolist() if len(anomaly_scores) > 0 else [],
            'anomaly_predictions': anomaly_preds.tolist() if len(anomaly_preds) > 0 else [],
            'failure_probabilities': failure_probs.tolist() if len(failure_probs) > 0 else [],
            'failure_predictions': failure_preds.tolist() if len(failure_preds) > 0 else [],
            'online_probabilities': online_probs.tolist() if len(online_probs) > 0 else [],
            'online_predictions': online_preds.tolist() if len(online_preds) > 0 else [],
            'risk_scores': risk_scores.tolist() if len(risk_scores) > 0 else [],
            'feature_columns': self.feature_columns
        }
        
        logger.info(f"Prediction completed for {len(df)} samples")
        return results
    
    def _calculate_risk_score(self, anomaly_scores: np.ndarray, 
                             failure_probs: np.ndarray, 
                             online_probs: np.ndarray) -> np.ndarray:
        """
        Calculate combined risk score from all models.
        
        Args:
            anomaly_scores: Anomaly scores from Isolation Forest
            failure_probs: Failure probabilities from Random Forest
            online_probs: Online learning probabilities
            
        Returns:
            Combined risk scores
        """
        if len(anomaly_scores) == 0 or len(failure_probs) == 0 or len(online_probs) == 0:
            return np.array([])
        
        # Normalize anomaly scores (convert to 0-1 scale where 1 is high risk)
        # Isolation Forest returns higher scores for normal data, lower for anomalies
        anomaly_risk = 1 - (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
        
        # Combine all scores with weights
        weights = {
            'anomaly': 0.3,
            'failure': 0.5,
            'online': 0.2
        }
        
        risk_scores = (
            weights['anomaly'] * anomaly_risk +
            weights['failure'] * failure_probs +
            weights['online'] * online_probs
        )
        
        return risk_scores
    
    def get_latest_prediction(self, df: pd.DataFrame) -> Dict:
        """
        Get prediction for the latest data point.
        
        Args:
            df: Raw sensor data (should be sorted by timestamp)
            
        Returns:
            Dictionary with latest prediction
        """
        if df.empty:
            return {}
        
        # Get latest data point
        latest_df = df.tail(1)
        
        # Get predictions
        results = self.predict_all(latest_df)
        
        if not results:
            return {}
        
        # Extract latest values
        latest_result = {
            'timestamp': latest_df['timestamp'].iloc[0] if 'timestamp' in latest_df.columns else None,
            'sensor_id': latest_df['sensor_id'].iloc[0] if 'sensor_id' in latest_df.columns else None,
            'anomaly_score': results['anomaly_scores'][0] if results['anomaly_scores'] else None,
            'failure_probability': results['failure_probabilities'][0] if results['failure_probabilities'] else None,
            'online_probability': results['online_probabilities'][0] if results['online_probabilities'] else None,
            'risk_score': results['risk_scores'][0] if results['risk_scores'] else None,
            'is_anomaly': results['anomaly_predictions'][0] == -1 if results['anomaly_predictions'] else None,
            'is_failure': results['failure_predictions'][0] == 1 if results['failure_predictions'] else None
        }
        
        return latest_result

def main():
    """Test the prediction functionality."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create predictor
    predictor = EquipmentFailurePredictor()
    
    # Load models
    if not predictor.load_models():
        print("Failed to load models. Make sure to run train_model.py first.")
        return
    
    # Create sample data
    sample_data = {
        'timestamp': [pd.Timestamp.now().isoformat()],
        'sensor_id': ['sensor_001'],
        'temperature': [75.5],  # High temperature (potential failure)
        'vibration': [4.2],     # High vibration
        'pressure': [2.1],     # Low pressure
        'label': ['failure']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Make prediction
    print("Making prediction for sample data:")
    print(df.to_string())
    
    result = predictor.get_latest_prediction(df)
    
    print("\nPrediction Results:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
