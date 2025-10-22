"""
Training script for ML models used in equipment failure prediction.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
from datetime import datetime, timedelta
import logging

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.clean_data import SensorDataCleaner
from features.feature_engineering import SensorFeatureEngineer
from config import ML_CONFIG, MODEL_DIR

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains and manages ML models for equipment failure prediction."""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.sgd_classifier = SGDClassifier(
            loss='log_loss',
            random_state=42,
            learning_rate='adaptive',
            eta0=0.01
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        
    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic sensor data with realistic failure patterns.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic sensor data
        """
        logger.info(f"Generating {n_samples} synthetic samples")
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=30)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
        
        # Generate sensor IDs
        sensor_ids = [f"sensor_{i:03d}" for i in range(1, 4)]
        
        data = []
        failure_rate = 0.15  # 15% failure rate
        
        for i, timestamp in enumerate(timestamps):
            sensor_id = np.random.choice(sensor_ids)
            
            # Determine if this is a failure case
            is_failure = np.random.random() < failure_rate
            
            if is_failure:
                # Generate failure patterns
                # Temperature increases significantly
                temp = np.random.normal(75, 10)  # High temperature
                # Vibration increases
                vibration = np.random.normal(4, 1)  # High vibration
                # Pressure decreases
                pressure = np.random.normal(2, 0.5)  # Low pressure
                label = "failure"
            else:
                # Generate normal patterns
                temp = np.random.normal(30, 5)  # Normal temperature
                vibration = np.random.normal(1, 0.3)  # Normal vibration
                pressure = np.random.normal(5, 0.8)  # Normal pressure
                label = "normal"
            
            # Add some noise
            temp += np.random.normal(0, 1)
            vibration += np.random.normal(0, 0.1)
            pressure += np.random.normal(0, 0.1)
            
            data.append({
                'timestamp': timestamp.isoformat(),
                'sensor_id': sensor_id,
                'temperature': round(temp, 2),
                'vibration': round(vibration, 2),
                'pressure': round(pressure, 2),
                'label': label
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated data with {df['label'].value_counts().to_dict()} distribution")
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare training data by cleaning and engineering features.
        
        Args:
            df: Raw sensor data
            
        Returns:
            Tuple of (X, y) for training
        """
        logger.info("Preparing training data")
        
        # Clean the data
        cleaner = SensorDataCleaner()
        cleaned_df = cleaner.clean_data(df)
        
        # Engineer features
        engineer = SensorFeatureEngineer()
        features_df = engineer.extract_all_features(cleaned_df)
        
        # Store feature columns
        self.feature_columns = engineer.get_feature_columns()
        
        # Prepare features and target
        X = features_df[self.feature_columns].fillna(0)
        y = features_df['label']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        
        return X, y_encoded
    
    def train_isolation_forest(self, X: pd.DataFrame) -> dict:
        """Train Isolation Forest for anomaly detection."""
        logger.info("Training Isolation Forest")
        
        # Fit the model
        self.isolation_forest.fit(X)
        
        # Get anomaly scores
        anomaly_scores = self.isolation_forest.decision_function(X)
        predictions = self.isolation_forest.predict(X)
        
        # Calculate metrics
        n_anomalies = (predictions == -1).sum()
        anomaly_rate = n_anomalies / len(predictions)
        
        metrics = {
            'anomaly_rate': anomaly_rate,
            'n_anomalies': n_anomalies,
            'mean_anomaly_score': anomaly_scores.mean(),
            'std_anomaly_score': anomaly_scores.std()
        }
        
        logger.info(f"Isolation Forest metrics: {metrics}")
        return metrics
    
    def train_random_forest(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """Train Random Forest classifier."""
        logger.info("Training Random Forest")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.random_forest.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.random_forest.predict(X_test_scaled)
        y_pred_proba = self.random_forest.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        metrics = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'n_features': X.shape[1],
            'n_estimators': self.random_forest.n_estimators
        }
        
        logger.info(f"Random Forest metrics: {metrics}")
        return metrics
    
    def train_sgd_classifier(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """Train SGD Classifier for online learning."""
        logger.info("Training SGD Classifier")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.sgd_classifier.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.sgd_classifier.predict(X_test_scaled)
        y_pred_proba = self.sgd_classifier.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        metrics = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'n_features': X.shape[1]
        }
        
        logger.info(f"SGD Classifier metrics: {metrics}")
        return metrics
    
    def train_all_models(self, df: pd.DataFrame) -> dict:
        """
        Train all models on the provided data.
        
        Args:
            df: Training data
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting model training")
        
        # Prepare data
        X, y = self.prepare_training_data(df)
        
        # Train models
        if_metrics = self.train_isolation_forest(X)
        rf_metrics = self.train_random_forest(X, y)
        sgd_metrics = self.train_sgd_classifier(X, y)
        
        all_metrics = {
            'isolation_forest': if_metrics,
            'random_forest': rf_metrics,
            'sgd_classifier': sgd_metrics,
            'feature_columns': self.feature_columns
        }
        
        logger.info("Model training completed")
        return all_metrics
    
    def save_models(self, model_dir: str = None) -> dict:
        """
        Save trained models to disk.
        
        Args:
            model_dir: Directory to save models
            
        Returns:
            Dictionary with saved file paths
        """
        if model_dir is None:
            model_dir = str(MODEL_DIR)
        
        os.makedirs(model_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save models
        models = {
            'isolation_forest.pkl': self.isolation_forest,
            'random_forest.pkl': self.random_forest,
            'sgd_classifier.pkl': self.sgd_classifier,
            'scaler.pkl': self.scaler,
            'label_encoder.pkl': self.label_encoder
        }
        
        for filename, model in models.items():
            filepath = os.path.join(model_dir, filename)
            joblib.dump(model, filepath)
            saved_files[filename] = filepath
            logger.info(f"Saved {filename} to {filepath}")
        
        # Save feature columns
        feature_file = os.path.join(model_dir, 'feature_columns.pkl')
        joblib.dump(self.feature_columns, feature_file)
        saved_files['feature_columns.pkl'] = feature_file
        
        logger.info(f"All models saved to {model_dir}")
        return saved_files

def main():
    """Main training function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Starting model training process")
    
    # Create trainer
    trainer = ModelTrainer()
    
    # Generate synthetic data
    logger.info("Generating synthetic training data")
    df = trainer.generate_synthetic_data(n_samples=10000)
    
    # Train all models
    logger.info("Training models")
    metrics = trainer.train_all_models(df)
    
    # Save models
    logger.info("Saving models")
    saved_files = trainer.save_models()
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL TRAINING SUMMARY")
    print("="*50)
    
    print(f"\nTraining data: {len(df)} samples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    print(f"\nModels saved to: {MODEL_DIR}")
    for filename, filepath in saved_files.items():
        print(f"  - {filename}")
    
    print(f"\nModel Performance:")
    for model_name, model_metrics in metrics.items():
        if model_name != 'feature_columns':
            print(f"\n{model_name.upper()}:")
            for metric, value in model_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    
    print(f"\nFeature columns: {len(metrics['feature_columns'])}")
    print("="*50)

if __name__ == "__main__":
    main()
