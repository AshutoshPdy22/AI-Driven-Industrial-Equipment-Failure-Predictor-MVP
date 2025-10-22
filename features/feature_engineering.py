"""
Feature engineering module for extracting meaningful features from sensor data.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class SensorFeatureEngineer:
    """Extracts and engineers features from sensor data."""
    
    def __init__(self):
        self.feature_columns = []
        self.rolling_windows = [5, 10, 20]
    
    def extract_rolling_features(self, df: pd.DataFrame, sensor_columns: List[str]) -> pd.DataFrame:
        """
        Extract rolling statistics features.
        
        Args:
            df: DataFrame with sensor data
            sensor_columns: List of sensor column names
            
        Returns:
            DataFrame with rolling features added
        """
        if df.empty:
            return df
        
        features_df = df.copy()
        
        for window in self.rolling_windows:
            for col in sensor_columns:
                if col in features_df.columns:
                    # Rolling mean
                    features_df[f'{col}_rolling_mean_{window}'] = (
                        features_df[col].rolling(window=window, min_periods=1).mean()
                    )
                    
                    # Rolling standard deviation
                    features_df[f'{col}_rolling_std_{window}'] = (
                        features_df[col].rolling(window=window, min_periods=1).std()
                    )
                    
                    # Rolling min/max
                    features_df[f'{col}_rolling_min_{window}'] = (
                        features_df[col].rolling(window=window, min_periods=1).min()
                    )
                    features_df[f'{col}_rolling_max_{window}'] = (
                        features_df[col].rolling(window=window, min_periods=1).max()
                    )
                    
                    # Rolling range (max - min)
                    features_df[f'{col}_rolling_range_{window}'] = (
                        features_df[f'{col}_rolling_max_{window}'] - 
                        features_df[f'{col}_rolling_min_{window}']
                    )
        
        logger.info(f"Added rolling features with windows: {self.rolling_windows}")
        return features_df
    
    def extract_rate_of_change(self, df: pd.DataFrame, sensor_columns: List[str]) -> pd.DataFrame:
        """
        Extract rate of change features.
        
        Args:
            df: DataFrame with sensor data
            sensor_columns: List of sensor column names
            
        Returns:
            DataFrame with rate of change features added
        """
        if df.empty:
            return df
        
        features_df = df.copy()
        
        for col in sensor_columns:
            if col in features_df.columns:
                # Rate of change (current - previous)
                features_df[f'{col}_rate_of_change'] = features_df[col].diff()
                
                # Rate of change percentage
                features_df[f'{col}_rate_of_change_pct'] = features_df[col].pct_change()
                
                # Acceleration (rate of change of rate of change)
                features_df[f'{col}_acceleration'] = features_df[f'{col}_rate_of_change'].diff()
        
        logger.info("Added rate of change features")
        return features_df
    
    def extract_spike_features(self, df: pd.DataFrame, sensor_columns: List[str]) -> pd.DataFrame:
        """
        Extract spike detection features.
        
        Args:
            df: DataFrame with sensor data
            sensor_columns: List of sensor column names
            
        Returns:
            DataFrame with spike features added
        """
        if df.empty:
            return df
        
        features_df = df.copy()
        
        for col in sensor_columns:
            if col in features_df.columns:
                # Calculate rolling statistics for spike detection
                rolling_mean = features_df[col].rolling(window=10, min_periods=1).mean()
                rolling_std = features_df[col].rolling(window=10, min_periods=1).std()
                
                # Z-score for spike detection
                features_df[f'{col}_z_score'] = (
                    (features_df[col] - rolling_mean) / (rolling_std + 1e-8)
                )
                
                # Spike indicator (z-score > 2 or < -2)
                features_df[f'{col}_spike'] = (
                    (features_df[f'{col}_z_score'].abs() > 2).astype(int)
                )
                
                # Spike magnitude
                features_df[f'{col}_spike_magnitude'] = features_df[f'{col}_z_score'].abs()
        
        logger.info("Added spike detection features")
        return features_df
    
    def extract_lag_features(self, df: pd.DataFrame, sensor_columns: List[str], lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """
        Extract lag features.
        
        Args:
            df: DataFrame with sensor data
            sensor_columns: List of sensor column names
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features added
        """
        if df.empty:
            return df
        
        features_df = df.copy()
        
        for col in sensor_columns:
            if col in features_df.columns:
                for lag in lags:
                    features_df[f'{col}_lag_{lag}'] = features_df[col].shift(lag)
        
        logger.info(f"Added lag features with lags: {lags}")
        return features_df
    
    def extract_interaction_features(self, df: pd.DataFrame, sensor_columns: List[str]) -> pd.DataFrame:
        """
        Extract interaction features between sensors.
        
        Args:
            df: DataFrame with sensor data
            sensor_columns: List of sensor column names
            
        Returns:
            DataFrame with interaction features added
        """
        if df.empty or len(sensor_columns) < 2:
            return df
        
        features_df = df.copy()
        
        # Temperature-Vibration interaction
        if 'temperature' in sensor_columns and 'vibration' in sensor_columns:
            features_df['temp_vib_interaction'] = (
                features_df['temperature'] * features_df['vibration']
            )
        
        # Temperature-Pressure interaction
        if 'temperature' in sensor_columns and 'pressure' in sensor_columns:
            features_df['temp_pressure_interaction'] = (
                features_df['temperature'] * features_df['pressure']
            )
        
        # Vibration-Pressure interaction
        if 'vibration' in sensor_columns and 'pressure' in sensor_columns:
            features_df['vib_pressure_interaction'] = (
                features_df['vibration'] * features_df['pressure']
            )
        
        # Combined sensor health score (normalized)
        if len(sensor_columns) >= 3:
            # Normalize each sensor to 0-1 scale
            temp_norm = (features_df['temperature'] - features_df['temperature'].min()) / (
                features_df['temperature'].max() - features_df['temperature'].min() + 1e-8
            )
            vib_norm = (features_df['vibration'] - features_df['vibration'].min()) / (
                features_df['vibration'].max() - features_df['vibration'].min() + 1e-8
            )
            pressure_norm = (features_df['pressure'] - features_df['pressure'].min()) / (
                features_df['pressure'].max() - features_df['pressure'].min() + 1e-8
            )
            
            # Combined health score (lower is better for vibration, higher is better for temp/pressure)
            features_df['combined_health_score'] = (
                temp_norm + (1 - vib_norm) + pressure_norm
            ) / 3
        
        logger.info("Added interaction features")
        return features_df
    
    def extract_all_features(self, df: pd.DataFrame, sensor_columns: List[str] = None) -> pd.DataFrame:
        """
        Extract all features from sensor data.
        
        Args:
            df: DataFrame with sensor data
            sensor_columns: List of sensor column names (default: ['temperature', 'vibration', 'pressure'])
            
        Returns:
            DataFrame with all engineered features
        """
        if df.empty:
            return df
        
        if sensor_columns is None:
            sensor_columns = ['temperature', 'vibration', 'pressure']
        
        # Filter to available columns
        available_sensor_columns = [col for col in sensor_columns if col in df.columns]
        
        if not available_sensor_columns:
            logger.warning("No sensor columns found for feature engineering")
            return df
        
        logger.info(f"Engineering features for columns: {available_sensor_columns}")
        
        # Start with original data
        features_df = df.copy()
        
        # Extract different types of features
        features_df = self.extract_rolling_features(features_df, available_sensor_columns)
        features_df = self.extract_rate_of_change(features_df, available_sensor_columns)
        features_df = self.extract_spike_features(features_df, available_sensor_columns)
        features_df = self.extract_lag_features(features_df, available_sensor_columns)
        features_df = self.extract_interaction_features(features_df, available_sensor_columns)
        
        # Store feature columns for later use
        self.feature_columns = [col for col in features_df.columns if col not in df.columns]
        
        logger.info(f"Total features engineered: {len(self.feature_columns)}")
        return features_df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of engineered feature columns."""
        return self.feature_columns
    
    def select_important_features(self, df: pd.DataFrame, target_column: str = 'label', 
                                top_k: int = 20) -> List[str]:
        """
        Select most important features using correlation with target.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            top_k: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        if target_column not in df.columns:
            logger.warning(f"Target column '{target_column}' not found")
            return self.feature_columns[:top_k]
        
        # Calculate correlation with target
        correlations = df[self.feature_columns + [target_column]].corr()[target_column].abs()
        correlations = correlations.drop(target_column).sort_values(ascending=False)
        
        selected_features = correlations.head(top_k).index.tolist()
        logger.info(f"Selected top {len(selected_features)} features")
        
        return selected_features

def engineer_features(df: pd.DataFrame, sensor_columns: List[str] = None) -> pd.DataFrame:
    """
    Convenience function to engineer features from sensor data.
    
    Args:
        df: DataFrame with sensor data
        sensor_columns: List of sensor column names
        
    Returns:
        DataFrame with engineered features
    """
    engineer = SensorFeatureEngineer()
    return engineer.extract_all_features(df, sensor_columns)

def main():
    """Test the feature engineering functionality."""
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    sample_data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'sensor_id': ['sensor_001'] * n_samples,
        'temperature': 30 + np.random.normal(0, 5, n_samples),
        'vibration': 1 + np.random.normal(0, 0.5, n_samples),
        'pressure': 5 + np.random.normal(0, 1, n_samples),
        'label': ['normal'] * n_samples
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original data shape:", df.shape)
    print("Original columns:", df.columns.tolist())
    
    # Engineer features
    engineer = SensorFeatureEngineer()
    features_df = engineer.extract_all_features(df)
    
    print(f"\nFeatures engineered: {len(engineer.get_feature_columns())}")
    print("New data shape:", features_df.shape)
    print("Sample engineered features:")
    print(features_df[engineer.get_feature_columns()].head())

if __name__ == "__main__":
    main()
