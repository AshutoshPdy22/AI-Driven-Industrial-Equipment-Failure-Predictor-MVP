"""
Data cleaning and preprocessing module for sensor data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SensorDataCleaner:
    """Handles cleaning and preprocessing of sensor data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean sensor data by handling missing values and outliers.
        
        Args:
            df: DataFrame with sensor readings
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return df
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in cleaned_df.columns:
            cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
        
        # Handle missing values
        cleaned_df = self._handle_missing_values(cleaned_df)
        
        # Handle outliers
        cleaned_df = self._handle_outliers(cleaned_df)
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['temperature', 'vibration', 'pressure']
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        logger.info(f"Cleaned data: {len(cleaned_df)} rows")
        return cleaned_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in sensor data."""
        # For sensor data, we can use forward fill for missing values
        # as they represent continuous measurements
        numeric_columns = ['temperature', 'vibration', 'pressure']
        
        for col in numeric_columns:
            if col in df.columns:
                # Forward fill missing values
                df[col] = df[col].fillna(method='ffill')
                # If still missing (e.g., first values), use backward fill
                df[col] = df[col].fillna(method='bfill')
                # If still missing, use median
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.warning(f"Filled missing values in {col} with median: {median_val}")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method."""
        numeric_columns = ['temperature', 'vibration', 'pressure']
        
        for col in numeric_columns:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Clip outliers instead of removing them
                original_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if original_outliers > 0:
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"Clipped {original_outliers} outliers in {col}")
        
        return df
    
    def normalize_data(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Normalize sensor data using StandardScaler.
        
        Args:
            df: DataFrame with sensor readings
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            Normalized DataFrame
        """
        if df.empty:
            return df
        
        # Select numeric columns for normalization
        numeric_columns = ['temperature', 'vibration', 'pressure']
        available_columns = [col for col in numeric_columns if col in df.columns]
        
        if not available_columns:
            logger.warning("No numeric columns found for normalization")
            return df
        
        normalized_df = df.copy()
        
        if fit_scaler or not self.is_fitted:
            # Fit the scaler
            normalized_df[available_columns] = self.scaler.fit_transform(
                normalized_df[available_columns]
            )
            self.is_fitted = True
            logger.info("Scaler fitted on data")
        else:
            # Transform using fitted scaler
            normalized_df[available_columns] = self.scaler.transform(
                normalized_df[available_columns]
            )
            logger.info("Data normalized using fitted scaler")
        
        return normalized_df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML models.
        
        Args:
            df: Cleaned DataFrame with sensor readings
            
        Returns:
            DataFrame with features ready for ML
        """
        if df.empty:
            return df
        
        features_df = df.copy()
        
        # Ensure we have the required columns
        required_columns = ['temperature', 'vibration', 'pressure']
        missing_columns = [col for col in required_columns if col not in features_df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        # Add sensor_id as categorical feature if available
        if 'sensor_id' in features_df.columns:
            # One-hot encode sensor_id
            sensor_dummies = pd.get_dummies(features_df['sensor_id'], prefix='sensor')
            features_df = pd.concat([features_df, sensor_dummies], axis=1)
        
        # Add timestamp features if timestamp is available
        if 'timestamp' in features_df.columns:
            features_df['hour'] = pd.to_datetime(features_df['timestamp']).dt.hour
            features_df['day_of_week'] = pd.to_datetime(features_df['timestamp']).dt.dayofweek
        
        return features_df
    
    def get_feature_columns(self) -> list:
        """Get list of feature columns for ML models."""
        base_features = ['temperature', 'vibration', 'pressure']
        # Add sensor dummy columns (assuming 3 sensors)
        sensor_features = [f'sensor_sensor_{i:03d}' for i in range(1, 4)]
        time_features = ['hour', 'day_of_week']
        
        return base_features + sensor_features + time_features

def clean_sensor_data(df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """
    Convenience function to clean sensor data.
    
    Args:
        df: Raw sensor data DataFrame
        normalize: Whether to normalize the data
        
    Returns:
        Cleaned and optionally normalized DataFrame
    """
    cleaner = SensorDataCleaner()
    cleaned_df = cleaner.clean_data(df)
    
    if normalize:
        cleaned_df = cleaner.normalize_data(cleaned_df, fit_scaler=True)
    
    return cleaned_df

def main():
    """Test the data cleaning functionality."""
    # Create sample data
    sample_data = {
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'sensor_id': ['sensor_001'] * 100,
        'temperature': np.random.normal(30, 5, 100),
        'vibration': np.random.normal(1, 0.5, 100),
        'pressure': np.random.normal(5, 1, 100),
        'label': ['normal'] * 100
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some missing values and outliers
    df.loc[10:15, 'temperature'] = np.nan
    df.loc[20, 'temperature'] = 150  # outlier
    df.loc[25, 'vibration'] = 20     # outlier
    
    print("Original data shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    
    # Clean the data
    cleaner = SensorDataCleaner()
    cleaned_df = cleaner.clean_data(df)
    normalized_df = cleaner.normalize_data(cleaned_df, fit_scaler=True)
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Missing values after cleaning:", cleaned_df.isnull().sum().sum())
    print("Normalized data sample:")
    print(normalized_df[['temperature', 'vibration', 'pressure']].head())

if __name__ == "__main__":
    main()
