"""
Configuration settings for the Equipment Failure Predictor system.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
DB_PATH = PROJECT_ROOT / "equipment_data.db"

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    "sensors": "/api/sensors",
    "health": "/api/health"
}

# Database Configuration
DB_CONFIG = {
    "path": str(DB_PATH),
    "table_name": "sensor_readings"
}

# ML Model Configuration
ML_CONFIG = {
    "failure_threshold": 0.7,
    "anomaly_threshold": -0.1,
    "models": {
        "isolation_forest": "isolation_forest.pkl",
        "random_forest": "random_forest.pkl", 
        "sgd_classifier": "sgd_classifier.pkl",
        "scaler": "scaler.pkl"
    }
}

# Alert Configuration
ALERT_CONFIG = {
    "console_enabled": True,
    "webhook_enabled": os.getenv("WEBHOOK_ENABLED", "false").lower() == "true",
    "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
    "telegram_webhook": os.getenv("TELEGRAM_WEBHOOK_URL"),
    "failure_threshold": 0.7,
    "anomaly_threshold": -0.1
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    "refresh_interval": 5,  # seconds
    "max_data_points": 100,
    "port": 8501
}

# Data Generation Configuration
DATA_CONFIG = {
    "sensor_count": 3,
    "update_interval": 2,  # seconds
    "failure_injection_rate": 0.1,  # 10% chance of failure patterns
    "normal_ranges": {
        "temperature": (20, 80),
        "vibration": (0, 5),
        "pressure": (1, 10)
    }
}
