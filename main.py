"""
Main orchestrator for the Equipment Failure Predictor system.
Runs the complete pipeline: fetch → preprocess → engineer → predict → alert
"""
import time
import logging
import os
import sys
from datetime import datetime
import signal
import threading

# Import our modules
from data_fetch.api_client import SensorDataClient
from preprocessing.clean_data import SensorDataCleaner
from features.feature_engineering import SensorFeatureEngineer
from model.predict import EquipmentFailurePredictor
from explainability.explain_model import ModelExplainer
from alerts.alert_handler import AlertHandler
from config import API_BASE_URL, DB_PATH, DASHBOARD_CONFIG, DATA_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('equipment_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EquipmentMonitor:
    """Main orchestrator for the equipment monitoring system."""
    
    def __init__(self):
        self.running = False
        self.client = None
        self.predictor = None
        self.explainer = None
        self.alert_handler = None
        self.cleaner = None
        self.engineer = None
        
    def initialize_components(self):
        """Initialize all system components."""
        logger.info("Initializing system components...")
        
        try:
            # Initialize data client
            self.client = SensorDataClient(API_BASE_URL, str(DB_PATH))
            logger.info("Data client initialized")
            
            # Initialize predictor
            self.predictor = EquipmentFailurePredictor()
            if not self.predictor.load_models():
                logger.error("Failed to load prediction models")
                return False
            logger.info("Prediction models loaded")
            
            # Initialize explainer
            self.explainer = ModelExplainer()
            if not self.explainer.load_models():
                logger.warning("Failed to load explainer (optional)")
            else:
                logger.info("Model explainer loaded")
            
            # Initialize alert handler
            self.alert_handler = AlertHandler()
            logger.info("Alert handler initialized")
            
            # Initialize data processors
            self.cleaner = SensorDataCleaner()
            self.engineer = SensorFeatureEngineer()
            logger.info("Data processors initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    def check_api_health(self):
        """Check if API is healthy."""
        if not self.client:
            return False
        
        return self.client.is_api_healthy()
    
    def fetch_sensor_data(self):
        """Fetch latest sensor data from API."""
        try:
            if not self.client:
                logger.error("Data client not initialized")
                return 0
            
            # Fetch data for all sensors
            stored_count = self.client.fetch_and_store()
            logger.info(f"Fetched and stored {stored_count} sensor readings")
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to fetch sensor data: {e}")
            return 0
    
    def process_predictions(self):
        """Process predictions for latest data."""
        try:
            if not self.predictor:
                logger.error("Predictor not initialized")
                return
            
            # Get latest data
            df = self.client.get_latest_readings(limit=50)
            if df.empty:
                logger.warning("No data available for prediction")
                return
            
            # Process each sensor
            for sensor_id in df['sensor_id'].unique():
                sensor_data = df[df['sensor_id'] == sensor_id].tail(1)
                
                # Get prediction
                prediction = self.predictor.get_latest_prediction(sensor_data)
                
                if prediction:
                    # Process alert
                    alert_result = self.alert_handler.process_prediction(prediction)
                    
                    if alert_result['alert_created']:
                        logger.info(f"Alert created for {sensor_id}: {alert_result['alert_level']}")
            
        except Exception as e:
            logger.error(f"Failed to process predictions: {e}")
    
    def run_monitoring_cycle(self):
        """Run one complete monitoring cycle."""
        try:
            logger.info("Starting monitoring cycle...")
            
            # Check API health
            if not self.check_api_health():
                logger.warning("API is not healthy, skipping cycle")
                return
            
            # Fetch new data
            fetched_count = self.fetch_sensor_data()
            if fetched_count == 0:
                logger.warning("No new data fetched")
                return
            
            # Process predictions
            self.process_predictions()
            
            logger.info("Monitoring cycle completed")
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
    
    def start_monitoring(self, interval: int = 30):
        """
        Start continuous monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        self.running = True
        
        try:
            while self.running:
                self.run_monitoring_cycle()
                
                # Wait for next cycle
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            self.running = False
            logger.info("Monitoring stopped")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        logger.info("Stopping monitoring...")
    
    def get_system_status(self):
        """Get current system status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'running': self.running,
            'api_healthy': self.check_api_health(),
            'models_loaded': self.predictor is not None and self.predictor.is_loaded,
            'explainer_loaded': self.explainer is not None and self.explainer.is_loaded,
            'alert_stats': self.alert_handler.get_alert_stats() if self.alert_handler else {}
        }
        
        return status

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    if 'monitor' in globals():
        monitor.stop_monitoring()
    sys.exit(0)

def main():
    """Main function."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Equipment Failure Predictor System")
    logger.info("=" * 60)
    
    # Create monitor
    monitor = EquipmentMonitor()
    
    # Initialize components
    if not monitor.initialize_components():
        logger.error("Failed to initialize system components")
        sys.exit(1)
    
    # Check system status
    status = monitor.get_system_status()
    logger.info(f"System status: {status}")
    
    # Start monitoring
    try:
        # Get monitoring interval from config
        interval = DATA_CONFIG.get('update_interval', 30)
        monitor.start_monitoring(interval)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
