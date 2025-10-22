"""
Alert handling system for equipment failure predictions.
"""
import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import os
from enum import Enum

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ALERT_CONFIG

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class AlertHandler:
    """Handles alert generation and delivery."""
    
    def __init__(self, config: Dict = None):
        self.config = config or ALERT_CONFIG
        self.alert_history = []
        self.console_enabled = self.config.get('console_enabled', True)
        self.webhook_enabled = self.config.get('webhook_enabled', False)
        self.failure_threshold = self.config.get('failure_threshold', 0.7)
        self.anomaly_threshold = self.config.get('anomaly_threshold', -0.1)
        
    def create_alert(self, prediction_data: Dict, alert_level: AlertLevel = AlertLevel.WARNING) -> Dict:
        """
        Create an alert from prediction data.
        
        Args:
            prediction_data: Dictionary with prediction results
            alert_level: Severity level of the alert
            
        Returns:
            Dictionary with alert information
        """
        timestamp = datetime.now().isoformat()
        
        # Extract key information
        sensor_id = prediction_data.get('sensor_id', 'unknown')
        risk_score = prediction_data.get('risk_score', 0)
        failure_prob = prediction_data.get('failure_probability', 0)
        anomaly_score = prediction_data.get('anomaly_score', 0)
        
        # Create alert message
        if alert_level == AlertLevel.EMERGENCY:
            message = f"🚨 EMERGENCY: Equipment {sensor_id} at CRITICAL risk of failure!"
        elif alert_level == AlertLevel.CRITICAL:
            message = f"⚠️ CRITICAL: Equipment {sensor_id} showing high failure risk!"
        elif alert_level == AlertLevel.WARNING:
            message = f"⚠️ WARNING: Equipment {sensor_id} showing elevated risk!"
        else:
            message = f"ℹ️ INFO: Equipment {sensor_id} status update"
        
        alert = {
            'timestamp': timestamp,
            'sensor_id': sensor_id,
            'alert_level': alert_level.value,
            'message': message,
            'risk_score': risk_score,
            'failure_probability': failure_prob,
            'anomaly_score': anomaly_score,
            'is_failure': prediction_data.get('is_failure', False),
            'is_anomaly': prediction_data.get('is_anomaly', False),
            'raw_data': prediction_data
        }
        
        return alert
    
    def determine_alert_level(self, prediction_data: Dict) -> AlertLevel:
        """
        Determine alert level based on prediction data.
        
        Args:
            prediction_data: Dictionary with prediction results
            
        Returns:
            AlertLevel enum value
        """
        risk_score = prediction_data.get('risk_score', 0)
        failure_prob = prediction_data.get('failure_probability', 0)
        is_failure = prediction_data.get('is_failure', False)
        is_anomaly = prediction_data.get('is_anomaly', False)
        
        # Emergency: Very high risk or confirmed failure
        if risk_score >= 0.9 or failure_prob >= 0.9 or is_failure:
            return AlertLevel.EMERGENCY
        
        # Critical: High risk
        elif risk_score >= 0.8 or failure_prob >= 0.8:
            return AlertLevel.CRITICAL
        
        # Warning: Elevated risk
        elif risk_score >= self.failure_threshold or failure_prob >= self.failure_threshold or is_anomaly:
            return AlertLevel.WARNING
        
        # Info: Normal operation
        else:
            return AlertLevel.INFO
    
    def send_console_alert(self, alert: Dict) -> bool:
        """
        Send alert to console/log.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.console_enabled:
            return True
        
        try:
            # Format console message
            timestamp = alert['timestamp']
            level = alert['alert_level']
            message = alert['message']
            sensor_id = alert['sensor_id']
            risk_score = alert['risk_score']
            
            console_message = f"""
{'='*60}
ALERT: {level}
Time: {timestamp}
Equipment: {sensor_id}
Message: {message}
Risk Score: {risk_score:.4f}
Failure Probability: {alert['failure_probability']:.4f}
Anomaly Score: {alert['anomaly_score']:.4f}
{'='*60}
"""
            
            # Log the alert
            if level == "EMERGENCY":
                logger.critical(console_message)
            elif level == "CRITICAL":
                logger.error(console_message)
            elif level == "WARNING":
                logger.warning(console_message)
            else:
                logger.info(console_message)
            
            print(console_message)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send console alert: {e}")
            return False
    
    def send_webhook_alert(self, alert: Dict) -> bool:
        """
        Send alert via webhook (Slack/Telegram).
        
        Args:
            alert: Alert dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.webhook_enabled:
            return True
        
        success = True
        
        # Try Slack webhook
        slack_webhook = self.config.get('slack_webhook')
        if slack_webhook:
            if not self._send_slack_alert(alert, slack_webhook):
                success = False
        
        # Try Telegram webhook
        telegram_webhook = self.config.get('telegram_webhook')
        if telegram_webhook:
            if not self._send_telegram_alert(alert, telegram_webhook):
                success = False
        
        return success
    
    def _send_slack_alert(self, alert: Dict, webhook_url: str) -> bool:
        """Send alert to Slack."""
        try:
            # Create Slack message
            color = {
                "EMERGENCY": "#FF0000",  # Red
                "CRITICAL": "#FF6600",   # Orange
                "WARNING": "#FFAA00",    # Yellow
                "INFO": "#00AA00"        # Green
            }.get(alert['alert_level'], "#000000")
            
            slack_message = {
                "attachments": [{
                    "color": color,
                    "title": f"Equipment Alert: {alert['sensor_id']}",
                    "text": alert['message'],
                    "fields": [
                        {
                            "title": "Risk Score",
                            "value": f"{alert['risk_score']:.4f}",
                            "short": True
                        },
                        {
                            "title": "Failure Probability",
                            "value": f"{alert['failure_probability']:.4f}",
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": alert['timestamp'],
                            "short": False
                        }
                    ],
                    "footer": "Equipment Failure Predictor",
                    "ts": int(datetime.now().timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=slack_message, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _send_telegram_alert(self, alert: Dict, webhook_url: str) -> bool:
        """Send alert to Telegram."""
        try:
            # Create Telegram message
            emoji = {
                "EMERGENCY": "🚨",
                "CRITICAL": "⚠️",
                "WARNING": "⚠️",
                "INFO": "ℹ️"
            }.get(alert['alert_level'], "📊")
            
            telegram_message = f"""
{emoji} *Equipment Alert*

*Equipment:* {alert['sensor_id']}
*Level:* {alert['alert_level']}
*Message:* {alert['message']}

*Risk Score:* {alert['risk_score']:.4f}
*Failure Probability:* {alert['failure_probability']:.4f}
*Anomaly Score:* {alert['anomaly_score']:.4f}

*Time:* {alert['timestamp']}
"""
            
            payload = {
                "text": telegram_message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Telegram alert sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    def process_prediction(self, prediction_data: Dict) -> Dict:
        """
        Process a prediction and generate alerts if necessary.
        
        Args:
            prediction_data: Dictionary with prediction results
            
        Returns:
            Dictionary with alert results
        """
        # Determine alert level
        alert_level = self.determine_alert_level(prediction_data)
        
        # Create alert
        alert = self.create_alert(prediction_data, alert_level)
        
        # Store in history
        self.alert_history.append(alert)
        
        # Send alerts
        console_success = self.send_console_alert(alert)
        webhook_success = self.send_webhook_alert(alert)
        
        # Return results
        return {
            'alert_created': True,
            'alert_level': alert_level.value,
            'alert': alert,
            'console_sent': console_success,
            'webhook_sent': webhook_success,
            'total_alerts': len(self.alert_history)
        }
    
    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """
        Get recent alert history.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        return self.alert_history[-limit:] if self.alert_history else []
    
    def get_alert_stats(self) -> Dict:
        """
        Get alert statistics.
        
        Returns:
            Dictionary with alert statistics
        """
        if not self.alert_history:
            return {
                'total_alerts': 0,
                'by_level': {},
                'recent_alerts': 0
            }
        
        # Count by level
        level_counts = {}
        for alert in self.alert_history:
            level = alert['alert_level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Count recent alerts (last hour)
        recent_cutoff = datetime.now().timestamp() - 3600
        recent_alerts = sum(
            1 for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']).timestamp() > recent_cutoff
        )
        
        return {
            'total_alerts': len(self.alert_history),
            'by_level': level_counts,
            'recent_alerts': recent_alerts
        }
    
    def clear_history(self):
        """Clear alert history."""
        self.alert_history = []
        logger.info("Alert history cleared")

def main():
    """Test the alert handler."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create alert handler
    handler = AlertHandler()
    
    # Test with different prediction scenarios
    test_predictions = [
        {
            'sensor_id': 'sensor_001',
            'risk_score': 0.3,
            'failure_probability': 0.2,
            'anomaly_score': 0.1,
            'is_failure': False,
            'is_anomaly': False
        },
        {
            'sensor_id': 'sensor_002',
            'risk_score': 0.8,
            'failure_probability': 0.75,
            'anomaly_score': -0.2,
            'is_failure': False,
            'is_anomaly': True
        },
        {
            'sensor_id': 'sensor_003',
            'risk_score': 0.95,
            'failure_probability': 0.9,
            'anomaly_score': -0.5,
            'is_failure': True,
            'is_anomaly': True
        }
    ]
    
    print("Testing Alert Handler")
    print("=" * 50)
    
    for i, prediction in enumerate(test_predictions):
        print(f"\nTest {i+1}: Processing prediction for {prediction['sensor_id']}")
        result = handler.process_prediction(prediction)
        print(f"Alert Level: {result['alert_level']}")
        print(f"Console Sent: {result['console_sent']}")
        print(f"Webhook Sent: {result['webhook_sent']}")
    
    # Get statistics
    stats = handler.get_alert_stats()
    print(f"\nAlert Statistics:")
    print(f"Total Alerts: {stats['total_alerts']}")
    print(f"By Level: {stats['by_level']}")
    print(f"Recent Alerts: {stats['recent_alerts']}")

if __name__ == "__main__":
    main()
