"""
Demo script to show the Equipment Failure Predictor system in action.
This script simulates the complete pipeline without requiring ML models.
"""
import time
import random
import json
from datetime import datetime
import os

def simulate_sensor_data():
    """Simulate sensor data generation."""
    sensors = ['sensor_001', 'sensor_002', 'sensor_003']
    sensor_id = random.choice(sensors)
    
    # Simulate failure patterns (10% chance)
    is_failure = random.random() < 0.1
    
    if is_failure:
        # Failure patterns
        temperature = random.uniform(70, 90)  # High temperature
        vibration = random.uniform(3, 6)      # High vibration
        pressure = random.uniform(1, 3)       # Low pressure
        label = "failure"
    else:
        # Normal patterns
        temperature = random.uniform(20, 40)  # Normal temperature
        vibration = random.uniform(0.5, 2)   # Normal vibration
        pressure = random.uniform(4, 8)       # Normal pressure
        label = "normal"
    
    return {
        'timestamp': datetime.now().isoformat(),
        'sensor_id': sensor_id,
        'temperature': round(temperature, 2),
        'vibration': round(vibration, 2),
        'pressure': round(pressure, 2),
        'label': label
    }

def calculate_risk_score(sensor_data):
    """Calculate a simple risk score based on sensor values."""
    temp = sensor_data['temperature']
    vib = sensor_data['vibration']
    pressure = sensor_data['pressure']
    
    # Simple risk calculation
    temp_risk = max(0, (temp - 30) / 60)  # Higher temp = higher risk
    vib_risk = max(0, (vib - 1) / 5)     # Higher vibration = higher risk
    pressure_risk = max(0, (5 - pressure) / 5)  # Lower pressure = higher risk
    
    # Combined risk score (0-1)
    risk_score = (temp_risk + vib_risk + pressure_risk) / 3
    return min(1.0, risk_score)

def determine_alert_level(risk_score):
    """Determine alert level based on risk score."""
    if risk_score >= 0.9:
        return "🚨 EMERGENCY"
    elif risk_score >= 0.8:
        return "⚠️ CRITICAL"
    elif risk_score >= 0.7:
        return "⚠️ WARNING"
    else:
        return "✅ NORMAL"

def print_sensor_status(sensor_data, risk_score, alert_level):
    """Print sensor status in a formatted way."""
    print(f"""
{'='*60}
SENSOR STATUS UPDATE
{'='*60}
Timestamp: {sensor_data['timestamp']}
Equipment: {sensor_data['sensor_id']}
Status: {alert_level}
Risk Score: {risk_score:.4f}

Sensor Readings:
  Temperature: {sensor_data['temperature']}°C
  Vibration: {sensor_data['vibration']}
  Pressure: {sensor_data['pressure']}
  Label: {sensor_data['label']}

Risk Assessment:
  Temperature Risk: {max(0, (sensor_data['temperature'] - 30) / 60):.3f}
  Vibration Risk: {max(0, (sensor_data['vibration'] - 1) / 5):.3f}
  Pressure Risk: {max(0, (5 - sensor_data['pressure']) / 5):.3f}
{'='*60}
""")

def main():
    """Main demo function."""
    print("🏭 Equipment Failure Predictor - Demo Mode")
    print("=" * 60)
    print("This demo simulates the complete system without ML models.")
    print("Press Ctrl+C to stop the demo.")
    print("=" * 60)
    
    try:
        while True:
            # Generate sensor data
            sensor_data = simulate_sensor_data()
            
            # Calculate risk score
            risk_score = calculate_risk_score(sensor_data)
            
            # Determine alert level
            alert_level = determine_alert_level(risk_score)
            
            # Print status
            print_sensor_status(sensor_data, risk_score, alert_level)
            
            # Simulate processing time
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")
        print("Thank you for trying the Equipment Failure Predictor!")

if __name__ == "__main__":
    main()
