"""
Simulated API server for generating realistic sensor data with failure patterns.
"""
import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Industrial Sensor API", version="1.0.0")

class SensorReading(BaseModel):
    timestamp: str
    sensor_id: str
    temperature: float
    vibration: float
    pressure: float
    label: str  # "normal" or "failure"

class SensorDataGenerator:
    """Generates realistic sensor data with configurable failure patterns."""
    
    def __init__(self, failure_rate: float = 0.1):
        self.failure_rate = failure_rate
        self.sensor_states = {}
        self.failure_countdown = {}
        
    def generate_normal_reading(self, sensor_id: str) -> Dict:
        """Generate normal sensor readings with realistic patterns."""
        if sensor_id not in self.sensor_states:
            self.sensor_states[sensor_id] = {
                'base_temp': random.uniform(25, 35),
                'base_vibration': random.uniform(0.5, 1.5),
                'base_pressure': random.uniform(3, 7)
            }
        
        state = self.sensor_states[sensor_id]
        
        # Add some realistic variation
        temp = state['base_temp'] + random.gauss(0, 2)
        vibration = state['base_vibration'] + random.gauss(0, 0.3)
        pressure = state['base_pressure'] + random.gauss(0, 0.5)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "sensor_id": sensor_id,
            "temperature": round(temp, 2),
            "vibration": round(vibration, 2),
            "pressure": round(pressure, 2),
            "label": "normal"
        }
    
    def generate_failure_reading(self, sensor_id: str) -> Dict:
        """Generate sensor readings showing failure patterns."""
        if sensor_id not in self.failure_countdown:
            self.failure_countdown[sensor_id] = random.randint(5, 15)
        
        countdown = self.failure_countdown[sensor_id]
        
        if countdown > 0:
            # Pre-failure: gradual degradation
            degradation = (15 - countdown) / 15
            
            temp = 30 + degradation * 40 + random.gauss(0, 3)
            vibration = 1.0 + degradation * 4 + random.gauss(0, 0.5)
            pressure = 5.0 - degradation * 2 + random.gauss(0, 0.3)
            
            self.failure_countdown[sensor_id] -= 1
            
            return {
                "timestamp": datetime.now().isoformat(),
                "sensor_id": sensor_id,
                "temperature": round(temp, 2),
                "vibration": round(vibration, 2),
                "pressure": round(pressure, 2),
                "label": "failure"
            }
        else:
            # Reset failure countdown
            self.failure_countdown[sensor_id] = random.randint(5, 15)
            return self.generate_normal_reading(sensor_id)
    
    def generate_reading(self, sensor_id: str) -> Dict:
        """Generate a sensor reading with configurable failure probability."""
        if random.random() < self.failure_rate:
            return self.generate_failure_reading(sensor_id)
        else:
            return self.generate_normal_reading(sensor_id)

# Global data generator
data_generator = SensorDataGenerator(failure_rate=0.1)

@app.get("/")
async def root():
    return {"message": "Industrial Sensor API", "status": "running"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/sensors")
async def get_sensor_data(sensor_id: Optional[str] = None):
    """Get sensor data for all sensors or a specific sensor."""
    sensor_ids = ["sensor_001", "sensor_002", "sensor_003"]
    
    if sensor_id and sensor_id not in sensor_ids:
        raise HTTPException(status_code=404, detail="Sensor not found")
    
    target_sensors = [sensor_id] if sensor_id else sensor_ids
    
    readings = []
    for sid in target_sensors:
        reading = data_generator.generate_reading(sid)
        readings.append(reading)
    
    return {"readings": readings, "count": len(readings)}

@app.get("/api/sensors/{sensor_id}")
async def get_sensor_data_by_id(sensor_id: str):
    """Get data for a specific sensor."""
    return await get_sensor_data(sensor_id)

@app.post("/api/sensors/batch")
async def get_batch_sensor_data(count: int = 10):
    """Get multiple readings for all sensors."""
    all_readings = []
    for _ in range(count):
        readings = await get_sensor_data()
        all_readings.extend(readings["readings"])
    
    return {"readings": all_readings, "count": len(all_readings)}

if __name__ == "__main__":
    print("Starting Industrial Sensor API server...")
    print("API will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
