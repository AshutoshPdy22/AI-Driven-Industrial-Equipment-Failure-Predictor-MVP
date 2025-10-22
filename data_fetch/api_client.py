"""
Client for fetching sensor data from the API and storing in SQLite database.
"""
import sqlite3
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorDataClient:
    """Client for fetching and storing sensor data."""
    
    def __init__(self, api_base_url: str, db_path: str):
        self.api_base_url = api_base_url
        self.db_path = db_path
        self.session = requests.Session()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with sensor readings table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                sensor_id TEXT NOT NULL,
                temperature REAL NOT NULL,
                vibration REAL NOT NULL,
                pressure REAL NOT NULL,
                label TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON sensor_readings(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sensor_id 
            ON sensor_readings(sensor_id)
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def fetch_sensor_data(self, sensor_id: Optional[str] = None) -> List[Dict]:
        """Fetch sensor data from API."""
        try:
            if sensor_id:
                url = f"{self.api_base_url}/api/sensors/{sensor_id}"
            else:
                url = f"{self.api_base_url}/api/sensors"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get("readings", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch sensor data: {e}")
            return []
    
    def store_sensor_data(self, readings: List[Dict]) -> int:
        """Store sensor readings in database."""
        if not readings:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stored_count = 0
        for reading in readings:
            try:
                cursor.execute('''
                    INSERT INTO sensor_readings 
                    (timestamp, sensor_id, temperature, vibration, pressure, label)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    reading['timestamp'],
                    reading['sensor_id'],
                    reading['temperature'],
                    reading['vibration'],
                    reading['label']
                ))
                stored_count += 1
            except sqlite3.Error as e:
                logger.error(f"Failed to store reading: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored {stored_count} sensor readings")
        return stored_count
    
    def get_latest_readings(self, limit: int = 100) -> pd.DataFrame:
        """Get latest sensor readings from database."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, sensor_id, temperature, vibration, pressure, label
            FROM sensor_readings
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df
    
    def get_sensor_history(self, sensor_id: str, hours: int = 24) -> pd.DataFrame:
        """Get sensor history for a specific sensor."""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        cutoff_str = datetime.fromtimestamp(cutoff_time).isoformat()
        
        query = '''
            SELECT timestamp, sensor_id, temperature, vibration, pressure, label
            FROM sensor_readings
            WHERE sensor_id = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=(sensor_id, cutoff_str))
        conn.close()
        
        return df
    
    def fetch_and_store(self, sensor_id: Optional[str] = None) -> int:
        """Fetch data from API and store in database."""
        readings = self.fetch_sensor_data(sensor_id)
        return self.store_sensor_data(readings)
    
    def is_api_healthy(self) -> bool:
        """Check if API is responding."""
        try:
            response = self.session.get(f"{self.api_base_url}/api/health", timeout=5)
            return response.status_code == 200
        except:
            return False

def main():
    """Test the API client."""
    from config import API_BASE_URL, DB_PATH
    
    client = SensorDataClient(API_BASE_URL, str(DB_PATH))
    
    if not client.is_api_healthy():
        print("API is not healthy. Make sure the API server is running.")
        return
    
    print("Fetching sensor data...")
    stored_count = client.fetch_and_store()
    print(f"Stored {stored_count} readings")
    
    # Get latest readings
    latest = client.get_latest_readings(10)
    print("\nLatest readings:")
    print(latest.to_string())

if __name__ == "__main__":
    main()
