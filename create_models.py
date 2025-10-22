"""
Simple script to create pre-trained models for the system.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Create model directory
os.makedirs("model", exist_ok=True)

print("Creating pre-trained models...")

# Generate synthetic data
np.random.seed(42)
n_samples = 5000

# Generate timestamps
timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1min')

# Generate sensor data
data = []
sensor_ids = ['sensor_001', 'sensor_002', 'sensor_003']
failure_rate = 0.15

for i, timestamp in enumerate(timestamps):
    sensor_id = np.random.choice(sensor_ids)
    is_failure = np.random.random() < failure_rate
    
    if is_failure:
        # Failure patterns
        temp = np.random.normal(75, 10)
        vibration = np.random.normal(4, 1)
        pressure = np.random.normal(2, 0.5)
        label = "failure"
    else:
        # Normal patterns
        temp = np.random.normal(30, 5)
        vibration = np.random.normal(1, 0.3)
        pressure = np.random.normal(5, 0.8)
        label = "normal"
    
    data.append({
        'timestamp': timestamp.isoformat(),
        'sensor_id': sensor_id,
        'temperature': round(temp, 2),
        'vibration': round(vibration, 2),
        'pressure': round(pressure, 2),
        'label': label
    })

df = pd.DataFrame(data)
print(f"Generated {len(df)} samples with {df['label'].value_counts().to_dict()} distribution")

# Create simple features
df['temp_vib_interaction'] = df['temperature'] * df['vibration']
df['temp_pressure_interaction'] = df['temperature'] * df['pressure']
df['vib_pressure_interaction'] = df['vibration'] * df['pressure']

# Prepare features
feature_columns = ['temperature', 'vibration', 'pressure', 'temp_vib_interaction', 'temp_pressure_interaction', 'vib_pressure_interaction']
X = df[feature_columns]
y = df['label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
print("Training Isolation Forest...")
isolation_forest = IsolationForest(contamination=0.1, random_state=42)
isolation_forest.fit(X_train_scaled)

print("Training Random Forest...")
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train_scaled, y_train)

print("Training SGD Classifier...")
sgd_classifier = SGDClassifier(loss='log_loss', random_state=42)
sgd_classifier.fit(X_train_scaled, y_train)

# Save models
print("Saving models...")
joblib.dump(isolation_forest, 'model/isolation_forest.pkl')
joblib.dump(random_forest, 'model/random_forest.pkl')
joblib.dump(sgd_classifier, 'model/sgd_classifier.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(label_encoder, 'model/label_encoder.pkl')
joblib.dump(feature_columns, 'model/feature_columns.pkl')

print("Models saved successfully!")
print(f"Feature columns: {feature_columns}")

# Test models
print("\nTesting models...")
isolation_scores = isolation_forest.decision_function(X_test_scaled)
rf_predictions = random_forest.predict(X_test_scaled)
rf_probabilities = random_forest.predict_proba(X_test_scaled)
sgd_predictions = sgd_classifier.predict(X_test_scaled)
sgd_probabilities = sgd_classifier.predict_proba(X_test_scaled)

print(f"Isolation Forest - Mean score: {isolation_scores.mean():.4f}")
print(f"Random Forest - Accuracy: {(rf_predictions == y_test).mean():.4f}")
print(f"SGD Classifier - Accuracy: {(sgd_predictions == y_test).mean():.4f}")

print("\nPre-trained models created successfully!")
