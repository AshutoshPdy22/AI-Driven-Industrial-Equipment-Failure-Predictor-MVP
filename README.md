# 🏭 AI-Driven Industrial Equipment Failure Predictor (MVP)

A complete Python system for real-time industrial equipment monitoring and failure prediction using classical machine learning models.

## 🎯 Overview

This system uses API-based sensor data (temperature, vibration, pressure) to:
- Monitor industrial equipment health in real-time
- Predict failures before they happen using classical ML
- Visualize results in a Streamlit dashboard
- Send alerts when risk is high

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Simulated     │    │   Data          │    │   SQLite        │
│   API Server    │───▶│   Ingestion     │───▶│   Database      │
│   (FastAPI)     │    │   (Client)      │    │   (Storage)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data          │    │   Feature       │    │   ML Models     │
│   Preprocessing │───▶│   Engineering   │───▶│   (3 Models)    │
│   (Cleaning)    │    │   (Rolling)     │    │   (Prediction)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SHAP          │    │   Alert         │    │   Streamlit     │
│   Explainability│    │   System        │    │   Dashboard     │
│   (Visualization)│   │   (Notifications)│   │   (Real-time)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd "AI-Driven Industrial Equipment Failure Predictor (MVP)"

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models (Optional - Pre-trained models included)

```bash
python model/train_model.py
```

### 3. Start the System

**Terminal 1 - API Server:**
```bash
python data_fetch/fetch_api_data.py
```

**Terminal 2 - Main Monitor:**
```bash
python main.py
```

**Terminal 3 - Dashboard:**
```bash
streamlit run dashboard/app.py
```

## 📁 Project Structure

```
/equipment-failure-predictor
│
├── data_fetch/
│   ├── __init__.py
│   ├── fetch_api_data.py          # Simulated API server
│   └── api_client.py              # Data fetching client
│
├── preprocessing/
│   ├── __init__.py
│   └── clean_data.py              # Data cleaning & normalization
│
├── features/
│   ├── __init__.py
│   └── feature_engineering.py    # Feature extraction
│
├── model/
│   ├── __init__.py
│   ├── train_model.py             # Model training
│   └── predict.py                 # Prediction engine
│
├── explainability/
│   ├── __init__.py
│   └── explain_model.py          # SHAP explanations
│
├── alerts/
│   ├── __init__.py
│   └── alert_handler.py           # Alert system
│
├── dashboard/
│   ├── __init__.py
│   └── app.py                     # Streamlit dashboard
│
├── main.py                        # Main orchestrator
├── config.py                      # Configuration
├── requirements.txt               # Dependencies
├── env.example                   # Environment variables
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

Copy `env.example` to `.env` and configure:

```bash
# Webhook Configuration (Optional)
WEBHOOK_ENABLED=false
SLACK_WEBHOOK_URL=your_slack_webhook_url
TELEGRAM_WEBHOOK_URL=your_telegram_webhook_url

# Database Configuration
DB_PATH=./equipment_data.db

# API Configuration
API_BASE_URL=http://localhost:8000
```

### Key Settings in `config.py`

- **Failure Threshold**: 0.7 (70% probability triggers alerts)
- **Anomaly Threshold**: -0.1 (Isolation Forest score)
- **Refresh Interval**: 5 seconds (Dashboard)
- **Update Interval**: 30 seconds (Data fetching)

## 🤖 Machine Learning Models

### 1. Isolation Forest
- **Purpose**: Anomaly detection
- **Output**: Anomaly scores (-1 to 1)
- **Use Case**: Identify unusual sensor patterns

### 2. Random Forest
- **Purpose**: Binary classification (fail/no-fail)
- **Output**: Failure probabilities (0 to 1)
- **Use Case**: Primary failure prediction

### 3. SGD Classifier
- **Purpose**: Online learning capability
- **Output**: Failure probabilities (0 to 1)
- **Use Case**: Adaptive learning from new data

### Combined Risk Score
```
Risk Score = 0.3 × Anomaly_Risk + 0.5 × Failure_Prob + 0.2 × Online_Prob
```

## 📊 Dashboard Features

### Real-time Monitoring
- **Equipment Status Cards**: Live health status for each sensor
- **Sensor Trends**: Line charts for temperature, vibration, pressure
- **Risk Gauge**: Overall system risk level
- **Alert History**: Recent alerts and notifications

### Model Explainability
- **SHAP Waterfall Plot**: Feature importance for latest prediction
- **Feature Importance**: Top contributing factors to predictions
- **Interactive Charts**: Zoom, pan, and explore data

### Auto-refresh
- **Configurable Interval**: 5-60 seconds
- **Real-time Updates**: Automatic data refresh
- **Live Predictions**: Continuous model inference

## 🚨 Alert System

### Alert Levels
- **🚨 EMERGENCY**: Risk ≥ 90% or confirmed failure
- **⚠️ CRITICAL**: Risk ≥ 80%
- **⚠️ WARNING**: Risk ≥ 70% or anomaly detected
- **ℹ️ INFO**: Normal operation updates

### Alert Channels
- **Console Logging**: Always enabled
- **Slack Webhooks**: Optional, via environment variables
- **Telegram Webhooks**: Optional, via environment variables

## 🔍 Feature Engineering

### Rolling Statistics
- **Windows**: 5, 10, 20 periods
- **Metrics**: Mean, std, min, max, range
- **Purpose**: Capture temporal patterns

### Rate of Change
- **First Derivative**: Current - Previous
- **Percentage Change**: Rate of change %
- **Acceleration**: Second derivative

### Spike Detection
- **Z-Score**: Standardized anomaly detection
- **Threshold**: |z-score| > 2
- **Magnitude**: Absolute z-score value

### Interaction Features
- **Sensor Cross-correlation**: Temperature × Vibration
- **Combined Health Score**: Multi-sensor health metric
- **Lag Features**: Historical values (1, 2, 3, 5 periods)

## 📈 Usage Examples

### Basic Monitoring
```python
from data_fetch.api_client import SensorDataClient
from model.predict import EquipmentFailurePredictor

# Fetch data
client = SensorDataClient("http://localhost:8000", "data.db")
client.fetch_and_store()

# Make prediction
predictor = EquipmentFailurePredictor()
predictor.load_models()
prediction = predictor.get_latest_prediction(df)
print(f"Risk Score: {prediction['risk_score']}")
```

### Custom Alerts
```python
from alerts.alert_handler import AlertHandler

handler = AlertHandler()
result = handler.process_prediction(prediction_data)
if result['alert_created']:
    print(f"Alert: {result['alert_level']}")
```

### SHAP Explanations
```python
from explainability.explain_model import ModelExplainer

explainer = ModelExplainer()
explainer.load_models()
explanation = explainer.get_explanation_summary(df)
print(f"Top Features: {explanation['feature_importance']['top_features']}")
```

## 🛠️ Development

### Running Tests
```bash
# Test individual components
python data_fetch/api_client.py
python model/train_model.py
python model/predict.py
python explainability/explain_model.py
python alerts/alert_handler.py
```

### Adding New Features
1. **New Sensors**: Update `DATA_CONFIG` in `config.py`
2. **New Models**: Extend `ModelTrainer` class in `train_model.py`
3. **New Alerts**: Add methods to `AlertHandler` class
4. **New Visualizations**: Extend `dashboard/app.py`

## 🐛 Troubleshooting

### Common Issues

**1. "Models not found" error**
```bash
# Solution: Train models first
python model/train_model.py
```

**2. "API not healthy" error**
```bash
# Solution: Start API server
python data_fetch/fetch_api_data.py
```

**3. "No data available" in dashboard**
```bash
# Solution: Check if main.py is running and collecting data
python main.py
```

**4. Import errors**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Log Files
- **Main Log**: `equipment_monitor.log`
- **Dashboard Logs**: Streamlit console output
- **API Logs**: FastAPI console output

## 📋 System Requirements

- **Python**: 3.8+
- **Memory**: 2GB+ RAM
- **Storage**: 1GB+ free space
- **Network**: Local (API runs on localhost)

## 🔮 Future Enhancements

- **Docker Support**: Containerized deployment
- **MQTT Integration**: Real-time sensor data streams
- **Advanced Models**: LSTM, Transformer architectures
- **Cloud Deployment**: AWS/Azure integration
- **Mobile App**: React Native dashboard
- **Edge Computing**: Raspberry Pi deployment

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

---

**Built with ❤️ for Industrial IoT and Predictive Maintenance**
