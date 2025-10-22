"""
Streamlit dashboard for real-time equipment failure monitoring.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
import sys
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetch.api_client import SensorDataClient
from model.predict import EquipmentFailurePredictor
from explainability.explain_model import ModelExplainer
from alerts.alert_handler import AlertHandler
from config import API_BASE_URL, DB_PATH, DASHBOARD_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Equipment Failure Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-emergency {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-critical {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-warning {
        background-color: #fffde7;
        border-left: 4px solid #ffeb3b;
    }
    .alert-info {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)  # Cache for 1 minute
def load_latest_data():
    """Load latest sensor data."""
    try:
        client = SensorDataClient(API_BASE_URL, str(DB_PATH))
        df = client.get_latest_readings(limit=100)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_prediction_models():
    """Load prediction models."""
    try:
        predictor = EquipmentFailurePredictor()
        if predictor.load_models():
            return predictor
        else:
            st.error("Failed to load prediction models")
            return None
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_explainer():
    """Load model explainer."""
    try:
        explainer = ModelExplainer()
        if explainer.load_models():
            return explainer
        else:
            return None
    except Exception as e:
        st.error(f"Failed to load explainer: {e}")
        return None

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a metric card."""
    if delta is not None:
        st.metric(title, value, delta=delta, delta_color=delta_color)
    else:
        st.metric(title, value)

def create_risk_gauge(risk_score):
    """Create a risk gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score (%)"},
        delta = {'reference': 70},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_sensor_trends(df):
    """Create sensor trend charts."""
    if df.empty:
        return go.Figure()
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Temperature (°C)', 'Vibration', 'Pressure'),
        vertical_spacing=0.1
    )
    
    # Temperature plot
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['temperature'],
            mode='lines+markers',
            name='Temperature',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Vibration plot
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['vibration'],
            mode='lines+markers',
            name='Vibration',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    # Pressure plot
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['pressure'],
            mode='lines+markers',
            name='Pressure',
            line=dict(color='green')
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=600,
        title_text="Sensor Trends (Last 100 Readings)",
        showlegend=False
    )
    
    return fig

def create_equipment_status_cards(df, predictor):
    """Create equipment status cards."""
    if df.empty or predictor is None:
        return
    
    # Get latest readings for each sensor
    latest_readings = df.groupby('sensor_id').tail(1)
    
    for _, row in latest_readings.iterrows():
        sensor_id = row['sensor_id']
        
        # Get prediction for this sensor
        sensor_data = df[df['sensor_id'] == sensor_id].tail(1)
        prediction = predictor.get_latest_prediction(sensor_data)
        
        if prediction:
            risk_score = prediction.get('risk_score', 0)
            failure_prob = prediction.get('failure_probability', 0)
            is_anomaly = prediction.get('is_anomaly', False)
            is_failure = prediction.get('is_failure', False)
            
            # Determine status
            if is_failure or risk_score >= 0.9:
                status = "🚨 CRITICAL"
                status_class = "alert-emergency"
            elif risk_score >= 0.7:
                status = "⚠️ WARNING"
                status_class = "alert-warning"
            elif is_anomaly:
                status = "⚠️ ANOMALY"
                status_class = "alert-critical"
            else:
                status = "✅ NORMAL"
                status_class = "alert-info"
            
            # Create card
            with st.container():
                st.markdown(f"""
                <div class="metric-card {status_class}">
                    <h4>{sensor_id}</h4>
                    <p><strong>Status:</strong> {status}</p>
                    <p><strong>Risk Score:</strong> {risk_score:.3f}</p>
                    <p><strong>Failure Probability:</strong> {failure_prob:.3f}</p>
                    <p><strong>Temperature:</strong> {row['temperature']:.1f}°C</p>
                    <p><strong>Vibration:</strong> {row['vibration']:.2f}</p>
                    <p><strong>Pressure:</strong> {row['pressure']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

def create_shap_visualization(df, explainer):
    """Create SHAP visualization."""
    if df.empty or explainer is None:
        st.warning("No data available for SHAP visualization")
        return
    
    try:
        # Get latest sample
        latest_sample = df.tail(1)
        
        # Get SHAP explanation
        explanation = explainer.get_explanation_summary(latest_sample)
        
        if explanation and 'feature_importance' in explanation:
            importance = explanation['feature_importance']
            
            if 'top_features' in importance:
                # Create feature importance chart
                features = importance['top_features'][:10]
                
                if features:
                    feature_names = [f['feature'] for f in features]
                    shap_values = [f['shap_value'] for f in features]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=shap_values,
                            y=feature_names,
                            orientation='h',
                            marker_color=['red' if x > 0 else 'blue' for x in shap_values]
                        )
                    ])
                    
                    fig.update_layout(
                        title="SHAP Feature Importance (Latest Prediction)",
                        xaxis_title="SHAP Value",
                        yaxis_title="Feature",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show feature details
                    st.subheader("Feature Details")
                    feature_df = pd.DataFrame(features)
                    st.dataframe(feature_df[['feature', 'value', 'shap_value']], use_container_width=True)
        
    except Exception as e:
        st.error(f"Failed to create SHAP visualization: {e}")

def main():
    """Main dashboard function."""
    st.title("🏭 Equipment Failure Predictor Dashboard")
    st.markdown("Real-time monitoring of industrial equipment health and failure prediction")
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 10)
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        df = load_latest_data()
        predictor = load_prediction_models()
        explainer = load_explainer()
    
    if df.empty:
        st.error("No data available. Make sure the API server is running and data is being collected.")
        return
    
    if predictor is None:
        st.error("Prediction models not available. Please run the training script first.")
        return
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sensors", len(df['sensor_id'].unique()))
    
    with col2:
        st.metric("Latest Readings", len(df))
    
    with col3:
        # Calculate average risk score
        if not df.empty:
            latest_predictions = []
            for sensor_id in df['sensor_id'].unique():
                sensor_data = df[df['sensor_id'] == sensor_id].tail(1)
                prediction = predictor.get_latest_prediction(sensor_data)
                if prediction and 'risk_score' in prediction:
                    latest_predictions.append(prediction['risk_score'])
            
            avg_risk = np.mean(latest_predictions) if latest_predictions else 0
            st.metric("Average Risk Score", f"{avg_risk:.3f}")
        else:
            st.metric("Average Risk Score", "N/A")
    
    with col4:
        # Count alerts
        alert_handler = AlertHandler()
        stats = alert_handler.get_alert_stats()
        st.metric("Total Alerts", stats['total_alerts'])
    
    # Equipment Status Cards
    st.header("📊 Equipment Status")
    create_equipment_status_cards(df, predictor)
    
    # Risk Assessment
    st.header("⚠️ Risk Assessment")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sensor trends
        st.subheader("Sensor Trends")
        trend_fig = create_sensor_trends(df)
        st.plotly_chart(trend_fig, use_container_width=True)
    
    with col2:
        # Risk gauge
        st.subheader("Overall Risk Level")
        if not df.empty:
            # Calculate overall risk score
            latest_predictions = []
            for sensor_id in df['sensor_id'].unique():
                sensor_data = df[df['sensor_id'] == sensor_id].tail(1)
                prediction = predictor.get_latest_prediction(sensor_data)
                if prediction and 'risk_score' in prediction:
                    latest_predictions.append(prediction['risk_score'])
            
            overall_risk = np.mean(latest_predictions) if latest_predictions else 0
            risk_gauge = create_risk_gauge(overall_risk)
            st.plotly_chart(risk_gauge, use_container_width=True)
    
    # SHAP Explanation
    if explainer:
        st.header("🔍 Model Explanation")
        create_shap_visualization(df, explainer)
    
    # Alert History
    st.header("🚨 Alert History")
    alert_handler = AlertHandler()
    alert_history = alert_handler.get_alert_history(limit=20)
    
    if alert_history:
        alert_df = pd.DataFrame(alert_history)
        st.dataframe(
            alert_df[['timestamp', 'sensor_id', 'alert_level', 'message', 'risk_score']],
            use_container_width=True
        )
    else:
        st.info("No alerts generated yet")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
