import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Riyadh Air Quality", layout="centered")
st.title("Riyadh Air Quality Risk")
st.markdown("Predict whether the next hour will have a high PM2.5 pollution risk.")

with st.form("prediction-form"):
    col1, col2 = st.columns(2)
    with col1:
        pm2_5 = st.number_input("Current PM2.5", 0, 1000, 25)
        pm10 = st.number_input("Current PM10", 0, 1500, 60)
        temperature = st.number_input("Temperature (°C)", -20, 65, 32)
        humidity = st.number_input("Relative humidity (%)", 0, 100, 25)
        wind = st.number_input("Wind speed (km/h)", 0, 200, 12)
    with col2:
        hour = st.slider("Hour (0-23)", 0, 23, 12)
        day = st.slider("Day of week (0=Mon, 6=Sun)", 0, 6, 2)
        lag_1 = st.number_input("PM2.5 one hour ago", 0, 1000, 24)
        lag_3 = st.number_input("PM2.5 three hours ago", 0, 1000, 22)
        rolling = st.number_input("Six-hour PM2.5 avg", 0, 1000, 23)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "pm2_5": pm2_5, "pm10": pm10, "temperature_2m": temperature,
        "relative_humidity_2m": humidity, "wind_speed_10m": wind,
        "hour": hour, "day_of_week": day, "pm2_5_lag_1": lag_1,
        "pm2_5_lag_3": lag_3, "pm2_5_rolling_mean_6": rolling,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        st.metric("High-risk probability", f"{result['probability']:.1%}")
        if result["risk_level"] == "high":
            st.error(f"Risk level: {result['risk_level'].upper()} — high pollution expected")
        else:
            st.success(f"Risk level: {result['risk_level'].upper()} — normal conditions")
    except requests.RequestException as error:
        st.error(f"API request failed: {error}")

st.divider()
try:
    status = requests.get(f"{API_URL}/health", timeout=5).json()
    st.caption(f"API status: {status['status']}")
except requests.RequestException:
    st.caption("API status: unavailable")
