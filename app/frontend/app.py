"""
BlazeNet - Forest Fire Prediction & Simulation Dashboard
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
import plotly.express as px
from datetime import date
import numpy as np

# Page config
st.set_page_config(
    page_title="BlazeNet - Fire Prediction Dashboard",
    page_icon="🔥",
    layout="wide"
)

# API URL
API_BASE_URL = "http://localhost:8000/api/v1"

def main():
    """Main dashboard."""
    
    st.title("🔥 BlazeNet - Forest Fire Prediction & Simulation")
    st.markdown("**Advanced Geospatial ML System for Fire Risk Assessment**")
    
    # Sidebar
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "🔥 Fire Prediction", "🌪️ Fire Simulation", "📊 Data Sources"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔥 Fire Prediction":
        show_fire_prediction()
    elif page == "🌪️ Fire Simulation":
        show_fire_simulation()
    elif page == "📊 Data Sources":
        show_data_sources()

def show_dashboard():
    """Dashboard overview."""
    
    st.header("🏠 Dashboard Overview")
    
    # Status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                st.metric("🟢 System Status", "Healthy")
            else:
                st.metric("🔴 System Status", "Error")
        except:
            st.metric("🔴 System Status", "Offline")
    
    with col2:
        st.metric("🤖 Models", "2", "U-Net, LSTM")
    
    with col3:
        st.metric("🌍 Regions", "India", "Focus: Uttarakhand")
    
    with col4:
        st.metric("📊 Data Sources", "4+", "Satellite, Weather")
    
    # Map
    st.subheader("🗺️ Uttarakhand Region")
    
    m = folium.Map(location=[30.0668, 79.0193], zoom_start=8)
    
    # Sample points
    points = [
        {"lat": 30.0668, "lon": 79.0193, "risk": "High", "name": "Dehradun"},
        {"lat": 29.3803, "lon": 79.4636, "risk": "Medium", "name": "Nainital"},
        {"lat": 30.7268, "lon": 79.0122, "risk": "Low", "name": "Rishikesh"},
    ]
    
    for point in points:
        color = "red" if point["risk"] == "High" else "orange" if point["risk"] == "Medium" else "green"
        folium.CircleMarker(
            location=[point["lat"], point["lon"]],
            radius=8,
            popup=f"{point['name']}<br>Risk: {point['risk']}",
            color=color,
            fill=True
        ).add_to(m)
    
    st_folium(m, width=1000, height=400)

def show_fire_prediction():
    """Fire prediction interface."""
    
    st.header("🔥 Fire Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_lat = st.number_input("Min Latitude", value=29.0)
            min_lon = st.number_input("Min Longitude", value=77.5)
            prediction_date = st.date_input("Date", value=date.today())
        
        with col2:
            max_lat = st.number_input("Max Latitude", value=31.5)
            max_lon = st.number_input("Max Longitude", value=81.0)
            model_type = st.selectbox("Model", ["unet", "lstm"])
        
        st.subheader("🌤️ Weather Data")
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.number_input("Temperature (°C)", value=30.0)
            humidity = st.number_input("Humidity (%)", value=40.0)
        
        with col2:
            wind_speed = st.number_input("Wind Speed (m/s)", value=5.0)
            wind_direction = st.number_input("Wind Direction (°)", value=180.0)
        
        submitted = st.form_submit_button("🔥 Run Prediction")
        
        if submitted:
            request_data = {
                "region": {
                    "min_lat": min_lat, "max_lat": max_lat,
                    "min_lon": min_lon, "max_lon": max_lon
                },
                "date": prediction_date.isoformat(),
                "weather_data": {
                    "temperature": temperature,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "wind_direction": wind_direction
                },
                "model_type": model_type
            }
            
            with st.spinner("Running prediction..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/predict/fire-probability",
                        json=request_data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Prediction completed!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        stats = result["statistics"]
                        
                        with col1:
                            st.metric("High Risk Area", f"{stats['high_risk_area_km2']:.1f} km²")
                        with col2:
                            st.metric("Medium Risk Area", f"{stats['medium_risk_area_km2']:.1f} km²")
                        with col3:
                            st.metric("Max Probability", f"{stats['max_probability']:.2f}")
                        with col4:
                            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                        
                        # Show map
                        center_lat = (min_lat + max_lat) / 2
                        center_lon = (min_lon + max_lon) / 2
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                        
                        bounds = [
                            [min_lat, min_lon], [max_lat, min_lon],
                            [max_lat, max_lon], [min_lat, max_lon]
                        ]
                        
                        folium.Polygon(
                            locations=bounds,
                            color='red',
                            weight=2,
                            fill=True,
                            fillOpacity=0.3,
                            popup=f"Prediction Region<br>Model: {model_type}"
                        ).add_to(m)
                        
                        st_folium(m, width=1000, height=400)
                        
                    else:
                        st.error(f"❌ Prediction failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

def show_fire_simulation():
    """Fire simulation interface."""
    
    st.header("🌪️ Fire Simulation")
    
    with st.form("simulation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_lat = st.number_input("Min Lat", value=29.0)
            min_lon = st.number_input("Min Lon", value=77.5)
            ignition_lat = st.number_input("Ignition Lat", value=30.0)
        
        with col2:
            max_lat = st.number_input("Max Lat", value=31.5)
            max_lon = st.number_input("Max Lon", value=81.0)
            ignition_lon = st.number_input("Ignition Lon", value=78.5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            simulation_hours = st.number_input("Hours", value=24, min_value=1, max_value=72)
            temperature = st.number_input("Temperature", value=35.0)
        
        with col2:
            wind_speed = st.number_input("Wind Speed", value=10.0)
            wind_direction = st.number_input("Wind Direction", value=225.0)
        
        submitted = st.form_submit_button("🌪️ Run Simulation")
        
        if submitted:
            request_data = {
                "region": {
                    "min_lat": min_lat, "max_lat": max_lat,
                    "min_lon": min_lon, "max_lon": max_lon
                },
                "ignition_points": [{"lat": ignition_lat, "lon": ignition_lon}],
                "weather_conditions": {
                    "temperature": temperature,
                    "wind_speed": wind_speed,
                    "wind_direction": wind_direction,
                    "humidity": 30.0
                },
                "simulation_hours": simulation_hours
            }
            
            with st.spinner("Running simulation..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/simulate/fire-spread",
                        json=request_data,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Simulation completed!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        stats = result["statistics"]
                        
                        with col1:
                            st.metric("Burned Area", f"{stats['total_burned_area_km2']:.1f} km²")
                        with col2:
                            st.metric("Fire Intensity", f"{stats['max_fire_intensity']:.2f}")
                        with col3:
                            st.metric("Steps", stats['simulation_steps'])
                        with col4:
                            st.metric("Time", f"{result['processing_time']:.2f}s")
                        
                        # Show simulation map
                        center_lat = (min_lat + max_lat) / 2
                        center_lon = (min_lon + max_lon) / 2
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                        
                        folium.Marker(
                            location=[ignition_lat, ignition_lon],
                            popup="🔥 Ignition Point",
                            icon=folium.Icon(color='red', icon='fire')
                        ).add_to(m)
                        
                        st_folium(m, width=1000, height=400)
                        
                    else:
                        st.error(f"❌ Simulation failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

def show_data_sources():
    """Data sources information."""
    
    st.header("📊 Data Sources")
    
    try:
        response = requests.get(f"{API_BASE_URL}/data/sources", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            sources = data["data_sources"]
            
            st.subheader("🌤️ Weather Data")
            for name, info in sources.get("weather", {}).items():
                with st.expander(f"📡 {name}"):
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Resolution:** {info['spatial_resolution']}")
                    st.write(f"**Coverage:** {info['coverage']}")
            
            st.subheader("🛰️ Satellite Data")
            for name, info in sources.get("satellite", {}).items():
                with st.expander(f"🛰️ {name}"):
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Resolution:** {info['spatial_resolution']}")
        else:
            st.error("❌ Failed to load data sources")
            
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {str(e)}")

if __name__ == "__main__":
    main()
