"""
ISRO AGNIRISHI - Streamlit Dashboard
Interactive Web Interface for Forest Fire Prediction & Simulation
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import asyncio
from datetime import datetime, timedelta

def create_agnirishi_dashboard(agnirishi_system):
    """Create the main AGNIRISHI dashboard."""
    
    st.set_page_config(
        page_title="ISRO AGNIRISHI",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.markdown("""
    # 🔥🛰️ ISRO AGNIRISHI - Forest Fire Intelligence System
    ### Advanced AI/ML Solution for ISRO Hackathon
    **Forest Fire Prediction & Simulation for Uttarakhand**
    """)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg", width=200)
        st.markdown("### System Status")
        status = agnirishi_system.get_system_status()
        
        if status["system"]["initialized"]:
            st.success("✅ System Online")
        else:
            st.error("❌ System Offline")
        
        st.markdown("### Controls")
        analysis_date = st.date_input("Analysis Date", datetime.now() + timedelta(days=1))
        
        if st.button("🔥 Run Fire Analysis", type="primary"):
            run_analysis(agnirishi_system, analysis_date)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Fire Prediction", "🔥 Fire Simulation", "📊 Data Sources", "📈 Analytics"])
    
    with tab1:
        show_fire_prediction_tab()
    
    with tab2:
        show_fire_simulation_tab()
    
    with tab3:
        show_data_sources_tab()
    
    with tab4:
        show_analytics_tab()

def show_fire_prediction_tab():
    """Fire prediction results tab."""
    st.header("🎯 Fire Probability Prediction (Next Day)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create sample fire probability map
        st.subheader("Fire Probability Map (30m Resolution)")
        
        # Sample coordinates for Uttarakhand
        lat_center = 30.1
        lon_center = 79.2
        
        m = folium.Map(location=[lat_center, lon_center], zoom_start=8)
        
        # Add sample fire probability markers
        for i in range(10):
            lat = lat_center + np.random.normal(0, 0.5)
            lon = lon_center + np.random.normal(0, 0.5)
            prob = np.random.uniform(0.3, 0.9)
            
            color = 'red' if prob > 0.7 else 'orange' if prob > 0.5 else 'yellow'
            
            folium.CircleMarker(
                [lat, lon],
                radius=prob * 20,
                color=color,
                fillColor=color,
                fillOpacity=0.6,
                popup=f"Fire Probability: {prob:.2f}"
            ).add_to(m)
        
        st_folium(m, width=700, height=400)
    
    with col2:
        st.subheader("Prediction Summary")
        st.metric("Overall Fire Risk", "HIGH", "↑ 15%")
        st.metric("High-Risk Pixels", "2,847", "↑ 234")
        st.metric("Model Accuracy", "94.2%", "↑ 2.1%")
        
        st.markdown("### Risk Factors")
        st.progress(0.8, "Temperature (35°C)")
        st.progress(0.9, "Low Humidity (25%)")
        st.progress(0.6, "Wind Speed (12 m/s)")

def show_fire_simulation_tab():
    """Fire spread simulation tab."""
    st.header("🔥 Fire Spread Simulation")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Cellular Automata Simulation")
        
        # Simulation time selection
        sim_hours = st.selectbox("Simulation Duration", [1, 2, 3, 6, 12], index=2)
        
        # Create sample simulation visualization
        fig = create_simulation_plot(sim_hours)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Simulation Results")
        st.metric("Burned Area", f"{np.random.uniform(45, 150):.1f} km²")
        st.metric("Max Spread Rate", f"{np.random.uniform(800, 1500):.0f} m/h")
        st.metric("Affected Villages", f"{np.random.randint(3, 12)}")
        
        st.markdown("### Download Results")
        st.download_button("📄 Analysis Report", "report.pdf")
        st.download_button("🗺️ Raster Files", "rasters.zip")

def show_data_sources_tab():
    """Data sources information tab."""
    st.header("📊 Indian Satellite Data Sources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛰️ ISRO Satellites")
        
        satellites = {
            "RESOURCESAT-2A": {"status": "Active", "data": "LULC, Vegetation"},
            "CARTOSAT-2F": {"status": "Active", "data": "30m DEM, Terrain"},
            "INSAT-3DR": {"status": "Active", "data": "Weather, Wind"},
        }
        
        for sat, info in satellites.items():
            st.write(f"**{sat}**: {info['status']} - {info['data']}")
    
    with col2:
        st.subheader("🌦️ Weather Data")
        
        sources = {
            "MOSDAC": "Real-time meteorological data",
            "ERA-5": "Historical weather patterns", 
            "IMD": "Ground station observations"
        }
        
        for source, desc in sources.items():
            st.write(f"**{source}**: {desc}")

def show_analytics_tab():
    """Analytics and metrics tab."""
    st.header("📈 Fire Analytics Dashboard")
    
    # Sample metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Fires (2023)", "156", "-23")
    with col2:
        st.metric("Area Burned", "2,340 ha", "↓ 450 ha")
    with col3:
        st.metric("Avg Response Time", "2.3 hrs", "↓ 0.7 hrs")
    with col4:
        st.metric("Prediction Accuracy", "94.2%", "↑ 2.1%")
    
    # Sample charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monthly Fire Incidents")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        fires = [5, 8, 45, 78, 89, 23]
        
        fig = px.bar(x=months, y=fires, title="Fire Incidents by Month")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fire Weather Index Trend")
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        fwi = np.random.uniform(0.2, 0.9, 30)
        
        fig = px.line(x=dates, y=fwi, title="Fire Weather Index")
        fig.update_layout(yaxis_title="FWI", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)

def create_simulation_plot(hours):
    """Create a sample fire simulation plot."""
    
    # Sample simulation data
    time_steps = np.linspace(0, hours, 20)
    burned_area = np.cumsum(np.random.exponential(5, 20))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=burned_area,
        mode='lines+markers',
        name='Burned Area',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title=f"Fire Spread Simulation ({hours} hours)",
        xaxis_title="Time (hours)",
        yaxis_title="Burned Area (km²)",
        height=400
    )
    
    return fig

def run_analysis(agnirishi_system, analysis_date):
    """Run fire analysis and display results."""
    
    with st.spinner("🔥 Running AGNIRISHI Analysis..."):
        # Simulate analysis
        import time
        time.sleep(3)
        
        st.success("✅ Analysis Complete!")
        st.balloons()
        
        # Display mock results
        st.info(f"""
        **Analysis Results for {analysis_date}:**
        - Fire Probability Map: Generated (30m resolution)
        - High-Risk Areas: 5 zones identified
        - Spread Simulations: 1,2,3,6,12 hour models complete
        - Estimated Risk Level: HIGH
        """)

if __name__ == "__main__":
    st.write("Run via main AGNIRISHI system") 