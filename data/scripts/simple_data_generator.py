"""
Simple Sample Data Generator (No Rasterio Dependencies)
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import os

def generate_simple_sample_data():
    """Generate sample data without rasterio dependencies."""
    
    print(" Generating Simple BlazeNet Sample Dataset")
    print("=" * 40)
    
    # Create directory
    os.makedirs("data/sample", exist_ok=True)
    
    # Generate weather data
    print(" Generating weather data...")
    stations = [
        {"id": "DEH001", "name": "Dehradun", "lat": 30.3165, "lon": 78.0322},
        {"id": "NAI001", "name": "Nainital", "lat": 29.3803, "lon": 79.4636},
        {"id": "RIS001", "name": "Rishikesh", "lat": 30.1022, "lon": 78.2676},
    ]
    
    weather_data = []
    start_date = datetime.now() - timedelta(days=7)
    
    for station in stations:
        for hour in range(7 * 24):  # 7 days of hourly data
            date = start_date + timedelta(hours=hour)
            
            weather_data.append({
                "station_id": station["id"],
                "station_name": station["name"],
                "timestamp": date.isoformat(),
                "latitude": station["lat"],
                "longitude": station["lon"],
                "temperature": round(25 + 10 * np.random.random(), 1),
                "humidity": round(30 + 40 * np.random.random(), 1),
                "wind_speed": round(5 * np.random.random(), 1),
                "wind_direction": round(360 * np.random.random(), 1),
                "precipitation": round(5 * np.random.random() if np.random.random() < 0.1 else 0, 2),
                "pressure": round(1013.25 + 20 * (np.random.random() - 0.5), 1)
            })
    
    df = pd.DataFrame(weather_data)
    df.to_csv("data/sample/weather_data.csv", index=False)
    print(f" Generated weather data: {len(df)} records")
    
    # Generate fire history
    print(" Generating fire history...")
    fires = []
    for i in range(50):
        fires.append({
            "fire_id": f"FIRE_{i+1:03d}",
            "date": (datetime.now() - timedelta(days=np.random.randint(0, 365))).date().isoformat(),
            "latitude": round(28.8 + 2.6 * np.random.random(), 4),
            "longitude": round(77.5 + 3.5 * np.random.random(), 4),
            "area_burned_ha": round(50 * np.random.exponential(1), 1),
            "duration_hours": np.random.randint(1, 48),
            "intensity": np.random.choice(["Low", "Medium", "High"], p=[0.5, 0.3, 0.2]),
            "cause": np.random.choice(["Human", "Lightning", "Unknown"], p=[0.7, 0.2, 0.1]),
            "suppressed": np.random.choice([True, False], p=[0.9, 0.1])
        })
    
    fire_df = pd.DataFrame(fires)
    fire_df.to_csv("data/sample/fire_history.csv", index=False)
    print(f" Generated fire history: {len(fire_df)} records")
    
    # Generate metadata
    metadata = {
        "dataset_info": {
            "name": "BlazeNet Simple Sample Dataset",
            "description": "Basic sample data for testing BlazeNet (no rasterio dependencies)",
            "region": "Uttarakhand, India",
            "created": datetime.now().isoformat(),
            "bounds": {
                "min_lat": 28.8, "max_lat": 31.4,
                "min_lon": 77.5, "max_lon": 81.0
            }
        },
        "files": {
            "weather_data.csv": {
                "type": "Weather Station Data",
                "records": len(df),
                "description": "7 days of hourly weather data"
            },
            "fire_history.csv": {
                "type": "Fire Incident Records", 
                "records": len(fire_df),
                "description": "Historical fire data"
            }
        }
    }
    
    with open("data/sample/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(" Generated metadata: data/sample/metadata.json")
    print(" Simple sample dataset complete!")

if __name__ == "__main__":
    generate_simple_sample_data()

