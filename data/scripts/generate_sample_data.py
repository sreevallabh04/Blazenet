"""
Generate Sample Geospatial Data for BlazeNet Testing
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
import json
from datetime import datetime, timedelta
import os

# Uttarakhand region bounds
UTTARAKHAND_BOUNDS = {
    'min_lat': 28.8,
    'max_lat': 31.4,
    'min_lon': 77.5,
    'max_lon': 81.0
}

def generate_dem_data():
    """Generate sample DEM (elevation) raster data."""
    
    bounds = UTTARAKHAND_BOUNDS
    width, height = 120, 104  # ~30km resolution
    
    # Create elevation grid
    elevation = np.random.normal(1500, 800, (height, width))
    elevation = np.clip(elevation, 200, 7000)  # Realistic elevation range
    
    # Add some mountain ridges
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    X, Y = np.meshgrid(x, y)
    
    # Add ridge patterns
    ridge1 = 2000 * np.exp(-((X-60)**2 + (Y-30)**2)/500)
    ridge2 = 1500 * np.exp(-((X-30)**2 + (Y-70)**2)/300)
    
    elevation += ridge1 + ridge2
    elevation = elevation.astype(np.float32)
    
    # Create transform
    transform = from_bounds(
        bounds['min_lon'], bounds['min_lat'],
        bounds['max_lon'], bounds['max_lat'],
        width, height
    )
    
    # Save as GeoTIFF
    os.makedirs('data/sample', exist_ok=True)
    
    with rasterio.open(
        'data/sample/uttarakhand_dem.tif',
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=elevation.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(elevation, 1)
    
    print("✅ Generated DEM data: data/sample/uttarakhand_dem.tif")

def generate_slope_aspect():
    """Generate slope and aspect from DEM."""
    
    # Read DEM
    with rasterio.open('data/sample/uttarakhand_dem.tif') as src:
        elevation = src.read(1)
        transform = src.transform
        profile = src.profile
    
    # Calculate slope and aspect
    dy, dx = np.gradient(elevation)
    
    # Slope in degrees
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi
    
    # Aspect in degrees
    aspect = np.arctan2(-dx, dy) * 180 / np.pi
    aspect = (aspect + 360) % 360  # Convert to 0-360
    
    # Save slope
    profile.update(dtype=slope.dtype)
    with rasterio.open('data/sample/uttarakhand_slope.tif', 'w', **profile) as dst:
        dst.write(slope.astype(np.float32), 1)
    
    # Save aspect
    with rasterio.open('data/sample/uttarakhand_aspect.tif', 'w', **profile) as dst:
        dst.write(aspect.astype(np.float32), 1)
    
    print("✅ Generated slope data: data/sample/uttarakhand_slope.tif")
    print("✅ Generated aspect data: data/sample/uttarakhand_aspect.tif")

def generate_weather_data():
    """Generate sample weather station data."""
    
    # Sample weather stations in Uttarakhand
    stations = [
        {"id": "DEH001", "name": "Dehradun", "lat": 30.3165, "lon": 78.0322},
        {"id": "NAI001", "name": "Nainital", "lat": 29.3803, "lon": 79.4636},
        {"id": "RIS001", "name": "Rishikesh", "lat": 30.1022, "lon": 78.2676},
        {"id": "ALM001", "name": "Almora", "lat": 29.5971, "lon": 79.6591},
        {"id": "HAR001", "name": "Haridwar", "lat": 29.9457, "lon": 78.1642}
    ]
    
    # Generate 30 days of hourly weather data
    start_date = datetime.now() - timedelta(days=30)
    dates = [start_date + timedelta(hours=h) for h in range(30*24)]
    
    weather_data = []
    
    for station in stations:
        for date in dates:
            # Seasonal temperature patterns
            base_temp = 25 + 10 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            daily_temp = base_temp + 8 * np.sin(2 * np.pi * date.hour / 24)
            
            # Add elevation effect
            elevation_effect = -6.5 * 1.5 / 1000  # Lapse rate
            temp = daily_temp + elevation_effect + np.random.normal(0, 2)
            
            # Humidity (inverse correlation with temperature)
            humidity = 80 - 0.5 * temp + np.random.normal(0, 10)
            humidity = np.clip(humidity, 20, 95)
            
            # Wind speed
            wind_speed = np.random.exponential(3) + 1
            wind_direction = np.random.uniform(0, 360)
            
            # Precipitation (sparse)
            precipitation = 0
            if np.random.random() < 0.1:  # 10% chance of rain
                precipitation = np.random.exponential(2)
            
            weather_data.append({
                'station_id': station['id'],
                'station_name': station['name'],
                'timestamp': date.isoformat(),
                'latitude': station['lat'],
                'longitude': station['lon'],
                'temperature': round(temp, 1),
                'humidity': round(humidity, 1),
                'wind_speed': round(wind_speed, 1),
                'wind_direction': round(wind_direction, 1),
                'precipitation': round(precipitation, 2),
                'pressure': round(1013.25 + np.random.normal(0, 10), 1)
            })
    
    # Save as CSV
    df = pd.DataFrame(weather_data)
    df.to_csv('data/sample/weather_data.csv', index=False)
    
    print(f"✅ Generated weather data: data/sample/weather_data.csv ({len(df)} records)")

def generate_fire_history():
    """Generate sample historical fire incident data."""
    
    fires = []
    
    # Generate 100 historical fire incidents
    for i in range(100):
        # Random location in Uttarakhand
        lat = np.random.uniform(UTTARAKHAND_BOUNDS['min_lat'], UTTARAKHAND_BOUNDS['max_lat'])
        lon = np.random.uniform(UTTARAKHAND_BOUNDS['min_lon'], UTTARAKHAND_BOUNDS['max_lon'])
        
        # Random date in past 5 years
        start_date = datetime.now() - timedelta(days=5*365)
        random_days = np.random.randint(0, 5*365)
        fire_date = start_date + timedelta(days=random_days)
        
        # Fire characteristics
        area_burned = np.random.exponential(50)  # Hectares
        duration = np.random.randint(1, 48)  # Hours
        intensity = np.random.choice(['Low', 'Medium', 'High'], p=[0.5, 0.3, 0.2])
        cause = np.random.choice(['Human', 'Lightning', 'Unknown'], p=[0.7, 0.2, 0.1])
        
        fires.append({
            'fire_id': f'FIRE_{i+1:03d}',
            'date': fire_date.date().isoformat(),
            'latitude': round(lat, 4),
            'longitude': round(lon, 4),
            'area_burned_ha': round(area_burned, 1),
            'duration_hours': duration,
            'intensity': intensity,
            'cause': cause,
            'suppressed': np.random.choice([True, False], p=[0.9, 0.1])
        })
    
    # Save as CSV
    df = pd.DataFrame(fires)
    df.to_csv('data/sample/fire_history.csv', index=False)
    
    print(f"✅ Generated fire history: data/sample/fire_history.csv ({len(df)} records)")

def generate_vegetation_index():
    """Generate sample NDVI raster data."""
    
    bounds = UTTARAKHAND_BOUNDS
    width, height = 120, 104
    
    # Create NDVI grid (0-1 range)
    ndvi = np.random.beta(3, 2, (height, width))  # Skewed towards higher values
    
    # Add elevation-based vegetation patterns
    with rasterio.open('data/sample/uttarakhand_dem.tif') as src:
        elevation = src.read(1)
        transform = src.transform
    
    # Vegetation decreases with elevation
    elevation_normalized = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    vegetation_factor = 1 - 0.7 * elevation_normalized
    
    ndvi = ndvi * vegetation_factor
    ndvi = np.clip(ndvi, 0, 1).astype(np.float32)
    
    # Save NDVI
    with rasterio.open(
        'data/sample/uttarakhand_ndvi.tif',
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=ndvi.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(ndvi, 1)
    
    print("✅ Generated NDVI data: data/sample/uttarakhand_ndvi.tif")

def generate_metadata():
    """Generate metadata file for the sample dataset."""
    
    metadata = {
        "dataset_info": {
            "name": "BlazeNet Sample Dataset",
            "description": "Sample geospatial data for testing BlazeNet fire prediction system",
            "region": "Uttarakhand, India",
            "created": datetime.now().isoformat(),
            "bounds": UTTARAKHAND_BOUNDS,
            "spatial_resolution": "~30km",
            "coordinate_system": "EPSG:4326"
        },
        "files": {
            "uttarakhand_dem.tif": {
                "type": "Digital Elevation Model",
                "units": "meters",
                "description": "Terrain elevation data"
            },
            "uttarakhand_slope.tif": {
                "type": "Slope",
                "units": "degrees",
                "description": "Terrain slope derived from DEM"
            },
            "uttarakhand_aspect.tif": {
                "type": "Aspect",
                "units": "degrees",
                "description": "Terrain aspect derived from DEM"
            },
            "uttarakhand_ndvi.tif": {
                "type": "Normalized Difference Vegetation Index",
                "units": "dimensionless",
                "range": "0-1",
                "description": "Vegetation density indicator"
            },
            "weather_data.csv": {
                "type": "Weather Station Data",
                "temporal_resolution": "hourly",
                "period": "30 days",
                "stations": 5,
                "description": "Historical weather observations"
            },
            "fire_history.csv": {
                "type": "Fire Incident Records",
                "period": "5 years",
                "records": 100,
                "description": "Historical fire occurrence data"
            }
        }
    }
    
    with open('data/sample/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Generated metadata: data/sample/metadata.json")

def main():
    """Generate all sample data."""
    
    print("🔥 Generating BlazeNet Sample Dataset")
    print("=" * 40)
    
    # Create directory
    os.makedirs('data/sample', exist_ok=True)
    
    # Generate all data types
    generate_dem_data()
    generate_slope_aspect()
    generate_vegetation_index()
    generate_weather_data()
    generate_fire_history()
    generate_metadata()
    
    print("\n🎉 Sample dataset generation complete!")
    print("\nGenerated files:")
    print("📁 data/sample/")
    print("  ├── uttarakhand_dem.tif      (Elevation)")
    print("  ├── uttarakhand_slope.tif    (Slope)")
    print("  ├── uttarakhand_aspect.tif   (Aspect)")
    print("  ├── uttarakhand_ndvi.tif     (Vegetation)")
    print("  ├── weather_data.csv         (Weather)")
    print("  ├── fire_history.csv         (Fire Records)")
    print("  └── metadata.json            (Dataset Info)")

if __name__ == "__main__":
    main() 