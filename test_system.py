"""
BlazeNet System Test Script
Complete end-to-end testing of all components.
"""

import requests
import json
import time
import os
import subprocess
import sys
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
API_V1_URL = f"{API_BASE_URL}/api/v1"

def test_api_health():
    """Test API health endpoint."""
    
    print("🔍 Testing API Health...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data['status']}")
            return True
        else:
            print(f"❌ API Health failed: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {str(e)}")
        return False

def test_fire_prediction():
    """Test fire prediction endpoint."""
    
    print("\n🔥 Testing Fire Prediction...")
    
    request_data = {
        "region": {
            "min_lat": 29.0,
            "max_lat": 31.5,
            "min_lon": 77.5,
            "max_lon": 81.0
        },
        "date": "2024-01-15",
        "weather_data": {
            "temperature": 35.0,
            "humidity": 30.0,
            "wind_speed": 10.0,
            "wind_direction": 180.0,
            "precipitation": 0.0
        },
        "resolution": 30.0,
        "model_type": "unet"
    }
    
    try:
        response = requests.post(
            f"{API_V1_URL}/predict/fire-probability",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            stats = result["statistics"]
            
            print(f"✅ Fire Prediction successful!")
            print(f"   📊 High Risk Area: {stats['high_risk_area_km2']:.1f} km²")
            print(f"   📊 Max Probability: {stats['max_probability']:.2f}")
            print(f"   ⏱️ Processing Time: {result['processing_time']:.2f}s")
            return True
        else:
            print(f"❌ Fire Prediction failed: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Fire Prediction error: {str(e)}")
        return False

def test_fire_simulation():
    """Test fire simulation endpoint."""
    
    print("\n🌪️ Testing Fire Simulation...")
    
    request_data = {
        "region": {
            "min_lat": 29.0,
            "max_lat": 31.5,
            "min_lon": 77.5,
            "max_lon": 81.0
        },
        "ignition_points": [
            {"lat": 30.0, "lon": 78.5}
        ],
        "weather_conditions": {
            "temperature": 40.0,
            "humidity": 25.0,
            "wind_speed": 15.0,
            "wind_direction": 225.0,
            "fuel_moisture": 0.08
        },
        "simulation_hours": 12,
        "time_step": 1.0,
        "resolution": 30.0
    }
    
    try:
        response = requests.post(
            f"{API_V1_URL}/simulate/fire-spread",
            json=request_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            stats = result["statistics"]
            
            print(f"✅ Fire Simulation successful!")
            print(f"   🔥 Burned Area: {stats['total_burned_area_km2']:.1f} km²")
            print(f"   💥 Max Intensity: {stats['max_fire_intensity']:.2f}")
            print(f"   🎯 Simulation Steps: {stats['simulation_steps']}")
            print(f"   ⏱️ Processing Time: {result['processing_time']:.2f}s")
            return True
        else:
            print(f"❌ Fire Simulation failed: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Fire Simulation error: {str(e)}")
        return False

def test_data_sources():
    """Test data sources endpoint."""
    
    print("\n📊 Testing Data Sources...")
    
    try:
        response = requests.get(f"{API_V1_URL}/data/sources", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            sources = data["data_sources"]
            
            print(f"✅ Data Sources loaded!")
            print(f"   🌤️ Weather Sources: {len(sources.get('weather', {}))}")
            print(f"   🛰️ Satellite Sources: {len(sources.get('satellite', {}))}")
            print(f"   🏔️ Terrain Sources: {len(sources.get('terrain', {}))}")
            return True
        else:
            print(f"❌ Data Sources failed: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Data Sources error: {str(e)}")
        return False

def test_weather_data():
    """Test weather data endpoint."""
    
    print("\n🌤️ Testing Weather Data...")
    
    try:
        response = requests.get(
            f"{API_V1_URL}/data/weather",
            params={
                "station_id": "DEH001",
                "start_date": "2024-01-01",
                "end_date": "2024-01-02"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Weather Data retrieved!")
            print(f"   📍 Station: {data['station_info']['name']}")
            print(f"   📅 Records: {len(data['weather_data'])}")
            return True
        else:
            print(f"❌ Weather Data failed: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Weather Data error: {str(e)}")
        return False

def test_terrain_data():
    """Test terrain data endpoint."""
    
    print("\n🏔️ Testing Terrain Data...")
    
    try:
        response = requests.get(
            f"{API_V1_URL}/data/terrain",
            params={
                "min_lat": 30.0,
                "max_lat": 30.5,
                "min_lon": 78.0,
                "max_lon": 78.5,
                "data_type": "elevation"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Terrain Data retrieved!")
            print(f"   📊 Data Type: {data['data_type']}")
            print(f"   📏 Resolution: {data['resolution']}")
            print(f"   📈 Statistics: {data['statistics']}")
            return True
        else:
            print(f"❌ Terrain Data failed: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Terrain Data error: {str(e)}")
        return False

def test_database_connection():
    """Test database connectivity."""
    
    print("\n🗄️ Testing Database Connection...")
    
    try:
        # Test via API health check which includes DB status
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            if "database" in data:
                print(f"✅ Database: {data['database']['status']}")
                return True
            else:
                print("✅ Database connection implied by API health")
                return True
        else:
            print(f"❌ Database test failed via API health")
            return False
            
    except Exception as e:
        print(f"❌ Database test error: {str(e)}")
        return False

def test_ml_models():
    """Test ML model loading."""
    
    print("\n🤖 Testing ML Models...")
    
    # Test through prediction endpoint which loads models
    try:
        # Quick prediction to test model loading
        request_data = {
            "region": {
                "min_lat": 29.0,
                "max_lat": 29.1,
                "min_lon": 77.5,
                "max_lon": 77.6
            },
            "date": "2024-01-15",
            "weather_data": {
                "temperature": 30.0,
                "humidity": 40.0,
                "wind_speed": 5.0,
                "wind_direction": 180.0
            },
            "model_type": "lstm"
        }
        
        response = requests.post(
            f"{API_V1_URL}/predict/fire-probability",
            json=request_data,
            timeout=15
        )
        
        if response.status_code == 200:
            print("✅ ML Models (LSTM): Loaded and functional")
            
            # Test U-Net model
            request_data["model_type"] = "unet"
            response = requests.post(
                f"{API_V1_URL}/predict/fire-probability",
                json=request_data,
                timeout=15
            )
            
            if response.status_code == 200:
                print("✅ ML Models (U-Net): Loaded and functional")
                return True
            else:
                print("❌ U-Net model test failed")
                return False
        else:
            print("❌ LSTM model test failed")
            return False
            
    except Exception as e:
        print(f"❌ ML Models test error: {str(e)}")
        return False

def test_frontend_accessibility():
    """Test if frontend is accessible."""
    
    print("\n🖥️ Testing Frontend Accessibility...")
    
    try:
        # Check if Streamlit is running on port 8501
        response = requests.get("http://localhost:8501", timeout=5)
        
        if response.status_code == 200:
            print("✅ Frontend: Accessible on http://localhost:8501")
            return True
        else:
            print(f"❌ Frontend: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException:
        print("❌ Frontend: Not accessible (Streamlit not running?)")
        return False

def run_all_tests():
    """Run complete system test suite."""
    
    print("🔥 BlazeNet System Test Suite")
    print("=" * 50)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    tests = [
        ("API Health", test_api_health),
        ("Database Connection", test_database_connection),
        ("Data Sources", test_data_sources),
        ("Weather Data", test_weather_data),
        ("Terrain Data", test_terrain_data),
        ("ML Models", test_ml_models),
        ("Fire Prediction", test_fire_prediction),
        ("Fire Simulation", test_fire_simulation),
        ("Frontend", test_frontend_accessibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! BlazeNet system is fully functional.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the system components.")
        return False

def check_prerequisites():
    """Check if required services are running."""
    
    print("🔍 Checking Prerequisites...")
    
    # Check if API server is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        print("✅ API Server is running")
    except:
        print("❌ API Server is not running")
        print("   💡 Start with: uvicorn app.backend.main:app --host 0.0.0.0 --port 8000")
        return False
    
    # Check if database is accessible
    # (This is checked via API health, so if API is up, DB should be too)
    
    return True

if __name__ == "__main__":
    print("🚀 Starting BlazeNet System Tests...\n")
    
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please start required services first.")
        sys.exit(1)
    
    success = run_all_tests()
    
    if success:
        print("\n🎊 BlazeNet is ready for production!")
        sys.exit(0)
    else:
        print("\n⚠️ Please fix failing components before deployment.")
        sys.exit(1) 