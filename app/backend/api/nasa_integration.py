"""
NASA FIRMS API Integration for BlazeNet
Real-time fire data from NASA FIRMS system.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from fastapi import HTTPException

# NASA FIRMS Configuration
NASA_FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api"

class NASAFireDataClient:
    """Client for NASA FIRMS real-time fire data."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NASA_FIRMS_API_KEY") or os.getenv("NASA_API_KEY")
        
        if not self.api_key:
            print(" NASA FIRMS API key not found. Real-time fire data unavailable.")
    
    def get_active_fires_for_india(self, days: int = 1) -> List[Dict]:
        """Get active fires for India from NASA FIRMS."""
        
        if not self.api_key:
            return self._get_mock_fire_data()
        
        try:
            # Use country API for India
            url = f"{NASA_FIRMS_BASE_URL}/country/csv/{self.api_key}/IND/{days}"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV response
            lines = response.text.strip().split("\n")
            if len(lines) < 2:
                return []
            
            headers = lines[0].split(",")
            fires = []
            
            for line in lines[1:]:
                values = line.split(",")
                if len(values) == len(headers):
                    fire_data = dict(zip(headers, values))
                    
                    # Convert to our format
                    try:
                        fires.append({
                            "latitude": float(fire_data.get("latitude", 0)),
                            "longitude": float(fire_data.get("longitude", 0)),
                            "brightness": float(fire_data.get("brightness", 0)),
                            "confidence": fire_data.get("confidence", ""),
                            "acq_date": fire_data.get("acq_date", ""),
                            "acq_time": fire_data.get("acq_time", ""),
                            "satellite": fire_data.get("satellite", ""),
                            "instrument": fire_data.get("instrument", ""),
                            "frp": float(fire_data.get("frp", 0)),  # Fire Radiative Power
                            "version": fire_data.get("version", ""),
                            "country_id": fire_data.get("country_id", "IND")
                        })
                    except (ValueError, TypeError):
                        continue
            
            print(f" Retrieved {len(fires)} active fires from NASA FIRMS for India")
            return fires
            
        except requests.exceptions.RequestException as e:
            print(f" NASA FIRMS API error: {str(e)}")
            return self._get_mock_fire_data()
        except Exception as e:
            print(f" Error processing NASA FIRMS data: {str(e)}")
            return self._get_mock_fire_data()
    
    def get_api_status(self) -> Dict:
        """Check NASA FIRMS API status and transaction count."""
        
        if not self.api_key:
            return {"status": "No API Key", "transactions": 0, "limit": 0}
        
        try:
            url = f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "Connected",
                    "transactions": data.get("current_transactions", 0),
                    "limit": data.get("transaction_limit", 5000),
                    "interval": data.get("transaction_interval", "10 minutes")
                }
            else:
                return {"status": "Error", "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"status": "Error", "error": str(e)}
    
    def _get_mock_fire_data(self) -> List[Dict]:
        """Generate mock fire data for India when API unavailable."""
        
        import random
        
        # Uttarakhand and surrounding regions
        fires = []
        num_fires = random.randint(5, 20)
        
        for i in range(num_fires):
            fires.append({
                "latitude": random.uniform(28.0, 32.0),
                "longitude": random.uniform(77.0, 82.0),
                "brightness": random.uniform(300, 400),
                "confidence": random.choice(["low", "nominal", "high"]),
                "acq_date": datetime.now().strftime("%Y-%m-%d"),
                "acq_time": f"{random.randint(0, 23):02d}{random.randint(0, 59):02d}",
                "satellite": random.choice(["Terra", "Aqua", "VIIRS"]),
                "instrument": random.choice(["MODIS", "VIIRS"]),
                "frp": random.uniform(10, 100),
                "version": "6.1NRT",
                "country_id": "IND"
            })
        
        print(f" Generated {len(fires)} mock fire points for India")
        return fires

# Global client instance
nasa_client = None

def get_nasa_client() -> NASAFireDataClient:
    """Get or create NASA client."""
    global nasa_client
    if nasa_client is None:
        nasa_client = NASAFireDataClient()
    return nasa_client

