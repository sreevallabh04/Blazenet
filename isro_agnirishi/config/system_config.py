"""
ISRO AGNIRISHI - Indigenous Forest Fire Intelligence System
Core Configuration Module

Developed for ISRO Hackathon
Advanced Satellite-based Fire Prediction and Monitoring
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

@dataclass
class ISROSatelliteConfig:
    """Configuration for ISRO satellite data sources."""
    resourcesat_endpoint: str = "https://bhuvan-app1.nrsc.gov.in/data/download/"
    cartosat_endpoint: str = "https://bhuvan-app1.nrsc.gov.in/cartosat/"
    mosdac_endpoint: str = "https://www.mosdac.gov.in/data/"
    access_token: Optional[str] = None

@dataclass
class MLModelConfig:
    """ML Model configuration for indigenous algorithms."""
    model_path: str = "./models/"
    use_gpu: bool = True
    prediction_resolution: float = 30.0  # meters
    temporal_window: int = 72  # hours
    confidence_threshold: float = 0.75

@dataclass
class SystemConfig:
    """Main system configuration."""
    
    # System Info
    PROJECT_NAME: str = "ISRO AGNIRISHI"
    VERSION: str = "1.0.0"
    DEVELOPED_BY: str = "ISRO Innovation Team"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG_MODE: bool = True
    
    # Frontend Configuration  
    DASHBOARD_PORT: int = 8501
    DASHBOARD_TITLE: str = "AGNIRISHI - Indigenous Fire Intelligence"
    
    # Database Configuration (SQLite for standalone)
    DATABASE_URL: str = "sqlite:///./agnirishi.db"
    
    # ISRO Satellite Configuration
    satellite: ISROSatelliteConfig = ISROSatelliteConfig()
    
    # ML Configuration
    ml_models: MLModelConfig = MLModelConfig()
    
    # NASA FIRMS API (International data validation)
    NASA_FIRMS_API_KEY: str = "904187de8a6aa5475740a5799d207041"
    
    # Target Regions (Focus on Indian subcontinent)
    TARGET_REGIONS: Dict = {
        "uttarakhand": {
            "name": "Uttarakhand",
            "bounds": {"min_lat": 28.8, "max_lat": 31.4, "min_lon": 77.5, "max_lon": 81.0},
            "priority": "HIGH"
        },
        "himachal": {
            "name": "Himachal Pradesh", 
            "bounds": {"min_lat": 30.2, "max_lat": 33.0, "min_lon": 75.5, "max_lon": 79.0},
            "priority": "HIGH"
        },
        "karnataka": {
            "name": "Karnataka",
            "bounds": {"min_lat": 11.5, "max_lat": 18.5, "min_lon": 74.0, "max_lon": 78.5},
            "priority": "MEDIUM"
        }
    }
    
    def save_config(self, filepath: str = "isro_agnirishi/config/system.json"):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        config_dict = {
            "project_info": {
                "name": self.PROJECT_NAME,
                "version": self.VERSION,
                "developed_by": self.DEVELOPED_BY
            },
            "api": {
                "host": self.API_HOST,
                "port": self.API_PORT,
                "debug": self.DEBUG_MODE
            },
            "dashboard": {
                "port": self.DASHBOARD_PORT,
                "title": self.DASHBOARD_TITLE
            },
            "database": {
                "url": self.DATABASE_URL
            },
            "satellite": {
                "resourcesat_endpoint": self.satellite.resourcesat_endpoint,
                "cartosat_endpoint": self.satellite.cartosat_endpoint,
                "mosdac_endpoint": self.satellite.mosdac_endpoint
            },
            "ml_models": {
                "model_path": self.ml_models.model_path,
                "use_gpu": self.ml_models.use_gpu,
                "prediction_resolution": self.ml_models.prediction_resolution,
                "confidence_threshold": self.ml_models.confidence_threshold
            },
            "nasa_firms_key": self.NASA_FIRMS_API_KEY,
            "target_regions": self.TARGET_REGIONS
        }
        
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f" Configuration saved to {filepath}")

# Global configuration instance
config = SystemConfig()

def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    return config

def initialize_system():
    """Initialize the ISRO AGNIRISHI system."""
    print(" Initializing ISRO AGNIRISHI System")
    print("=" * 50)
    print(f"Project: {config.PROJECT_NAME}")
    print(f"Version: {config.VERSION}")
    print(f"Developed by: {config.DEVELOPED_BY}")
    print("=" * 50)
    
    # Save configuration
    config.save_config()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/satellite", exist_ok=True)
    os.makedirs("data/predictions", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print(" System initialized successfully")

if __name__ == "__main__":
    initialize_system()

