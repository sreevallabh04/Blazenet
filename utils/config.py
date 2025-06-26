"""
Configuration management for BlazeNet system.
"""

import os
import logging
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Main configuration class for BlazeNet."""
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://blazenet:password@localhost:5432/blazenet_db")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "blazenet")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "blazenet_db")
    
    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "your-secret-key-here-change-in-production")
    
    # Streamlit Configuration
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    # ML Model Configuration
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", "./app/ml/weights/"))
    DEVICE: str = os.getenv("DEVICE", "cuda")
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))
    MAX_PREDICTION_SIZE: int = int(os.getenv("MAX_PREDICTION_SIZE", "1000000"))
    
    # Data Configuration
    DATA_PATH: Path = Path(os.getenv("DATA_PATH", "./data/"))
    RAW_DATA_PATH: Path = Path(os.getenv("RAW_DATA_PATH", "./data/raw/"))
    PROCESSED_DATA_PATH: Path = Path(os.getenv("PROCESSED_DATA_PATH", "./data/processed/"))
    RASTER_RESOLUTION: int = int(os.getenv("RASTER_RESOLUTION", "30"))
    
    # External API Keys
    OPENWEATHER_API_KEY: Optional[str] = os.getenv("OPENWEATHER_API_KEY")
    NASA_API_KEY: Optional[str] = os.getenv("NASA_API_KEY")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Path = Path(os.getenv("LOG_FILE", "./logs/blazenet.log"))
    
    # Simulation Parameters
    MAX_SIMULATION_HOURS: int = int(os.getenv("MAX_SIMULATION_HOURS", "24"))
    SIMULATION_TIME_STEP: int = int(os.getenv("SIMULATION_TIME_STEP", "1"))
    CELLULAR_AUTOMATA_GRID_SIZE: int = int(os.getenv("CELLULAR_AUTOMATA_GRID_SIZE", "512"))
    
    # Spatial Configuration
    TARGET_CRS: str = "EPSG:4326"  # WGS84
    UTM_ZONE: str = "43N"  # For Uttarakhand region
    
    # Fire Prediction Model Parameters
    FIRE_PREDICTION_CLASSES: int = 2  # fire / no fire
    FIRE_PROBABILITY_THRESHOLD: float = 0.5
    
    # Uttarakhand Bounding Box (for initial focus area)
    UTTARAKHAND_BBOX = {
        "min_lon": 77.5,
        "max_lon": 81.0,
        "min_lat": 29.0,
        "max_lat": 31.5
    }
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration."""
        cls.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler()
            ]
        )
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.MODEL_PATH,
            cls.DATA_PATH,
            cls.RAW_DATA_PATH,
            cls.PROCESSED_DATA_PATH,
            cls.LOG_FILE.parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Initialize configuration
config = Config()
config.setup_logging()
config.create_directories() 