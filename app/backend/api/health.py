"""
Enhanced API Health Check with NASA Integration
"""

from fastapi import APIRouter
from app.backend.api.nasa_integration import get_nasa_client
import asyncpg
import os
from datetime import datetime

router = APIRouter()

@router.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check including all services."""
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "services": {}
    }
    
    # Check database
    try:
        conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
        await conn.close()
        health_status["services"]["database"] = {"status": "healthy", "type": "PostgreSQL"}
    except Exception as e:
        health_status["services"]["database"] = {"status": "error", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check NASA FIRMS API
    try:
        nasa_client = get_nasa_client()
        nasa_status = nasa_client.get_api_status()
        health_status["services"]["nasa_firms"] = nasa_status
    except Exception as e:
        health_status["services"]["nasa_firms"] = {"status": "error", "error": str(e)}
    
    # Check ML models
    try:
        import torch
        weights_dir = "app/ml/weights"
        unet_exists = os.path.exists(os.path.join(weights_dir, "unet_best.pth"))
        lstm_exists = os.path.exists(os.path.join(weights_dir, "lstm_best.pth"))
        
        health_status["services"]["ml_models"] = {
            "status": "ready" if (unet_exists and lstm_exists) else "partial",
            "unet_available": unet_exists,
            "lstm_available": lstm_exists
        }
    except Exception as e:
        health_status["services"]["ml_models"] = {"status": "error", "error": str(e)}
    
    return health_status

@router.get("/health/nasa")
async def nasa_health_check():
    """Check NASA FIRMS API status specifically."""
    
    try:
        nasa_client = get_nasa_client()
        status = nasa_client.get_api_status()
        
        # Try to get a small sample of fire data
        if status.get("status") == "Connected":
            fires = nasa_client.get_active_fires_for_india(1)
            status["sample_data_count"] = len(fires)
            status["last_test"] = datetime.now().isoformat()
        
        return status
    except Exception as e:
        return {"status": "error", "error": str(e)}

