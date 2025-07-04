"""
Fire prediction API endpoints.
"""

import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import logging
from datetime import datetime, date
import uuid

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class PredictionRequest(BaseModel):
    """Request model for fire prediction."""
    region: Dict[str, float] = Field(
        description="Region bounds",
        example={
            "min_lat": 29.0,
            "max_lat": 31.5,
            "min_lon": 77.5,
            "max_lon": 81.0
        }
    )
    date: date = Field(
        description="Prediction date",
        example="2024-01-01"
    )
    weather_data: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weather parameters",
        example={
            "temperature": 30.0,
            "humidity": 40.0,
            "wind_speed": 5.0,
            "wind_direction": 180.0,
            "precipitation": 0.0
        }
    )
    resolution: float = Field(
        default=30.0,
        description="Output resolution in meters",
        example=30.0
    )
    model_type: str = Field(
        default="unet",
        description="Model type to use",
        example="unet"
    )

class PredictionResponse(BaseModel):
    """Response model for fire prediction."""
    prediction_id: str
    region: Dict[str, float]
    date: str
    resolution: float
    fire_probability_url: str
    statistics: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float

@router.post("/predict/fire-probability", response_model=PredictionResponse)
async def predict_fire_probability(request: PredictionRequest):
    """
    Predict fire probability for a given region and date.
    """
    try:
        logger.info(f"Received fire prediction request for region: {request.region}")
        
        # Validate region bounds
        if (request.region["max_lat"] <= request.region["min_lat"] or
            request.region["max_lon"] <= request.region["min_lon"]):
            raise HTTPException(
                status_code=400,
                detail="Invalid region bounds"
            )
        
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Simulate prediction processing
        start_time = datetime.now()
        
        # Create mock prediction results
        statistics = {
            "high_risk_area_km2": 145.2,
            "medium_risk_area_km2": 320.5,
            "low_risk_area_km2": 234.3,
            "max_probability": 0.85,
            "mean_probability": 0.23,
            "total_area_km2": 700.0
        }
        
        metadata = {
            "model_version": "1.0.0",
            "model_type": request.model_type,
            "prediction_date": request.date.isoformat(),
            "created_at": datetime.now().isoformat(),
            "weather_used": request.weather_data is not None
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create response
        response = PredictionResponse(
            prediction_id=prediction_id,
            region=request.region,
            date=request.date.isoformat(),
            resolution=request.resolution,
            fire_probability_url=f"/api/v1/data/prediction/{prediction_id}/probability.tif",
            statistics=statistics,
            metadata=metadata,
            processing_time=processing_time
        )
        
        logger.info(f"Fire prediction completed in {processing_time:.2f} seconds")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fire prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/predict/models")
async def get_available_models():
    """Get list of available prediction models."""
    try:
        models = [
            {
                "name": "unet",
                "description": "U-Net model for spatial fire prediction",
                "version": "1.0.0",
                "accuracy": 0.84,
                "training_date": "2024-01-01"
            },
            {
                "name": "lstm",
                "description": "LSTM model for temporal fire prediction",
                "version": "1.0.0",
                "accuracy": 0.79,
                "training_date": "2024-01-01"
            }
        ]
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Failed to get available models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve available models"
        )

@router.get("/predict/history")
async def get_prediction_history(limit: int = 10, offset: int = 0):
    """Get prediction history."""
    try:
        # Mock history data
        history = [
            {
                "prediction_id": str(uuid.uuid4()),
                "region": {"min_lat": 29.0, "max_lat": 30.0, "min_lon": 78.0, "max_lon": 79.0},
                "date": "2024-01-01",
                "model_type": "unet",
                "status": "completed",
                "created_at": "2024-01-01T10:00:00Z"
            }
            for _ in range(min(limit, 5))
        ]
        
        return {
            "predictions": history,
            "total": len(history),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to get prediction history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve prediction history"
        )
