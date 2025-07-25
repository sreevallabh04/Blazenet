"""
Fire simulation API endpoints.
"""

import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class SimulationRequest(BaseModel):
    """Request model for fire simulation."""
    region: Dict[str, float] = Field(
        description="Region bounds",
        example={
            "min_lat": 29.0,
            "max_lat": 31.5,
            "min_lon": 77.5,
            "max_lon": 81.0
        }
    )
    ignition_points: List[Dict[str, float]] = Field(
        description="Initial fire ignition points",
        example=[
            {"lat": 30.0, "lon": 78.5},
            {"lat": 30.2, "lon": 78.7}
        ]
    )
    weather_conditions: Dict[str, float] = Field(
        description="Weather parameters for simulation",
        example={
            "temperature": 35.0,
            "humidity": 30.0,
            "wind_speed": 10.0,
            "wind_direction": 225.0,
            "fuel_moisture": 0.08
        }
    )
    simulation_hours: int = Field(
        default=24,
        description="Simulation duration in hours",
        example=24
    )
    time_step: float = Field(
        default=1.0,
        description="Time step in hours",
        example=1.0
    )
    resolution: float = Field(
        default=30.0,
        description="Spatial resolution in meters",
        example=30.0
    )

class SimulationResponse(BaseModel):
    """Response model for fire simulation."""
    simulation_id: str
    region: Dict[str, float]
    ignition_points: List[Dict[str, float]]
    simulation_hours: int
    time_step: float
    animation_url: str
    final_state_url: str
    statistics: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float

@router.post("/simulate/fire-spread", response_model=SimulationResponse)
async def simulate_fire_spread(request: SimulationRequest):
    """
    Simulate fire spread from ignition points.
    """
    try:
        logger.info(f"Received fire simulation request for {len(request.ignition_points)} ignition points")
        
        # Validate request
        if not request.ignition_points:
            raise HTTPException(
                status_code=400,
                detail="At least one ignition point is required"
            )
        
        if request.simulation_hours > 72:  # Maximum 72 hours
            raise HTTPException(
                status_code=400,
                detail="Maximum simulation duration is 72 hours"
            )
        
        # Generate simulation ID
        simulation_id = str(uuid.uuid4())
        
        # Simulate processing
        start_time = datetime.now()
        
        # Create mock simulation results
        statistics = {
            "total_burned_area_km2": 85.4,
            "max_fire_intensity": 0.92,
            "simulation_steps": int(request.simulation_hours / request.time_step),
            "ignition_points_count": len(request.ignition_points),
            "final_perimeter_km": 42.3,
            "rate_of_spread_m_per_hour": 127.5
        }
        
        metadata = {
            "simulation_version": "1.0.0",
            "algorithm": "cellular_automata",
            "created_at": datetime.now().isoformat(),
            "weather_conditions": request.weather_conditions,
            "grid_size": "512x512",
            "cell_size_m": request.resolution
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create response
        response = SimulationResponse(
            simulation_id=simulation_id,
            region=request.region,
            ignition_points=request.ignition_points,
            simulation_hours=request.simulation_hours,
            time_step=request.time_step,
            animation_url=f"/api/v1/data/simulation/{simulation_id}/animation.mp4",
            final_state_url=f"/api/v1/data/simulation/{simulation_id}/final_state.tif",
            statistics=statistics,
            metadata=metadata,
            processing_time=processing_time
        )
        
        logger.info(f"Fire simulation completed in {processing_time:.2f} seconds")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fire simulation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}"
        )

@router.get("/simulate/presets")
async def get_simulation_presets():
    """Get predefined simulation scenarios."""
    try:
        presets = [
            {
                "name": "high_wind_scenario",
                "description": "High wind speed fire spread scenario",
                "weather_conditions": {
                    "temperature": 38.0,
                    "humidity": 25.0,
                    "wind_speed": 15.0,
                    "wind_direction": 270.0,
                    "fuel_moisture": 0.05
                },
                "expected_spread_rate": "fast"
            },
            {
                "name": "moderate_conditions",
                "description": "Typical moderate fire conditions",
                "weather_conditions": {
                    "temperature": 30.0,
                    "humidity": 40.0,
                    "wind_speed": 8.0,
                    "wind_direction": 180.0,
                    "fuel_moisture": 0.10
                },
                "expected_spread_rate": "moderate"
            },
            {
                "name": "humid_conditions",
                "description": "High humidity, low spread conditions",
                "weather_conditions": {
                    "temperature": 25.0,
                    "humidity": 60.0,
                    "wind_speed": 3.0,
                    "wind_direction": 90.0,
                    "fuel_moisture": 0.20
                },
                "expected_spread_rate": "slow"
            }
        ]
        
        return {"presets": presets}
        
    except Exception as e:
        logger.error(f"Failed to get simulation presets: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve simulation presets"
        )

@router.get("/simulate/history")
async def get_simulation_history(limit: int = 10, offset: int = 0):
    """Get simulation history."""
    try:
        # Mock history data
        history = [
            {
                "simulation_id": str(uuid.uuid4()),
                "region": {"min_lat": 29.0, "max_lat": 30.0, "min_lon": 78.0, "max_lon": 79.0},
                "ignition_points": [{"lat": 29.5, "lon": 78.5}],
                "simulation_hours": 24,
                "status": "completed",
                "burned_area_km2": 45.2,
                "created_at": "2024-01-01T10:00:00Z"
            }
            for _ in range(min(limit, 5))
        ]
        
        return {
            "simulations": history,
            "total": len(history),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to get simulation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve simulation history"
        )
