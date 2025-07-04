"""
Data access API endpoints for weather, terrain, and geospatial data.
"""

import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, date
import uuid

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class WeatherRequest(BaseModel):
    """Request model for weather data."""
    region: Dict[str, float] = Field(
        description="Region bounds",
        example={
            "min_lat": 29.0,
            "max_lat": 31.5,
            "min_lon": 77.5,
            "max_lon": 81.0
        }
    )
    start_date: date = Field(
        description="Start date for weather data",
        example="2024-01-01"
    )
    end_date: date = Field(
        description="End date for weather data",
        example="2024-01-07"
    )
    parameters: List[str] = Field(
        default=["temperature", "humidity", "wind_speed", "precipitation"],
        description="Weather parameters to retrieve",
        example=["temperature", "humidity", "wind_speed", "precipitation"]
    )

class TerrainRequest(BaseModel):
    """Request model for terrain data."""
    region: Dict[str, float] = Field(
        description="Region bounds",
        example={
            "min_lat": 29.0,
            "max_lat": 31.5,
            "min_lon": 77.5,
            "max_lon": 81.0
        }
    )
    resolution: float = Field(
        default=30.0,
        description="Spatial resolution in meters",
        example=30.0
    )
    layers: List[str] = Field(
        default=["elevation", "slope", "aspect", "land_cover"],
        description="Terrain layers to retrieve",
        example=["elevation", "slope", "aspect", "land_cover"]
    )

@router.post("/data/weather")
async def get_weather_data(request: WeatherRequest):
    """
    Retrieve weather data for a specific region and time period.
    """
    try:
        logger.info(f"Retrieving weather data for region: {request.region}")
        
        # Validate date range
        if request.end_date < request.start_date:
            raise HTTPException(
                status_code=400,
                detail="End date must be after start date"
            )
        
        # Mock weather data generation
        mock_data = []
        current_date = request.start_date
        
        while current_date <= request.end_date:
            daily_data = {
                "date": current_date.isoformat(),
                "region": request.region,
                "data": {}
            }
            
            # Generate mock weather values
            if "temperature" in request.parameters:
                daily_data["data"]["temperature"] = 25.0 + (hash(str(current_date)) % 20)
            if "humidity" in request.parameters:
                daily_data["data"]["humidity"] = 40.0 + (hash(str(current_date)) % 40)
            if "wind_speed" in request.parameters:
                daily_data["data"]["wind_speed"] = 2.0 + (hash(str(current_date)) % 15)
            if "precipitation" in request.parameters:
                daily_data["data"]["precipitation"] = max(0, (hash(str(current_date)) % 10) - 8)
            if "wind_direction" in request.parameters:
                daily_data["data"]["wind_direction"] = hash(str(current_date)) % 360
                
            mock_data.append(daily_data)
            current_date = date.fromordinal(current_date.toordinal() + 1)
        
        return {
            "status": "success",
            "region": request.region,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "parameters": request.parameters,
            "data": mock_data,
            "metadata": {
                "source": "ERA5-Land Reanalysis",
                "resolution": "0.1 degrees",
                "created_at": datetime.now().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Weather data retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve weather data: {str(e)}"
        )

@router.post("/data/terrain")
async def get_terrain_data(request: TerrainRequest):
    """
    Retrieve terrain data for a specific region.
    """
    try:
        logger.info(f"Retrieving terrain data for region: {request.region}")
        
        # Validate region size
        area_deg = ((request.region["max_lat"] - request.region["min_lat"]) * 
                   (request.region["max_lon"] - request.region["min_lon"]))
        if area_deg > 25.0:  # Maximum 25 square degrees for terrain data
            raise HTTPException(
                status_code=400,
                detail="Region too large for terrain data. Maximum area is 25 square degrees."
            )
        
        # Generate mock terrain data URLs
        data_id = str(uuid.uuid4())
        terrain_data = {
            "data_id": data_id,
            "region": request.region,
            "resolution": request.resolution,
            "layers": {},
            "download_urls": {}
        }
        
        # Generate URLs for each requested layer
        base_url = f"/api/v1/data/terrain/{data_id}"
        
        if "elevation" in request.layers:
            terrain_data["layers"]["elevation"] = {
                "description": "Digital Elevation Model",
                "units": "meters",
                "source": "SRTM 30m",
                "statistics": {
                    "min_elevation": 200.5,
                    "max_elevation": 3250.8,
                    "mean_elevation": 1125.3
                }
            }
            terrain_data["download_urls"]["elevation"] = f"{base_url}/elevation.tif"
        
        if "slope" in request.layers:
            terrain_data["layers"]["slope"] = {
                "description": "Terrain slope derived from DEM",
                "units": "degrees",
                "source": "Calculated from SRTM",
                "statistics": {
                    "min_slope": 0.0,
                    "max_slope": 65.2,
                    "mean_slope": 12.8
                }
            }
            terrain_data["download_urls"]["slope"] = f"{base_url}/slope.tif"
        
        if "aspect" in request.layers:
            terrain_data["layers"]["aspect"] = {
                "description": "Terrain aspect (slope direction)",
                "units": "degrees",
                "source": "Calculated from SRTM",
                "statistics": {
                    "min_aspect": 0.0,
                    "max_aspect": 360.0,
                    "mean_aspect": 180.0
                }
            }
            terrain_data["download_urls"]["aspect"] = f"{base_url}/aspect.tif"
        
        if "land_cover" in request.layers:
            terrain_data["layers"]["land_cover"] = {
                "description": "Land use and land cover classification",
                "units": "category",
                "source": "ESA WorldCover 2021",
                "classes": {
                    "1": "Forest",
                    "2": "Grassland",
                    "3": "Cropland",
                    "4": "Built-up",
                    "5": "Water"
                }
            }
            terrain_data["download_urls"]["land_cover"] = f"{base_url}/land_cover.tif"
        
        terrain_data["metadata"] = {
            "created_at": datetime.now().isoformat(),
            "coordinate_system": "EPSG:4326",
            "pixel_size_degrees": request.resolution / 111320.0,  # Convert meters to degrees
            "processing_time": 0.25
        }
        
        return terrain_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Terrain data retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve terrain data: {str(e)}"
        )

@router.get("/data/sources")
async def get_data_sources():
    """
    Get information about available data sources.
    """
    try:
        sources = {
            "weather": {
                "ERA5-Land": {
                    "description": "ECMWF Reanalysis v5 Land dataset",
                    "temporal_resolution": "hourly",
                    "spatial_resolution": "0.1 degrees (~11 km)",
                    "parameters": ["temperature", "humidity", "wind_speed", "wind_direction", "precipitation"],
                    "coverage": "Global",
                    "update_frequency": "Real-time with 5-day delay"
                },
                "IMD": {
                    "description": "India Meteorological Department data",
                    "temporal_resolution": "daily",
                    "spatial_resolution": "0.25 degrees (~25 km)",
                    "parameters": ["temperature", "rainfall", "humidity"],
                    "coverage": "India",
                    "update_frequency": "Daily"
                }
            },
            "terrain": {
                "SRTM": {
                    "description": "Shuttle Radar Topography Mission DEM",
                    "spatial_resolution": "30 meters",
                    "coverage": "Global (60°N to 56°S)",
                    "derived_products": ["elevation", "slope", "aspect"],
                    "accuracy": "±16 meters (90% confidence)"
                },
                "ESA_WorldCover": {
                    "description": "ESA WorldCover land cover classification",
                    "spatial_resolution": "10 meters",
                    "coverage": "Global",
                    "classes": 11,
                    "reference_year": "2021"
                }
            },
            "satellite": {
                "Landsat": {
                    "description": "Landsat 8/9 surface reflectance",
                    "spatial_resolution": "30 meters",
                    "temporal_resolution": "16 days",
                    "bands": ["Red", "Green", "Blue", "NIR", "SWIR"],
                    "derived_products": ["NDVI", "NDWI", "NBR"]
                },
                "Sentinel-2": {
                    "description": "Sentinel-2 MultiSpectral Instrument",
                    "spatial_resolution": "10/20 meters",
                    "temporal_resolution": "5 days",
                    "bands": 13,
                    "derived_products": ["NDVI", "LAI", "FAPAR"]
                }
            },
            "fire": {
                "VIIRS": {
                    "description": "VIIRS active fire detection",
                    "spatial_resolution": "375 meters",
                    "temporal_resolution": "daily",
                    "coverage": "Global",
                    "confidence_levels": ["low", "nominal", "high"]
                },
                "MODIS": {
                    "description": "MODIS active fire product",
                    "spatial_resolution": "1 km",
                    "temporal_resolution": "daily",
                    "coverage": "Global",
                    "products": ["MOD14", "MYD14"]
                }
            }
        }
        
        return {
            "data_sources": sources,
            "last_updated": datetime.now().isoformat(),
            "total_sources": sum(len(category) for category in sources.values())
        }
        
    except Exception as e:
        logger.error(f"Failed to get data sources: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve data sources"
        )

@router.get("/data/regions/uttarakhand")
async def get_uttarakhand_info():
    """
    Get information about Uttarakhand region for fire prediction.
    """
    try:
        uttarakhand_info = {
            "name": "Uttarakhand",
            "country": "India",
            "area_km2": 53483,
            "coordinates": {
                "min_lat": 28.8,
                "max_lat": 31.4,
                "min_lon": 77.3,
                "max_lon": 81.0
            },
            "districts": [
                "Almora", "Bageshwar", "Chamoli", "Champawat", "Dehradun",
                "Haridwar", "Nainital", "Pauri Garhwal", "Pithoragarh",
                "Rudraprayag", "Tehri Garhwal", "Udham Singh Nagar", "Uttarkashi"
            ],
            "fire_season": {
                "peak_months": ["March", "April", "May"],
                "secondary_peak": ["October", "November"],
                "low_risk_months": ["July", "August", "September"]
            },
            "vegetation_types": {
                "forest_cover_percent": 71.05,
                "major_types": [
                    "Subtropical Pine Forest",
                    "Temperate Coniferous Forest",
                    "Alpine Meadows",
                    "Deciduous Forest"
                ]
            },
            "climate": {
                "type": "Subtropical Highland",
                "average_temperature_range": "15-25°C",
                "monsoon_months": ["June", "July", "August", "September"],
                "dry_months": ["October", "November", "December", "January", "February"]
            },
            "fire_risk_factors": [
                "Dry deciduous forests in lower elevations",
                "Pine forests with resin content",
                "Agricultural residue burning",
                "Tourist activities and camping",
                "Traditional practices (slash and burn)",
                "Power line infrastructure"
            ],
            "recommended_monitoring": {
                "resolution": "30 meters",
                "frequency": "daily during fire season",
                "priority_areas": [
                    "Rajaji National Park",
                    "Jim Corbett National Park",
                    "Nanda Devi Biosphere Reserve",
                    "Valley of Flowers National Park"
                ]
            }
        }
        
        return uttarakhand_info
        
    except Exception as e:
        logger.error(f"Failed to get Uttarakhand info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve Uttarakhand information"
        )

@router.get("/data/prediction/{prediction_id}/probability.tif")
async def download_prediction_raster(prediction_id: str):
    """
    Download fire probability prediction raster.
    """
    try:
        # In a real implementation, this would serve the actual raster file
        # For now, return a placeholder response
        return Response(
            content=b"Mock raster data - implement actual file serving",
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename=fire_probability_{prediction_id}.tif"}
        )
        
    except Exception as e:
        logger.error(f"Failed to download prediction raster: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to download prediction raster"
        )
