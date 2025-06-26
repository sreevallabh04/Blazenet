"""
Main FastAPI application for BlazeNet backend.
"""

import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.config import config
from app.backend.api.fire_prediction import router as fire_prediction_router
from app.backend.api.fire_simulation import router as fire_simulation_router
from app.backend.api.data import router as data_router
from app.backend.database.connection import database_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="BlazeNet API",
    description="Geospatial ML Forest Fire Prediction & Simulation System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(fire_prediction_router, prefix="/api/v1", tags=["fire-prediction"])
app.include_router(fire_simulation_router, prefix="/api/v1", tags=["fire-simulation"])
app.include_router(data_router, prefix="/api/v1", tags=["data"])

@app.on_event("startup")
async def startup_event():
    """Initialize database and other services on startup."""
    logger.info("Starting BlazeNet API...")
    
    try:
        # Initialize database
        await database_manager.initialize()
        logger.info("Database initialized successfully")
        
        # Initialize other services (Redis, ML models, etc.)
        # await initialize_ml_models()
        # await initialize_redis()
        
        logger.info("BlazeNet API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start BlazeNet API: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    logger.info("Shutting down BlazeNet API...")
    
    try:
        # Close database connections
        await database_manager.close()
        logger.info("Database connections closed")
        
        logger.info("BlazeNet API shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to BlazeNet API",
        "description": "Geospatial ML Forest Fire Prediction & Simulation System",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "fire_prediction": "/api/v1/predict/fire-probability",
            "fire_simulation": "/api/v1/simulate/fire-spread",
            "weather_data": "/api/v1/data/weather",
            "terrain_data": "/api/v1/data/terrain"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db_status = await database_manager.health_check()
        
        return {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "services": {
                "database": "healthy" if db_status else "unhealthy",
                "api": "healthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info"
    ) 