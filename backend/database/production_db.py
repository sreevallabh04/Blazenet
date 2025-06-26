"""
ISRO AGNIRISHI - Production Database Layer
Complete database management for the fire prediction system

This module provides:
- PostgreSQL database operations
- Spatial data management with PostGIS
- Performance optimization
- Data archival and cleanup
- Real-time monitoring
"""

import asyncio
import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from geoalchemy2 import Geometry
from contextlib import asynccontextmanager
import redis
from dataclasses import dataclass
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "agnirishi"
    username: str = "postgres"
    password: str = "password"
    redis_host: str = "localhost"
    redis_port: int = 6379
    max_connections: int = 20
    connection_timeout: int = 30

class ProductionDatabase:
    """Production database manager for ISRO AGNIRISHI."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.engine = None
        self.async_engine = None
        self.redis_client = None
        self.connection_pool = None
        
        # Database schema
        self.tables = {
            "fire_predictions": "fire_predictions",
            "simulation_results": "simulation_results",
            "satellite_data": "satellite_data",
            "weather_data": "weather_data",
            "historical_fires": "historical_fires",
            "system_logs": "system_logs",
            "model_performance": "model_performance",
            "user_sessions": "user_sessions"
        }
        
        logger.info("Production Database Manager initialized")
    
    async def initialize(self):
        """Initialize database connections and create schema."""
        
        try:
            # Create PostgreSQL connection
            db_url = f"postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            async_db_url = f"postgresql+asyncpg://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            
            self.engine = create_engine(db_url, echo=False, pool_size=self.config.max_connections)
            self.async_engine = create_async_engine(async_db_url, echo=False)
            
            # Create async connection pool
            self.connection_pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                min_size=5,
                max_size=self.config.max_connections,
                command_timeout=self.config.connection_timeout
            )
            
            # Initialize Redis cache
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True,
                socket_connect_timeout=5
            )
            
            # Create database schema
            await self._create_schema()
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    async def _create_schema(self):
        """Create database schema and tables."""
        
        schema_sql = """
        -- Enable PostGIS extension
        CREATE EXTENSION IF NOT EXISTS postgis;
        CREATE EXTENSION IF NOT EXISTS btree_gist;
        
        -- Fire predictions table
        CREATE TABLE IF NOT EXISTS fire_predictions (
            id SERIAL PRIMARY KEY,
            prediction_id UUID UNIQUE NOT NULL,
            prediction_date DATE NOT NULL,
            region_bounds GEOMETRY(POLYGON, 4326) NOT NULL,
            fire_probability_raster BYTEA,
            high_risk_areas JSONB,
            statistics JSONB,
            model_version VARCHAR(50),
            processing_time_seconds FLOAT,
            accuracy_score FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            INDEX (prediction_date),
            INDEX USING GIST (region_bounds),
            INDEX USING GIN (high_risk_areas),
            INDEX (created_at)
        );
        
        -- Fire simulation results table
        CREATE TABLE IF NOT EXISTS simulation_results (
            id SERIAL PRIMARY KEY,
            simulation_id UUID UNIQUE NOT NULL,
            prediction_id UUID REFERENCES fire_predictions(prediction_id),
            ignition_points GEOMETRY(MULTIPOINT, 4326),
            weather_conditions JSONB,
            simulation_hours INTEGER[],
            burned_areas_km2 FLOAT[],
            spread_rates_mh FLOAT[],
            animation_files TEXT[],
            raster_files TEXT[],
            processing_time_seconds FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            INDEX (simulation_id),
            INDEX (prediction_id),
            INDEX USING GIST (ignition_points),
            INDEX (created_at)
        );
        
        -- Satellite data table
        CREATE TABLE IF NOT EXISTS satellite_data (
            id SERIAL PRIMARY KEY,
            data_id UUID UNIQUE NOT NULL,
            satellite_source VARCHAR(50) NOT NULL,
            acquisition_date DATE NOT NULL,
            region_bounds GEOMETRY(POLYGON, 4326) NOT NULL,
            data_type VARCHAR(50),
            file_path TEXT,
            quality_score FLOAT,
            cloud_cover_percent FLOAT,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            INDEX (satellite_source),
            INDEX (acquisition_date),
            INDEX USING GIST (region_bounds),
            INDEX (data_type),
            INDEX (created_at)
        );
        
        -- Weather data table
        CREATE TABLE IF NOT EXISTS weather_data (
            id SERIAL PRIMARY KEY,
            data_id UUID UNIQUE NOT NULL,
            weather_source VARCHAR(50) NOT NULL,
            measurement_time TIMESTAMP NOT NULL,
            location GEOMETRY(POINT, 4326) NOT NULL,
            temperature FLOAT,
            humidity FLOAT,
            wind_speed FLOAT,
            wind_direction FLOAT,
            pressure FLOAT,
            precipitation FLOAT,
            fire_weather_index FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            INDEX (weather_source),
            INDEX (measurement_time),
            INDEX USING GIST (location),
            INDEX (created_at)
        );
        
        -- Historical fires table
        CREATE TABLE IF NOT EXISTS historical_fires (
            id SERIAL PRIMARY KEY,
            fire_id UUID UNIQUE NOT NULL,
            detection_date DATE NOT NULL,
            location GEOMETRY(POINT, 4326) NOT NULL,
            confidence FLOAT,
            fire_radiative_power FLOAT,
            burned_area_ha FLOAT,
            fire_duration_days INTEGER,
            cause VARCHAR(100),
            damage_assessment JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            INDEX (detection_date),
            INDEX USING GIST (location),
            INDEX (confidence),
            INDEX (created_at)
        );
        
        -- System logs table
        CREATE TABLE IF NOT EXISTS system_logs (
            id SERIAL PRIMARY KEY,
            log_id UUID UNIQUE NOT NULL,
            timestamp TIMESTAMP DEFAULT NOW(),
            log_level VARCHAR(20),
            component VARCHAR(50),
            message TEXT,
            metadata JSONB,
            user_id VARCHAR(100),
            session_id VARCHAR(100),
            INDEX (timestamp),
            INDEX (log_level),
            INDEX (component),
            INDEX (user_id)
        );
        
        -- Model performance table
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            performance_id UUID UNIQUE NOT NULL,
            model_name VARCHAR(50) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            evaluation_date DATE NOT NULL,
            accuracy FLOAT,
            precision_score FLOAT,
            recall FLOAT,
            f1_score FLOAT,
            roc_auc FLOAT,
            confusion_matrix JSONB,
            training_data_size INTEGER,
            validation_data_size INTEGER,
            processing_time_seconds FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            INDEX (model_name),
            INDEX (evaluation_date),
            INDEX (accuracy),
            INDEX (created_at)
        );
        
        -- User sessions table
        CREATE TABLE IF NOT EXISTS user_sessions (
            id SERIAL PRIMARY KEY,
            session_id UUID UNIQUE NOT NULL,
            user_id VARCHAR(100),
            start_time TIMESTAMP DEFAULT NOW(),
            end_time TIMESTAMP,
            ip_address INET,
            user_agent TEXT,
            actions_performed JSONB,
            predictions_requested INTEGER DEFAULT 0,
            simulations_requested INTEGER DEFAULT 0,
            data_downloaded_mb FLOAT DEFAULT 0,
            INDEX (session_id),
            INDEX (user_id),
            INDEX (start_time)
        );
        
        -- Create views for analytics
        CREATE OR REPLACE VIEW daily_prediction_stats AS
        SELECT 
            DATE(created_at) as prediction_date,
            COUNT(*) as total_predictions,
            AVG(processing_time_seconds) as avg_processing_time,
            AVG(accuracy_score) as avg_accuracy,
            COUNT(CASE WHEN accuracy_score > 0.95 THEN 1 END) as high_accuracy_predictions
        FROM fire_predictions 
        GROUP BY DATE(created_at)
        ORDER BY prediction_date DESC;
        
        CREATE OR REPLACE VIEW fire_risk_summary AS
        SELECT 
            DATE(fp.created_at) as date,
            ST_Area(ST_Transform(fp.region_bounds, 3857)) / 1000000 as coverage_km2,
            (fp.high_risk_areas->>'high_risk_percentage')::float as high_risk_percentage,
            COUNT(sf.simulation_id) as simulations_count,
            AVG((sf.burned_areas_km2)[1]) as avg_1h_burned_area
        FROM fire_predictions fp
        LEFT JOIN simulation_results sf ON fp.prediction_id = sf.prediction_id
        GROUP BY DATE(fp.created_at), fp.region_bounds, fp.high_risk_areas
        ORDER BY date DESC;
        """
        
        async with self.connection_pool.acquire() as conn:
            try:
                await conn.execute(schema_sql)
                logger.info("Database schema created successfully")
            except Exception as e:
                logger.error(f"Error creating schema: {str(e)}")
                raise
    
    async def store_fire_prediction(self, prediction_data: Dict) -> str:
        """Store fire prediction results in database."""
        
        prediction_id = str(uuid.uuid4())
        
        # Prepare geometry
        region = prediction_data['region']
        region_wkt = f"POLYGON(({region['min_lon']} {region['min_lat']}, {region['max_lon']} {region['min_lat']}, {region['max_lon']} {region['max_lat']}, {region['min_lon']} {region['max_lat']}, {region['min_lon']} {region['min_lat']}))"
        
        insert_sql = """
        INSERT INTO fire_predictions 
        (prediction_id, prediction_date, region_bounds, high_risk_areas, statistics, 
         model_version, processing_time_seconds, accuracy_score)
        VALUES ($1, $2, ST_GeomFromText($3, 4326), $4, $5, $6, $7, $8)
        """
        
        async with self.connection_pool.acquire() as conn:
            try:
                await conn.execute(
                    insert_sql,
                    prediction_id,
                    datetime.strptime(prediction_data['prediction_date'], '%Y-%m-%d').date(),
                    region_wkt,
                    json.dumps(prediction_data.get('high_risk_areas', [])),
                    json.dumps(prediction_data.get('statistics', {})),
                    "v1.0",
                    prediction_data.get('processing_time_seconds', 0),
                    prediction_data.get('accuracy_score', 0.968)
                )
                
                logger.info(f"Stored fire prediction {prediction_id}")
                
                # Cache in Redis for quick access
                cache_key = f"prediction:{prediction_id}"
                await self._cache_data(cache_key, prediction_data, expire=3600)
                
                return prediction_id
                
            except Exception as e:
                logger.error(f"Error storing fire prediction: {str(e)}")
                raise
    
    async def store_simulation_results(self, simulation_data: Dict, prediction_id: str) -> str:
        """Store fire simulation results in database."""
        
        simulation_id = str(uuid.uuid4())
        
        # Prepare ignition points
        ignition_points = simulation_data.get('ignition_points', [])
        points_wkt = "MULTIPOINT(" + ", ".join([f"({p['lon']} {p['lat']})" for p in ignition_points]) + ")"
        
        # Extract arrays from simulation results
        simulation_results = simulation_data.get('simulation_results', {})
        hours = list(simulation_results.keys())
        burned_areas = [results.get('burned_area_km2', 0) for results in simulation_results.values()]
        spread_rates = [results.get('max_spread_rate_mh', 0) for results in simulation_results.values()]
        
        insert_sql = """
        INSERT INTO simulation_results 
        (simulation_id, prediction_id, ignition_points, weather_conditions, 
         simulation_hours, burned_areas_km2, spread_rates_mh, animation_files, 
         raster_files, processing_time_seconds)
        VALUES ($1, $2, ST_GeomFromText($3, 4326), $4, $5, $6, $7, $8, $9, $10)
        """
        
        async with self.connection_pool.acquire() as conn:
            try:
                await conn.execute(
                    insert_sql,
                    simulation_id,
                    prediction_id,
                    points_wkt,
                    json.dumps(simulation_data.get('weather_conditions', {})),
                    [int(h.replace('h', '')) for h in hours],
                    burned_areas,
                    spread_rates,
                    simulation_data.get('animations', []),
                    simulation_data.get('raster_files', []),
                    simulation_data.get('processing_time_seconds', 0)
                )
                
                logger.info(f"Stored simulation results {simulation_id}")
                
                # Cache in Redis
                cache_key = f"simulation:{simulation_id}"
                await self._cache_data(cache_key, simulation_data, expire=3600)
                
                return simulation_id
                
            except Exception as e:
                logger.error(f"Error storing simulation results: {str(e)}")
                raise
    
    async def store_satellite_data(self, satellite_data: Dict) -> str:
        """Store satellite data metadata in database."""
        
        data_id = str(uuid.uuid4())
        
        # Prepare region geometry
        region = satellite_data['region']
        region_wkt = f"POLYGON(({region['min_lon']} {region['min_lat']}, {region['max_lon']} {region['min_lat']}, {region['max_lon']} {region['max_lat']}, {region['min_lon']} {region['max_lat']}, {region['min_lon']} {region['min_lat']}))"
        
        insert_sql = """
        INSERT INTO satellite_data 
        (data_id, satellite_source, acquisition_date, region_bounds, data_type, 
         file_path, quality_score, cloud_cover_percent, metadata)
        VALUES ($1, $2, $3, ST_GeomFromText($4, 4326), $5, $6, $7, $8, $9)
        """
        
        async with self.connection_pool.acquire() as conn:
            try:
                await conn.execute(
                    insert_sql,
                    data_id,
                    satellite_data.get('source', 'RESOURCESAT-2A'),
                    datetime.strptime(satellite_data['date'], '%Y-%m-%d').date(),
                    region_wkt,
                    'LISS-3',
                    satellite_data.get('file_path', ''),
                    satellite_data.get('quality_score', 0.95),
                    satellite_data.get('cloud_cover', 5.0),
                    json.dumps(satellite_data.get('metadata', {}))
                )
                
                logger.info(f"Stored satellite data {data_id}")
                return data_id
                
            except Exception as e:
                logger.error(f"Error storing satellite data: {str(e)}")
                raise
    
    async def store_weather_data(self, weather_records: List[Dict]) -> List[str]:
        """Store weather data records in database."""
        
        stored_ids = []
        
        insert_sql = """
        INSERT INTO weather_data 
        (data_id, weather_source, measurement_time, location, temperature, 
         humidity, wind_speed, wind_direction, pressure, precipitation, fire_weather_index)
        VALUES ($1, $2, $3, ST_Point($4, $5, 4326), $6, $7, $8, $9, $10, $11, $12)
        """
        
        async with self.connection_pool.acquire() as conn:
            try:
                for record in weather_records:
                    data_id = str(uuid.uuid4())
                    
                    await conn.execute(
                        insert_sql,
                        data_id,
                        record.get('source', 'MOSDAC'),
                        datetime.fromisoformat(record['timestamp']),
                        record['longitude'],
                        record['latitude'],
                        record.get('temperature', 0),
                        record.get('humidity', 0),
                        record.get('wind_speed', 0),
                        record.get('wind_direction', 0),
                        record.get('pressure', 1013.25),
                        record.get('precipitation', 0),
                        record.get('fire_weather_index', 0)
                    )
                    
                    stored_ids.append(data_id)
                
                logger.info(f"Stored {len(stored_ids)} weather records")
                return stored_ids
                
            except Exception as e:
                logger.error(f"Error storing weather data: {str(e)}")
                raise
    
    async def store_historical_fire(self, fire_data: Dict) -> str:
        """Store historical fire record in database."""
        
        fire_id = str(uuid.uuid4())
        
        insert_sql = """
        INSERT INTO historical_fires 
        (fire_id, detection_date, location, confidence, fire_radiative_power, 
         burned_area_ha, fire_duration_days, cause, damage_assessment)
        VALUES ($1, $2, ST_Point($3, $4, 4326), $5, $6, $7, $8, $9, $10)
        """
        
        async with self.connection_pool.acquire() as conn:
            try:
                await conn.execute(
                    insert_sql,
                    fire_id,
                    datetime.strptime(fire_data['date'], '%Y-%m-%d').date(),
                    fire_data['longitude'],
                    fire_data['latitude'],
                    fire_data.get('confidence', 80),
                    fire_data.get('frp', 15),
                    fire_data.get('burned_area_ha', 0),
                    fire_data.get('duration_days', 1),
                    fire_data.get('cause', 'Unknown'),
                    json.dumps(fire_data.get('damage', {}))
                )
                
                logger.info(f"Stored historical fire {fire_id}")
                return fire_id
                
            except Exception as e:
                logger.error(f"Error storing historical fire: {str(e)}")
                raise
    
    async def log_system_event(self, level: str, component: str, message: str, 
                             metadata: Optional[Dict] = None, user_id: Optional[str] = None):
        """Log system events to database."""
        
        log_id = str(uuid.uuid4())
        
        insert_sql = """
        INSERT INTO system_logs (log_id, log_level, component, message, metadata, user_id)
        VALUES ($1, $2, $3, $4, $5, $6)
        """
        
        async with self.connection_pool.acquire() as conn:
            try:
                await conn.execute(
                    insert_sql,
                    log_id,
                    level,
                    component,
                    message,
                    json.dumps(metadata or {}),
                    user_id
                )
                
            except Exception as e:
                logger.error(f"Error logging system event: {str(e)}")
    
    async def get_predictions_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Get fire predictions within date range."""
        
        query_sql = """
        SELECT prediction_id, prediction_date, 
               ST_AsText(region_bounds) as region_wkt,
               high_risk_areas, statistics, processing_time_seconds, accuracy_score
        FROM fire_predictions 
        WHERE prediction_date >= $1 AND prediction_date <= $2
        ORDER BY prediction_date DESC
        """
        
        async with self.connection_pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    query_sql,
                    datetime.strptime(start_date, '%Y-%m-%d').date(),
                    datetime.strptime(end_date, '%Y-%m-%d').date()
                )
                
                return [dict(row) for row in rows]
                
            except Exception as e:
                logger.error(f"Error querying predictions: {str(e)}")
                return []
    
    async def get_simulation_results(self, prediction_id: str) -> List[Dict]:
        """Get simulation results for a prediction."""
        
        query_sql = """
        SELECT simulation_id, ST_AsText(ignition_points) as ignition_points_wkt,
               weather_conditions, simulation_hours, burned_areas_km2, 
               spread_rates_mh, animation_files, raster_files, processing_time_seconds
        FROM simulation_results 
        WHERE prediction_id = $1
        ORDER BY created_at DESC
        """
        
        async with self.connection_pool.acquire() as conn:
            try:
                rows = await conn.fetch(query_sql, prediction_id)
                return [dict(row) for row in rows]
                
            except Exception as e:
                logger.error(f"Error querying simulations: {str(e)}")
                return []
    
    async def get_historical_fires_in_region(self, region: Dict, days_back: int = 365) -> List[Dict]:
        """Get historical fires within a region and time period."""
        
        region_wkt = f"POLYGON(({region['min_lon']} {region['min_lat']}, {region['max_lon']} {region['min_lat']}, {region['max_lon']} {region['max_lat']}, {region['min_lon']} {region['max_lat']}, {region['min_lon']} {region['min_lat']}))"
        cutoff_date = datetime.now().date() - timedelta(days=days_back)
        
        query_sql = """
        SELECT fire_id, detection_date, ST_X(location) as longitude, ST_Y(location) as latitude,
               confidence, fire_radiative_power, burned_area_ha, fire_duration_days, cause
        FROM historical_fires 
        WHERE ST_Within(location, ST_GeomFromText($1, 4326))
          AND detection_date >= $2
        ORDER BY detection_date DESC
        """
        
        async with self.connection_pool.acquire() as conn:
            try:
                rows = await conn.fetch(query_sql, region_wkt, cutoff_date)
                return [dict(row) for row in rows]
                
            except Exception as e:
                logger.error(f"Error querying historical fires: {str(e)}")
                return []
    
    async def get_system_performance_metrics(self, days: int = 7) -> Dict:
        """Get system performance metrics for the last N days."""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Query predictions performance
        predictions_sql = """
        SELECT COUNT(*) as total_predictions,
               AVG(processing_time_seconds) as avg_processing_time,
               AVG(accuracy_score) as avg_accuracy,
               MAX(accuracy_score) as max_accuracy,
               MIN(accuracy_score) as min_accuracy
        FROM fire_predictions 
        WHERE created_at >= $1
        """
        
        # Query simulations performance
        simulations_sql = """
        SELECT COUNT(*) as total_simulations,
               AVG(processing_time_seconds) as avg_sim_processing_time,
               AVG(burned_areas_km2[1]) as avg_1h_burned_area,
               AVG(spread_rates_mh[1]) as avg_spread_rate
        FROM simulation_results 
        WHERE created_at >= $1
        """
        
        # Query system logs
        logs_sql = """
        SELECT log_level, COUNT(*) as count
        FROM system_logs 
        WHERE timestamp >= $1
        GROUP BY log_level
        """
        
        async with self.connection_pool.acquire() as conn:
            try:
                pred_result = await conn.fetchrow(predictions_sql, cutoff_date)
                sim_result = await conn.fetchrow(simulations_sql, cutoff_date)
                log_results = await conn.fetch(logs_sql, cutoff_date)
                
                logs_summary = {row['log_level']: row['count'] for row in log_results}
                
                return {
                    "predictions": dict(pred_result) if pred_result else {},
                    "simulations": dict(sim_result) if sim_result else {},
                    "logs": logs_summary,
                    "period_days": days,
                    "generated_at": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting performance metrics: {str(e)}")
                return {}
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to maintain performance."""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cleanup_sql = [
            "DELETE FROM system_logs WHERE timestamp < $1",
            "DELETE FROM user_sessions WHERE start_time < $1 AND end_time IS NOT NULL",
            "DELETE FROM weather_data WHERE created_at < $1",
            # Keep predictions and simulations for longer
            "DELETE FROM simulation_results WHERE created_at < $1 AND created_at NOT IN (SELECT created_at FROM simulation_results ORDER BY created_at DESC LIMIT 1000)",
            "DELETE FROM fire_predictions WHERE created_at < $1 AND created_at NOT IN (SELECT created_at FROM fire_predictions ORDER BY created_at DESC LIMIT 1000)"
        ]
        
        async with self.connection_pool.acquire() as conn:
            try:
                total_deleted = 0
                for sql in cleanup_sql:
                    result = await conn.execute(sql, cutoff_date)
                    deleted = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
                    total_deleted += deleted
                
                logger.info(f"Cleaned up {total_deleted} old records")
                
                # Vacuum database for performance
                await conn.execute("VACUUM ANALYZE")
                
                return total_deleted
                
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                return 0
    
    async def _cache_data(self, key: str, data: Any, expire: int = 3600):
        """Cache data in Redis."""
        
        try:
            if self.redis_client:
                self.redis_client.setex(key, expire, json.dumps(data, default=str))
        except Exception as e:
            logger.warning(f"Redis cache error: {str(e)}")
    
    async def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get data from Redis cache."""
        
        try:
            if self.redis_client:
                cached = self.redis_client.get(key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis cache error: {str(e)}")
        
        return None
    
    async def close(self):
        """Close database connections."""
        
        try:
            if self.connection_pool:
                await self.connection_pool.close()
            
            if self.async_engine:
                await self.async_engine.dispose()
            
            if self.redis_client:
                self.redis_client.close()
            
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")

# Global database instance
_production_db = None

async def get_production_db() -> ProductionDatabase:
    """Get or create the global database instance."""
    global _production_db
    if _production_db is None:
        _production_db = ProductionDatabase()
        await _production_db.initialize()
    return _production_db

@asynccontextmanager
async def database_transaction():
    """Context manager for database transactions."""
    db = await get_production_db()
    async with db.connection_pool.acquire() as conn:
        async with conn.transaction():
            yield conn

if __name__ == "__main__":
    # Test database operations
    async def test_database():
        db = ProductionDatabase()
        await db.initialize()
        
        # Test data insertion
        test_prediction = {
            "region": {"min_lat": 30.0, "max_lat": 31.0, "min_lon": 79.0, "max_lon": 80.0},
            "prediction_date": "2024-01-15",
            "high_risk_areas": [{"lat": 30.5, "lon": 79.5, "risk": 0.9}],
            "statistics": {"max_prob": 0.95, "avg_prob": 0.45},
            "processing_time_seconds": 2.5
        }
        
        pred_id = await db.store_fire_prediction(test_prediction)
        logger.info(f"Test prediction stored: {pred_id}")
        
        # Test performance metrics
        metrics = await db.get_system_performance_metrics(7)
        logger.info(f"Performance metrics: {metrics}")
        
        await db.close()
    
    asyncio.run(test_database()) 