-- BlazeNet Database Initialization Script
-- This script sets up the database schema for the BlazeNet fire prediction system

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create fire predictions table
CREATE TABLE IF NOT EXISTS fire_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_bounds GEOMETRY(POLYGON, 4326) NOT NULL,
    prediction_date DATE NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    resolution FLOAT NOT NULL DEFAULT 30.0,
    weather_data JSONB,
    statistics JSONB,
    metadata JSONB,
    raster_path TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_time FLOAT
);

-- Create fire simulations table
CREATE TABLE IF NOT EXISTS fire_simulations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_bounds GEOMETRY(POLYGON, 4326) NOT NULL,
    ignition_points GEOMETRY(MULTIPOINT, 4326) NOT NULL,
    weather_conditions JSONB NOT NULL,
    simulation_hours INTEGER NOT NULL DEFAULT 24,
    time_step FLOAT NOT NULL DEFAULT 1.0,
    resolution FLOAT NOT NULL DEFAULT 30.0,
    statistics JSONB,
    metadata JSONB,
    animation_path TEXT,
    final_state_path TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_time FLOAT
);

-- Create weather data table
CREATE TABLE IF NOT EXISTS weather_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    location GEOMETRY(POINT, 4326) NOT NULL,
    date_time TIMESTAMP WITH TIME ZONE NOT NULL,
    temperature FLOAT,
    humidity FLOAT,
    wind_speed FLOAT,
    wind_direction FLOAT,
    precipitation FLOAT,
    source VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create fire history table
CREATE TABLE IF NOT EXISTS fire_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fire_location GEOMETRY(POINT, 4326) NOT NULL,
    fire_polygon GEOMETRY(POLYGON, 4326),
    start_date TIMESTAMP WITH TIME ZONE,
    burned_area_ha FLOAT,
    cause VARCHAR(100),
    confidence FLOAT,
    source VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create spatial indexes
CREATE INDEX IF NOT EXISTS idx_fire_predictions_region ON fire_predictions USING GIST(region_bounds);
CREATE INDEX IF NOT EXISTS idx_fire_predictions_date ON fire_predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_fire_simulations_region ON fire_simulations USING GIST(region_bounds);
CREATE INDEX IF NOT EXISTS idx_weather_data_location ON weather_data USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_fire_history_location ON fire_history USING GIST(fire_location);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO blazenet;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO blazenet;
