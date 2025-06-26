"""
Geospatial utilities for BlazeNet system.
Handles raster operations, coordinate transformations, and spatial data processing.
"""

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.geometry import box, Point, Polygon
from pyproj import Transformer
import xarray as xr
from typing import Tuple, List, Optional, Union, Dict
import logging

logger = logging.getLogger(__name__)

class GeoUtils:
    """Utility class for geospatial operations."""
    
    @staticmethod
    def create_raster_from_bounds(
        bounds: Tuple[float, float, float, float],
        resolution: float,
        crs: str = "EPSG:4326",
        data: Optional[np.ndarray] = None,
        fill_value: float = 0.0
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        """
        Create a raster from geographic bounds.
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            resolution: Pixel resolution in degrees or meters
            crs: Coordinate reference system
            data: Optional data array
            fill_value: Fill value for empty pixels
            
        Returns:
            Tuple of (data_array, transform)
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Calculate dimensions
        width = int((max_lon - min_lon) / resolution)
        height = int((max_lat - min_lat) / resolution)
        
        # Create transform
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
        
        # Create data array if not provided
        if data is None:
            data = np.full((height, width), fill_value, dtype=np.float32)
        
        return data, transform
    
    @staticmethod
    def reproject_raster(
        src_array: np.ndarray,
        src_transform: rasterio.Affine,
        src_crs: str,
        dst_crs: str,
        dst_resolution: Optional[float] = None
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        """
        Reproject raster to different CRS.
        
        Args:
            src_array: Source data array
            src_transform: Source transform
            src_crs: Source CRS
            dst_crs: Destination CRS
            dst_resolution: Target resolution (optional)
            
        Returns:
            Tuple of (reprojected_array, new_transform)
        """
        # Calculate default transform and dimensions
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src_array.shape[1], src_array.shape[0], *src_transform[:6]
        )
        
        # Adjust for specific resolution if provided
        if dst_resolution:
            bounds = rasterio.transform.array_bounds(src_array.shape[0], src_array.shape[1], src_transform)
            transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            dst_bounds = transformer.transform_bounds(*bounds)
            
            dst_width = int((dst_bounds[2] - dst_bounds[0]) / dst_resolution)
            dst_height = int((dst_bounds[3] - dst_bounds[1]) / dst_resolution)
            dst_transform = from_bounds(*dst_bounds, dst_width, dst_height)
        
        # Create destination array
        dst_array = np.zeros((dst_height, dst_width), dtype=src_array.dtype)
        
        # Reproject
        reproject(
            source=src_array,
            destination=dst_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        
        return dst_array, dst_transform
    
    @staticmethod
    def clip_raster_to_geometry(
        data: np.ndarray,
        transform: rasterio.Affine,
        geometry: Union[Polygon, List[Polygon]],
        crs: str = "EPSG:4326"
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        """
        Clip raster data to geometry bounds.
        
        Args:
            data: Input raster data
            transform: Raster transform
            geometry: Clipping geometry
            crs: Coordinate reference system
            
        Returns:
            Tuple of (clipped_data, new_transform)
        """
        if isinstance(geometry, Polygon):
            geometries = [geometry]
        else:
            geometries = geometry
        
        # Create a temporary raster dataset
        profile = {
            'driver': 'MEM',
            'height': data.shape[0],
            'width': data.shape[1],
            'count': 1,
            'dtype': data.dtype,
            'crs': crs,
            'transform': transform
        }
        
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(data, 1)
                
                # Clip to geometry
                clipped_data, clipped_transform = mask(
                    dataset, geometries, crop=True, filled=False
                )
                
                return clipped_data[0], clipped_transform
    
    @staticmethod
    def calculate_slope_aspect(
        dem: np.ndarray,
        transform: rasterio.Affine,
        resolution: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate slope and aspect from DEM.
        
        Args:
            dem: Digital elevation model
            transform: Raster transform
            resolution: Pixel resolution in meters
            
        Returns:
            Tuple of (slope, aspect) arrays
        """
        # Calculate gradients
        dy, dx = np.gradient(dem, resolution)
        
        # Calculate slope in degrees
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi
        
        # Calculate aspect in degrees (0-360)
        aspect = np.arctan2(-dx, dy) * 180 / np.pi
        aspect = np.where(aspect < 0, 360 + aspect, aspect)
        
        return slope.astype(np.float32), aspect.astype(np.float32)
    
    @staticmethod
    def normalize_raster(
        data: np.ndarray,
        method: str = "minmax",
        percentiles: Tuple[float, float] = (2, 98)
    ) -> np.ndarray:
        """
        Normalize raster data.
        
        Args:
            data: Input data array
            method: Normalization method ('minmax', 'zscore', 'percentile')
            percentiles: Percentile range for percentile normalization
            
        Returns:
            Normalized data array
        """
        data = data.astype(np.float32)
        
        if method == "minmax":
            min_val = np.nanmin(data)
            max_val = np.nanmax(data)
            if max_val > min_val:
                normalized = (data - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(data)
                
        elif method == "zscore":
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)
            if std_val > 0:
                normalized = (data - mean_val) / std_val
            else:
                normalized = np.zeros_like(data)
                
        elif method == "percentile":
            low_val = np.nanpercentile(data, percentiles[0])
            high_val = np.nanpercentile(data, percentiles[1])
            if high_val > low_val:
                normalized = np.clip((data - low_val) / (high_val - low_val), 0, 1)
            else:
                normalized = np.zeros_like(data)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    @staticmethod
    def create_distance_raster(
        shape: Tuple[int, int],
        transform: rasterio.Affine,
        points: List[Tuple[float, float]],
        crs: str = "EPSG:4326"
    ) -> np.ndarray:
        """
        Create distance raster from points.
        
        Args:
            shape: Output raster shape (height, width)
            transform: Raster transform
            points: List of (x, y) coordinates
            crs: Coordinate reference system
            
        Returns:
            Distance raster array
        """
        height, width = shape
        
        # Create coordinate arrays
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
        
        # Convert to coordinate arrays
        coord_array = np.column_stack([xs, ys])
        
        # Calculate minimum distance to any point
        min_distances = np.full(len(coord_array), np.inf)
        
        for point in points:
            distances = np.sqrt(
                (coord_array[:, 0] - point[0])**2 + 
                (coord_array[:, 1] - point[1])**2
            )
            min_distances = np.minimum(min_distances, distances)
        
        return min_distances.reshape(height, width).astype(np.float32)
    
    @staticmethod
    def create_wind_effect_raster(
        shape: Tuple[int, int],
        wind_speed: float,
        wind_direction: float,
        terrain_slope: np.ndarray,
        terrain_aspect: np.ndarray
    ) -> np.ndarray:
        """
        Create wind effect raster for fire simulation.
        
        Args:
            shape: Output raster shape
            wind_speed: Wind speed in m/s
            wind_direction: Wind direction in degrees (0-360)
            terrain_slope: Terrain slope array
            terrain_aspect: Terrain aspect array
            
        Returns:
            Wind effect multiplier raster
        """
        height, width = shape
        
        # Convert wind direction to radians
        wind_dir_rad = np.radians(wind_direction)
        
        # Calculate wind alignment with terrain
        aspect_rad = np.radians(terrain_aspect)
        alignment = np.cos(aspect_rad - wind_dir_rad)
        
        # Wind effect increases with speed and slope alignment
        base_effect = 1.0 + (wind_speed / 10.0)  # Normalize wind speed
        slope_effect = 1.0 + (terrain_slope / 90.0) * 0.5  # Slope contributes to wind effect
        alignment_effect = 1.0 + alignment * 0.3  # Wind alignment effect
        
        wind_effect = base_effect * slope_effect * alignment_effect
        
        return wind_effect.astype(np.float32)
    
    @staticmethod
    def save_raster(
        data: np.ndarray,
        transform: rasterio.Affine,
        crs: str,
        output_path: str,
        dtype: str = "float32",
        nodata: Optional[float] = None
    ):
        """
        Save raster data to file.
        
        Args:
            data: Data array to save
            transform: Raster transform
            crs: Coordinate reference system
            output_path: Output file path
            dtype: Data type
            nodata: NoData value
        """
        profile = {
            'driver': 'GTiff',
            'height': data.shape[0],
            'width': data.shape[1],
            'count': 1,
            'dtype': dtype,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw'
        }
        
        if nodata is not None:
            profile['nodata'] = nodata
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)
    
    @staticmethod
    def load_raster(file_path: str) -> Tuple[np.ndarray, rasterio.Affine, str]:
        """
        Load raster from file.
        
        Args:
            file_path: Path to raster file
            
        Returns:
            Tuple of (data, transform, crs)
        """
        with rasterio.open(file_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs.to_string()
        
        return data, transform, crs
    
    @staticmethod
    def create_uttarakhand_boundary() -> Polygon:
        """
        Create a polygon representing Uttarakhand boundary.
        
        Returns:
            Uttarakhand boundary polygon
        """
        # Simplified Uttarakhand boundary
        from utils.config import config
        bbox = config.UTTARAKHAND_BBOX
        
        return box(
            bbox["min_lon"],
            bbox["min_lat"],
            bbox["max_lon"],
            bbox["max_lat"]
        ) 