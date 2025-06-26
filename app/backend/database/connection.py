"""
Database connection management for BlazeNet backend.
"""

import sys
from pathlib import Path
import asyncpg
import logging
from typing import Optional
from contextlib import asynccontextmanager

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from utils.config import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database connections."""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.connection_url = config.DATABASE_URL
    
    async def initialize(self):
        """Initialize database connection pool."""
        try:
            logger.info("Initializing database connection pool...")
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.connection_url,
                min_size=5,
                max_size=20,
                command_timeout=60,
                server_settings={
                    'jit': 'off'  # Disable JIT for better performance with small queries
                }
            )
            
            # Test connection
            async with self.pool.acquire() as connection:
                # Test basic connectivity
                result = await connection.fetchval('SELECT version()')
                logger.info(f"Connected to PostgreSQL: {result}")
                
                # Check PostGIS extension
                try:
                    postgis_version = await connection.fetchval('SELECT PostGIS_Version()')
                    logger.info(f"PostGIS version: {postgis_version}")
                except Exception as e:
                    logger.warning(f"PostGIS not available: {e}")
                
            logger.info("Database connection pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error(f"Database operation failed: {e}")
                raise
    
    async def health_check(self) -> bool:
        """Check database health."""
        try:
            if not self.pool:
                return False
            
            async with self.get_connection() as conn:
                await conn.fetchval('SELECT 1')
                return True
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def execute_query(self, query: str, *args):
        """Execute a query and return results."""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def execute_single(self, query: str, *args):
        """Execute a query and return single result."""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)
    
    async def execute_value(self, query: str, *args):
        """Execute a query and return single value."""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)
    
    async def execute_command(self, query: str, *args):
        """Execute a command (INSERT, UPDATE, DELETE)."""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)

# Global database manager instance
database_manager = DatabaseManager() 