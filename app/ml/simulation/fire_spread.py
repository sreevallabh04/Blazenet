"""
Fire spread simulation using cellular automata.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class FireSimulationParams:
    """Parameters for fire spread simulation."""
    wind_speed: float = 5.0  # m/s
    wind_direction: float = 180.0  # degrees
    temperature: float = 30.0  # Celsius
    humidity: float = 40.0  # percentage
    fuel_moisture: float = 0.1  # fraction
    time_step: float = 1.0  # hours
    max_steps: int = 24  # maximum simulation steps

class CellularAutomataFireModel:
    """Cellular automata model for fire spread simulation."""
    
    def __init__(self, grid_size: Tuple[int, int], cell_size: float = 30.0):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.height, self.width = grid_size
        
        # Fire states: 0=unburned, 1=burning, 2=burned
        self.fire_state = np.zeros(grid_size, dtype=np.int32)
        self.burn_time = np.zeros(grid_size, dtype=np.float32)
        
        # Environmental factors
        self.fuel_load = np.ones(grid_size, dtype=np.float32)
        self.fuel_type = np.ones(grid_size, dtype=np.int32)
        self.terrain_slope = np.zeros(grid_size, dtype=np.float32)
        
        # Fire behavior parameters
        self.fuel_burn_rate = {
            1: 2.0,  # Grass
            2: 4.0,  # Shrub
            3: 6.0,  # Forest
            4: 0.1,  # Urban
            5: 0.0   # Water/rock
        }
        
        # Wind effect directions
        self.neighbor_directions = np.array([
            [-1, 0],  # North
            [-1, 1],  # Northeast
            [0, 1],   # East
            [1, 1],   # Southeast
            [1, 0],   # South
            [1, -1],  # Southwest
            [0, -1],  # West
            [-1, -1]  # Northwest
        ])
    
    def set_environmental_data(self, fuel_load, fuel_type, terrain_slope):
        """Set environmental data for simulation."""
        self.fuel_load = fuel_load.astype(np.float32)
        self.fuel_type = fuel_type.astype(np.int32)
        self.terrain_slope = terrain_slope.astype(np.float32)
    
    def set_ignition_points(self, ignition_points: List[Tuple[int, int]]):
        """Set initial fire ignition points."""
        for row, col in ignition_points:
            if 0 <= row < self.height and 0 <= col < self.width:
                self.fire_state[row, col] = 1
                self.burn_time[row, col] = 0.0
    
    def calculate_spread_probability(self, source_row, source_col, target_row, target_col, params):
        """Calculate probability of fire spreading from source to target cell."""
        # Check bounds
        if not (0 <= target_row < self.height and 0 <= target_col < self.width):
            return 0.0
        
        # Can't spread to already burned/burning cells
        if self.fire_state[target_row, target_col] != 0:
            return 0.0
        
        # Base spread probability based on fuel type
        fuel_type = self.fuel_type[target_row, target_col]
        if fuel_type == 5:  # Non-flammable
            return 0.0
        
        base_prob = {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.1}.get(fuel_type, 0.0)
        
        # Wind effect
        wind_effect = self._calculate_wind_effect(source_row, source_col, target_row, target_col, params)
        
        # Environmental effects
        moisture_effect = max(0.1, 1.0 - params.fuel_moisture * 2.0)
        temp_effect = min(2.0, params.temperature / 25.0)
        humidity_effect = max(0.3, 1.0 - params.humidity / 100.0)
        fuel_effect = min(2.0, self.fuel_load[target_row, target_col])
        
        # Combine effects
        total_prob = (base_prob * wind_effect * moisture_effect * 
                     temp_effect * humidity_effect * fuel_effect)
        
        return min(1.0, total_prob)
    
    def _calculate_wind_effect(self, source_row, source_col, target_row, target_col, params):
        """Calculate wind effect on fire spread."""
        dy = target_row - source_row
        dx = target_col - source_col
        
        if dx == 0 and dy == 0:
            return 1.0
        
        # Calculate spread direction
        spread_angle = np.degrees(np.arctan2(dx, -dy))
        if spread_angle < 0:
            spread_angle += 360
        
        # Calculate alignment with wind
        angle_diff = abs(spread_angle - params.wind_direction)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        wind_strength = min(3.0, 1.0 + params.wind_speed / 10.0)
        alignment = np.cos(np.radians(angle_diff))
        wind_effect = 1.0 + (wind_strength - 1.0) * alignment * 0.5
        
        return max(0.1, wind_effect)
    
    def step(self, params: FireSimulationParams) -> bool:
        """Execute one simulation time step."""
        new_fire_state = self.fire_state.copy()
        new_burn_time = self.burn_time.copy()
        
        burning_cells = np.where(self.fire_state == 1)
        
        if len(burning_cells[0]) == 0:
            return False
        
        for i in range(len(burning_cells[0])):
            row, col = burning_cells[0][i], burning_cells[1][i]
            
            # Update burn time
            new_burn_time[row, col] += params.time_step
            
            # Check if finished burning
            fuel_type = self.fuel_type[row, col]
            burn_duration = self.fuel_burn_rate.get(fuel_type, 2.0)
            
            if new_burn_time[row, col] >= burn_duration:
                new_fire_state[row, col] = 2
                continue
            
            # Spread to neighbors
            for direction in self.neighbor_directions:
                neighbor_row = row + direction[0]
                neighbor_col = col + direction[1]
                
                spread_prob = self.calculate_spread_probability(
                    row, col, neighbor_row, neighbor_col, params
                )
                
                if np.random.random() < spread_prob * params.time_step:
                    if (0 <= neighbor_row < self.height and 
                        0 <= neighbor_col < self.width and
                        new_fire_state[neighbor_row, neighbor_col] == 0):
                        
                        new_fire_state[neighbor_row, neighbor_col] = 1
                        new_burn_time[neighbor_row, neighbor_col] = 0.0
        
        self.fire_state = new_fire_state
        self.burn_time = new_burn_time
        
        return np.any(self.fire_state == 1)
    
    def simulate(self, ignition_points, params):
        """Run complete fire spread simulation."""
        # Reset simulation
        self.fire_state = np.zeros(self.grid_size, dtype=np.int32)
        self.burn_time = np.zeros(self.grid_size, dtype=np.float32)
        
        # Set ignition points
        self.set_ignition_points(ignition_points)
        
        # Store history
        history = [self.fire_state.copy()]
        
        # Run simulation
        for step in range(params.max_steps):
            active = self.step(params)
            history.append(self.fire_state.copy())
            
            if not active:
                break
        
        return history

class FireSpreadSimulator:
    """High-level fire spread simulator."""
    
    def __init__(self, grid_size: Tuple[int, int] = (512, 512)):
        self.grid_size = grid_size
        self.ca_model = CellularAutomataFireModel(grid_size)
    
    def run_simulation(self, environmental_data, ignition_points, weather_params, max_hours=24):
        """Run fire spread simulation."""
        # Set environmental data
        self.ca_model.set_environmental_data(**environmental_data)
        
        # Create parameters
        params = FireSimulationParams(
            wind_speed=weather_params.get('wind_speed', 5.0),
            wind_direction=weather_params.get('wind_direction', 180.0),
            temperature=weather_params.get('temperature', 30.0),
            humidity=weather_params.get('humidity', 40.0),
            fuel_moisture=weather_params.get('fuel_moisture', 0.1),
            time_step=1.0,
            max_steps=max_hours
        )
        
        # Run simulation
        history = self.ca_model.simulate(ignition_points, params)
        
        # Calculate statistics
        total_burned = np.sum(history[-1] >= 1)
        burned_area_ha = total_burned * (30 * 30) / 10000
        
        return {
            'fire_history': history,
            'final_state': history[-1],
            'total_burned_cells': total_burned,
            'burned_area_hectares': burned_area_ha,
            'simulation_steps': len(history) - 1,
            'parameters': params
        } 