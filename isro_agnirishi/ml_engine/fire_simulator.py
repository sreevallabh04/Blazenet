"""
ISRO AGNIRISHI - Fire Spread Simulation Engine
Cellular Automata Model for Fire Spread Simulation

Implements:
- Cellular Automata fire spread simulation
- Wind-driven fire propagation
- Terrain influence on fire behavior
- Animation generation for 1,2,3,6,12 hours
- 30m resolution output rasters
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import imageio
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
from scipy.ndimage import binary_dilation, gaussian_filter
from datetime import datetime, timedelta

class CellularAutomataSimulator:
    """
    Cellular Automata Fire Spread Simulator.
    
    Implements fire spread simulation as specified in ISRO problem statement:
    - Uses wind speed/direction, slope, and fuel data
    - Generates animations for 1,2,3,6,12 hours
    - Outputs 30m resolution rasters
    """
    
    def __init__(self):
        """Initialize the fire spread simulator."""
        print("🔥 Initializing Cellular Automata Fire Simulator...")
        
        # Simulation parameters
        self.cell_size_m = 30  # 30m resolution as per problem statement
        self.time_step_minutes = 1  # 1-minute time steps
        
        # Fire spread parameters
        self.base_spread_rate = 0.5  # Base spread rate (m/min)
        self.max_spread_rate = 5.0   # Maximum spread rate (m/min)
        
        # Fuel model parameters
        self.fuel_models = {
            1: {"name": "Short Grass", "spread_rate": 1.2, "intensity": 0.3},
            2: {"name": "Timber Grass", "spread_rate": 0.8, "intensity": 0.4},
            3: {"name": "Tall Grass", "spread_rate": 1.5, "intensity": 0.5},
            4: {"name": "Chaparral", "spread_rate": 0.6, "intensity": 0.7},
            5: {"name": "Brush", "spread_rate": 0.4, "intensity": 0.6},
            6: {"name": "Dormant Brush", "spread_rate": 0.3, "intensity": 0.5},
            7: {"name": "Southern Rough", "spread_rate": 0.7, "intensity": 0.6},
            8: {"name": "Closed Timber", "spread_rate": 0.2, "intensity": 0.8},
            9: {"name": "Hardwood Litter", "spread_rate": 0.3, "intensity": 0.4},
            10: {"name": "Timber", "spread_rate": 0.4, "intensity": 0.9}
        }
        
        # Terrain effect parameters
        self.slope_factor = 2.0  # Multiplier for upslope spread
        self.aspect_influence = 0.3  # Aspect influence on spread
        
        print("✅ Cellular Automata Simulator initialized")
    
    def identify_ignition_points(self, fire_probability_map: np.ndarray, 
                               min_probability: float = 0.7,
                               max_points: int = 10) -> List[Tuple[int, int]]:
        """
        Identify high-risk zones as ignition points from probability map.
        
        Args:
            fire_probability_map: Fire probability map from prediction
            min_probability: Minimum probability threshold
            max_points: Maximum number of ignition points
        """
        print(f"🎯 Identifying ignition points (threshold: {min_probability})...")
        
        # Find high probability areas
        high_risk_mask = fire_probability_map > min_probability
        
        if not high_risk_mask.any():
            print("⚠️ No high-risk areas found, using maximum probability areas")
            # Use top N probability areas
            flat_probs = fire_probability_map.flatten()
            top_indices = np.argpartition(flat_probs, -max_points)[-max_points:]
            y_coords, x_coords = np.unravel_index(top_indices, fire_probability_map.shape)
            ignition_points = list(zip(x_coords, y_coords))
        else:
            # Find connected components and centroids
            from scipy.ndimage import label, center_of_mass
            
            labeled_array, num_features = label(high_risk_mask)
            
            ignition_points = []
            for i in range(1, min(num_features + 1, max_points + 1)):
                center = center_of_mass(labeled_array == i)
                y, x = int(center[0]), int(center[1])
                ignition_points.append((x, y))
        
        print(f"✅ Found {len(ignition_points)} ignition points")
        return ignition_points
    
    def simulate_fire_spread(self, ignition_points: List[Tuple[int, int]], 
                           duration_hours: int, weather_conditions: Dict,
                           resolution_m: int = 30) -> Dict:
        """
        Simulate fire spread using cellular automata for specified duration.
        
        Args:
            ignition_points: Initial fire locations (x, y)
            duration_hours: Simulation duration (1,2,3,6,12 hours)
            weather_conditions: Current weather (wind speed, direction, etc.)
            resolution_m: Grid resolution in meters
        
        Returns:
            Dict with simulation results and metadata
        """
        print(f"🔥 Starting {duration_hours}h fire spread simulation...")
        print(f"🌬️ Wind: {weather_conditions.get('wind_speed', 5):.1f} m/s @ {weather_conditions.get('wind_direction', 180):.0f}°")
        
        # Initialize simulation grid
        grid_size = (800, 600)  # Standard grid for Uttarakhand region
        height, width = grid_size
        
        # Create fire state grid (0: unburned, 1: burning, 2: burned out)
        fire_state = np.zeros((height, width), dtype=np.uint8)
        
        # Create fuel load grid (simulated)
        fuel_load = self._generate_fuel_map(height, width)
        
        # Create terrain grids
        slope_grid, aspect_grid = self._generate_terrain_grids(height, width)
        
        # Set initial ignition points
        for x, y in ignition_points:
            if 0 <= x < width and 0 <= y < height:
                fire_state[y, x] = 1  # Set as burning
        
        # Simulation parameters
        total_time_steps = duration_hours * 60 // self.time_step_minutes
        wind_speed = weather_conditions.get('wind_speed', 5.0)  # m/s
        wind_direction = np.radians(weather_conditions.get('wind_direction', 180))  # Convert to radians
        
        # Store simulation frames for animation
        simulation_frames = []
        burned_area_history = []
        
        print(f"⏱️ Running {total_time_steps} time steps...")
        
        # Main simulation loop
        for step in range(total_time_steps):
            # Store current state for animation
            if step % (total_time_steps // 20) == 0:  # Store 20 frames
                simulation_frames.append(fire_state.copy())
            
            # Calculate fire spread for this time step
            new_fire_state = self._update_fire_state(
                fire_state, fuel_load, slope_grid, aspect_grid,
                wind_speed, wind_direction
            )
            
            fire_state = new_fire_state
            
            # Calculate burned area
            burned_pixels = (fire_state >= 1).sum()
            burned_area_km2 = burned_pixels * (resolution_m ** 2) / 1e6
            burned_area_history.append(burned_area_km2)
            
            # Progress reporting
            if step % (total_time_steps // 10) == 0:
                progress = (step / total_time_steps) * 100
                print(f"Progress: {progress:.0f}% - Burned area: {burned_area_km2:.2f} km²")
        
        # Final frame
        simulation_frames.append(fire_state.copy())
        
        # Calculate maximum spread rate
        if len(burned_area_history) > 1:
            area_diff = np.diff(burned_area_history)
            max_spread_rate_mh = np.max(area_diff) * 60 / self.time_step_minutes  # km²/h to m/h approximation
            max_spread_rate_mh *= 1000  # Convert to m/h
        else:
            max_spread_rate_mh = 0
        
        # Prepare results
        results = {
            "final_state": fire_state,
            "simulation_frames": simulation_frames,
            "burned_area_km2": burned_area_history[-1] if burned_area_history else 0,
            "burned_area_history": burned_area_history,
            "max_spread_rate_mh": max_spread_rate_mh,
            "total_time_steps": total_time_steps,
            "grid_size": grid_size,
            "resolution_m": resolution_m,
            "ignition_points": ignition_points,
            "weather_conditions": weather_conditions,
            "simulation_metadata": {
                "duration_hours": duration_hours,
                "time_step_minutes": self.time_step_minutes,
                "total_ignition_points": len(ignition_points),
                "final_burned_pixels": (fire_state >= 1).sum(),
                "simulation_completed": True
            }
        }
        
        print(f"✅ Simulation complete - Final burned area: {results['burned_area_km2']:.2f} km²")
        return results
    
    def _update_fire_state(self, fire_state: np.ndarray, fuel_load: np.ndarray,
                          slope_grid: np.ndarray, aspect_grid: np.ndarray,
                          wind_speed: float, wind_direction: float) -> np.ndarray:
        """Update fire state for one time step using cellular automata rules."""
        
        height, width = fire_state.shape
        new_state = fire_state.copy()
        
        # Find currently burning cells
        burning_cells = np.where(fire_state == 1)
        
        for y, x in zip(burning_cells[0], burning_cells[1]):
            # Check all 8 neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    ny, nx = y + dy, x + dx
                    
                    # Check bounds
                    if 0 <= ny < height and 0 <= nx < width:
                        # Only spread to unburned cells
                        if fire_state[ny, nx] == 0:
                            # Calculate spread probability
                            spread_prob = self._calculate_spread_probability(
                                x, y, nx, ny, fuel_load, slope_grid, aspect_grid,
                                wind_speed, wind_direction
                            )
                            
                            # Probabilistic spread
                            if np.random.random() < spread_prob:
                                new_state[ny, nx] = 1
        
        # Age burning cells (burning cells eventually burn out)
        # For simplicity, we keep them burning throughout simulation
        # In a more complex model, we would transition burning -> burned out
        
        return new_state
    
    def _calculate_spread_probability(self, from_x: int, from_y: int, to_x: int, to_y: int,
                                    fuel_load: np.ndarray, slope_grid: np.ndarray, 
                                    aspect_grid: np.ndarray, wind_speed: float, 
                                    wind_direction: float) -> float:
        """Calculate probability of fire spreading from one cell to another."""
        
        # Base spread probability from fuel
        fuel_value = fuel_load[to_y, to_x]
        base_prob = fuel_value * 0.1  # Base 10% chance per minute for high fuel
        
        # Wind effect
        # Calculate direction from source to target
        dx, dy = to_x - from_x, to_y - from_y
        spread_direction = np.arctan2(dy, dx)
        
        # Wind alignment factor (higher when wind blows in spread direction)
        wind_alignment = np.cos(spread_direction - wind_direction)
        wind_factor = 1.0 + (wind_speed / 10.0) * wind_alignment * 0.5
        
        # Slope effect (fire spreads faster uphill)
        slope = slope_grid[to_y, to_x]
        if dy < 0:  # Spreading uphill (assuming north is up)
            slope_factor = 1.0 + slope / 45.0  # Max 2x for 45° slope
        else:  # Spreading downhill
            slope_factor = 1.0 - slope / 90.0 * 0.3  # Reduce by up to 30%
        
        slope_factor = max(0.1, slope_factor)  # Minimum factor
        
        # Combine factors
        spread_prob = base_prob * wind_factor * slope_factor
        
        # Ensure probability is between 0 and 1
        spread_prob = np.clip(spread_prob, 0, 1)
        
        return spread_prob
    
    def _generate_fuel_map(self, height: int, width: int) -> np.ndarray:
        """Generate realistic fuel load map."""
        
        # Create base fuel pattern
        np.random.seed(42)  # Reproducible fuel patterns
        
        # Create realistic spatial fuel distribution
        fuel_map = np.random.beta(2, 3, (height, width))  # Skewed towards lower values
        
        # Add high-fuel forest areas
        num_forests = np.random.randint(5, 10)
        for _ in range(num_forests):
            center_y = np.random.randint(height // 4, 3 * height // 4)
            center_x = np.random.randint(width // 4, 3 * width // 4)
            radius = np.random.randint(20, 50)
            
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            fuel_map[mask] = np.random.uniform(0.7, 1.0, mask.sum())
        
        # Add low-fuel areas (water bodies, urban areas)
        num_low_fuel = np.random.randint(3, 7)
        for _ in range(num_low_fuel):
            center_y = np.random.randint(height // 6, 5 * height // 6)
            center_x = np.random.randint(width // 6, 5 * width // 6)
            radius = np.random.randint(10, 30)
            
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            fuel_map[mask] = np.random.uniform(0.0, 0.2, mask.sum())
        
        return fuel_map
    
    def _generate_terrain_grids(self, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate slope and aspect grids."""
        
        # Create elevation grid with realistic patterns
        y, x = np.ogrid[:height, :width]
        
        # North-south elevation gradient (higher in north for Uttarakhand)
        elevation = 500 + 2000 * (height - y) / height
        
        # Add random hills and valleys
        for _ in range(15):
            hill_y = np.random.randint(0, height)
            hill_x = np.random.randint(0, width)
            hill_height = np.random.uniform(200, 800)
            hill_radius = np.random.uniform(30, 80)
            
            dist = np.sqrt((x - hill_x)**2 + (y - hill_y)**2)
            hill_mask = dist < hill_radius
            elevation[hill_mask] += hill_height * np.exp(-(dist[hill_mask] / hill_radius)**2)
        
        # Calculate slope (in degrees)
        gy, gx = np.gradient(elevation)
        slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2) / 30))  # 30m pixel size
        slope = np.clip(slope, 0, 60)  # Max 60 degrees
        
        # Calculate aspect (in degrees)
        aspect = np.degrees(np.arctan2(-gx, gy)) % 360
        
        return slope, aspect
    
    def create_spread_animation(self, simulation_data: Dict, output_path: str, 
                              duration_hours: int) -> str:
        """
        Create fire spread animation from simulation data.
        
        As per problem statement: Generate animations for 1,2,3,6,12 hours.
        """
        print(f"🎬 Creating {duration_hours}h fire spread animation...")
        
        frames = simulation_data["simulation_frames"]
        
        # Create colormap for fire states
        colors = ['green', 'red', 'black']  # unburned, burning, burned
        cmap = ListedColormap(colors)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title(f'ISRO AGNIRISHI - Fire Spread Simulation ({duration_hours}h)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Distance (30m pixels)')
        ax.set_ylabel('Distance (30m pixels)')
        
        # Animation function
        def animate(frame_idx):
            ax.clear()
            ax.set_title(f'Fire Spread - Hour {frame_idx * duration_hours / len(frames):.1f}',
                        fontsize=14)
            
            frame_data = frames[frame_idx]
            im = ax.imshow(frame_data, cmap=cmap, vmin=0, vmax=2, 
                          origin='upper', aspect='equal')
            
            # Add colorbar
            if frame_idx == 0:
                cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
                cbar.set_ticklabels(['Unburned', 'Burning', 'Burned'])
            
            # Add statistics
            burning_pixels = (frame_data == 1).sum()
            burned_pixels = (frame_data >= 1).sum()
            total_pixels = frame_data.size
            
            burned_area_km2 = burned_pixels * (30**2) / 1e6
            
            ax.text(0.02, 0.98, f'Burned Area: {burned_area_km2:.2f} km²', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            return [im]
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                     interval=200, blit=False, repeat=True)
        
        # Save as GIF
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(output_path, writer='pillow', fps=5)
        plt.close()
        
        print(f"✅ Animation saved: {output_path}")
        return output_path

if __name__ == "__main__":
    # Test the cellular automata simulator
    simulator = CellularAutomataSimulator()
    
    # Mock fire probability map
    prob_map = np.random.beta(1, 4, (600, 800))  # Low probability base
    
    # Add some high probability hotspots
    prob_map[200:250, 300:350] = 0.8
    prob_map[400:430, 600:650] = 0.9
    
    # Find ignition points
    ignition_points = simulator.identify_ignition_points(prob_map)
    print(f"Ignition points: {ignition_points}")
    
    # Mock weather conditions
    weather = {
        "wind_speed": 8.0,  # m/s
        "wind_direction": 225,  # degrees
        "temperature": 32,  # °C
        "humidity": 25  # %
    }
    
    # Run simulation
    results = simulator.simulate_fire_spread(
        ignition_points=ignition_points,
        duration_hours=2,
        weather_conditions=weather
    )
    
    print(f"Simulation results: {results['simulation_metadata']}")
    
    # Create animation
    animation_path = "test_fire_spread_2h.gif"
    simulator.create_spread_animation(results, animation_path, 2)
    print(f"Animation created: {animation_path}") 