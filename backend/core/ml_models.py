"""
ISRO AGNIRISHI - Production ML Models
Complete U-NET and LSTM implementation for fire prediction

This module contains the actual working ML models that power
the revolutionary fire prediction system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
import logging
from pathlib import Path
import cv2
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UNetFirePredictor(nn.Module):
    """Production U-NET model for spatial fire prediction."""
    
    def __init__(self, in_channels: int = 9, out_channels: int = 1):
        super(UNetFirePredictor, self).__init__()
        
        # Encoder path
        self.enc1 = self._double_conv(in_channels, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.enc4 = self._double_conv(256, 512)
        
        # Center
        self.center = self._double_conv(512, 1024)
        
        # Decoder path
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, 1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        logger.info("U-NET Fire Predictor initialized with {} input channels".format(in_channels))
    
    def _double_conv(self, in_ch: int, out_ch: int) -> nn.Module:
        """Double convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Center
        center = self.center(self.pool(enc4))
        
        # Decoder with skip connections
        up4 = self.up4(center)
        merge4 = torch.cat([up4, enc4], dim=1)
        dec4 = self.dec4(merge4)
        
        up3 = self.up3(dec4)
        merge3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(merge3)
        
        up2 = self.up2(dec3)
        merge2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(merge2)
        
        up1 = self.up1(dec2)
        merge1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(merge1)
        
        return torch.sigmoid(self.final(dec1))

class LSTMFirePredictor(nn.Module):
    """Production LSTM model for temporal fire prediction."""
    
    def __init__(self, input_size: int = 9, hidden_size: int = 128, 
                 num_layers: int = 3, output_size: int = 1, dropout: float = 0.2):
        super(LSTMFirePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_size * 2, num_heads=8, batch_first=True
        )
        
        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
        
        logger.info("LSTM Fire Predictor initialized with {} input features".format(input_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last time step
        last_output = attn_out[:, -1, :]
        
        # Final prediction
        return self.fc_layers(last_output)

class CellularAutomataFireSim:
    """Production Cellular Automata fire spread simulation."""
    
    def __init__(self, cell_size_m: float = 30.0):
        self.cell_size_m = cell_size_m
        self.time_step_minutes = 1.0
        
        # Fire spread parameters based on scientific research
        self.fuel_models = {
            1: {"name": "Grass", "spread_rate": 1.2, "intensity": 0.3},
            2: {"name": "Shrub", "spread_rate": 0.8, "intensity": 0.6},
            3: {"name": "Timber", "spread_rate": 0.4, "intensity": 0.9},
            4: {"name": "Slash", "spread_rate": 1.8, "intensity": 0.7}
        }
        
        logger.info("Cellular Automata Fire Simulator initialized")
    
    def simulate_spread(self, ignition_points: List[Tuple[int, int]], 
                       weather: Dict, terrain: Dict, fuel: np.ndarray,
                       duration_hours: int) -> Dict:
        """Run complete fire spread simulation."""
        
        height, width = fuel.shape
        fire_state = np.zeros((height, width), dtype=np.uint8)
        
        # Set ignition points
        for x, y in ignition_points:
            if 0 <= x < width and 0 <= y < height:
                fire_state[y, x] = 1
        
        # Simulation parameters
        total_steps = int(duration_hours * 60 / self.time_step_minutes)
        wind_speed = weather.get('wind_speed', 5.0)
        wind_direction = np.radians(weather.get('wind_direction', 180))
        
        # Simulation history
        history = []
        burned_area_history = []
        
        logger.info(f"Starting {duration_hours}h simulation with {len(ignition_points)} ignition points")
        
        for step in range(total_steps):
            new_fire_state = self._update_fire_state(
                fire_state, fuel, terrain, wind_speed, wind_direction
            )
            
            fire_state = new_fire_state
            
            # Record state every 10 steps for animation
            if step % 10 == 0:
                history.append(fire_state.copy())
            
            # Calculate burned area
            burned_pixels = (fire_state > 0).sum()
            burned_area_km2 = burned_pixels * (self.cell_size_m / 1000) ** 2
            burned_area_history.append(burned_area_km2)
        
        # Calculate final metrics
        final_burned_area = burned_area_history[-1] if burned_area_history else 0
        max_spread_rate = self._calculate_max_spread_rate(burned_area_history)
        
        logger.info(f"Simulation complete: {final_burned_area:.2f} km² burned")
        
        return {
            "final_state": fire_state,
            "history": history,
            "burned_area_km2": final_burned_area,
            "burned_area_history": burned_area_history,
            "max_spread_rate_mh": max_spread_rate,
            "simulation_metadata": {
                "duration_hours": duration_hours,
                "total_steps": total_steps,
                "ignition_points": len(ignition_points),
                "wind_speed": wind_speed,
                "wind_direction": np.degrees(wind_direction)
            }
        }
    
    def _update_fire_state(self, fire_state: np.ndarray, fuel: np.ndarray,
                          terrain: Dict, wind_speed: float, wind_direction: float) -> np.ndarray:
        """Update fire state for one time step."""
        
        height, width = fire_state.shape
        new_state = fire_state.copy()
        
        # Get terrain data
        slope = terrain.get('slope', np.zeros((height, width)))
        aspect = terrain.get('aspect', np.zeros((height, width)))
        
        # Find burning cells
        burning_cells = np.where(fire_state == 1)
        
        for y, x in zip(burning_cells[0], burning_cells[1]):
            # Check 8 neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    ny, nx = y + dy, x + dx
                    
                    if (0 <= ny < height and 0 <= nx < width and 
                        fire_state[ny, nx] == 0):
                        
                        # Calculate spread probability
                        spread_prob = self._calculate_spread_probability(
                            x, y, nx, ny, fuel, slope, aspect,
                            wind_speed, wind_direction
                        )
                        
                        if np.random.random() < spread_prob:
                            new_state[ny, nx] = 1
        
        return new_state
    
    def _calculate_spread_probability(self, from_x: int, from_y: int, 
                                    to_x: int, to_y: int, fuel: np.ndarray,
                                    slope: np.ndarray, aspect: np.ndarray,
                                    wind_speed: float, wind_direction: float) -> float:
        """Calculate fire spread probability based on multiple factors."""
        
        # Base probability from fuel
        fuel_value = fuel[to_y, to_x]
        base_prob = fuel_value * 0.1  # Base 10% per minute for max fuel
        
        # Wind effect
        dx, dy = to_x - from_x, to_y - from_y
        spread_angle = np.arctan2(dy, dx)
        wind_alignment = np.cos(spread_angle - wind_direction)
        wind_factor = 1.0 + (wind_speed / 15.0) * max(0, wind_alignment) * 0.5
        
        # Slope effect
        slope_value = slope[to_y, to_x]
        if dy < 0:  # Uphill spread
            slope_factor = 1.0 + slope_value / 45.0  # Max 2x for 45° slope
        else:  # Downhill spread
            slope_factor = max(0.5, 1.0 - slope_value / 90.0 * 0.3)
        
        # Combine factors
        total_prob = base_prob * wind_factor * slope_factor
        
        return np.clip(total_prob, 0, 1)
    
    def _calculate_max_spread_rate(self, area_history: List[float]) -> float:
        """Calculate maximum fire spread rate in m/h."""
        if len(area_history) < 2:
            return 0.0
        
        # Calculate rate of change in area
        area_changes = np.diff(area_history)
        max_change = np.max(area_changes) if len(area_changes) > 0 else 0
        
        # Convert to approximate linear spread rate
        # Assuming circular spread: A = πr², so r = √(A/π)
        # Rate = Δr/Δt
        if max_change > 0:
            radius_change = np.sqrt(max_change / np.pi) * 1000  # Convert km to m
            rate_mh = radius_change * (60 / self.time_step_minutes)  # Convert to m/h
            return rate_mh
        
        return 0.0

class ProductionMLPipeline:
    """Complete ML pipeline for production fire prediction."""
    
    def __init__(self, model_dir: str = "models/production"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.unet_model = UNetFirePredictor()
        self.lstm_model = LSTMFirePredictor()
        self.fire_sim = CellularAutomataFireSim()
        
        # Move to device
        self.unet_model.to(self.device)
        self.lstm_model.to(self.device)
        
        # Load pre-trained weights if available
        self._load_models()
        
        logger.info(f"Production ML Pipeline initialized on {self.device}")
    
    def _load_models(self):
        """Load pre-trained model weights."""
        unet_path = self.model_dir / "unet_fire_model.pth"
        lstm_path = self.model_dir / "lstm_fire_model.pth"
        
        if unet_path.exists():
            self.unet_model.load_state_dict(torch.load(unet_path, map_location=self.device))
            logger.info("Loaded pre-trained U-NET model")
        else:
            logger.warning("No pre-trained U-NET model found, using random weights")
        
        if lstm_path.exists():
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
            logger.info("Loaded pre-trained LSTM model")
        else:
            logger.warning("No pre-trained LSTM model found, using random weights")
    
    def predict_fire_probability(self, feature_data: np.ndarray, 
                                temporal_data: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict fire probability using ensemble of models."""
        
        self.unet_model.eval()
        self.lstm_model.eval()
        
        with torch.no_grad():
            # U-NET spatial prediction
            if feature_data.ndim == 3:
                feature_data = feature_data[np.newaxis, ...]
            
            features_tensor = torch.FloatTensor(feature_data).to(self.device)
            unet_pred = self.unet_model(features_tensor).cpu().numpy().squeeze()
            
            # LSTM temporal prediction (if temporal data available)
            if temporal_data is not None:
                if temporal_data.ndim == 2:
                    temporal_data = temporal_data[np.newaxis, ...]
                
                temporal_tensor = torch.FloatTensor(temporal_data).to(self.device)
                lstm_pred = self.lstm_model(temporal_tensor).cpu().numpy().squeeze()
                
                # Ensemble prediction (weighted average)
                final_pred = 0.7 * unet_pred + 0.3 * lstm_pred
            else:
                final_pred = unet_pred
        
        logger.info(f"Fire probability prediction complete - max probability: {final_pred.max():.3f}")
        return final_pred
    
    def simulate_fire_spread(self, fire_probability: np.ndarray, 
                           weather_data: Dict, terrain_data: Dict,
                           simulation_hours: List[int] = [1, 2, 3, 6, 12]) -> Dict:
        """Run complete fire spread simulation for multiple time periods."""
        
        # Identify ignition points from high probability areas
        ignition_points = self._identify_ignition_points(fire_probability)
        
        # Create fuel map from probability
        fuel_map = np.clip(fire_probability * 1.5, 0, 1)
        
        results = {}
        
        for hours in simulation_hours:
            logger.info(f"Running {hours}-hour fire spread simulation")
            
            sim_result = self.fire_sim.simulate_spread(
                ignition_points=ignition_points,
                weather=weather_data,
                terrain=terrain_data,
                fuel=fuel_map,
                duration_hours=hours
            )
            
            results[f"{hours}h"] = sim_result
        
        return results
    
    def _identify_ignition_points(self, fire_probability: np.ndarray, 
                                 threshold: float = 0.7, max_points: int = 20) -> List[Tuple[int, int]]:
        """Identify potential ignition points from fire probability map."""
        
        high_prob_mask = fire_probability > threshold
        y_coords, x_coords = np.where(high_prob_mask)
        
        if len(y_coords) == 0:
            # If no high probability areas, use top probability points
            flat_indices = np.argpartition(fire_probability.flatten(), -max_points)[-max_points:]
            y_coords, x_coords = np.unravel_index(flat_indices, fire_probability.shape)
        
        # Limit number of ignition points
        if len(y_coords) > max_points:
            indices = np.random.choice(len(y_coords), max_points, replace=False)
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]
        
        ignition_points = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
        
        logger.info(f"Identified {len(ignition_points)} ignition points")
        return ignition_points
    
    def train_models(self, training_data: Dict, epochs: int = 100) -> Dict:
        """Train the ML models on historical data."""
        
        logger.info("Starting model training...")
        
        # Extract training data
        X_spatial = training_data['spatial_features']
        y_spatial = training_data['spatial_targets']
        X_temporal = training_data.get('temporal_features')
        y_temporal = training_data.get('temporal_targets')
        
        # Training results
        results = {"unet_losses": [], "lstm_losses": [], "training_time": 0}
        
        start_time = datetime.now()
        
        # Train U-NET
        if X_spatial is not None and y_spatial is not None:
            unet_results = self._train_unet(X_spatial, y_spatial, epochs)
            results["unet_losses"] = unet_results["losses"]
        
        # Train LSTM
        if X_temporal is not None and y_temporal is not None:
            lstm_results = self._train_lstm(X_temporal, y_temporal, epochs)
            results["lstm_losses"] = lstm_results["losses"]
        
        # Save trained models
        self._save_models()
        
        end_time = datetime.now()
        results["training_time"] = (end_time - start_time).total_seconds()
        
        logger.info(f"Model training completed in {results['training_time']:.2f} seconds")
        return results
    
    def _train_unet(self, X: np.ndarray, y: np.ndarray, epochs: int) -> Dict:
        """Train U-NET model."""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.unet_model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        self.unet_model.train()
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Mini-batch training
            batch_size = 4
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.unet_model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / max(1, len(X_tensor) // batch_size)
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"U-NET Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return {"losses": losses}
    
    def _train_lstm(self, X: np.ndarray, y: np.ndarray, epochs: int) -> Dict:
        """Train LSTM model."""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        self.lstm_model.train()
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Mini-batch training
            batch_size = 16
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / max(1, len(X_tensor) // batch_size)
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return {"losses": losses}
    
    def _save_models(self):
        """Save trained models."""
        torch.save(self.unet_model.state_dict(), self.model_dir / "unet_fire_model.pth")
        torch.save(self.lstm_model.state_dict(), self.model_dir / "lstm_fire_model.pth")
        logger.info("Models saved successfully")
    
    def get_model_info(self) -> Dict:
        """Get information about the models."""
        return {
            "unet_parameters": sum(p.numel() for p in self.unet_model.parameters()),
            "lstm_parameters": sum(p.numel() for p in self.lstm_model.parameters()),
            "device": str(self.device),
            "model_dir": str(self.model_dir),
            "unet_model_size_mb": sum(p.numel() * 4 for p in self.unet_model.parameters()) / (1024**2),
            "lstm_model_size_mb": sum(p.numel() * 4 for p in self.lstm_model.parameters()) / (1024**2)
        }

# Global ML pipeline instance
_ml_pipeline = None

def get_ml_pipeline() -> ProductionMLPipeline:
    """Get or create the global ML pipeline instance."""
    global _ml_pipeline
    if _ml_pipeline is None:
        _ml_pipeline = ProductionMLPipeline()
    return _ml_pipeline

if __name__ == "__main__":
    # Test the ML pipeline
    pipeline = get_ml_pipeline()
    logger.info("ML Pipeline test completed successfully")
    logger.info(f"Model info: {pipeline.get_model_info()}") 