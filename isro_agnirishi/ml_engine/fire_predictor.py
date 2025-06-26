"""
ISRO AGNIRISHI - Fire Prediction Engine
U-NET and LSTM Models for Forest Fire Prediction

Implements:
- U-NET model for spatial fire probability prediction
- LSTM model for temporal fire pattern analysis
- Binary classification (fire/no fire) at 30m resolution
- Model training and inference pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import joblib

class UNetFirePredictor(nn.Module):
    """
    U-NET architecture for spatial fire prediction.
    
    As specified in ISRO problem statement:
    - Input: Multi-band feature stack (weather, terrain, LULC)
    - Output: Binary fire probability map (30m resolution)
    """
    
    def __init__(self, in_channels: int = 9, num_classes: int = 1):
        super(UNetFirePredictor, self).__init__()
        
        # Encoder (Contracting Path)
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder (Expanding Path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Final classifier
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        print("🧠 U-NET Fire Predictor initialized")
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Convolutional block with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Final prediction
        output = torch.sigmoid(self.classifier(dec1))
        return output

class LSTMFirePredictor(nn.Module):
    """
    LSTM model for temporal fire pattern analysis.
    
    Analyzes time series of fire conditions and weather patterns.
    """
    
    def __init__(self, input_size: int = 9, hidden_size: int = 128, num_layers: int = 2, output_size: int = 1):
        super(LSTMFirePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
        
        print("⏰ LSTM Fire Predictor initialized")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last time step for prediction
        last_output = attn_out[:, -1, :]
        
        # Final prediction
        prediction = self.classifier(last_output)
        return prediction

class FirePredictionEngine:
    """
    Main fire prediction engine combining U-NET and LSTM models.
    
    Implements the complete ML pipeline for fire prediction as per
    ISRO problem statement requirements.
    """
    
    def __init__(self):
        """Initialize the prediction engine."""
        print("🔥 Initializing Fire Prediction Engine...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"💻 Using device: {self.device}")
        
        # Initialize models
        self.unet_model = UNetFirePredictor(in_channels=9, num_classes=1)
        self.lstm_model = LSTMFirePredictor(input_size=9, hidden_size=128)
        
        # Move models to device
        self.unet_model.to(self.device)
        self.lstm_model.to(self.device)
        
        # Model states
        self.models_trained = False
        self.training_history = {}
        
        # Load pre-trained weights if available
        self._load_pretrained_weights()
        
        print("✅ Fire Prediction Engine initialized")
    
    def _load_pretrained_weights(self):
        """Load pre-trained model weights if available."""
        unet_path = Path("models/trained/unet_fire_predictor.pth")
        lstm_path = Path("models/trained/lstm_fire_predictor.pth")
        
        if unet_path.exists():
            print("🔄 Loading pre-trained U-NET weights...")
            self.unet_model.load_state_dict(torch.load(unet_path, map_location=self.device))
            self.models_trained = True
        else:
            print("⚠️ No pre-trained U-NET weights found, using random initialization")
        
        if lstm_path.exists():
            print("🔄 Loading pre-trained LSTM weights...")
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
        else:
            print("⚠️ No pre-trained LSTM weights found, using random initialization")
    
    def train_models(self, X_spatial: np.ndarray, y_spatial: np.ndarray, 
                    X_temporal: Optional[np.ndarray] = None, y_temporal: Optional[np.ndarray] = None,
                    epochs: int = 50, batch_size: int = 4) -> Dict:
        """
        Train U-NET and LSTM models on fire data.
        
        Args:
            X_spatial: Spatial features (B, C, H, W)
            y_spatial: Spatial targets (B, H, W)
            X_temporal: Temporal features (B, T, F) - optional
            y_temporal: Temporal targets (B,) - optional
            epochs: Training epochs
            batch_size: Batch size
        """
        print("🎓 Training fire prediction models...")
        
        training_results = {
            "unet_losses": [],
            "lstm_losses": [],
            "unet_accuracy": [],
            "final_accuracy": 0.0
        }
        
        # Train U-NET for spatial prediction
        if X_spatial is not None and y_spatial is not None:
            print("🧠 Training U-NET model...")
            unet_results = self._train_unet(X_spatial, y_spatial, epochs, batch_size)
            training_results["unet_losses"] = unet_results["losses"]
            training_results["unet_accuracy"] = unet_results["accuracy"]
        
        # Train LSTM for temporal prediction (if temporal data provided)
        if X_temporal is not None and y_temporal is not None:
            print("⏰ Training LSTM model...")
            lstm_results = self._train_lstm(X_temporal, y_temporal, epochs, batch_size)
            training_results["lstm_losses"] = lstm_results["losses"]
        
        # Save trained models
        self._save_models()
        
        self.models_trained = True
        self.training_history = training_results
        
        # Calculate final accuracy
        training_results["final_accuracy"] = training_results["unet_accuracy"][-1] if training_results["unet_accuracy"] else 0.0
        
        print(f"✅ Model training complete - Final accuracy: {training_results['final_accuracy']:.1%}")
        return training_results
    
    def _train_unet(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> Dict:
        """Train the U-NET model."""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)  # Add channel dim
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.unet_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        losses = []
        accuracies = []
        
        self.unet_model.train()
        
        for epoch in range(epochs):
            epoch_losses = []
            epoch_accuracies = []
            
            # Process in batches
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.unet_model(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                pred_binary = (predictions > 0.5).float()
                accuracy = (pred_binary == batch_y).float().mean()
                
                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy.item())
            
            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            
            losses.append(avg_loss)
            accuracies.append(avg_accuracy)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.1%}")
        
        return {"losses": losses, "accuracy": accuracies}
    
    def _train_lstm(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> Dict:
        """Train the LSTM model."""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        losses = []
        self.lstm_model.train()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                predictions = self.lstm_model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"LSTM Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        return {"losses": losses}
    
    def predict_fire_probability(self, features: np.ndarray, target_date: str) -> np.ndarray:
        """
        Predict fire probability for next day using trained models.
        
        As per problem statement:
        - Input: Feature stack at 30m resolution
        - Output: Fire probability map
        """
        print(f"🔮 Predicting fire probability for {target_date}...")
        
        if not self.models_trained:
            print("⚠️ Models not trained yet, using mock predictions")
            return self._generate_mock_predictions(features)
        
        self.unet_model.eval()
        
        with torch.no_grad():
            # Convert features to tensor
            if features.ndim == 3:
                features = features[np.newaxis, ...]  # Add batch dimension
            
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            # U-NET prediction
            unet_pred = self.unet_model(features_tensor)
            
            # Convert back to numpy
            predictions = unet_pred.cpu().numpy().squeeze()
        
        print(f"✅ Fire probability prediction complete - Max probability: {predictions.max():.3f}")
        return predictions
    
    def _generate_mock_predictions(self, features: np.ndarray) -> np.ndarray:
        """Generate realistic mock predictions for demonstration."""
        
        if features.ndim == 3:
            _, height, width = features.shape
        else:
            height, width = features.shape[-2:]
        
        # Create realistic fire probability patterns
        np.random.seed(42)
        
        # Base probability from terrain and weather
        elevation = features[6] if features.ndim == 3 else np.random.rand(height, width)
        temperature = features[0] if features.ndim == 3 else np.random.rand(height, width)
        
        # Higher probability in certain conditions
        base_prob = 0.1 + 0.3 * (temperature / temperature.max()) * (1 - elevation / elevation.max())
        
        # Add hotspots
        num_hotspots = np.random.randint(3, 8)
        for _ in range(num_hotspots):
            center_y = np.random.randint(height//4, 3*height//4)
            center_x = np.random.randint(width//4, 3*width//4)
            
            # Create gaussian hotspot
            y, x = np.ogrid[:height, :width]
            mask = ((x - center_x)**2 + (y - center_y)**2) < (20**2)
            base_prob[mask] = np.clip(base_prob[mask] + 0.6, 0, 1)
        
        # Add noise
        noise = np.random.normal(0, 0.05, (height, width))
        predictions = np.clip(base_prob + noise, 0, 1)
        
        return predictions
    
    def create_binary_fire_map(self, probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Create binary fire classification map.
        
        As per problem statement:
        - Output: Binary classification (fire/no fire)
        """
        print(f"🎯 Creating binary fire map (threshold: {threshold})...")
        
        binary_map = (probabilities > threshold).astype(np.uint8)
        
        fire_pixels = binary_map.sum()
        total_pixels = binary_map.size
        fire_percentage = (fire_pixels / total_pixels) * 100
        
        print(f"✅ Binary map created - {fire_pixels} fire pixels ({fire_percentage:.2f}%)")
        return binary_map
    
    def _save_models(self):
        """Save trained models."""
        models_dir = Path("models/trained")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.unet_model.state_dict(), models_dir / "unet_fire_predictor.pth")
        torch.save(self.lstm_model.state_dict(), models_dir / "lstm_fire_predictor.pth")
        
        # Save training history
        joblib.dump(self.training_history, models_dir / "training_history.pkl")
        
        print("💾 Models saved successfully")
    
    def get_model_info(self) -> Dict:
        """Get information about the models."""
        return {
            "unet_parameters": sum(p.numel() for p in self.unet_model.parameters()),
            "lstm_parameters": sum(p.numel() for p in self.lstm_model.parameters()),
            "device": str(self.device),
            "models_trained": self.models_trained,
            "training_history": self.training_history
        }

if __name__ == "__main__":
    # Test the fire prediction engine
    engine = FirePredictionEngine()
    
    # Mock data for testing
    batch_size, channels, height, width = 2, 9, 256, 256
    X_spatial = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    y_spatial = np.random.randint(0, 2, (batch_size, height, width)).astype(np.float32)
    
    # Test prediction
    features = np.random.randn(channels, height, width).astype(np.float32)
    predictions = engine.predict_fire_probability(features, "2024-01-15")
    
    print(f"Prediction shape: {predictions.shape}")
    print(f"Prediction range: {predictions.min():.3f} - {predictions.max():.3f}")
    
    # Test binary classification
    binary_map = engine.create_binary_fire_map(predictions, threshold=0.5)
    print(f"Binary map shape: {binary_map.shape}")
    print(f"Fire pixels: {binary_map.sum()}") 