"""
Training script for fire prediction models.
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from utils.config import config
from utils.ml_utils import FireDataset, ModelMetrics, DataPreprocessor, EarlyStopping, LossUtils

logger = logging.getLogger(__name__)

class FirePredictionTrainer:
    """Trainer for fire prediction models."""
    
    def __init__(self, model_type='unet', config_dict=None):
        self.model_type = model_type
        self.config = config_dict or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.epochs = self.config.get('epochs', 100)
        self.batch_size = self.config.get('batch_size', 16)
        self.learning_rate = self.config.get('learning_rate', 1e-3)
        self.weight_decay = self.config.get('weight_decay', 1e-5)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.preprocessor = DataPreprocessor()
        self.early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def create_sample_data(self):
        """Create sample training data for demonstration."""
        logger.info("Creating sample training data...")
        
        # Create synthetic spatial data
        batch_size = 100
        channels = 6  # DEM, slope, aspect, LULC, NDVI, distance
        height, width = 128, 128
        
        # Generate random features
        features = np.random.randn(batch_size, channels, height, width).astype(np.float32)
        
        # Generate corresponding labels (fire probability maps)
        labels = np.random.rand(batch_size, height, width).astype(np.float32)
        labels = (labels > 0.7).astype(np.float32)  # Binary fire/no-fire
        
        return features, labels
    
    def prepare_data(self, data_path=None):
        """Prepare training and validation data."""
        logger.info("Preparing training data...")
        
        # For now, create sample data
        features, labels = self.create_sample_data()
        
        logger.info(f"Loaded {len(features)} training samples")
        
        # Create datasets
        dataset = FireDataset(features, labels)
        
        # Split into train/validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True
        )
        
        logger.info(f"Created train loader: {len(self.train_loader)} batches")
        logger.info(f"Created validation loader: {len(self.val_loader)} batches")
    
    def create_simple_unet(self):
        """Create a simple U-Net model."""
        class SimpleUNet(nn.Module):
            def __init__(self, in_channels=6, out_channels=1):
                super(SimpleUNet, self).__init__()
                
                # Encoder
                self.enc1 = nn.Sequential(
                    nn.Conv2d(in_channels, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
                
                self.pool1 = nn.MaxPool2d(2)
                
                self.enc2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )
                
                self.pool2 = nn.MaxPool2d(2)
                
                # Bottleneck
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
                
                # Decoder
                self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec2 = nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )
                
                self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec1 = nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
                
                self.final = nn.Conv2d(64, out_channels, 1)
            
            def forward(self, x):
                # Encoder
                enc1 = self.enc1(x)
                enc2 = self.enc2(self.pool1(enc1))
                
                # Bottleneck
                bottleneck = self.bottleneck(self.pool2(enc2))
                
                # Decoder
                dec2 = self.upconv2(bottleneck)
                dec2 = torch.cat([dec2, enc2], dim=1)
                dec2 = self.dec2(dec2)
                
                dec1 = self.upconv1(dec2)
                dec1 = torch.cat([dec1, enc1], dim=1)
                dec1 = self.dec1(dec1)
                
                return self.final(dec1)
        
        return SimpleUNet()
    
    def setup_model(self):
        """Setup model, optimizer, and loss function."""
        logger.info(f"Setting up {self.model_type} model...")
        
        # Create model
        if self.model_type == 'unet':
            self.model = self.create_simple_unet()
        else:
            raise ValueError(f"Model type {self.model_type} not implemented yet")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Setup loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Reshape output to match target
            output = output.squeeze(1)  # Remove channel dimension for loss calculation
            
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions for metrics
            with torch.no_grad():
                predictions = torch.sigmoid(output).cpu().numpy()
                targets = target.cpu().numpy()
                all_predictions.append(predictions)
                all_targets.append(targets)
            
            if batch_idx % 10 == 0:
                logger.info(f'Train Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}')
        
        # Calculate metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        metrics = ModelMetrics.calculate_metrics(all_targets, all_predictions)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, metrics
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                output = output.squeeze(1)  # Remove channel dimension
                
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Store predictions for metrics
                predictions = torch.sigmoid(output).cpu().numpy()
                targets = target.cpu().numpy()
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Calculate metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        metrics = ModelMetrics.calculate_metrics(all_targets, all_predictions)
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, metrics
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            logger.info(f"Train F1: {train_metrics['f1_score']:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', epoch, val_loss)
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                logger.info("Early stopping triggered")
                break
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, filename, epoch, loss):
        """Save model checkpoint."""
        checkpoint_path = config.MODEL_PATH / filename
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train fire prediction model')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'lstm'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Training configuration
    train_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'model_type': args.model
    }
    
    try:
        # Initialize trainer
        trainer = FirePredictionTrainer(args.model, train_config)
        
        # Prepare data
        trainer.prepare_data()
        
        # Setup model
        trainer.setup_model()
        
        # Train model
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
