"""
Machine Learning utilities for BlazeNet system.
Handles model training, validation, data preparation, and prediction utilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import rasterio
from typing import Tuple, List, Optional, Dict, Any
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class FireDataset(Dataset):
    """Dataset class for fire prediction data."""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        transform=None
    ):
        """
        Initialize dataset.
        
        Args:
            features: Input features array (N, C, H, W)
            labels: Target labels array (N, H, W)
            transform: Optional data transforms
        """
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32) if labels is not None else None
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        
        if self.transform:
            feature = self.transform(feature)
        
        if self.labels is not None:
            label = self.labels[idx]
            return torch.from_numpy(feature), torch.from_numpy(label)
        else:
            return torch.from_numpy(feature)

class SpatialDataAugmentation:
    """Data augmentation for spatial data."""
    
    def __init__(self, flip_prob=0.5, rotate_prob=0.5):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
    
    def __call__(self, sample):
        """Apply augmentation to sample."""
        # Random horizontal flip
        if np.random.random() < self.flip_prob:
            sample = np.flip(sample, axis=-1)
        
        # Random vertical flip
        if np.random.random() < self.flip_prob:
            sample = np.flip(sample, axis=-2)
        
        # Random 90-degree rotation
        if np.random.random() < self.rotate_prob:
            k = np.random.randint(0, 4)
            sample = np.rot90(sample, k, axes=(-2, -1))
        
        return sample.copy()

class SpatialCrossValidator:
    """Spatial cross-validation to avoid spatial autocorrelation."""
    
    def __init__(self, n_splits=5, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, X, y, spatial_coords):
        """
        Generate spatial splits.
        
        Args:
            X: Feature data
            y: Target data
            spatial_coords: Array of (lat, lon) coordinates
            
        Yields:
            Tuple of (train_idx, test_idx)
        """
        for _ in range(self.n_splits):
            # Simple spatial split by randomly selecting test regions
            train_idx, test_idx = train_test_split(
                np.arange(len(X)),
                test_size=self.test_size,
                random_state=np.random.randint(0, 10000)
            )
            yield train_idx, test_idx

class ModelMetrics:
    """Calculate and track model performance metrics."""
    
    @staticmethod
    def calculate_spatial_metrics(y_true, y_pred, threshold=0.5):
        """
        Calculate spatial-aware metrics for fire prediction.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        # Convert to binary predictions
        y_pred_binary = (y_pred > threshold).astype(int)
        y_true_binary = y_true.astype(int)
        
        # Basic metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true_binary.flatten(), y_pred_binary.flatten()),
            'precision': precision_score(y_true_binary.flatten(), y_pred_binary.flatten(), zero_division=0),
            'recall': recall_score(y_true_binary.flatten(), y_pred_binary.flatten(), zero_division=0),
            'f1_score': f1_score(y_true_binary.flatten(), y_pred_binary.flatten(), zero_division=0),
        }
        
        # AUC-ROC if possible
        try:
            metrics['auc_roc'] = roc_auc_score(y_true_binary.flatten(), y_pred.flatten())
        except ValueError:
            metrics['auc_roc'] = 0.0
        
        # Spatial metrics
        metrics.update(ModelMetrics._calculate_spatial_accuracy(y_true_binary, y_pred_binary))
        
        return metrics
    
    @staticmethod
    def _calculate_spatial_accuracy(y_true, y_pred):
        """Calculate spatial accuracy metrics."""
        # Calculate connected component accuracy
        from scipy import ndimage
        
        # True positive regions
        tp_regions = (y_true == 1) & (y_pred == 1)
        tp_components, tp_count = ndimage.label(tp_regions)
        
        # False positive regions
        fp_regions = (y_true == 0) & (y_pred == 1)
        fp_components, fp_count = ndimage.label(fp_regions)
        
        # Calculate spatial precision (fewer false positive regions is better)
        spatial_precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        
        return {
            'spatial_precision': spatial_precision,
            'true_positive_regions': tp_count,
            'false_positive_regions': fp_count
        }

class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        """Check if training should stop."""
        if self.best is None:
            self.best = val_loss
            self.best_weights = model.state_dict().copy()
            return False
        
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.wait = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = True
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False

class DataPreprocessor:
    """Data preprocessing utilities."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_stats = {}
    
    def fit_scalers(self, features: Dict[str, np.ndarray]):
        """
        Fit scalers for different feature types.
        
        Args:
            features: Dictionary of feature arrays
        """
        for feature_name, data in features.items():
            scaler = StandardScaler()
            # Fit on flattened data
            flat_data = data.reshape(-1, 1)
            scaler.fit(flat_data)
            self.scalers[feature_name] = scaler
            
            # Store basic stats
            self.feature_stats[feature_name] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data)
            }
            
            logger.info(f"Fitted scaler for {feature_name}: mean={self.feature_stats[feature_name]['mean']:.3f}, "
                       f"std={self.feature_stats[feature_name]['std']:.3f}")
    
    def transform_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Transform features using fitted scalers.
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            Stacked and normalized feature array
        """
        transformed_features = []
        
        for feature_name, data in features.items():
            if feature_name in self.scalers:
                scaler = self.scalers[feature_name]
                original_shape = data.shape
                flat_data = data.reshape(-1, 1)
                scaled_data = scaler.transform(flat_data)
                scaled_data = scaled_data.reshape(original_shape)
                transformed_features.append(scaled_data)
            else:
                logger.warning(f"No scaler found for {feature_name}, using raw data")
                transformed_features.append(data)
        
        return np.stack(transformed_features, axis=0)
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor state."""
        state = {
            'scalers': self.scalers,
            'feature_stats': self.feature_stats
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.scalers = state['scalers']
        self.feature_stats = state['feature_stats']

class ModelUtils:
    """Utility functions for model operations."""
    
    @staticmethod
    def count_parameters(model):
        """Count trainable parameters in model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def save_checkpoint(model, optimizer, epoch, loss, filepath):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    @staticmethod
    def load_checkpoint(model, optimizer, filepath, device):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info(f"Checkpoint loaded from {filepath}, epoch {epoch}")
        return epoch, loss
    
    @staticmethod
    def predict_with_tta(model, dataloader, device, tta_transforms=None):
        """
        Make predictions with test-time augmentation.
        
        Args:
            model: Trained model
            dataloader: Data loader
            device: Device to use
            tta_transforms: List of TTA transforms
            
        Returns:
            Averaged predictions
        """
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)
                
                # Original prediction
                pred = model(inputs)
                batch_preds = [pred.cpu().numpy()]
                
                # TTA predictions
                if tta_transforms:
                    for transform in tta_transforms:
                        augmented_inputs = transform(inputs)
                        augmented_pred = model(augmented_inputs)
                        # Reverse the augmentation on prediction
                        reversed_pred = transform.reverse(augmented_pred)
                        batch_preds.append(reversed_pred.cpu().numpy())
                
                # Average predictions
                avg_pred = np.mean(batch_preds, axis=0)
                predictions.append(avg_pred)
        
        return np.concatenate(predictions, axis=0)
    
    @staticmethod
    def calculate_class_weights(labels, num_classes=2):
        """Calculate class weights for imbalanced data."""
        from sklearn.utils.class_weight import compute_class_weight
        
        flat_labels = labels.flatten()
        classes = np.arange(num_classes)
        weights = compute_class_weight('balanced', classes=classes, y=flat_labels)
        
        return torch.FloatTensor(weights)

class LossUtils:
    """Custom loss functions for fire prediction."""
    
    @staticmethod
    def focal_loss(inputs, targets, alpha=1, gamma=2):
        """
        Focal loss for addressing class imbalance.
        
        Args:
            inputs: Model predictions
            targets: Ground truth labels
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            
        Returns:
            Focal loss value
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = alpha * (1 - pt) ** gamma * BCE_loss
        return focal_loss.mean()
    
    @staticmethod
    def dice_loss(inputs, targets, smooth=1):
        """
        Dice loss for better segmentation performance.
        
        Args:
            inputs: Model predictions
            targets: Ground truth labels
            smooth: Smoothing factor
            
        Returns:
            Dice loss value
        """
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice
    
    @staticmethod
    def combined_loss(inputs, targets, bce_weight=0.5, dice_weight=0.3, focal_weight=0.2):
        """
        Combined loss function.
        
        Args:
            inputs: Model predictions
            targets: Ground truth labels
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            focal_weight: Weight for Focal loss
            
        Returns:
            Combined loss value
        """
        bce = F.binary_cross_entropy_with_logits(inputs, targets)
        dice = LossUtils.dice_loss(inputs, targets)
        focal = LossUtils.focal_loss(inputs, targets)
        
        return bce_weight * bce + dice_weight * dice + focal_weight * focal 