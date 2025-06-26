"""
U-Net model for spatial fire prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle different input sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionBlock(nn.Module):
    """Attention mechanism for U-Net."""
    
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class UNet(nn.Module):
    """U-Net model for fire prediction."""
    
    def __init__(self, n_channels=6, n_classes=1, bilinear=False, use_attention=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Attention blocks
        if use_attention:
            self.att1 = AttentionBlock(F_g=512, F_l=512, F_int=256)
            self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
            self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
            self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        
        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with attention
        if self.use_attention:
            x4_att = self.att1(g=x5, x=x4)
            x = self.up1(x5, x4_att)
            
            x3_att = self.att2(g=x, x=x3)
            x = self.up2(x, x3_att)
            
            x2_att = self.att3(g=x, x=x2)
            x = self.up3(x, x2_att)
            
            x1_att = self.att4(g=x, x=x1)
            x = self.up4(x, x1_att)
        else:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        
        x = self.dropout(x)
        logits = self.outc(x)
        return logits

class FirePredictionUNet(nn.Module):
    """Specialized U-Net for fire prediction with additional features."""
    
    def __init__(self, n_channels=6, n_classes=1, use_attention=True, weather_features=4):
        super(FirePredictionUNet, self).__init__()
        
        # Main U-Net
        self.unet = UNet(n_channels, n_classes, use_attention=use_attention)
        
        # Weather feature processing
        self.weather_fc = nn.Sequential(
            nn.Linear(weather_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(n_classes + 1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )
    
    def forward(self, spatial_features, weather_features=None):
        # Get spatial predictions
        spatial_pred = self.unet(spatial_features)
        
        if weather_features is not None:
            # Process weather features
            weather_embed = self.weather_fc(weather_features)
            
            # Expand weather features to spatial dimensions
            B, _ = weather_embed.shape
            H, W = spatial_pred.shape[2], spatial_pred.shape[3]
            weather_spatial = weather_embed.view(B, -1, 1, 1).expand(-1, -1, H, W)
            
            # Combine spatial and weather features
            combined = torch.cat([spatial_pred, weather_spatial[:, :1]], dim=1)
            final_pred = self.fusion_conv(combined)
            
            return final_pred
        else:
            return spatial_pred

def create_fire_unet(config):
    """Create U-Net model based on configuration."""
    model = FirePredictionUNet(
        n_channels=6,  # DEM, slope, aspect, LULC, NDVI, distance_to_roads
        n_classes=1,   # Binary fire prediction
        use_attention=True,
        weather_features=4  # temperature, humidity, wind_speed, precipitation
    )
    
    return model 