"""
LSTM model for temporal fire prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell."""
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class TemporalFireLSTM(nn.Module):
    """LSTM model for temporal fire prediction."""
    
    def __init__(self, input_features=6, hidden_size=128, num_layers=2, 
                 sequence_length=7, output_size=1, dropout=0.2):
        super(TemporalFireLSTM, self).__init__()
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.output_size = output_size
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_size, kernel_size=1)
        )
    
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        
        # Extract features for each time step
        features = []
        for t in range(seq_len):
            feat = self.feature_extractor(x[:, t])
            # Global average pooling
            feat_pooled = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
            features.append(feat_pooled)
        
        # Stack features
        feature_sequence = torch.stack(features, dim=1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(feature_sequence)
        
        # Get the last output and reshape for spatial prediction
        last_output = lstm_out[:, -1]  # (batch, hidden_size)
        spatial_output = last_output.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
        
        # Generate prediction
        prediction = self.output_conv(spatial_output)
        
        return prediction

def create_fire_lstm(config):
    """Create LSTM model based on configuration."""
    model = TemporalFireLSTM(
        input_features=6,
        hidden_size=128,
        num_layers=2,
        sequence_length=7,
        output_size=1,
        dropout=0.2
    )
    
    return model
