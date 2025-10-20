#!/usr/bin/env python3

import os
import warnings
warnings.filterwarnings('ignore')

# Core ML/DL Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchvision.transforms as transforms

# Scientific Computing
import numpy as np
import pandas as pd
import dask.dataframe as dd
from scipy import optimize, stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Geospatial and Time Series
import xarray as xr
from datetime import datetime, timedelta
import geopandas as gpd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Bayesian Optimization (optional)
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    print("  scikit-optimize not available. Bayesian optimization will be disabled.")
    SKOPT_AVAILABLE = False

# Explainability (optional)
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    print("  LIME not available. LIME explanations will be disabled.")
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("  SHAP not available. SHAP explanations will be disabled.")
    SHAP_AVAILABLE = False

# Advanced Architectures
from transformers import AutoModel, AutoConfig
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, GraphConv

# Memory and Performance
import gc
from tqdm import tqdm
import pickle
import joblib
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Check MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(" Using Apple Metal Performance Shaders (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(" Using CUDA GPU")
else:
    device = torch.device("cpu")
    print(" Using CPU")

print(f"Device: {device}")

class ZeroCurtainTemporalPatternAnalyzer(nn.Module):
    """
    Advanced temporal pattern analysis for discriminating between different
    zero-curtain event types based on physics and ecological memory.
    
    Event Types:
    1. Abrupt/Rapid Events: Single, short-duration isolated events
    2. Consecutive Abrupt Events: Multiple rapid events in sequence
    3. Extended Persistent Events: Long-duration continuous events
    4. Composite Extended Events: Extended periods with variable internal dynamics
    """
    
    def __init__(self, d_model=256, memory_depth=50):
        super().__init__()
        
        self.d_model = d_model
        self.memory_depth = memory_depth
        
        # Temporal pattern detection components
        self.pattern_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=3,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Multi-scale temporal convolutions for different event timescales
        self.rapid_detector = nn.Conv1d(d_model, d_model//2, kernel_size=3, padding=1)  # Hours-days
        self.consecutive_detector = nn.Conv1d(d_model, d_model//2, kernel_size=7, padding=3)  # Days-weeks
        self.extended_detector = nn.Conv1d(d_model, d_model//2, kernel_size=15, padding=7)  # Weeks-months
        self.seasonal_detector = nn.Conv1d(d_model, d_model//2, kernel_size=31, padding=15)  # Months-seasons
        
        # Pattern classification heads
        self.pattern_classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 for bidirectional LSTM
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 4)  # 4 pattern types
        )
        
        # Continuity analysis
        self.continuity_analyzer = nn.Sequential(
            nn.Linear(d_model * 2, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, 1),
            nn.Sigmoid()  # Continuity score [0,1]
        )
        
        # Inter-event relationship detector
        self.relationship_detector = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Physics-informed constraints
        self.physics_constraints = PhysicsInformedTemporalConstraints()
        
    def forward(self, event_sequence, timestamps, event_features):
        """
        Analyze temporal patterns in zero-curtain event sequences.
        
        Args:
            event_sequence: (batch_size, seq_len, d_model) - Sequence of event representations
            timestamps: (batch_size, seq_len) - Event timestamps
            event_features: (batch_size, seq_len, 3) - [duration, intensity, extent]
        """
        batch_size, seq_len, _ = event_sequence.shape
        
        # 1. Bidirectional LSTM for temporal context
        lstm_output, (hidden, cell) = self.pattern_encoder(event_sequence)
        
        # 2. Multi-scale temporal pattern detection
        # Transpose for Conv1d: (batch, features, time)
        conv_input = event_sequence.transpose(1, 2)
        
        rapid_patterns = self.rapid_detector(conv_input)
        consecutive_patterns = self.consecutive_detector(conv_input)
        extended_patterns = self.extended_detector(conv_input)
        seasonal_patterns = self.seasonal_detector(conv_input)
        
        # Combine multi-scale patterns
        multi_scale = torch.cat([
            rapid_patterns, consecutive_patterns,
            extended_patterns, seasonal_patterns
        ], dim=1).transpose(1, 2)  # Back to (batch, time, features)
        
        # 3. Inter-event relationship analysis
        relationship_output, attention_weights = self.relationship_detector(
            lstm_output, lstm_output, lstm_output
        )
        
        # 4. Pattern classification for each timestep
        pattern_logits = self.pattern_classifier(lstm_output)
        pattern_probs = F.softmax(pattern_logits, dim=-1)
        
        # 5. Continuity analysis
        continuity_scores = self.continuity_analyzer(lstm_output)
        
        # 6. Physics-informed pattern validation
        physics_validated_patterns = self.physics_constraints(
            pattern_probs, event_features, timestamps, continuity_scores
        )
        
        return {
            'pattern_probabilities': physics_validated_patterns,
            'continuity_scores': continuity_scores,
            'inter_event_attention': attention_weights,
            'multi_scale_features': multi_scale,
            'temporal_encoding': lstm_output
        }

class PhysicsInformedTemporalConstraints(nn.Module):
    """
    Apply physics-based constraints to temporal pattern classification.
    """
    
    def __init__(self):
        super().__init__()
        
        # Physics constants for zero-curtain events
        self.MIN_RAPID_DURATION = 6    # hours
        self.MAX_RAPID_DURATION = 72   # hours (3 days)
        self.MIN_EXTENDED_DURATION = 168  # hours (1 week)
        self.CONSECUTIVE_GAP_THRESHOLD = 48  # hours between events
        
    def forward(self, pattern_probs, event_features, timestamps, continuity_scores):
        """
        Apply physics constraints to refine pattern probabilities.
        """
        batch_size, seq_len, num_patterns = pattern_probs.shape
        durations = event_features[:, :, 0]  # Duration is first feature
        
        # Create physics-based masks
        physics_mask = torch.ones_like(pattern_probs)
        
        for b in range(batch_size):
            for t in range(seq_len):
                duration = durations[b, t]
                
                # Pattern 0: Rapid/Abrupt Events
                if duration > self.MAX_RAPID_DURATION:
                    physics_mask[b, t, 0] *= 0.1  # Strongly discourage
                    
                # Pattern 1: Consecutive Abrupt Events
                if t > 0:
                    time_gap = timestamps[b, t] - timestamps[b, t-1]
                    if time_gap > self.CONSECUTIVE_GAP_THRESHOLD:
                        physics_mask[b, t, 1] *= 0.3  # Discourage if gap too large
                        
                # Pattern 2: Extended Persistent Events
                if duration < self.MIN_EXTENDED_DURATION:
                    physics_mask[b, t, 2] *= 0.2  # Discourage short durations
                    
                # Pattern 3: Composite Extended Events
                # Require high continuity score and variable internal dynamics
                if continuity_scores[b, t] < 0.5:
                    physics_mask[b, t, 3] *= 0.4
        
        # Apply physics constraints
        constrained_probs = pattern_probs * physics_mask
        
        # Renormalize
        constrained_probs = F.softmax(constrained_probs, dim=-1)
        
        return constrained_probs

class ExtendedEventDetector(nn.Module):
    """
    Specialized detector for identifying extended zero-curtain events
    that may appear as consecutive short events but represent a single
    persistent phenomenon.
    """
    
    def __init__(self, d_model=256, max_gap_hours=72):
        super().__init__()
        
        self.d_model = d_model
        self.max_gap_hours = max_gap_hours
        
        # Event similarity encoder
        self.similarity_encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Spatial continuity detector
        self.spatial_continuity = nn.Sequential(
            nn.Linear(6, d_model//4),  # lat, lon, depth for two events
            nn.GELU(),
            nn.Linear(d_model//4, 1),
            nn.Sigmoid()
        )
        
        # Extended event classifier
        self.extended_classifier = nn.Sequential(
            nn.Linear(d_model + 3, d_model//2),  # features + similarity + spatial + temporal
            nn.GELU(),
            nn.Linear(d_model//2, 2),  # extended vs separate
            nn.Softmax(dim=-1)
        )
        
    def forward(self, event_pairs, spatial_features, temporal_gaps):
        """
        Determine if consecutive events are part of an extended event.
        
        Args:
            event_pairs: (batch_size, num_pairs, 2, d_model) - Pairs of consecutive events
            spatial_features: (batch_size, num_pairs, 6) - Spatial info for each pair
            temporal_gaps: (batch_size, num_pairs) - Time gaps between events
        """
        batch_size, num_pairs, _, _ = event_pairs.shape
        
        # Calculate event similarity
        event1 = event_pairs[:, :, 0, :]  # First event in pair
        event2 = event_pairs[:, :, 1, :]  # Second event in pair
        
        # Concatenate for similarity analysis
        paired_features = torch.cat([event1, event2], dim=-1)
        similarity_scores = self.similarity_encoder(paired_features)
        
        # Spatial continuity
        spatial_continuity_scores = self.spatial_continuity(spatial_features)
        
        # Temporal continuity (based on gap duration)
        temporal_continuity = torch.exp(-temporal_gaps / self.max_gap_hours).unsqueeze(-1)
        
        # Combine all features
        combined_features = torch.cat([
            event1,  # Use first event as representative
            similarity_scores,
            spatial_continuity_scores,
            temporal_continuity
        ], dim=-1)
        
        # Classify as extended vs separate events
        extended_probs = self.extended_classifier(combined_features)

class PhysicsInformedLoss(nn.Module):
    """
    Enhanced physics-informed loss function with temporal pattern awareness.
    Incorporates constraints for different zero-curtain event types.
    """
    
    def __init__(self, alpha_mse=1.0, alpha_physics=0.5, alpha_temporal=0.3, alpha_pattern=0.2):
        super().__init__()
        self.alpha_mse = alpha_mse          # Standard MSE weight
        self.alpha_physics = alpha_physics   # Physics constraint weight
        self.alpha_temporal = alpha_temporal # Temporal consistency weight
        self.alpha_pattern = alpha_pattern   # Pattern classification weight
        
        # Physical constants from LPJ-EOSIM/CryoGrid
        self.LHEAT = 3.34E8          # Latent heat of fusion [J m-3]
        self.TMFW = 273.15           # Freezing temperature [K]
                # Pattern-specific constraints
        self.RAPID_MAX_DURATION = 72    # hours
        self.EXTENDED_MIN_DURATION = 168  # hours
        self.CONSECUTIVE_GAP_MAX = 48   # hours
        
    def forward(self, predictions, targets, physics_features=None, pattern_predictions=None, pattern_targets=None):
        """
        Calculate enhanced physics-informed loss with pattern awareness.
        
        Args:
            predictions: (batch_size, 3) - [intensity, duration, extent]
            targets: (batch_size, 3) - [intensity, duration, extent]
            physics_features: (batch_size, n_features) - Additional physics data
            pattern_predictions: (batch_size, 4) - Pattern type probabilities
            pattern_targets: (batch_size,) - True pattern labels
        """
        
        # Extract predictions and targets
        pred_intensity, pred_duration, pred_extent = predictions[:, 0], predictions[:, 1], predictions[:, 2]
        true_intensity, true_duration, true_extent = targets[:, 0], targets[:, 1], targets[:, 2]
        
        # 1. Standard MSE Loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # 2. Enhanced Physics-Informed Constraints
        physics_loss = 0.0
        
        # Constraint 1: Pattern-aware duration constraints
        if pattern_predictions is not None:
            # Get dominant pattern type
            dominant_patterns = torch.argmax(pattern_predictions, dim=-1)
            
            # Apply pattern-specific duration constraints
            for pattern_type in range(4):
                pattern_mask = (dominant_patterns == pattern_type)
                if pattern_mask.any():
                    if pattern_type == 0:  # Rapid events
                        duration_violation = F.relu(pred_duration[pattern_mask] - self.RAPID_MAX_DURATION)
                        physics_loss += duration_violation.mean()
                    elif pattern_type == 2:  # Extended events
                        duration_violation = F.relu(self.EXTENDED_MIN_DURATION - pred_duration[pattern_mask])
                        physics_loss += duration_violation.mean()
        
        # Constraint 2: Zero-curtain intensity should correlate with duration
        # Longer events tend to have higher intensity (thermal stability)
        intensity_duration_correlation = F.relu(pred_intensity - 0.1) * F.relu(pred_duration - 6.0)
        expected_correlation = true_intensity * true_duration
        correlation_loss = F.mse_loss(intensity_duration_correlation, expected_correlation)
        physics_loss += correlation_loss
        
        # Constraint 3: Spatial extent physics bounds
        extent_bounds_loss = F.relu(pred_extent - 5.0) + F.relu(-pred_extent + 0.05)
        physics_loss += extent_bounds_loss.mean()
        
        # Constraint 4: Energy conservation with pattern awareness
        if physics_features is not None and physics_features.shape[1] >= 3:
            thermal_diffusivity = physics_features[:, 0]
            permafrost_prob = physics_features[:, 1]
            
            # Stefan problem: extent ∝ sqrt(thermal_diffusivity * time) * intensity_factor
            # Stefan problem with numerical stability
            thermal_diffusivity_stable = torch.clamp(thermal_diffusivity, min=1e-10, max=1e-3)
            pred_duration_stable = torch.clamp(pred_duration, min=1e-3, max=100.0)
            pred_intensity_stable = torch.clamp(pred_intensity, min=1e-3, max=1.0)

            expected_extent = torch.sqrt(thermal_diffusivity_stable * pred_duration_stable / 3600.0) * pred_intensity_stable
            expected_extent = torch.clamp(expected_extent, min=1e-6, max=10.0)
            
            # Pattern-specific modifications
            if pattern_predictions is not None:
                pattern_weights = torch.ones_like(pred_extent)
                
                # Extended events should have larger spatial extent
                extended_mask = (torch.argmax(pattern_predictions, dim=-1) == 2)
                pattern_weights[extended_mask] *= 1.5
                
                # Rapid events should have smaller spatial extent
                rapid_mask = (torch.argmax(pattern_predictions, dim=-1) == 0)
                pattern_weights[rapid_mask] *= 0.7
                
                expected_extent *= pattern_weights
            
            energy_conservation_loss = F.mse_loss(pred_extent, expected_extent.clamp(0.05, 5.0))
            physics_loss += energy_conservation_loss
        
        # Constraint 5: Permafrost enhancement with pattern interaction
        if physics_features is not None and physics_features.shape[1] >= 2:
            permafrost_prob = physics_features[:, 1]
            permafrost_enhancement = 1.0 + 0.5 * permafrost_prob
            
            # Pattern-specific permafrost effects
            if pattern_predictions is not None:
                # Extended events more influenced by permafrost
                extended_prob = pattern_predictions[:, 2]  # Extended pattern probability
                permafrost_enhancement *= (1.0 + 0.3 * extended_prob)
            
            enhanced_intensity = pred_intensity * permafrost_enhancement
            permafrost_loss = F.mse_loss(enhanced_intensity, true_intensity)
            physics_loss += 0.3 * permafrost_loss
        
        # 3. Temporal Consistency Loss (for sequential data)
        temporal_loss = 0.0
        if predictions.shape[0] > 1:
            # Penalize unrealistic temporal jumps with pattern awareness
            pred_diff = torch.diff(predictions, dim=0)
            temporal_variance = torch.var(pred_diff, dim=0)
            
            # Pattern-specific temporal smoothness expectations
            if pattern_predictions is not None and pattern_predictions.shape[0] > 1:
                pattern_diff = torch.diff(pattern_predictions, dim=0)
                pattern_consistency = -torch.mean(torch.sum(pattern_diff**2, dim=-1))  # Reward consistency
                temporal_loss += -0.1 * pattern_consistency  # Negative because we want to minimize loss
            
            temporal_loss += temporal_variance.mean()
        
        # 4. Pattern Classification Loss
        pattern_loss = 0.0
        if pattern_predictions is not None and pattern_targets is not None:
            pattern_loss = F.cross_entropy(pattern_predictions, pattern_targets)
        
        # 5. Combined Loss with NaN protection
        # Clamp individual losses to prevent numerical instability
        mse_loss_clamped = torch.clamp(mse_loss, max=100.0)
        physics_loss_clamped = torch.clamp(physics_loss, max=100.0) if isinstance(physics_loss, torch.Tensor) else min(physics_loss, 100.0)
        temporal_loss_clamped = torch.clamp(temporal_loss, max=100.0) if isinstance(temporal_loss, torch.Tensor) else min(temporal_loss, 100.0)
        pattern_loss_clamped = torch.clamp(pattern_loss, max=100.0) if isinstance(pattern_loss, torch.Tensor) else min(pattern_loss, 100.0)

        total_loss = (self.alpha_mse * mse_loss_clamped +
                     self.alpha_physics * physics_loss_clamped +
                     self.alpha_temporal * temporal_loss_clamped +
                     self.alpha_pattern * pattern_loss_clamped)

        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: NaN/Inf detected in loss, using MSE only")
            total_loss = mse_loss_clamped
        
        return total_loss, {
            'mse_loss': mse_loss.item(),
            'physics_loss': physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss,
            'temporal_loss': temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else temporal_loss,
            'pattern_loss': pattern_loss.item() if isinstance(pattern_loss, torch.Tensor) else pattern_loss,
            'total_loss': total_loss.item()
        }

class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism for capturing spatial, temporal, and physics-informed patterns.
    """
    
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Spatial attention for geographic patterns
        self.spatial_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Temporal attention for seasonal/long-term patterns
        self.temporal_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Physics-informed attention for thermodynamic relationships
        self.physics_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention between scales
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Layer normalization and feedforward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, spatial_mask=None, temporal_mask=None, physics_mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            spatial_mask, temporal_mask, physics_mask: Attention masks
        """
        batch_size, seq_len, d_model = x.shape
        
        # Spatial attention
        spatial_out, spatial_weights = self.spatial_attention(x, x, x, attn_mask=spatial_mask)
        x = self.norm1(x + spatial_out)
        
        # Temporal attention
        temporal_out, temporal_weights = self.temporal_attention(x, x, x, attn_mask=temporal_mask)
        x = self.norm2(x + temporal_out)
        
        # Physics-informed attention
        physics_out, physics_weights = self.physics_attention(x, x, x, attn_mask=physics_mask)
        x = self.norm3(x + physics_out)
        
        # Cross-scale integration
        cross_out, cross_weights = self.cross_attention(x, x, x)
        x = self.norm4(x + cross_out)
        
        # Feedforward
        ff_out = self.ff(x)
        x = x + ff_out
        
        return x, {
            'spatial_weights': spatial_weights,
            'temporal_weights': temporal_weights,
            'physics_weights': physics_weights,
            'cross_weights': cross_weights
        }

class LiquidNeuralUnit(nn.Module):
    """
    Liquid Neural Network unit for modeling ecological memory and continuous dynamics.
    Based on Neural ODEs and Liquid Time Constants.
    """
    
    def __init__(self, input_size, hidden_size, time_constant=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.time_constant = nn.Parameter(torch.tensor(time_constant))
        
        # Continuous dynamics
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        
        # Adaptive time constants - fix input size
        self.time_mlp = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, hidden_state=None, dt=1.0):
        """
        Args:
            x: (batch_size, input_size)
            hidden_state: (batch_size, hidden_size)
            dt: Time step
        """
        batch_size = x.shape[0]
        
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Ensure input dimensions are correct
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        
        # Project input to correct size if needed
        if x.shape[1] != self.input_size:
            # Add a projection layer on the fly
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(x.shape[1], self.input_size).to(x.device)
            x = self.input_projection(x)
        
        # Adaptive time constant with numerical stability
        time_input = torch.cat([x, hidden_state], dim=-1)
        adaptive_tau = torch.clamp(self.time_mlp(time_input) * self.time_constant, min=0.1, max=10.0)

        # Continuous dynamics with clamping
        input_contribution = torch.clamp(self.input_layer(x), min=-10.0, max=10.0)
        hidden_contribution = torch.clamp(self.hidden_layer(hidden_state), min=-10.0, max=10.0)

        # Leaky integration with numerical stability
        dh_dt = (input_contribution + hidden_contribution - hidden_state) / (adaptive_tau + 0.1)
        dh_dt = torch.clamp(dh_dt, min=-1.0, max=1.0)  # Prevent explosive gradients

        # Euler integration with smaller time step
        dt_stable = min(dt, 0.1)  # Cap time step
        new_hidden = hidden_state + dt_stable * dh_dt

        # Apply nonlinearity with gradient-friendly function
        new_hidden = torch.clamp(new_hidden, min=-5.0, max=5.0)
        new_hidden = torch.tanh(new_hidden)

        # Output with final clamping
        output = torch.clamp(self.output_layer(new_hidden), min=-5.0, max=5.0)

        # NaN check and fallback
        if torch.isnan(output).any() or torch.isnan(new_hidden).any():
            print("Warning: NaN detected in Liquid Neural Unit, using fallback")
            output = torch.zeros_like(output)
            new_hidden = torch.zeros_like(new_hidden)
        
        return output, new_hidden

class SpatioTemporalUNet(nn.Module):
    """
    U-Net architecture adapted for spatiotemporal zero-curtain prediction.
    Handles multi-scale spatial patterns with skip connections.
    """
    
    def __init__(self, in_channels=10, out_channels=3, base_filters=64):
        super().__init__()
        
        # Encoder (Downsampling)
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.enc4 = self._conv_block(base_filters * 4, base_filters * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 8, base_filters * 16)
        
        # Decoder (Upsampling)
        self.dec4 = self._upconv_block(base_filters * 16, base_filters * 8)
        self.dec3 = self._upconv_block(base_filters * 8, base_filters * 4)
        self.dec2 = self._upconv_block(base_filters * 4, base_filters * 2)
        self.dec1 = self._upconv_block(base_filters * 2, base_filters)
        
        # Output layer
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, height, width)
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        # Final output
        output = self.final_conv(d1)
        
        return output

class ZeroCurtainHybridModel(nn.Module):
    """
    Hybrid model combining Transformer, U-Net, Liquid Neural Networks,
    and physics-informed components for zero-curtain prediction.
    """
    
    def __init__(self,
                 input_features=34,      # From Part 1 dataframe
                 spatial_features=10,    # Spatial grid features
                 d_model=256,           # Reduced from 512
                 n_heads=8,             # Attention heads
                 n_layers=4,            # Reduced from 6
                 liquid_hidden=128,     # Reduced from 256
                 output_features=3,     # intensity, duration, extent
                 dropout=0.1):
        
        super().__init__()
        
        self.d_model = d_model
        self.liquid_hidden = liquid_hidden
        self.output_features = output_features
        
        # Enable shape debugging for first few iterations
        self.debug_shapes = False  # Disabled after confirming dimensions work
        
        # Input processing and embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Positional encoding for spatiotemporal data
        self.positional_encoding = PositionalEncoding(d_model, max_len=10000)
        
        # Multi-scale attention layers
        self.attention_layers = nn.ModuleList([
            MultiScaleAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Liquid Neural Network for ecological memory
        self.liquid_layers = nn.ModuleList([
            LiquidNeuralUnit(d_model, liquid_hidden),
            LiquidNeuralUnit(liquid_hidden, liquid_hidden),
            LiquidNeuralUnit(liquid_hidden, liquid_hidden)  # Changed to output liquid_hidden instead of d_model
        ])
        
        # Spatial processing with U-Net
        self.spatial_processor = SpatioTemporalUNet(
            in_channels=spatial_features,
            out_channels=d_model // 8
        )

        # PART II: GeoCryoAI spatiotemporal graph integration
        self.geocryoai_enabled = True  # Toggle for ablation studies
        
        if self.geocryoai_enabled:
            from geocryoai_integration import GeoCryoAIHybridGraphModel
            
            self.geocryoai_graph_model = GeoCryoAIHybridGraphModel(
                node_features=d_model,
                spatial_hidden=d_model // 2,
                temporal_hidden=d_model // 2,
                spatial_threshold_km=50.0  # 50km spatial connectivity threshold
            )
            
            print(" GeoCryoAI graph model initialized")
        
        # Physics-informed processing
        self.physics_encoder = nn.Sequential(
            nn.Linear(input_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 8)
        )
        
        # Feature fusion - adjusted for liquid_hidden output
        fusion_input_size = liquid_hidden + self.d_model // 8 + self.d_model // 8  # liquid + spatial + physics
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task output heads
        self.intensity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()  # Intensity is [0, 1]
        )
        
        self.duration_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.ReLU()  # Duration is positive
        )
        
        self.extent_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.ReLU()  # Spatial extent is positive
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, features, spatial_data=None, sequence_length=None, batch_metadata=None):
        """
        Args:
            features: (batch_size, sequence_length, input_features)
            spatial_data: (batch_size, spatial_features, height, width)
            sequence_length: Actual sequence lengths for variable-length sequences
        """

        # Store metadata for GeoCryoAI processing
        if batch_metadata is not None:
            self._current_batch_metadata = batch_metadata
        elif not hasattr(self, '_current_batch_metadata'):
            # Create dummy metadata if none provided
            batch_size = features.shape[0]
            self._current_batch_metadata = [
                {'site_lat': 0.0, 'site_lon': 0.0} for _ in range(batch_size)
            ]
        
        batch_size, seq_len, _ = features.shape
        
        # 1. Feature embedding and positional encoding
        embedded_features = self.feature_embedding(features)
        embedded_features = self.positional_encoding(embedded_features)
        
        # 2. Multi-scale attention processing
        attention_output = embedded_features
        attention_weights = {}
        
        for i, attention_layer in enumerate(self.attention_layers):
            attention_output, weights = attention_layer(attention_output)
            attention_weights[f'layer_{i}'] = weights
        
        # 3. Liquid Neural Network processing for ecological memory
        liquid_output = attention_output
        liquid_hidden = None
        
        # Process sequence through liquid layers
        liquid_outputs = []
        for t in range(seq_len):
            current_input = liquid_output[:, t, :]
            
            # Ensure current_input has correct dimensions
            if current_input.dim() == 1:
                current_input = current_input.unsqueeze(0)
            
            # Process through liquid layers with proper dimension handling
            for i, liquid_layer in enumerate(self.liquid_layers):
                # For the first layer, use d_model as input size
                if i == 0:
                    expected_input_size = self.d_model
                else:
                    expected_input_size = liquid_layer.hidden_size
                
                # Adjust input size if needed
                if current_input.shape[-1] != expected_input_size:
                    if not hasattr(self, f'liquid_projection_{i}'):
                        setattr(self, f'liquid_projection_{i}',
                               nn.Linear(current_input.shape[-1], expected_input_size).to(current_input.device))
                    projection = getattr(self, f'liquid_projection_{i}')
                    current_input = projection(current_input)
                
                current_input, liquid_hidden = liquid_layer(current_input, liquid_hidden)
            
            liquid_outputs.append(current_input)
        
        liquid_sequence = torch.stack(liquid_outputs, dim=1)
        
        # 3.5. PART II: GeoCryoAI spatiotemporal graph processing
        if hasattr(self, 'geocryoai_enabled') and self.geocryoai_enabled:
            if hasattr(self, 'geocryoai_graph_model'):
                # Extract timestamps from features if available
                # Assumes timestamps are encoded in features or passed separately
                # For demonstration, create synthetic timestamps based on sequence position
                timestamps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).float()
                
                try:
                    # Process through GeoCryoAI graph model
                    # Note: This requires metadata to be passed through forward()
                    # We'll need to modify the forward signature
                    geocryoai_features = self.geocryoai_graph_model(
                        batch_features=embedded_features,
                        batch_metadata=getattr(self, '_current_batch_metadata', None),
                        timestamps=timestamps
                    )
                    
                    # Blend GeoCryoAI features with liquid features
                    liquid_sequence = 0.7 * liquid_sequence + 0.3 * geocryoai_features
                    
                except Exception as e:
                    if self.debug_shapes:
                        print(f"Warning: GeoCryoAI processing failed: {e}")
                    # Continue without GeoCryoAI enhancement
                    pass

        # 4. Spatial processing (if spatial data available)
        spatial_features = None
        if spatial_data is not None:
            spatial_output = self.spatial_processor(spatial_data)
            # Global average pooling to get spatial features
            spatial_features = F.adaptive_avg_pool2d(spatial_output, (1, 1))
            spatial_features = spatial_features.view(batch_size, -1)
            # Expand to sequence length
            spatial_features = spatial_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 5. Physics-informed processing
        physics_features = self.physics_encoder(features)
        
        # 6. Feature fusion - handle dimension compatibility
        fusion_inputs = []
        
        # Add liquid features
        fusion_inputs.append(liquid_sequence)
        
        # Add spatial features or dummy
        if spatial_features is not None:
            fusion_inputs.append(spatial_features)
        else:
            dummy_spatial = torch.zeros(batch_size, seq_len, self.d_model // 8, device=features.device)
            fusion_inputs.append(dummy_spatial)
        
        # Add physics features
        fusion_inputs.append(physics_features)
        
        # Concatenate features - ensure all have compatible dimensions
        fused_features = torch.cat(fusion_inputs, dim=-1)
        
        # Debug: print shapes for troubleshooting
        if hasattr(self, 'debug_shapes') and self.debug_shapes:
            print(f"Liquid sequence shape: {liquid_sequence.shape}")
            print(f"Physics features shape: {physics_features.shape}")
            print(f"Fused features shape: {fused_features.shape}")
            print(f"Expected fusion input size: {self.fusion_layer[0].in_features}")
        
        # Apply fusion layer
        fused_output = self.fusion_layer(fused_features)
        
        # 7. Aggregate sequence information (use last timestep or attention pooling)
        if sequence_length is not None:
            # Use actual sequence lengths
            indices = torch.clamp(sequence_length - 1, 0, seq_len - 1)
            final_output = fused_output[range(batch_size), indices]
        else:
            # Use last timestep
            final_output = fused_output[:, -1, :]
        
        # 8. Multi-task prediction heads
        intensity = self.intensity_head(final_output)
        duration = self.duration_head(final_output)
        extent = self.extent_head(final_output)
        
        # Combine outputs
        predictions = torch.cat([intensity, duration, extent], dim=-1)
        
        return predictions, {
            'attention_weights': attention_weights,
            'liquid_features': liquid_sequence,
            'spatial_features': spatial_features,
            'physics_features': physics_features,
            'fused_features': fused_output
        }

class PositionalEncoding(nn.Module):
    """Positional encoding for spatiotemporal data."""
    
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class ZeroCurtainDataset(Dataset):
    """
    Enhanced PyTorch Dataset for zero-curtain data with temporal pattern analysis
    and extended event detection capabilities.
    """
    
    def __init__(self,
                 dataframe,
                 feature_columns=None,
                 target_columns=['intensity_percentile', 'duration_hours', 'spatial_extent_meters'],
                 sequence_length=10,
                 scaler=None,
                 target_scaler=None,
                 spatial_data=None,
                 enable_pattern_analysis=True):
        
        # Reset index to ensure continuous indexing
        self.df = dataframe.copy().reset_index(drop=True)
        self.sequence_length = sequence_length
        self.target_columns = target_columns
        self.spatial_data = spatial_data
        self.enable_pattern_analysis = enable_pattern_analysis
        
        # Define feature columns if not provided
        if feature_columns is None:
            self.feature_columns = [
                'mean_temperature', 'temperature_variance', 'permafrost_probability',
                'phase_change_energy', 'freeze_penetration_depth', 'thermal_diffusivity',
                'snow_insulation_factor', 'cryogrid_thermal_conductivity',
                'cryogrid_heat_capacity', 'cryogrid_enthalpy_stability',
                'surface_energy_balance', 'lateral_thermal_effects',
                'van_genuchten_alpha', 'van_genuchten_n', 'latitude', 'longitude'
            ]
        else:
            self.feature_columns = feature_columns
        
        # Remove any columns that don't exist in the dataframe
        self.feature_columns = [col for col in self.feature_columns if col in self.df.columns]
        
        # Handle missing values
        self.df[self.feature_columns] = self.df[self.feature_columns].fillna(0)
        self.df[self.target_columns] = self.df[self.target_columns].fillna(0)
        
        # Scaling
        if scaler is None:
            self.scaler = RobustScaler()
            self.features = self.scaler.fit_transform(self.df[self.feature_columns])
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.df[self.feature_columns])
        
        if target_scaler is None:
            # Use MinMaxScaler with specific ranges for better numerical stability
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
            self.targets = self.target_scaler.fit_transform(self.df[self.target_columns])
            
        else:
            self.target_scaler = target_scaler
            self.targets = self.target_scaler.transform(self.df[self.target_columns])
            #self.targets = self.target_scaler.transform(df_targets_transformed)
            
        # DEBUG TARGET SCALING (for both train and val/test datasets)
        print(f"Target scaling stats:")
        print(f"  Duration: min={self.targets[:, 1].min():.3f}, max={self.targets[:, 1].max():.3f}")
        print(f"  Intensity: min={self.targets[:, 0].min():.3f}, max={self.targets[:, 0].max():.3f}")
        print(f"  Extent: min={self.targets[:, 2].min():.3f}, max={self.targets[:, 2].max():.3f}")
        
        # Group by site for sequence creation
        self.df['site_id'] = self.df['latitude'].astype(str) + '_' + self.df['longitude'].astype(str)
        
        # Enhanced sequence creation with temporal pattern analysis
        self.sequences = []
        self.sequence_targets = []
        self.sequence_lengths = []
        self.temporal_patterns = []
        self.event_metadata = []
        
        # Group by site and create sequences with pattern analysis
        for site_id, group in self.df.groupby('site_id'):
            # Sort by time if time column exists
            if 'start_time' in group.columns:
                group_sorted = group.sort_values('start_time')
            else:
                group_sorted = group.sort_index()
            
            # Get the indices within our reset dataframe
            group_indices = group_sorted.index.tolist()
            
            if len(group_indices) >= sequence_length:
                # Analyze temporal patterns for this site
                site_patterns = self._analyze_site_temporal_patterns(group_sorted) if enable_pattern_analysis else None
                
                # Create overlapping sequences
                for i in range(len(group_indices) - sequence_length + 1):
                    seq_indices = group_indices[i:i + sequence_length]
                    
                    # Use the reset indices to access features and targets
                    seq_features = self.features[seq_indices]
                    seq_targets = self.targets[seq_indices[-1]]  # Predict last timestep
                    
                    # Extract temporal metadata
                    seq_metadata = self._extract_sequence_metadata(group_sorted.iloc[i:i + sequence_length])
                    
                    self.sequences.append(seq_features)
                    self.sequence_targets.append(seq_targets)
                    self.sequence_lengths.append(sequence_length)
                    self.event_metadata.append(seq_metadata)
                    
                    if site_patterns is not None:
                        # Extract pattern for this specific sequence
                        seq_pattern = site_patterns[i:i + sequence_length] if i + sequence_length <= len(site_patterns) else site_patterns[-sequence_length:]
                        self.temporal_patterns.append(seq_pattern)
                    else:
                        self.temporal_patterns.append(None)
                        
            elif len(group_indices) > 0:
                # For sites with fewer than sequence_length samples,
                # pad with the last available sample
                seq_indices = group_indices
                seq_features = self.features[seq_indices]
                
                # Pad sequence to required length
                while len(seq_features) < sequence_length:
                    seq_features = np.vstack([seq_features, seq_features[-1:]])
                
                seq_targets = self.targets[seq_indices[-1]]  # Use last available target
                seq_metadata = self._extract_sequence_metadata(group_sorted)
                
                self.sequences.append(seq_features)
                self.sequence_targets.append(seq_targets)
                self.sequence_lengths.append(len(group_indices))  # Store actual length
                self.event_metadata.append(seq_metadata)
                self.temporal_patterns.append(None)
        
        # Convert to tensors
        self.sequences = [torch.FloatTensor(seq) for seq in self.sequences]
        self.sequence_targets = torch.FloatTensor(self.sequence_targets)
        self.sequence_lengths = torch.LongTensor(self.sequence_lengths)
        
        print(f"Created {len(self.sequences)} sequences from {len(self.df.groupby('site_id'))} sites")
        print(f"Feature shape: {self.sequences[0].shape}")
        print(f"Target shape: {self.sequence_targets.shape}")
        print(f"Sequence length stats: min={min(self.sequence_lengths)}, max={max(self.sequence_lengths)}, mean={float(self.sequence_lengths.float().mean()):.1f}")
        
        if enable_pattern_analysis:
            non_none_patterns = [p for p in self.temporal_patterns if p is not None]
            print(f"Temporal patterns analyzed for {len(non_none_patterns)} sequences")

        # PART II: Enhanced temporal pattern discrimination
            if enable_pattern_analysis and len(non_none_patterns) > 0:
                from temporal_pattern_analyzer import TemporalPatternDiscriminator
                
                self.pattern_discriminator = TemporalPatternDiscriminator(
                    rapid_max_duration=72.0,
                    extended_min_duration=168.0,
                    consecutive_gap_threshold=48.0
                )
                
                # Apply advanced pattern classification
                print(" Applying advanced temporal pattern discrimination...")
                enhanced_patterns = []
                
                for site_id, group in self.df.groupby('site_id'):
                    if len(group) >= 2:
                        try:
                            pattern_labels, pattern_features = self.pattern_discriminator.classify_event_sequence(group)
                            enhanced_patterns.extend(pattern_labels.tolist())
                        except Exception as e:
                            print(f"Warning: Pattern classification failed for site {site_id}: {e}")
                            enhanced_patterns.extend([0] * len(group))
                    else:
                        enhanced_patterns.extend([0] * len(group))
                
                # Update temporal patterns with enhanced classification
                if len(enhanced_patterns) == len(self.df):
                    self.enhanced_temporal_patterns = enhanced_patterns
                    print(f" Enhanced patterns: {pd.Series(enhanced_patterns).value_counts().to_dict()}")
    
    def _analyze_site_temporal_patterns(self, site_group):
        """
        Analyze temporal patterns for events at a single site.
        
        Returns pattern classifications:
        0: Rapid/Abrupt Events
        1: Consecutive Abrupt Events  
        2: Extended Persistent Events
        3: Composite Extended Events
        """
        
        if len(site_group) < 2:
            return [0]  # Single event = rapid
        
        patterns = []
        durations = site_group['duration_hours'].values
        timestamps = pd.to_datetime(site_group['start_time']).values if 'start_time' in site_group.columns else np.arange(len(site_group))
        
        for i in range(len(site_group)):
            duration = durations[i]
            
            # Base classification on duration
            if duration <= 72:  # <= 3 days
                if i > 0:
                    # Check if consecutive with previous event
                    if isinstance(timestamps[0], pd.Timestamp):
                        time_gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # hours
                    else:
                        time_gap = timestamps[i] - timestamps[i-1]  # assume hours
                    
                    if time_gap <= 48:  # Within 48 hours
                        patterns.append(1)  # Consecutive abrupt
                    else:
                        patterns.append(0)  # Isolated rapid
                else:
                    patterns.append(0)  # First event, assume rapid
                    
            elif duration >= 168:  # >= 1 week
                # Check for internal variability (composite vs persistent)
                if i > 0 and i < len(site_group) - 1:
                    # Look at surrounding events for variability
                    prev_duration = durations[i-1]
                    next_duration = durations[i+1] if i+1 < len(durations) else duration
                    
                    duration_variance = np.var([prev_duration, duration, next_duration])
                    if duration_variance > 1000:  # High variance suggests composite
                        patterns.append(3)  # Composite extended
                    else:
                        patterns.append(2)  # Persistent extended
                else:
                    patterns.append(2)  # Extended persistent
            else:
                patterns.append(0)  # Default to rapid
        
        return patterns
    
    def _extract_sequence_metadata(self, sequence_group):
        """Extract metadata for a sequence of events."""
        
        metadata = {
            'site_lat': sequence_group['latitude'].iloc[0] if 'latitude' in sequence_group.columns else 0,
            'site_lon': sequence_group['longitude'].iloc[0] if 'longitude' in sequence_group.columns else 0,
            'sequence_start': sequence_group.index[0],
            'sequence_end': sequence_group.index[-1],
            'total_duration': sequence_group['duration_hours'].sum() if 'duration_hours' in sequence_group.columns else 0,
            'mean_intensity': sequence_group['intensity_percentile'].mean() if 'intensity_percentile' in sequence_group.columns else 0,
            'mean_extent': sequence_group['spatial_extent_meters'].mean() if 'spatial_extent_meters' in sequence_group.columns else 0,
            'duration_variance': sequence_group['duration_hours'].var() if 'duration_hours' in sequence_group.columns else 0,
            'num_events': len(sequence_group)
        }
        
        return metadata
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = {
            'features': self.sequences[idx],
            'targets': self.sequence_targets[idx],
            'sequence_length': self.sequence_lengths[idx],
            'metadata': self.event_metadata[idx]
        }
        
        if self.enable_pattern_analysis and self.temporal_patterns[idx] is not None:
            item['temporal_pattern'] = torch.LongTensor(self.temporal_patterns[idx])
        
        return item
        
class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning.
    """
    
    def __init__(self, model_class, train_loader, val_loader, device):
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization. Please install it with: pip install scikit-optimize")
        
        self.model_class = model_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Define search space
        from skopt.space import Real, Integer
        self.search_space = [
            Real(1e-5, 1e-2, name='learning_rate', prior='log-uniform'),
            Integer(4, 16, name='n_heads'),
            Integer(2, 8, name='n_layers'),
            Integer(128, 1024, name='d_model'),
            Integer(64, 512, name='liquid_hidden'),
            Real(0.0, 0.5, name='dropout'),
            Real(0.1, 2.0, name='alpha_physics'),
            Real(0.1, 1.0, name='alpha_temporal')
        ]
    
    def objective(self, params_list):
        """Objective function for Bayesian optimization."""
        
        # Convert parameter list to dictionary
        params = {
            'learning_rate': params_list[0],
            'n_heads': params_list[1],
            'n_layers': params_list[2],
            'd_model': params_list[3],
            'liquid_hidden': params_list[4],
            'dropout': params_list[5],
            'alpha_physics': params_list[6],
            'alpha_temporal': params_list[7]
        }
        
        try:
            # Create model with current hyperparameters
            model = self.model_class(
                d_model=params['d_model'],
                n_heads=params['n_heads'],
                n_layers=params['n_layers'],
                liquid_hidden=params['liquid_hidden'],
                dropout=params['dropout']
            ).to(self.device)
            
            # Create optimizer and loss function
            optimizer = AdamW(model.parameters(), lr=params['learning_rate'])
            criterion = PhysicsInformedLoss(
                alpha_physics=params['alpha_physics'],
                alpha_temporal=params['alpha_temporal']
            )
            
            # Train for a few epochs
            model.train()
            for epoch in range(3):  # Quick evaluation
                for batch in self.train_loader:
                    features = batch['features'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    
                    optimizer.zero_grad()
                    predictions, _ = model(features)
                    loss, _ = criterion(predictions, targets)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on validation set
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in self.val_loader:
                    features = batch['features'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    
                    predictions, _ = model(features)
                    loss, _ = criterion(predictions, targets)
                    val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            
            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            return val_loss
            
        except Exception as e:
            print(f"Error in optimization: {e}")
            return 1e6  # Return high loss for failed configurations
    
    def optimize(self, n_calls=20):
        """Run Bayesian optimization."""
        
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")
        
        from skopt import gp_minimize
        
        print("Starting Bayesian optimization...")
        result = gp_minimize(
            func=self.objective,
            dimensions=self.search_space,
            n_calls=n_calls,
            random_state=42,
            acq_func='EI'
        )
        
        # Extract best parameters
        best_params = {
            'learning_rate': result.x[0],
            'n_heads': result.x[1],
            'n_layers': result.x[2],
            'd_model': result.x[3],
            'liquid_hidden': result.x[4],
            'dropout': result.x[5],
            'alpha_physics': result.x[6],
            'alpha_temporal': result.x[7]
        }
        
        print(f"Best parameters: {best_params}")
        print(f"Best validation loss: {result.fun}")
        
        return best_params, result

class ModelExplainer:
    """
    Model explainability using LIME and SHAP.
    """
    
    def __init__(self, model, dataset, feature_names):
        self.model = model
        self.dataset = dataset
        self.feature_names = feature_names
        self.device = next(model.parameters()).device
        
    def explain_with_lime(self, instance_idx, num_features=10):
        """Explain prediction using LIME."""
        
        if not LIME_AVAILABLE:
            print("  LIME not available. Please install with: pip install lime")
            return None
        
        # Get instance
        sample = self.dataset[instance_idx]
        features = sample['features'].numpy()
        
        # Flatten sequence for LIME (use last timestep)
        features_flat = features[-1]
        
        # Create prediction function for LIME
        def predict_fn(X):
            predictions = []
            for x in X:
                # Reshape for model
                x_tensor = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred, _ = self.model(x_tensor)
                    predictions.append(pred.cpu().numpy()[0])
            return np.array(predictions)
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.random.randn(100, len(features_flat)),  # Dummy training data
            feature_names=self.feature_names,
            mode='regression'
        )
        
        # Generate explanation
        explanation = explainer.explain_instance(
            features_flat,
            predict_fn,
            num_features=num_features
        )
        
        return explanation
    
    def explain_with_shap(self, sample_size=100):
        """Explain model using SHAP."""
        
        if not SHAP_AVAILABLE:
            print("  SHAP not available. Please install with: pip install shap")
            return None, None
        
        # Prepare data
        X_sample = []
        for i in range(min(sample_size, len(self.dataset))):
            sample = self.dataset[i]
            features = sample['features'].numpy()
            X_sample.append(features[-1])  # Use last timestep
        
        X_sample = np.array(X_sample)
        
        # Create prediction function for SHAP
        def predict_fn(X):
            predictions = []
            for x in X:
                x_tensor = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred, _ = self.model(x_tensor)
                    predictions.append(pred.cpu().numpy()[0])
            return np.array(predictions)
        
        # Create SHAP explainer
        explainer = shap.Explainer(predict_fn, X_sample)
        
        # Calculate SHAP values
        shap_values = explainer(X_sample[:10])  # Explain first 10 samples
        
        return explainer, shap_values

class ZeroCurtainTrainer:
    """
    Complete training pipeline for the zero-curtain prediction model.
    """
    
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 test_loader,
                 device,
                 save_dir='./models'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize training components
        self.criterion = PhysicsInformedLoss(
            alpha_mse=1.0,      # ADDED
            alpha_physics=0.1,  # ENHANCE FROM 0.1 # REDUCE FROM 0.5
            alpha_temporal=0.05,  # ENHANCE FROM 0.05 # REDUCE FROM 0.3
            alpha_pattern=0.1
        )
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)
        
        # Enable mixed precision for faster training
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.use_amp = torch.cuda.is_available() or torch.backends.mps.is_available()
        
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'train_physics': [],
            'val_physics': []
        }
        
        self.best_val_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0
    
    def train_epoch(self):
        """Train for one epoch."""
        
        self.model.train()
        epoch_losses = []
        epoch_mse = []
        epoch_physics = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            # predictions, model_outputs = self.model(features)
            
            if self.use_amp:
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    predictions, model_outputs = self.model(features)
                    loss, loss_components = self.criterion(predictions, targets)
            else:
                predictions, model_outputs = self.model(features)
                loss, loss_components = self.criterion(predictions, targets)
            
            # Calculate loss with NaN checking
            # loss, loss_components = self.criterion(predictions, targets)

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf detected at batch {len(epoch_losses)}! Skipping batch.")
                print(f"Predictions range: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
                print(f"Targets range: [{targets.min().item():.3f}, {targets.max().item():.3f}]")
                continue

            # Backward pass
            loss.backward()

            # Check gradients for NaN
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm):
                print("NaN gradients detected! Skipping optimizer step.")
                self.optimizer.zero_grad()
                continue

            self.optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            epoch_mse.append(loss_components['mse_loss'])
            epoch_physics.append(loss_components['physics_loss'])
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'MSE': f"{loss_components['mse_loss']:.4f}",
                'Physics': f"{loss_components['physics_loss']:.4f}"
            })
        
        return {
            'loss': np.mean(epoch_losses),
            'mse': np.mean(epoch_mse),
            'physics': np.mean(epoch_physics)
        }
    
    def validate_epoch(self):
        """Validate for one epoch."""
        
        self.model.eval()
        epoch_losses = []
        epoch_mse = []
        epoch_physics = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                predictions, _ = self.model(
                    features,
                    batch_metadata=batch.get('metadata', None)
                )
                loss, loss_components = self.criterion(predictions, targets)
                
                epoch_losses.append(loss.item())
                epoch_mse.append(loss_components['mse_loss'])
                epoch_physics.append(loss_components['physics_loss'])
        
        return {
            'loss': np.mean(epoch_losses),
            'mse': np.mean(epoch_mse),
            'physics': np.mean(epoch_physics)
        }
    
    def train(self, epochs=100):
        """Complete training loop."""
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['train_physics'].append(train_metrics['physics'])
            self.history['val_physics'].append(val_metrics['physics'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"Train MSE: {train_metrics['mse']:.4f} | Val MSE: {val_metrics['mse']:.4f}")
            print(f"Train Physics: {train_metrics['physics']:.4f} | Val Physics: {val_metrics['physics']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_model('best_model.pth')
                self.patience_counter = 0
                print(" New best model saved!")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pth')
        
        print("Training completed!")
        
        # Load best model
        self.load_model('best_model.pth')
        
        # Final evaluation
        test_metrics = self.evaluate_test_set()
        print(f"\nFinal Test Metrics:")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test MSE: {test_metrics['mse']:.4f}")
        print(f"Test R²: {test_metrics['r2']:.4f}")
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # MSE
        axes[0, 1].plot(self.history['train_mse'], label='Train')
        axes[0, 1].plot(self.history['val_mse'], label='Validation')
        axes[0, 1].set_title('MSE Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].legend()
        
        # Physics Loss
        axes[1, 0].plot(self.history['train_physics'], label='Train')
        axes[1, 0].plot(self.history['val_physics'], label='Validation')
        axes[1, 0].set_title('Physics Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Physics Loss')
        axes[1, 0].legend()
        
        # Learning Rate
        lr_values = [self.scheduler.get_last_lr()[0] for _ in range(len(self.history['train_loss']))]
        axes[1, 1].plot(lr_values)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=300)
        plt.show()
    
    def evaluate_test_set(self):
        """Evaluate on test set."""
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        epoch_losses = []
        epoch_mse = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                predictions, _ = self.model(
                    features,
                    batch_metadata=batch.get('metadata', None)
                )
                loss, loss_components = self.criterion(predictions, targets)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                epoch_losses.append(loss.item())
                epoch_mse.append(loss_components['mse_loss'])
        
        # Concatenate all predictions and targets
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # Calculate R² for each target
        r2_scores = []
        for i in range(all_targets.shape[1]):
            r2 = r2_score(all_targets[:, i], all_predictions[:, i])
            r2_scores.append(r2)
        
        return {
            'loss': np.mean(epoch_losses),
            'mse': np.mean(epoch_mse),
            'r2': np.mean(r2_scores),
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def save_model(self, filename):
        """Save model state."""
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, filepath)
    
    def load_model(self, filename):
        """Load model state."""
        
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']

class TeacherForcingTrainer(ZeroCurtainTrainer):
    """
    Enhanced trainer implementing teacher forcing with PINSZC ground truth.
    Extends ZeroCurtainTrainer with curriculum learning and scheduled sampling.
    """
    
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 test_loader,
                 device,
                 save_dir='./models',
                 teacher_forcing_ratio=1.0,
                 curriculum_schedule='exponential',
                 pinszc_ground_truth=None):
        
        super().__init__(model, train_loader, val_loader, test_loader, device, save_dir)
        
        # Teacher forcing configuration
        self.teacher_forcing_ratio = teacher_forcing_ratio  # Start with 100% ground truth
        self.initial_tf_ratio = teacher_forcing_ratio
        self.curriculum_schedule = curriculum_schedule
        self.pinszc_ground_truth = pinszc_ground_truth  # Part I PINSZC dataframe
        
        # OPTIMIZATION: Pre-index PINSZC by site coordinates for faster lookup
        if self.pinszc_ground_truth is not None:
            print(" Pre-indexing PINSZC ground truth for fast lookup...")
            self.pinszc_site_index = {}
            
            for idx, row in self.pinszc_ground_truth.iterrows():
                # Round coordinates to 2 decimal places for grouping
                lat_key = round(row['latitude'], 2)
                lon_key = round(row['longitude'], 2)
                site_key = (lat_key, lon_key)
                
                if site_key not in self.pinszc_site_index:
                    self.pinszc_site_index[site_key] = []
                
                self.pinszc_site_index[site_key].append({
                    'intensity': row['intensity_percentile'],
                    'duration': row['duration_hours'],
                    'extent': row['spatial_extent_meters']
                })
            
            print(f" Indexed {len(self.pinszc_site_index)} unique sites from PINSZC")
        else:
            self.pinszc_site_index = None

        # Curriculum learning parameters
        self.tf_decay_rate = 0.95  # Exponential decay per epoch
        self.tf_min_ratio = 0.1    # Minimum teacher forcing ratio
        
        # Track teacher forcing effectiveness
        self.tf_history = {
            'ratio': [],
            'accuracy_with_tf': [],
            'accuracy_without_tf': []
        }
    
    def update_teacher_forcing_ratio(self, epoch):
        """
        Update teacher forcing ratio based on curriculum schedule.
        
        Schedules:
        - 'exponential': Exponential decay from initial ratio
        - 'linear': Linear decay to minimum ratio
        - 'inverse_sigmoid': Slow start, rapid middle, slow end decay
        """
        
        if self.curriculum_schedule == 'exponential':
            self.teacher_forcing_ratio = max(
                self.tf_min_ratio,
                self.initial_tf_ratio * (self.tf_decay_rate ** epoch)
            )
        
        elif self.curriculum_schedule == 'linear':
            decay_per_epoch = (self.initial_tf_ratio - self.tf_min_ratio) / self.epochs
            self.teacher_forcing_ratio = max(
                self.tf_min_ratio,
                self.initial_tf_ratio - decay_per_epoch * epoch
            )
        
        elif self.curriculum_schedule == 'inverse_sigmoid':
            k = 10 / self.epochs  # Steepness parameter
            self.teacher_forcing_ratio = self.tf_min_ratio + (
                (self.initial_tf_ratio - self.tf_min_ratio) / 
                (1 + np.exp(k * (epoch - self.epochs / 2)))
            )
        
        self.tf_history['ratio'].append(self.teacher_forcing_ratio)
        
        return self.teacher_forcing_ratio
    
    def train_epoch_with_teacher_forcing(self):
        """
        Enhanced training epoch with teacher forcing and PINSZC guidance.
        """
        
        self.model.train()
        epoch_losses = []
        epoch_mse = []
        epoch_physics = []
        epoch_tf_accuracy = []
        
        pbar = tqdm(self.train_loader, desc=f"Training (TF ratio: {self.teacher_forcing_ratio:.3f})")
        
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)

            # Get PINSZC ground truth if available
            if self.pinszc_ground_truth is not None and 'metadata' in batch:
                # DEBUG: Verify metadata structure (only first batch)
                if batch_idx == 0:
                    print(f"DEBUG: metadata type: {type(batch['metadata'])}")
                    print(f"DEBUG: metadata length: {len(batch['metadata'])}")
                    if len(batch['metadata']) > 0:
                        print(f"DEBUG: first metadata item type: {type(batch['metadata'][0])}")
                        print(f"DEBUG: first metadata item: {batch['metadata'][0]}")
                
                pinszc_targets = self._get_pinszc_ground_truth(batch['metadata'])
            else:
                pinszc_targets = targets
            
            self.optimizer.zero_grad()
            
            # Determine whether to use teacher forcing for this batch
            use_teacher_forcing = np.random.random() < self.teacher_forcing_ratio
            
            if use_teacher_forcing:
                # Use PINSZC ground truth to guide predictions
                # Pass metadata to enable GeoCryoAI graph processing
                predictions, model_outputs = self.model(
                    features, 
                    batch_metadata=batch.get('metadata', None)
                )
                
                # Blend model predictions with PINSZC ground truth
                alpha = self.teacher_forcing_ratio
                guided_predictions = alpha * pinszc_targets + (1 - alpha) * predictions
                
                # Calculate loss with guided predictions
                loss, loss_components = self.criterion(guided_predictions, targets)
                
            else:
                # Standard forward pass without teacher forcing
                predictions, model_outputs = self.model(
                    features,
                    batch_metadata=batch.get('metadata', None)
                )
                loss, loss_components = self.criterion(predictions, targets)
            
            # Backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf detected! Skipping batch {batch_idx}")
                continue
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            if torch.isnan(grad_norm):
                print("NaN gradients! Skipping optimizer step.")
                self.optimizer.zero_grad()
                continue
            
            self.optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            epoch_mse.append(loss_components['mse_loss'])
            epoch_physics.append(loss_components['physics_loss'])
            
            # Calculate accuracy of teacher forcing guidance
            if use_teacher_forcing:
                with torch.no_grad():
                    tf_accuracy = 1.0 - F.mse_loss(guided_predictions, targets).item()
                    epoch_tf_accuracy.append(tf_accuracy)
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'MSE': f"{loss_components['mse_loss']:.4f}",
                'TF': 'Yes' if use_teacher_forcing else 'No'
            })
        
        # Track teacher forcing effectiveness
        if epoch_tf_accuracy:
            self.tf_history['accuracy_with_tf'].append(np.mean(epoch_tf_accuracy))
        
        return {
            'loss': np.mean(epoch_losses),
            'mse': np.mean(epoch_mse),
            'physics': np.mean(epoch_physics),
            'tf_accuracy': np.mean(epoch_tf_accuracy) if epoch_tf_accuracy else 0.0
        }
    
    def _get_pinszc_ground_truth(self, metadata_batch):
        """
        Retrieve PINSZC ground truth for batch based on metadata.
        
        Args:
            metadata_batch: List of metadata dicts with site/temporal info
        
        Returns:
            torch.Tensor: Ground truth targets from PINSZC dataframe
        """
        
        pinszc_targets = []
        
        # Ensure metadata_batch is a list
        if not isinstance(metadata_batch, list):
            print(f"Warning: metadata_batch is not a list, type: {type(metadata_batch)}")
            # Try to convert to list
            try:
                metadata_batch = list(metadata_batch)
            except:
                # Fallback: return zeros
                return torch.zeros((1, 3), dtype=torch.float32).to(self.device)
        
        # Process each metadata dict in the batch
        for metadata in metadata_batch:
            # Verify metadata is a dict
            if not isinstance(metadata, dict):
                print(f"Warning: metadata item is not a dict, type: {type(metadata)}")
                pinszc_targets.append([0.0, 0.0, 0.0])
                continue
            
            # Extract site coordinates - they should be scalar values
            site_lat = metadata.get('site_lat', 0.0)
            site_lon = metadata.get('site_lon', 0.0)
            
            # DEFENSIVE: Convert to float if needed
            if isinstance(site_lat, torch.Tensor):
                if site_lat.numel() == 1:
                    site_lat = site_lat.item()
                else:
                    print(f"Warning: site_lat is a tensor with {site_lat.numel()} elements - using first element")
                    site_lat = site_lat[0].item() if site_lat.numel() > 0 else 0.0
            
            if isinstance(site_lon, torch.Tensor):
                if site_lon.numel() == 1:
                    site_lon = site_lon.item()
                else:
                    print(f"Warning: site_lon is a tensor with {site_lon.numel()} elements - using first element")
                    site_lon = site_lon[0].item() if site_lon.numel() > 0 else 0.0
            
            # Ensure they're Python floats
            try:
                site_lat = float(site_lat)
                site_lon = float(site_lon)
            except (TypeError, ValueError) as e:
                print(f"Warning: Could not convert coordinates to float: {e}")
                pinszc_targets.append([0.0, 0.0, 0.0])
                continue
            
            # Query PINSZC using pre-computed index
            if hasattr(self, 'pinszc_site_index') and self.pinszc_site_index is not None:
                try:
                    # Round to match index precision
                    lat_key = round(site_lat, 2)
                    lon_key = round(site_lon, 2)
                    site_key = (lat_key, lon_key)
                    
                    if site_key in self.pinszc_site_index:
                        # Get all matching events for this site
                        matching_events = self.pinszc_site_index[site_key]
                        
                        # Calculate mean values
                        gt_intensity = np.mean([e['intensity'] for e in matching_events])
                        gt_duration = np.mean([e['duration'] for e in matching_events])
                        gt_extent = np.mean([e['extent'] for e in matching_events])
                        
                        pinszc_targets.append([gt_intensity, gt_duration, gt_extent])
                    else:
                        # No matching site in index - use zeros
                        pinszc_targets.append([0.0, 0.0, 0.0])
                        
                except Exception as e:
                    print(f"Warning: Error querying PINSZC index: {e}")
                    pinszc_targets.append([0.0, 0.0, 0.0])
            else:
                # No index available - use zeros
                pinszc_targets.append([0.0, 0.0, 0.0])
        
        # Convert to tensor
        if len(pinszc_targets) == 0:
            return torch.zeros((1, 3), dtype=torch.float32).to(self.device)
        
        return torch.FloatTensor(pinszc_targets).to(self.device)
    
    def train(self, epochs=100):
        """
        Training loop with teacher forcing curriculum learning.
        """
        
        print(f"Starting Teacher Forcing Training for {epochs} epochs...")
        print(f"Initial TF Ratio: {self.teacher_forcing_ratio:.3f}")
        print(f"Curriculum Schedule: {self.curriculum_schedule}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Update teacher forcing ratio
            current_tf_ratio = self.update_teacher_forcing_ratio(epoch)
            print(f"Teacher Forcing Ratio: {current_tf_ratio:.3f}")
            
            # Train with teacher forcing
            train_metrics = self.train_epoch_with_teacher_forcing()
            
            # Validate (no teacher forcing during validation)
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['train_physics'].append(train_metrics['physics'])
            self.history['val_physics'].append(val_metrics['physics'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"TF Accuracy: {train_metrics['tf_accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_model('best_model_tf.pth')
                self.patience_counter = 0
                print(" New best model with teacher forcing saved!")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        
        print("Teacher Forcing Training completed!")
        
        # Plot teacher forcing curriculum
        self.plot_teacher_forcing_curriculum()
        
        return self.history
    
    def plot_teacher_forcing_curriculum(self):
        """
        Visualize teacher forcing curriculum schedule.
        """
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Teacher forcing ratio decay
        axes[0].plot(self.tf_history['ratio'], linewidth=2)
        axes[0].set_title('Teacher Forcing Ratio Decay')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('TF Ratio')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=self.tf_min_ratio, color='r', linestyle='--', 
                       label=f'Min Ratio: {self.tf_min_ratio}')
        axes[0].legend()
        
        # Teacher forcing accuracy
        if self.tf_history['accuracy_with_tf']:
            axes[1].plot(self.tf_history['accuracy_with_tf'], 
                        label='Accuracy with TF', linewidth=2)
            axes[1].set_title('Teacher Forcing Effectiveness')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'teacher_forcing_curriculum.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

def load_and_preprocess_data(parquet_file):
    """Load and preprocess the zero-curtain dataset with comprehensive outlier handling."""
    
    print("Loading zero-curtain dataset...")
    
    # Check if file exists
    if not os.path.exists(parquet_file):
        print(f" File not found: {parquet_file}")
        print("Looking for alternative file paths...")
        
        # Try different possible paths
        possible_paths = [
            parquet_file,
            os.path.join(os.getcwd(), "zero_curtain_enhanced_cryogrid_physics_dataset.parquet"),
            os.path.join(os.path.dirname(__file__), "zero_curtain_enhanced_cryogrid_physics_dataset.parquet"),
            "/Users/bradleygay/part1_output/zero_curtain_enhanced_cryogrid_physics_dataset.parquet",
        ]
        
        # Look for any parquet files in current directory
        current_dir_parquets = [f for f in os.listdir('.') if f.endswith('.parquet')]
        if current_dir_parquets:
            print(f"Found parquet files in current directory: {current_dir_parquets}")
            possible_paths.extend([os.path.join('.', f) for f in current_dir_parquets])
        
        # Try each path
        found_file = None
        for path in possible_paths:
            if os.path.exists(path):
                found_file = path
                break
        
        if found_file:
            print(f" Using file: {found_file}")
            parquet_file = found_file
        else:
            print(" No valid parquet file found!")
            print("Please ensure you have the zero-curtain dataset file in one of these locations:")
            for path in possible_paths:
                print(f"  - {path}")
            
            # Create sample data for demonstration
            # print(" Creating sample dataset for demonstration...")
            # return create_sample_dataset()
    
    try:
        # Load with Dask
        df = dd.read_parquet(parquet_file)
        
        # Convert to pandas for processing (sample if too large)
        print("Converting to pandas DataFrame...")
        df_pd = df.compute()
        
    except Exception as e:
        print(f"Error loading with Dask: {e}")
        print("Trying with pandas...")
        try:
            df_pd = pd.read_parquet(parquet_file)
        except Exception as e2:
            print(f"Error loading with pandas: {e2}")
            # print("Creating sample dataset for demonstration...")
            # return create_sample_dataset()
    
    print(f"Dataset shape before cleaning: {df_pd.shape}")
    print(f"Memory usage: {df_pd.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # ============================================================================
    # COMPREHENSIVE OUTLIER DETECTION AND REMOVAL
    # ============================================================================
    
    print("\n OUTLIER ANALYSIS AND CLEANING")
    print("=" * 60)
    
    # Analyze target variables before cleaning
    target_vars = ['duration_hours', 'intensity_percentile', 'spatial_extent_meters']
    
    print(" Target variable statistics BEFORE cleaning:")
    for var in target_vars:
        if var in df_pd.columns:
            stats = {
                'min': df_pd[var].min(),
                'max': df_pd[var].max(),
                'mean': df_pd[var].mean(),
                'median': df_pd[var].median(),
                'std': df_pd[var].std(),
                'q1': df_pd[var].quantile(0.1),
                'q99': df_pd[var].quantile(0.99)
            }
            print(f"  {var}:")
            print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"    Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}")
            print(f"    Q1: {stats['q1']:.3f}, Q99: {stats['q99']:.3f}")
    
    initial_count = len(df_pd)
    
    # 1. DURATION OUTLIER REMOVAL
    print("\n Duration outlier analysis:")
    duration_q1 = df_pd['duration_hours'].quantile(0.1)
    duration_q99 = df_pd['duration_hours'].quantile(0.99)
    duration_iqr = duration_q99 - duration_q1
    duration_median = df_pd['duration_hours'].median()
    
    # Physics-informed duration limits based on Arctic zero-curtain literature
    # Kane et al. (1991), Outcalt et al. (1990): typical range 6 hours to 6 months
    DURATION_MIN = 6.0      # Minimum 6 hours (very short events)
    DURATION_MAX = 4500.0   # Maximum ~6 months (187.5 days * 24 hours)
    
    # Alternative: IQR-based outlier detection (more permissive)
    duration_lower_iqr = duration_q1 - 5 * duration_iqr  # 5*IQR instead of 1.5*IQR
    duration_upper_iqr = duration_q99 + 5 * duration_iqr
    
    # Use the more permissive of physics-based or IQR-based limits
    duration_min_limit = min(DURATION_MIN, duration_lower_iqr) if duration_lower_iqr > 0 else DURATION_MIN
    duration_max_limit = DURATION_MAX
    
    print(f"  Physics limits: [{DURATION_MIN}, {DURATION_MAX}] hours")
    print(f"  IQR limits (5×IQR): [{duration_lower_iqr:.1f}, {duration_upper_iqr:.1f}] hours")
    print(f"  Applied limits: [{duration_min_limit:.1f}, {duration_max_limit:.1f}] hours")
    
    duration_mask = (df_pd['duration_hours'] >= duration_min_limit) & (df_pd['duration_hours'] <= duration_max_limit)
    duration_outliers = (~duration_mask).sum()
    print(f"  Removing {duration_outliers} duration outliers ({duration_outliers/len(df_pd)*100:.2f}%)")
    
    # 2. INTENSITY OUTLIER REMOVAL
    print("\n  Intensity outlier analysis:")
    # Intensity should be [0, 1] by definition, but check for anomalies
    intensity_mask = (df_pd['intensity_percentile'] >= 0.0) & (df_pd['intensity_percentile'] <= 1.0)
    intensity_outliers = (~intensity_mask).sum()
    print(f"  Intensity range: [{df_pd['intensity_percentile'].min():.3f}, {df_pd['intensity_percentile'].max():.3f}]")
    print(f"  Removing {intensity_outliers} intensity outliers (outside [0,1])")
    
    # 3. SPATIAL EXTENT OUTLIER REMOVAL
    print("\n Spatial extent outlier analysis:")
    extent_q1 = df_pd['spatial_extent_meters'].quantile(0.1)
    extent_q99 = df_pd['spatial_extent_meters'].quantile(0.99)
    extent_iqr = extent_q99 - extent_q1
    
    # Physics-informed spatial extent limits
    # Zero-curtain penetration depth typically 0.1m - 3.0m (Outcalt et al., 1990)
    EXTENT_MIN = 0.01   # 1cm minimum
    EXTENT_MAX = 5.0    # 5m maximum (already capped in Part 1)
    
    # IQR-based detection (more permissive)
    extent_lower_iqr = extent_q1 - 5 * extent_iqr
    extent_upper_iqr = extent_q99 + 5 * extent_iqr
    
    extent_min_limit = max(EXTENT_MIN, extent_lower_iqr) if extent_lower_iqr > 0 else EXTENT_MIN
    extent_max_limit = min(EXTENT_MAX, extent_upper_iqr) if extent_upper_iqr < EXTENT_MAX else EXTENT_MAX
    
    print(f"  Physics limits: [{EXTENT_MIN}, {EXTENT_MAX}] meters")
    print(f"  IQR limits (5×IQR): [{extent_lower_iqr:.3f}, {extent_upper_iqr:.3f}] meters")
    print(f"  Applied limits: [{extent_min_limit:.3f}, {extent_max_limit:.3f}] meters")
    
    extent_mask = (df_pd['spatial_extent_meters'] >= extent_min_limit) & (df_pd['spatial_extent_meters'] <= extent_max_limit)
    extent_outliers = (~extent_mask).sum()
    print(f"  Removing {extent_outliers} spatial extent outliers ({extent_outliers/len(df_pd)*100:.2f}%)")
    
    # 4. ADDITIONAL PHYSICS-BASED OUTLIER DETECTION
    print("\n Additional physics variable outlier analysis:")
    
    additional_masks = []
    
    # Temperature outliers (should be reasonable for Arctic conditions)
    if 'mean_temperature' in df_pd.columns:
        temp_mask = (df_pd['mean_temperature'] >= -50) & (df_pd['mean_temperature'] <= 50)
        temp_outliers = (~temp_mask).sum()
        print(f"  Temperature outliers: {temp_outliers} (outside [-50°C, 50°C])")
        additional_masks.append(temp_mask)
    
    # Temperature variance outliers
    if 'temperature_variance' in df_pd.columns:
        temp_var_q99 = df_pd['temperature_variance'].quantile(0.99)
        temp_var_mask = (df_pd['temperature_variance'] >= 0) & (df_pd['temperature_variance'] <= temp_var_q99)
        temp_var_outliers = (~temp_var_mask).sum()
        print(f"  Temperature variance outliers: {temp_var_outliers} (above 99th percentile: {temp_var_q99:.2f})")
        additional_masks.append(temp_var_mask)
    
    # Permafrost probability should be [0, 1]
    if 'permafrost_probability' in df_pd.columns:
        pf_mask = (df_pd['permafrost_probability'].isna()) | ((df_pd['permafrost_probability'] >= 0) & (df_pd['permafrost_probability'] <= 1))
        pf_outliers = (~pf_mask).sum()
        print(f"  Permafrost probability outliers: {pf_outliers} (outside [0,1])")
        additional_masks.append(pf_mask)
    
    # Thermal conductivity outliers (reasonable physical bounds)
    if 'cryogrid_thermal_conductivity' in df_pd.columns:
        tc_mask = (df_pd['cryogrid_thermal_conductivity'].isna()) | ((df_pd['cryogrid_thermal_conductivity'] >= 0.1) & (df_pd['cryogrid_thermal_conductivity'] <= 10.0))
        tc_outliers = (~tc_mask).sum()
        print(f"  Thermal conductivity outliers: {tc_outliers} (outside [0.1, 10.0] W/m/K)")
        additional_masks.append(tc_mask)
    
    # 5. COMBINE ALL MASKS
    print(f"\n Applying combined outlier filters...")
    combined_mask = duration_mask & intensity_mask & extent_mask
    
    # Add additional masks
    for mask in additional_masks:
        combined_mask = combined_mask & mask
    
    # Apply the mask
    df_cleaned = df_pd[combined_mask].copy()
    
    removed_count = initial_count - len(df_cleaned)
    removal_percent = (removed_count / initial_count) * 100
    
    print(f" OUTLIER REMOVAL SUMMARY:")
    print(f"  Initial samples: {initial_count:,}")
    print(f"  Samples removed: {removed_count:,} ({removal_percent:.2f}%)")
    print(f"  Final samples: {len(df_cleaned):,}")
    
    # Analyze target variables after cleaning
    print(f"\n Target variable statistics AFTER cleaning:")
    for var in target_vars:
        if var in df_cleaned.columns:
            stats = {
                'min': df_cleaned[var].min(),
                'max': df_cleaned[var].max(),
                'mean': df_cleaned[var].mean(),
                'median': df_cleaned[var].median(),
                'std': df_cleaned[var].std(),
                'q1': df_cleaned[var].quantile(0.1),
                'q99': df_cleaned[var].quantile(0.99)
            }
            print(f"  {var}:")
            print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"    Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}")
            print(f"    Q1: {stats['q1']:.3f}, Q99: {stats['q99']:.3f}")
    
    # Use cleaned dataframe
    df_pd = df_cleaned
    
    # ============================================================================
    # CONTINUE WITH REGULAR PREPROCESSING
    # ============================================================================
    
    # Handle datetime columns
    if 'start_time' in df_pd.columns:
        df_pd['start_time'] = pd.to_datetime(df_pd['start_time'])
    if 'end_time' in df_pd.columns:
        df_pd['end_time'] = pd.to_datetime(df_pd['end_time'])
    
    # Create additional temporal features
    if 'start_time' in df_pd.columns:
        df_pd['year'] = df_pd['start_time'].dt.year
        df_pd['month'] = df_pd['start_time'].dt.month
        df_pd['day_of_year'] = df_pd['start_time'].dt.dayofyear
        df_pd['season'] = df_pd['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
    
    # Encode categorical variables
    if 'depth_zone' in df_pd.columns:
        df_pd['depth_zone_encoded'] = df_pd['depth_zone'].map({
            'surface': 0, 'shallow': 1, 'intermediate': 2, 'deep': 3, 'very_deep': 4
        })
        df_pd['depth_zone_encoded'] = df_pd['depth_zone_encoded'].fillna(2)  # Default to intermediate
    
    if 'permafrost_zone' in df_pd.columns:
        df_pd['permafrost_zone_encoded'] = df_pd['permafrost_zone'].map({
            'continuous': 4, 'discontinuous': 3, 'sporadic': 2, 'isolated': 1
        })
        df_pd['permafrost_zone_encoded'] = df_pd['permafrost_zone_encoded'].fillna(0)  # Default to no permafrost
    
    print(f"\n Data preprocessing completed!")
    print(f"Final dataset shape: {df_pd.shape}")
    
    return df_pd

def custom_collate_fn(batch):
    """
    Custom collate function to properly handle metadata batching.
    Prevents PyTorch from converting metadata dicts into tensors.
    """
    # Extract individual components
    features = torch.stack([item['features'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    sequence_lengths = torch.stack([item['sequence_length'] for item in batch])
    
    # CRITICAL: Keep metadata as list of dicts (don't let default collate mess it up)
    metadata = [item['metadata'] for item in batch]
    
    # Handle temporal patterns if present
    has_patterns = 'temporal_pattern' in batch[0]
    
    result = {
        'features': features,
        'targets': targets,
        'sequence_length': sequence_lengths,
        'metadata': metadata  # List of dicts, NOT a batched tensor
    }
    
    if has_patterns:
        temporal_patterns = [item['temporal_pattern'] for item in batch if 'temporal_pattern' in item]
        if temporal_patterns:
            result['temporal_pattern'] = temporal_patterns
    
    return result

def create_data_loaders(df, test_size=0.2, val_size=0.1, batch_size=32, sequence_length=10, enable_pattern_analysis=True):
    """Create train/val/test data loaders with FIXED sequence length handling."""
    
    print("Creating data loaders...")
    
    # Define feature columns
    feature_columns = [
        'mean_temperature', 'temperature_variance', 'permafrost_probability',
        'phase_change_energy', 'freeze_penetration_depth', 'thermal_diffusivity',
        'snow_insulation_factor', 'cryogrid_thermal_conductivity',
        'cryogrid_heat_capacity', 'cryogrid_enthalpy_stability',
        'surface_energy_balance', 'lateral_thermal_effects',
        'van_genuchten_alpha', 'van_genuchten_n', 'latitude', 'longitude',
        'year', 'month', 'day_of_year', 'depth_zone_encoded', 'permafrost_zone_encoded'
    ]
    
    # Filter columns that exist in dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"Using {len(available_features)} features")
    
    # CRITICAL FIX: Remove sites with insufficient samples BEFORE splitting
    sites = df['latitude'].astype(str) + '_' + df['longitude'].astype(str)
    site_counts = sites.value_counts()
    
    # Require at least sequence_length + 5 samples per site
    min_required = sequence_length + 5
    valid_sites = site_counts[site_counts >= min_required].index
    
    print(f"Original sites: {len(site_counts)}")
    print(f"Sites with ≥{min_required} samples: {len(valid_sites)}")
    
    # Filter to valid sites only
    df_filtered = df[sites.isin(valid_sites)].copy()
    sites_filtered = sites[sites.isin(valid_sites)]
    
    print(f"Samples before filtering: {len(df)}")
    print(f"Samples after filtering: {len(df_filtered)}")
    
    # Split filtered sites
    unique_sites = sites_filtered.unique()
    train_sites, temp_sites = train_test_split(unique_sites, test_size=test_size + val_size, random_state=42)
    val_sites, test_sites = train_test_split(temp_sites, test_size=test_size/(test_size + val_size), random_state=42)
    
    # Create site-based splits
    train_mask = sites_filtered.isin(train_sites)
    val_mask = sites_filtered.isin(val_sites)
    test_mask = sites_filtered.isin(test_sites)
    
    train_df = df_filtered[train_mask].copy()
    val_df = df_filtered[val_mask].copy()
    test_df = df_filtered[test_mask].copy()
    
    print(f"Train sites: {len(train_sites)}, Val sites: {len(val_sites)}, Test sites: {len(test_sites)}")
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    # REMOVE THIS BROKEN LOGIC ENTIRELY:
    # min_samples_per_site = min(...)
    # effective_sequence_length = min(sequence_length, min_samples_per_site)
    
    # USE FIXED SEQUENCE LENGTH:
    effective_sequence_length = sequence_length
    print(f"Using sequence length: {effective_sequence_length}")
    
    # Create datasets
    train_dataset = ZeroCurtainDataset(
        train_df,
        feature_columns=available_features,
        sequence_length=effective_sequence_length,
        enable_pattern_analysis=enable_pattern_analysis
    )
    
    val_dataset = ZeroCurtainDataset(
        val_df,
        feature_columns=available_features,
        sequence_length=effective_sequence_length,
        scaler=train_dataset.scaler,
        target_scaler=train_dataset.target_scaler,
        enable_pattern_analysis=enable_pattern_analysis
    )
    
    test_dataset = ZeroCurtainDataset(
        test_df,
        feature_columns=available_features,
        sequence_length=effective_sequence_length,
        scaler=train_dataset.scaler,
        target_scaler=train_dataset.target_scaler,
        enable_pattern_analysis=enable_pattern_analysis
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=0, pin_memory=False, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=0, pin_memory=False, collate_fn=custom_collate_fn)
    
    print("Data loaders created successfully!")
    
    return train_loader, val_loader, test_loader, train_dataset

def calculate_optimal_sequence_length(df, target_temporal_coverage='seasonal'):
    """
    Calculate sequence length based on ACTUAL temporal patterns in your data.
    """
    
    # Analyze your actual temporal spacing
    if 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'])
        
        # Calculate typical time gaps between events at each site
        temporal_gaps = []
        for site_id, group in df.groupby(['latitude', 'longitude']):
            if len(group) > 1:
                group_sorted = group.sort_values('start_time')
                gaps = group_sorted['start_time'].diff().dt.total_seconds() / 3600  # hours
                temporal_gaps.extend(gaps.dropna().tolist())
        
        median_gap = np.median(temporal_gaps)
        print(f"Median time gap between events: {median_gap:.1f} hours")
        
        # Define temporal coverage targets
        if target_temporal_coverage == 'seasonal':
            target_hours = 2160  # 3 months
        elif target_temporal_coverage == 'annual':
            target_hours = 8760  # 1 year
        elif target_temporal_coverage == 'extended_event':
            target_hours = 720   # 1 month
        else:
            target_hours = 168   # 1 week minimum
        
        # Calculate required sequence length
        required_sequence_length = int(target_hours / median_gap)
        
        print(f"Target temporal coverage: {target_hours} hours")
        print(f"Required sequence length: {required_sequence_length}")
        
        # REMOVE ARBITRARY BOUNDS - USE COMPUTATIONAL LIMITS ONLY
        # max_reasonable = 200  # Based on memory/computation, not arbitrary
        # min_reasonable = 3    # Minimum for any temporal pattern
        
        return required_sequence_length
    
    else:
        raise ValueError(
            " NO TEMPORAL INFORMATION FOUND!\n"
            "Zero-curtain temporal modeling requires time-ordered data.\n"
            "Available columns: {}\n"
            "Need: 'start_time' or equivalent temporal column".format(list(df.columns))
        )

def main(config_path='../configs/part2_config.yaml'):
    """
    Main training pipeline for Part II: GeoCryoAI Teacher Forcing.
    
    Args:
        config_path: Path to configuration file
    """
    
    print(" Zero-Curtain Spatiotemporal ML Framework - Part II")
    print("=" * 60)
    print(" GeoCryoAI Integration with Teacher Forcing")
    print("=" * 60)
    
    # Load configuration
    from config_loader import load_config
    
    try:
        config = load_config(config_path)
        print(f" Configuration loaded from: {config_path}")
    except FileNotFoundError:
        print(f"  Configuration file not found: {config_path}")
        print("Using default configuration...")
        
        # Fallback to hardcoded config
        config = {
            'data': {
                'parquet_file': "./outputs/zero_curtain_enhanced_cryogrid_physics_dataset.parquet",
                'batch_size': 128,
                'temporal_coverage': 'seasonal'
            },
            'training': {
                'epochs': 25,
                'learning_rate': 1e-4
            },
            'model': {
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'liquid_hidden': 64,
                'dropout': 0.1,
                'geocryoai': {
                    'enabled': True,
                    'spatial_threshold_km': 50.0
                },
                'pattern_analysis': {
                    'enabled': True
                }
            },
            'teacher_forcing': {
                'enabled': True,
                'initial_ratio': 0.9,
                'curriculum_schedule': 'exponential'
            },
            'explainability': {
                'enabled': True
            },
            'output': {
                'save_dir': './outputs',
                'models_dir': './outputs/models'
            }
        }
    
    # CHECK FOR EXISTING MODEL FIRST - BEFORE ANY HEAVY PROCESSING
    best_model_path = config['output']['models_dir'] + '/best_model.pth'
    if os.path.exists(best_model_path):
        print(f" Found existing model at {best_model_path}")
        print(" Skipping data loading and preprocessing - using cached model")
        
        # Load minimal data just for explainability
        print("Loading minimal dataset for explainability...")
        df = load_and_preprocess_data(config['data']['parquet_file'])
        
        optimal_sequence_length = calculate_optimal_sequence_length(
            df, 
            target_temporal_coverage=config['data'].get('temporal_coverage', 'seasonal')
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader, train_dataset = create_data_loaders(
            df,
            batch_size=config['data']['batch_size'],
            sequence_length=optimal_sequence_length,
            enable_pattern_analysis=config['model']['pattern_analysis']['enabled']
        )
        
        # Initialize model for loading
        model = ZeroCurtainHybridModel(
            input_features=len(train_dataset.feature_columns),
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            n_layers=config['model']['n_layers'],
            liquid_hidden=config['model']['liquid_hidden'],
            dropout=config['model']['dropout']
        )
        
        # PART II: Initialize Teacher Forcing Trainer with PINSZC ground truth
        print("\n PART II: Initializing Teacher Forcing Trainer with PINSZC")
        
        # Load PINSZC ground truth from Part I
        pinszc_path = config['data'].get('pinszc_ground_truth', 
                                         './outputs/zero_curtain_enhanced_cryogrid_physics_dataset.parquet')
        
        if os.path.exists(pinszc_path):
            print(f" Loading PINSZC ground truth from: {pinszc_path}")
            pinszc_df = pd.read_parquet(pinszc_path)
            print(f"PINSZC samples: {len(pinszc_df)}")
        else:
            print(f"  PINSZC file not found at {pinszc_path}")
            print("Proceeding without teacher forcing ground truth")
            pinszc_df = None
        
        # Initialize Teacher Forcing Trainer
        trainer = TeacherForcingTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            save_dir=config['output']['models_dir'],
            teacher_forcing_ratio=config['teacher_forcing']['initial_ratio'],
            curriculum_schedule=config['teacher_forcing']['curriculum_schedule'],
            pinszc_ground_truth=pinszc_df
        )
        
        trainer.load_model('best_model.pth')
        print(" Model loaded successfully!")
        
        # Skip to evaluation
        test_results = trainer.evaluate_test_set()
        print(f" Loaded model Test R²: {test_results['r2']:.4f}")
        
    else:
        print(" No existing model found - starting fresh training...")
        
        # Load and preprocess data - CORRECTED
        df = load_and_preprocess_data(config['data']['parquet_file'])
        
        # Calculate optimal sequence length
        optimal_sequence_length = calculate_optimal_sequence_length(
            df, 
            target_temporal_coverage=config['data'].get('temporal_coverage', 'seasonal')
        )
        
        print(f"Optimal sequence length: {optimal_sequence_length}")
        
        # Create data loaders
        train_loader, val_loader, test_loader, train_dataset = create_data_loaders(
            df,
            batch_size=config['data']['batch_size'],
            sequence_length=optimal_sequence_length,
            enable_pattern_analysis=config['model']['pattern_analysis']['enabled']
        )
        
        # Initialize model
        model = ZeroCurtainHybridModel(
            input_features=len(train_dataset.feature_columns),
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            n_layers=config['model']['n_layers'],
            liquid_hidden=config['model']['liquid_hidden'],
            dropout=config['model']['dropout']
        )
        
        print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Bayesian optimization (optional)
        if config.get('bayesian_optimization', {}).get('enabled', False) and SKOPT_AVAILABLE:
            print("\n Running Bayesian Optimization...")
            optimizer = BayesianOptimizer(ZeroCurtainHybridModel, train_loader, val_loader, device)
            best_params, opt_result = optimizer.optimize(n_calls=20)
            
            # Update model with best parameters
            model = ZeroCurtainHybridModel(
                input_features=len(train_dataset.feature_columns),
                **best_params
            )
        elif config.get('bayesian_optimization', {}).get('enabled', False) and not SKOPT_AVAILABLE:
            print("  Bayesian optimization requested but scikit-optimize not available. Skipping...")
        
        # PART II: Initialize Teacher Forcing Trainer with PINSZC ground truth
        print("\n PART II: Initializing Teacher Forcing Trainer with PINSZC")
        
        # Load PINSZC ground truth from Part I
        pinszc_path = config['data'].get('pinszc_ground_truth', 
                                         './outputs/zero_curtain_enhanced_cryogrid_physics_dataset.parquet')
        
        if os.path.exists(pinszc_path):
            print(f" Loading PINSZC ground truth from: {pinszc_path}")
            pinszc_df = pd.read_parquet(pinszc_path)
            print(f"PINSZC samples: {len(pinszc_df)}")
        else:
            print(f"  PINSZC file not found at {pinszc_path}")
            print("Proceeding without teacher forcing ground truth")
            pinszc_df = None
        
        # Initialize Teacher Forcing Trainer
        trainer = TeacherForcingTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            save_dir=config['output']['models_dir'],
            teacher_forcing_ratio=config['teacher_forcing']['initial_ratio'],
            curriculum_schedule=config['teacher_forcing']['curriculum_schedule'],
            pinszc_ground_truth=pinszc_df
        )
        
        # Start training
        print(" Starting Training...")
        history = trainer.train(epochs=config['training']['epochs'])
        trainer.plot_training_history()
        
        # Evaluate after training
        test_results = trainer.evaluate_test_set()
    
    # Model explainability (runs regardless of whether model was loaded or trained)
    if config.get('explainability', {}).get('enabled', False):
        print("\n Running Model Explainability Analysis...")
        
        explainer = ModelExplainer(
            model=trainer.model,
            dataset=train_dataset,
            feature_names=train_dataset.feature_columns
        )
        
        # LIME explanation for a sample
        if LIME_AVAILABLE:
            try:
                lime_explanation = explainer.explain_with_lime(instance_idx=0)
                if lime_explanation is not None:
                    lime_explanation.show_in_notebook(show_table=True)
            except Exception as e:
                print(f"LIME explanation failed: {e}")
        
        # SHAP explanation
        if SHAP_AVAILABLE:
            try:
                shap_explainer, shap_values = explainer.explain_with_shap(sample_size=50)
                
                if shap_values is not None:
                    # Plot SHAP summary
                    shap.summary_plot(shap_values, features=train_dataset.feature_columns, show=False)
                    plt.savefig('./models/shap_summary.png', dpi=300, bbox_inches='tight')
                    plt.show()
            except Exception as e:
                print(f"SHAP explanation failed: {e}")
    
    # Final evaluation and results (runs regardless of path taken)
    print("\n Final Model Performance:")
    
    # Save results
    results_df = pd.DataFrame({
        'predictions_intensity': test_results['predictions'][:, 0],
        'predictions_duration': test_results['predictions'][:, 1],
        'predictions_extent': test_results['predictions'][:, 2],
        'targets_intensity': test_results['targets'][:, 0],
        'targets_duration': test_results['targets'][:, 1],
        'targets_extent': test_results['targets'][:, 2]
    })
    
    results_df.to_csv('./models/test_predictions.csv', index=False)
    
    # Create prediction plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    target_names = ['Intensity', 'Duration (hours)', 'Spatial Extent (m)']
    
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        ax.scatter(test_results['targets'][:, i], test_results['predictions'][:, i], alpha=0.6)
        ax.plot([test_results['targets'][:, i].min(), test_results['targets'][:, i].max()],
                [test_results['targets'][:, i].min(), test_results['targets'][:, i].max()], 'r--')
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name} Predictions')
        
        # Calculate R²
        r2 = r2_score(test_results['targets'][:, i], test_results['predictions'][:, i])
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('./models/prediction_scatter_plots.png', dpi=300)
    plt.show()
    
    print(" Analysis completed successfully!")
    print(f" Results saved in './models/' directory")
    
    return trainer, test_results

if __name__ == "__main__":
    main()
