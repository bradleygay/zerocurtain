#!/usr/bin/env python3

"""
PART IV: COMPLETE TRANSFER LEARNING PIPELINE WITH TRAINING
==========================================================
Trains GeoCryoAI model (Part II) on PIRSZC remote sensing data (Part III)
for circumpolar Arctic predictions (2015-2024)

[RESEARCHER] Gay
Arctic Zero-Curtain Detection Research
[RESEARCHER] Sciences Laboratory

Training Architecture:
- Part II: GeoCryoAI LSTM-Attention-Liquid model (pre-trained on PINSZC)
- Part III: PIRSZC physics-informed remote sensing detections (18.6M events)
- Part IV: Transfer learning → Fine-tune on PIRSZC → Circumpolar predictions

CRITICAL: Full training pipeline, NOT inference-only
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import warnings
import time
import json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm

# Deep learning frameworks
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

# NUMBA acceleration
from numba import jit, prange

print("="*80)
print("PART IV: GEOCRYOAI TRANSFER LEARNING WITH FULL TRAINING")
print("Arctic Zero-Curtain Detection: PINSZC → PIRSZC Transfer Learning")
print("="*80)
print(f"Timestamp: {datetime.now()}")
print()

# ============================================================================
# GEOCRYOAI MODEL ARCHITECTURE - EXACT REPRODUCTION FROM PART II
# ============================================================================

class LiquidNeuralNetwork(nn.Module):
    """Liquid Neural Network - Exact architecture from Part II"""
    def __init__(self, input_dim, hidden_dim, time_constant_init=1.0, time_mlp_input_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.time_constant = nn.Parameter(torch.tensor(time_constant_init))
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        
        if time_mlp_input_dim is None:
            time_mlp_input_dim = input_dim + hidden_dim
            
        self.time_mlp = nn.Sequential(
            nn.Linear(time_mlp_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, prev_state=None):
        batch_size = x.size(0)
        
        if prev_state is None:
            prev_state = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        input_contrib = torch.tanh(self.input_layer(x))
        hidden_contrib = torch.tanh(self.hidden_layer(prev_state))
        
        time_input = torch.cat([x, prev_state], dim=-1)
        time_factor = self.time_mlp(time_input)
        
        new_state = (1 - time_factor) * prev_state + time_factor * (input_contrib + hidden_contrib)
        output = self.output_layer(new_state)
        
        return output, new_state

class MultiHeadAttention(nn.Module):
    """Multi-head attention - Exact architecture from Part II"""
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.in_proj_weight = nn.Parameter(torch.randn(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.randn(3 * d_model))
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        return self.out_proj(context)

class AttentionLayer(nn.Module):
    """Complete attention layer from Part II"""
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        
        self.spatial_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.temporal_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.physics_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, d_model),
        )
        
    def forward(self, x):
        spatial_out = self.norm1(x + self.spatial_attention(x))
        temporal_out = self.norm2(spatial_out + self.temporal_attention(spatial_out))
        physics_out = self.norm3(temporal_out + self.physics_attention(temporal_out))
        cross_out = self.norm4(physics_out + self.cross_attention(physics_out))
        
        output = cross_out + self.ff(cross_out)
        
        return output

class GeoCryoAIModel(nn.Module):
    """Complete GeoCryoAI model - Exact architecture from Part II training"""
    def __init__(self, input_features=21, d_model=128, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128)
        )
        
        self.positional_encoding = nn.Parameter(torch.randn(10000, 1, 128))
        
        self.attention_layers = nn.ModuleList([
            AttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        
        self.liquid_layers = nn.ModuleList([
            LiquidNeuralNetwork(128, 64, 1.0, 192),
            LiquidNeuralNetwork(64, 64, 1.0, 128),
            LiquidNeuralNetwork(64, 64, 1.0, 128)
        ])
        
        self.physics_encoder = nn.Sequential(
            nn.Linear(21, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(96, 128),
            nn.BatchNorm1d(128)
        )
        
        # Output heads with domain-adaptive constraints
        self.intensity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # [0,1] range
        )
        
        self.duration_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Log-space output
        )
        
        self.extent_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Positive output for log-space
        )
        
    def forward(self, x, spatial_data=None):
        batch_size, seq_len, features = x.size()
        
        # Feature embedding
        x_flat = x.reshape(-1, features)
        embedded = self.feature_embedding(x_flat)
        embedded = embedded.view(batch_size, seq_len, -1)
        
        # Positional encoding
        embedded = embedded + self.positional_encoding[:seq_len].transpose(0, 1)
        
        # Attention processing
        for attention_layer in self.attention_layers:
            embedded = attention_layer(embedded)
        
        # Liquid neural network processing
        liquid_out = embedded.mean(dim=1)
        for liquid_layer in self.liquid_layers:
            liquid_out, _ = liquid_layer(liquid_out)
        
        # Physics encoding
        physics_features = self.physics_encoder(x_flat[:, :21])
        physics_features = physics_features.view(batch_size, seq_len, -1).mean(dim=1)
        
        # Spatial features (placeholder if not provided)
        if spatial_data is not None:
            spatial_features = spatial_data
        else:
            spatial_features = torch.zeros(batch_size, 16, device=x.device)
        
        # Feature fusion
        fused_features = torch.cat([liquid_out, physics_features, spatial_features], dim=1)
        final_features = self.fusion_layer(fused_features)
        
        # Multi-task predictions
        intensity = self.intensity_head(final_features)
        duration_raw = self.duration_head(final_features)
        duration = torch.clamp(duration_raw, min=0.0, max=8.95)  # [0, log(6570/6)]
        extent = self.extent_head(final_features)
        
        return {
            'intensity': intensity,
            'duration': duration,
            'extent': extent,
            'features': final_features
        }

# ============================================================================
# PIRSZC DATA PROCESSING - NUMBA ACCELERATED
# ============================================================================

@jit(nopython=True)
def normalize_features_numba(features):
    """NUMBA-accelerated feature normalization"""
    n_samples, n_features = features.shape
    normalized = np.empty_like(features)
    
    for j in range(n_features):
        col = features[:, j]
        valid_mask = np.isfinite(col)
        if np.sum(valid_mask) > 0:
            col_mean = np.mean(col[valid_mask])
            col_std = np.std(col[valid_mask])
            if col_std > 0:
                normalized[:, j] = (col - col_mean) / col_std
            else:
                normalized[:, j] = col - col_mean
        else:
            normalized[:, j] = 0.0
    
    for i in range(n_samples):
        for j in range(n_features):
            if not np.isfinite(normalized[i, j]):
                normalized[i, j] = 0.0
    
    return normalized

@jit(nopython=True, parallel=True)
def create_sequences_vectorized(features, targets, sequence_length=24):
    """NUMBA-accelerated sequence creation"""
    n_samples, n_features = features.shape
    n_sequences = max(0, n_samples - sequence_length + 1)
    
    if n_sequences == 0:
        return np.empty((0, sequence_length, n_features), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    
    sequences = np.empty((n_sequences, sequence_length, n_features), dtype=np.float32)
    sequence_targets = np.empty((n_sequences, 3), dtype=np.float32)
    
    for i in prange(n_sequences):
        for j in range(sequence_length):
            for k in range(n_features):
                sequences[i, j, k] = features[i + j, k]
        
        target_row = i + sequence_length - 1
        sequence_targets[i, 0] = targets[target_row, 0]  # intensity
        sequence_targets[i, 1] = targets[target_row, 1]  # duration
        sequence_targets[i, 2] = targets[target_row, 2]  # extent
    
    return sequences, sequence_targets

class PIRSZCDataset(Dataset):
    """Optimized PIRSZC dataset for transfer learning"""
    
    def __init__(self, pirszc_path, cluster_ids, sequence_length=24):
        self.sequence_length = sequence_length
        
        print(f"Loading PIRSZC data for {len(cluster_ids)} clusters...")
        
        # Load PIRSZC data
        required_columns = [
            'cluster_id', 'start_time', 'duration_hours', 'intensity_percentile', 
            'spatial_extent_meters', 'latitude', 'longitude', 'mean_temperature',
            'temperature_variance', 'mean_moisture', 'moisture_variance',
            'mean_displacement', 'consensus_confidence', 'permafrost_probability',
            'phase_change_energy', 'freeze_penetration_depth', 'thermal_diffusivity',
            'snow_insulation_factor', 'cryogrid_thermal_conductivity',
            'cryogrid_heat_capacity', 'surface_energy_balance',
            'van_genuchten_alpha', 'van_genuchten_n'
        ]
        
        df = pd.read_parquet(pirszc_path, columns=required_columns)
        df = df[df['cluster_id'].isin(cluster_ids)]
        df = df.sort_values(['cluster_id', 'start_time'])
        
        print(f"Loaded {len(df):,} PIRSZC records")
        
        # Process clusters
        all_sequences = []
        all_targets = []
        cluster_info = []
        temporal_spatial_data = []
        
        for cluster_id in tqdm(cluster_ids, desc="Processing clusters"):
            cluster_df = df[df['cluster_id'] == cluster_id]
            
            if len(cluster_df) < sequence_length:
                continue
            
            # Extract features (21 features)
            features = cluster_df[[
                'duration_hours', 'intensity_percentile', 'spatial_extent_meters',
                'latitude', 'longitude', 'mean_temperature', 'temperature_variance',
                'mean_moisture', 'moisture_variance', 'mean_displacement',
                'consensus_confidence', 'permafrost_probability', 'phase_change_energy',
                'freeze_penetration_depth', 'thermal_diffusivity', 'snow_insulation_factor',
                'cryogrid_thermal_conductivity', 'cryogrid_heat_capacity',
                'surface_energy_balance', 'van_genuchten_alpha', 'van_genuchten_n'
            ]].values.astype(np.float32)
            
            targets = cluster_df[[
                'intensity_percentile', 'duration_hours', 'spatial_extent_meters'
            ]].values.astype(np.float32)
            
            # Temporal-spatial metadata
            temporal_spatial = cluster_df[['start_time', 'latitude', 'longitude']].values
            
            # Normalize features
            features = normalize_features_numba(features)
            
            # Transform targets to log-space (matching Part II training)
            targets_transformed = np.empty_like(targets)
            for i in range(len(targets)):
                # Intensity: keep as-is [0,1]
                targets_transformed[i, 0] = np.clip(targets[i, 0], 0.0, 1.0)
                
                # Duration: log-space [0, log(6570/6)]
                duration = np.clip(targets[i, 1], 6.0, 6570.0)
                targets_transformed[i, 1] = np.log(duration) - np.log(6.0)
                
                # Extent: log-space
                extent = np.maximum(0.01, targets[i, 2])
                targets_transformed[i, 2] = np.log1p(extent)
            
            # Create sequences
            sequences, sequence_targets = create_sequences_vectorized(features, targets_transformed, sequence_length)
            
            if len(sequences) > 0:
                all_sequences.append(sequences)
                all_targets.append(sequence_targets)
                cluster_info.extend([cluster_id] * len(sequences))
                
                # Extract temporal-spatial for each sequence
                for seq_idx in range(len(sequences)):
                    temporal_idx = seq_idx + sequence_length - 1
                    if temporal_idx < len(temporal_spatial):
                        temporal_spatial_data.append(temporal_spatial[temporal_idx])
                    else:
                        temporal_spatial_data.append(temporal_spatial[-1])
        
        if all_sequences:
            self.sequences = np.vstack(all_sequences)
            self.targets = np.vstack(all_targets)
            self.cluster_info = cluster_info
            self.temporal_spatial_data = np.array(temporal_spatial_data)
        else:
            self.sequences = np.empty((0, sequence_length, 21), dtype=np.float32)
            self.targets = np.empty((0, 3), dtype=np.float32)
            self.cluster_info = []
            self.temporal_spatial_data = np.empty((0, 3), dtype=object)
        
        print(f"Generated {len(self.sequences):,} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        features = torch.from_numpy(self.sequences[idx]).float()
        targets = {
            'intensity': torch.tensor([self.targets[idx, 0]], dtype=torch.float32),
            'duration': torch.tensor([self.targets[idx, 1]], dtype=torch.float32),
            'extent': torch.tensor([self.targets[idx, 2]], dtype=torch.float32)
        }
        return features, targets

# ============================================================================
# TRANSFER LEARNING TRAINER
# ============================================================================

class TransferLearningTrainer:
    """Transfer learning trainer for GeoCryoAI on PIRSZC data"""
    
    def __init__(self, pretrained_model_path, device):
        self.device = device
        self.pretrained_model_path = pretrained_model_path
        
        # Initialize model
        self.model = GeoCryoAIModel().to(device)
        
        # Load pretrained weights
        self._load_pretrained_weights()
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        
        # Optimizer
        self.optimizer = self._setup_optimizer()
        
    def _load_pretrained_weights(self):
        """Load pretrained weights from Part II"""
        print("Loading pretrained GeoCryoAI model...")
        
        checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        else:
            pretrained_dict = checkpoint
        
        # Load weights
        self.model.load_state_dict(pretrained_dict, strict=False)
        print(" Pretrained weights loaded successfully")
        
    def _setup_optimizer(self):
        """Setup optimizer for transfer learning"""
        
        # Freeze early layers initially
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in ['feature_embedding', 'positional_encoding']):
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Optimizer with lower learning rate for transfer learning
        optimizer_params = [p for p in self.model.parameters() if p.requires_grad]
        
        return torch.optim.AdamW(optimizer_params, lr=1e-5, weight_decay=1e-5)
    
    def compute_loss(self, predictions, targets):
        """Multi-task loss computation"""
        
        intensity_loss = self.mse_loss(predictions['intensity'], targets['intensity'])
        duration_loss = self.mse_loss(predictions['duration'], targets['duration'])
        extent_loss = self.mse_loss(predictions['extent'], targets['extent'])
        
        # Check for NaN/inf
        if not torch.isfinite(intensity_loss):
            intensity_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        if not torch.isfinite(duration_loss):
            duration_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        if not torch.isfinite(extent_loss):
            extent_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        
        total_loss = (intensity_loss + duration_loss + extent_loss) / 3.0
        
        return {
            'total': total_loss,
            'intensity': intensity_loss,
            'duration': duration_loss,
            'extent': extent_loss
        }
    
    def train_epoch(self, dataloader, gradient_accumulation_steps=8):
        """Train one epoch"""
        self.model.train()
        epoch_losses = {'total': 0, 'intensity': 0, 'duration': 0, 'extent': 0}
        
        batch_count = 0
        
        for batch_idx, (features, targets) in enumerate(tqdm(dataloader, desc="Training")):
            features = features.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            predictions = self.model(features)
            losses = self.compute_loss(predictions, targets)
            
            loss = losses['total'] / gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            batch_count += 1
        
        for key in epoch_losses:
            epoch_losses[key] /= batch_count
        
        return epoch_losses
    
    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        epoch_losses = {'total': 0, 'intensity': 0, 'duration': 0, 'extent': 0}
        
        all_predictions = {'intensity': [], 'duration': [], 'extent': []}
        all_targets = {'intensity': [], 'duration': [], 'extent': []}
        
        with torch.no_grad():
            for features, targets in tqdm(dataloader, desc="Evaluating"):
                features = features.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                predictions = self.model(features)
                losses = self.compute_loss(predictions, targets)
                
                for key in epoch_losses:
                    epoch_losses[key] += losses[key].item()
                
                # Collect predictions
                for key in all_predictions:
                    all_predictions[key].extend(predictions[key].cpu().numpy().flatten())
                    all_targets[key].extend(targets[key].cpu().numpy().flatten())
        
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
        
        return epoch_losses, all_predictions, all_targets
    
    def unfreeze_all_layers(self):
        """Unfreeze all layers for fine-tuning"""
        print("Unfreezing all layers for fine-tuning...")
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Update optimizer with differentiated learning rates
        pretrained_params = []
        new_params = []
        
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in ['attention_layers', 'liquid_layers']):
                pretrained_params.append(param)
            else:
                new_params.append(param)
        
        optimizer_params = [
            {'params': pretrained_params, 'lr': 1e-5},
            {'params': new_params, 'lr': 1e-4}
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_params, weight_decay=1e-5)

# ============================================================================
# PART IV PIPELINE WITH FULL TRAINING
# ============================================================================

class Part4TransferLearningPipeline:
    """Complete transfer learning pipeline with training"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = Path.home() / "arctic_zero_curtain_pipeline"
        self.base_dir = Path(base_dir)
        
        self.dirs = {
            'part2': self.base_dir / 'outputs' / 'part2_geocryoai',
            'part3': self.base_dir / 'outputs' / 'part3_pirszc',
            'part4': self.base_dir / 'outputs' / 'part4_transfer_learning'
        }
        
        # Create Part IV directories
        self.dirs['part4'].mkdir(parents=True, exist_ok=True)
        (self.dirs['part4'] / 'checkpoints').mkdir(exist_ok=True)
        (self.dirs['part4'] / 'predictions').mkdir(exist_ok=True)
        (self.dirs['part4'] / 'metrics').mkdir(exist_ok=True)
        (self.dirs['part4'] / 'logs').mkdir(exist_ok=True)
        
        # Updated file paths
        self.files = {
            'geocryoai_checkpoint': self.dirs['part2'] / 'models' / 'best_model_tf.pth',
            'pirszc_data': self.dirs['part3'] / 'remote_sensing_physics_zero_curtain_comprehensive.parquet',
            'best_checkpoint': self.dirs['part4'] / 'checkpoints' / 'best_transfer_model.pth',
            'predictions_output': self.dirs['part4'] / 'predictions' / 'circumarctic_predictions.parquet'
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Part II model: {self.files['geocryoai_checkpoint']}")
        print(f"Part III data: {self.files['pirszc_data']}")
        print(f"Part IV output: {self.dirs['part4']}")
    
    def prepare_data(self):
        """Prepare train/val/test splits from PIRSZC"""
        print("\n" + "="*80)
        print("PREPARING PIRSZC DATA FOR TRANSFER LEARNING")
        print("="*80)
        
        if not self.files['pirszc_data'].exists():
            raise FileNotFoundError(f"PIRSZC data not found: {self.files['pirszc_data']}")
        
        # Load cluster distribution
        df_clusters = pd.read_parquet(self.files['pirszc_data'], columns=['cluster_id'])
        cluster_counts = df_clusters['cluster_id'].value_counts()
        unique_clusters = cluster_counts.index.tolist()
        
        print(f"Total PIRSZC clusters: {len(unique_clusters)}")
        
        # Split: 70% train, 15% val, 15% test
        np.random.seed(42)
        np.random.shuffle(unique_clusters)
        
        n_clusters = len(unique_clusters)
        train_clusters = unique_clusters[:int(0.7 * n_clusters)]
        val_clusters = unique_clusters[int(0.7 * n_clusters):int(0.85 * n_clusters)]
        test_clusters = unique_clusters[int(0.85 * n_clusters):]
        
        print(f"Split: {len(train_clusters)} train, {len(val_clusters)} val, {len(test_clusters)} test")
        
        # Create datasets
        print("\nCreating datasets...")
        self.train_dataset = PIRSZCDataset(self.files['pirszc_data'], train_clusters)
        self.val_dataset = PIRSZCDataset(self.files['pirszc_data'], val_clusters)
        self.test_dataset = PIRSZCDataset(self.files['pirszc_data'], test_clusters)
        
        print(f" Datasets created")
        print(f"  Train: {len(self.train_dataset):,} sequences")
        print(f"  Val: {len(self.val_dataset):,} sequences")
        print(f"  Test: {len(self.test_dataset):,} sequences")
    
    def train(self, n_epochs=10, batch_size=64):
        """Train model on PIRSZC data"""
        print("\n" + "="*80)
        print("TRAINING GEOCRYOAI ON PIRSZC DATA")
        print("="*80)
        
        if not self.files['geocryoai_checkpoint'].exists():
            raise FileNotFoundError(f"Part II model not found: {self.files['geocryoai_checkpoint']}")
        
        # Create dataloaders
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Initialize trainer
        trainer = TransferLearningTrainer(self.files['geocryoai_checkpoint'], self.device)
        
        # Training loop
        best_val_loss = float('inf')
        training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            
            # Train
            train_losses = trainer.train_epoch(train_loader)
            print(f"Train Loss: {train_losses['total']:.6f} "
                  f"(I: {train_losses['intensity']:.6f}, "
                  f"D: {train_losses['duration']:.6f}, "
                  f"E: {train_losses['extent']:.6f})")
            
            # Validate
            val_losses, _, _ = trainer.evaluate(val_loader)
            print(f"Val Loss: {val_losses['total']:.6f} "
                  f"(I: {val_losses['intensity']:.6f}, "
                  f"D: {val_losses['duration']:.6f}, "
                  f"E: {val_losses['extent']:.6f})")
            
            training_history['train_loss'].append(train_losses['total'])
            training_history['val_loss'].append(val_losses['total'])
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'training_history': training_history
                }, self.files['best_checkpoint'])
                print(f" Best model saved (val_loss: {best_val_loss:.6f})")
            
            # Unfreeze after 5 epochs
            if epoch == 4:
                trainer.unfreeze_all_layers()
        
        print(f"\n Training completed. Best val loss: {best_val_loss:.6f}")
        
        return trainer
    
    def generate_predictions(self, trainer, batch_size=32):
        """Generate predictions on test set"""
        print("\n" + "="*80)
        print("GENERATING CIRCUMPOLAR PREDICTIONS")
        print("="*80)
        
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Load best model
        checkpoint = torch.load(self.files['best_checkpoint'])
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Generate predictions
        _, predictions, targets = trainer.evaluate(test_loader)
        
        # Convert to original space
        pred_intensity = np.array(predictions['intensity'])
        pred_duration = np.exp(np.clip(np.array(predictions['duration']), 0, 8.95)) * 6.0
        pred_extent = np.expm1(np.clip(np.array(predictions['extent']), 0.001, 10.0))
        
        # Create dataframe
        predictions_df = pd.DataFrame({
            'cluster_id': self.test_dataset.cluster_info,
            'start_time': [pd.to_datetime(ts[0]) for ts in self.test_dataset.temporal_spatial_data],
            'latitude': [float(ts[1]) for ts in self.test_dataset.temporal_spatial_data],
            'longitude': [float(ts[2]) for ts in self.test_dataset.temporal_spatial_data],
            'predicted_intensity_percentile': pred_intensity,
            'predicted_duration_hours': pred_duration,
            'predicted_spatial_extent_meters': pred_extent
        })
        
        # Add temporal features
        predictions_df['year'] = predictions_df['start_time'].dt.year
        predictions_df['month'] = predictions_df['start_time'].dt.month
        predictions_df['season'] = predictions_df['month'].apply(
            lambda m: 'Winter' if m in [12,1,2] else 'Spring' if m in [3,4,5] 
                     else 'Summer' if m in [6,7,8] else 'Fall'
        )
        
        # Validation flags
        predictions_df['duration_valid'] = (
            (predictions_df['predicted_duration_hours'] >= 6) & 
            (predictions_df['predicted_duration_hours'] <= 6570)
        )
        predictions_df['intensity_valid'] = (
            (predictions_df['predicted_intensity_percentile'] >= 0) & 
            (predictions_df['predicted_intensity_percentile'] <= 1)
        )
        predictions_df['extent_valid'] = predictions_df['predicted_spatial_extent_meters'] >= 0.001
        predictions_df['prediction_valid'] = (
            predictions_df['duration_valid'] & 
            predictions_df['intensity_valid'] & 
            predictions_df['extent_valid']
        )
        
        return predictions_df
    
    def save_results(self, predictions_df):
        """Save results"""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        # Save predictions
        predictions_df.to_parquet(self.files['predictions_output'], index=False)
        predictions_df.to_csv(self.files['predictions_output'].with_suffix('.csv'), index=False)
        
        print(f" Predictions saved:")
        print(f"  Parquet: {self.files['predictions_output']}")
        print(f"  CSV: {self.files['predictions_output'].with_suffix('.csv')}")
        print(f"\nSummary:")
        print(f"  Total predictions: {len(predictions_df):,}")
        print(f"  Valid predictions: {predictions_df['prediction_valid'].sum():,} "
              f"({predictions_df['prediction_valid'].mean()*100:.1f}%)")
        print(f"  Temporal range: {predictions_df['start_time'].min()} to {predictions_df['start_time'].max()}")
        print(f"  Years covered: {sorted(predictions_df['year'].unique())}")
    
    def run_pipeline(self, n_epochs=10, batch_size=64):
        """Execute complete Part IV pipeline with training"""
        print("="*80)
        print("EXECUTING PART IV: FULL TRANSFER LEARNING WITH TRAINING")
        print("="*80)
        
        try:
            # Prepare data
            self.prepare_data()
            
            # Train model
            trainer = self.train(n_epochs=n_epochs, batch_size=batch_size)
            
            # Generate predictions
            predictions_df = self.generate_predictions(trainer)
            
            # Save results
            self.save_results(predictions_df)
            
            print("\n" + "="*80)
            print(" PART IV PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            
            return predictions_df
            
        except Exception as e:
            print(f"\n ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Part IV Transfer Learning with Training')
    parser.add_argument('--base-dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    
    args = parser.parse_args()
    
    pipeline = Part4TransferLearningPipeline(base_dir=args.base_dir)
    predictions_df = pipeline.run_pipeline(n_epochs=args.epochs, batch_size=args.batch_size)
    
    return predictions_df

if __name__ == '__main__':
    main()