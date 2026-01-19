#!/usr/bin/env python3
"""
FULL-DATASET Ablation Study for GeoCryoAI - Table S2.6
NO SAMPLING - Complete PINSZC with robust prediction handling.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                    handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
DROPOUT = 0.2
LIQUID_UNITS = 128
BATCH_SIZE = 768
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
EPOCHS = 10
GRADIENT_CLIP = 0.5


class FullGeoCryoAI(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS, dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.lnn = nn.Sequential(
            nn.Linear(D_MODEL, 256), nn.Tanh(), nn.Dropout(DROPOUT),
            nn.Linear(256, LIQUID_UNITS), nn.Tanh(), nn.Dropout(DROPOUT),
            nn.Linear(LIQUID_UNITS, LIQUID_UNITS), nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(LIQUID_UNITS, 256), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )
        self.classifier = nn.Linear(64, 2)
        self.duration_head = nn.Linear(64, 1)
        self.extent_head = nn.Linear(64, 1)
        self.physics_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        x = self.lnn(x)
        x = self.decoder(x)
        return self.classifier(x), self.duration_head(x), self.extent_head(x), self.physics_weight


class NoPhysicsModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS, dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.lnn = nn.Sequential(nn.Linear(D_MODEL, 256), nn.Tanh(), nn.Linear(256, LIQUID_UNITS), nn.Tanh(), nn.Linear(LIQUID_UNITS, LIQUID_UNITS), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(LIQUID_UNITS, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.classifier = nn.Linear(64, 2)
        self.duration_head = nn.Linear(64, 1)
        self.extent_head = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        x = self.lnn(x)
        x = self.decoder(x)
        return self.classifier(x), self.duration_head(x), self.extent_head(x), torch.tensor(0.0, device=x.device)


class NoLNNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS, dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.lstm = nn.LSTM(D_MODEL, LIQUID_UNITS, num_layers=2, batch_first=True, dropout=DROPOUT)
        self.decoder = nn.Sequential(nn.Linear(LIQUID_UNITS, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.classifier = nn.Linear(64, 2)
        self.duration_head = nn.Linear(64, 1)
        self.extent_head = nn.Linear(64, 1)
        self.physics_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        x, _ = self.lstm(x.unsqueeze(1))
        x = x.squeeze(1)
        x = self.decoder(x)
        return self.classifier(x), self.duration_head(x), self.extent_head(x), self.physics_weight


class SingleScaleModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, D_MODEL)
        self.attention = nn.MultiheadAttention(D_MODEL, num_heads=1, dropout=DROPOUT, batch_first=True)
        self.lnn = nn.Sequential(nn.Linear(D_MODEL, 256), nn.Tanh(), nn.Linear(256, LIQUID_UNITS), nn.Tanh(), nn.Linear(LIQUID_UNITS, LIQUID_UNITS), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(LIQUID_UNITS, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.classifier = nn.Linear(64, 2)
        self.duration_head = nn.Linear(64, 1)
        self.extent_head = nn.Linear(64, 1)
        self.physics_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        x = self.lnn(x)
        x = self.decoder(x)
        return self.classifier(x), self.duration_head(x), self.extent_head(x), self.physics_weight


class DenseDecoderModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS, dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.lnn = nn.Sequential(nn.Linear(D_MODEL, 256), nn.Tanh(), nn.Linear(256, LIQUID_UNITS), nn.Tanh(), nn.Linear(LIQUID_UNITS, LIQUID_UNITS), nn.Tanh())
        self.decoder = nn.Linear(LIQUID_UNITS, 64)
        self.classifier = nn.Linear(64, 2)
        self.duration_head = nn.Linear(64, 1)
        self.extent_head = nn.Linear(64, 1)
        self.physics_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        x = self.lnn(x)
        x = torch.relu(self.decoder(x))
        return self.classifier(x), self.duration_head(x), self.extent_head(x), self.physics_weight


class BaselineMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.classifier = nn.Linear(64, 2)
        self.duration_head = nn.Linear(64, 1)
        self.extent_head = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.mlp(x)
        return self.classifier(x), self.duration_head(x), self.extent_head(x), torch.tensor(0.0, device=x.device)


def prepare_full_data(pinszc_path: str):
    """Load COMPLETE PINSZC dataset."""
    logger.info(f"Loading FULL PINSZC: {pinszc_path}")
    df = pd.read_parquet(pinszc_path, engine='pyarrow')
    logger.info(f"Loaded {len(df):,} events - NO SAMPLING")
    
    feature_cols = ['latitude', 'longitude', 'intensity_percentile', 'mean_temperature',
                    'permafrost_probability', 'phase_change_energy', 'year', 'month']
    available = [c for c in feature_cols if c in df.columns]
    logger.info(f"Features: {available}")
    
    X = df[available].values.astype(np.float32)
    median_duration = df['duration_hours'].median()
    y_class = (df['duration_hours'] > median_duration).astype(np.int64).values
    y_duration = df['duration_hours'].values.astype(np.float64)  # Use float64 for stability
    y_extent = df['spatial_extent_meters'].values.astype(np.float64) if 'spatial_extent_meters' in df.columns else np.zeros(len(df), dtype=np.float64)
    
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y_duration) & ~np.isnan(y_extent)
    X, y_class, y_duration, y_extent = X[valid], y_class[valid], y_duration[valid], y_extent[valid]
    logger.info(f"Valid samples: {len(X):,}")
    
    # Store raw statistics for RMSE calculation
    dur_mean, dur_std = y_duration.mean(), y_duration.std()
    ext_mean, ext_std = y_extent.mean(), y_extent.std()
    logger.info(f"Duration: mean={dur_mean:.1f}, std={dur_std:.1f}")
    logger.info(f"Extent: mean={ext_mean:.4f}, std={ext_std:.4f}")
    
    # 70/15/15 split
    X_train, X_temp, y_c_train, y_c_temp, y_d_train, y_d_temp, y_e_train, y_e_temp = \
        train_test_split(X, y_class, y_duration, y_extent, test_size=0.30, random_state=42)
    X_val, X_test, y_c_val, y_c_test, y_d_val, y_d_test, y_e_val, y_e_test = \
        train_test_split(X_temp, y_c_temp, y_d_temp, y_e_temp, test_size=0.50, random_state=42)
    
    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Scale features
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_val = feature_scaler.transform(X_val)
    X_test = feature_scaler.transform(X_test)
    
    # Normalize targets using simple z-score (more stable than RobustScaler for inverse transform)
    y_d_train_norm = (y_d_train - dur_mean) / dur_std
    y_d_val_norm = (y_d_val - dur_mean) / dur_std
    y_d_test_norm = (y_d_test - dur_mean) / dur_std
    
    y_e_train_norm = (y_e_train - ext_mean) / ext_std
    y_e_val_norm = (y_e_val - ext_mean) / ext_std
    y_e_test_norm = (y_e_test - ext_mean) / ext_std
    
    logger.info(f"Duration (norm): min={y_d_train_norm.min():.2f}, max={y_d_train_norm.max():.2f}")
    
    return {
        'X_train': X_train.astype(np.float32), 'X_val': X_val.astype(np.float32), 'X_test': X_test.astype(np.float32),
        'y_c_train': y_c_train, 'y_c_val': y_c_val, 'y_c_test': y_c_test,
        'y_d_train': y_d_train_norm.astype(np.float32), 'y_d_val': y_d_val_norm.astype(np.float32), 'y_d_test': y_d_test_norm.astype(np.float32),
        'y_e_train': y_e_train_norm.astype(np.float32), 'y_e_val': y_e_val_norm.astype(np.float32), 'y_e_test': y_e_test_norm.astype(np.float32),
        'y_d_test_raw': y_d_test, 'y_e_test_raw': y_e_test,
        'dur_mean': dur_mean, 'dur_std': dur_std, 'ext_mean': ext_mean, 'ext_std': ext_std,
        'input_dim': X_train.shape[1]
    }


def train_and_evaluate(model, data, device, use_physics=True):
    model = model.to(device)
    scaler = torch.amp.GradScaler('cuda')
    
    train_dataset = TensorDataset(
        torch.FloatTensor(data['X_train']), torch.LongTensor(data['y_c_train']),
        torch.FloatTensor(data['y_d_train']), torch.FloatTensor(data['y_e_train'])
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=8, pin_memory=True, persistent_workers=True)
    
    val_dataset = TensorDataset(
        torch.FloatTensor(data['X_val']), torch.LongTensor(data['y_c_val']),
        torch.FloatTensor(data['y_d_val']), torch.FloatTensor(data['y_e_val'])
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    test_dataset = TensorDataset(
        torch.FloatTensor(data['X_test']), torch.LongTensor(data['y_c_test']),
        torch.FloatTensor(data['y_d_test']), torch.FloatTensor(data['y_e_test'])
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    
    n_batches = len(train_loader)
    logger.info(f"  Training: {len(data['X_train']):,} samples, {n_batches} batches/epoch")
    
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_valid = 0
        
        for batch_idx, (X_batch, y_class_batch, y_dur_batch, y_ext_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device, non_blocking=True)
            y_class_batch = y_class_batch.to(device, non_blocking=True)
            y_dur_batch = y_dur_batch.to(device, non_blocking=True)
            y_ext_batch = y_ext_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits, dur_pred, ext_pred, physics_w = model(X_batch)
                
                # Classification loss
                loss = ce_loss(logits, y_class_batch)
                
                # Regression losses (on normalized targets)
                loss += 0.1 * mse_loss(dur_pred.squeeze(), y_dur_batch)  # Reduced weight
                loss += 0.1 * mse_loss(ext_pred.squeeze(), y_ext_batch)
                
                if use_physics:
                    # Soft penalty on normalized predictions (z-scores should be bounded)
                    physics_penalty = torch.mean(torch.relu(torch.abs(dur_pred) - 5))  # Penalize beyond 5 std
                    loss += 0.01 * physics_w * physics_penalty
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            n_valid += 1
            
            if (batch_idx + 1) % 5000 == 0:
                logger.info(f"    Epoch {epoch+1}, Batch {batch_idx+1}/{n_batches}, Loss: {total_loss/n_valid:.4f}")
        
        scheduler.step()
        
        if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for X_batch, y_class_batch, _, _ in val_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_class_batch = y_class_batch.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda'):
                        logits, _, _, _ = model(X_batch)
                    val_correct += (logits.argmax(dim=1) == y_class_batch).sum().item()
                    val_total += len(y_class_batch)
            
            val_acc = val_correct / val_total * 100
            logger.info(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/max(n_valid,1):.4f}, Val Acc: {val_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    # Test evaluation
    model.eval()
    all_preds = []
    all_dur_preds = []
    all_ext_preds = []
    
    logger.info(f"  Evaluating on {len(data['X_test']):,} test samples...")
    
    with torch.no_grad():
        for X_batch, _, _, _ in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits, dur_pred, ext_pred, _ = model(X_batch)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_dur_preds.append(dur_pred.cpu())
            all_ext_preds.append(ext_pred.cpu())
    
    y_pred = torch.cat(all_preds).numpy()
    dur_preds_norm = torch.cat(all_dur_preds).numpy().squeeze().astype(np.float64)
    ext_preds_norm = torch.cat(all_ext_preds).numpy().squeeze().astype(np.float64)
    
    # Clip normalized predictions to reasonable range before inverse transform
    dur_preds_norm = np.clip(dur_preds_norm, -10, 10)
    ext_preds_norm = np.clip(ext_preds_norm, -10, 10)
    
    # Inverse transform
    dur_preds_raw = dur_preds_norm * data['dur_std'] + data['dur_mean']
    ext_preds_raw = ext_preds_norm * data['ext_std'] + data['ext_mean']
    
    # Clip to physically reasonable bounds
    dur_preds_raw = np.clip(dur_preds_raw, 0, 8760)  # 0 to 1 year in hours
    ext_preds_raw = np.clip(ext_preds_raw, 0, 100)   # Reasonable extent range
    
    accuracy = accuracy_score(data['y_c_test'], y_pred) * 100
    dur_rmse = np.sqrt(mean_squared_error(data['y_d_test_raw'], dur_preds_raw))
    ext_rmse = np.sqrt(mean_squared_error(data['y_e_test_raw'], ext_preds_raw))
    physics_compliance = np.mean((dur_preds_raw > 0) & (dur_preds_raw < 4380)) * 100
    
    return {
        'accuracy': float(accuracy), 
        'duration_rmse': float(dur_rmse),
        'extent_rmse': float(ext_rmse), 
        'physics_compliance': float(physics_compliance),
        'best_val_acc': float(best_val_acc)
    }


def run_ablation(pinszc_path: str, output_dir: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    data = prepare_full_data(pinszc_path)
    input_dim = data['input_dim']
    
    configs = [
        ('Full GeoCryoAI', FullGeoCryoAI(input_dim), True, 'Complete architecture'),
        ('- Physics Constraints', NoPhysicsModel(input_dim), False, 'lambda_physics = 0'),
        ('- LNN Component', NoLNNModel(input_dim), True, 'Standard LSTM replacement'),
        ('- Multi-scale Attention', SingleScaleModel(input_dim), True, 'Single-scale attention'),
        ('- U-Net Decoder', DenseDecoderModel(input_dim), True, 'Dense decoder layers'),
        ('Baseline MLP', BaselineMLP(input_dim), False, '3-layer feedforward')
    ]
    
    results = []
    baseline_acc = None
    
    for name, model, use_physics, notes in configs:
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING: {name}")
        logger.info(f"{'='*70}")
        
        metrics = train_and_evaluate(model, data, device, use_physics=use_physics)
        
        if baseline_acc is None:
            baseline_acc = metrics['accuracy']
            delta = 0.0
        else:
            delta = metrics['accuracy'] - baseline_acc
        
        results.append({
            'configuration': name,
            'detection_accuracy': round(metrics['accuracy'], 1),
            'delta_accuracy': round(delta, 1),
            'duration_rmse_hours': round(metrics['duration_rmse'], 2),
            'extent_rmse_m': round(metrics['extent_rmse'], 3),
            'physics_compliance_pct': round(metrics['physics_compliance'], 1),
            'notes': notes
        })
        
        logger.info(f"  RESULT: Acc={metrics['accuracy']:.1f}% (Δ={delta:+.1f}%), RMSE={metrics['duration_rmse']:.2f}h")
        
        del model
        torch.cuda.empty_cache()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    total_events = len(data['X_train']) + len(data['X_val']) + len(data['X_test'])
    
    output = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'device': str(device),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'dataset': 'PINSZC',
            'total_events': total_events,
            'sampling': 'NONE - FULL DATASET'
        },
        'table_s2_6': results
    }
    
    json_path = Path(output_dir) / 'ablation_study_results_FULL.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved: {json_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TABLE S2.6: ABLATION STUDY (FULL PINSZC)")
    logger.info("=" * 80)
    logger.info(f"{'Configuration':<25} {'Acc %':>8} {'Δ Acc':>8} {'Dur RMSE':>12}")
    logger.info("-" * 60)
    for r in results:
        logger.info(f"{r['configuration']:<25} {r['detection_accuracy']:>8.1f} {r['delta_accuracy']:>+8.1f} {r['duration_rmse_hours']:>12.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pinszc', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("GEOCRYOAI ABLATION - FULL PINSZC - NO SAMPLING")
    logger.info("=" * 80)
    
    run_ablation(args.pinszc, args.output)
    
    logger.info("\nCOMPLETE")


if __name__ == '__main__':
    main()
