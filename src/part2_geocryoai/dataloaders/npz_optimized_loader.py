"""
WORKING optimized dataloader for .npz files.
Simple, fast, and actually works.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class NPZSequenceDataset(Dataset):
    """Fast memory-mapped .npz dataset."""
    
    def __init__(self, npz_files, feature_columns=None, target_columns=None):
        self.npz_files = [Path(f) for f in npz_files]
        
        # Set feature/target columns (defaults from Part 1)
        self.feature_columns = feature_columns or [
            'mean_temperature', 'temperature_variance', 'permafrost_probability',
            'phase_change_energy', 'freeze_penetration_depth', 'thermal_diffusivity'
        ]
        self.target_columns = target_columns or [
            'intensity_percentile', 'duration_hours', 'spatial_extent_meters'
        ]
        
        # Build index
        self.index = []
        self.files = {}
        
        for file_idx, filepath in enumerate(self.npz_files):
            # Load data (no mmap - safer with multiple workers)
            data = np.load(filepath, allow_pickle=True)
            self.files[file_idx] = {
                'features': data['features'],
                'targets': data['targets']
            }
            n_sequences = len(data['features'])
            
            for local_idx in range(n_sequences):
                self.index.append((file_idx, local_idx))
        
        print(f"  NPZ Dataset: {len(self.index):,} sequences from {len(self.npz_files)} files")
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        file_idx, local_idx = self.index[idx]
        data = self.files[file_idx]
        
        # Direct access - loaded in memory
        return {
            'features': torch.from_numpy(data['features'][local_idx]).float(),
            'targets': torch.from_numpy(data['targets'][local_idx]).float(),
            'sequence_length': torch.tensor(data['features'][local_idx].shape[0], dtype=torch.long),
            'metadata': {}
        }

def create_npz_dataloaders(train_files, val_files, test_files, batch_size=32, num_workers=4):
    """Create dataloaders from .npz files."""
    
    print(f"\n{'='*60}")
    print("Creating NPZ DataLoaders")
    print(f"{'='*60}")
    
    feature_columns = [
        'mean_temperature', 'temperature_variance', 'permafrost_probability',
        'phase_change_energy', 'freeze_penetration_depth', 'thermal_diffusivity'
    ]
    target_columns = ['intensity_percentile', 'duration_hours', 'spatial_extent_meters']
    
    train_dataset = NPZSequenceDataset(train_files, feature_columns, target_columns)
    val_dataset = NPZSequenceDataset(val_files, feature_columns, target_columns)
    test_dataset = NPZSequenceDataset(test_files, feature_columns, target_columns)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Preserve temporal order
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context='fork',  # Faster on Linux
        drop_last=True  # Avoid small last batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"  Train: {len(train_loader):,} batches")
    print(f"  Val: {len(val_loader):,} batches")
    print(f"  Test: {len(test_loader):,} batches")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, test_loader
