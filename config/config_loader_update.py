#!/usr/bin/env python3
"""Configuration loader for Part II GeoCryoAI training."""
import yaml
from pathlib import Path

def load_config(config_path='config/part2_config.yaml'):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        if not config_path.exists():
            project_root = Path.cwd()
            config_path = project_root / config_path
    
    if not config_path.exists():
        print(f" Configuration file not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f" Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        print(f" Error loading configuration: {e}")
        return None

def get_default_config():
    return {
        'data': {
            'parquet_file': 'outputs/part1_pinszc/consolidated_datasets/physics_informed_zero_curtain_events_COMPLETE.parquet',
            'pinszc_ground_truth': 'outputs/part1_pinszc/consolidated_datasets/physics_informed_zero_curtain_events_COMPLETE.parquet',
            'batch_size': 128,
            'temporal_coverage': 'seasonal',
            'test_size': 0.2,
            'val_size': 0.1
        },
        'model': {'d_model': 128, 'n_heads': 4, 'n_layers': 4, 'liquid_hidden': 256, 'dropout': 0.1,
                  'pattern_analysis': {'enabled': True}},
        'training': {'epochs': 25, 'learning_rate': 0.0001, 'weight_decay': 0.01},
        'teacher_forcing': {'enabled': True, 'initial_ratio': 0.9, 'curriculum_schedule': 'exponential'},
        'output': {'models_dir': 'outputs/part2_geocryoai/models'},
        'explainability': {'enabled': False}
    }
