#!/usr/bin/env python3

"""
Configuration loader for Part II GeoCryoAI training.
"""

import yaml
from pathlib import Path

def load_config(config_path='config/part2_config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (relative to project root or absolute)
    
    Returns:
        dict: Configuration dictionary or None if not found
    """
    # Convert to Path object
    config_path = Path(config_path)
    
    # If relative path, make it relative to project root
    if not config_path.is_absolute():
        # Get project root (3 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / config_path
    
    # Check if file exists
    if not config_path.exists():
        print(f"  Configuration file not found: {config_path}")
        return None
    
    # Load YAML
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f" Configuration loaded from: {config_path}")
        return config
    
    except Exception as e:
        print(f"  Error loading configuration: {e}")
        return None

def get_default_config():
    """
    Get default configuration when config file is not found.
    
    Returns:
        dict: Default configuration
    """
    return {
        'data': {
            'parquet_file': 'outputs/part1_pinszc/zero_curtain_enhanced_cryogrid_physics_dataset.parquet',
            'pinszc_ground_truth': 'outputs/part1_pinszc/zero_curtain_enhanced_cryogrid_physics_dataset.parquet',
            'batch_size': 128,
            'temporal_coverage': 'seasonal'
        },
        'model': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 4,
            'liquid_hidden': 256,
            'dropout': 0.1,
            'pattern_analysis': {
                'enabled': True
            }
        },
        'training': {
            'epochs': 25,
            'learning_rate': 0.0001,
            'weight_decay': 0.01
        },
        'teacher_forcing': {
            'initial_ratio': 0.9,
            'curriculum_schedule': 'exponential'
        },
        'output': {
            'models_dir': './outputs/part2_geocryoai/models'
        },
        'explainability': {
            'enabled': False
        }
    }