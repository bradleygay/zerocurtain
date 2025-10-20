#!/usr/bin/env python3
"""
Configuration loader for Part II experiments.
"""

import yaml
import os
from pathlib import Path


def load_config(config_path='../config/part2_config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        config: Dictionary with configuration parameters
    """
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    for dir_key in ['save_dir', 'models_dir', 'predictions_dir', 'explainability_dir']:
        dir_path = config['output'][dir_key]
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return config


def update_config_from_args(config, args):
    """
    Update configuration from command-line arguments.
    
    Args:
        config: Base configuration dictionary
        args: argparse.Namespace with command-line arguments
    
    Returns:
        Updated configuration
    """
    
    # Update specific fields from command-line args
    if hasattr(args, 'epochs') and args.epochs is not None:
        config['training']['epochs'] = args.epochs
    
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    
    if hasattr(args, 'teacher_forcing_ratio') and args.teacher_forcing_ratio is not None:
        config['teacher_forcing']['initial_ratio'] = args.teacher_forcing_ratio
    
    return config