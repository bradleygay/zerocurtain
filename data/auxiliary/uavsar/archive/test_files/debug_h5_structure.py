#!/usr/bin/env python3
"""
Debug H5 file structure
"""

import h5py
from pathlib import Path

def print_h5_structure(h5_file, max_depth=3):
    """Print H5 file structure"""
    print(f"\n{'='*80}")
    print(f"FILE: {h5_file.name}")
    print('='*80)
    
    def print_group(name, obj, depth=0):
        if depth > max_depth:
            return
        
        indent = "  " * depth
        
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name} [Dataset] shape={obj.shape} dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}{name}/ [Group]")
    
    with h5py.File(h5_file, 'r') as f:
        f.visititems(print_group)

# Find H5 files
base_dir = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar/nisar_downloads")
h5_files = list(base_dir.rglob("*.h5"))

print(f"Found {len(h5_files)} H5 files\n")

for h5_file in h5_files:
    try:
        print_h5_structure(h5_file)
    except Exception as e:
        print(f"Error reading {h5_file}: {e}")
