#!/usr/bin/env python3
"""
inventory_existing_data.py
Create inventory of existing UAVSAR/NISAR data
"""

import os
from pathlib import Path
import json
from datetime import datetime

def scan_for_data(base_dir):
    """Scan for existing H5 files and related data"""
    
    base_path = Path(base_dir)
    
    print("="*80)
    print(f"SCANNING: {base_dir}")
    print("="*80)
    
    # Search for H5 files
    h5_files = list(base_path.rglob("*.h5"))
    
    print(f"\nFound {len(h5_files)} H5 files")
    
    if h5_files:
        print("\nSample H5 files:")
        for h5_file in h5_files[:5]:
            try:
                rel_path = h5_file.relative_to(base_path)
            except ValueError:
                rel_path = h5_file
            size_mb = h5_file.stat().st_size / (1024 * 1024)
            print(f"  {rel_path} ({size_mb:.2f} MB)")
    
    # Search for GRD files
    grd_files = list(base_path.rglob("*.grd"))
    print(f"\nFound {len(grd_files)} GRD files")
    
    if grd_files:
        print("Sample GRD files:")
        for grd_file in grd_files[:3]:
            try:
                rel_path = grd_file.relative_to(base_path)
            except ValueError:
                rel_path = grd_file
            print(f"  {rel_path}")
    
    # Search for annotation files
    ann_files = list(base_path.rglob("*.ann"))
    print(f"\nFound {len(ann_files)} ANN files")
    
    if ann_files:
        print("Sample ANN files:")
        for ann_file in ann_files[:3]:
            try:
                rel_path = ann_file.relative_to(base_path)
            except ValueError:
                rel_path = ann_file
            print(f"  {rel_path}")
    
    # Create inventory
    inventory = {
        'scan_date': datetime.now().isoformat(),
        'base_directory': str(base_dir),
        'h5_files': [str(f) for f in h5_files],
        'grd_files': [str(f) for f in grd_files],
        'ann_files': [str(f) for f in ann_files],
        'total_h5': len(h5_files),
        'total_grd': len(grd_files),
        'total_ann': len(ann_files)
    }
    
    # Save inventory
    inventory_file = Path("utilities/data_inventory.json")
    with open(inventory_file, 'w') as f:
        json.dump(inventory, f, indent=2)
    
    print(f"\n Inventory saved to: {inventory_file}")
    
    return inventory

if __name__ == '__main__':
    base_dir = "/Users/[USER]/arctic_zero_curtain_pipeline"
    
    inventory = scan_for_data(base_dir)
    
    print("\n" + "="*80)
    print("INVENTORY SUMMARY")
    print("="*80)
    print(f"H5 files:  {inventory['total_h5']}")
    print(f"GRD files: {inventory['total_grd']}")
    print(f"ANN files: {inventory['total_ann']}")
    print("="*80)
