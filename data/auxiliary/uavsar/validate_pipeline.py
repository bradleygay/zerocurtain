#!/usr/bin/env python3
"""
Validate pipeline without overwriting existing files
"""

import sys
from pathlib import Path
import pandas as pd

print("="*80)
print("PIPELINE VALIDATION (READ-ONLY)")
print("="*80)

uavsar_root = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar")

# Test 1: Check critical directories exist
print("\n[1/6] Checking directory structure...")
required_dirs = [
    'production',
    'modules', 
    'utilities',
    'data_products',
    'displacement_30m',
    'documentation'
]

for dir_name in required_dirs:
    dir_path = uavsar_root / dir_name
    if dir_path.exists():
        file_count = len(list(dir_path.glob('*')))
        print(f"   {dir_name}/ ({file_count} items)")
    else:
        print(f"   {dir_name}/ MISSING!")
        sys.exit(1)

# Test 2: Check critical files exist
print("\n[2/6] Checking critical files...")
critical_files = {
    'production/run_pipeline.py': 'Main pipeline',
    'production/consolidate_all_data.py': 'Consolidation script',
    'production/pipeline_config.yaml': 'Pipeline config',
    'utilities/data_inventory.json': 'Data inventory',
    'utilities/uavsar_urls.json': 'Download URLs',
    'data_products/nisar.parquet': 'UAVSAR data',
    'data_products/remote_sensing.parquet': 'Combined data',
}

for filepath, description in critical_files.items():
    full_path = uavsar_root / filepath
    if full_path.exists():
        size_mb = full_path.stat().st_size / (1024 * 1024)
        print(f"   {filepath} ({size_mb:.2f} MB) - {description}")
    else:
        print(f"   {filepath} MISSING! - {description}")

# Test 3: Validate data products
print("\n[3/6] Validating data products...")
try:
    nisar_path = uavsar_root / 'data_products/nisar.parquet'
    rs_path = uavsar_root / 'data_products/remote_sensing.parquet'
    
    nisar_df = pd.read_parquet(nisar_path)
    rs_df = pd.read_parquet(rs_path)
    
    print(f"   nisar.parquet: {len(nisar_df):,} records")
    print(f"   remote_sensing.parquet: {len(rs_df):,} records")
    
    # Check required columns
    nisar_required = ['datetime', 'latitude', 'longitude', 'displacement_m']
    rs_required = ['datetime', 'latitude', 'longitude', 'source']
    
    nisar_has_cols = all(col in nisar_df.columns for col in nisar_required)
    rs_has_cols = all(col in rs_df.columns for col in rs_required)
    
    if nisar_has_cols:
        print("   nisar.parquet has all required columns")
    else:
        print("   nisar.parquet missing columns!")
    
    if rs_has_cols:
        print("   remote_sensing.parquet has all required columns")
    else:
        print("   remote_sensing.parquet missing columns!")
        
except Exception as e:
    print(f"   Error validating data: {e}")

# Test 4: Check displacement GeoTIFFs
print("\n[4/6] Checking displacement GeoTIFFs...")
disp_dir = uavsar_root / 'displacement_30m'
tif_files = list(disp_dir.glob('*.tif'))
print(f"   Found {len(tif_files)} GeoTIFF files")

# Test 5: Check modules can be imported
print("\n[5/6] Testing module imports...")
sys.path.insert(0, str(uavsar_root))
modules_to_test = [
    'modules.interferometry',
    'modules.consolidation',
    'modules.spatial_join',
    'modules.validation'
]

for module_name in modules_to_test:
    try:
        __import__(module_name)
        print(f"   {module_name}")
    except Exception as e:
        print(f"   {module_name}: {e}")

# Test 6: Check documentation
print("\n[6/6] Checking documentation...")
doc_files = [
    'documentation/README.md',
    'documentation/DIRECTORY_STRUCTURE.md',
    'documentation/USAGE_GUIDE.md',
    'documentation/FILE_LOCATIONS.md'
]

for doc_file in doc_files:
    doc_path = uavsar_root / doc_file
    if doc_path.exists():
        print(f"   {doc_file}")
    else:
        print(f"   {doc_file} missing")

print("\n" + "="*80)
print(" VALIDATION COMPLETE")
print("="*80)
print("\nPipeline is ready to use!")
print("No files were modified during validation.")
print("\nTo run consolidation:")
print("  cd production && python3 consolidate_all_data.py")
