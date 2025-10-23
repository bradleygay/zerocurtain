#!/usr/bin/env python3
"""
Update file paths in all scripts after reorganization
"""

import os
import re
from pathlib import Path

uavsar_root = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar")

# Path mappings (old → new)
path_updates = {
    # JSON files moved to utilities
    'data_inventory.json': 'utilities/data_inventory.json',
    'uavsar_urls.json': 'utilities/uavsar_urls.json',
    
    # Config files
    'pipeline_config.yaml': 'production/pipeline_config.yaml',
    'test_config.yaml': 'testing/test_config.yaml',
    
    # Output paths
    'output/nisar.parquet': 'data_products/nisar.parquet',
    'output/remote_sensing.parquet': 'data_products/remote_sensing.parquet',
    
    # Test data
    'test_data/': 'testing/test_data/',
}

print("="*80)
print("UPDATING FILE PATHS IN ALL SCRIPTS")
print("="*80)

# Find all Python scripts
python_files = []
for directory in ['production', 'utilities', 'testing', 'modules', 'archived_pipelines']:
    dir_path = uavsar_root / directory
    if dir_path.exists():
        python_files.extend(dir_path.glob("*.py"))

print(f"\nFound {len(python_files)} Python files to update")

updates_made = {}

for py_file in python_files:
    try:
        with open(py_file, 'r') as f:
            content = f.read()
        
        original_content = content
        file_updated = False
        
        # Update each path
        for old_path, new_path in path_updates.items():
            # Handle various quote styles and path formats
            patterns = [
                (f'"{old_path}"', f'"{new_path}"'),
                (f"'{old_path}'", f"'{new_path}'"),
                (f'Path("{old_path}")', f'Path("{new_path}")'),
                (f"Path('{old_path}')", f"Path('{new_path}')"),
            ]
            
            for old_pattern, new_pattern in patterns:
                if old_pattern in content:
                    content = content.replace(old_pattern, new_pattern)
                    file_updated = True
                    
                    if py_file.name not in updates_made:
                        updates_made[py_file.name] = []
                    updates_made[py_file.name].append(f"{old_path} → {new_path}")
        
        # Write back if changed
        if content != original_content:
            with open(py_file, 'w') as f:
                f.write(content)
    
    except Exception as e:
        print(f"  ERROR updating {py_file.name}: {e}")

# Print summary
print("\n" + "="*80)
print("UPDATE SUMMARY")
print("="*80)

if updates_made:
    for filename, changes in sorted(updates_made.items()):
        print(f"\n{filename}:")
        for change in changes:
            print(f"  • {change}")
else:
    print("No path updates needed - all paths are relative or already correct!")

print("\n Path updates complete")
