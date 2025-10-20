#!/usr/bin/env python3

import re
from pathlib import Path

def update_script_paths():
    """
    Update all scripts to use new organized output directory structure.
    """
    print("="*90)
    print("UPDATING SCRIPT FILEPATHS FOR ORGANIZED STRUCTURE")
    print("="*90)
    
    project_root = Path("/path/to/user/arctic_zero_curtain_pipeline")
    
    path_mappings = {
        'outputs/physics_informed_zero_curtain_events_COMPLETE.parquet': 
            'outputs/consolidated_datasets/physics_informed_zero_curtain_events_COMPLETE.parquet',
        
        'outputs/physics_informed_events_train.parquet':
            'outputs/splits/physics_informed_events_train.parquet',
        
        'outputs/physics_informed_events_val.parquet':
            'outputs/splits/physics_informed_events_val.parquet',
        
        'outputs/physics_informed_events_test.parquet':
            'outputs/splits/physics_informed_events_test.parquet',
        
        'outputs/consolidated_physics_statistics.json':
            'outputs/statistics/consolidated_physics_statistics.json',
    }
    
    scripts_to_update = [
        'scripts/consolidate_physics_results.py',
        'scripts/qa_physics_results.py',
    ]
    
    print("\nUpdating script filepaths...")
    
    for script_path in scripts_to_update:
        full_path = project_root / script_path
        
        if not full_path.exists():
            print(f"    Script not found: {script_path}")
            continue
        
        with open(full_path, 'r') as f:
            content = f.read()
        
        original_content = content
        updates_made = 0
        
        for old_path, new_path in path_mappings.items():
            if old_path in content:
                content = content.replace(old_path, new_path)
                updates_made += 1
        
        if updates_made > 0:
            with open(full_path, 'w') as f:
                f.write(content)
            print(f"   Updated {script_path}: {updates_made} path(s) changed")
        else:
            print(f"  ℹ  {script_path}: no changes needed")
    
    print("\n All scripts updated with new filepaths")
    print("="*90)


if __name__ == "__main__":
    update_script_paths()