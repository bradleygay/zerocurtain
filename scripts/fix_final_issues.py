#!/usr/bin/env python3
"""
Fix the final remaining path issues found by audit.
"""

from pathlib import Path
import json

def fix_file(filepath: Path, replacements: list):
    """Apply replacements to a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        for old, new in replacements:
            content = content.replace(old, new)
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed: {filepath.name}")
            return True
        
        return False
    
    except Exception as e:
        print(f"✗ Error fixing {filepath}: {e}")
        return False

def main():
    root = Path(__file__).parent.parent
    
    print("Fixing final path issues...")
    print("=" * 60)
    
    # 1. Fix METHODOLOGY.md
    methodology = root / 'METHODOLOGY.md'
    fix_file(methodology, [
        ('outputs/teacher_forcing_in_situ_database_train.parquet', 
         'outputs/part2_geocryoai/teacher_forcing_in_situ_database_train.parquet'),
        ('outputs/teacher_forcing_in_situ_database_val.parquet',
         'outputs/part2_geocryoai/teacher_forcing_in_situ_database_val.parquet'),
        ('outputs/teacher_forcing_in_situ_database_test.parquet',
         'outputs/part2_geocryoai/teacher_forcing_in_situ_database_test.parquet'),
        ('outputs/teacher_forcing_in_situ_database_metadata.json',
         'outputs/part2_geocryoai/teacher_forcing_in_situ_database_metadata.json'),
    ])
    
    # 2. Fix README.md
    readme = root / 'README.md'
    fix_file(readme, [
        ('outputs/splits/physics_informed_events_train.parquet',
         'outputs/part1_pinszc/splits/physics_informed_events_train.parquet'),
        ('outputs/splits/physics_informed_events_val.parquet',
         'outputs/part1_pinszc/splits/physics_informed_events_val.parquet'),
        ('outputs/splits/physics_informed_events_test.parquet',
         'outputs/part1_pinszc/splits/physics_informed_events_test.parquet'),
        ('outputs/consolidated_datasets/physics_informed_zero_curtain_events_COMPLETE.parquet',
         'outputs/part1_pinszc/consolidated_datasets/physics_informed_zero_curtain_events_COMPLETE.parquet'),
        ('src/physics_detection/physics_config.py',
         'src/part1_physics_detection/physics_config.py'),
        ('src/physics_detection/README.md',
         'src/part1_physics_detection/README.md'),
        ('src/data_ingestion/inspect_parquet.py',
         'src/preprocessing/inspect_parquet.py'),
    ])
    
    # 3. Fix config/part2_config.yaml
    config_yaml = root / 'config/part2_config.yaml'
    fix_file(config_yaml, [
        ('../outputs/zero_curtain_enhanced_cryogrid_physics_dataset.parquet',
         '../outputs/part1_pinszc/zero_curtain_enhanced_cryogrid_physics_dataset.parquet'),
    ])
    
    # 4. Fix data/auxiliary/DATA_SOURCES.md
    data_sources = root / 'data/auxiliary/DATA_SOURCES.md'
    fix_file(data_sources, [
        ('src/physics_detection/physics_config.py',
         'src/part1_physics_detection/physics_config.py'),
    ])
    
    # 5. Fix teacher_forcing_in_situ_database_metadata.json
    metadata_json = root / 'outputs/part2_geocryoai/teacher_forcing_in_situ_database_metadata.json'
    if metadata_json.exists():
        try:
            with open(metadata_json, 'r') as f:
                data = json.load(f)
            
            # Update paths in JSON
            for split in ['train', 'validation', 'test']:
                if split in data:
                    old_path = data[split].get('file', '')
                    if 'outputs/teacher_forcing_in_situ_database' in old_path:
                        new_path = old_path.replace(
                            'outputs/teacher_forcing_in_situ_database',
                            'outputs/part2_geocryoai/teacher_forcing_in_situ_database'
                        )
                        data[split]['file'] = new_path
            
            with open(metadata_json, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Fixed: {metadata_json.name}")
        
        except Exception as e:
            print(f"✗ Error fixing JSON: {e}")
    
    # 6. Fix src/part1_physics_detection/README.md
    part1_readme = root / 'src/part1_physics_detection/README.md'
    fix_file(part1_readme, [
        ('from physics_detection.zero_curtain_detector',
         'from src.part1_physics_detection.zero_curtain_detector'),
        ('from physics_detection.physics_config',
         'from src.part1_physics_detection.physics_config'),
    ])
    
    # 7. Fix src/part1_physics_detection/run_zero_curtain_detection.py
    run_detection = root / 'src/part1_physics_detection/run_zero_curtain_detection.py'
    fix_file(run_detection, [
        ('python src/detection/run_zero_curtain_detection.py',
         'python src/part1_physics_detection/run_zero_curtain_detection.py'),
    ])
    
    # Note: Don't fix scripts/audit_all_paths.py or scripts/validate_pipeline_paths.py
    # as those contain the patterns intentionally for checking purposes
    
    print("=" * 60)
    print("Final fixes complete!")
    print("\nNote: Scripts audit_all_paths.py and validate_pipeline_paths.py")
    print("      contain patterns intentionally - they are checking tools.")

if __name__ == "__main__":
    main()