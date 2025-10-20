#!/usr/bin/env python3
"""
Comprehensive filepath update script.
Updates ALL import statements and file paths after reorganization.
"""

import re
from pathlib import Path
from typing import List, Tuple

class FilepathUpdater:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.changes_made = []
        
        # Define all path mappings
        self.import_mappings = {
            # Part 1 imports
            r'from physics_detection\.': 'from src.part1_physics_detection.',
            r'import physics_detection\.': 'import src.part1_physics_detection.',
            r'from src\.physics_detection\.': 'from src.part1_physics_detection.',
            r'import src\.physics_detection\.': 'import src.part1_physics_detection.',
            r'from detection\.': 'from src.part1_physics_detection.',
            r'import detection\.': 'import src.part1_physics_detection.',
            r'from src\.detection\.': 'from src.part1_physics_detection.',
            
            # Part 2 imports
            r'from src.part2_geocryoai.geocryoai_integration': 'from src.part2_geocryoai.geocryoai_integration',
            r'import src.part2_geocryoai.geocryoai_integration': 'import src.part2_geocryoai.geocryoai_integration',
            r'from src.part2_geocryoai.zero_curtain_ml_model': 'from src.part2_geocryoai.zero_curtain_ml_model',
            r'import src.part2_geocryoai.zero_curtain_ml_model': 'import src.part2_geocryoai.zero_curtain_ml_model',
            r'from src.part2_geocryoai.temporal_pattern_analyzer': 'from src.part2_geocryoai.temporal_pattern_analyzer',
            r'import src.part2_geocryoai.temporal_pattern_analyzer': 'import src.part2_geocryoai.temporal_pattern_analyzer',
            r'from src.part2_geocryoai.config_loader': 'from src.part2_geocryoai.config_loader',
            r'import src.part2_geocryoai.config_loader': 'import src.part2_geocryoai.config_loader',
            r'from modeling\.teacher_forcing_prep': 'from src.part2_geocryoai.teacher_forcing_prep',
            r'from src\.modeling\.teacher_forcing_prep': 'from src.part2_geocryoai.teacher_forcing_prep',
            r'from modeling\.': 'from src.part2_geocryoai.',
            r'import modeling\.': 'import src.part2_geocryoai.',
            
            # Preprocessing imports
            r'from data_ingestion\.': 'from src.preprocessing.',
            r'import data_ingestion\.': 'import src.preprocessing.',
            r'from src\.data_ingestion\.': 'from src.preprocessing.',
            r'from processing\.merge_datasets': 'from src.preprocessing.merge_datasets',
            r'from src\.processing\.merge_datasets': 'from src.preprocessing.merge_datasets',
            r'from processing\.merging_module': 'from src.preprocessing.merging_module',
            r'from src\.processing\.merging_module': 'from src.preprocessing.merging_module',
            r'from processing\.': 'from src.preprocessing.',
            r'from transformations\.': 'from src.preprocessing.',
            r'from src\.transformations\.': 'from src.preprocessing.',
            
            # Utils imports
            r'from visualization\.arctic_projections': 'from src.utils.arctic_projections',
            r'from src\.visualization\.arctic_projections': 'from src.utils.arctic_projections',
            r'from visualization\.': 'from src.utils.',
            r'from common\.': 'from src.utils.',
            r'from src\.common\.': 'from src.utils.',
        }
        
        self.filepath_mappings = {
            # Part 1 output paths
            r'outputs/zero_curtain_enhanced_cryogrid_physics_dataset\.parquet': 
                'outputs/part1_pinszc/zero_curtain_enhanced_cryogrid_physics_dataset.parquet',
            r'outputs/zero_curtain_quality_corrected\.parquet':
                'outputs/part1_pinszc/zero_curtain_quality_corrected.parquet',
            r'outputs/part1_pinszc/statistics/': 'outputs/part1_pinszc/statistics/',
            r'outputs/part1_pinszc/metadata/': 'outputs/part1_pinszc/metadata/',
            r'outputs/part1_pinszc/splits/': 'outputs/part1_pinszc/splits/',
            r'outputs/part1_pinszc/consolidated_datasets/': 'outputs/part1_pinszc/consolidated_datasets/',
            r'outputs/archive/emergency_saves/': 'outputs/archive/emergency_saves/',
            r'outputs/archive/incremental_saves/': 'outputs/archive/incremental_saves/',
            
            # Part 2 output paths
            r'outputs/part2_geocryoai/teacher_forcing_in_situ_database': 'outputs/part2_geocryoai/teacher_forcing_in_situ_database',
            r'outputs/part2_geocryoai/models/': 'outputs/part2_geocryoai/models/',
            r'src/outputs/part2_geocryoai/models/': 'outputs/part2_geocryoai/models/',
            
            # Config paths
            r'configs/part2_config\.yaml': 'config/part2_config.yaml',
        }
    
    def update_file(self, filepath: Path) -> bool:
        """Update a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError):
            return False
        
        original_content = content
        changes_in_file = []
        
        # Update imports
        for pattern, replacement in self.import_mappings.items():
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                changes_in_file.append(f"Import: {pattern} -> {replacement}")
                content = new_content
        
        # Update file paths
        for pattern, replacement in self.filepath_mappings.items():
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                changes_in_file.append(f"Path: {pattern} -> {replacement}")
                content = new_content
        
        # Write back if changes were made
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            rel_path = filepath.relative_to(self.root_dir)
            self.changes_made.append((str(rel_path), changes_in_file))
            return True
        
        return False
    
    def update_all_files(self):
        """Update all Python files in the repository."""
        print("=" * 80)
        print("COMPREHENSIVE FILEPATH UPDATE")
        print("=" * 80)
        print()
        
        # Find all Python files
        python_files = [
            f for f in self.root_dir.rglob("*.py")
            if '.git' not in str(f) and '__pycache__' not in str(f)
        ]
        
        # Also check shell scripts and config files
        script_files = list(self.root_dir.glob("scripts/*.sh"))
        config_files = list(self.root_dir.glob("config/*.py"))
        
        all_files = python_files + script_files + config_files
        
        print(f"Found {len(all_files)} files to check")
        print()
        
        updated_count = 0
        for filepath in sorted(all_files):
            if self.update_file(filepath):
                updated_count += 1
        
        print()
        print("=" * 80)
        print(f"SUMMARY: Updated {updated_count} files")
        print("=" * 80)
        print()
        
        if self.changes_made:
            print("DETAILED CHANGES:")
            print("-" * 80)
            for filepath, changes in self.changes_made:
                print(f"\n{filepath}:")
                for change in changes:
                    print(f"  • {change}")
        
        return updated_count

def main():
    root_dir = Path(__file__).parent.parent
    updater = FilepathUpdater(root_dir)
    updater.update_all_files()

if __name__ == "__main__":
    main()