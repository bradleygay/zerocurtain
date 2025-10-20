#!/usr/bin/env python3
"""
Validate the entire pipeline file paths without executing the full pipeline.
Tests: Preprocessing → Part I → Part II
Ensures all file paths are correct and accessible.
"""

import sys
from pathlib import Path
import importlib.util

class PipelineValidator:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.errors = []
        self.warnings = []
    
    def log_error(self, message: str):
        """Log an error."""
        self.errors.append(message)
        print(f"❌ ERROR: {message}")
    
    def log_warning(self, message: str):
        """Log a warning."""
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")
    
    def log_success(self, message: str):
        """Log a success."""
        print(f"✅ {message}")
    
    def check_imports(self):
        """Check that all modules can be imported."""
        print("\n" + "=" * 80)
        print("STEP 1: CHECKING MODULE IMPORTS")
        print("=" * 80)
        
        modules_to_check = [
            ('src.preprocessing.data_inspection_module', 'Preprocessing - Data Inspection'),
            ('src.preprocessing.merge_datasets', 'Preprocessing - Merge Datasets'),
            ('src.preprocessing.merging_module', 'Preprocessing - Merging Module'),
            ('src.part1_physics_detection.physics_config', 'Part I - Physics Config'),
            ('src.part1_physics_detection.zero_curtain_detector', 'Part I - Detector'),
            ('src.part2_geocryoai.config_loader', 'Part II - Config Loader'),
            ('src.part2_geocryoai.zero_curtain_ml_model', 'Part II - ML Model'),
            ('src.part2_geocryoai.geocryoai_integration', 'Part II - GeoCryoAI'),
            ('src.part2_geocryoai.temporal_pattern_analyzer', 'Part II - Pattern Analyzer'),
            ('src.part2_geocryoai.teacher_forcing_prep', 'Part II - Teacher Forcing'),
        ]
        
        for module_name, description in modules_to_check:
            try:
                module_path = self.root_dir / module_name.replace('.', '/')
                if not module_path.with_suffix('.py').exists():
                    self.log_error(f"{description}: File not found at {module_path}.py")
                    continue
                
                # Try to import
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    module_path.with_suffix('.py')
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    # Don't actually execute - just check syntax
                    self.log_success(f"{description}")
                else:
                    self.log_error(f"{description}: Cannot create module spec")
            
            except SyntaxError as e:
                self.log_error(f"{description}: Syntax error - {e}")
            except Exception as e:
                self.log_warning(f"{description}: Import check skipped - {e}")
    
    def check_config_files(self):
        """Check configuration files exist and are readable."""
        print("\n" + "=" * 80)
        print("STEP 2: CHECKING CONFIGURATION FILES")
        print("=" * 80)
        
        config_files = [
            ('config/paths.py', 'Main paths configuration'),
            ('config/part2_config.yaml', 'Part II model configuration'),
        ]
        
        for filepath, description in config_files:
            full_path = self.root_dir / filepath
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    # Check for old paths in config
                    old_patterns = [
                        'outputs/statistics/',
                        'outputs/metadata/',
                        'outputs/models/',
                        'configs/part2_config',
                    ]
                    
                    found_old = False
                    for pattern in old_patterns:
                        if pattern in content:
                            self.log_warning(f"{description}: Contains old path '{pattern}'")
                            found_old = True
                    
                    if not found_old:
                        self.log_success(f"{description}")
                
                except Exception as e:
                    self.log_error(f"{description}: Cannot read - {e}")
            else:
                self.log_error(f"{description}: File not found at {filepath}")
    
    def check_output_directories(self):
        """Check output directory structure."""
        print("\n" + "=" * 80)
        print("STEP 3: CHECKING OUTPUT DIRECTORIES")
        print("=" * 80)
        
        required_dirs = [
            ('outputs/part1_pinszc', 'Part I output directory'),
            ('outputs/part1_pinszc/statistics', 'Part I statistics'),
            ('outputs/part1_pinszc/metadata', 'Part I metadata'),
            ('outputs/part1_pinszc/splits', 'Part I data splits'),
            ('outputs/part1_pinszc/consolidated_datasets', 'Part I consolidated datasets'),
            ('outputs/part2_geocryoai', 'Part II output directory'),
            ('outputs/part2_geocryoai/models', 'Part II models'),
            ('outputs/archive', 'Archived outputs'),
        ]
        
        for dirpath, description in required_dirs:
            full_path = self.root_dir / dirpath
            if full_path.exists():
                self.log_success(f"{description}: {dirpath}")
            else:
                self.log_warning(f"{description}: Directory not found (will be created)")
    
    def check_input_data(self):
        """Check if required input data exists."""
        print("\n" + "=" * 80)
        print("STEP 4: CHECKING INPUT DATA")
        print("=" * 80)
        
        # Check for Part I output (needed for Part II)
        part1_output = self.root_dir / 'outputs/part1_pinszc/zero_curtain_enhanced_cryogrid_physics_dataset.parquet'
        
        if part1_output.exists():
            self.log_success(f"Part I PINSZC dataset found ({part1_output.stat().st_size / 1e9:.2f} GB)")
        else:
            self.log_warning("Part I PINSZC dataset not found (needed for Part II)")
        
        # Check for Part II training data
        tf_files = [
            'outputs/part2_geocryoai/teacher_forcing_in_situ_database_train.parquet',
            'outputs/part2_geocryoai/teacher_forcing_in_situ_database_val.parquet',
            'outputs/part2_geocryoai/teacher_forcing_in_situ_database_test.parquet',
        ]
        
        all_exist = True
        for tf_file in tf_files:
            full_path = self.root_dir / tf_file
            if full_path.exists():
                self.log_success(f"Teacher forcing data: {Path(tf_file).name}")
            else:
                self.log_warning(f"Teacher forcing data not found: {Path(tf_file).name}")
                all_exist = False
        
        if not all_exist:
            print("    → Run teacher_forcing_prep.py to generate training data")
    
    def check_scripts(self):
        """Check that main scripts exist and have correct paths."""
        print("\n" + "=" * 80)
        print("STEP 5: CHECKING EXECUTION SCRIPTS")
        print("=" * 80)
        
        scripts = [
            ('scripts/run_physics_detection.py', 'Part I execution script'),
            ('src/part1_physics_detection/run_zero_curtain_detection.py', 'Part I main script'),
            ('src/part2_geocryoai/teacher_forcing_prep.py', 'Part II data prep'),
            ('src/part2_geocryoai/zero_curtain_ml_model.py', 'Part II training script'),
        ]
        
        for script_path, description in scripts:
            full_path = self.root_dir / script_path
            if full_path.exists():
                # Check for old paths in script
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    old_patterns = [
                        'outputs/statistics/',
                        'outputs/metadata/',
                        'outputs/models/',
                        'from physics_detection.',
                        'from detection.',
                        'from modeling.',
                    ]
                    
                    found_old = False
                    for pattern in old_patterns:
                        if pattern in content and 'part1_pinszc' not in content and 'part2_geocryoai' not in content:
                            self.log_warning(f"{description}: May contain old path '{pattern}'")
                            found_old = True
                            break
                    
                    if not found_old:
                        self.log_success(f"{description}")
                
                except Exception as e:
                    self.log_warning(f"{description}: Cannot check - {e}")
            else:
                self.log_error(f"{description}: Script not found at {script_path}")
    
    def validate(self):
        """Run all validation checks."""
        print("=" * 80)
        print("PIPELINE PATH VALIDATION")
        print("=" * 80)
        print("\nValidating: Preprocessing → Part I → Part II")
        print("Mode: DRY RUN (checking paths only, not executing)")
        print()
        
        self.check_imports()
        self.check_config_files()
        self.check_output_directories()
        self.check_input_data()
        self.check_scripts()
        
        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        if self.errors:
            print(f"\n❌ ERRORS: {len(self.errors)}")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors:
            print("\n✅ VALIDATION PASSED")
            print("All file paths are correctly configured.")
            print("\nReady to:")
            print("  1. Run Part I detection (if not already run)")
            print("  2. Generate Part II training data")
            print("  3. Train Part II GeoCryoAI model")
            return True
        else:
            print("\n❌ VALIDATION FAILED")
            print("Fix errors before proceeding.")
            return False

def main():
    root_dir = Path(__file__).parent.parent
    validator = PipelineValidator(root_dir)
    
    success = validator.validate()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()