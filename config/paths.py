"""Path configuration for Arctic zero-curtain pipeline."""

from pathlib import Path
import os
import sys

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path('/Volumes/levelup/bradleygay')

if not DATA_DIR.exists():
    print(f"⚠️  WARNING: External drive not mounted at {DATA_DIR}")
    sys.exit(1)

OUTPUT_DIR = BASE_DIR / 'outputs'
CACHE_DIR = BASE_DIR / 'cache'

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / 'figures').mkdir(exist_ok=True)
(OUTPUT_DIR / 'reports').mkdir(exist_ok=True)
(BASE_DIR / 'logs').mkdir(exist_ok=True)

INPUT_PATHS = {
    'in_situ': DATA_DIR / 'merged_compressed_corrected_final.parquet',
    'arctic_consolidated': DATA_DIR / 'final_arctic_consolidated_working_nisar_smap_corrected_verified_kelvin_corrected_verified.parquet',
    'circumarctic_predictions': DATA_DIR / 'P1' / 'part4_transfer_learning_new' / 'predictions' / 'circumarctic_zero_curtain_predictions_20250730_160518.parquet',
    'physics_zero_curtain': DATA_DIR / 'zero_curtain_results_comprehensive_physics.parquet',
}

INTERMEDIATE_PATHS = {
    'quality_controlled': CACHE_DIR / 'quality_controlled.parquet',
}

OUTPUT_PATHS = {
    'teacher_forcing_dataset': OUTPUT_DIR / 'teacher_forcing_in_situ_database.parquet',
    'data_summary_stats': OUTPUT_DIR / 'reports' / 'summary_statistics.json',
}

PATHS = {**INPUT_PATHS, **INTERMEDIATE_PATHS, **OUTPUT_PATHS}

def validate_paths():
    missing = []
    for name, path in INPUT_PATHS.items():
        if not Path(path).exists():
            missing.append(f"{name}: {path}")
    if missing:
        print(f"⚠️  Warning: {len(missing)} file(s) not found:")
        for m in missing:
            print(f"  - {m}")
        return False
    return True

if __name__ == "__main__":
    print("=" * 80)
    print("ARCTIC PIPELINE CONFIGURATION")
    print("=" * 80)
    print(f"\nData Directory: {DATA_DIR}")
    print(f"\nInput Files:")
    
    total_gb = 0
    for name, path in INPUT_PATHS.items():
        if Path(path).exists():
            size_gb = Path(path).stat().st_size / 1024**3
            total_gb += size_gb
            print(f"  ✓ {name}: {size_gb:.2f} GB")
        else:
            print(f"  ✗ {name}: NOT FOUND")
    
    print(f"\nTotal: {total_gb:.2f} GB")
