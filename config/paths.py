"""Path configuration for Arctic zero-curtain pipeline."""

from pathlib import Path
import os

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get('ARCTIC_DATA_DIR', '/Users/username'))
OUTPUT_DIR = BASE_DIR / 'outputs'
CACHE_DIR = BASE_DIR / 'cache'

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / 'figures').mkdir(exist_ok=True)
(OUTPUT_DIR / 'reports').mkdir(exist_ok=True)
(BASE_DIR / 'logs').mkdir(exist_ok=True)

INPUT_PATHS = {
    'in_situ': DATA_DIR / 'merged_compressed_corrected_final.parquet',
    'uavsar_nisar_raw': DATA_DIR / 'merged_uavsar_nisar.parquet',
    'smap_raw': DATA_DIR / 'smap_master.parquet',
    'zero_curtain_detections': DATA_DIR / 'part1_pipeline_optimized' / 'performance_fixed_zero_curtain_events.parquet',
    'remote_sensing_detections': DATA_DIR / 'vectorized_output' / 'vectorized_high_performance_zero_curtain.parquet',
    'final_predictions': DATA_DIR / 'part4_transfer_learning_new' / 'predictions' / 'circumarctic_zero_curtain_predictions_20250730_160518.parquet',
}

INTERMEDIATE_PATHS = {
    'uavsar_nisar_transformed': CACHE_DIR / 'uavsar_nisar_transformed.parquet',
    'smap_transformed': CACHE_DIR / 'smap_transformed.parquet',
}

OUTPUT_PATHS = {
    'merged_final': OUTPUT_DIR / 'final_arctic_consolidated.parquet',
    'teacher_forcing_dataset': OUTPUT_DIR / 'teacher_forcing_in_situ_database.parquet',
    'quality_report': OUTPUT_DIR / 'reports' / 'quality_control_report.html',
    'data_summary_stats': OUTPUT_DIR / 'reports' / 'summary_statistics.json',
}

PATHS = {**INPUT_PATHS, **INTERMEDIATE_PATHS, **OUTPUT_PATHS}

def validate_paths():
    """Validate that required input paths exist."""
    missing = []
    for name, path in INPUT_PATHS.items():
        if not Path(path).exists():
            missing.append(f"{name}: {path}")
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} files:\n" + "\n".join(missing))
    return True

if __name__ == "__main__":
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"\nInput Paths:")
    for name, path in INPUT_PATHS.items():
        status = "" if Path(path).exists() else ""
        print(f"  {status} {name}: {path}")
