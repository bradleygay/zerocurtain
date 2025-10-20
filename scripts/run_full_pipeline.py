#!/usr/bin/env python3
"""
Pipeline with COLD SEASON filtering (Sept-May only).
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import INPUT_PATHS, OUTPUT_PATHS
from config.pipeline_parameters import SPLIT_RATIOS, STRATIFICATION, SEQUENCE_PARAMS, SEASONAL_FILTER
from src.preprocessing.inspect_parquet import inspect_parquet
from src.part2_geocryoai.teacher_forcing_prep import prepare_teacher_forcing_dataset


def main():
    print("\n" + "=" * 80)
    print("ARCTIC PIPELINE - COLD SEASON ZERO-CURTAIN")
    print("=" * 80)
    print("\n CRITICAL: Summer months EXCLUDED")
    print(f"  Zero-curtain only occurs Sept-May (freeze-thaw transitions)")
    print(f"  Excluding: June, July, August (no freeze-thaw dynamics)")
    print(f"\nMethodology:")
    print(f"  Seasonal filter: {SEASONAL_FILTER['exclude_months']} excluded")
    print(f"  Splits: {SPLIT_RATIOS['train']*100:.0f}% / {SPLIT_RATIOS['validation']*100:.0f}% / {SPLIT_RATIOS['test']*100:.0f}%")
    print(f"  Stratification: {STRATIFICATION['stratify_columns']}")
    
    # Inspect
    print("\n" + "=" * 80)
    print("STAGE 1: Data Inspection")
    print("=" * 80)
    inspect_parquet(INPUT_PATHS['in_situ'], n_rows=5, show_preview=True)
    
    # Teacher Forcing with SUMMER FILTER
    print("\n" + "=" * 80)
    print("STAGE 2: Cold Season Stratified Dataset")
    print("=" * 80)
    metadata = prepare_teacher_forcing_dataset(
        input_path=INPUT_PATHS['in_situ'],
        output_path=OUTPUT_PATHS['teacher_forcing_dataset'],
        train_ratio=SPLIT_RATIOS['train'],
        val_ratio=SPLIT_RATIOS['validation'],
        test_ratio=SPLIT_RATIOS['test'],
        stratify_columns=STRATIFICATION['stratify_columns'],
        random_state=STRATIFICATION['random_state'],
        exclude_months=SEASONAL_FILTER['exclude_months'],  # CRITICAL
        min_duration_hours=SEQUENCE_PARAMS['min_duration_hours'],
        max_duration_hours=SEQUENCE_PARAMS['max_duration_hours']
    )
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE - COLD SEASON ONLY")
    print("=" * 80)
    print(f"\nCold season observations: {metadata['total_observations_after_filter']:,}")
    print(f"Summer observations removed: {metadata['observations_removed']:,}")
    print(f"Ready for zero-curtain model training")


if __name__ == "__main__":
    main()
