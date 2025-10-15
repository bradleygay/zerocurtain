#!/usr/bin/env python3
"""
Main pipeline runner for Arctic zero-curtain teacher forcing dataset.
Orchestrates all stages from data loading to final output.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import INPUT_PATHS, OUTPUT_PATHS
from src.data_ingestion.inspect_parquet import inspect_parquet
from src.visualization.arctic_projections import analyze_data_coverage
from src.modeling.teacher_forcing_prep import prepare_teacher_forcing_dataset
import pandas as pd


def main():
    """Execute complete pipeline."""
    print("\n" + "=" * 80)
    print("ARCTIC ZERO-CURTAIN PIPELINE - TEACHER FORCING DATASET")
    print("=" * 80 + "\n")
    
    # Stage 1: Inspect data
    print("STAGE 1: Data Inspection")
    print("-" * 80)
    inspect_parquet(INPUT_PATHS['in_situ'], n_rows=5, show_preview=True)
    
    # Stage 2: Coverage analysis
    print("\n" + "=" * 80)
    print("STAGE 2: Coverage Analysis")
    print("-" * 80)
    df = pd.read_parquet(INPUT_PATHS['in_situ'])
    sample = df.sample(n=min(50000, len(df)))
    results = analyze_data_coverage(sample)
    
    # Stage 3: Teacher forcing preparation
    print("\n" + "=" * 80)
    print("STAGE 3: Teacher Forcing Dataset Preparation")
    print("-" * 80)
    metadata = prepare_teacher_forcing_dataset(
        input_path=INPUT_PATHS['in_situ'],
        output_path=OUTPUT_PATHS['teacher_forcing_dataset'],
        sequence_length=30,
        prediction_horizon=7,
        validation_split=0.2,
        test_split=0.1
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {OUTPUT_PATHS['teacher_forcing_dataset'].parent}")
    print(f"\nDataset ready for zero-curtain model training!")
    
    return metadata


if __name__ == "__main__":
    main()
