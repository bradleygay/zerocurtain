"""Teacher forcing dataset preparation."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import json
from datetime import datetime


def prepare_teacher_forcing_dataset(input_path, output_path, 
                                    sequence_length=30, prediction_horizon=7,
                                    validation_split=0.2, test_split=0.1):
    """Prepare dataset for teacher forcing training."""
    print("TEACHER FORCING DATASET PREPARATION")
    print(f"Loading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Total: {len(df):,} observations")
    
    # Split
    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.9)
    
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    
    print(f"Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}")
    
    # Save
    output_base = Path(output_path).parent / Path(output_path).stem
    df_train.to_parquet(f"{output_base}_train.parquet")
    df_val.to_parquet(f"{output_base}_val.parquet")
    df_test.to_parquet(f"{output_base}_test.parquet")
    
    print("Complete!")
    return {'train': len(df_train), 'val': len(df_val), 'test': len(df_test)}


if __name__ == "__main__":
    print("Teacher forcing module ready")
