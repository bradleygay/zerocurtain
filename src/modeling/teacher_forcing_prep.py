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
    print("=" * 80)
    print("TEACHER FORCING DATASET PREPARATION")
    print("=" * 80)
    
    print(f"\nLoading: {input_path}")
    df = pd.read_parquet(input_path)
    
    print(f"Total observations: {len(df):,}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Sort by datetime
    df = df.sort_values('datetime')
    
    # Split data
    train_end = int(len(df) * (1 - validation_split - test_split))
    val_end = int(len(df) * (1 - test_split))
    
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    
    print(f"\nSplits:")
    print(f"  Train: {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(df_val):,} ({len(df_val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(df_test):,} ({len(df_test)/len(df)*100:.1f}%)")
    
    # Save
    output_base = Path(output_path).parent / Path(output_path).stem
    
    train_path = f"{output_base}_train.parquet"
    val_path = f"{output_base}_val.parquet"
    test_path = f"{output_base}_test.parquet"
    
    df_train.to_parquet(train_path, engine='pyarrow', compression='snappy')
    df_val.to_parquet(val_path, engine='pyarrow', compression='snappy')
    df_test.to_parquet(test_path, engine='pyarrow', compression='snappy')
    
    # Metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'total_observations': len(df),
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon,
        'splits': {
            'train': {'count': len(df_train), 'file': train_path},
            'val': {'count': len(df_val), 'file': val_path},
            'test': {'count': len(df_test), 'file': test_path}
        }
    }
    
    metadata_path = f"{output_base}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")
    print(f"  Meta:  {metadata_path}")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    
    return metadata


if __name__ == "__main__":
    print("Teacher forcing module ready")
