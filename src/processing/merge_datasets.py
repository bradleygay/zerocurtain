"""
Dataset merging and consolidation utilities.
Combines multiple Arctic data sources into unified format.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dask.dataframe as dd
import pandas as pd


def merge_arctic_datasets(paths_dict, output_path):
    """
    Merge multiple Arctic datasets into consolidated file.
    
    Args:
        paths_dict: Dictionary of {name: path} for input files
        output_path: Path for merged output
    
    Returns:
        Merged Dask DataFrame
    """
    print(f"\nMERGING ARCTIC DATASETS")
    print("=" * 70)
    
    dataframes = []
    
    for name, path in paths_dict.items():
        print(f"\nLoading: {name}")
        print(f"  Path: {path}")
        
        df = dd.read_parquet(path)
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        
        dataframes.append(df)
    
    print(f"\nConcatenating {len(dataframes)} datasets...")
    df_merged = dd.concat(dataframes, axis=0, ignore_index=True)
    
    print(f"Sorting by datetime...")
    df_merged = df_merged.sort_values('datetime').reset_index(drop=True)
    
    print(f"Writing to: {output_path}")
    df_merged.to_parquet(output_path, engine='pyarrow', compression='snappy')
    
    print(f"\n{'='*70}")
    print(f"MERGE COMPLETE")
    print(f"{'='*70}")
    print(f"Total records: {len(df_merged):,}")
    print(f"Output: {output_path}")
    
    return df_merged


if __name__ == "__main__":
    """Test merging functions."""
    print("Merge module loaded successfully")
    print("\nAvailable functions:")
    print("  - merge_arctic_datasets(paths_dict, output_path)")
