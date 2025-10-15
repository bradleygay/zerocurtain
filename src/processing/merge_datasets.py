"""
Dataset merging utilities.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dask.dataframe as dd


def merge_arctic_datasets(paths_dict, output_path):
    """Merge multiple Arctic datasets."""
    print(f"\nMerging {len(paths_dict)} datasets...")
    
    dataframes = []
    for name, path in paths_dict.items():
        print(f"  Loading: {name}")
        df = dd.read_parquet(path)
        dataframes.append(df)
    
    df_merged = dd.concat(dataframes, axis=0, ignore_index=True)
    df_merged = df_merged.sort_values('datetime').reset_index(drop=True)
    
    df_merged.to_parquet(output_path, engine='pyarrow', compression='snappy')
    print(f"Saved: {output_path}")
    
    return df_merged


if __name__ == "__main__":
    print("Merge module ready")
