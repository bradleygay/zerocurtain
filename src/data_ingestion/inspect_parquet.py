"""Data inspection utilities from playground notebook."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import dask.dataframe as dd
import pyarrow.parquet as pq
import polars as pl
from config.parameters import PARAMETERS

def inspect_parquet(filepath, n_rows=10, show_schema=True, show_preview=True):
    """
    Inspect parquet file structure and contents.
    
    Args:
        filepath: Path to parquet file
        n_rows: Number of preview rows to display
        show_schema: Whether to show schema information
        show_preview: Whether to show data preview
    
    Returns:
        Dask DataFrame
    """
    print(f"\nINSPECTING: {filepath}\n")
    
    if show_schema:
        pq_file = pq.ParquetFile(filepath)
        print("PyArrow Schema:")
        print(pq_file.schema)
    
    df_dask = dd.read_parquet(filepath)
    print(f"\nDask Columns:")
    print(df_dask.columns.tolist())
    
    if show_preview:
        print(f"\nData Preview (first {n_rows} rows):")
        preview = df_dask.head(n_rows)
        print(preview)
    
    df_lazy = pl.scan_parquet(filepath)
    print(f"\nPolars Schema:")
    print(df_lazy.schema)
    
    return df_dask

if __name__ == "__main__":
    # Test the function
    from config.paths import INPUT_PATHS
    
    test_file = INPUT_PATHS['in_situ']
    if Path(test_file).exists():
        print("Testing inspect_parquet function...")
        df = inspect_parquet(test_file, n_rows=5)
        print(f"\nReturned DataFrame has {len(df.columns)} columns")
    else:
        print(f"Test file not found: {test_file}")
