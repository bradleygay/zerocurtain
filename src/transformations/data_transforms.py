"""
Data transformation utilities for Arctic datasets.
Harmonizes schemas across different data sources.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import dask.dataframe as dd


def classify_depth_zone(depth):
    """Classify soil depth into zones."""
    if depth < 0.2:
        return 'surface'
    elif depth < 0.5:
        return 'intermediate'
    elif depth < 1.0:
        return 'deep'
    else:
        return 'very_deep'


def transform_uavsar_nisar(input_path, output_path):
    """Transform UAVSAR/NISAR data to standard schema."""
    print(f"\nTransforming UAVSAR/NISAR: {input_path}")
    df = dd.read_parquet(input_path)
    
    df_transformed = df[['longitude', 'latitude', 'thickness_m', 
                         'thickness_m_standardized', 'source', 'data_type']].copy()
    
    df_transformed['datetime'] = df['second_retrieval_dt']
    df_transformed['soil_temp'] = np.nan
    df_transformed['soil_moist'] = np.nan
    
    df_transformed.to_parquet(output_path, engine='pyarrow', compression='snappy')
    print(f"Saved: {output_path}")
    return df_transformed


if __name__ == "__main__":
    print("Transformation module ready")
