"""
Data transformation utilities for Arctic datasets.
Harmonizes schemas across different data sources (UAVSAR, NISAR, SMAP, in situ).
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import dask.dataframe as dd


def classify_depth_zone(depth):
    """
    Classify soil depth into zones.
    
    Args:
        depth: Depth in meters
    
    Returns:
        str: 'surface', 'intermediate', 'deep', or 'very_deep'
    """
    if depth < 0.2:
        return 'surface'
    elif depth < 0.5:
        return 'intermediate'
    elif depth < 1.0:
        return 'deep'
    else:
        return 'very_deep'


def transform_uavsar_nisar(input_path, output_path):
    """
    Transform UAVSAR/NISAR data to standard schema.
    
    Note: Your consolidated file already has NISAR+SMAP merged,
    so this function is for reference/future use.
    
    Args:
        input_path: Path to UAVSAR/NISAR parquet file
        output_path: Path for transformed output
    
    Returns:
        Transformed DataFrame
    """
    print(f"\nTRANSFORMING UAVSAR/NISAR DATA")
    print(f"Input: {input_path}")
    
    df = dd.read_parquet(input_path)
    
    # Create standardized schema
    df_transformed = df[[
        'longitude', 'latitude', 'thickness_m', 'thickness_m_standardized',
        'first_retrieval_dt', 'second_retrieval_dt', 'duration_days',
        'period', 'season', 'year', 'source', 'data_type'
    ]].copy()
    
    # Use second retrieval date as datetime
    df_transformed['datetime'] = df_transformed['second_retrieval_dt']
    
    # Add placeholder columns for soil measurements (not in UAVSAR/NISAR)
    df_transformed['soil_temp'] = np.nan
    df_transformed['soil_temp_standardized'] = np.nan
    df_transformed['soil_temp_depth'] = np.nan
    df_transformed['soil_temp_depth_zone'] = pd.NA
    df_transformed['soil_moist'] = np.nan
    df_transformed['soil_moist_standardized'] = np.nan
    df_transformed['soil_moist_depth'] = np.nan
    
    # Reorder columns to match standard schema
    column_order = [
        'datetime', 'year', 'season', 'latitude', 'longitude',
        'thickness_m', 'thickness_m_standardized',
        'soil_temp', 'soil_temp_standardized', 'soil_temp_depth', 
        'soil_temp_depth_zone',
        'soil_moist', 'soil_moist_standardized', 'soil_moist_depth',
        'source', 'data_type'
    ]
    
    df_transformed = df_transformed[column_order]
    
    # Write output
    df_transformed.to_parquet(output_path, engine='pyarrow', compression='snappy')
    
    print(f"Output: {output_path}")
    print(f"Rows: {len(df_transformed):,}")
    
    return df_transformed


def transform_smap(input_path, output_path):
    """
    Transform SMAP data to standard schema.
    Expands multi-layer soil data into separate records.
    
    Args:
        input_path: Path to SMAP parquet file
        output_path: Path for transformed output
    
    Returns:
        Transformed DataFrame
    """
    print(f"\nTRANSFORMING SMAP DATA")
    print(f"Input: {input_path}")
    
    df = dd.read_parquet(input_path)
    
    # Rename columns
    df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
    
    # Define depth mappings
    temp_depths = {
        'soil_temp_layer1': 0.1,
        'soil_temp_layer2': 0.2,
        'soil_temp_layer3': 0.4,
        'soil_temp_layer4': 0.75,
        'soil_temp_layer5': 1.5,
        'soil_temp_layer6': 10.0
    }
    
    moist_depths = {
        'sm_surface': 0.05,
        'sm_rootzone': 1.0
    }
    
    # Expand each row into multiple records (one per layer)
    records = []
    
    for _, row in df.iterrows():
        base_record = {
            'datetime': row['datetime'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'year': pd.to_datetime(row['datetime']).year,
            'season': 'unknown',  # Could derive from month if needed
            'thickness_m': np.nan,
            'thickness_m_standardized': np.nan,
            'source': 'SMAP',
            'data_type': None
        }
        
        # Create records for temperature layers
        for layer, depth in temp_depths.items():
            if pd.notna(row.get(layer)):
                record = base_record.copy()
                record.update({
                    'soil_temp': row[layer],
                    'soil_temp_standardized': row[layer],
                    'soil_temp_depth': depth,
                    'soil_temp_depth_zone': classify_depth_zone(depth),
                    'soil_moist': np.nan,
                    'soil_moist_standardized': np.nan,
                    'soil_moist_depth': np.nan,
                    'data_type': 'soil_temperature'
                })
                records.append(record)
        
        # Create records for moisture layers
        for layer, depth in moist_depths.items():
            if pd.notna(row.get(layer)):
                record = base_record.copy()
                record.update({
                    'soil_temp': np.nan,
                    'soil_temp_standardized': np.nan,
                    'soil_temp_depth': np.nan,
                    'soil_temp_depth_zone': pd.NA,
                    'soil_moist': row[layer],
                    'soil_moist_standardized': row[layer],
                    'soil_moist_depth': depth,
                    'data_type': 'soil_moisture'
                })
                records.append(record)
    
    df_transformed = pd.DataFrame(records)
    
    # Write output
    df_transformed.to_parquet(output_path, engine='pyarrow', compression='snappy')
    
    print(f"Output: {output_path}")
    print(f"Rows: {len(df_transformed):,}")
    
    return df_transformed


if __name__ == "__main__":
    """Test transformation functions."""
    print("Transformation module loaded successfully")
    print("\nAvailable functions:")
    print("  - classify_depth_zone(depth)")
    print("  - transform_uavsar_nisar(input_path, output_path)")
    print("  - transform_smap(input_path, output_path)")
