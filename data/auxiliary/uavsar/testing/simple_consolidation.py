#!/usr/bin/env python3
"""
Simple Consolidation: UAVSAR + SMAP → Unified DataFrame
"""

import pandas as pd
import numpy as np
import rasterio
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

print("="*80)
print("SIMPLE CONSOLIDATION: UAVSAR + SMAP")
print("="*80)

# ============================================================================
# STEP 1: LOAD DISPLACEMENT DATA → nisar.parquet
# ============================================================================
print("\n[1/3] Loading UAVSAR displacement data...")

displacement_dir = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar/displacement_30m")
displacement_files = list(displacement_dir.glob("*_displacement_*.tif"))

print(f"Found {len(displacement_files)} displacement files")

all_displacement = []

for tif_file in displacement_files:
    print(f"  Reading: {tif_file.name}")
    
    with rasterio.open(tif_file) as src:
        displacement = src.read(1, masked=True)
        transform = src.transform
        
        rows, cols = np.where(~displacement.mask)
        
        if len(rows) == 0:
            continue
        
        lons, lats = rasterio.transform.xy(transform, rows, cols)
        disp_values = displacement.data[rows, cols]
        
        # Extract date from filename (YYMMDD)
        import re
        dates = re.findall(r'(\d{6})', tif_file.stem)
        if dates:
            try:
                date = datetime.strptime(dates[0], "%y%m%d")
            except:
                date = datetime.now()
        else:
            date = datetime.now()
        
        for lat, lon, disp in zip(lats, lons, disp_values):
            if not np.isnan(disp):
                all_displacement.append({
                    'datetime': date,
                    'latitude': lat,
                    'longitude': lon,
                    'displacement_m': disp,
                    'source': 'UAVSAR'
                })

nisar_df = pd.DataFrame(all_displacement)
print(f"Loaded {len(nisar_df):,} UAVSAR records")

# Save nisar.parquet
nisar_output = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar/output/nisar.parquet")
nisar_output.parent.mkdir(parents=True, exist_ok=True)
nisar_df.to_parquet(nisar_output, compression='snappy', index=False)
print(f" Saved: {nisar_output}")

# ============================================================================
# STEP 2: LOAD SMAP DATA (IN CHUNKS) WITH COLUMN STANDARDIZATION
# ============================================================================
print("\n[2/3] Loading SMAP data (chunked)...")

smap_path = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/smap/smap.parquet")
parquet_file = pq.ParquetFile(smap_path)

print(f"SMAP file has {parquet_file.num_row_groups} row groups")

# Filter to Alaska region only (save memory)
lat_min, lat_max = 60.0, 63.0
lon_min, lon_max = -167.0, -163.0

smap_chunks = []

for batch_idx in range(parquet_file.num_row_groups):
    if batch_idx % 200 == 0:
        print(f"  Processing batch {batch_idx}/{parquet_file.num_row_groups}")
    
    table = parquet_file.read_row_group(batch_idx)
    df_chunk = table.to_pandas()
    
    # Standardize column names BEFORE filtering
    # datetime: keep as is
    # lat → latitude
    # lon → longitude
    column_mapping = {}
    
    for col in df_chunk.columns:
        col_lower = col.lower()
        if col_lower in ['lat', 'latitude']:
            column_mapping[col] = 'latitude'
        elif col_lower in ['lon', 'long', 'longitude']:
            column_mapping[col] = 'longitude'
        elif col_lower in ['datetime', 'date', 'time']:
            column_mapping[col] = 'datetime'
    
    df_chunk = df_chunk.rename(columns=column_mapping)
    
    # Filter by region
    if 'latitude' in df_chunk.columns and 'longitude' in df_chunk.columns:
        mask = (
            (df_chunk['latitude'] >= lat_min) &
            (df_chunk['latitude'] <= lat_max) &
            (df_chunk['longitude'] >= lon_min) &
            (df_chunk['longitude'] <= lon_max)
        )
        
        filtered = df_chunk[mask].copy()
        
        if len(filtered) > 0:
            # Process SMAP columns
            # Soil temperature: average across layers, ensure in °C (-60 to 60)
            temp_cols = [c for c in filtered.columns if 'soil_temp' in c.lower() and 'layer' in c.lower()]
            
            if temp_cols:
                filtered['soil_temp_c'] = filtered[temp_cols].mean(axis=1, skipna=True)
                
                # Check if in Kelvin (> 200) and convert
                if filtered['soil_temp_c'].max() > 200:
                    filtered['soil_temp_c'] = filtered['soil_temp_c'] - 273.15
                
                # Sanity check: -60 to 60°C
                filtered['soil_temp_c'] = filtered['soil_temp_c'].clip(-60, 60)
            
            # Soil moisture: use surface, ensure 0-1
            if 'sm_surface' in filtered.columns:
                filtered['soil_moist_frac'] = filtered['sm_surface']
            elif 'sm_rootzone' in filtered.columns:
                filtered['soil_moist_frac'] = filtered['sm_rootzone']
            
            if 'soil_moist_frac' in filtered.columns:
                # Ensure 0-1 range
                filtered['soil_moist_frac'] = filtered['soil_moist_frac'].clip(0, 1)
            
            # Keep only essential columns
            keep_cols = ['datetime', 'latitude', 'longitude']
            if 'soil_temp_c' in filtered.columns:
                keep_cols.append('soil_temp_c')
            if 'soil_moist_frac' in filtered.columns:
                keep_cols.append('soil_moist_frac')
            
            # Add source
            filtered['source'] = 'SMAP'
            keep_cols.append('source')
            
            filtered = filtered[keep_cols]
            smap_chunks.append(filtered)

# Concatenate all SMAP chunks
if smap_chunks:
    smap_df = pd.concat(smap_chunks, ignore_index=True)
    print(f"Loaded {len(smap_df):,} SMAP records (Alaska region)")
else:
    print("No SMAP data in region")
    smap_df = pd.DataFrame()

# ============================================================================
# STEP 3: CONCATENATE & SAVE
# ============================================================================
print("\n[3/3] Concatenating UAVSAR + SMAP...")

# Ensure datetime is parsed
nisar_df['datetime'] = pd.to_datetime(nisar_df['datetime'])
if len(smap_df) > 0:
    smap_df['datetime'] = pd.to_datetime(smap_df['datetime'])

# Concatenate
unified_df = pd.concat([nisar_df, smap_df], ignore_index=True)

print(f"Total unified records: {len(unified_df):,}")
print(f"  UAVSAR: {len(nisar_df):,}")
print(f"  SMAP: {len(smap_df):,}")

# Save
output_path = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar/output/remote_sensing.parquet")
unified_df.to_parquet(output_path, compression='snappy', index=False)

file_size_mb = output_path.stat().st_size / (1024 * 1024)
print(f"\n Saved: {output_path} ({file_size_mb:.2f} MB)")

# ============================================================================
# DISPLAY FIRST 50 ROWS
# ============================================================================
print("\n" + "="*80)
print("FIRST 50 ROWS OF remote_sensing.parquet")
print("="*80)
print(unified_df.head(50))

print("\n" + "="*80)
print("COLUMN STATISTICS")
print("="*80)
print(unified_df.describe())

print("\n CONSOLIDATION COMPLETE")
