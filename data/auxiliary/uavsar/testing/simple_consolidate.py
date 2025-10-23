#!/usr/bin/env python3
"""
Simple data consolidation: Displacement + SMAP → One DataFrame
"""

import pandas as pd
import numpy as np
import rasterio
from pathlib import Path

print("="*80)
print("SIMPLE CONSOLIDATION: DISPLACEMENT + SMAP")
print("="*80)

# 1. Load displacement data
disp_file = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar/displacement_30m/chevak_01811_17055_004_170527_PL09043020_30_CX_02_chevak_01811_17080_010_170814_PL09043020_30_CX_02_HHHV_displacement_30m.tif")

print("\n1. Loading displacement data...")
with rasterio.open(disp_file) as src:
    displacement = src.read(1, masked=True)
    transform = src.transform
    
    rows, cols = np.where(~displacement.mask)
    lons, lats = rasterio.transform.xy(transform, rows, cols)
    disp_values = displacement.data[rows, cols]

disp_df = pd.DataFrame({
    'datetime': pd.Timestamp('2017-07-05'),  # Midpoint between May 27 and Aug 14
    'latitude': lats,
    'longitude': lons,
    'displacement_m': disp_values,
    'source': 'UAVSAR',
    'polarization': 'HHHV'
})

print(f"  Loaded {len(disp_df):,} displacement records")

# 2. Load SMAP data
smap_file = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/smap/smap.parquet")

print("\n2. Loading SMAP data...")
smap_df = pd.read_parquet(smap_file)

# Rename columns to standard names
smap_df = smap_df.rename(columns={
    'lat': 'latitude',
    'lon': 'longitude'
})

# Process SMAP columns
temp_cols = [c for c in smap_df.columns if 'soil_temp' in c]
if temp_cols:
    smap_df['soil_temp_c'] = smap_df[temp_cols].mean(axis=1, skipna=True)

if 'sm_surface' in smap_df.columns:
    smap_df['soil_moist_frac'] = smap_df['sm_surface']
elif 'sm_rootzone' in smap_df.columns:
    smap_df['soil_moist_frac'] = smap_df['sm_rootzone']

# Keep only essential columns
smap_keep = ['datetime', 'latitude', 'longitude', 'soil_temp_c', 'soil_moist_frac']
smap_keep = [c for c in smap_keep if c in smap_df.columns]
smap_df = smap_df[smap_keep].copy()
smap_df['source'] = 'SMAP'

print(f"  Loaded {len(smap_df):,} SMAP records")

# 3. Combine
print("\n3. Combining datasets...")

# Ensure datetime is datetime type
disp_df['datetime'] = pd.to_datetime(disp_df['datetime'])
smap_df['datetime'] = pd.to_datetime(smap_df['datetime'])

# Concatenate
combined_df = pd.concat([disp_df, smap_df], ignore_index=True)

print(f"  Combined: {len(combined_df):,} total records")

# 4. Validate units
print("\n4. Validating units...")

# Temperature: should be -60 to 60°C
if 'soil_temp_c' in combined_df.columns:
    temp_valid = combined_df['soil_temp_c'].notna()
    temp_range = combined_df.loc[temp_valid, 'soil_temp_c']
    if len(temp_range) > 0:
        print(f"  Soil temp (°C): min={temp_range.min():.1f}, max={temp_range.max():.1f}")
        if temp_range.min() < -60 or temp_range.max() > 60:
            print("    WARNING: Temperature out of range!")

# Moisture: should be 0-1
if 'soil_moist_frac' in combined_df.columns:
    moist_valid = combined_df['soil_moist_frac'].notna()
    moist_range = combined_df.loc[moist_valid, 'soil_moist_frac']
    if len(moist_range) > 0:
        print(f"  Soil moisture (0-1): min={moist_range.min():.3f}, max={moist_range.max():.3f}")
        if moist_range.min() < 0 or moist_range.max() > 1:
            print("    WARNING: Moisture out of range!")

# Displacement: should be -10 to 10m
if 'displacement_m' in combined_df.columns:
    disp_valid = combined_df['displacement_m'].notna()
    disp_range = combined_df.loc[disp_valid, 'displacement_m']
    if len(disp_range) > 0:
        print(f"  Displacement (m): min={disp_range.min():.3f}, max={disp_range.max():.3f}")
        if disp_range.min() < -10 or disp_range.max() > 10:
            print("    WARNING: Displacement out of range!")

# 5. Save
output_file = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar/consolidated_data.parquet")
output_file.parent.mkdir(parents=True, exist_ok=True)

print(f"\n5. Saving to {output_file}...")
combined_df.to_parquet(output_file, index=False, compression='snappy')

file_size_mb = output_file.stat().st_size / (1024**2)
print(f"  Saved: {file_size_mb:.2f} MB")

# 6. Show first 50 rows
print("\n" + "="*80)
print("FIRST 50 ROWS:")
print("="*80)
print(combined_df.head(50).to_string())

print("\n" + "="*80)
print("SUMMARY STATISTICS:")
print("="*80)
print(combined_df.describe())

print("\n Done!")
