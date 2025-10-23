#!/usr/bin/env python3
"""
Complete Data Consolidation: All UAVSAR + SMAP → Unified Dataset
"""

import pandas as pd
import numpy as np
import rasterio
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
import re

print("="*80)
print("COMPLETE DATA CONSOLIDATION: ALL UAVSAR + SMAP")
print("="*80)

# ============================================================================
# STEP 1: LOAD ALL DISPLACEMENT DATA
# ============================================================================
print("\n[1/3] Loading ALL UAVSAR displacement data...")

displacement_dir = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar/displacement_30m")
displacement_files = list(displacement_dir.glob("*.tif"))

print(f"Found {len(displacement_files)} displacement GeoTIFF files")

all_displacement = []
file_stats = []

for idx, tif_file in enumerate(displacement_files, 1):
    if idx % 10 == 0:
        print(f"  Processing {idx}/{len(displacement_files)} files...")
    
    try:
        with rasterio.open(tif_file) as src:
            displacement = src.read(1, masked=True)
            transform = src.transform
            
            rows, cols = np.where(~displacement.mask)
            
            if len(rows) == 0:
                continue
            
            lons, lats = rasterio.transform.xy(transform, rows, cols)
            disp_values = displacement.data[rows, cols]
            
            # Extract metadata from filename
            filename = tif_file.stem
            
            # Extract dates (look for YYMMDD or YYYYMMDD patterns)
            date_matches = re.findall(r'(\d{6})', filename)
            
            if date_matches:
                # Use first date found
                try:
                    date = datetime.strptime(date_matches[0], "%y%m%d")
                except:
                    date = datetime.now()
            else:
                # Use file modification time
                date = datetime.fromtimestamp(tif_file.stat().st_mtime)
            
            # Extract polarization
            pol = 'unknown'
            for p in ['HHHV', 'HHVV', 'HVVV', 'HHHH', 'HVHV', 'VVVV', 'VH', 'VV', 'HH', 'HV']:
                if p in filename:
                    pol = p
                    break
            
            # Extract frequency if present
            freq = 'unknown'
            if 'frequencyA' in filename:
                freq = 'A'
            elif 'frequencyB' in filename:
                freq = 'B'
            
            valid_count = 0
            for lat, lon, disp in zip(lats, lons, disp_values):
                if not np.isnan(disp):
                    all_displacement.append({
                        'datetime': date,
                        'latitude': lat,
                        'longitude': lon,
                        'displacement_m': disp,
                        'polarization': pol,
                        'frequency': freq,
                        'source': 'UAVSAR',
                        'filename': tif_file.name
                    })
                    valid_count += 1
            
            file_stats.append({
                'filename': tif_file.name,
                'date': date,
                'polarization': pol,
                'frequency': freq,
                'valid_pixels': valid_count
            })
    
    except Exception as e:
        print(f"  ERROR reading {tif_file.name}: {e}")
        continue

if not all_displacement:
    print("ERROR: No valid displacement data found!")
    exit(1)

nisar_df = pd.DataFrame(all_displacement)
print(f"\n Loaded {len(nisar_df):,} total UAVSAR observations")
print(f"  From {len(file_stats)} files")
print(f"  Date range: {nisar_df['datetime'].min()} to {nisar_df['datetime'].max()}")

# Save nisar.parquet
nisar_output = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar/output/nisar.parquet")
nisar_output.parent.mkdir(parents=True, exist_ok=True)
nisar_df.to_parquet(nisar_output, compression='snappy', index=False)

nisar_size_mb = nisar_output.stat().st_size / (1024 * 1024)
print(f" Saved: {nisar_output} ({nisar_size_mb:.2f} MB)")

# Save file inventory
inventory_df = pd.DataFrame(file_stats)
inventory_df.to_csv(nisar_output.parent / 'nisar_file_inventory.csv', index=False)
print(f" Saved file inventory: {len(file_stats)} files")

# ============================================================================
# STEP 2: LOAD SMAP DATA (CHUNKED)
# ============================================================================
print("\n[2/3] Loading SMAP data (chunked, Alaska region only)...")

smap_path = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/smap/smap.parquet")
parquet_file = pq.ParquetFile(smap_path)

print(f"SMAP file has {parquet_file.num_row_groups} row groups")

# Get bounds from UAVSAR data
lat_min = nisar_df['latitude'].min() - 2.0  # Add 2° buffer
lat_max = nisar_df['latitude'].max() + 2.0
lon_min = nisar_df['longitude'].min() - 2.0
lon_max = nisar_df['longitude'].max() + 2.0

print(f"Filtering SMAP to bounds: lat [{lat_min:.1f}, {lat_max:.1f}], lon [{lon_min:.1f}, {lon_max:.1f}]")

smap_chunks = []

for batch_idx in range(parquet_file.num_row_groups):
    if batch_idx % 200 == 0:
        print(f"  Processing batch {batch_idx}/{parquet_file.num_row_groups}")
    
    try:
        table = parquet_file.read_row_group(batch_idx)
        df_chunk = table.to_pandas()
        
        # Standardize column names
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
                # Process soil temperature
                temp_cols = [c for c in filtered.columns if 'soil_temp' in c.lower() and 'layer' in c.lower()]
                
                if temp_cols:
                    filtered['soil_temp_c'] = filtered[temp_cols].mean(axis=1, skipna=True)
                    
                    # Convert from Kelvin if needed
                    if filtered['soil_temp_c'].max() > 200:
                        filtered['soil_temp_c'] = filtered['soil_temp_c'] - 273.15
                    
                    # Clip to reasonable range
                    filtered['soil_temp_c'] = filtered['soil_temp_c'].clip(-60, 60)
                
                # Process soil moisture
                if 'sm_surface' in filtered.columns:
                    filtered['soil_moist_frac'] = filtered['sm_surface']
                elif 'sm_rootzone' in filtered.columns:
                    filtered['soil_moist_frac'] = filtered['sm_rootzone']
                
                if 'soil_moist_frac' in filtered.columns:
                    filtered['soil_moist_frac'] = filtered['soil_moist_frac'].clip(0, 1)
                
                # Keep essential columns
                keep_cols = ['datetime', 'latitude', 'longitude']
                if 'soil_temp_c' in filtered.columns:
                    keep_cols.append('soil_temp_c')
                if 'soil_moist_frac' in filtered.columns:
                    keep_cols.append('soil_moist_frac')
                
                filtered['source'] = 'SMAP'
                keep_cols.append('source')
                
                filtered = filtered[keep_cols]
                smap_chunks.append(filtered)
    
    except Exception as e:
        continue

# Concatenate SMAP chunks
if smap_chunks:
    smap_df = pd.concat(smap_chunks, ignore_index=True)
    print(f"\n Loaded {len(smap_df):,} SMAP observations")
    print(f"  Date range: {smap_df['datetime'].min()} to {smap_df['datetime'].max()}")
else:
    print("WARNING: No SMAP data found in region")
    smap_df = pd.DataFrame()

# ============================================================================
# STEP 3: CONCATENATE & SAVE
# ============================================================================
print("\n[3/3] Concatenating datasets...")

# Ensure datetime is parsed
nisar_df['datetime'] = pd.to_datetime(nisar_df['datetime'])
if len(smap_df) > 0:
    smap_df['datetime'] = pd.to_datetime(smap_df['datetime'])

# Concatenate
unified_df = pd.concat([nisar_df, smap_df], ignore_index=True, sort=False)

print(f"\n Total unified records: {len(unified_df):,}")
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
print(unified_df.head(50).to_string())

print("\n" + "="*80)
print("COLUMN STATISTICS")
print("="*80)
print(unified_df.describe())

print("\n" + "="*80)
print("DATA QUALITY CHECKS")
print("="*80)

if 'displacement_m' in unified_df.columns:
    disp_stats = unified_df['displacement_m'].dropna()
    print(f"Displacement (m):")
    print(f"  Range: [{disp_stats.min():.3f}, {disp_stats.max():.3f}]")
    print(f"  Mean: {disp_stats.mean():.3f}")

if 'soil_temp_c' in unified_df.columns:
    temp_stats = unified_df['soil_temp_c'].dropna()
    print(f"Soil Temperature (°C):")
    print(f"  Range: [{temp_stats.min():.1f}, {temp_stats.max():.1f}]")
    print(f"  Mean: {temp_stats.mean():.1f}")
    if temp_stats.min() < -60 or temp_stats.max() > 60:
        print("    WARNING: Temperature outside expected range!")

if 'soil_moist_frac' in unified_df.columns:
    moist_stats = unified_df['soil_moist_frac'].dropna()
    print(f"Soil Moisture (fraction):")
    print(f"  Range: [{moist_stats.min():.3f}, {moist_stats.max():.3f}]")
    print(f"  Mean: {moist_stats.mean():.3f}")
    if moist_stats.min() < 0 or moist_stats.max() > 1:
        print("    WARNING: Moisture outside 0-1 range!")

print("\n CONSOLIDATION COMPLETE")
