#!/usr/bin/env python3
"""
SMAP Data Consolidation Pipeline
Converts downloaded .h5 files to consolidated parquet format
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import gc

# Directories
BASE_DIR = Path.home() / "arctic_zero_curtain_pipeline"
RAW_DIR = BASE_DIR / "data" / "auxiliary" / "smap" / "raw"
CONSOLIDATED_DIR = BASE_DIR / "data" / "auxiliary" / "smap" / "consolidated"
LOG_DIR = BASE_DIR / "logs" / "smap"

# Create output directory
CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)

# Target variables from SPL4SMGP
TARGET_VARS = [
    'sm_surface',
    'sm_rootzone',
    'soil_temp_layer1',
    'soil_temp_layer2',
    'soil_temp_layer3',
    'soil_temp_layer4',
    'soil_temp_layer5',
    'soil_temp_layer6'
]

# Bounding box for circumarctic region
BBOX = (-180, 49, 180, 85.044)


class SMAPConsolidator:
    """Consolidate SMAP .h5 files into efficient parquet format"""
    
    def __init__(self):
        self.schema = self.create_schema()
        self.processed_files = set()
        
    def create_schema(self):
        """Create PyArrow schema for SMAP data"""
        fields = [
            pa.field('datetime', pa.timestamp('us')),
            pa.field('year', pa.int16()),
            pa.field('month', pa.int8()),
            pa.field('day', pa.int8()),
            pa.field('x', pa.float64()),
            pa.field('y', pa.float64()),
            pa.field('latitude', pa.float32()),
            pa.field('longitude', pa.float32()),
        ]
        
        # Add variable fields
        for var in TARGET_VARS:
            fields.append(pa.field(var, pa.float32()))
        
        return pa.schema(fields)
    
    def find_h5_files(self):
        """Find all .h5 files in raw directory"""
        h5_files = list(RAW_DIR.glob("*.h5")) + list(RAW_DIR.glob("*.he5"))
        h5_files.sort()
        print(f"[INFO] Found {len(h5_files):,} .h5 files")
        return h5_files
    
    def extract_datetime(self, h5_file):
        """Extract datetime from HDF5 file"""
        try:
            with h5py.File(h5_file, 'r') as f:
                # Try different attribute locations
                for attr_name in ['model_run_date', 'RangeBeginningDate']:
                    if attr_name in f.attrs:
                        attr = f.attrs[attr_name]
                        if isinstance(attr, bytes):
                            attr = attr.decode()
                        
                        # Parse datetime
                        for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d', '%Y%m%d']:
                            try:
                                return datetime.strptime(attr, fmt)
                            except:
                                continue
        except Exception as e:
            print(f"[WARNING] Could not extract datetime from {h5_file.name}: {e}")
        
        return None
    
    def find_variable_path(self, h5f, varname):
        """Find full path to variable in HDF5 file"""
        # Check root level
        if varname in h5f:
            return varname
        
        # Check in groups
        for group_name in h5f.keys():
            if isinstance(h5f[group_name], h5py.Group):
                if varname in h5f[group_name]:
                    return f"{group_name}/{varname}"
                
                # Check subgroups
                for subgroup in h5f[group_name].keys():
                    if isinstance(h5f[group_name][subgroup], h5py.Group):
                        if varname in h5f[group_name][subgroup]:
                            return f"{group_name}/{subgroup}/{varname}"
        
        return None
    
    def process_h5_file(self, h5_file):
        """Extract data from single HDF5 file"""
        try:
            dt = self.extract_datetime(h5_file)
            if not dt:
                print(f"[SKIP] {h5_file.name} - no datetime found")
                return None
            
            with h5py.File(h5_file, 'r') as f:
                # Get geolocation
                lat_path = self.find_variable_path(f, 'cell_lat')
                lon_path = self.find_variable_path(f, 'cell_lon')
                
                if not lat_path or not lon_path:
                    print(f"[SKIP] {h5_file.name} - no geolocation found")
                    return None
                
                lats = f[lat_path][()]
                lons = f[lon_path][()]
                
                # Apply bounding box mask
                mask = (
                    (lats >= BBOX[1]) & (lats <= BBOX[3]) &
                    (lons >= BBOX[0]) & (lons <= BBOX[2])
                )
                
                if not mask.any():
                    print(f"[SKIP] {h5_file.name} - no data in bounding box")
                    return None
                
                # Convert to projected coordinates (Arctic Polar Stereographic)
                from pyproj import Transformer
                transformer = Transformer.from_crs(4326, 3413, always_xy=True)
                x, y = transformer.transform(lons[mask], lats[mask])
                
                # Extract variables
                data_dict = {
                    'datetime': np.full(mask.sum(), dt, dtype='datetime64[us]'),
                    'year': np.full(mask.sum(), dt.year, dtype=np.int16),
                    'month': np.full(mask.sum(), dt.month, dtype=np.int8),
                    'day': np.full(mask.sum(), dt.day, dtype=np.int8),
                    'x': x.astype(np.float64),
                    'y': y.astype(np.float64),
                    'latitude': lats[mask].astype(np.float32),
                    'longitude': lons[mask].astype(np.float32),
                }
                
                # Extract target variables
                for var in TARGET_VARS:
                    var_path = self.find_variable_path(f, var)
                    if var_path:
                        data = f[var_path][()]
                        if data.shape == mask.shape:
                            data_dict[var] = data[mask].astype(np.float32)
                        else:
                            data_dict[var] = np.full(mask.sum(), np.nan, dtype=np.float32)
                    else:
                        data_dict[var] = np.full(mask.sum(), np.nan, dtype=np.float32)
                
                # Create PyArrow table
                table = pa.table(data_dict, schema=self.schema)
                return table
                
        except Exception as e:
            print(f"[ERROR] {h5_file.name}: {e}")
            return None
    
    def consolidate_by_year(self):
        """Consolidate files by year"""
        h5_files = self.find_h5_files()
        
        if not h5_files:
            print("[ERROR] No .h5 files found")
            return
        
        # Group files by year
        files_by_year = {}
        for h5_file in h5_files:
            dt = self.extract_datetime(h5_file)
            if dt:
                year = dt.year
                if year not in files_by_year:
                    files_by_year[year] = []
                files_by_year[year].append(h5_file)
        
        print(f"[INFO] Processing {len(files_by_year)} years: {sorted(files_by_year.keys())}")
        
        # Process each year
        for year in sorted(files_by_year.keys()):
            output_file = CONSOLIDATED_DIR / f"smap_{year}.parquet"
            
            if output_file.exists():
                print(f"[SKIP] {year} - already consolidated")
                continue
            
            print(f"\n[YEAR] Processing {year} ({len(files_by_year[year]):,} files)")
            
            tables = []
            for h5_file in tqdm(files_by_year[year], desc=f"Processing {year}"):
                table = self.process_h5_file(h5_file)
                if table is not None:
                    tables.append(table)
                
                # Periodic garbage collection
                if len(tables) % 100 == 0:
                    gc.collect()
            
            if tables:
                # Concatenate all tables
                print(f"[MERGE] Concatenating {len(tables):,} tables for {year}")
                combined_table = pa.concat_tables(tables)
                
                # Write to parquet
                print(f"[WRITE] Writing {len(combined_table):,} rows to {output_file.name}")
                pq.write_table(
                    combined_table,
                    output_file,
                    compression='zstd',
                    compression_level=5
                )
                
                file_size = output_file.stat().st_size / (1024**2)
                print(f"[OK] {year} complete: {file_size:.1f} MB")
                
                # Clean up
                del tables, combined_table
                gc.collect()
            else:
                print(f"[WARNING] No valid data found for {year}")


def main():
    """Main entry point"""
    print("=" * 80)
    print("SMAP Data Consolidation Pipeline")
    print("=" * 80)
    
    consolidator = SMAPConsolidator()
    
    try:
        consolidator.consolidate_by_year()
        print("\n[SUCCESS] Consolidation complete")
        return 0
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Consolidation stopped by user")
        return 130
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())