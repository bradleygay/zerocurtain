#!/usr/bin/env python3
"""
modules/consolidation.py
Multi-Source Consolidation - CORRECTED for actual SMAP schema


SMAP schema: lat, lon, sm_surface, sm_rootzone, soil_temp_layer1-6
"""

import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime


class MultiSourceConsolidator:
    """Memory-efficient multi-source data consolidator"""
    
    def __init__(self,
                 displacement_dir: str,
                 smap_path: str,
                 output_path: str,
                 spatial_joiner: 'SpatioTemporalJoiner',
                 require_complete_records: bool = True,
                 logger: Optional[logging.Logger] = None):
        
        self.displacement_dir = Path(displacement_dir)
        self.smap_path = Path(smap_path)
        self.output_path = Path(output_path)
        self.spatial_joiner = spatial_joiner
        self.require_complete_records = require_complete_records
        self.logger = logger or logging.getLogger(__name__)
        
        if not self.displacement_dir.exists():
            self.logger.warning(f"Displacement directory not found: {displacement_dir}")
        
        if not self.smap_path.exists():
            self.logger.warning(f"SMAP file not found: {smap_path}")
        
        self.stats = {
            'displacement_files_found': 0,
            'displacement_records_loaded': 0,
            'smap_records_loaded': 0,
            'total_complete_records': 0,
            'rejected_incomplete': 0,
            'match_success_rate': 0.0
        }
    
    def consolidate_all_sources(self) -> Dict:
        """Consolidate displacement and SMAP data"""
        
        self.logger.info("="*80)
        self.logger.info("MULTI-SOURCE CONSOLIDATION")
        self.logger.info("="*80)
        
        consolidation_start = datetime.now()
        
        # Step 1: Load displacement data
        self.logger.info("Step 1: Loading displacement data...")
        displacement_df = self._load_displacement_data()
        self.stats['displacement_records_loaded'] = len(displacement_df)
        
        if len(displacement_df) == 0:
            self.logger.warning("No displacement data found - skipping consolidation")
            return self.stats
        
        # Step 2: Load SMAP data IN CHUNKS
        self.logger.info("Step 2: Loading SMAP data (chunked for memory efficiency)...")
        
        # Get spatial bounds of displacement data to filter SMAP
        lat_min = displacement_df['latitude'].min()
        lat_max = displacement_df['latitude'].max()
        lon_min = displacement_df['longitude'].min()
        lon_max = displacement_df['longitude'].max()
        
        self.logger.info(f"  Displacement bounds: lat [{lat_min:.2f}, {lat_max:.2f}], "
                        f"lon [{lon_min:.2f}, {lon_max:.2f}]")
        
        smap_df = self._load_smap_data_chunked(lat_min, lat_max, lon_min, lon_max)
        self.stats['smap_records_loaded'] = len(smap_df)
        
        if len(smap_df) == 0:
            self.logger.warning("No SMAP data in displacement region - skipping join")
            return self.stats
        
        # Step 3: Spatiotemporal join
        self.logger.info("Step 3: Performing spatiotemporal join...")
        unified_df = self.spatial_joiner.join_displacement_soil(displacement_df, smap_df)
        
        # Step 4: Enforce completeness
        if self.require_complete_records:
            self.logger.info("Step 4: Enforcing completeness guarantee...")
            unified_df, rejected_count = self._enforce_completeness(unified_df)
            self.stats['rejected_incomplete'] = rejected_count
        
        self.stats['total_complete_records'] = len(unified_df)
        self.stats['match_success_rate'] = (
            len(unified_df) / self.stats['displacement_records_loaded']
            if self.stats['displacement_records_loaded'] > 0 else 0.0
        )
        
        # Step 5: Save consolidated output
        if len(unified_df) > 0:
            self.logger.info("Step 5: Saving consolidated output...")
            self._save_consolidated_output(unified_df)
        else:
            self.logger.warning("No unified records to save")
        
        # Log summary
        duration = datetime.now() - consolidation_start
        self.logger.info("-"*80)
        self.logger.info("Consolidation Summary:")
        self.logger.info(f"  Duration: {duration}")
        self.logger.info(f"  Displacement records: {self.stats['displacement_records_loaded']:,}")
        self.logger.info(f"  SMAP records: {self.stats['smap_records_loaded']:,}")
        self.logger.info(f"  Complete records generated: {self.stats['total_complete_records']:,}")
        self.logger.info(f"  Match success rate: {self.stats['match_success_rate']:.1%}")
        if self.require_complete_records:
            self.logger.info(f"  Incomplete records rejected: {self.stats['rejected_incomplete']:,}")
        self.logger.info("-"*80)
        
        return self.stats
    
    def _load_displacement_data(self) -> pd.DataFrame:
        """Load all displacement GeoTIFFs and extract metadata"""
        import rasterio
        
        displacement_files = list(self.displacement_dir.rglob("*_displacement_30m.tif"))
        
        if not displacement_files:
            self.logger.warning(f"No displacement files found in {self.displacement_dir}")
            return pd.DataFrame()
        
        self.stats['displacement_files_found'] = len(displacement_files)
        self.logger.info(f"Found {len(displacement_files):,} displacement files")
        
        all_records = []
        
        for idx, disp_file in enumerate(displacement_files):
            if idx % 10 == 0 and idx > 0:
                self.logger.info(f"  Processed {idx}/{len(displacement_files)} files")
            
            try:
                with rasterio.open(disp_file) as src:
                    displacement = src.read(1, masked=True)
                    transform = src.transform
                    
                    rows, cols = np.where(~displacement.mask)
                    
                    if len(rows) == 0:
                        continue
                    
                    lons, lats = rasterio.transform.xy(transform, rows, cols)
                    disp_values = displacement.data[rows, cols]
                    
                    # Extract metadata from filename
                    metadata = self._extract_displacement_metadata(disp_file)
                    
                    for lat, lon, disp in zip(lats, lons, disp_values):
                        if not np.isnan(disp):
                            record = {
                                'datetime': metadata.get('datetime'),
                                'latitude': lat,
                                'longitude': lon,
                                'displacement_m': disp,
                                'source': 'UAVSAR',
                                'polarization': metadata.get('polarization', 'unknown'),
                                'file_path': str(disp_file)
                            }
                            all_records.append(record)
            
            except Exception as e:
                self.logger.warning(f"Error processing {disp_file}: {e}")
                continue
        
        if not all_records:
            self.logger.warning("No valid displacement records extracted")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['year'] = df['datetime'].dt.year
        df['season'] = df['datetime'].dt.month.map(self._month_to_season)
        
        self.logger.info(f"Loaded {len(df):,} displacement observations")
        
        return df
    
    def _extract_displacement_metadata(self, file_path: Path) -> Dict:
        """Extract metadata from displacement filename"""
        import re
        
        filename = file_path.stem
        metadata = {}
        
        # Extract polarization
        for pol in ['HHHV', 'HHVV', 'HVVV', 'HHHH', 'HVHV', 'VVVV']:
            if pol in filename:
                metadata['polarization'] = pol
                break
        
        # Extract date (YYMMDD or YYYYMMDD)
        date_patterns = [r'(\d{6})', r'(\d{8})']
        
        for pattern in date_patterns:
            matches = re.findall(pattern, filename)
            if matches:
                date_str = matches[0]
                try:
                    if len(date_str) == 6:
                        metadata['datetime'] = datetime.strptime(date_str, "%y%m%d")
                    elif len(date_str) == 8:
                        metadata['datetime'] = datetime.strptime(date_str, "%Y%m%d")
                    break
                except ValueError:
                    continue
        
        if 'datetime' not in metadata:
            metadata['datetime'] = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        return metadata
    
    def _load_smap_data_chunked(self, lat_min: float, lat_max: float, 
                                lon_min: float, lon_max: float) -> pd.DataFrame:
        """Load SMAP data in chunks, filtering by bounds - CORRECTED SCHEMA"""
        
        self.logger.info(f"Loading SMAP data from {self.smap_path}")
        self.logger.info("  Using chunked reading to avoid memory issues...")
        
        # Read parquet in batches
        parquet_file = pq.ParquetFile(self.smap_path)
        
        self.logger.info(f"  SMAP file has {parquet_file.num_row_groups} row groups")
        
        # Add buffer to bounds
        buffer = 1.0  # degrees
        lat_min_buf = lat_min - buffer
        lat_max_buf = lat_max + buffer
        lon_min_buf = lon_min - buffer
        lon_max_buf = lon_max + buffer
        
        filtered_chunks = []
        
        for batch_idx in range(parquet_file.num_row_groups):
            if batch_idx % 100 == 0:
                self.logger.info(f"  Processing batch {batch_idx}/{parquet_file.num_row_groups}")
            
            try:
                # Read one row group at a time
                table = parquet_file.read_row_group(batch_idx)
                df_chunk = table.to_pandas()
                
                # CORRECTED: Use 'lat' and 'lon' instead of 'latitude' and 'longitude'
                mask = (
                    (df_chunk['lat'] >= lat_min_buf) &
                    (df_chunk['lat'] <= lat_max_buf) &
                    (df_chunk['lon'] >= lon_min_buf) &
                    (df_chunk['lon'] <= lon_max_buf)
                )
                
                filtered = df_chunk[mask]
                
                if len(filtered) > 0:
                    # Rename to standard names
                    filtered = filtered.rename(columns={
                        'lat': 'latitude',
                        'lon': 'longitude'
                    })
                    filtered_chunks.append(filtered)
                
                # Clear memory
                del df_chunk, table
            
            except Exception as e:
                self.logger.warning(f"Error reading batch {batch_idx}: {e}")
                continue
        
        if not filtered_chunks:
            self.logger.warning("No SMAP data found in region")
            return pd.DataFrame()
        
        # Combine all chunks
        smap_df = pd.concat(filtered_chunks, ignore_index=True)
        
        # Ensure datetime is parsed
        if 'datetime' in smap_df.columns:
            smap_df['datetime'] = pd.to_datetime(smap_df['datetime'])
        
        # Process SMAP columns: use surface moisture and mean soil temp
        smap_df = self._process_smap_columns(smap_df)
        
        self.logger.info(f"Loaded {len(smap_df):,} SMAP observations")
        
        return smap_df
    
    def _process_smap_columns(self, smap_df: pd.DataFrame) -> pd.DataFrame:
        """Process SMAP columns to unified schema"""
        
        # Use surface soil moisture
        if 'sm_surface' in smap_df.columns:
            smap_df['soil_moist_frac'] = smap_df['sm_surface']
        elif 'sm_rootzone' in smap_df.columns:
            smap_df['soil_moist_frac'] = smap_df['sm_rootzone']
        
        # Calculate mean soil temperature from available layers
        temp_layers = [col for col in smap_df.columns if col.startswith('soil_temp_layer')]
        
        if temp_layers:
            # Take mean of non-NaN values across layers
            smap_df['soil_temp_c'] = smap_df[temp_layers].mean(axis=1, skipna=True)
        
        # Keep essential columns
        essential_cols = ['datetime', 'latitude', 'longitude', 'soil_temp_c', 'soil_moist_frac']
        
        # Add columns that exist
        keep_cols = [col for col in essential_cols if col in smap_df.columns]
        
        return smap_df[keep_cols]
    
    def _month_to_season(self, month: int) -> str:
        """Convert month to season"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def _enforce_completeness(self, unified_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Enforce completeness guarantee"""
        
        initial_count = len(unified_df)
        
        critical_columns = ['displacement_m', 'soil_temp_c', 'soil_moist_frac']
        
        complete_mask = pd.Series(True, index=unified_df.index)
        
        for col in critical_columns:
            if col in unified_df.columns:
                complete_mask &= unified_df[col].notna()
        
        clean_df = unified_df[complete_mask].copy()
        
        rejected_count = initial_count - len(clean_df)
        
        if rejected_count > 0:
            self.logger.warning(f"Rejected {rejected_count:,} incomplete records")
        
        return clean_df, rejected_count
    
    def _save_consolidated_output(self, unified_df: pd.DataFrame):
        """Save consolidated dataset"""
        
        self.logger.info(f"Saving {len(unified_df):,} records to {self.output_path}")
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet with compression
        unified_df.to_parquet(
            self.output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"Output saved: {file_size_mb:.2f} MB")
