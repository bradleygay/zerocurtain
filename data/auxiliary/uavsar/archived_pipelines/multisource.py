#!/usr/bin/env python3
"""
Multi-Source Consolidation Module
Version: 2.0.0

Consolidates displacement and SMAP data into unified dataset with
guaranteed completeness (no null values in critical measurements).
"""

import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Optional
import logging
from datetime import datetime
import glob


class MultiSourceConsolidator:
    """
    Multi-source data consolidator with completeness guarantee.
    
    This consolidator ensures the final output contains ONLY complete records
    where every observation has displacement, soil temperature, and soil moisture.
    """
    
    def __init__(self,
                 displacement_dir: str,
                 smap_path: str,
                 output_path: str,
                 spatial_joiner: 'SpatioTemporalJoiner',
                 require_complete_records: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize consolidator
        
        Parameters:
        -----------
        displacement_dir : str
            Directory containing displacement GeoTIFFs
        smap_path : str
            Path to SMAP master parquet file
        output_path : str
            Path for final consolidated output
        spatial_joiner : SpatioTemporalJoiner
            Configured spatial-temporal joiner
        require_complete_records : bool
            Enforce completeness guarantee
        logger : logging.Logger, optional
            Logger instance
        """
        self.displacement_dir = Path(displacement_dir)
        self.smap_path = Path(smap_path)
        self.output_path = Path(output_path)
        self.spatial_joiner = spatial_joiner
        self.require_complete_records = require_complete_records
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate inputs
        if not self.displacement_dir.exists():
            raise FileNotFoundError(f"Displacement directory not found: {displacement_dir}")
        
        if not self.smap_path.exists():
            raise FileNotFoundError(f"SMAP file not found: {smap_path}")
        
        # Statistics
        self.stats = {
            'displacement_files_found': 0,
            'displacement_records_loaded': 0,
            'smap_records_loaded': 0,
            'total_complete_records': 0,
            'rejected_incomplete': 0,
            'match_success_rate': 0.0
        }
    
    def consolidate_all_sources(self) -> Dict:
        """
        Consolidate displacement and SMAP data with completeness guarantee
        
        Returns:
        --------
        dict : Consolidation statistics
        """
        self.logger.info("="*80)
        self.logger.info("MULTI-SOURCE CONSOLIDATION")
        self.logger.info("="*80)
        
        consolidation_start = datetime.now()
        
        # Step 1: Load displacement data
        self.logger.info("Step 1: Loading displacement data...")
        displacement_df = self._load_displacement_data()
        self.stats['displacement_records_loaded'] = len(displacement_df)
        
        # Step 2: Load SMAP data
        self.logger.info("Step 2: Loading SMAP data...")
        smap_df = self._load_smap_data()
        self.stats['smap_records_loaded'] = len(smap_df)
        
        # Step 3: Spatiotemporal join
        self.logger.info("Step 3: Performing spatiotemporal join...")
        unified_df = self.spatial_joiner.join_displacement_soil(displacement_df, smap_df)
        
        # Step 4: Enforce completeness (if required)
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
        self.logger.info("Step 5: Saving consolidated output...")
        self._save_consolidated_output(unified_df)
        
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
        """
        Load all displacement GeoTIFFs and extract metadata
        
        Returns:
        --------
        pd.DataFrame : Displacement observations
        """
        # Find all displacement GeoTIFFs
        displacement_files = list(self.displacement_dir.rglob("*_displacement_30m.tif"))
        
        if not displacement_files:
            raise RuntimeError(f"No displacement files found in {self.displacement_dir}")
        
        self.stats['displacement_files_found'] = len(displacement_files)
        self.logger.info(f"Found {len(displacement_files):,} displacement files")
        
        # Extract data from each file
        import rasterio
        
        all_records = []
        
        for idx, disp_file in enumerate(displacement_files):
            if idx % 100 == 0 and idx > 0:
                self.logger.info(f"  Processed {idx}/{len(displacement_files)} files")
            
            try:
                with rasterio.open(disp_file) as src:
                    # Read displacement values
                    displacement = src.read(1, masked=True)
                    
                    # Get geotransform
                    transform = src.transform
                    
                    # Create coordinate arrays
                    rows, cols = np.where(~displacement.mask)
                    
                    if len(rows) == 0:
                        continue
                    
                    # Convert pixel coordinates to geographic coordinates
                    lons, lats = rasterio.transform.xy(transform, rows, cols)
                    
                    # Extract displacement values
                    disp_values = displacement.data[rows, cols]
                    
                    # Extract metadata from filename
                    metadata = self._extract_displacement_metadata(disp_file)
                    
                    # Create records
                    for lat, lon, disp in zip(lats, lons, disp_values):
                        if not np.isnan(disp):
                            record = {
                                'datetime': metadata.get('datetime'),
                                'latitude': lat,
                                'longitude': lon,
                                'displacement_m': disp,
                                'source': metadata.get('source', 'UAVSAR'),
                                'polarization': metadata.get('polarization', 'unknown'),
                                'file_path': str(disp_file)
                            }
                            all_records.append(record)
            
            except Exception as e:
                self.logger.warning(f"Error processing {disp_file}: {e}")
                continue
        
        if not all_records:
            raise RuntimeError("No valid displacement records extracted")
        
        df = pd.DataFrame(all_records)
        
        # Ensure datetime is parsed
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Add year and season
        df['year'] = df['datetime'].dt.year
        df['season'] = df['datetime'].dt.month.map(self._month_to_season)
        
        self.logger.info(f"Loaded {len(df):,} displacement observations")
        
        return df
    
    def _extract_displacement_metadata(self, file_path: Path) -> Dict:
        """
        Extract metadata from displacement filename
        
        Expected format: SITE_REF_SEC_POL_displacement_30m.tif
        
        Parameters:
        -----------
        file_path : Path
            Displacement file path
        
        Returns:
        --------
        dict : Extracted metadata
        """
        filename = file_path.stem
        
        # Try to parse filename components
        parts = filename.split('_')
        
        metadata = {}
        
        # Look for date patterns
        import re
        date_patterns = [
            r'(\d{6})',  # YYMMDD
            r'(\d{8})',  # YYYYMMDD
        ]
        
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
        
        # If no date found, use file modification time
        if 'datetime' not in metadata:
            metadata['datetime'] = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        # Extract polarization if present
        for pol in ['HH', 'HV', 'VH', 'VV']:
            if pol in filename:
                metadata['polarization'] = pol
                break
        
        # Determine source
        if 'nisar' in filename.lower():
            metadata['source'] = 'NISAR'
        else:
            metadata['source'] = 'UAVSAR'
        
        return metadata
    
    def _load_smap_data(self) -> pd.DataFrame:
        """
        Load SMAP master parquet file
        
        Returns:
        --------
        pd.DataFrame : SMAP observations
        """
        self.logger.info(f"Loading SMAP data from {self.smap_path}")
        
        # Read parquet file
        table = pq.read_table(self.smap_path)
        df = table.to_pandas()
        
        # Ensure datetime is parsed
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Filter for soil temperature and moisture records
        if 'data_type' in df.columns:
            # Keep both temperature and moisture records
            df = df[df['data_type'].isin(['soil_temperature', 'soil_moisture'])].copy()
        
        # For each unique location/time, we need both temperature and moisture
        # Pivot to wide format to have both in same row
        df = self._pivot_smap_data(df)
        
        self.logger.info(f"Loaded {len(df):,} SMAP observation pairs")
        
        return df
    
    def _pivot_smap_data(self, smap_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot SMAP data so each record has both temperature and moisture
        
        Parameters:
        -----------
        smap_df : pd.DataFrame
            Long-format SMAP data
        
        Returns:
        --------
        pd.DataFrame : Wide-format with both measurements
        """
        # Group by location and time
        grouped = smap_df.groupby(['latitude', 'longitude', 'datetime'])
        
        unified_records = []
        
        for (lat, lon, dt), group in grouped:
            # Separate temperature and moisture records
            temp_records = group[group['data_type'] == 'soil_temperature']
            moist_records = group[group['data_type'] == 'soil_moisture']
            
            if len(temp_records) == 0 or len(moist_records) == 0:
                continue
            
            # Take first (or surface layer) measurements
            temp_row = temp_records.iloc[0]
            moist_row = moist_records.iloc[0]
            
            record = {
                'datetime': dt,
                'latitude': lat,
                'longitude': lon,
                'soil_temp_c': temp_row['soil_temp_c'],
                'soil_temp_c_std': temp_row.get('soil_temp_c_std', np.nan),
                'soil_temp_depth_m': temp_row.get('soil_temp_depth_m', np.nan),
                'soil_temp_depth_zone': temp_row.get('soil_temp_depth_zone', ''),
                'soil_moist_frac': moist_row['soil_moist_frac'],
                'soil_moist_frac_std': moist_row.get('soil_moist_frac_std', np.nan),
                'soil_moist_depth_m': moist_row.get('soil_moist_depth_m', np.nan),
                'qc_flag': temp_row.get('qc_flag', 'valid')
            }
            
            unified_records.append(record)
        
        return pd.DataFrame(unified_records)
    
    def _month_to_season(self, month: int) -> str:
        """Convert month number to season"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def _enforce_completeness(self, unified_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Enforce completeness guarantee: remove any records with null values
        
        Parameters:
        -----------
        unified_df : pd.DataFrame
            Unified records
        
        Returns:
        --------
        tuple : (clean_df, rejected_count)
        """
        self.logger.info("Enforcing data completeness guarantee...")
        
        initial_count = len(unified_df)
        
        # Define critical columns
        critical_columns = ['displacement_m', 'soil_temp_c', 'soil_moist_frac']
        
        # Create completeness mask
        complete_mask = pd.Series(True, index=unified_df.index)
        
        for col in critical_columns:
            complete_mask &= unified_df[col].notna()
        
        # Filter to complete records only
        clean_df = unified_df[complete_mask].copy()
        
        rejected_count = initial_count - len(clean_df)
        
        if rejected_count > 0:
            self.logger.warning(f"Rejected {rejected_count:,} incomplete records")
            self.logger.info(f"Retained {len(clean_df):,} complete records")
        else:
            self.logger.info("All records complete - no rejections")
        
        # Final validation
        for col in critical_columns:
            null_count = clean_df[col].isna().sum()
            if null_count > 0:
                raise RuntimeError(f"Completeness enforcement failed: {null_count} nulls in {col}")
        
        return clean_df, rejected_count
    
    def _save_consolidated_output(self, unified_df: pd.DataFrame):
        """
        Save consolidated dataset as compressed parquet
        
        Parameters:
        -----------
        unified_df : pd.DataFrame
            Unified records
        """
        self.logger.info(f"Saving {len(unified_df):,} records to {self.output_path}")
        
        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define schema for consistency with original fifth.py
        schema = pa.schema([
            pa.field('datetime', pa.timestamp('us')),
            pa.field('year', pa.int64()),
            pa.field('season', pa.string()),
            pa.field('latitude', pa.float64()),
            pa.field('longitude', pa.float64()),
            pa.field('source_displacement', pa.string()),
            pa.field('source_soil', pa.string()),
            pa.field('data_type', pa.string()),
            pa.field('displacement_m', pa.float64()),
            pa.field('displacement_m_std', pa.float64()),
            pa.field('soil_temp_c', pa.float64()),
            pa.field('soil_temp_c_std', pa.float64()),
            pa.field('soil_temp_depth_m', pa.float64()),
            pa.field('soil_temp_depth_zone', pa.string()),
            pa.field('soil_moist_frac', pa.float64()),
            pa.field('soil_moist_frac_std', pa.float64()),
            pa.field('soil_moist_depth_m', pa.float64()),
            pa.field('qc_flag_displacement', pa.string()),
            pa.field('qc_flag_soil', pa.string()),
            pa.field('spatial_distance_m', pa.float64()),
            pa.field('temporal_offset_hours', pa.float64())
        ])
        
        # Convert to PyArrow table
        table = pa.Table.from_pandas(unified_df, schema=schema)
        
        # Write with compression
        pq.write_table(
            table,
            self.output_path,
            compression='ZSTD',
            compression_level=9
        )
        
        # Log file size
        file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"Output saved: {file_size_mb:.2f} MB")
        
        # Save metadata
        self._save_metadata(unified_df)
    
    def _save_metadata(self, unified_df: pd.DataFrame):
        """Save metadata about consolidated dataset"""
        
        metadata = {
            'created': datetime.now().isoformat(),
            'total_records': len(unified_df),
            'temporal_extent': {
                'start': unified_df['datetime'].min().isoformat(),
                'end': unified_df['datetime'].max().isoformat()
            },
            'spatial_extent': {
                'lat_min': float(unified_df['latitude'].min()),
                'lat_max': float(unified_df['latitude'].max()),
                'lon_min': float(unified_df['longitude'].min()),
                'lon_max': float(unified_df['longitude'].max())
            },
            'data_completeness': {
                'guaranteed': self.require_complete_records,
                'null_displacement': int(unified_df['displacement_m'].isna().sum()),
                'null_temperature': int(unified_df['soil_temp_c'].isna().sum()),
                'null_moisture': int(unified_df['soil_moist_frac'].isna().sum())
            },
            'statistics': {
                'displacement': {
                    'mean': float(unified_df['displacement_m'].mean()),
                    'std': float(unified_df['displacement_m'].std()),
                    'min': float(unified_df['displacement_m'].min()),
                    'max': float(unified_df['displacement_m'].max())
                },
                'temperature': {
                    'mean': float(unified_df['soil_temp_c'].mean()),
                    'std': float(unified_df['soil_temp_c'].std()),
                    'min': float(unified_df['soil_temp_c'].min()),
                    'max': float(unified_df['soil_temp_c'].max())
                },
                'moisture': {
                    'mean': float(unified_df['soil_moist_frac'].mean()),
                    'std': float(unified_df['soil_moist_frac'].std()),
                    'min': float(unified_df['soil_moist_frac'].min()),
                    'max': float(unified_df['soil_moist_frac'].max())
                }
            }
        }
        
        metadata_path = self.output_path.with_suffix('.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved: {metadata_path}")