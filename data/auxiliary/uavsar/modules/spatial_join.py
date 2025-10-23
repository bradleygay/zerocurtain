#!/usr/bin/env python3
"""
modules/spatial_join.py
Spatial-Temporal Join Module - COMPLETE IMPLEMENTATION


Advanced spatiotemporal data fusion for displacement and soil property measurements.
Implements intelligent matching with completeness guarantee.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from datetime import timedelta
from typing import Tuple, List, Dict, Optional
import logging


class SpatioTemporalJoiner:
    """
    Sophisticated spatial-temporal joining for multi-source remote sensing data.
    """
    
    def __init__(self, 
                 temporal_window_days: int = 3,
                 spatial_tolerance_m: float = 100.0,
                 max_neighbors: int = 5,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize spatiotemporal joiner
        """
        self.temporal_window = timedelta(days=temporal_window_days)
        self.spatial_tolerance = spatial_tolerance_m
        self.max_neighbors = max_neighbors
        self.logger = logger or logging.getLogger(__name__)
        
        # Conversion constants
        self.meters_per_degree = 111000.0
        
        # Statistics
        self.stats = {
            'total_displacement_records': 0,
            'matched_records': 0,
            'unmatched_spatial': 0,
            'unmatched_temporal': 0,
            'unmatched_quality': 0
        }
    
    def join_displacement_soil(self, 
                              displacement_df: pd.DataFrame,
                              smap_df: pd.DataFrame) -> pd.DataFrame:
        """
        Join displacement and SMAP data with completeness guarantee.
        """
        self.logger.info("="*80)
        self.logger.info("SPATIOTEMPORAL JOIN WITH COMPLETENESS GUARANTEE")
        self.logger.info("="*80)
        
        self.stats['total_displacement_records'] = len(displacement_df)
        
        self.logger.info(f"Input displacement records: {len(displacement_df):,}")
        self.logger.info(f"Input SMAP records: {len(smap_df):,}")
        self.logger.info(f"Temporal window: ±{self.temporal_window.days} days")
        self.logger.info(f"Spatial tolerance: {self.spatial_tolerance:.0f} meters")
        
        # Validate input DataFrames
        self._validate_input_dataframes(displacement_df, smap_df)
        
        # Build spatial index for SMAP data
        self.logger.info("Building spatial index...")
        spatial_index = self._build_spatial_index(smap_df)
        
        # Process each displacement observation
        self.logger.info("Matching observations...")
        unified_records = []
        
        for idx, disp_row in displacement_df.iterrows():
            if idx % 10000 == 0 and idx > 0:
                self.logger.info(f"  Processed {idx:,}/{len(displacement_df):,} records "
                               f"({100*idx/len(displacement_df):.1f}%)")
            
            # Find matching SMAP observation
            match = self._find_best_match(disp_row, smap_df, spatial_index)
            
            if match is not None:
                # Create unified record
                unified_record = self._create_unified_record(disp_row, match)
                unified_records.append(unified_record)
                self.stats['matched_records'] += 1
        
        # Convert to DataFrame
        unified_df = pd.DataFrame(unified_records)
        
        # Log statistics
        self._log_join_statistics()
        
        # Critical validation: ensure no nulls
        self._validate_completeness(unified_df)
        
        self.logger.info("="*80)
        self.logger.info(f"JOIN COMPLETE: {len(unified_df):,} records with complete measurements")
        self.logger.info("="*80)
        
        return unified_df
    
    def _validate_input_dataframes(self, displacement_df: pd.DataFrame, smap_df: pd.DataFrame):
        """Validate input DataFrames have required columns"""
        
        required_disp_cols = ['datetime', 'latitude', 'longitude', 'displacement_m']
        required_smap_cols = ['datetime', 'latitude', 'longitude', 'soil_temp_c', 'soil_moist_frac']
        
        missing_disp = [col for col in required_disp_cols if col not in displacement_df.columns]
        missing_smap = [col for col in required_smap_cols if col not in smap_df.columns]
        
        if missing_disp:
            raise ValueError(f"Displacement DataFrame missing columns: {missing_disp}")
        
        if missing_smap:
            raise ValueError(f"SMAP DataFrame missing columns: {missing_smap}")
        
        # Ensure datetime columns are parsed
        if not pd.api.types.is_datetime64_any_dtype(displacement_df['datetime']):
            displacement_df['datetime'] = pd.to_datetime(displacement_df['datetime'])
        
        if not pd.api.types.is_datetime64_any_dtype(smap_df['datetime']):
            smap_df['datetime'] = pd.to_datetime(smap_df['datetime'])
    
    def _build_spatial_index(self, smap_df: pd.DataFrame) -> cKDTree:
        """Build k-d tree for efficient spatial queries"""
        coords = np.column_stack([
            smap_df['longitude'].values,
            smap_df['latitude'].values
        ])
        
        tree = cKDTree(coords)
        
        self.logger.info(f"Spatial index built with {len(coords):,} points")
        
        return tree
    
    def _find_best_match(self, 
                        disp_row: pd.Series,
                        smap_df: pd.DataFrame,
                        spatial_index: cKDTree) -> Optional[pd.Series]:
        """Find best matching SMAP observation for displacement record"""
        query_point = np.array([[disp_row['longitude'], disp_row['latitude']]])
        
        tolerance_degrees = self.spatial_tolerance / self.meters_per_degree
        
        try:
            distances, indices = spatial_index.query(
                query_point,
                k=self.max_neighbors,
                distance_upper_bound=tolerance_degrees
            )
        except Exception as e:
            self.logger.warning(f"Spatial query failed: {e}")
            return None
        
        if np.isscalar(distances):
            distances = [distances]
            indices = [indices]
        else:
            distances = distances[0]
            indices = indices[0]
        
        valid_mask = distances < tolerance_degrees
        valid_distances = distances[valid_mask]
        valid_indices = indices[valid_mask]
        
        if len(valid_indices) == 0:
            self.stats['unmatched_spatial'] += 1
            return None
        
        disp_time = pd.to_datetime(disp_row['datetime'])
        
        temporal_candidates = []
        
        for dist, idx in zip(valid_distances, valid_indices):
            if idx >= len(smap_df):
                continue
            
            smap_row = smap_df.iloc[idx]
            smap_time = pd.to_datetime(smap_row['datetime'])
            
            time_diff = abs(smap_time - disp_time)
            
            if time_diff <= self.temporal_window:
                temporal_candidates.append({
                    'row': smap_row,
                    'spatial_distance': dist * self.meters_per_degree,
                    'temporal_offset': time_diff,
                    'score': self._compute_match_score(dist, time_diff)
                })
        
        if not temporal_candidates:
            self.stats['unmatched_temporal'] += 1
            return None
        
        temporal_candidates.sort(key=lambda x: x['score'])
        
        best_match = temporal_candidates[0]
        
        if not self._validate_soil_measurements(best_match['row']):
            self.stats['unmatched_quality'] += 1
            return None
        
        best_match['row'] = best_match['row'].copy()
        best_match['row']['_spatial_distance_m'] = best_match['spatial_distance']
        best_match['row']['_temporal_offset_hours'] = best_match['temporal_offset'].total_seconds() / 3600
        
        return best_match['row']
    
    def _compute_match_score(self, spatial_distance: float, temporal_offset: timedelta) -> float:
        """Compute match quality score"""
        spatial_norm = spatial_distance / (self.spatial_tolerance / self.meters_per_degree)
        temporal_norm = temporal_offset.total_seconds() / self.temporal_window.total_seconds()
        
        score = 0.4 * spatial_norm + 0.6 * temporal_norm
        
        return score
    
    def _validate_soil_measurements(self, smap_row: pd.Series) -> bool:
        """Validate SMAP measurements are not null and within reasonable range"""
        if pd.isna(smap_row['soil_temp_c']) or pd.isna(smap_row['soil_moist_frac']):
            return False
        
        if smap_row['soil_temp_c'] < -60.0 or smap_row['soil_temp_c'] > 60.0:
            return False
        
        if smap_row['soil_moist_frac'] < 0.0 or smap_row['soil_moist_frac'] > 1.0:
            return False
        
        return True
    
    def _create_unified_record(self, 
                              disp_row: pd.Series,
                              smap_row: pd.Series) -> Dict:
        """Create unified record combining displacement and soil measurements"""
        record = {
            'datetime': disp_row['datetime'],
            'year': disp_row.get('year', pd.to_datetime(disp_row['datetime']).year),
            'season': disp_row.get('season', self._get_season(pd.to_datetime(disp_row['datetime']))),
            'latitude': disp_row['latitude'],
            'longitude': disp_row['longitude'],
            'source_displacement': disp_row.get('source', 'UAVSAR'),
            'source_soil': 'SMAP',
            'data_type': 'unified',
            'displacement_m': disp_row['displacement_m'],
            'displacement_m_std': disp_row.get('displacement_m_std', np.nan),
            'soil_temp_c': smap_row['soil_temp_c'],
            'soil_temp_c_std': smap_row.get('soil_temp_c_std', np.nan),
            'soil_temp_depth_m': smap_row.get('soil_temp_depth_m', np.nan),
            'soil_temp_depth_zone': smap_row.get('soil_temp_depth_zone', ''),
            'soil_moist_frac': smap_row['soil_moist_frac'],
            'soil_moist_frac_std': smap_row.get('soil_moist_frac_std', np.nan),
            'soil_moist_depth_m': smap_row.get('soil_moist_depth_m', np.nan),
            'qc_flag_displacement': disp_row.get('qc_flag', 'valid'),
            'qc_flag_soil': smap_row.get('qc_flag', 'valid'),
            'spatial_distance_m': smap_row.get('_spatial_distance_m', np.nan),
            'temporal_offset_hours': smap_row.get('_temporal_offset_hours', np.nan)
        }
        
        return record
    
    def _get_season(self, dt: pd.Timestamp) -> str:
        """Get meteorological season from datetime"""
        month = dt.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def _validate_completeness(self, unified_df: pd.DataFrame):
        """CRITICAL: Validate no null values in unified DataFrame"""
        critical_columns = ['displacement_m', 'soil_temp_c', 'soil_moist_frac']
        
        for col in critical_columns:
            null_count = unified_df[col].isna().sum()
            if null_count > 0:
                raise RuntimeError(
                    f"Completeness validation failed: {null_count} null values in {col}"
                )
        
        self.logger.info(" Completeness validation passed: No null values in critical columns")
    
    def _log_join_statistics(self):
        """Log detailed join statistics"""
        total = self.stats['total_displacement_records']
        matched = self.stats['matched_records']
        
        self.logger.info("-"*80)
        self.logger.info("Join Statistics:")
        self.logger.info(f"  Total displacement records: {total:,}")
        self.logger.info(f"  Successfully matched: {matched:,} ({100*matched/total:.1f}%)")
        self.logger.info(f"  Unmatched (spatial): {self.stats['unmatched_spatial']:,}")
        self.logger.info(f"  Unmatched (temporal): {self.stats['unmatched_temporal']:,}")
        self.logger.info(f"  Unmatched (quality): {self.stats['unmatched_quality']:,}")
        self.logger.info("-"*80)
    
    def get_statistics(self) -> Dict:
        """Return join statistics"""
        return self.stats.copy()
