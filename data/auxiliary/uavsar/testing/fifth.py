#!/usr/bin/env python3

"""
Arctic Remote Sensing Data Consolidation - Production Version
Consolidates UAVSAR, NISAR, and SMAP 30m remote sensing data with comprehensive validation.

Purpose: Merge circumarctic displacement and soil property retrievals
Excludes: In situ measurements (remote sensing only)
"""

import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
import time
import os
import gc
import shutil
import glob
from datetime import datetime
from pathlib import Path
import traceback
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration with validation thresholds"""
    
    # File paths (modify these for your system)
    UAVSAR_PATH = '/Users/bradleygay/merged_uavsar_nisar.parquet'
    SMAP_PATH = '/Users/bradleygay/smap_master.parquet'
    OUTPUT_PATH = '/Users/bradleygay/final_remote_sensing_consolidated.parquet'
    TEMP_DIR = '/tmp/arctic_remote_sensing'
    
    # Physical validation thresholds
    DISPLACEMENT_MIN = -10.0  # meters (subsidence)
    DISPLACEMENT_MAX = 10.0   # meters (uplift)
    TEMP_MIN = -60.0          # °C (Arctic winter minimum)
    TEMP_MAX = 60.0           # °C (Surface maximum)
    MOISTURE_MIN = 0.0        # Decimal form (0%)
    MOISTURE_MAX = 1.0        # Decimal form (100%)
    
    # Kelvin conversion bounds (for detection)
    KELVIN_MIN = 200.0
    KELVIN_MAX = 400.0
    
    # Processing parameters
    BATCH_SIZE_UAVSAR = 500      # Row groups per batch
    BATCH_SIZE_SMAP = 200        # Row groups per batch
    CHUNK_SIZE_RECORDS = 5000000 # Records per chunk
    
    # Logging configuration
    LOG_INTERVAL = 10  # Log every N batches

# ============================================================================
# SCHEMA DEFINITION
# ============================================================================

def create_remote_sensing_schema():
    """
    Standardized schema for all remote sensing products.
    Ensures consistent structure across UAVSAR, NISAR, and SMAP.
    """
    return pa.schema([
        # Temporal metadata
        pa.field('datetime', pa.timestamp('us')),
        pa.field('year', pa.int64()),
        pa.field('season', pa.string()),
        
        # Spatial metadata
        pa.field('latitude', pa.float64()),
        pa.field('longitude', pa.float64()),
        
        # Source identification
        pa.field('source', pa.string()),           # 'UAVSAR', 'NISAR', 'SMAP'
        pa.field('data_type', pa.string()),        # 'displacement', 'soil_temperature', 'soil_moisture'
        
        # Displacement measurements (UAVSAR/NISAR)
        pa.field('displacement_m', pa.float64()),
        pa.field('displacement_m_std', pa.float64()),
        
        # Soil temperature measurements (SMAP)
        pa.field('soil_temp_c', pa.float64()),
        pa.field('soil_temp_c_std', pa.float64()),
        pa.field('soil_temp_depth_m', pa.float64()),
        pa.field('soil_temp_depth_zone', pa.string()),
        
        # Soil moisture measurements (SMAP)
        pa.field('soil_moist_frac', pa.float64()),
        pa.field('soil_moist_frac_std', pa.float64()),
        pa.field('soil_moist_depth_m', pa.float64()),
        
        # Quality flags
        pa.field('qc_flag', pa.string())  # 'valid', 'out_of_range', 'converted'
    ])

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_elapsed_time(start_time):
    """Format elapsed time as HH:MM:SS"""
    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def log_message(message: str, start_time=None, records_processed=0):
    """Enhanced logging with timestamps and processing rates"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if start_time and records_processed > 0:
        elapsed = time.time() - start_time
        if elapsed > 0:
            rate = records_processed / elapsed
            print(f"[{timestamp}] {message} | Rate: {rate:,.0f} rec/sec | Elapsed: {get_elapsed_time(start_time)}")
        else:
            print(f"[{timestamp}] {message}")
    else:
        print(f"[{timestamp}] {message}")

def get_season_from_month(month_values):
    """Convert month numbers to meteorological seasons"""
    seasons = []
    for m in month_values:
        if m in [12, 1, 2]:
            seasons.append('Winter')
        elif m in [3, 4, 5]:
            seasons.append('Spring')
        elif m in [6, 7, 8]:
            seasons.append('Summer')
        else:
            seasons.append('Fall')
    return seasons

# ============================================================================
# VALIDATION AND STANDARDIZATION FUNCTIONS
# ============================================================================

def validate_and_standardize_displacement(values, source='UAVSAR'):
    """
    Validate and standardize displacement measurements.
    
    Displacement convention:
    - Negative values: subsidence (ground lowering)
    - Positive values: uplift (ground rising)
    
    Returns:
    --------
    standardized_values : np.ndarray
        Validated displacement in meters
    qc_flags : list
        Quality control flags for each measurement
    statistics : dict
        Processing statistics
    """
    if len(values) == 0:
        return values, [], {'total': 0, 'valid': 0, 'out_of_range': 0}
    
    values_np = np.array(values, dtype=np.float64)
    qc_flags = ['valid'] * len(values_np)
    
    # Check for unrealistic values
    out_of_range_mask = (values_np < Config.DISPLACEMENT_MIN) | (values_np > Config.DISPLACEMENT_MAX)
    out_of_range_count = np.sum(out_of_range_mask)
    
    if out_of_range_count > 0:
        log_message(f"    {source}: {out_of_range_count:,} displacement values out of range "
                   f"({Config.DISPLACEMENT_MIN}m to {Config.DISPLACEMENT_MAX}m)")
        # Mark as invalid but keep for analysis
        for idx in np.where(out_of_range_mask)[0]:
            qc_flags[idx] = 'out_of_range'
    
    # Apply physical constraints (clip to valid range)
    values_np = np.clip(values_np, Config.DISPLACEMENT_MIN, Config.DISPLACEMENT_MAX)
    
    statistics = {
        'total': len(values_np),
        'valid': len(values_np) - out_of_range_count,
        'out_of_range': out_of_range_count,
        'mean': np.nanmean(values_np),
        'std': np.nanstd(values_np),
        'min': np.nanmin(values_np),
        'max': np.nanmax(values_np)
    }
    
    return values_np, qc_flags, statistics

def validate_and_standardize_temperature(values, source='SMAP'):
    """
    Validate and standardize soil temperature measurements.
    
    Converts Kelvin to Celsius if detected.
    Applies physical constraints for Arctic environments.
    
    Returns:
    --------
    standardized_values : np.ndarray
        Validated temperature in °C
    qc_flags : list
        Quality control flags
    statistics : dict
        Processing statistics
    """
    if len(values) == 0:
        return values, [], {'total': 0, 'valid': 0, 'converted': 0, 'out_of_range': 0}
    
    values_np = np.array(values, dtype=np.float64)
    qc_flags = ['valid'] * len(values_np)
    converted_count = 0
    
    # Detect and convert Kelvin to Celsius
    kelvin_mask = (values_np >= Config.KELVIN_MIN) & (values_np <= Config.KELVIN_MAX)
    converted_count = np.sum(kelvin_mask)
    
    if converted_count > 0:
        log_message(f"    {source}: Converting {converted_count:,} temperature values from Kelvin to Celsius")
        values_np[kelvin_mask] = values_np[kelvin_mask] - 273.15
        for idx in np.where(kelvin_mask)[0]:
            qc_flags[idx] = 'converted'
    
    # Check for unrealistic temperatures
    out_of_range_mask = (values_np < Config.TEMP_MIN) | (values_np > Config.TEMP_MAX)
    out_of_range_count = np.sum(out_of_range_mask)
    
    if out_of_range_count > 0:
        log_message(f"    {source}: {out_of_range_count:,} temperature values out of range "
                   f"({Config.TEMP_MIN}°C to {Config.TEMP_MAX}°C)")
        for idx in np.where(out_of_range_mask)[0]:
            qc_flags[idx] = 'out_of_range'
    
    # Apply physical constraints
    values_np = np.clip(values_np, Config.TEMP_MIN, Config.TEMP_MAX)
    
    statistics = {
        'total': len(values_np),
        'valid': len(values_np) - out_of_range_count - converted_count,
        'converted': converted_count,
        'out_of_range': out_of_range_count,
        'mean': np.nanmean(values_np),
        'std': np.nanstd(values_np),
        'min': np.nanmin(values_np),
        'max': np.nanmax(values_np)
    }
    
    return values_np, qc_flags, statistics

def validate_and_standardize_moisture(values, source='SMAP'):
    """
    Validate and standardize soil moisture measurements.
    
    Converts percentage (1-100) to decimal fraction (0-1) if detected.
    Applies physical constraints (0-100% saturation).
    
    Returns:
    --------
    standardized_values : np.ndarray
        Validated moisture as decimal fraction (0-1)
    qc_flags : list
        Quality control flags
    statistics : dict
        Processing statistics
    """
    if len(values) == 0:
        return values, [], {'total': 0, 'valid': 0, 'converted': 0, 'out_of_range': 0}
    
    values_np = np.array(values, dtype=np.float64)
    qc_flags = ['valid'] * len(values_np)
    converted_count = 0
    
    # Detect and convert percentage to fraction
    percentage_mask = (values_np > 1.0) & (values_np <= 100.0)
    converted_count = np.sum(percentage_mask)
    
    if converted_count > 0:
        log_message(f"   {source}: Converting {converted_count:,} moisture values from percentage to fraction")
        values_np[percentage_mask] = values_np[percentage_mask] / 100.0
        for idx in np.where(percentage_mask)[0]:
            qc_flags[idx] = 'converted'
    
    # Check for unrealistic values
    out_of_range_mask = (values_np < Config.MOISTURE_MIN) | (values_np > Config.MOISTURE_MAX)
    out_of_range_count = np.sum(out_of_range_mask)
    
    if out_of_range_count > 0:
        log_message(f"    {source}: {out_of_range_count:,} moisture values out of range "
                   f"({Config.MOISTURE_MIN} to {Config.MOISTURE_MAX})")
        for idx in np.where(out_of_range_mask)[0]:
            qc_flags[idx] = 'out_of_range'
    
    # Apply physical constraints
    values_np = np.clip(values_np, Config.MOISTURE_MIN, Config.MOISTURE_MAX)
    
    statistics = {
        'total': len(values_np),
        'valid': len(values_np) - out_of_range_count - converted_count,
        'converted': converted_count,
        'out_of_range': out_of_range_count,
        'mean': np.nanmean(values_np),
        'std': np.nanstd(values_np),
        'min': np.nanmin(values_np),
        'max': np.nanmax(values_np)
    }
    
    return values_np, qc_flags, statistics

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def process_uavsar_nisar(file_path, batch_dir, start_time, source_name='UAVSAR'):
    """
    Process UAVSAR or NISAR displacement data with comprehensive validation.
    
    Parameters:
    -----------
    file_path : str
        Path to parquet file containing displacement measurements
    batch_dir : str
        Directory for batch output files
    start_time : float
        Processing start time for elapsed time calculation
    source_name : str
        'UAVSAR' or 'NISAR' for source identification
        
    Returns:
    --------
    batch_files : list
        List of batch file paths
    total_rows : int
        Total valid records processed
    statistics : dict
        Comprehensive processing statistics
    """
    log_message(f"  Processing {source_name} displacement data", start_time, 0)
    
    try:
        pf = pq.ParquetFile(file_path)
        total_row_groups = pf.num_row_groups
        
        log_message(f" {source_name}: {total_row_groups:,} row groups to process")
        
        batch_files = []
        schema = create_remote_sensing_schema()
        batch_idx = 0
        total_output_rows = 0
        
        # Statistics tracking
        stats = {
            'total_input_rows': 0,
            'total_output_rows': 0,
            'total_valid': 0,
            'total_out_of_range': 0,
            'displacement_stats': []
        }
        
        # Process in manageable batches
        row_groups_per_batch = Config.BATCH_SIZE_UAVSAR
        
        for rg_start in range(0, total_row_groups, row_groups_per_batch):
            rg_end = min(rg_start + row_groups_per_batch, total_row_groups)
            
            # Process batch of row groups
            batch_tables = []
            
            for rg_idx in range(rg_start, rg_end):
                try:
                    rg_table = pf.read_row_group(rg_idx)
                    stats['total_input_rows'] += len(rg_table)
                    
                    # Find displacement column
                    displacement_col = None
                    for col in ['thickness_m', 'thickness_m_standardized', 'displacement', 'displacement_m']:
                        if col in rg_table.column_names:
                            displacement_col = col
                            break
                    
                    if displacement_col is None:
                        del rg_table
                        continue
                    
                    # Filter valid data
                    displacement_values = rg_table[displacement_col]
                    valid_mask = pa.compute.is_valid(displacement_values)
                    
                    if pa.compute.sum(valid_mask).as_py() == 0:
                        del rg_table
                        continue
                    
                    filtered_table = pa.compute.filter(rg_table, valid_mask)
                    num_valid = len(filtered_table)
                    
                    if num_valid == 0:
                        del rg_table, filtered_table
                        continue
                    
                    # Extract displacement values
                    displacement_raw = filtered_table[displacement_col].to_numpy()
                    
                    # Validate and standardize
                    displacement_std, qc_flags, disp_stats = validate_and_standardize_displacement(
                        displacement_raw, source=source_name
                    )
                    stats['displacement_stats'].append(disp_stats)
                    stats['total_valid'] += disp_stats['valid']
                    stats['total_out_of_range'] += disp_stats['out_of_range']
                    
                    # Process datetime
                    if 'datetime' in filtered_table.column_names:
                        datetime_raw = filtered_table['datetime'].to_numpy()
                        datetime_arr = pa.array(datetime_raw, type=pa.timestamp('us'))
                        datetime_series = pd.to_datetime(datetime_raw)
                        year_arr = pa.array(datetime_series.year.values, type=pa.int64())
                        season_arr = pa.array(get_season_from_month(datetime_series.month.values), type=pa.string())
                    else:
                        datetime_arr = pa.nulls(num_valid, pa.timestamp('us'))
                        year_arr = pa.nulls(num_valid, pa.int64())
                        season_arr = pa.nulls(num_valid, pa.string())
                    
                    # Extract coordinates
                    lat_arr = None
                    lon_arr = None
                    for lat_col in ['latitude', 'lat']:
                        if lat_col in filtered_table.column_names:
                            lat_arr = filtered_table[lat_col].cast(pa.float64())
                            break
                    if lat_arr is None:
                        lat_arr = pa.nulls(num_valid, pa.float64())
                        
                    for lon_col in ['longitude', 'lon']:
                        if lon_col in filtered_table.column_names:
                            lon_arr = filtered_table[lon_col].cast(pa.float64())
                            break
                    if lon_arr is None:
                        lon_arr = pa.nulls(num_valid, pa.float64())
                    
                    # Create standardized output table
                    processed_table = pa.table([
                        datetime_arr,
                        year_arr,
                        season_arr,
                        lat_arr,
                        lon_arr,
                        pa.array([source_name] * num_valid, type=pa.string()),
                        pa.array(['displacement'] * num_valid, type=pa.string()),
                        pa.array(displacement_raw, type=pa.float64()),
                        pa.array(displacement_std, type=pa.float64()),
                        pa.nulls(num_valid, pa.float64()),  # soil_temp_c
                        pa.nulls(num_valid, pa.float64()),  # soil_temp_c_std
                        pa.nulls(num_valid, pa.float64()),  # soil_temp_depth_m
                        pa.nulls(num_valid, pa.string()),   # soil_temp_depth_zone
                        pa.nulls(num_valid, pa.float64()),  # soil_moist_frac
                        pa.nulls(num_valid, pa.float64()),  # soil_moist_frac_std
                        pa.nulls(num_valid, pa.float64()),  # soil_moist_depth_m
                        pa.array(qc_flags, type=pa.string())
                    ], schema=schema)
                    
                    batch_tables.append(processed_table)
                    
                    del rg_table, filtered_table, processed_table
                    
                except Exception as e:
                    log_message(f"    Error processing row group {rg_idx}: {e}")
                    continue
            
            # Combine and write batch
            if batch_tables:
                combined_table = pa.concat_tables(batch_tables)
                batch_file = os.path.join(batch_dir, f"{source_name.lower()}_{batch_idx:04d}.parquet")
                pq.write_table(combined_table, batch_file, compression="ZSTD")
                batch_files.append(batch_file)
                
                batch_rows = len(combined_table)
                total_output_rows += batch_rows
                stats['total_output_rows'] += batch_rows
                
                # Progress logging
                if batch_idx % Config.LOG_INTERVAL == 0:
                    progress = (rg_end / total_row_groups) * 100
                    log_message(
                        f"  {source_name} batch {batch_idx+1}: {progress:.1f}% complete, "
                        f"{total_output_rows:,} valid records",
                        start_time, total_output_rows
                    )
                
                del combined_table, batch_tables
                gc.collect()
            
            batch_idx += 1
    
    except Exception as e:
        log_message(f" ERROR in {source_name} processing: {e}")
        traceback.print_exc()
        return [], 0, {}
    
    # Final statistics
    log_message(
        f" {source_name} complete: {len(batch_files)} batches, "
        f"{total_output_rows:,} valid records",
        start_time, total_output_rows
    )
    
    # Aggregate displacement statistics
    if stats['displacement_stats']:
        all_means = [s['mean'] for s in stats['displacement_stats']]
        all_stds = [s['std'] for s in stats['displacement_stats']]
        all_mins = [s['min'] for s in stats['displacement_stats']]
        all_maxs = [s['max'] for s in stats['displacement_stats']]
        
        log_message(f" {source_name} Displacement Statistics:")
        log_message(f"   Mean: {np.nanmean(all_means):.3f} m ± {np.nanmean(all_stds):.3f} m")
        log_message(f"   Range: [{np.nanmin(all_mins):.3f}, {np.nanmax(all_maxs):.3f}] m")
        log_message(f"   Valid: {stats['total_valid']:,} / {stats['total_input_rows']:,} "
                   f"({100*stats['total_valid']/max(stats['total_input_rows'],1):.1f}%)")
    
    return batch_files, total_output_rows, stats

def process_smap(file_path, batch_dir, start_time, records_processed):
    """
    Process SMAP soil temperature and moisture data with comprehensive validation.
    
    Parameters:
    -----------
    file_path : str
        Path to SMAP parquet file
    batch_dir : str
        Directory for batch output files
    start_time : float
        Processing start time
    records_processed : int
        Cumulative records processed (for rate calculation)
        
    Returns:
    --------
    batch_files : list
        List of batch file paths
    total_rows : int
        Total valid records processed
    statistics : dict
        Comprehensive processing statistics
    """
    log_message(f" Processing SMAP soil data", start_time, records_processed)
    
    # SMAP layer definitions
    soil_temp_layers = {
        'soil_temp_layer1': (0.1, 'surface'),
        'soil_temp_layer2': (0.2, 'shallow'),
        'soil_temp_layer3': (0.4, 'intermediate'),
        'soil_temp_layer4': (0.75, 'deep'),
        'soil_temp_layer5': (1.5, 'very_deep'),
        'soil_temp_layer6': (10.0, 'very_deep')
    }
    
    moisture_layers = {
        'sm_surface': 0.05,
        'sm_rootzone': 1.0
    }
    
    try:
        pf = pq.ParquetFile(file_path)
        total_row_groups = pf.num_row_groups
        log_message(f" SMAP: {total_row_groups:,} row groups to process")
        
        batch_files = []
        schema = create_remote_sensing_schema()
        batch_idx = 0
        total_output_rows = 0
        
        # Statistics tracking
        stats = {
            'total_input_rows': 0,
            'total_output_rows': 0,
            'temperature_layers_processed': 0,
            'moisture_layers_processed': 0,
            'temp_stats': [],
            'moisture_stats': []
        }
        
        # Process in batches
        row_groups_per_batch = Config.BATCH_SIZE_SMAP
        
        for rg_start in range(0, total_row_groups, row_groups_per_batch):
            rg_end = min(rg_start + row_groups_per_batch, total_row_groups)
            
            all_layer_tables = []
            
            # Process each row group in batch
            for rg_idx in range(rg_start, rg_end):
                try:
                    rg_table = pf.read_row_group(rg_idx)
                    stats['total_input_rows'] += len(rg_table)
                    
                    # Extract common metadata
                    datetime_raw = rg_table['datetime'].to_numpy() if 'datetime' in rg_table.column_names else None
                    if datetime_raw is None:
                        del rg_table
                        continue
                    
                    # Extract coordinates
                    lat_raw = None
                    lon_raw = None
                    for lat_col in ['lat', 'latitude']:
                        if lat_col in rg_table.column_names:
                            lat_raw = rg_table[lat_col].to_numpy()
                            break
                    for lon_col in ['lon', 'longitude']:
                        if lon_col in rg_table.column_names:
                            lon_raw = rg_table[lon_col].to_numpy()
                            break
                    
                    if lat_raw is None or lon_raw is None:
                        del rg_table
                        continue
                    
                    # Process soil temperature layers
                    for layer_col, (depth, zone) in soil_temp_layers.items():
                        if layer_col not in rg_table.column_names:
                            continue
                        
                        layer_values = rg_table[layer_col]
                        valid_mask = pa.compute.is_valid(layer_values)
                        
                        if pa.compute.sum(valid_mask).as_py() == 0:
                            continue
                        
                        valid_indices = pa.compute.indices_nonzero(valid_mask).to_numpy()
                        if len(valid_indices) == 0:
                            continue
                        
                        valid_temps = layer_values.to_numpy()[valid_indices]
                        valid_datetime = datetime_raw[valid_indices]
                        valid_lat = lat_raw[valid_indices]
                        valid_lon = lon_raw[valid_indices]
                        
                        # Validate and standardize temperature
                        temp_std, qc_flags, temp_stats = validate_and_standardize_temperature(
                            valid_temps, source=f'SMAP_{layer_col}'
                        )
                        stats['temp_stats'].append(temp_stats)
                        stats['temperature_layers_processed'] += 1
                        
                        # Create temporal arrays
                        num_valid = len(valid_indices)
                        datetime_series = pd.to_datetime(valid_datetime)
                        
                        layer_table = pa.table([
                            pa.array(valid_datetime, type=pa.timestamp('us')),
                            pa.array(datetime_series.year.values, type=pa.int64()),
                            pa.array(get_season_from_month(datetime_series.month.values), type=pa.string()),
                            pa.array(valid_lat, type=pa.float64()),
                            pa.array(valid_lon, type=pa.float64()),
                            pa.array(['SMAP'] * num_valid, type=pa.string()),
                            pa.array(['soil_temperature'] * num_valid, type=pa.string()),
                            pa.nulls(num_valid, pa.float64()),  # displacement_m
                            pa.nulls(num_valid, pa.float64()),  # displacement_m_std
                            pa.array(valid_temps, type=pa.float64()),
                            pa.array(temp_std, type=pa.float64()),
                            pa.array([depth] * num_valid, type=pa.float64()),
                            pa.array([zone] * num_valid, type=pa.string()),
                            pa.nulls(num_valid, pa.float64()),  # soil_moist_frac
                            pa.nulls(num_valid, pa.float64()),  # soil_moist_frac_std
                            pa.nulls(num_valid, pa.float64()),  # soil_moist_depth_m
                            pa.array(qc_flags, type=pa.string())
                        ], schema=schema)
                        
                        all_layer_tables.append(layer_table)
                        del layer_table
                    
                    # Process moisture layers
                    for moisture_col, depth in moisture_layers.items():
                        if moisture_col not in rg_table.column_names:
                            continue
                        
                        moisture_values = rg_table[moisture_col]
                        valid_mask = pa.compute.is_valid(moisture_values)
                        
                        if pa.compute.sum(valid_mask).as_py() == 0:
                            continue
                        
                        valid_indices = pa.compute.indices_nonzero(valid_mask).to_numpy()
                        if len(valid_indices) == 0:
                            continue
                        
                        valid_moisture = moisture_values.to_numpy()[valid_indices]
                        valid_datetime = datetime_raw[valid_indices]
                        valid_lat = lat_raw[valid_indices]
                        valid_lon = lon_raw[valid_indices]
                        
                        # Validate and standardize moisture
                        moisture_std, qc_flags, moist_stats = validate_and_standardize_moisture(
                            valid_moisture, source=f'SMAP_{moisture_col}'
                        )
                        stats['moisture_stats'].append(moist_stats)
                        stats['moisture_layers_processed'] += 1
                        
                        # Create temporal arrays
                        num_valid = len(valid_indices)
                        datetime_series = pd.to_datetime(valid_datetime)
                        
                        moisture_table = pa.table([
                            pa.array(valid_datetime, type=pa.timestamp('us')),
                            pa.array(datetime_series.year.values, type=pa.int64()),
                            pa.array(get_season_from_month(datetime_series.month.values), type=pa.string()),
                            pa.array(valid_lat, type=pa.float64()),
                            pa.array(valid_lon, type=pa.float64()),
                            pa.array(['SMAP'] * num_valid, type=pa.string()),
                            pa.array(['soil_moisture'] * num_valid, type=pa.string()),
                            pa.nulls(num_valid, pa.float64()),  # displacement_m
                            pa.nulls(num_valid, pa.float64()),  # displacement_m_std
                            pa.nulls(num_valid, pa.float64()),  # soil_temp_c
                            pa.nulls(num_valid, pa.float64()),  # soil_temp_c_std
                            pa.nulls(num_valid, pa.float64()),  # soil_temp_depth_m
                            pa.nulls(num_valid, pa.string()),   # soil_temp_depth_zone
                            pa.array(valid_moisture, type=pa.float64()),
                            pa.array(moisture_std, type=pa.float64()),
                            pa.array([depth] * num_valid, type=pa.float64()),
                            pa.array(qc_flags, type=pa.string())
                        ], schema=schema)
                        
                        all_layer_tables.append(moisture_table)
                        del moisture_table
                    
                    del rg_table
                    gc.collect()
                    
                except Exception as e:
                    log_message(f"    Error processing SMAP row group {rg_idx}: {e}")
                    continue
            
            # Combine and write batch
            if all_layer_tables:
                combined_table = pa.concat_tables(all_layer_tables)
                batch_file = os.path.join(batch_dir, f"smap_{batch_idx:04d}.parquet")
                pq.write_table(combined_table, batch_file, compression="ZSTD")
                batch_files.append(batch_file)
                
                batch_rows = len(combined_table)
                total_output_rows += batch_rows
                stats['total_output_rows'] += batch_rows
                
                # Progress logging
                if batch_idx % Config.LOG_INTERVAL == 0:
                    progress = (rg_end / total_row_groups) * 100
                    log_message(
                        f" SMAP batch {batch_idx+1}: {progress:.1f}% complete, "
                        f"{total_output_rows:,} records",
                        start_time, records_processed + total_output_rows
                    )
                
                batch_idx += 1
                del combined_table, all_layer_tables
                gc.collect()
    
    except Exception as e:
        log_message(f" ERROR in SMAP processing: {e}")
        traceback.print_exc()
        return [], 0, {}
    
    # Final statistics
    log_message(
        f" SMAP complete: {len(batch_files)} batches, "
        f"{total_output_rows:,} records",
        start_time, records_processed + total_output_rows
    )
    
    # Aggregate statistics
    if stats['temp_stats']:
        all_temp_means = [s['mean'] for s in stats['temp_stats']]
        all_temp_converted = sum([s['converted'] for s in stats['temp_stats']])
        log_message(f"  SMAP Temperature Statistics:")
        log_message(f"   Layers processed: {stats['temperature_layers_processed']}")
        log_message(f"   Mean temperature: {np.nanmean(all_temp_means):.2f}°C")
        log_message(f"   Converted from Kelvin: {all_temp_converted:,}")
    
    if stats['moisture_stats']:
        all_moist_means = [s['mean'] for s in stats['moisture_stats']]
        all_moist_converted = sum([s['converted'] for s in stats['moisture_stats']])
        log_message(f" SMAP Moisture Statistics:")
        log_message(f"   Layers processed: {stats['moisture_layers_processed']}")
        log_message(f"   Mean moisture: {np.nanmean(all_moist_means):.3f} (fraction)")
        log_message(f"   Converted from percentage: {all_moist_converted:,}")
    
    return batch_files, total_output_rows, stats

def stream_merge_parquet(files, output_path, start_time):
    """
    Efficient streaming merge of batch files into final consolidated dataset.
    
    Parameters:
    -----------
    files : list
        List of batch file paths to merge
    output_path : str
        Path for final consolidated output file
    start_time : float
        Processing start time
        
    Returns:
    --------
    success : bool
        True if merge completed successfully
    """
    log_message(f" Streaming merge of {len(files)} batch files", start_time, 0)
    log_message(f" Output: {output_path}")

    writer = None
    total_rows = 0
    file_count = 0

    for i, file in enumerate(files):
        if i % 10 == 0:
            log_message(f"[{get_elapsed_time(start_time)}] Merging file {i+1}/{len(files)}: {os.path.basename(file)}")
        
        try:
            reader = pq.ParquetFile(file)
            for rg in range(reader.num_row_groups):
                row_group = reader.read_row_group(rg)
                
                if writer is None:
                    writer = pq.ParquetWriter(output_path, row_group.schema, compression="ZSTD")
                
                writer.write_table(row_group)
                total_rows += row_group.num_rows
            
            file_count += 1
            
        except Exception as e:
            log_message(f"  Error merging {file}: {e}")
            continue

    if writer:
        writer.close()

    log_message(f" Streaming merge complete: {total_rows:,} rows from {file_count} files", 
               start_time, total_rows)
    return True

# ============================================================================
# MAIN CONSOLIDATION FUNCTION
# ============================================================================

def consolidate_remote_sensing():
    """
    Main consolidation workflow for Arctic remote sensing data.
    
    Processes and merges:
    - UAVSAR 30m displacement maps
    - NISAR 30m displacement maps
    - SMAP 30m soil temperature and moisture
    
    Excludes: In situ measurements
    
    Returns:
    --------
    success : bool
        True if consolidation completed successfully
    """
    start_time = time.time()
    
    log_message("=" * 80)
    log_message(" ARCTIC REMOTE SENSING DATA CONSOLIDATION")
    log_message("=" * 80)
    log_message(f"Version: Production v1.0")
    log_message(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Configuration:")
    log_message(f"  Displacement range: [{Config.DISPLACEMENT_MIN}, {Config.DISPLACEMENT_MAX}] m")
    log_message(f"  Temperature range: [{Config.TEMP_MIN}, {Config.TEMP_MAX}] °C")
    log_message(f"  Moisture range: [{Config.MOISTURE_MIN}, {Config.MOISTURE_MAX}] (fraction)")
    log_message("=" * 80)
    
    # Setup directories
    TEMP_DIR = Config.TEMP_DIR
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    BATCH_DIR = os.path.join(TEMP_DIR, "batches")
    os.makedirs(BATCH_DIR)
    
    # Verify input files
    log_message("\n Verifying input files...")
    input_files = {
        'UAVSAR/NISAR': Config.UAVSAR_PATH,
        'SMAP': Config.SMAP_PATH
    }
    
    for name, path in input_files.items():
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024**3)
            log_message(f"   {name}: {size_gb:.2f} GB - {path}")
        else:
            log_message(f"   {name}: NOT FOUND - {path}")
            log_message(f"\n ERROR: Missing input file. Aborting.")
            return False
    
    all_batch_files = []
    total_records = 0
    all_statistics = {}
    
    try:
        # PHASE 1: Process UAVSAR/NISAR displacement data
        log_message("\n" + "=" * 80)
        log_message("PHASE 1: PROCESSING DISPLACEMENT DATA")
        log_message("=" * 80)
        
        if os.path.exists(Config.UAVSAR_PATH):
            batch_files, records, stats = process_uavsar_nisar(
                Config.UAVSAR_PATH, 
                BATCH_DIR, 
                start_time,
                source_name='UAVSAR'
            )
            all_batch_files.extend(batch_files)
            total_records += records
            all_statistics['UAVSAR'] = stats
        
        # PHASE 2: Process SMAP soil data
        log_message("\n" + "=" * 80)
        log_message("PHASE 2: PROCESSING SMAP SOIL DATA")
        log_message("=" * 80)
        
        if os.path.exists(Config.SMAP_PATH):
            batch_files, records, stats = process_smap(
                Config.SMAP_PATH, 
                BATCH_DIR, 
                start_time, 
                total_records
            )
            all_batch_files.extend(batch_files)
            total_records += records
            all_statistics['SMAP'] = stats
        
        # PHASE 3: Merge all batches
        log_message("\n" + "=" * 80)
        log_message("PHASE 3: MERGING ALL BATCHES")
        log_message("=" * 80)
        
        if all_batch_files:
            log_message(f" Total batches to merge: {len(all_batch_files)}")
            success = stream_merge_parquet(all_batch_files, Config.OUTPUT_PATH, start_time)
            
            if not success:
                log_message(" Merge failed")
                return False
        else:
            log_message(" No batch files created!")
            return False
        
        # PHASE 4: Cleanup and verification
        log_message("\n" + "=" * 80)
        log_message("PHASE 4: CLEANUP AND VERIFICATION")
        log_message("=" * 80)
        
        try:
            shutil.rmtree(TEMP_DIR)
            log_message(f"🧹 Cleaned up temporary directory: {TEMP_DIR}")
        except Exception as e:
            log_message(f"  Could not remove temp directory: {e}")
        
        # Final verification and statistics
        if os.path.exists(Config.OUTPUT_PATH):
            final_size_gb = os.path.getsize(Config.OUTPUT_PATH) / (1024**3)
            total_time = time.time() - start_time
            
            # Read sample for verification
            sample_table = pq.read_table(Config.OUTPUT_PATH, columns=['source', 'data_type'])
            sample_df = sample_table.to_pandas()
            source_counts = sample_df['source'].value_counts()
            data_type_counts = sample_df['data_type'].value_counts()
            
            log_message("\n" + "=" * 80)
            log_message(" CONSOLIDATION COMPLETE")
            log_message("=" * 80)
            log_message(f" Total Records: {total_records:,}")
            log_message(f" Total Time: {get_elapsed_time(start_time)}")
            log_message(f" Average Rate: {total_records/total_time:,.0f} records/second")
            log_message(f" Output: {Config.OUTPUT_PATH}")
            log_message(f" Size: {final_size_gb:.2f} GB")
            log_message(f"\n Data Distribution:")
            for source, count in source_counts.items():
                pct = (count / len(sample_df)) * 100
                log_message(f"   {source}: {count:,} records ({pct:.1f}%)")
            log_message(f"\n Data Types:")
            for data_type, count in data_type_counts.items():
                pct = (count / len(sample_df)) * 100
                log_message(f"   {data_type}: {count:,} records ({pct:.1f}%)")
            
            # Save processing statistics
            stats_file = Config.OUTPUT_PATH.replace('.parquet', '_statistics.json')
            with open(stats_file, 'w') as f:
                json.dump({
                    'processing_time': total_time,
                    'total_records': total_records,
                    'output_size_gb': final_size_gb,
                    'sources': {k: v for k, v in source_counts.items()},
                    'data_types': {k: v for k, v in data_type_counts.items()},
                    'statistics': all_statistics,
                    'configuration': {
                        'displacement_range': [Config.DISPLACEMENT_MIN, Config.DISPLACEMENT_MAX],
                        'temperature_range': [Config.TEMP_MIN, Config.TEMP_MAX],
                        'moisture_range': [Config.MOISTURE_MIN, Config.MOISTURE_MAX]
                    }
                }, f, indent=2)
            log_message(f" Statistics saved: {stats_file}")
            
            log_message("=" * 80)
            
            return True
        else:
            log_message(" Output file not created")
            return False
        
    except Exception as e:
        log_message(f" CRITICAL ERROR: {str(e)}")
        traceback.print_exc()
        return False

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Arctic Remote Sensing Data Consolidation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python arctic_remote_sensing_consolidation.py
  python arctic_remote_sensing_consolidation.py --uavsar /path/to/uavsar.parquet --smap /path/to/smap.parquet
  
Configuration:
  Edit the Config class in the script to adjust file paths and validation thresholds.
        """
    )
    
    parser.add_argument('--uavsar', type=str, help='Path to UAVSAR/NISAR parquet file')
    parser.add_argument('--smap', type=str, help='Path to SMAP parquet file')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--temp-dir', type=str, help='Temporary directory for processing')
    
    args = parser.parse_args()
    
    # Override configuration if command-line arguments provided
    if args.uavsar:
        Config.UAVSAR_PATH = args.uavsar
    if args.smap:
        Config.SMAP_PATH = args.smap
    if args.output:
        Config.OUTPUT_PATH = args.output
    if args.temp_dir:
        Config.TEMP_DIR = args.temp_dir
    
    # Run consolidation
    success = consolidate_remote_sensing()
    exit(0 if success else 1)