"""
Transformation module for Arctic zero-curtain pipeline.
Auto-generated from Jupyter notebook.
"""

from src.common.imports import *
from src.common.utilities import *

# Archived
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
import warnings
import os
import re
warnings.filterwarnings('ignore')

def merge_all_permafrost_datasets(alt_datasets, soil_temp_datasets, soil_moisture_datasets, output_file=None):
    print("Starting comprehensive dataset merging process...")
    print(f"Processing {len(alt_datasets)} active layer thickness datasets...")
    alt_dataframes = []
    for name, df in alt_datasets.items():
        print(f"  Preparing {name} ALT data ({len(df)} records)...")
        if name == 'calm_above_neon_tundra_alt':
            prepared_df = prepare_calm_alt_data(df)
        elif name == 'derived_alt':
            prepared_df = prepare_derived_alt_data(df)
        elif name == 'gtnp_alt':
            prepared_df = prepare_gtnp_alt_data(df)
        else:
            print(f"  Unknown ALT dataset type: {name}, using generic preparation")
            prepared_df = prepare_generic_alt_data(df)
        alt_dataframes.append(prepared_df)
    print(f"Processing {len(soil_temp_datasets)} soil temperature datasets...")
    st_dataframes = []
    for name, df in soil_temp_datasets.items():
        print(f"  Preparing {name} soil temperature data ({len(df)} records)...")
        if name == 'ru_dst':
            prepared_df = prepare_ru_dst_data(df)
        elif name == 'gtnp_st':
            prepared_df = prepare_gtnp_st_data(df)
        elif name == 'tundrafielddb_st':
            prepared_df = prepare_tundrafield_st_data(df)
        else:
            print(f"  Unknown soil temperature dataset type: {name}, using generic preparation")
            prepared_df = prepare_generic_st_data(df)
        st_dataframes.append(prepared_df)
    print(f"Processing {len(soil_moisture_datasets)} soil moisture datasets...")
    sm_dataframes = []
    for name, df in soil_moisture_datasets.items():
        print(f"  Preparing {name} soil moisture data ({len(df)} records)...")
        if name == 'tundrafielddb_smc':
            prepared_df = prepare_tundrafield_smc_data(df)
        else:
            print(f"  Unknown soil moisture dataset type: {name}, using generic preparation")
            prepared_df = prepare_generic_sm_data(df)
        sm_dataframes.append(prepared_df)
    print("\nMerging all datasets...")
    all_dataframes = alt_dataframes + st_dataframes + sm_dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    print("Performing post-processing steps...")
    combined_df = standardize_datetime(combined_df)
    combined_df['merged_date'] = datetime.now().strftime('%Y-%m-%d')
    combined_df['merged_version'] = '3.0'
    combined_df = assign_depth_zones(combined_df)
    try:
        combined_df = combined_df.sort_values(['datetime', 'latitude', 'longitude'])
    except:
        print("Warning: Sorting failed, proceeding with unsorted data")
    print("Validating merged dataset...")
    validation_results = validate_merged_dataset(combined_df)
    print_validation_results(validation_results)
    print("\nMerged Comprehensive Dataset Statistics:")
    print(f"Total records: {len(combined_df)}")
    print(f"Active layer thickness records: {combined_df['alt_m'].notna().sum()}")
    print(f"Soil temperature records: {combined_df['temperature'].notna().sum()}")
    print(f"Soil moisture records: {combined_df['vwc'].notna().sum()}")
    valid_dates = combined_df['datetime'].dropna()
    if len(valid_dates) > 0:
        print(f"Date range: {valid_dates.min()} to {valid_dates.max()}")
    else:
        print("Date range: No valid datetime values found")
    print(f"Unique sites: {combined_df['site_name'].nunique()}")
    print(f"Unique data sources: {combined_df['data_source'].nunique()}")
    measurement_types = combined_df['measurement_type'].value_counts()
    print("\nMeasurement types distribution:")
    for mtype, count in measurement_types.items():
        print(f"  {mtype}: {count} records ({count/len(combined_df)*100:.2f}%)")
    if output_file:
        if len(combined_df) > 1000000:
            chunk_size = 1000000
            for i, chunk_df in enumerate(np.array_split(combined_df, len(combined_df) // chunk_size + 1)):
                if i == 0:
                    chunk_df.to_csv(output_file, index=False, mode='w')
                else:
                    chunk_df.to_csv(output_file, index=False, mode='a', header=False)
            print(f"\nSaved merged dataset in chunks to {output_file}")
        else:
            combined_df.to_csv(output_file, index=False)
            print(f"\nSaved merged dataset to {output_file}")
    return combined_df

def prepare_calm_alt_data(df):
    prep_df = df.copy()
    prep_df['datetime'] = pd.to_datetime(prep_df['datetime'], errors='coerce')
    if 'thickness_m' in prep_df.columns and 'alt_m' not in prep_df.columns:
        prep_df['alt_m'] = prep_df['thickness_m']
    elif 'thickness_m_standardized' in prep_df.columns and 'alt_m' not in prep_df.columns:
        prep_df['alt_m'] = prep_df['thickness_m_standardized']
    prep_df['measurement_type'] = 'active_layer_thickness'
    prep_df['data_source'] = prep_df.get('source', 'CALM')
    if 'lat_bin' not in prep_df.columns and 'latitude' in prep_df.columns:
        prep_df['lat_bin'] = np.floor(prep_df['latitude']).astype(int)
    if 'lon_bin' not in prep_df.columns and 'longitude' in prep_df.columns:
        prep_df['lon_bin'] = np.floor(prep_df['longitude']).astype(int)
    if 'is_spatial_outlier' in prep_df.columns or 'is_physical_outlier' in prep_df.columns:
        prep_df['quality_flag'] = 'valid'
        if 'is_spatial_outlier' in prep_df.columns:
            spatial_outliers = prep_df['is_spatial_outlier'] == True
            prep_df.loc[spatial_outliers, 'quality_flag'] = 'spatial_outlier'
        if 'is_physical_outlier' in prep_df.columns:
            physical_outliers = prep_df['is_physical_outlier'] == True
            prep_df.loc[physical_outliers, 'quality_flag'] = 'physical_outlier'
    else:
        prep_df['quality_flag'] = 'valid'
    prep_df = standardize_common_columns(prep_df)
    return prep_df

def prepare_derived_alt_data(df):
    prep_df = df.copy()
    prep_df['datetime'] = pd.to_datetime(prep_df['datetime'], errors='coerce')
    if 'alt_m' in prep_df.columns:
        pass
    elif 'thickness_m_standardized' in prep_df.columns:
        prep_df['alt_m'] = prep_df['thickness_m_standardized']
    prep_df['measurement_type'] = 'active_layer_thickness_derived'
    prep_df['data_source'] = prep_df.get('data_source', 'Derived_from_GTNP_Borehole')
    if 'standardization_confidence' in prep_df.columns:
        prep_df['quality_flag'] = prep_df['standardization_confidence'].map(lambda x: 'valid' if x == 'high' else 'low_confidence')
    else:
        prep_df['quality_flag'] = 'valid'
    if 'is_physical_outlier' in prep_df.columns:
        physical_outliers = prep_df['is_physical_outlier'] == True
        prep_df.loc[physical_outliers, 'quality_flag'] = 'physical_outlier'
    prep_df = standardize_common_columns(prep_df)
    return prep_df

def prepare_gtnp_alt_data(df):
    prep_df = df.copy()
    prep_df['datetime'] = pd.to_datetime(prep_df['datetime'], errors='coerce')
    if 'thickness_m' in prep_df.columns and 'alt_m' not in prep_df.columns:
        prep_df['alt_m'] = prep_df['thickness_m']
    prep_df['measurement_type'] = 'active_layer_thickness'
    prep_df['data_source'] = prep_df.get('data_source', 'GTNP_ActiveLayer')
    if 'measurement_flag' in prep_df.columns:
        prep_df['quality_flag'] = prep_df['measurement_flag'].map(lambda x: 'valid' if x == 'valid' else 'suspicious')
    else:
        prep_df['quality_flag'] = 'valid'
    prep_df = standardize_common_columns(prep_df)
    return prep_df

def prepare_generic_alt_data(df):
    prep_df = df.copy()
    if 'datetime' in prep_df.columns:
        prep_df['datetime'] = pd.to_datetime(prep_df['datetime'], errors='coerce')
    elif 'date' in prep_df.columns:
        prep_df['datetime'] = pd.to_datetime(prep_df['date'], errors='coerce')
    alt_column = None
    for col in ['alt_m', 'thickness_m', 'thickness', 'alt', 'thickness_cm']:
        if col in prep_df.columns:
            alt_column = col
            break
    if alt_column:
        if alt_column != 'alt_m':
            if 'cm' in alt_column:
                prep_df['alt_m'] = prep_df[alt_column] / 100
            else:
                prep_df['alt_m'] = prep_df[alt_column]
    else:
        print("    Warning: No ALT column found in generic ALT dataset")
        prep_df['alt_m'] = np.nan
    prep_df['measurement_type'] = 'active_layer_thickness'
    if 'data_source' not in prep_df.columns and 'source' in prep_df.columns:
        prep_df['data_source'] = prep_df['source']
    elif 'data_source' not in prep_df.columns:
        prep_df['data_source'] = 'Unknown_ALT_Source'
    if 'quality_flag' not in prep_df.columns:
        prep_df['quality_flag'] = 'valid'
    prep_df = standardize_common_columns(prep_df)
    return prep_df

def prepare_ru_dst_data(df):
    prep_df = df.copy()
    prep_df['datetime'] = pd.to_datetime(prep_df['datetime'], errors='coerce')
    if 'soil_temp' in prep_df.columns and 'temperature' not in prep_df.columns:
        prep_df['temperature'] = prep_df['soil_temp']
    elif 'soil_temp_standardized' in prep_df.columns and 'temperature' not in prep_df.columns:
        prep_df['temperature'] = prep_df['soil_temp_standardized']
    prep_df['measurement_type'] = 'soil_temperature'
    prep_df['data_source'] = prep_df.get('data_source', 'RU_Meteo')
    if 'soil_temp_depth' in prep_df.columns and 'depth_m' not in prep_df.columns:
        prep_df['depth_m'] = prep_df['soil_temp_depth']
    if 'soil_temp_depth_zone' in prep_df.columns and 'depth_zone' not in prep_df.columns:
        prep_df['depth_zone'] = prep_df['soil_temp_depth_zone']
    if 'temp_quality_flag' in prep_df.columns and 'quality_flag' not in prep_df.columns:
        prep_df['quality_flag'] = prep_df['temp_quality_flag']
    else:
        prep_df['quality_flag'] = 'valid'
    prep_df = standardize_common_columns(prep_df)
    return prep_df

def prepare_gtnp_st_data(df):
    prep_df = df.copy()
    prep_df['datetime'] = pd.to_datetime(prep_df['datetime'], errors='coerce')
    if 'temperature' in prep_df.columns:
        pass
    prep_df['measurement_type'] = 'soil_temperature'
    prep_df['data_source'] = prep_df.get('data_source', 'GTNP_Borehole')
    if 'depth_m' in prep_df.columns:
        pass
    if 'depth_zone' in prep_df.columns:
        pass
    else:
        prep_df['depth_zone'] = add_depth_zones(prep_df['depth_m'])
    if 'temp_quality' in prep_df.columns and 'quality_flag' not in prep_df.columns:
        prep_df['quality_flag'] = prep_df['temp_quality']
    else:
        prep_df['quality_flag'] = 'valid'
    prep_df = standardize_common_columns(prep_df)
    return prep_df

def prepare_tundrafield_st_data(df):
    prep_df = df.copy()
    prep_df['datetime'] = pd.to_datetime(prep_df['datetime'], errors='coerce')
    if 'lat' in prep_df.columns and 'latitude' not in prep_df.columns:
        prep_df['latitude'] = prep_df['lat']
    if 'lon' in prep_df.columns and 'longitude' not in prep_df.columns:
        prep_df['longitude'] = prep_df['lon']
    if 'st' in prep_df.columns and 'temperature' not in prep_df.columns:
        prep_df['temperature'] = prep_df['st']
    prep_df['measurement_type'] = 'soil_temperature'
    if 'source' in prep_df.columns and 'data_source' not in prep_df.columns:
        prep_df['data_source'] = prep_df['source']
    else:
        prep_df['data_source'] = 'TundraFieldDB'
    if 'depth_m' not in prep_df.columns:
        prep_df['depth_m'] = 0.1
    prep_df['depth_zone'] = add_depth_zones(prep_df['depth_m'])
    prep_df['quality_flag'] = 'valid'
    if 'site_name' not in prep_df.columns:
        prep_df['site_name'] = 'TundraField_' + prep_df['latitude'].round(2).astype(str) + '_' + prep_df['longitude'].round(2).astype(str)
    prep_df = standardize_common_columns(prep_df)
    return prep_df

def prepare_generic_st_data(df):
    prep_df = df.copy()
    if 'datetime' in prep_df.columns:
        prep_df['datetime'] = pd.to_datetime(prep_df['datetime'], errors='coerce')
    elif 'date' in prep_df.columns:
        prep_df['datetime'] = pd.to_datetime(prep_df['date'], errors='coerce')
    temp_column = None
    for col in ['temperature', 'soil_temp', 'temp', 'soil_temperature', 'st']:
        if col in prep_df.columns:
            temp_column = col
            break
    if temp_column:
        if temp_column != 'temperature':
            prep_df['temperature'] = prep_df[temp_column]
    else:
        print("    Warning: No temperature column found in generic soil temperature dataset")
        prep_df['temperature'] = np.nan
    depth_column = None
    for col in ['depth_m', 'depth', 'soil_depth', 'measurement_depth']:
        if col in prep_df.columns:
            depth_column = col
            break
    if depth_column:
        if depth_column != 'depth_m':
            prep_df['depth_m'] = prep_df[depth_column]
    else:
        print("    Warning: No depth column found in generic soil temperature dataset")
        prep_df['depth_m'] = np.nan
    prep_df['measurement_type'] = 'soil_temperature'
    if 'data_source' not in prep_df.columns and 'source' in prep_df.columns:
        prep_df['data_source'] = prep_df['source']
    elif 'data_source' not in prep_df.columns:
        prep_df['data_source'] = 'Unknown_ST_Source'
    if pd.notna(prep_df['depth_m']).any():
        prep_df['depth_zone'] = add_depth_zones(prep_df['depth_m'])
    else:
        prep_df['depth_zone'] = 'unknown'
    if 'quality_flag' not in prep_df.columns:
        prep_df['quality_flag'] = 'valid'
    if 'site_name' not in prep_df.columns and 'latitude' in prep_df.columns and 'longitude' in prep_df.columns:
        prep_df['site_name'] = 'Generic_' + prep_df['latitude'].round(2).astype(str) + '_' + prep_df['longitude'].round(2).astype(str)
    prep_df = standardize_common_columns(prep_df)
    return prep_df

def prepare_tundrafield_smc_data(df):
    prep_df = df.copy()
    prep_df['datetime'] = pd.to_datetime(prep_df['datetime'], errors='coerce')
    if 'lat' in prep_df.columns and 'latitude' not in prep_df.columns:
        prep_df['latitude'] = prep_df['lat']
    if 'lon' in prep_df.columns and 'longitude' not in prep_df.columns:
        prep_df['longitude'] = prep_df['lon']
    if 'smc' in prep_df.columns and 'vwc' not in prep_df.columns:
        prep_df['vwc'] = prep_df['smc']
    prep_df['measurement_type'] = 'soil_moisture'
    if 'source' in prep_df.columns and 'data_source' not in prep_df.columns:
        prep_df['data_source'] = prep_df['source']
    else:
        prep_df['data_source'] = 'TundraFieldDB'
    if 'depth_m' not in prep_df.columns:
        prep_df['depth_m'] = 0.1
    prep_df['quality_flag'] = 'valid'
    if 'site_name' not in prep_df.columns:
        prep_df['site_name'] = 'TundraField_' + prep_df['latitude'].round(2).astype(str) + '_' + prep_df['longitude'].round(2).astype(str)
    prep_df['depth_zone'] = 'surface'
    prep_df['vwc_std'] = np.nan
    prep_df['vwc_min'] = np.nan
    prep_df['vwc_max'] = np.nan
    prep_df = standardize_common_columns(prep_df)
    return prep_df

def prepare_generic_sm_data(df):
    prep_df = df.copy()
    if 'datetime' in prep_df.columns:
        prep_df['datetime'] = pd.to_datetime(prep_df['datetime'], errors='coerce')
    elif 'date' in prep_df.columns:
        prep_df['datetime'] = pd.to_datetime(prep_df['date'], errors='coerce')
    sm_column = None
    for col in ['vwc', 'smc', 'soil_moisture', 'volumetric_water_content', 'water_content']:
        if col in prep_df.columns:
            sm_column = col
            break
    if sm_column:
        if sm_column != 'vwc':
            prep_df['vwc'] = prep_df[sm_column]
    else:
        print("    Warning: No soil moisture column found in generic soil moisture dataset")
        prep_df['vwc'] = np.nan
    depth_column = None
    for col in ['depth_m', 'depth', 'soil_depth', 'measurement_depth']:
        if col in prep_df.columns:
            depth_column = col
            break
    if depth_column:
        if depth_column != 'depth_m':
            prep_df['depth_m'] = prep_df[depth_column]
    else:
        print("    Warning: No depth column found in generic soil moisture dataset")
        prep_df['depth_m'] = 0.1
    prep_df['measurement_type'] = 'soil_moisture'
    if 'data_source' not in prep_df.columns and 'source' in prep_df.columns:
        prep_df['data_source'] = prep_df['source']
    elif 'data_source' not in prep_df.columns:
        prep_df['data_source'] = 'Unknown_SM_Source'
    prep_df['depth_zone'] = add_depth_zones(prep_df['depth_m'])
    if 'vwc_std' not in prep_df.columns:
        prep_df['vwc_std'] = np.nan
    if 'vwc_min' not in prep_df.columns:
        prep_df['vwc_min'] = np.nan
    if 'vwc_max' not in prep_df.columns:
        prep_df['vwc_max'] = np.nan
    if 'quality_flag' not in prep_df.columns:
        prep_df['quality_flag'] = 'valid'
    if 'site_name' not in prep_df.columns and 'latitude' in prep_df.columns and 'longitude' in prep_df.columns:
        prep_df['site_name'] = 'Generic_' + prep_df['latitude'].round(2).astype(str) + '_' + prep_df['longitude'].round(2).astype(str)
    prep_df = standardize_common_columns(prep_df)
    return prep_df

def standardize_common_columns(df):
    prep_df = df.copy()
    standard_columns = ['datetime', 'year', 'latitude', 'longitude', 'site_name', 'site_id', 'dataset_id', 
                        'alt_m', 'depth_m', 'temperature', 'vwc', 'vwc_std', 'vwc_min', 'vwc_max',
                        'measurement_type', 'measurement_method', 'data_source', 'quality_flag', 'season', 'depth_zone',
                        'depth_top_st', 'depth_bottom_st', 'depth_top_vwc', 'depth_bottom_vwc']
    for col in standard_columns:
        if col not in prep_df.columns:
            prep_df[col] = np.nan
    if pd.isna(prep_df['year']).all() and not pd.isna(prep_df['datetime']).all():
        valid_datetime = prep_df['datetime'].notna()
        prep_df.loc[valid_datetime, 'year'] = prep_df.loc[valid_datetime, 'datetime'].dt.year
    if pd.isna(prep_df['season']).all() and not pd.isna(prep_df['datetime']).all():
        prep_df['season'] = determine_season(prep_df['datetime'])
    if pd.isna(prep_df['site_id']).all() and not pd.isna(prep_df['site_name']).all():
        site_id_from_name = prep_df['site_name'].fillna('').apply(lambda x: x.replace(' ', '_').lower() if isinstance(x, str) else '')
        prep_df['site_id'] = site_id_from_name
    if pd.isna(prep_df['depth_m']).all():
        st_mask = (prep_df['measurement_type'] == 'soil_temperature') & pd.notna(prep_df['depth_bottom_st'])
        if st_mask.any():
            prep_df.loc[st_mask, 'depth_m'] = prep_df.loc[st_mask, 'depth_bottom_st']
        vwc_mask = (prep_df['measurement_type'] == 'soil_moisture') & pd.notna(prep_df['depth_bottom_vwc'])
        if vwc_mask.any():
            prep_df.loc[vwc_mask, 'depth_m'] = prep_df.loc[vwc_mask, 'depth_bottom_vwc']
    return prep_df[standard_columns]

def standardize_datetime(df):
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    valid_datetime = df['datetime'].notna()
    if 'year' in df.columns:
        missing_year = df['year'].isna()
        df.loc[valid_datetime & missing_year, 'year'] = df.loc[valid_datetime & missing_year, 'datetime'].dt.year
    else:
        df['year'] = np.nan
        df.loc[valid_datetime, 'year'] = df.loc[valid_datetime, 'datetime'].dt.year
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    return df

def determine_season(dates):
    def get_season(date):
        if pd.isna(date):
            return 'unknown'
        month = date.month
        if 3 <= month <= 5:
            return 'Spring'
        elif 6 <= month <= 8:
            return 'Summer'
        elif 9 <= month <= 11:
            return 'Fall'
        else:
            return 'Winter'
    return dates.apply(get_season)

def add_depth_zones(depths):
    if isinstance(depths, pd.Series):
        result = pd.Series(index=depths.index, data='unknown', dtype='object')
        surface_mask = (depths <= 0.1) & (depths >= 0)
        result.loc[surface_mask] = 'surface'
        shallow_mask = (depths > 0.1) & (depths <= 0.5)
        result.loc[shallow_mask] = 'shallow_soil'
        deep_soil_mask = (depths > 0.5) & (depths <= 1.0)
        result.loc[deep_soil_mask] = 'deep_soil'
        upper_pf_mask = (depths > 1.0) & (depths <= 3.0)
        result.loc[upper_pf_mask] = 'upper_permafrost'
        deep_pf_mask = (depths > 3.0)
        result.loc[deep_pf_mask] = 'deep_permafrost'
        return result
    else:
        if pd.isna(depths):
            return 'unknown'
        elif depths <= 0.1 and depths >= 0:
            return 'surface'
        elif depths > 0.1 and depths <= 0.5:
            return 'shallow_soil'
        elif depths > 0.5 and depths <= 1.0:
            return 'deep_soil'
        elif depths > 1.0 and depths <= 3.0:
            return 'upper_permafrost'
        elif depths > 3.0:
            return 'deep_permafrost'
        else:
            return 'unknown'

def assign_depth_zones(df):
    if 'depth_zone' not in df.columns:
        df['depth_zone'] = 'unknown'
    else:
        df['depth_zone'] = df['depth_zone'].fillna('unknown')
    chunk_size = 1000000
    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:i+chunk_size]
        alt_mask = chunk_df['measurement_type'].isin(['active_layer_thickness', 'active_layer_thickness_derived'])
        if alt_mask.any():
            df.loc[df.index[i:i+chunk_size][alt_mask], 'depth_zone'] = 'active_layer'
    return df

def validate_merged_dataset(df):
    results = {}
    missing_lat = df['latitude'].isna().sum()
    missing_lon = df['longitude'].isna().sum()
    results['missing_coordinates'] = {'latitude': missing_lat, 'longitude': missing_lon, 'percentage': (missing_lat + missing_lon) / (2 * len(df)) * 100}
    missing_datetime = df['datetime'].isna().sum()
    results['missing_datetime'] = {'count': missing_datetime, 'percentage': missing_datetime / len(df) * 100}
    temp_values = df['temperature'].dropna()
    if len(temp_values) > 0:
        extreme_low = (temp_values < -60).sum()
        extreme_high = (temp_values > 60).sum()
        results['extreme_temperatures'] = {'below_minus_60': extreme_low, 'above_60': extreme_high, 'percentage': (extreme_low + extreme_high) / len(temp_values) * 100}
    else:
        results['extreme_temperatures'] = {'below_minus_60': 0, 'above_60': 0, 'percentage': 0}
    alt_values = df['alt_m'].dropna()
    if len(alt_values) > 0:
        extreme_alt_low = (alt_values < 0).sum()
        extreme_alt_high = (alt_values > 10).sum()
        results['extreme_alt'] = {'below_0': extreme_alt_low, 'above_10m': extreme_alt_high, 'percentage': (extreme_alt_low + extreme_alt_high) / len(alt_values) * 100}
    else:
        results['extreme_alt'] = {'below_0': 0, 'above_10m': 0, 'percentage': 0}
    vwc_values = df['vwc'].dropna()
    if len(vwc_values) > 0:
        extreme_vwc_low = (vwc_values < 0).sum()
        extreme_vwc_high = (vwc_values > 100).sum()
        results['extreme_vwc'] = {'below_0': extreme_vwc_low, 'above_100': extreme_vwc_high, 'percentage': (extreme_vwc_low + extreme_vwc_high) / len(vwc_values) * 100}
    else:
        results['extreme_vwc'] = {'below_0': 0, 'above_100': 0, 'percentage': 0}
    valid_dates = df['datetime'].dropna()
    if len(valid_dates) > 0:
        future_dates = (valid_dates > pd.Timestamp.now()).sum()
        results['future_dates'] = {'count': future_dates, 'percentage': future_dates / len(valid_dates) * 100}
    else:
        results['future_dates'] = {'count': 0, 'percentage': 0}
    measurement_types = df['measurement_type'].value_counts()
    results['measurement_types'] = measurement_types.to_dict()
    quality_flags = df['quality_flag'].value_counts()
    results['quality_flags'] = quality_flags.to_dict()
    return results

def print_validation_results(results):
    print("\nValidation Results:")
    print(f"Missing Coordinates:")
    print(f"  Latitude: {results['missing_coordinates']['latitude']} records")
    print(f"  Longitude: {results['missing_coordinates']['longitude']} records")
    print(f"  Percentage: {results['missing_coordinates']['percentage']:.2f}%")
    print(f"Missing Datetime:")
    print(f"  Count: {results['missing_datetime']['count']} records")
    print(f"  Percentage: {results['missing_datetime']['percentage']:.2f}%")
    print(f"Extreme Temperature Values:")
    print(f"  Below -60°C: {results['extreme_temperatures']['below_minus_60']} records")
    print(f"  Above 60°C: {results['extreme_temperatures']['above_60']} records")
    print(f"  Percentage: {results['extreme_temperatures']['percentage']:.2f}%")
    print(f"Extreme Active Layer Thickness Values:")
    print(f"  Below 0m: {results['extreme_alt']['below_0']} records")
    print(f"  Above 10m: {results['extreme_alt']['above_10m']} records")
    print(f"  Percentage: {results['extreme_alt']['percentage']:.2f}%")
    print(f"Extreme Volumetric Water Content Values:")
    print(f"  Below 0%: {results['extreme_vwc']['below_0']} records")
    print(f"  Above 100%: {results['extreme_vwc']['above_100']} records")
    print(f"  Percentage: {results['extreme_vwc']['percentage']:.2f}%")
    print(f"Future Dates:")
    print(f"  Count: {results['future_dates']['count']} records")
    print(f"  Percentage: {results['future_dates']['percentage']:.2f}%")
    print(f"Measurement Types Distribution:")
    for mtype, count in results['measurement_types'].items():
        print(f"  {mtype}: {count} records")
    print(f"Quality Flags Distribution:")
    for flag, count in results['quality_flags'].items():
        print(f"  {flag}: {count} records")

def combine_permafrost_datasets(data_df, ru_dst_df, spatial_threshold=0.01, temporal_threshold='1D'):
    data = data_df.copy()
    ru_dst = ru_dst_df.copy()
    column_mapping = {'temperature': 'soil_temp', 'depth': 'soil_temp_depth', 'depth_zone': 'soil_temp_depth_zone', 'source': 'data_source', 
                      'soil_temp': 'soil_temp', 'soil_temp_depth': 'soil_temp_depth', 'soil_temp_depth_zone': 'soil_temp_depth_zone'}
    data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})
    ru_dst = ru_dst.rename(columns={k: v for k, v in column_mapping.items() if k in ru_dst.columns})
    for df in [data, ru_dst]:
        if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    data['dataset_origin'] = 'global_dataset'
    ru_dst['dataset_origin'] = 'russian_dataset'
    common_columns = ['datetime', 'soil_temp', 'latitude', 'longitude', 'soil_temp_depth', 'soil_temp_depth_zone', 
                     'data_source', 'site_id', 'season', 'station_name', 'dataset_origin']
    data_subset = data[[col for col in common_columns if col in data.columns]]
    ru_dst_subset = ru_dst[[col for col in common_columns if col in ru_dst.columns]]
    for col in common_columns:
        if col not in data_subset.columns:
            data_subset[col] = np.nan
        if col not in ru_dst_subset.columns:
            ru_dst_subset[col] = np.nan
    combined_df = pd.concat([data_subset, ru_dst_subset], ignore_index=True)
    depth_zone_mapping = {'shallow': 'shallow', 'intermediate': 'intermediate', 'deep': 'deep', 'very_deep': 'very_deep'}
    combined_df['soil_temp_depth_zone'] = combined_df['soil_temp_depth_zone'].map(lambda x: depth_zone_mapping.get(str(x).lower(), 'unknown'))
    combined_df['calculated_depth_zone'] = combined_df['soil_temp_depth'].apply(add_depth_zones)
    geometry = [Point(xy) for xy in zip(combined_df['longitude'], combined_df['latitude'])]
    gdf = gpd.GeoDataFrame(combined_df, geometry=geometry)
    temporal_coverage = gdf.groupby(['site_id'])['datetime'].agg(['min', 'max', 'count'])
    temporal_coverage['duration_days'] = (temporal_coverage['max'] - temporal_coverage['min']).dt.days
    temporal_coverage['density'] = temporal_coverage['count'] / temporal_coverage['duration_days'].clip(lower=1)
    site_density_map = dict(zip(temporal_coverage.index, temporal_coverage['density']))
    gdf['temporal_density'] = gdf['site_id'].map(site_density_map)
    harmonized_df = pd.DataFrame(gdf.drop(columns='geometry'))
    harmonized_df['harmonization_date'] = datetime.now().strftime('%Y-%m-%d')
    harmonized_df['harmonization_version'] = '1.0'
    return harmonized_df

def transform_station_names(df):
    df_copy = df.copy()
    cyrillic_names = {'Yakutsk': 'Якутск', 'Pokrovsk': 'Покровск', 'Verkhoyansk': 'Верхоянск', 'Isit': 'Изит', 'Churapcha': 'Чурапча'}
    def transform_siberia_name(name):
        if not isinstance(name, str):
            return name
        if name.startswith('siberia_'):
            parts = name.split('_')
            if len(parts) >= 2:
                location = parts[1]
                location = re.sub(r'_[a-z]{3}_size]
        st_mask = chunk_df['measurement_type'] == 'soil_temperature'
        if st_mask.any():
            use_st_depth = chunk_df['depth_m'].copy()
            mask = (chunk_df['depth_zone'] == 'unknown') & st_mask & use_st_depth.notna()
            if mask.any():
                surface_mask = mask & (use_st_depth <= 0.1) & (use_st_depth >= 0)
                df.loc[df.index[i:i+chunk_size][surface_mask], 'depth_zone'] = 'surface'
                shallow_mask = mask & (use_st_depth > 0.1) & (use_st_depth <= 0.5)
                df.loc[df.index[i:i+chunk_size][shallow_mask], 'depth_zone'] = 'shallow_soil'
                deep_soil_mask = mask & (use_st_depth > 0.5) & (use_st_depth <= 1.0)
                df.loc[df.index[i:i+chunk_size][deep_soil_mask], 'depth_zone'] = 'deep_soil'
                upper_pf_mask = mask & (use_st_depth > 1.0) & (use_st_depth <= 3.0)
                df.loc[df.index[i:i+chunk_size][upper_pf_mask], 'depth_zone'] = 'upper_permafrost'
                deep_pf_mask = mask & (use_st_depth > 3.0)
                df.loc[df.index[i:i+chunk_size][deep_pf_mask], 'depth_zone'] = 'deep_permafrost'
        vwc_mask = chunk_df['measurement_type'] == 'soil_moisture'
        if vwc_mask.any():
            use_vwc_depth = chunk_df['depth_m'].copy()
            mask = (chunk_df['depth_zone'] == 'unknown') & vwc_mask & use_vwc_depth.notna()
            if mask.any():
                surface_mask = mask & (use_vwc_depth <= 0.1) & (use_vwc_depth >= 0)
                df.loc[df.index[i:i+chunk_size][surface_mask], 'depth_zone'] = 'surface'
                shallow_mask = mask & (use_vwc_depth > 0.1) & (use_vwc_depth <= 0.5)
                df.loc[df.index[i:i+chunk_size][shallow_mask], 'depth_zone'] = 'shallow_soil'
                deep_soil_mask = mask & (use_vwc_depth > 0.5) & (use_vwc_depth <= 1.0)
                df.loc[df.index[i:i+chunk_size][deep_soil_mask], 'depth_zone'] = 'deep_soil'
                upper_pf_mask = mask & (use_vwc_depth > 1.0) & (use_vwc_depth <= 3.0)
                df.loc[df.index[i:i+chunk_size][upper_pf_mask], 'depth_zone'] = 'upper_permafrost'
                deep_pf_mask = mask & (use_vwc_depth > 3.0)
                df.loc[df.index[i:i+chunk_size][deep_pf_mask], 'depth_zone'] = 'deep_permafrost'
    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:i+chunk, '', location)
                return cyrillic_names.get(location, location)
        return name
    def transform_borehole_name(name):
        if not isinstance(name, str):
            return name
        if name.startswith('Borehole_'):
            match = re.search(r'Borehole_(\d+)', name)
            if match:
                return f'Borehole_{match.group(1)}'
        return name
    df_copy['station_name'] = df_copy['station_name'].apply(transform_siberia_name)
    df_copy['station_name'] = df_copy['station_name'].apply(transform_borehole_name)
    return df_copy

def process_soil_temperatures_final(combined_df):
    df = combined_df.copy()
    site_ids_to_divide = ['RU_Meteo_30879', 'RU_Meteo_30825', 'RU_Meteo_28367', 'RU_Meteo_28573', 'RU_Meteo_25138']
    station_names_to_divide = ['Нерчинский з-д', 'Иволгинск', 'Тюмень', 'Ишим', 'Островное']
    borehole_station_patterns = ['Borehole_']
    division_mask = (df['site_id'].isin(site_ids_to_divide) | df['station_name'].isin(station_names_to_divide))
    df.loc[division_mask, 'soil_temp'] = df.loc[division_mask, 'soil_temp'] / 10
    df['temp_divided'] = division_mask
    divided_count = division_mask.sum()
    borehole_mask = df['station_name'].apply(lambda x: isinstance(x, str) and any(x.startswith(pattern) for pattern in borehole_station_patterns))
    high_mask = (df['soil_temp'] > 40) & borehole_mask
    low_mask = (df['soil_temp'] < -40) & borehole_mask
    extreme_mask = high_mask | low_mask
    high_count = high_mask.sum()
    low_count = low_mask.sum()
    df.loc[extreme_mask, 'soil_temp'] = np.nan
    df['extreme_filtered'] = extreme_mask
    extreme_count = extreme_mask.sum()
    still_extreme = ((df['soil_temp'] > 40) | (df['soil_temp'] < -40))
    still_extreme_count = still_extreme.sum()
    print(f"Soil temperature processing results:")
    print(f"1. Division by 10:")
    print(f"   - Applied to {divided_count} records ({divided_count/len(df)*100:.2f}% of dataset)")
    print(f"   - Sites affected: {df.loc[division_mask, 'site_id'].nunique()}")
    print(f"   - Stations affected: {df.loc[division_mask, 'station_name'].nunique()}")
    print(f"\n2. Extreme value filtering:")
    print(f"   - Applied to {extreme_count} records ({extreme_count/len(df)*100:.2f}% of dataset)")
    print(f"   - High values (>40°C): {high_count}")
    print(f"   - Low values (<-40°C): {low_count}")
    print(f"   - Borehole stations affected: {df.loc[extreme_mask, 'station_name'].nunique()}")
    if still_extreme_count > 0:
        print(f"\nWARNING: {still_extreme_count} records still have extreme temperatures")
    else:
        print(f"\nAll extreme values have been successfully filtered")
    print(f"\nFinal temperature range: {df['soil_temp'].min()} to {df['soil_temp'].max()}°C")
    return df

def extract_siberian_data(file_paths):
    all_data = []
    for file_path in file_paths:
        site_name = os.path.basename(file_path).split('_')[1]
        print(f"Processing {site_name} file...")
        try:
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            jan_sheet = next((s for s in sheet_names if 'jan' in s.lower()), None)
            jul_sheet = next((s for s in sheet_names if 'jul' in s.lower()), None)
            if not jan_sheet or not jul_sheet:
                continue
            jan_metadata = pd.read_excel(file_path, sheet_name=jan_sheet, header=None, nrows=7, skiprows=2)
            site_coords = {'Yakutsk': (62.0, 129.7, "24959"), 'Verkhoyansk': (67.5, 133.4, "24266"), 
                          'Pokrovsk': (61.5, 129.1, "24065"), 'Isit': (61.0, 125.3, "24679"), 'Churapcha': (62.0, 132.4, "24859")}
            default_lat, default_lon, default_wmo = site_coords.get(site_name, (0.0, 0.0, ""))
            wmo = str(jan_metadata.iloc[2, 1]) if jan_metadata.iloc[2, 1] != 'nan' else default_wmo
            name = jan_metadata.iloc[3, 1] if isinstance(jan_metadata.iloc[3, 1], str) else site_name
            latitude = float(jan_metadata.iloc[4, 1]) if pd.notna(jan_metadata.iloc[4, 1]) else default_lat
            longitude = float(jan_metadata.iloc[5, 1]) if pd.notna(jan_metadata.iloc[5, 1]) else default_lon
            header_df = pd.read_excel(file_path, sheet_name=jan_sheet, skiprows=10, nrows=1)
            data_df = pd.read_excel(file_path, sheet_name=jan_sheet, skiprows=11)
            column_mapping = {data_df.columns[i]: header_df.columns[i] for i in range(min(len(data_df.columns), len(header_df.columns)))}
            jan_data = data_df.rename(columns=column_mapping)
            jul_data = pd.read_excel(file_path, sheet_name=jul_sheet, skiprows=11).rename(columns=column_mapping)
            if 'YEAR' in jan_data.columns:
                jan_processed = process_monthly_data(jan_data, 1, site_name, wmo, name, latitude, longitude)
                jul_processed = process_monthly_data(jul_data, 7, site_name, wmo, name, latitude, longitude)
                site_data = pd.concat([jan_processed, jul_processed], ignore_index=True)
                all_data.append(site_data)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def process_monthly_data(df, month, site_name, wmo, name, latitude, longitude, year_column=None):
    data = df.copy()
    if year_column is not None and year_column != 'YEAR':
        data = data.rename(columns={year_column: 'YEAR'})
    if 'YEAR' not in data.columns:
        return pd.DataFrame()
    data = data.dropna(subset=['YEAR'])
    data['YEAR'] = data['YEAR'].astype(int)
    processed_records = []
    standard_depths = {'0.20 m': 0.2, '0.40 m': 0.4, '0.60 m ': 0.6, '0.60 m': 0.6, '0.80 m': 0.8, 
                      '1.20 m': 1.2, '1.60 m': 1.6, '2.00 m': 2.0, '2.40 m': 2.4, '3.20 m': 3.2}
    depth_columns = [col for col in data.columns if str(col) in standard_depths]
    depth_values = {col: standard_depths[str(col)] for col in depth_columns}
    for _, row in data.iterrows():
        year = int(row['YEAR'])
        date_str = f"{year}-{month:02d}-01"
        for col in depth_columns:
            value = row[col]
            if value == -999 or pd.isna(value):
                continue
            record = {'datetime': date_str, 'year': year, 'month': month, 'temperature': float(value), 'depth': depth_values[col],
                     'latitude': latitude, 'longitude': longitude, 'site_id': str(wmo), 'site_name': name,
                     'source': f"siberia_{site_name}_{'jan' if month == 1 else 'jul'}", 
                     'depth_zone': add_depth_zones(depth_values[col]), 'season': 'Winter' if month == 1 else 'Summer'}
            processed_records.append(record)
    return pd.DataFrame(processed_records)

def merge_permafrost_multimodal_datasets(calm_df, above_df, gtnp_df, output_file=None):
    print("Starting multimodal dataset merging process...")
    print(f"Preparing CALM ALT data ({len(calm_df)} records)...")
    calm_prep = prepare_calm_data(calm_df)
    print(f"Preparing ABOVE/NEON VWC data ({len(above_df)} records)...")
    above_prep = prepare_above_data(above_df)
    print(f"Preparing GTNP combined data ({len(gtnp_df)} records)...")
    gtnp_prep = prepare_gtnp_data(gtnp_df)
    print("Merging all datasets...")
    combined_df = pd.concat([calm_prep, above_prep, gtnp_prep], ignore_index=True, sort=False)
    combined_df = standardize_datetime(combined_df)
    combined_df['merged_date'] = datetime.now().strftime('%Y-%m-%d')
    combined_df['merged_version'] = '2.0'
    combined_df = assign_depth_zones(combined_df)
    try:
        combined_df = combined_df.sort_values(['datetime', 'latitude', 'longitude'])
    except:
        print("Warning: Sorting failed")
    print("\nMerged Multimodal Dataset Statistics:")
    print(f"Total records: {len(combined_df)}")
    print(f"ALT records: {combined_df['alt_m'].notna().sum()}")
    print(f"Temperature records: {combined_df['temperature'].notna().sum()}")
    print(f"VWC records: {combined_df['vwc'].notna().sum()}")
    if output_file:
        combined_df.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")
    return combined_df

def prepare_calm_data(df):
    prep_df = df.copy()
    prep_df['datetime'] = pd.to_datetime(prep_df['datetime'], errors='coerce')
    if 'thickness' in prep_df.columns:
        prep_df.rename(columns={'thickness': 'alt_m'}, inplace=True)
    prep_df['measurement_type'] = 'active_layer_thickness'
    prep_df['measurement_method'] = 'Mechanical_Probing'
    prep_df['quality_flag'] = 'valid'
    if 'site_name' not in prep_df.columns:
        prep_df['site_name'] = 'CALM_' + prep_df['latitude'].round(2).astype(str) + '_' + prep_df['longitude'].round(2).astype(str)
    prep_df['data_source'] = prep_df.get('source', 'CALM')
    prep_df['depth_m'] = np.nan
    prep_df['temperature'] = np.nan
    prep_df['vwc'] = np.nan
    if 'season' not in prep_df.columns:
        prep_df['season'] = determine_season(prep_df['datetime'])
    prep_df['depth_zone'] = 'active_layer'
    return prep_df

def prepare_above_data(df):
    prep_df = df.copy()
    prep_df['datetime'] = pd.to_datetime(prep_df['datetime'], errors='coerce')
    prep_df['measurement_type'] = 'soil_moisture'
    prep_df['measurement_method'] = 'TDR_Probe'
    if 'site_name' not in prep_df.columns:
        prep_df['site_name'] = prep_df.get('site_id', 'ABOVE_NEON_' + prep_df['latitude'].round(2).astype(str) + '_' + prep_df['longitude'].round(2).astype(str))
    prep_df['data_source'] = prep_df.get('source', 'ABOVE_NEON')
    prep_df['quality_flag'] = 'valid'
    prep_df['alt_m'] = np.nan
    if 'temperature' not in prep_df.columns:
        prep_df['temperature'] = prep_df.get('soil_temp', np.nan)
    if 'season' not in prep_df.columns:
        prep_df['season'] = determine_season(prep_df['datetime'])
    return prep_df

def prepare_gtnp_data(df):
    prep_df = df.copy()
    prep_df['datetime'] = pd.to_datetime(prep_df['datetime'], errors='coerce')
    if 'vwc' not in prep_df.columns:
        prep_df['vwc'] = np.nan
    if 'year' not in prep_df.columns:
        prep_df['year'] = prep_df['datetime'].dt.year
    if 'depth_zone' not in prep_df.columns:
        prep_df['depth_zone'] = 'unknown'
    return prep_df_size]
        st_mask = chunk_df['measurement_type'] == 'soil_temperature'
        if st_mask.any():
            use_st_depth = chunk_df['depth_m'].copy()
            mask = (chunk_df['depth_zone'] == 'unknown') & st_mask & use_st_depth.notna()
            if mask.any():
                surface_mask = mask & (use_st_depth <= 0.1) & (use_st_depth >= 0)
                df.loc[df.index[i:i+chunk_size][surface_mask], 'depth_zone'] = 'surface'
                shallow_mask = mask & (use_st_depth > 0.1) & (use_st_depth <= 0.5)
                df.loc[df.index[i:i+chunk_size][shallow_mask], 'depth_zone'] = 'shallow_soil'
                deep_soil_mask = mask & (use_st_depth > 0.5) & (use_st_depth <= 1.0)
                df.loc[df.index[i:i+chunk_size][deep_soil_mask], 'depth_zone'] = 'deep_soil'
                upper_pf_mask = mask & (use_st_depth > 1.0) & (use_st_depth <= 3.0)
                df.loc[df.index[i:i+chunk_size][upper_pf_mask], 'depth_zone'] = 'upper_permafrost'
                deep_pf_mask = mask & (use_st_depth > 3.0)
                df.loc[df.index[i:i+chunk_size][deep_pf_mask], 'depth_zone'] = 'deep_permafrost'
        vwc_mask = chunk_df['measurement_type'] == 'soil_moisture'
        if vwc_mask.any():
            use_vwc_depth = chunk_df['depth_m'].copy()
            mask = (chunk_df['depth_zone'] == 'unknown') & vwc_mask & use_vwc_depth.notna()
            if mask.any():
                surface_mask = mask & (use_vwc_depth <= 0.1) & (use_vwc_depth >= 0)
                df.loc[df.index[i:i+chunk_size][surface_mask], 'depth_zone'] = 'surface'
                shallow_mask = mask & (use_vwc_depth > 0.1) & (use_vwc_depth <= 0.5)
                df.loc[df.index[i:i+chunk_size][shallow_mask], 'depth_zone'] = 'shallow_soil'
                deep_soil_mask = mask & (use_vwc_depth > 0.5) & (use_vwc_depth <= 1.0)
                df.loc[df.index[i:i+chunk_size][deep_soil_mask], 'depth_zone'] = 'deep_soil'
                upper_pf_mask = mask & (use_vwc_depth > 1.0) & (use_vwc_depth <= 3.0)
                df.loc[df.index[i:i+chunk_size][upper_pf_mask], 'depth_zone'] = 'upper_permafrost'
                deep_pf_mask = mask & (use_vwc_depth > 3.0)
                df.loc[df.index[i:i+chunk_size][deep_pf_mask], 'depth_zone'] = 'deep_permafrost'
    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:i+chunk

