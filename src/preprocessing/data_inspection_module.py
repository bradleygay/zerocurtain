"""
Data Inspection module for Arctic zero-curtain pipeline.
Auto-generated from Jupyter notebook.
"""

from src.utils.imports import *
from src.utils.utilities import *

# Playground

import pandas as pd
import numpy as np
import dask.dataframe as dd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.patheffects as pe
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import geodatasets
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap, MarkerCluster
import rasterio
from rasterio.plot import show
import contextily as ctx
from datetime import datetime
from sklearn.cluster import DBSCAN
from IPython.display import display
from tabulate import tabulate
import os

def inspect_parquet(filepath, n_rows=10, show_schema=True, show_preview=True):
    print(f"\nINSPECTING: {filepath}\n")
    if show_schema:
        pq_file = pq.ParquetFile(filepath)
        print("PyArrow Schema:")
        print(pq_file.schema)
    df_dask = dd.read_parquet(filepath)
    print(f"\nDask Columns:")
    print(df_dask.columns.tolist())
    if show_preview:
        print(f"\nData Preview (first {n_rows} rows):")
        preview = df_dask.head(n_rows)
        display(preview)
    df_lazy = pl.scan_parquet(filepath)
    print(f"\nPolars Schema:")
    print(df_lazy.schema)
    return df_dask

def normalize_band(band_data, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.percentile(band_data, 5)
    if max_val is None:
        max_val = np.percentile(band_data, 95)
    return np.clip((band_data - min_val) / (max_val - min_val), 0, 1)

def create_spatial_df(df):
    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

def plot_arctic_projection(df, column, cmap='viridis', title=None, output_path=None):
    projection = ccrs.NorthPolarStereo(central_longitude=0.0)
    fig = plt.figure(figsize=(14, 14))
    ax = plt.axes(projection=projection)
    ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='white', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6, edgecolor='white')
    vmin = np.percentile(df[column].dropna(), 5)
    vmax = np.percentile(df[column].dropna(), 95)
    norm = Normalize(vmin=vmin, vmax=vmax)
    sc = ax.scatter(df['longitude'], df['latitude'], c=df[column], s=3, alpha=0.7,
                    transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
    cbar = plt.colorbar(sc, ax=ax, pad=0.05, shrink=0.8)
    cbar.set_label(column, fontsize=14)
    if title:
        plt.title(title, fontsize=16)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_data_coverage(df, output_dir="coverage_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nDATA COVERAGE ANALYSIS\n")
    print(f"Total observations: {len(df):,}")
    print(f"Longitude range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
    print(f"Latitude range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
    hist, xedges, yedges = np.histogram2d(df['longitude'], df['latitude'], bins=[36, 8],
                                          range=[[-180, 180], [50, 90]])
    total_bins = hist.size
    empty_bins = np.sum(hist == 0)
    coverage_pct = 100 * (1 - empty_bins / total_bins)
    print(f"Spatial coverage: {coverage_pct:.2f}% of Arctic grid cells")
    print(f"Empty cells: {empty_bins}/{total_bins}")
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"Temporal range: {df['datetime'].min()} to {df['datetime'].max()}")
    return {"total_observations": len(df), "coverage_percentage": coverage_pct,
            "empty_cells": empty_bins, "total_cells": total_bins}

def transform_uavsar_nisar(input_path, output_path):
    print(f"\nTRANSFORMING UAVSAR/NISAR DATA\n")
    df = dd.read_parquet(input_path)
    df_transformed = df[['longitude', 'latitude', 'thickness_m', 'thickness_m_standardized',
                         'first_retrieval_dt', 'second_retrieval_dt', 'duration_days',
                         'period', 'season', 'year', 'source', 'data_type']].copy()
    df_transformed['datetime'] = df_transformed['second_retrieval_dt']
    df_transformed['soil_temp'] = np.nan
    df_transformed['soil_temp_standardized'] = np.nan
    df_transformed['soil_temp_depth'] = np.nan
    df_transformed['soil_temp_depth_zone'] = pd.NA
    df_transformed['soil_moist'] = np.nan
    df_transformed['soil_moist_standardized'] = np.nan
    df_transformed['soil_moist_depth'] = np.nan
    column_order = ['datetime', 'year', 'season', 'latitude', 'longitude',
                    'thickness_m', 'thickness_m_standardized',
                    'soil_temp', 'soil_temp_standardized', 'soil_temp_depth', 'soil_temp_depth_zone',
                    'soil_moist', 'soil_moist_standardized', 'soil_moist_depth',
                    'source', 'data_type']
    df_transformed = df_transformed[column_order]
    df_transformed.to_parquet(output_path, engine='pyarrow', compression='snappy')
    print(f"Transformation complete: {output_path}")
    return df_transformed

def transform_smap(input_path, output_path):
    print(f"\nTRANSFORMING SMAP DATA\n")
    df = dd.read_parquet(input_path)
    df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
    temp_depths = {'soil_temp_layer1': 0.1, 'soil_temp_layer2': 0.2, 'soil_temp_layer3': 0.4,
                   'soil_temp_layer4': 0.75, 'soil_temp_layer5': 1.5, 'soil_temp_layer6': 10.0}
    moist_depths = {'sm_surface': 0.05, 'sm_rootzone': 1.0}
    records = []
    for _, row in df.iterrows():
        base_record = {'datetime': row['datetime'], 'latitude': row['latitude'], 'longitude': row['longitude'],
                      'year': pd.to_datetime(row['datetime']).year, 'season': pd.to_datetime(row['datetime']).season,
                      'thickness_m': np.nan, 'thickness_m_standardized': np.nan, 'source': 'SMAP', 'data_type': None}
        for layer, depth in temp_depths.items():
            if pd.notna(row[layer]):
                record = base_record.copy()
                record.update({'soil_temp': row[layer], 'soil_temp_standardized': row[layer],
                              'soil_temp_depth': depth, 'soil_temp_depth_zone': classify_depth_zone(depth),
                              'soil_moist': np.nan, 'soil_moist_standardized': np.nan,
                              'soil_moist_depth': np.nan, 'data_type': 'soil_temperature'})
                records.append(record)
        for layer, depth in moist_depths.items():
            if pd.notna(row[layer]):
                record = base_record.copy()
                record.update({'soil_temp': np.nan, 'soil_temp_standardized': np.nan,
                              'soil_temp_depth': np.nan, 'soil_temp_depth_zone': pd.NA,
                              'soil_moist': row[layer], 'soil_moist_standardized': row[layer],
                              'soil_moist_depth': depth, 'data_type': 'soil_moisture'})
                records.append(record)
    df_transformed = pd.DataFrame(records)
    df_transformed.to_parquet(output_path, engine='pyarrow', compression='snappy')
    print(f"Transformation complete: {output_path}")
    return df_transformed

def classify_depth_zone(depth):
    if depth < 0.2:
        return 'surface'
    elif depth < 0.5:
        return 'intermediate'
    elif depth < 1.0:
        return 'deep'
    else:
        return 'very_deep'

def merge_arctic_datasets(paths_dict, output_path):
    print(f"\nMERGING ARCTIC DATASETS\n")
    dataframes = []
    for name, path in paths_dict.items():
        print(f"Loading {name}...")
        df = dd.read_parquet(path)
        dataframes.append(df)
    df_merged = dd.concat(dataframes, axis=0, ignore_index=True)
    df_merged = df_merged.sort_values('datetime').reset_index(drop=True)
    df_merged.to_parquet(output_path, engine='pyarrow', compression='snappy')
    print(f"\nMerged dataset saved: {output_path}")
    print(f"Total records: {len(df_merged):,}")
    return df_merged

def main():
    PATHS = {
        'in_situ': '/Users/[USER]/merged_compressed_corrected_final.parquet',
        'uavsar_nisar_raw': '/Users/[USER]/merged_uavsar_nisar.parquet',
        'smap_raw': '/Users/[USER]/smap_master.parquet',
        'uavsar_nisar_transformed': '/Users/[USER]/uavsar_nisar_transformed.parquet',
        'smap_transformed': '/Users/[USER]/smap_transformed.parquet',
        'merged_final': '/Users/[USER]/final_arctic_consolidated.parquet',
        'zero_curtain_detections': '/Users/[USER]/part1_pipeline_optimized/performance_fixed_zero_curtain_events.parquet',
        'remote_sensing_detections': '/Users/[USER]/vectorized_output/vectorized_high_performance_zero_curtain.parquet',
        'final_predictions': '/Users/[USER]/part4_transfer_learning_new/predictions/circumarctic_zero_curtain_predictions_20250730_160518.parquet'
    }
    print("\nSTEP 1: DATA INSPECTION")
    for name, path in [('In Situ', PATHS['in_situ']), ('UAVSAR/NISAR', PATHS['uavsar_nisar_raw']), ('SMAP', PATHS['smap_raw'])]:
        if os.path.exists(path):
            inspect_parquet(path, n_rows=5, show_preview=True)
    print("\nSTEP 2: DATA TRANSFORMATION")
    if not os.path.exists(PATHS['uavsar_nisar_transformed']):
        transform_uavsar_nisar(PATHS['uavsar_nisar_raw'], PATHS['uavsar_nisar_transformed'])
    if not os.path.exists(PATHS['smap_transformed']):
        transform_smap(PATHS['smap_raw'], PATHS['smap_transformed'])
    print("\nSTEP 3: DATASET MERGING")
    if not os.path.exists(PATHS['merged_final']):
        merge_datasets = {'in_situ': PATHS['in_situ'], 'uavsar_nisar': PATHS['uavsar_nisar_transformed'],
                         'smap': PATHS['smap_transformed']}
        merge_arctic_datasets(merge_datasets, PATHS['merged_final'])
    print("\nSTEP 4: FINAL PREDICTIONS ANALYSIS")
    df_predictions = dd.read_parquet(PATHS['final_predictions']).compute()
    print(f"\nPrediction Dataset Summary:")
    print(f"Total predictions: {len(df_predictions):,}")
    print(f"Spatial extent: {df_predictions['latitude_center'].min():.2f}N to {df_predictions['latitude_center'].max():.2f}N")
    print(f"Temporal range: {df_predictions['year_mean'].min():.0f} to {df_predictions['year_mean'].max():.0f}")
    print(f"Mean confidence: {df_predictions['zc_confidence_score'].mean():.3f}")
    print(f"High confidence detections: {(df_predictions['zc_confidence_category'] == 'High').sum():,}")
    print("\nSTEP 5: VISUALIZATION")
    plot_arctic_projection(df_predictions, column='zc_presence_probability', cmap='Spectral_r',
                          title='Zero-Curtain Presence Probability - Circumarctic Domain',
                          output_path='arctic_zero_curtain_probability_map.png')
    print("\nPIPELINE COMPLETE")

if __name__ == "__main__":
    main()

# RuMeteo
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import json
import re
import os

df = pd.read_csv('/Users/[USER]/Library/CloudStorage/OneDrive-[REDACTED_AFFILIATION]/zerocurtain/ru/new/latlon_ru_update.csv')
indices_dailysoiltemperature = df[df['data_source'] == 'dailysoiltemperature_1984-2012']['index'].unique()
latlon_extracted = df[(df['index'].isin(indices_dailysoiltemperature)) & (df['data_source'] != 'dailysoiltemperature_1984-2012')][['index', 'latitude', 'longitude', 'data_source']]
latlon_lookup = latlon_extracted.groupby('index')[['latitude', 'longitude']].first()
mask = (df['data_source'] == 'dailysoiltemperature_1984-2012') & (df['latitude'].isna() | df['longitude'].isna())
df.loc[mask, ['latitude', 'longitude']] = df.loc[mask, 'index'].map(latlon_lookup.to_dict()).apply(pd.Series)
print(df[df['data_source'] == 'dailysoiltemperature_1984-2012'].isna().sum())
df[(df['data_source'] == 'dailysoiltemperature_1984-2012')]
latlon_lookup = latlon_extracted.groupby('index')[['latitude', 'longitude']].first().to_dict(orient='index')
for idx in df[df['data_source'] == 'dailysoiltemperature_1984-2012'].index:
    station_index = df.at[idx, 'index']
    if station_index in latlon_lookup:
        df.at[idx, 'latitude'] = latlon_lookup[station_index]['latitude']
        df.at[idx, 'longitude'] = latlon_lookup[station_index]['longitude']
df[df['data_source'] == 'dailysoiltemperature_1984-2012'].isna().sum()
df[(df['data_source'] == 'dailysoiltemperature_1984-2012')]
df.to_csv('/Users/[USER]/Downloads/latlon_ru_update.csv')
df = df.drop(columns=['Unnamed: 0'])

class ColumnDefinitions:
    MONTHLY_FORMATS = {
        'T_mes.tab': {'name': 'air_temperature', 'description': 'Monthly average air temperature'},
        'Pm_mes.tab': {'name': 'air_pressure_sea_level', 'description': 'Air pressure at sea level'},
        'Ob_mes.tab': {'name': 'cloud_cover', 'description': 'Average cloud cover'},
        'Nd_mes.tab': {'name': 'precipitation_days', 'description': 'Number of days with precipitation > 1mm'},
        'E_mes.tab': {'name': 'water_vapor_pressure', 'description': 'Partial pressure of water vapor'},
        'Pss_mes.tab': {'name': 'sunshine_duration', 'description': 'Sunshine duration'},
        'Ps_mes.tab': {'name': 'air_pressure', 'description': 'Air pressure at station'}
    }
    DAILY_FORMATS = {
        'Tttr': {
            'columns': ['station_id', 'year', 'month', 'day', 'tflag', 'tmin', 'qtmin', 'tmean', 'qtmean', 'tmax', 'qtmax', 'precip', 'precip_flag', 'precip_quality'],
            'description': 'Daily temperature and precipitation'
        },
        'Snow': {
            'columns': ['station_id', 'year', 'month', 'day', 'snow_depth', 'snow_coverage', 'depth_info', 'depth_quality', 'temp_info'],
            'description': 'Daily snow characteristics'
        }
    }
    SOIL_DEPTHS = {
        'Tpgks': [5, 10, 15, 20],
        'Tpg': [2, 5, 10, 15, 20, 40, 60, 80, 120, 160, 240, 320]
    }
    PRESSURE_LEVELS = list(range(1000, 0, -10))
    PRECIP_COLUMNS = ['observed_precip', 'corrected_precip', 'liquid_corrected_precip', 'mixed_corrected_precip', 'solid_corrected_precip']

class DataParser:
    @staticmethod
    def parse_monthly_tab(file_path: Path, value_name: str) -> Optional[pd.DataFrame]:
        try:
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) >= 14:
                        try:
                            station_id = values[0]
                            year = int(values[1])
                            for month, value in enumerate(values[2:14], 1):
                                try:
                                    val = float(value)
                                    if val != 9999:
                                        data.append({'station_id': station_id, 'year': year, 'month': month, 'value': val})
                                except ValueError:
                                    continue
                        except (ValueError, IndexError):
                            continue
            if not data:
                return None
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01')
            df = df.rename(columns={'value': value_name})
            return df
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    @staticmethod
    def parse_snow_data(file_path: Path) -> Optional[pd.DataFrame]:
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) < 5:
                        continue
                    try:
                        row_data = {'station_id': values[0], 'year': int(values[1]), 'month': int(values[2]), 'day': int(values[3])}
                        year = row_data['year']
                        month = row_data['month']
                        day = row_data['day']
                        if not (1900 <= year <= 2024 and 1 <= month <= 12 and 1 <= day <= 31):
                            continue
                        if len(values) > 4:
                            try:
                                depth = float(values[4])
                                if depth not in [9999, 9, -999] and 0 <= depth <= 1000:
                                    row_data['snow_depth'] = depth
                            except ValueError:
                                pass
                        if len(values) > 5:
                            try:
                                coverage = float(values[5])
                                if coverage not in [9999, 9, -999] and 0 <= coverage <= 100:
                                    row_data['snow_coverage'] = coverage
                            except ValueError:
                                pass
                        if len(values) > 6:
                            row_data['depth_info'] = values[6]
                        if len(values) > 7:
                            row_data['depth_quality'] = values[7]
                        if len(values) > 8:
                            row_data['temp_info'] = values[8]
                        if 'snow_depth' in row_data or 'snow_coverage' in row_data:
                            data.append(row_data)
                    except (ValueError, IndexError):
                        continue
            if not data:
                print(f"No valid data found in {file_path}")
                return None
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-' + df['day'].astype(str).str.zfill(2))
            print(f"Successfully processed {file_path}")
            print(f"Found {len(df)} valid records")
            return df
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    @staticmethod
    def parse_soil_temperature(file_path: Path, depths: List[int]) -> Optional[pd.DataFrame]:
        try:
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) < 4 + len(depths):
                        continue
                    try:
                        station = values[0]
                        year = int(values[1])
                        month = int(values[2])
                        day = int(values[3])
                        if not (1900 <= year <= 2024 and 1 <= month <= 12 and 1 <= day <= 31):
                            continue
                        for i, depth in enumerate(depths):
                            try:
                                temp = float(values[4 + i])
                                if temp not in [9999, 9, -999] and -90 <= temp <= 50:
                                    data.append({'station_id': station, 'year': year, 'month': month, 'day': day, 'depth_cm': depth, 'temperature': temp})
                            except ValueError:
                                continue
                    except ValueError:
                        continue
            if not data:
                print(f"No valid data found in {file_path}")
                return None
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-' + df['day'].astype(str).str.zfill(2))
            print(f"Successfully processed {file_path}")
            print(f"Found {len(df)} valid records")
            return df
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    @staticmethod
    def parse_pressure_data(file_path: Path, pressure_levels: List[int]) -> Optional[pd.DataFrame]:
        try:
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) < 4 + len(pressure_levels):
                        continue
                    try:
                        station = values[0]
                        year = int(values[1])
                        month = int(values[2])
                        day = int(values[3])
                        if not (1900 <= year <= 2024 and 1 <= month <= 12 and 1 <= day <= 31):
                            continue
                        for i, pressure in enumerate(pressure_levels):
                            try:
                                value = float(values[4 + i])
                                if value not in [9999, 9, -999]:
                                    data.append({'station_id': station, 'year': year, 'month': month, 'day': day, 'pressure_hpa': pressure, 'value': value})
                            except ValueError:
                                continue
                    except ValueError:
                        continue
            if not data:
                return None
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-' + df['day'].astype(str).str.zfill(2))
            return df
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    @staticmethod
    def parse_daily_data(file_path: Path, columns: List[str]) -> Optional[pd.DataFrame]:
        try:
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) >= len(columns):
                        try:
                            row_data = {'station_id': values[0], 'year': int(values[1]), 'month': int(values[2]), 'day': int(values[3])}
                            year = row_data['year']
                            month = row_data['month']
                            day = row_data['day']
                            if not (1900 <= year <= 2024 and 1 <= month <= 12 and 1 <= day <= 31):
                                continue
                            for i, col_name in enumerate(columns[4:], start=4):
                                if i < len(values):
                                    try:
                                        val = values[i]
                                        if col_name in ['tmin', 'tmean', 'tmax', 'precip']:
                                            val = float(val)
                                            if val not in [9999, 9]:
                                                row_data[col_name] = val
                                        elif col_name in ['tflag', 'qtmin', 'qtmean', 'qtmax', 'precip_flag', 'precip_quality']:
                                            if val not in ['9999', '9']:
                                                row_data[col_name] = val
                                    except ValueError:
                                        continue
                            if len(row_data) > 4:
                                data.append(row_data)
                        except (ValueError, IndexError):
                            continue
            if not data:
                print(f"No valid data found in {file_path}")
                return None
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-' + df['day'].astype(str).str.zfill(2))
            print(f"Successfully processed {file_path}")
            print(f"Found {len(df)} valid records")
            return df
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

class RussianMeteoProcessor:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.columns = ColumnDefinitions()
        self.parser = DataParser()
        self.known_subdirs = {
            'dailyairtemperatureandprecipitation_1812-2001': ('Tttr', self.columns.DAILY_FORMATS['Tttr']),
            'dailysoiltemperature_1984-2012': ('Tpgks', {'depths': self.columns.SOIL_DEPTHS['Tpgks']}),
            'dailysoiltemperatureatdepth_1963-2022': ('Tpg', {'depths': self.columns.SOIL_DEPTHS['Tpg']}),
            'dailysnowcovercharacteristics_1812-2001': ('Snow', self.columns.DAILY_FORMATS['Snow']),
            'atmboundarylevelparameters': ('Bound', {'pressure_levels': self.columns.PRESSURE_LEVELS}),
            'atmparameters': ('Ammv', {'pressure_levels': self.columns.PRESSURE_LEVELS})
        }
    
    def process_file(self, file_path: Path, directory_name: str) -> Optional[pd.DataFrame]:
        try:
            if file_path.name in self.columns.MONTHLY_FORMATS:
                format_info = self.columns.MONTHLY_FORMATS[file_path.name]
                return self.parser.parse_monthly_tab(file_path, format_info['name'])
            if directory_name in self.known_subdirs:
                subdir_name, format_info = self.known_subdirs[directory_name]
                if 'depths' in format_info:
                    return self.parser.parse_soil_temperature(file_path, format_info['depths'])
                elif 'pressure_levels' in format_info:
                    return self.parser.parse_pressure_data(file_path, format_info['pressure_levels'])
                elif 'columns' in format_info:
                    if 'Snow' in subdir_name:
                        return self.parser.parse_snow_data(file_path)
                    else:
                        return self.parser.parse_daily_data(file_path, format_info['columns'])
            return None
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def process_directory(self, directory: Path) -> Optional[pd.DataFrame]:
        data_frames = []
        dir_name = directory.name
        for tab_file in directory.glob('*.tab'):
            print(f"Processing {tab_file.name}")
            df = self.process_file(tab_file, dir_name)
            if df is not None:
                df['source_file'] = tab_file.name
                data_frames.append(df)
        if dir_name in self.known_subdirs:
            subdir_name, _ = self.known_subdirs[dir_name]
            subdir = directory / subdir_name
            if subdir.exists():
                for data_file in subdir.glob('*.*'):
                    print(f"Processing {data_file.relative_to(self.base_path)}")
                    df = self.process_file(data_file, dir_name)
                    if df is not None:
                        df['source_file'] = f"{subdir_name}/{data_file.name}"
                        data_frames.append(df)
        if data_frames:
            combined_df = pd.concat(data_frames, ignore_index=True)
            combined_df['source_dir'] = dir_name
            return combined_df
        return None
    
    def process_all_data(self) -> Dict[str, pd.DataFrame]:
        results = {}
        for directory in self.base_path.iterdir():
            if directory.is_dir():
                print(f"\nProcessing {directory.name} data...")
                df = self.process_directory(directory)
                if df is not None:
                    results[directory.name] = df
                    print(f"\nSummary for {directory.name}:")
                    print(f"Total records: {len(df)}")
                    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
                    print(f"Unique stations: {df['station_id'].nunique()}")
                    if 'depth_cm' in df.columns:
                        print("\nDepth measurements:")
                        print(df.groupby('depth_cm')['temperature'].describe())
                    if 'snow_depth' in df.columns:
                        print("\nSnow depth statistics:")
                        print(df['snow_depth'].describe())
                    if 'pressure_hpa' in df.columns:
                        print("\nPressure levels recorded:")
                        print(sorted(df['pressure_hpa'].unique()))
        return results
    
    def save_results(self, results: Dict[str, pd.DataFrame], output_dir: str = '/Users/[USER]/Library/CloudStorage/OneDrive-[REDACTED_AFFILIATION]/zerocurtain/newrun'):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        for data_type, df in results.items():
            output_file = output_dir / f"{data_type}_data.csv"
            df.to_csv(output_file, index=False)
            metadata = {
                'total_records': len(df),
                'date_range': {'start': df['datetime'].min().strftime('%Y-%m-%d'), 'end': df['datetime'].max().strftime('%Y-%m-%d')},
                'stations': {'count': df['station_id'].nunique(), 'ids': sorted(df['station_id'].unique().tolist())},
                'columns': {col: str(df[col].dtype) for col in df.columns}
            }
            if 'depth_cm' in df.columns:
                metadata['depths'] = sorted(df['depth_cm'].unique().tolist())
            if 'pressure_hpa' in df.columns:
                metadata['pressure_levels'] = sorted(df['pressure_hpa'].unique().tolist())
            if 'snow_depth' in df.columns:
                metadata['snow_depth_range'] = {'min': float(df['snow_depth'].min()), 'max': float(df['snow_depth'].max()), 'mean': float(df['snow_depth'].mean())}
            metadata_file = output_dir / f"{data_type}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"\nSaved {data_type} data to {output_file}")
            print(f"Saved metadata to {metadata_file}")

def main():
    base_path = '/Users/[USER]/Library/CloudStorage/OneDrive-[REDACTED_AFFILIATION]/zerocurtain/ru/old'
    processor = RussianMeteoProcessor(base_path)
    print("Starting data processing...")
    results = processor.process_all_data()
    print("\nSaving results...")
    processor.save_results(results)
    print("\nProcessing complete!")
    return processor, results

if __name__ == '__main__':
    processor, results = main()

def convert_to_decimal(coord):
    coord = str(coord).strip()
    try:
        return float(coord)
    except ValueError:
        pass
    match = re.match(r"(\d+)°\s*(\d+)'?", coord)
    if match:
        degrees = int(match.group(1))
        minutes = int(match.group(2))
        decimal_value = degrees + (minutes / 60)
        return round(decimal_value, 6)
    return None

def process_meteo_file(input_path, output_path, latlon_df):
    try:
        meteo_df = pd.read_csv(input_path)
        merged_df = meteo_df.merge(latlon_df, on="station_id", how="left")
        merged_df['latitude'] = merged_df['latitude'].apply(convert_to_decimal)
        merged_df['longitude'] = merged_df['longitude'].apply(convert_to_decimal)
        merged_df.to_csv(output_path)
        print(f"Successfully processed: {Path(input_path).name}")
        return True
    except Exception as e:
        print(f"Error processing {Path(input_path).name}: {str(e)}")
        return False

def batch_process_meteo_files(input_dir, output_dir, latlon_file_path):
    latlon_df = pd.read_csv(latlon_file_path)
    latlon_df = latlon_df.drop(columns=['Unnamed: 0'])
    latlon_df.columns = ['station_id', 'station_name', 'latitude', 'longitude', 'data_source']
    os.makedirs(output_dir, exist_ok=True)
    input_files = list(Path(input_dir).glob('*.csv'))
    results = {'successful': [], 'failed': []}
    for input_file in input_files:
        output_file = Path(output_dir) / f"{input_file.stem}_ru_update.csv"
        if process_meteo_file(input_file, output_file, latlon_df):
            results['successful'].append(input_file.name)
        else:
            results['failed'].append(input_file.name)
    print(f"\nProcessing Summary:")
    print(f"Successfully processed: {len(results['successful'])} files")
    print(f"Failed to process: {len(results['failed'])} files")
    if results['failed']:
        print("\nFailed files:")
        for file in results['failed']:
            print(f"- {file}")

input_directory = "/Users/[USER]/Library/CloudStorage/OneDrive-[REDACTED_AFFILIATION]/zerocurtain/processed_meteo_data"
output_directory = "/Users/[USER]/Downloads"
latlon_file_path = "/Users/[USER]/Downloads/latlon_ru_update.csv"
batch_process_meteo_files(input_directory, output_directory, latlon_file_path)

directory = '/Users/[USER]/Desktop/ru/new'
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        variable_name = os.path.splitext(filename)[0].split('_')[0]
        try:
            globals()[variable_name] = pd.read_csv(filepath).drop(columns=['Unnamed: 0'])
            print(f"Successfully read {filename} into DataFrame: {variable_name}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

dailyairtemperatureandprecipitation
snowmeasurementsurveys, dailysnowcovercharacteristics, dailysoiltemperature, dailyairtemperatureandprecipitation, dailysoiltemperatureatdepth
snowmeasurementsurveys

dst1 = pd.read_csv('/Users/[USER]/Desktop/ru/new/dailysoiltemperature_1984-2012_data_ru_update.csv').drop(columns=['Unnamed: 0'])
dst2 = pd.read_csv('/Users/[USER]/Desktop/ru/new/dailysoiltemperatureatdepth_1963-2022_data_ru_update.csv').drop(columns=['Unnamed: 0'])
dst1 = dst1[['station_id','datetime','latitude','longitude','depth_cm','temperature']].sort_values('datetime').reset_index(drop=True)
dst2 = dst2[['station_id','datetime','latitude','longitude','depth_cm','temperature']].sort_values('datetime').reset_index(drop=True)
ru_dst = pd.concat([dst1,dst2]).sort_values('datetime').reset_index(drop=True)

def add_depth_zones(depth):
    if depth <= 0.25:
        return 'shallow'
    elif 0.25 < depth <= 0.5:
        return 'intermediate'
    elif 0.5 < depth <= 1.0:
        return 'deep'
    elif depth > 1.0:
        return 'very_deep'
    else:
        return 'unknown'

ru_dst['depth'] = ru_dst.depth_cm/100
ru_dst = ru_dst.drop('depth_cm',axis=1)
ru_dst['site_id'] = 'RU_Meteo_' + ru_dst['station_id'].astype(str)
ru_dst['depth_zone'] = ru_dst['depth'].apply(add_depth_zones)
ru_dst

def locate_missing_coordinates(ru_dst, base_path='/Users/[USER]/Desktop/ru/old'):
    missing_stations = ru_dst[ru_dst['latitude'].isna() | ru_dst['longitude'].isna()]['station_id'].unique()
    station_coords = {}
    meteo_files = ['monthlyaverageairtemperature_1743-2023_data.csv', 'monthlyprecipitation_1936-2015_data.csv', 'dailyairtemperatureandprecipitation_1812-2001_data.csv', 'averagemonthlytotalclouds_1966-2023_data.csv']
    for file in meteo_files:
        try:
            file_path = Path(base_path).parent / 'new' / file
            if file_path.exists():
                df = pd.read_csv(file_path)
                for station in missing_stations:
                    station_data = df[df['station_id'] == station]
                    if not station_data.empty:
                        lat = station_data['latitude'].iloc[0]
                        lon = station_data['longitude'].iloc[0]
                        if pd.notna(lat) and pd.notna(lon):
                            station_coords[station] = (lat, lon)
                print(f"Found coordinates for {len(station_coords)} stations in {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    for subdir in Path(base_path).glob('**/'):
        for metadata_file in subdir.glob('*metadata.json'):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if 'stations' in metadata:
                        for station_info in metadata.get('stations', {}).get('data', []):
                            station_id = str(station_info.get('id'))
                            if station_id in missing_stations:
                                lat = station_info.get('latitude')
                                lon = station_info.get('longitude')
                                if lat is not None and lon is not None:
                                    station_coords[station_id] = (lat, lon)
            except Exception as e:
                print(f"Error processing {metadata_file}: {e}")
    doc_extensions = ['.txt', '.doc', '.docx', '.dll']
    for subdir in Path(base_path).glob('**/'):
        for ext in doc_extensions:
            for doc_file in subdir.glob(f'*{ext}'):
                try:
                    if ext == '.txt':
                        with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            pattern = r'Station\s+(\d+)[:\s]+lat\s*([-]?\d+\.?\d*)[,\s]+lon\s*([-]?\d+\.?\d*)'
                            matches = re.findall(pattern, content)
                            for match in matches:
                                station_id, lat, lon = match
                                if station_id in missing_stations:
                                    station_coords[station_id] = (float(lat), float(lon))
                except Exception as e:
                    print(f"Error processing {doc_file}: {e}")
    updated_df = ru_dst.copy()
    for station, (lat, lon) in station_coords.items():
        mask = (updated_df['station_id'] == station) & (updated_df['latitude'].isna() | updated_df['longitude'].isna())
        updated_df.loc[mask, 'latitude'] = lat
        updated_df.loc[mask, 'longitude'] = lon
    initial_missing = ru_dst['latitude'].isna().sum()
    final_missing = updated_df['latitude'].isna().sum()
    print(f"\nInitial missing coordinates: {initial_missing}")
    print(f"Final missing coordinates: {final_missing}")
    print(f"Recovered coordinates for {initial_missing - final_missing} rows")
    return updated_df

ru_dst_updated = locate_missing_coordinates(ru_dst)

def analyze_station_coverage(ru_dst, base_path='/Users/[USER]/Desktop/ru/old'):
    station_coords = {}
    data_sources = {}
    missing_stations = set(ru_dst[ru_dst['latitude'].isna() | ru_dst['longitude'].isna()]['station_id'].unique())
    print(f"Number of stations with missing coordinates: {len(missing_stations)}")
    new_dir = Path(base_path).parent / 'new'
    print("\nScanning /new directory for CSVs...")
    for csv_file in new_dir.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            if 'station_id' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
                valid_coords = df[df['station_id'].notna() & df['latitude'].notna() & df['longitude'].notna()]
                stations_found = set(valid_coords['station_id'].unique())
                stations_recovered = stations_found.intersection(missing_stations)
                if stations_recovered:
                    print(f"\nFound {len(stations_recovered)} matching stations in {csv_file.name}")
                    for station in stations_recovered:
                        station_data = valid_coords[valid_coords['station_id'] == station].iloc[0]
                        station_coords[station] = {'latitude': station_data['latitude'], 'longitude': station_data['longitude'], 'source': csv_file.name}
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    meteo_dir = Path('/Users/[USER]/Desktop/Research/Code/processed_meteo_data')
    print("\nScanning processed_meteo_data directory...")
    for csv_file in meteo_dir.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            if 'station_id' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
                valid_coords = df[df['station_id'].notna() & df['latitude'].notna() & df['longitude'].notna()]
                new_stations = set(valid_coords['station_id'].unique()) - set(station_coords.keys())
                new_stations = new_stations.intersection(missing_stations)
                if new_stations:
                    print(f"\nFound {len(new_stations)} additional stations in {csv_file.name}")
                    for station in new_stations:
                        station_data = valid_coords[valid_coords['station_id'] == station].iloc[0]
                        station_coords[station] = {'latitude': station_data['latitude'], 'longitude': station_data['longitude'], 'source': csv_file.name}
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    print("\nAnalyzing patterns in missing data:")
    missing_df = ru_dst[ru_dst['latitude'].isna() | ru_dst['longitude'].isna()]
    print("\nSample of records with missing coordinates:")
    print(missing_df[['station_id', 'site_id', 'datetime']].head())
    print("\nStation ID patterns:")
    missing_station_ids = missing_df['station_id'].value_counts()
    print(f"Number of unique missing station IDs: {len(missing_station_ids)}")
    print("\nMost common missing station IDs:")
    print(missing_station_ids.head())
    return station_coords, missing_stations

station_coords, missing_stations = analyze_station_coverage(ru_dst)
if station_coords:
    print("\nApplying recovered coordinates...")
    ru_dst_updated = ru_dst.copy()
    for station, coord_info in station_coords.items():
        mask = (ru_dst_updated['station_id'] == station) & (ru_dst_updated['latitude'].isna() | ru_dst_updated['longitude'].isna())
        ru_dst_updated.loc[mask, 'latitude'] = coord_info['latitude']
        ru_dst_updated.loc[mask, 'longitude'] = coord_info['longitude']
    initial_missing = ru_dst['latitude'].isna().sum()
    final_missing = ru_dst_updated['latitude'].isna().sum()
    print(f"\nInitial missing coordinates: {initial_missing}")
    print(f"Final missing coordinates: {final_missing}")
    print(f"Recovered coordinates for {initial_missing - final_missing} rows")
else:
    print("\nNo coordinates were recovered. Need to investigate alternative sources.")

ru_dst_updated = ru_dst_updated.sort_values('datetime').reset_index(drop=True)
ru_dst_updated = ru_dst_updated[ru_dst_updated.latitude>=30]
ru_dst = ru_dst_updated
ru_dst = ru_dst.sort_values('datetime').reset_index(drop=True)
ru_dst.to_csv('ru_dst.csv', index=False)

ru_dst = pd.read_csv('ru_dst_final_022425.csv')
ru_dst['year'] = pd.to_datetime(ru_dst.datetime).dt.year
ru_dst = ru_dst[['datetime', 'year', 'season', 'latitude', 'longitude', 'station_id', 'site_id', 'station_name', 'data_source', 'soil_temp', 'soil_temp_depth', 'soil_temp_depth_zone']]

def standardize_soil_temperature(ru_dst):
    std_df = ru_dst.copy()
    if 'temp_quality_flag' not in std_df.columns:
        std_df['temp_quality_flag'] = 'valid'
    impossible_mask = (std_df['soil_temp'] < -70) | (std_df['soil_temp'] > 50)
    std_df.loc[impossible_mask, 'temp_quality_flag'] = 'physically_impossible'
    suspicious_mask = ((std_df['soil_temp'] < -60) | (std_df['soil_temp'] > 40)) & ~impossible_mask
    std_df.loc[suspicious_mask, 'temp_quality_flag'] = 'suspicious'
    grouped = std_df.groupby(['station_id', 'datetime'])
    inconsistent_indices = []
    for _, group in grouped:
        if len(group) >= 2:
            sorted_group = group.sort_values('soil_temp_depth')
            is_winter = sorted_group['season'].iloc[0] in ['Winter', 'Fall']
            depths = sorted_group['soil_temp_depth'].values
            temps = sorted_group['soil_temp'].values
            for i in range(len(depths)-1):
                if depths[i+1] - depths[i] > 0:
                    temp_gradient = (temps[i+1] - temps[i]) / (depths[i+1] - depths[i])
                    if abs(temp_gradient) > 15:
                        inconsistent_indices.extend(sorted_group.iloc[i:i+2].index.tolist())
    std_df.loc[inconsistent_indices, 'temp_quality_flag'] = 'inconsistent_profile'
    total_records = len(std_df)
    valid_records = (std_df['temp_quality_flag'] == 'valid').sum()
    print(f"Total records: {total_records}")
    print(f"Valid records: {valid_records} ({valid_records/total_records*100:.1f}%)")
    print(f"Invalid records: {total_records - valid_records} ({(total_records - valid_records)/total_records*100:.1f}%)")
    flag_counts = std_df['temp_quality_flag'].value_counts()
    for flag, count in flag_counts.items():
        print(f"  {flag}: {count} ({count/total_records*100:.1f}%)")
    valid_temps = std_df.loc[std_df['temp_quality_flag'] == 'valid', 'soil_temp']
    print(f"\nValid temperature range: {valid_temps.min():.1f}°C to {valid_temps.max():.1f}°C")
    print(f"Valid temperature mean: {valid_temps.mean():.1f}°C")
    std_df['soil_temp_standardized'] = std_df['soil_temp']
    return std_df

standardized_ru_dst = standardize_soil_temperature(ru_dst)
valid_ru_dst = standardized_ru_dst[standardized_ru_dst['temp_quality_flag'] == 'valid']
valid_ru_dst = valid_ru_dst.sort_values('datetime').reset_index(drop=True)
ru_dst = valid_ru_dst
ru_dst.to_csv('ru_dst_030525.csv', index=False)
len(np.unique(pd.read_csv('ru_dst_final_022425.csv').sort_values('datetime').reset_index(drop=True).site_id))

# Archived
import re
from pathlib import Path
from datetime import datetime
import calendar
import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
warnings.filterwarnings('ignore')

class ValidationMetrics:
    def __init__(self):
        self.total_rows_processed = 0
        self.rows_dropped = 0
        self.invalid_dates = 0
        self.invalid_measurements = 0
        self.site_metrics: Dict[str, Dict] = {}
        self.processing_errors: List[str] = []
    
    def log_error(self, error_msg: str):
        self.processing_errors.append(error_msg)
    
    def update_site_metrics(self, site_id: str, metric_name: str, value: int):
        if site_id not in self.site_metrics:
            self.site_metrics[site_id] = {}
        self.site_metrics[site_id][metric_name] = self.site_metrics[site_id].get(metric_name, 0) + value

class ALTDataParser:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.min_year = 1950
        self.max_year = 2024
        self.metrics = ValidationMetrics()
        self.debug_info = {}
        self.month_map = self._initialize_month_mapping()
        self.special_handlers = {
            'R23': self.handle_r23_format,
            'R32': self.handle_r32_format,
            'R38': self.handle_r38_format,
            'R18A': self.handle_r38_format,
            'R40': self.handle_r40_format,
            'R24': self.handle_r24_format,
            'R27': self.handle_r27_format
        }
        self.processed_sites: Set[str] = set()
    
    def _initialize_month_mapping(self) -> Dict[str, int]:
        month_map = {}
        for i, month in enumerate(calendar.month_name[1:], 1):
            month_map[month.lower()] = i
            month_map[month[:3].lower()] = i
        variations = {'sept': 9, 'sep': 9, 'july': 7, 'jun': 6, 'jul': 7, 'aug': 8}
        month_map.update(variations)
        return month_map
    
    def standardize_site_id(self, site_id: str) -> str:
        site_id = site_id.upper().strip()
        if not site_id.startswith('R'):
            site_id = f"R{site_id}"
        match = re.match(r'(R\d+)([A-Za-z])?', site_id)
        if match:
            base, suffix = match.groups()
            if suffix:
                site_id = f"{base}{suffix.upper()}"
        return site_id
    
    def convert_two_digit_year(self, year: int) -> int:
        if year < 100:
            return 2000 + year if year < 50 else 1900 + year
        return year
    
    def validate_date(self, year: int, month: int, day: int) -> bool:
        try:
            if self.min_year <= year <= self.max_year and 1 <= month <= 12 and 1 <= day <= 31:
                datetime(year, month, day)
                return True
        except ValueError:
            return False
        return False
    
    def validate_measurement(self, value: float, site_id: str) -> bool:
        if pd.isna(value):
            return False
        if not 0 <= value <= 1000:
            self.metrics.update_site_metrics(site_id, 'invalid_measurements', 1)
            return False
        return True
    
    def _validate_year_range(self, start_year: int, end_year: int) -> bool:
        if not (self.min_year <= start_year <= self.max_year and 
                self.min_year <= end_year <= self.max_year):
            return False
        if end_year < start_year:
            return False
        return True
    
    def extract_year_from_filename(self, filename: str) -> Tuple[Optional[int], Optional[int]]:
        range_patterns = [
            r'(?:19|20)?(\d{2})[-_](?:19|20)?(\d{2})',
            r'(\d{4})[-_](\d{4})',
            r'(\d{4})[-_](\d{2})',
        ]
        for pattern in range_patterns:
            if match := re.search(pattern, filename):
                start_year = int(match.group(1))
                end_year = int(match.group(2))
                if start_year < 100:
                    start_year = 2000 + start_year if start_year < 50 else 1900 + start_year
                if end_year < 100:
                    end_year = 2000 + end_year if end_year < 50 else 1900 + end_year
                return start_year, end_year
        year_matches = re.findall(r'(?:19|20)?(\d{2}|\d{4})', filename)
        if year_matches:
            years = []
            for year in year_matches:
                year = int(year)
                if year < 100:
                    year = 2000 + year if year < 50 else 1900 + year
                if self.min_year <= year <= self.max_year:
                    years.append(year)
            if years:
                return min(years), max(years)
        return None, None
    
    def identify_alt_columns(self, df: pd.DataFrame, filename: str) -> dict:
        alt_patterns = [
            r'al-', r'al-\*', r'alt_', r'alt\(cm\)', r'thaw depth',
            r'TD\d{6}', r'alt', r'ALT', r'thaw', r'al\(cm\)'
        ]
        exclusion_patterns = [
            'temp', 'temperature', 'moisture', 'moist', 'precip',
            'rain', 'snow', 'depth_to'
        ]
        file_start_year, file_end_year = self.extract_year_from_filename(filename)
        alt_cols_with_dates = {}
        for col in df.columns:
            col_str = str(col).lower()
            if any(term in col_str for term in exclusion_patterns):
                continue
            if any(re.search(pattern, col_str, re.IGNORECASE) for pattern in alt_patterns):
                year_match = re.search(r'(?:19|20)?(\d{2})', col_str)
                if year_match:
                    year = self.convert_two_digit_year(int(year_match.group(1)))
                    if self.min_year <= year <= self.max_year:
                        alt_cols_with_dates[col] = f"{year}-07-15"
                elif file_start_year and file_end_year:
                    alt_cols_with_dates[col] = f"{file_start_year}-07-15"
        return alt_cols_with_dates
    
    def parse_complex_date(self, date_str: str) -> str:
        try:
            date_str = str(date_str).lower().strip()
            if re.match(r'\d{1,2}/\d{1,2}/\d{2}', date_str):
                month, day, year = map(int, date_str.split('/'))
                year = 2000 + year if year < 50 else 1900 + year
                return f"{year:04d}-{month:02d}-{day:02d}"
            match = re.match(r'(\d{1,2})-([a-zA-Z]{3})-(\d{2})', date_str)
            if match:
                day, month_str, year = match.groups()
                month = self.month_map.get(month_str.lower())
                if month:
                    year = 2000 + int(year) if int(year) < 50 else 1900 + int(year)
                    return f"{year:04d}-{month:02d}-{int(day):02d}"
            match = re.match(r'(\d{4}),\s*([a-zA-Z]{3,})\.?\s*(\d{1,2})', date_str)
            if match:
                year, month_str, day = match.groups()
                month = self.month_map.get(month_str.lower())
                if month:
                    return f"{year}-{month:02d}-{int(day):02d}"
            if re.match(r'\d{8}', date_str):
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                return f"{year}-{month:02d}-{day:02d}"
            if re.match(r'\d{4}', date_str):
                year = int(date_str)
                return f"{year}-07-15"
        except (ValueError, AttributeError, KeyError) as e:
            print(f"Error parsing date {date_str}: {str(e)}")
            return None
        return None
    
    def handle_r23_format(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        dfs = []
        td_cols = [col for col in df.columns
                  if isinstance(col, str) and
                  col.startswith('TD') and
                  len(col) == 8 and
                  col[2:].isdigit()]
        for col in td_cols:
            date_str = col[2:]
            try:
                yy = int(date_str[:2])
                mm = int(date_str[2:4])
                dd = int(date_str[4:])
                year = 2000 + yy if yy < 50 else 1900 + yy
                date_str = f"{year:04d}-{mm:02d}-{dd:02d}"
                alt_values = pd.to_numeric(df[col], errors='coerce')
                valid_values = alt_values.dropna()
                if not valid_values.empty:
                    temp_df = pd.DataFrame({
                        'datetime': date_str,
                        'site_id': 'R23',
                        'alt': valid_values
                    })
                    dfs.append(temp_df)
            except (ValueError, IndexError) as e:
                print(f"Error processing column {col}: {str(e)}")
                continue
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    def handle_r27_format(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        dfs = []
        site_id = kwargs.get('site_id', 'R27')
        def identify_alt_columns(df):
            alt_patterns = [
                r'ActiveLayerDepth',
                r'TDYYMMDD',
                r'TD\d{6}',
                'Глубина СТС',
                'СТС'
            ]
            alt_cols = []
            for col in df.columns:
                col_str = str(col).strip()
                if any(re.search(pattern, col_str, re.IGNORECASE) for pattern in alt_patterns):
                    alt_cols.append(col)
            return alt_cols
        try:
            alt_cols = identify_alt_columns(df)
            for col in alt_cols:
                col_str = str(col)
                date_str = None
                if match := re.search(r'TD(\d{6})', col_str):
                    date_str = match.group(1)
                    yy = int(date_str[:2])
                    mm = int(date_str[2:4])
                    dd = int(date_str[4:])
                    year = 2000 + yy if yy < 50 else 1900 + yy
                    date_str = f"{year:04d}-{mm:02d}-{dd:02d}"
                elif match := re.search(r'(\d{4})', col_str):
                    year = int(match.group(1))
                    if self.min_year <= year <= self.max_year:
                        date_str = f"{year}-07-15"
                if date_str:
                    values = df[col].astype(str).apply(lambda x: 
                        re.sub(r'[^\d.]', '', str(x).replace(',', '.'))
                    )
                    alt_values = pd.to_numeric(values, errors='coerce')
                    valid_values = alt_values[alt_values.apply(lambda x: self.validate_measurement(x, site_id))]
                    if not valid_values.empty:
                        temp_df = pd.DataFrame({
                            'datetime': date_str,
                            'site_id': site_id,
                            'alt': valid_values
                        })
                        dfs.append(temp_df)
        except Exception as e:
            self.metrics.log_error(f"Error processing R27 format: {str(e)}")
        if dfs:
            result_df = pd.concat(dfs, ignore_index=True)
            print(f"Successfully extracted {len(result_df)} measurements for R27")
            return result_df
        return pd.DataFrame()
    
    def handle_r32_format(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        dfs = []
        date_cols = [col for col in df.columns
                    if isinstance(col, (str, int)) and
                    str(col).isdigit() and
                    len(str(col)) == 8]
        for col in date_cols:
            try:
                date_str = str(col)
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:])
                if self.validate_date(year, month, day):
                    date_str = f"{year:04d}-{month:02d}-{day:02d}"
                    alt_values = pd.to_numeric(df[col], errors='coerce')
                    valid_values = alt_values.dropna()
                    if not valid_values.empty:
                        temp_df = pd.DataFrame({
                            'datetime': date_str,
                            'site_id': 'R32',
                            'alt': valid_values
                        })
                        dfs.append(temp_df)
            except (ValueError, IndexError) as e:
                print(f"Error processing column {col}: {str(e)}")
                continue
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    def handle_r38_format(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            header_row = None
            for i in range(min(10, len(df))):
                if df.iloc[i].astype(str).str.contains('al\(cm\)|alt\(cm\)').any():
                    header_row = i
                    break
            if header_row is None:
                return pd.DataFrame()
            headers = df.iloc[header_row]
            data = df.iloc[header_row + 1:].reset_index(drop=True)
            df_cleaned = pd.DataFrame(data.values, columns=headers)
            date_patterns = [
                (r'alt\(cm\)-(\d{1,2})/(\d{1,2})/(\d{4})', 'MDY'),
                (r'al\(cm\)-(\d{1,2})/(\d{1,2})/(\d{2})', 'MDY'),
                (r'al\(cm\)-(\d{1,2})/(\d{1,2})/(\d{4})', 'MDY')
            ]
            dfs = []
            for col in df_cleaned.columns:
                col_str = str(col).strip()
                if pd.isna(col_str) or col_str == '' or col_str == '-':
                    continue
                date_str = None
                for pattern, date_format in date_patterns:
                    if match := re.search(pattern, col_str, re.IGNORECASE):
                        try:
                            if date_format == 'MDY':
                                month, day, year = map(int, match.groups())
                                if year < 100:
                                    year = 2000 + year if year < 50 else 1900 + year
                                if self.validate_date(year, month, day):
                                    date_str = f"{year:04d}-{month:02d}-{day:02d}"
                                    break
                        except (ValueError, TypeError, AttributeError) as e:
                            print(f"Error parsing date from column {col}: {str(e)}")
                            continue
                if date_str:
                    alt_values = df_cleaned[col].replace(['no data', '-'], np.nan)
                    alt_values = alt_values.apply(lambda x: str(x).strip('-') if isinstance(x, str) else x)
                    alt_values = pd.to_numeric(alt_values, errors='coerce')
                    valid_values = alt_values.dropna()
                    if not valid_values.empty:
                        temp_df = pd.DataFrame({
                            'datetime': date_str,
                            'site_id': kwargs.get('site_id', 'R38'),
                            'alt': valid_values
                        })
                        dfs.append(temp_df)
            if dfs:
                result_df = pd.concat(dfs, ignore_index=True)
                print(f"Successfully extracted {len(result_df)} measurements for {kwargs.get('site_id', 'R38')}")
                return result_df
        except Exception as e:
            print(f"Error processing R38 format: {str(e)}")
        return pd.DataFrame()
    
    def handle_r40_format(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if 'datetime' not in df.columns:
            return pd.DataFrame()
        df['datetime'] = df['datetime'].apply(lambda x:
            f"{str(x)[:4]}-{str(x)[4:6]}-{str(x)[6:8]}"
            if len(str(x)) == 8 else None)
        alt_cols = [col for col in df.columns
                   if isinstance(col, str) and 'alt' in col.lower()]
        if not alt_cols:
            return pd.DataFrame()
        df['alt'] = pd.to_numeric(df[alt_cols[0]], errors='coerce')
        df['site_id'] = 'R40'
        return df[['datetime', 'site_id', 'alt']].dropna()
    
    def handle_r24_format(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        dfs = []
        for col in df.columns:
            if not isinstance(col, str):
                continue
            date_str = None
            if re.match(r'\d{1,2}/\d{1,2}/\d{2}', col):
                date_str = self.parse_complex_date(col)
            elif re.match(r'\d{4},\s*[a-zA-Z]{3,}\.?\s*\d{1,2}', col):
                date_str = self.parse_complex_date(col)
            if date_str:
                temp_df = pd.DataFrame({
                    'datetime': date_str,
                    'site_id': 'R24',
                    'alt': pd.to_numeric(df[col], errors='coerce').dropna()
                })
                dfs.append(temp_df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def default_process(self, df: pd.DataFrame, filename: str, site_id: str) -> pd.DataFrame:
        dfs = []
        df.columns = [str(col).strip() for col in df.columns]
        alt_cols_with_dates = self.identify_alt_columns(df, filename)
        for col, date_str in alt_cols_with_dates.items():
            if date_str:
                alt_values = pd.to_numeric(df[col], errors='coerce')
                valid_values = alt_values.dropna()
                if not valid_values.empty:
                    temp_df = pd.DataFrame({
                        'datetime': date_str,
                        'site_id': site_id,
                        'alt': valid_values
                    })
                    dfs.append(temp_df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def validate_and_clean_dataframe(self, df: pd.DataFrame, site_id: str) -> pd.DataFrame:
        initial_rows = len(df)
        self.metrics.update_site_metrics(site_id, 'total_rows', initial_rows)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        invalid_dates = df['datetime'].isna().sum()
        self.metrics.update_site_metrics(site_id, 'invalid_dates', invalid_dates)
        df = df.dropna(subset=['datetime'])
        df['alt'] = pd.to_numeric(df['alt'], errors='coerce')
        valid_mask = df['alt'].apply(lambda x: self.validate_measurement(x, site_id))
        df = df[valid_mask]
        df = df[
            (df['datetime'].dt.year >= self.min_year) &
            (df['datetime'].dt.year <= self.max_year)
        ]
        df = df.sort_values(['datetime', 'alt'])
        df = df.drop_duplicates(['site_id', 'datetime', 'alt'], keep='first')
        final_rows = len(df)
        self.metrics.update_site_metrics(site_id, 'valid_rows', final_rows)
        self.metrics.rows_dropped += (initial_rows - final_rows)
        return df
    
    def process_file(self, file_path: Path) -> pd.DataFrame:
        try:
            site_id_match = re.match(r'(R\d+[A-Za-z]?)', file_path.stem, re.IGNORECASE)
            if not site_id_match:
                self.metrics.log_error(f"Invalid site ID format in filename: {file_path.name}")
                return pd.DataFrame()
            site_id = self.standardize_site_id(site_id_match.group(1))
            self.processed_sites.add(site_id)
            base_site = re.match(r'(R\d+)', site_id).group(1)
            self.debug_info[file_path.name] = {
                'site_id': site_id,
                'sheets_processed': [],
                'data_points': 0,
                'processing_method': 'default'
            }
            special_handler = self.special_handlers.get(base_site)
            xl = pd.ExcelFile(file_path)
            dfs = []
            for sheet in xl.sheet_names:
                try:
                    df = pd.read_excel(xl, sheet_name=sheet)
                    if df.empty:
                        continue
                    self.debug_info[file_path.name]['sheets_processed'].append(sheet)
                    result_df = pd.DataFrame()
                    if special_handler:
                        try:
                            self.debug_info[file_path.name]['processing_method'] = f'special_{base_site}'
                            result_df = special_handler(df, site_id=site_id, sheet_name=sheet)
                        except Exception as e:
                            print(f"Special handler failed for {file_path.name}, sheet {sheet}: {str(e)}")
                            result_df = pd.DataFrame()
                    if result_df.empty:
                        self.debug_info[file_path.name]['processing_method'] = 'default'
                        result_df = self.default_process(df, file_path.name, site_id)
                    if not result_df.empty:
                        dfs.append(result_df)
                except Exception as e:
                    print(f"Error processing sheet {sheet} in {file_path.name}: {str(e)}")
            if dfs:
                final_df = pd.concat(dfs, ignore_index=True)
                self.debug_info[file_path.name]['data_points'] = len(final_df)
                return self.validate_and_clean_dataframe(final_df, site_id)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
        return pd.DataFrame()
    
    def process_r_series(self, start_r: int = 11, end_r: int = 61) -> pd.DataFrame:
        combined_data = []
        processed_files = set()
        for file_path in self.base_dir.glob('**/*'):
            if ('alt' in file_path.name.lower() or 'ALT' in file_path.name) and \
               file_path.suffix.lower() in ['.xls', '.xlsx']:
                site_id_match = re.search(r'R(\d+)', file_path.stem, re.IGNORECASE)
                if site_id_match:
                    site_num = int(site_id_match.group(1))
                    if start_r <= site_num <= end_r:
                        file_key = (site_num, file_path.stem)
                        if file_key not in processed_files:
                            print(f"Processing {file_path.name}...")
                            df = self.process_file(file_path)
                            if not df.empty:
                                combined_data.append(df)
                            processed_files.add(file_key)
        if combined_data:
            final_df = pd.concat(combined_data, ignore_index=True)
            final_df['datetime'] = pd.to_datetime(final_df['datetime'])
            final_df = final_df[
                (final_df['datetime'].dt.year >= self.min_year) &
                (final_df['datetime'].dt.year <= self.max_year)
            ]
            final_df = final_df.sort_values(['site_id', 'datetime'])
            final_df = final_df.drop_duplicates(['site_id', 'datetime', 'alt'])
            return final_df
        return pd.DataFrame()
    
    def print_debug_summary(self):
        print("\nProcessing Summary:")
        print("-" * 50)
        total_data_points = 0
        sites_with_data = []
        processing_methods = {}
        for filename, info in self.debug_info.items():
            print(f"\nFile: {filename}")
            print(f"Site ID: {info['site_id']}")
            print(f"Sheets processed: {len(info['sheets_processed'])}")
            print(f"Processing method: {info['processing_method']}")
            print(f"Data points extracted: {info['data_points']}")
            total_data_points += info['data_points']
            if info['data_points'] > 0:
                sites_with_data.append(info['site_id'])
                processing_methods[info['site_id']] = info['processing_method']
        print("\nOverall Summary:")
        print("-" * 50)
        print(f"Total data points extracted: {total_data_points}")
        print(f"Total sites with data: {len(set(sites_with_data))}")
        print("Sites with data and their processing methods:")
        for site in sorted(set(sites_with_data)):
            print(f"  {site}: {processing_methods[site]}")
    
    def print_validation_summary(self):
        print("\nValidation Summary")
        print("=" * 50)
        print(f"Total sites processed: {len(self.processed_sites)}")
        print(f"Total rows processed: {self.metrics.total_rows_processed}")
        print(f"Total rows dropped: {self.metrics.rows_dropped}")
        print(f"Invalid dates encountered: {self.metrics.invalid_dates}")
        print(f"Invalid measurements: {self.metrics.invalid_measurements}")
        print("\nPer-site Statistics")
        print("=" * 50)
        for site_id, metrics in sorted(self.metrics.site_metrics.items()):
            print(f"\nSite: {site_id}")
            print(f"  Total rows: {metrics.get('total_rows', 0)}")
            print(f"  Valid rows: {metrics.get('valid_rows', 0)}")
            print(f"  Invalid dates: {metrics.get('invalid_dates', 0)}")
            print(f"  Invalid measurements: {metrics.get('invalid_measurements', 0)}")
        if self.metrics.processing_errors:
            print("\nProcessing Errors")
            print("=" * 50)
            for error in self.metrics.processing_errors:
                print(f"- {error}")

def clean_alt_data(df, alt_column='alt', min_valid_alt=0, max_valid_alt=500):
    df_clean = df.copy()
    df_clean['alt_quality'] = 'valid'
    total_records = len(df_clean)
    print(f"Total records before cleaning: {total_records}")
    negative_mask = df_clean[alt_column] < min_valid_alt
    df_clean.loc[negative_mask, 'alt_quality'] = 'invalid_negative'
    print(f"Negative values: {negative_mask.sum()} ({negative_mask.sum()/total_records:.2%})")
    likely_mm_mask = (df_clean[alt_column] > 1000) & (df_clean[alt_column] <= 5000)
    df_clean.loc[likely_mm_mask, 'alt_quality'] = 'likely_mm'
    print(f"Likely mm values (>1000 cm): {likely_mm_mask.sum()} ({likely_mm_mask.sum()/total_records:.2%})")
    extreme_mask = df_clean[alt_column] > max_valid_alt
    df_clean.loc[extreme_mask, 'alt_quality'] = 'invalid_extreme'
    print(f"Extreme values (>500 cm): {extreme_mask.sum()} ({extreme_mask.sum()/total_records:.2%})")
    for site in df_clean['site_id'].unique():
        site_mask = df_clean['site_id'] == site
        site_data = df_clean.loc[site_mask, alt_column]
        if len(site_data) < 5:
            continue
        Q1 = site_data.quantile(0.25)
        Q3 = site_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - 3 * IQR)
        upper_bound = Q3 + 3 * IQR
        outlier_mask = (site_mask & 
                        ((df_clean[alt_column] < lower_bound) | 
                         (df_clean[alt_column] > upper_bound)) & 
                        (df_clean['alt_quality'] == 'valid'))
        df_clean.loc[outlier_mask, 'alt_quality'] = 'statistical_outlier'
        print(f"Site {site} statistical outliers: {outlier_mask.sum()} ({outlier_mask.sum()/site_mask.sum():.2%})")
    df_clean['alt_corrected'] = df_clean[alt_column].copy()
    df_clean.loc[df_clean['alt_quality'] == 'likely_mm', 'alt_corrected'] = \
        df_clean.loc[df_clean['alt_quality'] == 'likely_mm', alt_column] / 10
    invalid_mask = df_clean['alt_quality'].isin(['invalid_negative', 'invalid_extreme'])
    df_clean.loc[invalid_mask, 'alt_corrected'] = np.nan
    valid_records = df_clean['alt_corrected'].notna().sum()
    print(f"\nRecords after cleaning: {valid_records} ({valid_records/total_records:.2%} of original)")
    print(f"Min ALT: {df_clean['alt_corrected'].min():.1f} cm")
    print(f"Max ALT: {df_clean['alt_corrected'].max():.1f} cm")
    print(f"Mean ALT: {df_clean['alt_corrected'].mean():.1f} cm")
    print(f"Median ALT: {df_clean['alt_corrected'].median():.1f} cm")
    return df_clean

def analyze_site_trends(df, alt_column='alt_corrected', time_column='datetime'):
    df[time_column] = pd.to_datetime(df[time_column])
    df['year'] = df[time_column].dt.year
    annual_means = df.groupby(['site_id', 'year'])[alt_column].agg(
        mean_alt=np.mean,
        min_alt=np.min,
        max_alt=np.max,
        std_alt=np.std,
        count=len
    ).reset_index()
    trend_analysis = []
    for site in annual_means['site_id'].unique():
        site_data = annual_means[annual_means['site_id'] == site].sort_values('year')
        if len(site_data) < 5:
            continue
        years = site_data['year'] - site_data['year'].min()
        alt_values = site_data['mean_alt']
        if len(years) > 1 and len(alt_values) > 1:
            slope, intercept = np.polyfit(years, alt_values, 1)
            alt_pred = slope * years + intercept
            ss_total = np.sum((alt_values - np.mean(alt_values))**2)
            ss_residual = np.sum((alt_values - alt_pred)**2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            trend_analysis.append({
                'site_id': site,
                'n_years': len(site_data),
                'start_year': site_data['year'].min(),
                'end_year': site_data['year'].max(),
                'mean_alt': site_data['mean_alt'].mean(),
                'trend_cm_per_year': slope,
                'r_squared': r_squared,
                'p_value': None
            })
    trend_df = pd.DataFrame(trend_analysis)
    if not trend_df.empty:
        trend_df = trend_df.sort_values('trend_cm_per_year', ascending=False)
    return trend_df

# Archived
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os

class ALTDataValidator:
    def __init__(self):
        self.alt_thresholds = {
            'min': 5,
            'max': 300,
            'year_min': 1990,
            'year_max': 2024
        }
        
    def validate_measurements(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)
        valid_alt_mask = (df['alt'] >= self.alt_thresholds['min']) & \
                        (df['alt'] <= self.alt_thresholds['max'])
        valid_date_mask = (df['datetime'].dt.year >= self.alt_thresholds['year_min']) & \
                         (df['datetime'].dt.year <= self.alt_thresholds['year_max'])
        df_validated = df[valid_alt_mask & valid_date_mask].copy()
        df_validated['suspicious'] = False
        for site in df_validated['site_id'].unique():
            site_data = df_validated[df_validated['site_id'] == site].sort_values('datetime')
            if len(site_data) > 1:
                alt_changes = site_data['alt'].diff().abs()
                suspicious_changes = alt_changes > 50
                df_validated.loc[site_data[suspicious_changes].index, 'suspicious'] = True
        return df_validated, {
            'initial_count': initial_count,
            'final_count': len(df_validated),
            'invalid_alt': (~valid_alt_mask).sum(),
            'invalid_dates': (~valid_date_mask).sum(),
            'suspicious_count': df_validated['suspicious'].sum()
        }

def process_all_sites():
    print("Initializing ALT data processing...")
    start_time = time.time()
    parser = ALTDataParser('/Users/[USER]/Downloads/calm/')
    print("\nProcessing all R-series sites...")
    master_df = parser.process_r_series(start_r=11, end_r=61)
    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = Path('/Users/[USER]/Downloads/calm/processed/')
    output_dir.mkdir(exist_ok=True)
    master_output = output_dir / f'CALM_ALT_master_{timestamp}.csv'
    master_df.to_csv(master_output, index=False)
    print(f"\nMaster dataset saved to: {master_output}")
    site_stats = master_df.groupby('site_id').agg({
        'alt': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'datetime': ['min', 'max']
    }).round(2)
    site_stats.columns = ['count', 'mean', 'std', 'min', 'max', 'median', 'first_date', 'last_date']
    site_stats = site_stats.reset_index()
    stats_output = output_dir / f'CALM_ALT_summary_{timestamp}.csv'
    site_stats.to_csv(stats_output, index=False)
    print(f"Site statistics saved to: {stats_output}")
    yearly_summary = master_df.groupby(['site_id', master_df['datetime'].dt.year]).agg({
        'alt': ['count', 'mean', 'median', 'std', 'min', 'max']
    }).round(2)
    yearly_summary.columns = ['count', 'mean', 'std', 'min', 'max', 'median']
    yearly_summary = yearly_summary.reset_index()
    yearly_output = output_dir / f'CALM_ALT_yearly_{timestamp}.csv'
    yearly_summary.to_csv(yearly_output, index=False)
    print(f"Yearly summary saved to: {yearly_output}")
    print("\nSummary Statistics")
    print("=" * 50)
    print(f"Total sites: {len(site_stats)}")
    print(f"Total measurements: {site_stats['count'].sum():,}")
    print(f"Date range: {site_stats['first_date'].min()} to {site_stats['last_date'].max()}")
    print("\nTop 10 sites by measurement count:")
    print(site_stats.nlargest(10, 'count')[['site_id', 'count']])
    print(f"\nProcessing completed in {(time.time() - start_time):.1f} seconds")
    return master_df, site_stats, yearly_summary

def process_r40():
    df = pd.read_excel('/Users/[USER]/Downloads/calm/alt/R40_Igarka_alt_2008_2023.xlsx', sheet_name='data')
    dates_raw = [''''''', 
                 ''''''', 
                 '''''']
    dates = [datetime.strptime(date, '%m/%d/%y') for date in dates_raw]
    date_columns = df.columns[6:]
    alt_data = []
    for _, row in df.iloc[:-5].iterrows():
        for date, value in zip(dates, row[date_columns]):
            if pd.notna(value) and not isinstance(value, str):
                alt_data.append({'datetime': date, 'site_id': 'R40', 'alt': value})
    df_alt = pd.DataFrame(alt_data)
    df_alt = df_alt.sort_values('datetime').reset_index(drop=True)
    df_alt.to_csv('/Users/[USER]/Downloads/R40_alt_data.csv', index=False)
    return df_alt

def process_r38b():
    df = pd.read_excel('/Users/[USER]/Downloads/calm/alt/R38b_(Burn)_ALT_2003_2017.xls', sheet_name='data')
    df = df.iloc[4:-6,1:]
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.reset_index(drop=True)
    df = df.melt()
    df.columns=['datetime','alt']
    df['site_id']='R38B'
    df.alt=df.alt/100
    df.datetime = pd.to_datetime(df.datetime)
    df = df.sort_values('datetime').reset_index(drop=True)
    df.to_csv('R38B_alt_data.csv', index=False)
    return df

def process_r57():
    df = pd.read_csv('/Users/[USER]/Downloads/r57.csv')
    df=df.melt()
    df.columns=['datetime','alt']
    df['site_id']='R57'
    df.alt=df.alt/100
    df.datetime = pd.to_datetime(df.datetime)
    df = df.sort_values('datetime').reset_index(drop=True)
    df.to_csv('R57_alt_data.csv', index=False)
    return df

def process_r56():
    df = pd.read_csv('/Users/[USER]/Downloads/r56.csv')
    df=df.melt()
    df.columns=['datetime','alt']
    df['site_id']='R56'
    df.alt=df.alt/100
    df.datetime = pd.to_datetime(df.datetime)
    df = df.sort_values('datetime').reset_index(drop=True)
    df.to_csv('R56_alt_data.csv', index=False)
    return df

def update_dataframes(alt_data, calm, latlon):
    site_name_dict = {}
    for site_id in latlon['site_id']:
        if site_id.startswith('R'):
            base_id = site_id[:3]
            site_name_dict[base_id] = latlon.loc[latlon['site_id'] == site_id, 'site_name'].iloc[0]
        else:
            site_name_dict[site_id] = latlon.loc[latlon['site_id'] == site_id, 'site_name'].iloc[0]
    for site_id in calm['site_id'].unique():
        base_id = site_id[:3] if site_id.startswith('R') else site_id
        if base_id in site_name_dict:
            mask = calm['site_id'] == site_id
            calm.loc[mask, 'site_name'] = site_name_dict[base_id]
    alt_data['site_name'] = None
    for site_id in alt_data['site_id'].unique():
        base_id = site_id[:3] if site_id.startswith('R') else site_id
        if base_id in site_name_dict:
            mask = alt_data['site_id'] == site_id
            alt_data.loc[mask, 'site_name'] = site_name_dict[base_id]
    coord_dict = {}
    for site_id in latlon['site_id']:
        if site_id.startswith('R'):
            base_id = site_id[:3]
            coord_dict[base_id] = {
                'latitude': latlon.loc[latlon['site_id'] == site_id, 'latitude'].iloc[0],
                'longitude': latlon.loc[latlon['site_id'] == site_id, 'longitude'].iloc[0]
            }
        else:
            coord_dict[site_id] = {
                'latitude': latlon.loc[latlon['site_id'] == site_id, 'latitude'].iloc[0],
                'longitude': latlon.loc[latlon['site_id'] == site_id, 'longitude'].iloc[0]
            }
    for site_id in alt_data['site_id'].unique():
        base_id = site_id[:3] if site_id.startswith('R') else site_id
        if base_id in coord_dict:
            mask = alt_data['site_id'] == site_id
            alt_data.loc[mask, 'latitude'] = coord_dict[base_id]['latitude']
            alt_data.loc[mask, 'longitude'] = coord_dict[base_id]['longitude']
    alt_data['datetime'] = pd.to_datetime(alt_data['datetime'])
    return alt_data, calm

def analyze_single_month(calm, alt, combined, year=2000, month=1):
    for df in [calm, alt, combined]:
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])
    date_filter = lambda df: df[(df['datetime'].dt.year == year) & (df['datetime'].dt.month == month)]
    calm_month = date_filter(calm)
    alt_month = date_filter(alt)
    combined_month = date_filter(combined)
    print(f"\nAnalyzing data for {year}-{month:02d}")
    print(f"CALM records: {len(calm_month)}")
    print(f"ALT records: {len(alt_month)}")
    print(f"Combined records: {len(combined_month)}")
    def check_overlaps(df1, df2, name1, name2):
        merge_keys = ['datetime', 'site_id', 'latitude', 'longitude']
        merged = pd.merge(df1[merge_keys], df2[merge_keys], on=merge_keys, how='inner')
        print(f"\nExact matches between {name1} and {name2}: {len(merged)}")
        if len(merged) > 0:
            print("\nSample overlapping records:")
            print(merged.head())
    check_overlaps(calm_month, alt_month, 'CALM', 'ALT')
    check_overlaps(calm_month, combined_month, 'CALM', 'Combined')
    check_overlaps(alt_month, combined_month, 'ALT', 'Combined')
    return calm_month, alt_month, combined_month

def clean_alt_data(df, alt_column='alt', min_valid_alt=0, max_valid_alt=500):
    df_clean = df.copy()
    df_clean['alt_quality'] = 'valid'
    total_records = len(df_clean)
    print(f"Total records before cleaning: {total_records}")
    negative_mask = df_clean[alt_column] < min_valid_alt
    df_clean.loc[negative_mask, 'alt_quality'] = 'invalid_negative'
    print(f"Negative values: {negative_mask.sum()} ({negative_mask.sum()/total_records:.2%})")
    likely_mm_mask = (df_clean[alt_column] > 1000) & (df_clean[alt_column] <= 5000)
    df_clean.loc[likely_mm_mask, 'alt_quality'] = 'likely_mm'
    print(f"Likely mm values (>1000 cm): {likely_mm_mask.sum()} ({likely_mm_mask.sum()/total_records:.2%})")
    extreme_mask = df_clean[alt_column] > max_valid_alt
    df_clean.loc[extreme_mask, 'alt_quality'] = 'invalid_extreme'
    print(f"Extreme values (>500 cm): {extreme_mask.sum()} ({extreme_mask.sum()/total_records:.2%})")
    for site in df_clean['site_id'].unique():
        site_mask = df_clean['site_id'] == site
        site_data = df_clean.loc[site_mask, alt_column]
        if len(site_data) < 5:
            continue
        Q1 = site_data.quantile(0.25)
        Q3 = site_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - 3 * IQR)
        upper_bound = Q3 + 3 * IQR
        outlier_mask = (site_mask & ((df_clean[alt_column] < lower_bound) | (df_clean[alt_column] > upper_bound)) & (df_clean['alt_quality'] == 'valid'))
        df_clean.loc[outlier_mask, 'alt_quality'] = 'statistical_outlier'
        print(f"Site {site} statistical outliers: {outlier_mask.sum()} ({outlier_mask.sum()/site_mask.sum():.2%})")
    df_clean['alt_corrected'] = df_clean[alt_column].copy()
    df_clean.loc[df_clean['alt_quality'] == 'likely_mm', 'alt_corrected'] = df_clean.loc[df_clean['alt_quality'] == 'likely_mm', alt_column] / 10
    invalid_mask = df_clean['alt_quality'].isin(['invalid_negative', 'invalid_extreme'])
    df_clean.loc[invalid_mask, 'alt_corrected'] = np.nan
    valid_records = df_clean['alt_corrected'].notna().sum()
    print(f"\nRecords after cleaning: {valid_records} ({valid_records/total_records:.2%} of original)")
    print(f"Min ALT: {df_clean['alt_corrected'].min():.1f} cm")
    print(f"Max ALT: {df_clean['alt_corrected'].max():.1f} cm")
    print(f"Mean ALT: {df_clean['alt_corrected'].mean():.1f} cm")
    print(f"Median ALT: {df_clean['alt_corrected'].median():.1f} cm")
    return df_clean

def analyze_site_trends(df, alt_column='alt_corrected', time_column='datetime'):
    df[time_column] = pd.to_datetime(df[time_column])
    df['year'] = df[time_column].dt.year
    annual_means = df.groupby(['site_id', 'year'])[alt_column].agg(
        mean_alt=np.mean, min_alt=np.min, max_alt=np.max, std_alt=np.std, count=len
    ).reset_index()
    trend_analysis = []
    for site in annual_means['site_id'].unique():
        site_data = annual_means[annual_means['site_id'] == site].sort_values('year')
        if len(site_data) < 5:
            continue
        years = site_data['year'] - site_data['year'].min()
        alt_values = site_data['mean_alt']
        if len(years) > 1 and len(alt_values) > 1:
            slope, intercept = np.polyfit(years, alt_values, 1)
            alt_pred = slope * years + intercept
            ss_total = np.sum((alt_values - np.mean(alt_values))**2)
            ss_residual = np.sum((alt_values - alt_pred)**2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            trend_analysis.append({
                'site_id': site, 'n_years': len(site_data), 'start_year': site_data['year'].min(),
                'end_year': site_data['year'].max(), 'mean_alt': site_data['mean_alt'].mean(),
                'trend_cm_per_year': slope, 'r_squared': r_squared, 'p_value': None
            })
    trend_df = pd.DataFrame(trend_analysis)
    if not trend_df.empty:
        trend_df = trend_df.sort_values('trend_cm_per_year', ascending=False)
    return trend_df

def has_measurements(row):
    alt_columns = ['alt', 'alt_m', 'thickness']
    temp_columns = ['soil_temp', 'temperature']
    moisture_columns = ['soil_moist', 'vwc']
    has_alt = any(col in row.index and pd.notna(row[col]) for col in alt_columns)
    has_temp = any(col in row.index and pd.notna(row[col]) for col in temp_columns)
    has_moisture = any(col in row.index and pd.notna(row[col]) for col in moisture_columns)
    return has_alt or has_temp or has_moisture

def process_chunks_efficiently(calm, alt, combined, output_dir='chunk_files', days_per_chunk=30):
    print("Starting chunked processing...")
    os.makedirs(output_dir, exist_ok=True)
    print("Preparing datasets...")
    print(f"Processing CALM ({len(calm)} rows)")
    calm_copy = calm.copy()
    if not pd.api.types.is_datetime64_any_dtype(calm_copy['datetime']):
        calm_copy['datetime'] = pd.to_datetime(calm_copy['datetime'])
    if 'site_id' in calm_copy.columns:
        calm_copy['site_id'] = calm_copy['site_id'].astype(str)
    print(f"Processing ALT ({len(alt)} rows)")
    alt_copy = alt.copy()
    if not pd.api.types.is_datetime64_any_dtype(alt_copy['datetime']):
        alt_copy['datetime'] = pd.to_datetime(alt_copy['datetime'])
    if 'site_id' in alt_copy.columns:
        alt_copy['site_id'] = alt_copy['site_id'].astype(str)
    print(f"Processing Combined ({len(combined)} rows)")
    combined_copy = combined.copy()
    if not pd.api.types.is_datetime64_any_dtype(combined_copy['datetime']):
        combined_copy['datetime'] = pd.to_datetime(combined_copy['datetime'])
    if 'site_id' in combined_copy.columns:
        combined_copy['site_id'] = combined_copy['site_id'].astype(str)
    min_date = min(df['datetime'].min() for df in [calm_copy, alt_copy, combined_copy])
    max_date = max(df['datetime'].max() for df in [calm_copy, alt_copy, combined_copy])
    print(f"Processing data from {min_date.date()} to {max_date.date()}")
    date_chunks = pd.date_range(start=min_date, end=max_date, freq=f'{days_per_chunk}D')
    chunk_files = []
    total_rows_kept = 0
    for i in range(len(date_chunks)-1):
        start_date = date_chunks[i]
        end_date = date_chunks[i+1]
        print(f"\nProcessing chunk {i+1}/{len(date_chunks)-1}: {start_date.date()} to {end_date.date()}")
        chunk_filter = lambda df: df[(df['datetime'] >= start_date) & (df['datetime'] < end_date)]
        calm_chunk = chunk_filter(calm_copy)
        alt_chunk = chunk_filter(alt_copy)
        combined_chunk = chunk_filter(combined_copy)
        calm_chunk['data_source'] = 'CALM'
        alt_chunk['data_source'] = 'ALT'
        combined_chunk['data_source'] = 'Combined'
        calm_chunk_filtered = calm_chunk[calm_chunk.apply(has_measurements, axis=1)]
        alt_chunk_filtered = alt_chunk[alt_chunk.apply(has_measurements, axis=1)]
        combined_chunk_filtered = combined_chunk[combined_chunk.apply(has_measurements, axis=1)]
        total_size = len(calm_chunk_filtered) + len(alt_chunk_filtered) + len(combined_chunk_filtered)
        if total_size > 0:
            print(f"Found {total_size} rows with measurements")
            merged_chunk = pd.concat([calm_chunk_filtered, alt_chunk_filtered, combined_chunk_filtered], axis=0)
            latitude_filter = merged_chunk['latitude'] >= 49
            date_filter = merged_chunk['datetime'] < pd.Timestamp('2025-01-01')
            filtered_chunk = merged_chunk[latitude_filter & date_filter]
            post_filter_size = len(filtered_chunk)
            print(f"After latitude/date filtering: {post_filter_size} rows")
            if post_filter_size > 0:
                chunk_file = os.path.join(output_dir, f'chunk_{i+1:04d}.csv')
                filtered_chunk.to_csv(chunk_file, index=False)
                chunk_files.append(chunk_file)
                total_rows_kept += post_filter_size
                print(f"Written {post_filter_size} rows to {chunk_file}")
            del merged_chunk, filtered_chunk
            gc.collect()
    print(f"\nTotal rows saved across all chunks: {total_rows_kept}")
    return chunk_files

def combine_chunks(chunk_files, output_file='merged_permafrost.csv'):
    print("\nCombining chunks into final file...")
    if not chunk_files:
        print("No chunks to combine!")
        return pd.DataFrame()
    first_chunk = pd.read_csv(chunk_files[0], nrows=0)
    first_chunk.to_csv(output_file, index=False)
    total_rows = 0
    for chunk_file in chunk_files:
        chunk = pd.read_csv(chunk_file)
        total_rows += len(chunk)
        chunk.to_csv(output_file, mode='a', header=False, index=False)
        print(f"Added {len(chunk)} rows from {chunk_file}")
    print(f"\nFinal dataset statistics:")
    print(f"Total rows: {total_rows:,}")
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    os.rmdir(output_dir)
    return pd.read_csv(output_file, nrows=1000)

if __name__ == "__main__":
    print("Initializing parser...")
    parser = ALTDataParser('/Users/[USER]/Downloads/calm/alt')
    print("Processing files...")
    alt_data = parser.process_all_files()
    alt_data.alt = alt_data.alt/100
    alt_data = alt_data[alt_data.alt < 7]
    alt_data.datetime = pd.to_datetime(alt_data.datetime)
    alt_data = alt_data.sort_values('datetime').reset_index(drop=True)
    output_path = Path('/Users/[USER]/Downloads/calm/processed/calm_alt_parsed_all_files.csv')
    alt_data.to_csv(output_path, index=False)
    print("\nData Quality Checks")
    print("=" * 50)
    print(f"Total rows in final dataset: {len(alt_data)}")
    print(f"Unique sites: {sorted(alt_data['site_id'].unique())}")
    print("\nDate range:")
    print(f"Earliest: {alt_data['datetime'].min()}")
    print(f"Latest: {alt_data['datetime'].max()}")
    print("\nMeasurements by site:")
    site_counts = alt_data.groupby('site_id').size().sort_values(ascending=False)
    for site, count in site_counts.items():
        print(f"{site}: {count:,}")
    print(f"\nProcessed data saved to: {output_path}")

# ST
merged_df = pd.read_feather('merged_compressed.feather')
merged_df_st = merged_df[merged_df.soil_temp_standardized.isna()!=True].sort_values('datetime').reset_index(drop=True)

def analyze_temperature_stats(df):
    """Analyze temperature statistics to determine appropriate colorbar bounds"""
    # Calculate site means and standard deviations
    site_stats = df.groupby(['latitude', 'longitude'])['soil_temp_standardized'].agg(['mean', 'std']).reset_index()
    
    print("Temperature Distribution Statistics:")
    print("\nMean Temperature:")
    print(site_stats['mean'].describe())
    print("\nStandard Deviation:")
    print(site_stats['std'].describe())
    
    # Calculate robust bounds using percentiles
    mean_bounds = (
        site_stats['mean'].quantile(0.05),
        site_stats['mean'].quantile(0.95)
    )
    std_bounds = (
        site_stats['std'].quantile(0.05),
        site_stats['std'].quantile(0.95)
    )
    
    return mean_bounds, std_bounds

# valid_data = np.ma.masked_invalid(merged_df_st.soil_temp_standardized).compressed()

# print("Absolute min:", valid_data.min())
# print("Absolute max:", valid_data.max())

# p2 = np.percentile(valid_data, 2)
# p98 = np.percentile(valid_data, 98)

# print("2nd percentile (vmin):", p2)
# print("98th percentile (vmax):", p98)

def create_arctic_temp_map(df, value_column, title, vmin=None, vmax=None, std_dev=False, cmap=None):
    """
    Create a Circumarctic map with enhanced soil temperature distribution visualization
    """
    # Calculate site-specific statistics based on mode
    if std_dev:
        # Calculate site-specific standard deviation
        stats = df.groupby(['latitude', 'longitude'])[value_column].std().reset_index()
        plot_value = 'std'
        stats = stats.rename(columns={value_column: plot_value})
        cbar_label = 'Standard Deviation, Standardized Soil Temperature (°C)'
        
        if cmap is None:
            cmap = 'viridis'
            
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = stats[plot_value].quantile(0.95)
    else:
        # Calculate site-specific means
        stats = df.groupby(['latitude', 'longitude'])[value_column].mean().reset_index()
        plot_value = 'mean'
        stats = stats.rename(columns={value_column: plot_value})
        cbar_label = 'Mean, Standardized Soil Temperature (°C)'
        
        if cmap is None:
            cmap = 'cmr.guppy'
            
        if vmin is None:
            vmin = stats[plot_value].quantile(0.05)
        if vmax is None:
            vmax = stats[plot_value].quantile(0.95)
    
    # Print statistics for reference
    print(f"\nValue distribution for {plot_value}:")
    print(stats[plot_value].describe())
    print(f"Value range used for visualization: {vmin:.3f} to {vmax:.3f}")
    
    # Create figure
    fig = plt.figure(figsize=(12, 12))
    
    # Create map axes with specific dimensions 
    # (left, bottom, width, height) - all values as fractions of figure
    map_ax = fig.add_axes([0.05, 0.1, 0.9, 0.95], projection=ccrs.NorthPolarStereo())
    map_ax.set_extent([-180, 180, 48.5, 90], ccrs.PlateCarree())
    
    # Add map features with white background (no ocean)
    map_ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.5)
    map_ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
    map_ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    map_ax.gridlines()
    
    # Create scatter plot
    scatter = map_ax.scatter(
        stats['longitude'], 
        stats['latitude'],
        c=stats[plot_value],
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        s=50,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Create colorbar with EXACTLY the same width as the map
    cbar_ax = fig.add_axes([0.05, 0.075, 0.9, 0.02])
    cbar = plt.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(cbar_label, size=12, labelpad=10)
    
    # Add title with padding
    map_ax.set_title(title, pad=25, size=16)
    
    return fig

# Create maps with data-driven bounds
def create_maps(merged_df_st):
    """Generate both temperature maps with data-driven bounds"""
    # Calculate statistics for proper bounds
    mean_stats = merged_df_st.groupby(['latitude', 'longitude'])['soil_temp_standardized'].mean()
    std_stats = merged_df_st.groupby(['latitude', 'longitude'])['soil_temp_standardized'].std()
    
    # For mean temperature map - use 5th to 95th percentile for better visualization
    mean_vmin = mean_stats.quantile(0.05)
    mean_vmax = mean_stats.quantile(0.95)
    
    # For standard deviation map - start at 0 with 95th percentile upper bound
    std_vmin = 0.0
    std_vmax = std_stats.quantile(0.95)
    
    # Create maps with calculated bounds
    fig1 = create_arctic_temp_map(
        merged_df_st, 
        'soil_temp_standardized', 
        'Circumarctic Standardized Soil Temperature, In Situ \n 1891-01-15 to 2023-08-01',
        vmin=mean_vmin,
        vmax=mean_vmax,
        cmap='cmr.guppy'
    )
    
    fig2 = create_arctic_temp_map(
        merged_df_st,
        'soil_temp_standardized',
        'Circumarctic Standardized Soil Temperature Variability, In Situ \n 1891-01-15 to 2023-08-01',
        vmin=std_vmin,
        vmax=std_vmax,
        std_dev=True,
        cmap='viridis'
    )
    
    return fig1, fig2

fig1, fig2 = create_maps(merged_df_st)
plt.show()

fig1.savefig("meansoiltemperature_insitu_zerocurtain.png", bbox_inches='tight', pad_inches=0.5, dpi=1000)
fig2.savefig("stdsoiltemperature_insitu_zerocurtain.png", bbox_inches='tight', pad_inches=0.5, dpi=1000)


# import cmasher as cmr

# def create_arctic_temp_map(df, value_column, title, vbounds=None, std_dev=False):
#     """
# Create a Circumarctic map with enhanced soil...
#     """
#     # More sophisticated grouping that considers site-specific characteristics
#     # if std_dev:
#     #     # Calculate site-specific standard deviation
#     #     #stats = (df.groupby(['latitude', 'longitude'])
#     #     #        .agg({value_column: ['std', 'count']})
#     #     #        .reset_index())
#     #     #stats.columns = ['latitude', 'longitude', 'std', 'count']
#     #     stats = df.groupby(['latitude', 'longitude'])[value_column].std().reset_index()
#     #     # Filter for sites with sufficient measurements
# # #stats = stats[stats['count'] >= 10] #...
#     #     plot_value = value_column
#     #     cbar_label = 'Standard Deviation, Standardized Soil Temperature (°C)'
#     #     #vmin, vmax = 0, stats[value_column].quantile(0.95)
#     #     vmin, vmax = 0.0, 10.0
#     #     cmap = 'viridis'
#     # else:
#     #     # Calculate site-specific means
#     #     #stats = (df.groupby(['latitude', 'longitude'])
#     #     #        .agg({value_column: ['mean', 'count']})
#     #     #        .reset_index())
#     #     #stats.columns = ['latitude', 'longitude', 'mean', 'count']
#     #     stats = df.groupby(['latitude', 'longitude'])[value_column].mean().reset_index()
#     #     # Filter for sites with sufficient measurements
#     #     #stats = stats[stats['count'] >= 10]
#     #     plot_value = value_column
#     #     cbar_label = 'Mean, Standardized Soil Temperature (°C)'
#     #     #vmin, vmax = stats[value_column].quantile(0.05), stats[value_column].quantile(0.95)
#     #     vmin, vmax = 0.0, 7.0
#     #     cmap = 'cmr.guppy'

#     import cmasher as cmr
#     cmap = cmr.guppy

#     if std_dev:
# vmin, vmax = vbounds if vbounds is...
#         cmap = 'viridis'
#         cbar_label = 'Standard Deviation, Standardized Soil Temperature (°C)'
#     else:
# vmin, vmax = vbounds if vbounds is...
#         cmap = 'cmr.guppy'
#         cbar_label = 'Mean, Standardized Soil Temperature (°C)'

#     if std_dev:
#         cmap = cmr.ember  # you can choose e.g., cmr.ember or 'viridis'
#         cbar_label = 'Standard Deviation, Standardized Soil Temperature (°C)'
#     else:
#         cmap = cmr.guppy
#         cbar_label = 'Mean, Standardized Soil Temperature (°C)'
    
#     # Calculate fallback bounds from 5–95th percentiles
#     fallback_vmin = stats[plot_value].quantile(0.05)
#     fallback_vmax = stats[plot_value].quantile(0.95)
    
#     # Use vbounds if provided, otherwise fallback to quantiles
# vmin, vmax = vbounds if vbounds is...

#     #print(f"\nValue distribution for {plot_value}:")
#     #print(stats[plot_value].describe())
    
#     fig = plt.figure(figsize=(12, 12), constrained_layout=True)
#     gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.05])
#     ax = fig.add_subplot(gs[0], projection=ccrs.NorthPolarStereo())
#     ax.set_extent([-180, 180, 45, 90], ccrs.PlateCarree())
    
#     # Enhanced map features
#     ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.5)
#     #ax.add_feature(cfeature.OCEAN, alpha=0.5)
#     ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
#     ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
#     ax.gridlines()
    
#     # Create scatter plot with size variation based on measurement count
#     # Determine color scale bounds
#     if vbounds is not None:
#         vmin, vmax = vbounds
#     else:
#         vmin = stats[plot_value].quantile(0.02)
#         vmax = stats[plot_value].quantile(0.98)
#     scatter = ax.scatter(stats['longitude'], 
#                         stats['latitude'],
#                         c=stats[plot_value],
#     #scatter = ax.scatter(stats['longitude'], stats['latitude'],
#                         #c=stats[value_column],
#                         cmap=cmap,
#                         transform=ccrs.PlateCarree(),
#                         vmin=vmin,
#                         vmax=vmax,
#                         s=50,#stats['count'].clip(50, 200),  # Size varies with measurement count
#                         alpha=0.7)#,
#                         #edgecolor='black',
#                         #linewidth=0.5)
    
#     # Enhanced colorbar
#     cax = fig.add_subplot(gs[1])
#     cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')
#     cbar.set_label(cbar_label, size=12)
    
#     # Add measurement depth annotation if available
#     #if not std_dev and 'depth' in stats.columns:
#     #    depths = stats['depth'].unique()
#     #    ax.text(0.02, 0.98, f'Measurement depths (m): {", ".join(map(str, sorted(depths)))}',
#     #            transform=ax.transAxes, fontsize=8, verticalalignment='top')
    
#     ax.set_title(title, pad=25, size=16)

#     #print(f"Value range: {vmin:.3f} to {vmax:.3f}")

#     print(f"\nValue distribution for {plot_value}:")
#     print(stats[plot_value].describe())
#     print(f"Value range: {vmin:.3f} to {vmax:.3f}")
    
#     return fig

# ALT
merged_df = pd.read_feather('merged_compressed.feather')
merged_df_alt = merged_df[merged_df.thickness_m_standardized.isna() != True].sort_values('datetime').reset_index(drop=True)
print(f"Number of ALT measurements: {len(merged_df_alt)}")
print(merged_df_alt.thickness_m_standardized.describe())

def analyze_alt_stats(df):
    """Analyze alt statistics to determine appropriate colorbar bounds"""
    # Calculate site means and standard deviations
    site_stats = df.groupby(['latitude', 'longitude'])['thickness_m_standardized'].agg(['mean', 'std']).reset_index()
    
    print("ALT Distribution Statistics:")
    print("\nMean Standardized ALT:")
    print(site_stats['mean'].describe())
    print("\nStandard Deviation:")
    print(site_stats['std'].describe())
    
    # Calculate robust bounds using percentiles
    mean_bounds = (
        site_stats['mean'].quantile(0.05),
        site_stats['mean'].quantile(0.95)
    )
    std_bounds = (
        site_stats['std'].quantile(0.05),
        site_stats['std'].quantile(0.95)
    )
    
    return mean_bounds, std_bounds

mean_bounds, std_bounds = analyze_alt_stats(merged_df_alt[~merged_df_alt.thickness_m_standardized.isna()==True])

def create_arctic_alt_map(df, value_column, title, vmin=None, vmax=None, std_dev=False, cmap=None):
    """
    Create a Circumarctic map with enhanced ALT distribution visualization
    """
    # Calculate site-specific statistics based on mode
    if std_dev:
        # Calculate site-specific standard deviation
        stats = df.groupby(['latitude', 'longitude'])[value_column].std().reset_index()
        plot_value = 'std'
        stats = stats.rename(columns={value_column: plot_value})

        # Debug: Print out some stats information
        print("Number of sites with std data:", len(stats))
        print("Min std value:", stats[plot_value].min())
        print("Max std value:", stats[plot_value].max())
        
        cbar_label = 'Standard Deviation, Standardized Active Layer Thickness (m)'
        
        if cmap is None:
            cmap = 'viridis'
            
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = stats[plot_value].quantile(0.95)

        # Filter out points with zero or NaN standard deviation
        stats = stats[stats[plot_value] > 0].dropna(subset=[plot_value])
        print("After filtering, number of sites:", len(stats))
        
    else:
        # Calculate site-specific means
        stats = df.groupby(['latitude', 'longitude'])[value_column].mean().reset_index()
        plot_value = 'mean'
        stats = stats.rename(columns={value_column: plot_value})
        cbar_label = 'Mean, Standardized Active Layer Thickness (m)'
        
        if cmap is None:
            cmap = 'cmr.guppy'
            
        if vmin is None:
            vmin = stats[plot_value].quantile(0.05)
        if vmax is None:
            vmax = stats[plot_value].quantile(0.95)
    
    # Print statistics for reference
    print(f"\nValue distribution for {plot_value}:")
    print(stats[plot_value].describe())
    print(f"Value range used for visualization: {vmin:.3f} to {vmax:.3f}")
    
    # Create figure
    fig = plt.figure(figsize=(12, 12))
    
    # Create map axes with specific dimensions 
    # (left, bottom, width, height) - all values as fractions of figure
    map_ax = fig.add_axes([0.05, 0.1, 0.9, 0.95], projection=ccrs.NorthPolarStereo())
    map_ax.set_extent([-180, 180, 48.5, 90], ccrs.PlateCarree())
    
    # Add map features with white background (no ocean)
    map_ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.5)
    map_ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
    map_ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    map_ax.gridlines()
    
    # Create scatter plot
    scatter = map_ax.scatter(
        stats['longitude'], 
        stats['latitude'],
        c=stats[plot_value],
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        s=50,
        alpha=0.8,
        edgecolor='gray',
        linewidth=0.5
    )
    
    # Create colorbar with EXACTLY the same width as the map
    cbar_ax = fig.add_axes([0.05, 0.075, 0.9, 0.02])
    cbar = plt.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(cbar_label, size=12, labelpad=10)
    
    # Add title with padding
    map_ax.set_title(title, pad=25, size=16)
    
    return fig

def create_maps(merged_df_alt):
    """Generate both ALT maps with data-driven bounds"""
    # Calculate statistics for proper bounds
    mean_stats = merged_df_alt.groupby(['latitude', 'longitude'])['thickness_m_standardized'].mean()
    std_stats = merged_df_alt.groupby(['latitude', 'longitude'])['thickness_m_standardized'].std()
    
    # For mean temperature map - use 5th to 95th percentile for better visualization
    mean_vmin = mean_stats.quantile(0.05)
    mean_vmax = mean_stats.quantile(0.95)
    
    # For standard deviation map - start at 0 with 95th percentile upper bound
    std_vmin = 0.0
    std_vmax = std_stats.quantile(0.95)
    
    # Create maps with calculated bounds
    fig1 = create_arctic_alt_map(
        merged_df_alt, 
        'thickness_m_standardized', 
        'Circumarctic Standardized Active Layer Thickness, In Situ \n 1899-05-15 to 2024-08-31',
        vmin=mean_vmin,
        vmax=mean_vmax,
        cmap='cmr.guppy'
    )
    
    fig2 = create_arctic_alt_map(
        merged_df_alt,
        'thickness_m_standardized',
        'Circumarctic Standardized Active Layer Thickness Variability, In Situ \n 1899-05-15 to 2024-08-31',
        vmin=std_vmin,
        vmax=std_vmax,
        std_dev=True,
        cmap='viridis'
    )
    
    return fig1, fig2

fig1, fig2 = create_maps(merged_df_alt)
plt.show()

fig1.savefig("meanalt_insitu_zerocurtain.png", bbox_inches='tight', pad_inches=0.5, dpi=1000)
fig2.savefig("stdalt_insitu_zerocurtain.png", bbox_inches='tight', pad_inches=0.5, dpi=1000)

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import cmasher as cmr
# import cmocean as cmo

# def create_arctic_thickness_map(df, value_column, title, vbounds=None, std_dev=False):
#     """
#     Create a Circumarctic map with controlled colorbar bounds
#     """
#     # Calculate statistics for each location
#     if std_dev:
#         stats = df.groupby(['latitude', 'longitude'])[value_column].std().reset_index()
#         cbar_label = 'Standard Deviation, Active Layer Thickness (m)'
#         cmap = 'viridis'
#     else:
#         stats = df.groupby(['latitude', 'longitude'])[value_column].mean().reset_index()
#         cbar_label = 'Mean, Active Layer Thickness (m)'
#         cmap = 'cmr.guppy'
    
#     fig = plt.figure(figsize=(12, 12), constrained_layout=True)
#     gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.05])
#     ax = fig.add_subplot(gs[0], projection=ccrs.NorthPolarStereo())
#     ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    
#     # Add map features
#     ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.5)
#     ax.add_feature(cfeature.OCEAN, alpha=0.5)
#     ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
#     ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
#     ax.gridlines()
    
#     # Create scatter plot with bounded colorbar
#     scatter = ax.scatter(stats['longitude'], stats['latitude'],
#                         c=stats[value_column],
#                         cmap=cmap,
#                         transform=ccrs.PlateCarree(),
#                         vmin=vbounds[0], vmax=vbounds[1],
#                         s=50, alpha=0.7)

#     cax = fig.add_subplot(gs[1])
#     cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')#, pad=0.05)
#     cbar.set_label(cbar_label, size=12)#, labelpad=0.15)

#     ax.set_title(title, pad=25, size=16)
    
#     return fig

# fig1 = create_arctic_thickness_map(thickness_clean, 'thickness', 
# 'Circumarctic Distribution of Active Layer Thickness, In...
#                            vbounds=mean_bounds)
# fig2 = create_arctic_thickness_map(thickness_clean, 'thickness',
#                            'Circumarctic Distribution of Active Layer Thickness Variability, In Si...
#                            vbounds=std_bounds,
#                            std_dev=True)
# plt.show()

# SMC
merged_df = pd.read_feather('merged_compressed.feather')
merged_df_smc = merged_df[merged_df.soil_moist_standardized.isna() != True].sort_values('datetime').reset_index(drop=True)
print(f"Number of SMC measurements: {len(merged_df_smc)}")
print(merged_df_smc.soil_moist_standardized.describe())

def analyze_smc_stats(df):
    """Analyze alt statistics to determine appropriate colorbar bounds"""
    # Calculate site means and standard deviations
    site_stats = df.groupby(['latitude', 'longitude'])['soil_moist_standardized'].agg(['mean', 'std']).reset_index()
    
    print("SMC Dynamics and Distribution Statistics:")
    print("\nMean Standardized SMC:")
    print(site_stats['mean'].describe())
    print("\nStandard Deviation:")
    print(site_stats['std'].describe())
    
    # Calculate robust bounds using percentiles
    mean_bounds = (
        site_stats['mean'].quantile(0.05),
        site_stats['mean'].quantile(0.95)
    )
    std_bounds = (
        site_stats['std'].quantile(0.05),
        site_stats['std'].quantile(0.95)
    )
    
    return mean_bounds, std_bounds

mean_bounds, std_bounds = analyze_alt_stats(merged_df_alt[~merged_df_alt.thickness_m_standardized.isna()==True])

def create_arctic_smc_map(df, value_column, title, vmin=None, vmax=None, std_dev=False, cmap=None):
    """
    Create a Circumarctic map with enhanced SMC distribution visualization
    """
    # Calculate site-specific statistics based on mode
    if std_dev:
        # Calculate site-specific standard deviation
        stats = df.groupby(['latitude', 'longitude'])[value_column].std().reset_index()
        plot_value = 'std'
        stats = stats.rename(columns={value_column: plot_value})

        # Debug: Print out some stats information
        print("Number of sites with std data:", len(stats))
        print("Min std value:", stats[plot_value].min())
        print("Max std value:", stats[plot_value].max())
        
        cbar_label = 'Standard Deviation, Standardized Soil Moisture Content (%)'
        
        if cmap is None:
            cmap = 'viridis'
            
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = stats[plot_value].quantile(0.95)

        # Filter out points with zero or NaN standard deviation
        stats = stats[stats[plot_value] > 0].dropna(subset=[plot_value])
        print("After filtering, number of sites:", len(stats))
        
    else:
        # Calculate site-specific means
        stats = df.groupby(['latitude', 'longitude'])[value_column].mean().reset_index()
        plot_value = 'mean'
        stats = stats.rename(columns={value_column: plot_value})
        cbar_label = 'Mean, Standardized Soil Moisture Content (%)'
        
        if cmap is None:
            cmap = 'cmr.guppy'
            
        if vmin is None:
            vmin = stats[plot_value].quantile(0.05)
        if vmax is None:
            vmax = stats[plot_value].quantile(0.95)
    
    # Print statistics for reference
    print(f"\nValue distribution for {plot_value}:")
    print(stats[plot_value].describe())
    print(f"Value range used for visualization: {vmin:.3f} to {vmax:.3f}")
    
    # Create figure
    fig = plt.figure(figsize=(12, 12))
    
    # Create map axes with specific dimensions 
    # (left, bottom, width, height) - all values as fractions of figure
    map_ax = fig.add_axes([0.05, 0.1, 0.9, 0.95], projection=ccrs.NorthPolarStereo())
    map_ax.set_extent([-180, 180, 48.5, 90], ccrs.PlateCarree())
    
    # Add map features with white background (no ocean)
    map_ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.5)
    map_ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
    map_ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    map_ax.gridlines()
    
    # Create scatter plot
    scatter = map_ax.scatter(
        stats['longitude'], 
        stats['latitude'],
        c=stats[plot_value],
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        s=50,
        alpha=0.8,
        edgecolor='gray',
        linewidth=0.5
    )
    
    # Create colorbar with EXACTLY the same width as the map
    cbar_ax = fig.add_axes([0.05, 0.075, 0.9, 0.02])
    cbar = plt.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(cbar_label, size=12, labelpad=10)
    
    # Add title with padding
    map_ax.set_title(title, pad=25, size=16)
    
    return fig

def create_maps(merged_df_smc):
    """Generate both SMC maps with data-driven bounds"""
    # Calculate statistics for proper bounds
    mean_stats = merged_df_smc.groupby(['latitude', 'longitude'])['soil_moist_standardized'].mean()
    std_stats = merged_df_smc.groupby(['latitude', 'longitude'])['soil_moist_standardized'].std()
    
    # For mean SMC map - use 5th to 95th percentile for better visualization
    mean_vmin = mean_stats.quantile(0.05)
    mean_vmax = mean_stats.quantile(0.95)
    
    # For standard deviation map - start at 0 with 95th percentile upper bound
    std_vmin = 0.0
    std_vmax = std_stats.quantile(0.95)
    
    # Create maps with calculated bounds
    fig1 = create_arctic_smc_map(
        merged_df_smc, 
        'soil_moist_standardized', 
        'Circumarctic Standardized Soil Moisture Content, In Situ \n 1952-06-08 to 2024-12-31',
        vmin=mean_vmin,
        vmax=mean_vmax,
        cmap='cmr.guppy'
    )
    
    fig2 = create_arctic_smc_map(
        merged_df_smc,
        'soil_moist_standardized',
        'Circumarctic Standardized Soil Moisture Content Variability, In Situ \n 1952-06-08 to 2024-12-31',
        vmin=std_vmin,
        vmax=std_vmax,
        std_dev=True,
        cmap='viridis'
    )
    
    return fig1, fig2

fig1, fig2 = create_maps(merged_df_smc)
plt.show()

fig1.savefig("meansmc_insitu_zerocurtain.png", bbox_inches='tight', pad_inches=0.5, dpi=1000)
fig2.savefig("stdsmc_insitu_zerocurtain.png", bbox_inches='tight', pad_inches=0.5, dpi=1000)

# Soil Temperature Recall
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import PowerNorm, Normalize, LogNorm
from datetime import timedelta
from tqdm.auto import tqdm
from pathlib import Path
import logging
import gc
import psutil
import pickle
from scipy import stats
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import geopandas as gpd
from matplotlib.gridspec import GridSpec
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm.keras import TqdmCallback

class SoilTempParser:
    @staticmethod
    def parse_tpgks(file_path: Path) -> pd.DataFrame:
        depths = [5, 10, 15, 20]
        data = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) < 12:
                        continue
                    try:
                        station = values[0]
                        year = int(values[1])
                        month = int(values[2])
                        day = int(values[3])
                        for i, depth in enumerate(depths):
                            temp = values[4 + i*2]
                            flag = values[5 + i*2]
                            if temp != '9999':
                                data.append({
                                    'station_id': station,
                                    'datetime': f"{year}-{month:02d}-{day:02d}",
                                    'soil_temp_depth': depth,
                                    'soil_temp': float(temp)/10,
                                    'quality_flag': flag
                                })
                    except (ValueError, IndexError):
                        continue
            if data:
                df = pd.DataFrame(data)
                df['datetime'] = pd.to_datetime(df['datetime'])
                return df
            return None
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    @staticmethod
    def parse_tpg(file_path: Path) -> pd.DataFrame:
        depths = [2, 5, 10, 15, 20, 40, 60, 80, 120, 160, 240, 320]
        data = []
        try:
            with open(file_path, 'r') as f:
                next(f)
                for line in f:
                    values = line.strip().split()
                    if len(values) < 28:
                        continue
                    try:
                        station = values[0]
                        year = int(values[1])
                        month = int(values[2])
                        day = int(values[3])
                        for i, depth in enumerate(depths):
                            temp = values[4 + i*2]
                            flag = values[5 + i*2]
                            if temp != '9999':
                                data.append({
                                    'station_id': station,
                                    'datetime': f"{year}-{month:02d}-{day:02d}",
                                    'soil_temp_depth': depth,
                                    'soil_temp': float(temp)/10,
                                    'quality_flag': flag
                                })
                    except (ValueError, IndexError):
                        continue
            if data:
                df = pd.DataFrame(data)
                df['datetime'] = pd.to_datetime(df['datetime'])
                return df
            return None
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

def process_soil_temp_data(base_path: str) -> dict:
    parser = SoilTempParser()
    results = {}
    base_path = Path(base_path)
    tpgks_dir = base_path / 'dailysoiltemperature_1984-2012' / 'Tpgks'
    if tpgks_dir.exists():
        tpgks_data = []
        for file in tpgks_dir.glob('*.*'):
            df = parser.parse_tpgks(file)
            if df is not None:
                tpgks_data.append(df)
        if tpgks_data:
            results['tpgks'] = pd.concat(tpgks_data, ignore_index=True)
    tpg_dir = base_path / 'dailysoiltemperatureatdepth_1963-2022' / 'Tpg'
    if tpg_dir.exists():
        tpg_data = []
        for file in tpg_dir.glob('*.*'):
            df = parser.parse_tpg(file)
            if df is not None:
                tpg_data.append(df)
        if tpg_data:
            results['tpg'] = pd.concat(tpg_data, ignore_index=True)
    return results

def clean_coordinates(df):
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    mask = (
        (df['latitude'].between(-90, 90)) &
        (df['longitude'].between(-180, 180))
    )
    df_clean = df[mask].copy()
    removed = len(df) - len(df_clean)
    if removed > 0:
        print(f"Removed {removed} rows with invalid coordinates")
    return df_clean

def run_sensitivity_analysis(merged_df):
    df_clean = clean_coordinates(merged_df)
    epsilon_range = [0.025, 0.05, 0.075, 0.1, 0.15, 0.20, 0.25, 0.50, 1.0, 2.0]
    duration_range = [6, 12, 24, 36, 72, 168, 730, 2190, 4380, 8760]
    results = []
    for epsilon in epsilon_range:
        for duration in duration_range:
            zc_mask = df_clean['temperature'].abs() <= epsilon
            zc_data = df_clean[zc_mask].copy()
            if len(zc_data) > 0:
                zc_data['event_group'] = (~zc_mask).cumsum()[zc_mask]
                events = zc_data.groupby(['site_id', 'event_group']).agg({
                    'datetime': ['min', 'max', 'count'],
                    'temperature': ['mean', 'std']
                }).reset_index()
                events['duration_hours'] = (
                    (events['datetime']['max'] - events['datetime']['min'])
                    .dt.total_seconds() / 3600
                )
                valid_events = events[events['duration_hours'] >= duration]
                results.append({
                    'epsilon': epsilon,
                    'min_duration': duration,
                    'events': len(valid_events),
                    'sites': valid_events['site_id'].nunique(),
                    'mean_duration': valid_events['duration_hours'].mean(),
                    'total_duration': valid_events['duration_hours'].sum()
                })
            else:
                results.append({
                    'epsilon': epsilon,
                    'min_duration': duration,
                    'events': 0,
                    'sites': 0,
                    'mean_duration': 0,
                    'total_duration': 0
                })
    sensitivity = pd.DataFrame(results)
    sensitivity['events_per_site'] = sensitivity['events'] / sensitivity['sites']
    return sensitivity, df_clean

def visualize_parameter_sensitivity(sensitivity_df, stats, output_dir='figures'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    pivot_events = sensitivity_df.pivot(
        index='min_duration',
        columns='epsilon',
        values='events'
    )
    sns.heatmap(pivot_events, cmap='viridis', annot=True, fmt='.0f', ax=axes[0])
    axes[0].set_title('Number of Zero Curtain Events')
    axes[0].set_xlabel('Temperature Threshold (°C)')
    axes[0].set_ylabel('Minimum Duration (hours)')
    sns.lineplot(
        data=sensitivity_df,
        x='epsilon',
        y='events_per_site',
        hue='min_duration',
        ax=axes[1]
    )
    axes[1].set_title('Events per Site')
    axes[1].set_xlabel('Temperature Threshold (°C)')
    axes[1].set_ylabel('Average Events per Site')
    plt.tight_layout()
    plt.savefig(output_path / 'parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    optimal = sensitivity_df[
        (sensitivity_df['events'] > 0) & 
        (sensitivity_df['events_per_site'] >= 1)
    ].sort_values('events', ascending=False).iloc[0]
    return {
        'optimal_parameters': {
            'epsilon': optimal['epsilon'],
            'min_duration': optimal['min_duration'],
            'events': optimal['events'],
            'sites': optimal['sites'],
            'mean_duration': optimal['mean_duration']
        }
    }

class EnhancedZeroCurtainAnalyzer:
    def __init__(self, epsilon=2, min_duration=6):
        self.epsilon = epsilon
        self.min_duration = min_duration
        self.column_order = [
            'datetime', 'latitude', 'longitude', 'temperature', 'depth', 'year', 
            'source', 'season', 'depth_zone', 'start_date', 'end_date', 
            'duration_hours', 'mean_temp', 'std_temp', 'measurement_method',
            'temporal_res', 'site_name', 'region'
        ]
        self._setup_logging()
        
    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _validate_temperature_data(self, data):
        original_len = len(data)
        valid_mask = (
            (data['temperature'].between(-100, 40)) &
            (data['depth'] >= -0.1) &
            (data['depth'] <= 1000) &
            (data['latitude'].between(-90, 90)) &
            (data['longitude'].between(-180, 180))
        )
        if 'measurement_method' in data.columns:
            data.loc[data['measurement_method'] == 'satellite', 'temperature'] = \
                data.loc[data['measurement_method'] == 'satellite', 'temperature'].clip(-80, 30)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            invalid_breakdown = {
                'temperature_range': len(data[~data['temperature'].between(-100, 40)]),
                'depth_range': len(data[~data['depth'].between(-0.1, 1000)]),
                'coordinate_range': len(data[
                    ~(data['latitude'].between(-90, 90)) | 
                    ~(data['longitude'].between(-180, 180))
                ])
            }
            self.logger.warning(
                f"Validation Summary:\n"
                f"- Original records: {original_len}\n"
                f"- Invalid records: {invalid_count}\n"
                f"- Invalid temperature range: {invalid_breakdown['temperature_range']}\n"
                f"- Invalid depth range: {invalid_breakdown['depth_range']}\n"
                f"- Invalid coordinates: {invalid_breakdown['coordinate_range']}"
            )
        return data[valid_mask]

    def _check_temporal_consistency(self, data):
        if 'temporal_res' in data.columns:
            temporal_groups = data.groupby('temporal_res')
            for res, group in temporal_groups:
                time_diffs = pd.Series(pd.to_datetime(group['datetime'])).diff()
                inconsistent = time_diffs.value_counts()
                if len(inconsistent) > 1:
                    self.logger.warning(
                        f"Inconsistent measurement intervals found for {res}: {inconsistent.head()}"
                    )
        return data

    def _process_chunk(self, chunk):
        try:
            chunk = chunk.dropna(subset=['datetime', 'temperature', 'source'])
            chunk['datetime'] = pd.to_datetime(chunk['datetime'], errors='coerce')
            chunk = chunk.dropna(subset=['datetime'])
            if len(chunk) == 0:
                return None
            chunk = self._validate_temperature_data(chunk)
            chunk = self._check_temporal_consistency(chunk)
            chunk['is_zc'] = (chunk['temperature'].abs() <= self.epsilon)
            chunk['event_id'] = (chunk['is_zc'] != chunk['is_zc'].shift()).cumsum()
            metrics = []
            for event_id in chunk[chunk['is_zc']]['event_id'].unique():
                event = chunk[chunk['event_id'] == event_id]
                if len(event) >= self.min_duration:
                    start_date = event['datetime'].min()
                    metric = {
                        'source': event['source'].iloc[0],
                        'latitude': float(event['latitude'].iloc[0]),
                        'longitude': float(event['longitude'].iloc[0]),
                        'depth': float(event['depth'].iloc[0]),
                        'season': event['season'].iloc[0],
                        'depth_zone': str(event['depth_zone'].iloc[0]) if 'depth_zone' in event.columns else 'unknown',
                        'start_date': start_date,
                        'end_date': event['datetime'].max(),
                        'duration_hours': len(event),
                        'mean_temp': event['temperature'].mean(),
                        'std_temp': event['temperature'].std(),
                        'measurement_method': event['measurement_method'].iloc[0] if 'measurement_method' in event.columns else None,
                        'temporal_res': event['temporal_res'].iloc[0] if 'temporal_res' in event.columns else None,
                        'site_name': event['site_name'].iloc[0] if 'site_name' in event.columns else None,
                        'region': event['region'].iloc[0] if 'region' in event.columns else None
                    }
                    metrics.append(metric)
            if metrics:
                df = pd.DataFrame(metrics)
                return df
            return None
        except Exception as e:
            self.logger.error(f"Error processing chunk: {str(e)}")
            self.logger.error(f"Current chunk columns: {chunk.columns}")
            return None

    def analyze_file(self, input_file, output_file, chunk_size=100000):
        self.logger.info(f"Starting analysis of {input_file}")
        try:
            preview = pd.read_csv(input_file, nrows=5, engine='python')
            self.logger.info(f"Input file columns: {list(preview.columns)}")
            self.logger.info(f"Input file dtypes: {preview.dtypes}")
            chunks = pd.read_csv(
                input_file,
                chunksize=chunk_size,
                engine='python',
                on_bad_lines='warn',
                dtype={
                    'temperature': 'float64',
                    'latitude': 'float64',
                    'longitude': 'float64',
                    'depth': 'float64'
                }
            )
            metrics_list = []
            total_chunks = 0
            processed_chunks = 0
            total_events = 0
            for chunk in tqdm(chunks, desc="Processing chunks"):
                total_chunks += 1
                chunk_metrics = self._process_chunk(chunk)
                if chunk_metrics is not None and not chunk_metrics.empty:
                    metrics_list.append(chunk_metrics)
                    processed_chunks += 1
                    total_events += len(chunk_metrics)
                if len(metrics_list) >= 10:
                    self._save_batch(metrics_list, output_file)
                    metrics_list = []
                    gc.collect()
            if metrics_list:
                self._save_batch(metrics_list, output_file)
            self.logger.info(
                f"Analysis completed:\n"
                f"- Total chunks processed: {total_chunks}\n"
                f"- Chunks with zero curtain events: {processed_chunks}\n"
                f"- Total zero curtain events detected: {total_events}"
            )
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            raise

    def _save_batch(self, metrics_list, output_file):
        if metrics_list:
            try:
                combined = pd.concat(metrics_list, ignore_index=True)
                for col in self.column_order:
                    if col not in combined.columns:
                        combined[col] = None
                combined = combined[self.column_order]
                output_path = Path(output_file).parent
                output_path.mkdir(parents=True, exist_ok=True)
                mode = 'a' if os.path.exists(output_file) else 'w'
                combined.to_csv(
                    output_file,
                    mode=mode,
                    header=(mode=='w'),
                    index=False,
                    columns=self.column_order
                )
            except Exception as e:
                self.logger.error(f"Error saving batch: {str(e)}")
                raise

class ZeroCurtainAnalyzer:
    def __init__(self, epsilon=2, min_duration=6):
        self.epsilon = epsilon
        self.min_duration = min_duration
        self.logger = logging.getLogger(__name__)
        
    def _normalize_datetime(self, dt_series):
        def parse_date(date_str):
            if pd.isna(date_str):
                return None
            try:
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except ValueError:
                        continue
                return pd.to_datetime(date_str)
            except Exception:
                return None
        return dt_series.apply(parse_date)

    def _process_chunk(self, chunk):
        try:
            if 'datetime' in chunk.columns:
                chunk['datetime'] = self._normalize_datetime(chunk['datetime'])
                chunk = chunk.dropna(subset=['datetime'])
            chunk['is_zc'] = (chunk['temperature'].abs() <= self.epsilon)
            chunk['event_id'] = (chunk['is_zc'] != chunk['is_zc'].shift()).cumsum()
            metrics = []
            for event_id in chunk[chunk['is_zc']]['event_id'].unique():
                event = chunk[chunk['event_id'] == event_id]
                if len(event) >= self.min_duration:
                    metric = {
                        'site_id': event['site_id'].iloc[0],
                        'site_name': str(event['site_name'].iloc[0]),
                        'depth': event['depth'].iloc[0],
                        'year': event['year'].iloc[0],
                        'season': str(event['season'].iloc[0]),
                        'depth_zone': str(event['depth_zone'].iloc[0]),
                        'start_date': event['datetime'].min(),
                        'end_date': event['datetime'].max(),
                        'duration_hours': len(event),
                        'mean_temp': event['temperature'].mean(),
                        'std_temp': event['temperature'].std()
                    }
                    metrics.append(metric)
            return pd.DataFrame(metrics) if metrics else None
        except Exception as e:
            self.logger.error(f"Error processing chunk: {str(e)}")
            return None

    def analyze_file(self, input_file, output_file, chunk_size=100000):
        self.logger.info(f"Starting analysis of {input_file}")
        output_path = Path(output_file).parent
        output_path.mkdir(parents=True, exist_ok=True)
        try:
            dtypes = {
                'season': 'object',
                'site_name': 'object',
                'station_name': 'object',
                'site_id': 'object',
                'depth_zone': 'object'
            }
            chunks = pd.read_csv(input_file, chunksize=chunk_size, dtype=dtypes)
            metrics_list = []
            total_chunks = 0
            for chunk in tqdm(chunks):
                chunk_metrics = self._process_chunk(chunk)
                if chunk_metrics is not None and not chunk_metrics.empty:
                    metrics_list.append(chunk_metrics)
                    total_chunks += 1
                if len(metrics_list) >= 10:
                    self._save_batch(metrics_list, output_file)
                    metrics_list = []
                    gc.collect()
            if metrics_list:
                self._save_batch(metrics_list, output_file)
            if total_chunks > 0:
                self._save_summary(output_file)
            self.logger.info(f"Completed analysis: processed {total_chunks} chunks")
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            raise
            
    def _save_batch(self, metrics_list, output_file):
        if metrics_list:
            combined = pd.concat(metrics_list, ignore_index=True)
            mode = 'a' if Path(output_file).exists() else 'w'
            header = not Path(output_file).exists()
            combined.to_csv(output_file, mode=mode, header=header, index=False)
            
    def _save_summary(self, metrics_file):
        try:
            metrics = pd.read_csv(metrics_file)
            def parse_dates(date_series):
                return pd.to_datetime(date_series, infer_datetime_format=True, format='mixed')
            metrics['start_date'] = parse_dates(metrics['start_date'])
            metrics['end_date'] = parse_dates(metrics['end_date'])
            date_range = f"{metrics['start_date'].min():%Y-%m-%d} to {metrics['end_date'].max():%Y-%m-%d}"
            summary = {
                'total_events': len(metrics),
                'unique_sites': metrics['site_id'].nunique(),
                'temporal_coverage': date_range,
                'events_by_season': metrics['season'].value_counts().to_dict(),
                'mean_duration_hours': float(metrics['duration_hours'].mean()),
                'depth_stats': {
                    'min': float(metrics['depth'].min()),
                    'max': float(metrics['depth'].max()),
                    'mean': float(metrics['depth'].mean())
                },
                'temperature_stats': {
                    'mean': float(metrics['mean_temp'].mean()),
                    'std': float(metrics['std_temp'].mean())
                }
            }
            if 'depth_zone' in metrics.columns:
                summary['events_by_depth_zone'] = metrics['depth_zone'].value_counts().to_dict()
            summary_file = Path(metrics_file).with_suffix('.summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            self.logger.info(f"Summary saved to {summary_file}")
        except Exception as e:
            self.logger.error(f"Error saving summary: {str(e)}")

def calculate_site_metrics(data, site_id):
    site_data = data[data['site_id'] == site_id].copy()
    metrics = {
        'site_id': site_id,
        'site_name': site_data['site_name'].iloc[0],
        'latitude': site_data['latitude'].iloc[0],
        'longitude': site_data['longitude'].iloc[0],
        'n_depths': site_data['depth'].nunique(),
        'depth_range': site_data['depth'].max() - site_data['depth'].min(),
        'n_years': site_data['year'].nunique(),
        'year_range': site_data['year'].max() - site_data['year'].min(),
        'n_seasons': site_data['season'].nunique(),
        'n_zones': site_data['depth_zone'].nunique(),
        'total_events': len(site_data)
    }
    yearly_stats = site_data.groupby('year').agg({
        'duration_days': ['mean', 'std', 'count'],
        'mean_temp': ['mean', 'std']
    }).reset_index()
    if len(yearly_stats) >= 3:
        years = yearly_stats['year'].values
        durations = yearly_stats['duration_days']['mean'].values
        temps = yearly_stats['mean_temp']['mean'].values
        duration_trend = stats.linregress(years, durations)
        temp_trend = stats.linregress(years, temps)
        metrics.update({
            'duration_trend': duration_trend.slope,
            'duration_trend_p': duration_trend.pvalue,
            'temp_trend': temp_trend.slope,
            'temp_trend_p': temp_trend.pvalue
        })
    zone_stats = site_data.groupby('depth_zone').agg({
        'duration_days': ['mean', 'std', 'count'],
        'mean_temp': ['mean', 'std']
    }).reset_index()
    if len(zone_stats) >= 2:
        zone_corr = site_data.pivot_table(
            index='year',
            columns='depth_zone',
            values='duration_days',
            aggfunc='mean'
        ).corr()
        metrics['zone_correlation'] = zone_corr.mean().mean()
    season_stats = site_data.groupby('season').agg({
        'duration_days': ['mean', 'std', 'count'],
        'mean_temp': ['mean', 'std']
    }).reset_index()
    if len(season_stats) >= 2:
        f_stat, p_value = stats.f_oneway(
            *[group['duration_days'].values 
              for name, group in site_data.groupby('season')]
        )
        metrics['seasonal_f_stat'] = f_stat
        metrics['seasonal_p_value'] = p_value
    return metrics

def identify_exemplar_sites(data, min_criteria=None):
    if min_criteria is None:
        min_criteria = {
            'n_depths': 3,
            'n_years': 2,
            'n_seasons': 2,
            'n_zones': 2,
            'total_events': 20
        }
    site_metrics = []
    for site_id in data['site_id'].unique():
        metrics = calculate_site_metrics(data, site_id)
        site_metrics.append(metrics)
    metrics_df = pd.DataFrame(site_metrics)
    mask = (
        (metrics_df['n_depths'] >= min_criteria['n_depths']) &
        (metrics_df['n_years'] >= min_criteria['n_years']) &
        (metrics_df['n_seasons'] >= min_criteria['n_seasons']) &
        (metrics_df['n_zones'] >= min_criteria['n_zones']) &
        (metrics_df['total_events'] >= min_criteria['total_events'])
    )
    exemplar_sites = metrics_df[mask].copy()
    if len(exemplar_sites) == 0:
        print("No sites met the minimum criteria.")
        return None
    for col in ['n_depths', 'n_years', 'n_seasons', 'n_zones', 'total_events']:
        exemplar_sites[f'{col}_score'] = (
            exemplar_sites[col] - exemplar_sites[col].min()
        ) / (exemplar_sites[col].max() - exemplar_sites[col].min())
    exemplar_sites['coverage_score'] = exemplar_sites[[
        'n_depths_score', 'n_years_score', 'n_seasons_score',
        'n_zones_score', 'total_events_score'
    ]].mean(axis=1)
    if 'duration_trend_p' in exemplar_sites.columns:
        exemplar_sites['trend_score'] = (
            (1 - exemplar_sites['duration_trend_p']) + 
            (1 - exemplar_sites['temp_trend_p'])
        ) / 2
    if 'seasonal_p_value' in exemplar_sites.columns:
        exemplar_sites['seasonal_score'] = 1 - exemplar_sites['seasonal_p_value']
    score_columns = ['coverage_score']
    if 'trend_score' in exemplar_sites.columns:
        score_columns.append('trend_score')
    if 'seasonal_score' in exemplar_sites.columns:
        score_columns.append('seasonal_score')
    exemplar_sites['total_score'] = exemplar_sites[score_columns].mean(axis=1)
    exemplar_sites = exemplar_sites.sort_values('total_score', ascending=False)
    print(f"\nFound {len(exemplar_sites)} exemplar sites meeting criteria:")
    print(f"Minimum criteria:")
    for criterion, value in min_criteria.items():
        print(f"- {criterion}: {value}")
    print("\nTop exemplar sites:")
    for idx, site in exemplar_sites.head().iterrows():
        print(f"\nSite: {site['site_id']}")
        print(f"Location: {site['latitude']:.3f}°N, {site['longitude']:.3f}°E")
        print(f"Coverage: {site['n_years']} years, {site['n_seasons']} seasons")
        print(f"Depths: {site['n_depths']} (range: {site['depth_range']:.1f}m)")
        if 'duration_trend' in site:
            print(f"Duration trend: {site['duration_trend']:.3f} days/year (p={site['duration_trend_p']:.3f})")
        if 'temp_trend' in site:
            print(f"Temperature trend: {site['temp_trend']:.3f}°C/year (p={site['temp_trend_p']:.3f})")
        if 'seasonal_p_value' in site:
            print(f"Seasonal strength: p={site['seasonal_p_value']:.3f}")
        print(f"Total score: {site['total_score']:.3f}")
    return exemplar_sites

def plot_site(data, site_id, output_dir="site_plots"):
    site_data = data[data['site_id'] == site_id].copy()
    site_name = site_data['site_name'].iloc[0]
    fig = plt.figure(figsize=(10, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 1], hspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    vmin = site_data['mean_temp'].min()
    vmax = site_data['mean_temp'].max()
    scatter = ax1.scatter(site_data['depth'], 
                         site_data['duration_days'],
                         c=site_data['mean_temp'],
                         cmap='RdBu_r',
                         alpha=0.7,
                         s=80,
                         vmin=vmin,
                         vmax=vmax)
    cax = fig.add_axes([0.92, 0.65, 0.02, 0.2])
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label('Mean\nTemperature (°C)', fontsize=10, labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    ymax = site_data['duration_days'].max()
    ax1.set_ylim(-0.5, ymax * 1.1)
    seasons = sorted(site_data['season'].unique())
    leg_elements = []
    for season in seasons:
        leg_elements.append(ax1.scatter([], [], marker='o', label=season, s=50))
    ax1.set_xlabel('Depth (m)', fontsize=10)
    ax1.set_ylabel('Duration (days)', fontsize=10)
    site_lat = site_data.latitude.iloc[0]
    site_lon = site_data.longitude.iloc[0]
    ax1.set_title(f'Zero Curtain Events\n({site_lat:.3f}°N, {site_lon:.3f}°E)',
                  pad=15, fontsize=11)
    ax1.legend(handles=leg_elements, title='Season', 
              bbox_to_anchor=(1.15, 0.5), 
              loc='center left',
              fontsize=9,
              title_fontsize=10)
    ax2 = fig.add_subplot(gs[1])
    years = sorted(site_data['year'].unique())
    zones = sorted(site_data['depth_zone'].unique())
    for zone in zones:
        zone_data = site_data[site_data['depth_zone'] == zone]
        yearly_mean = zone_data.groupby('year')['duration_days'].mean()
        ax2.plot(yearly_mean.index, yearly_mean.values, 'o-', 
                label=zone.replace('_', ' ').title(),
                markersize=5, linewidth=1.5)
    y_vals = [v for zone in zones 
              for v in site_data[site_data['depth_zone'] == zone].groupby('year')['duration_days'].mean()]
    ymin, ymax = min(y_vals), max(y_vals)
    ax2.set_ylim(ymin * 0.9, ymax * 1.1)
    ax2.set_xlabel('Year', fontsize=10)
    ax2.set_ylabel('Mean Duration (days)', fontsize=10)
    ax2.set_title('Duration by Year and Depth Zone', pad=15, fontsize=11)
    ax2.legend(title='Depth Zone', bbox_to_anchor=(1.15, 0.5),
              loc='center left', fontsize=9, title_fontsize=10)
    ax3 = fig.add_subplot(gs[2])
    boxplot = ax3.boxplot([site_data[site_data['depth_zone'] == zone]['mean_temp'] 
                          for zone in zones],
                         labels=[z.replace('_', ' ').title() for z in zones],
                         widths=0.7,
                         medianprops=dict(color="orange", linewidth=1.5),
                         flierprops=dict(marker='o', markersize=4, alpha=0.5))
    temp_min = site_data['mean_temp'].min()
    temp_max = site_data['mean_temp'].max()
    ax3.set_ylim(temp_min * 1.1, temp_max * 1.1)
    ax3.set_xlabel('Depth Zone', fontsize=10)
    ax3.set_ylabel('Temperature (°C)', fontsize=10)
    ax3.set_title('Temperature Distribution by Depth Zone', pad=15, fontsize=11)
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(labelsize=9)
    plt.subplots_adjust(right=0.85, left=0.12, top=0.95, bottom=0.08)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"site_{site_id.split('-')[1][:30]}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def spatial_temporal_clustering(df):
    clustered_df = df.copy()
    clustering_features = clustered_df[['latitude', 'longitude', 'duration_hours']].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(clustering_features)
    db = DBSCAN(eps=2.0, min_samples=5).fit(X)
    clustered_df['cluster'] = db.labels_
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(
        clustered_df['longitude'], 
        clustered_df['latitude'], 
        c=clustered_df['cluster'], 
        cmap='viridis'
    )
    plt.colorbar(scatter)
    plt.title('Spatial Clusters of Zero Curtain Events')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.subplot(2, 2, 2)
    cluster_summary = clustered_df.groupby('cluster').agg({
        'duration_hours': ['mean', 'count'],
        'latitude': 'mean',
        'longitude': 'mean'
    })
    cluster_summary.columns = ['mean_duration', 'event_count', 'mean_lat', 'mean_lon']
    cluster_summary['mean_duration'].plot(kind='bar')
    plt.title('Mean Duration by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Mean Duration (hours)')
    plt.subplot(2, 2, 3)
    cluster_summary['event_count'].plot(kind='bar')
    plt.title('Number of Events by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Events')
    plt.subplot(2, 2, 4)
    cluster_depth = clustered_df.groupby(['cluster', 'depth_zone']).size().unstack(fill_value=0)
    cluster_depth.plot(kind='bar', stacked=True)
    plt.title('Depth Zone Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Events')
    plt.tight_layout()
    plt.show()
    print("\nCluster Analysis Summary:")
    print(cluster_summary)
    cluster_groups = [
        group['duration_hours'].values 
        for name, group in clustered_df.groupby('cluster') 
        if name != -1
    ]
    if len(cluster_groups) > 1:
        f_statistic, p_value = stats.f_oneway(*cluster_groups)
        print("\nANOVA Test for Cluster Duration Differences:")
        print(f"F-statistic: {f_statistic:.4f}")
        print(f"p-value: {p_value:.4f}")
    return clustered_df, cluster_summary

def analyze_zc_alt_dynamics(data, output_path):
    def calc_location_trend(group):
        if len(group) > 2 and len(group['year'].unique()) > 1:
            try:
                return stats.linregress(group['year'], group['duration_hours']).slope
            except:
                return np.nan
        return np.nan
    location_trends = data.groupby(['latitude', 'longitude']).apply(
        calc_location_trend
    ).reset_index(name='trend')
    location_trends = location_trends.dropna(subset=['trend'])
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.NorthPolarStereo())
    ax1.add_feature(cfeature.LAND, facecolor='white', alpha=0.3)
    ax1.add_feature(cfeature.OCEAN, alpha=0.3)
    ax1.add_feature(cfeature.COASTLINE, color='black')
    if len(location_trends) > 0:
        scatter = ax1.scatter(location_trends['longitude'],
                            location_trends['latitude'],
                            c=location_trends['trend'],
                            transform=ccrs.PlateCarree(),
                            cmap='RdBu_r',
                            s=100,
                            alpha=0.7)
        plt.colorbar(scatter, ax=ax1, label='Trend in Duration (hours/year)')
    ax1.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    ax1.set_title('Spatial Distribution of Zero Curtain Trends')
    ax2 = fig.add_subplot(gs[0, 1])
    annual_means = data.groupby('year').agg({
        'duration_hours': ['mean', 'std'],
        'depth': ['mean', 'std']
    }).reset_index()
    annual_means.columns = ['year', 'duration_mean', 'duration_std', 
                          'depth_mean', 'depth_std']
    color1 = 'tab:blue'
    ax2.errorbar(annual_means['year'], 
                annual_means['duration_mean'],
                yerr=annual_means['duration_std'],
                color=color1, 
                marker='o',
                label='Duration')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Mean Duration (hours)', color=color1)
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2_twin = ax2.twinx()
    color2 = 'tab:red'
    ax2_twin.errorbar(annual_means['year'],
                     annual_means['depth_mean'],
                     yerr=annual_means['depth_std'],
                     color=color2,
                     marker='s',
                     label='Depth')
    ax2_twin.set_ylabel('Mean Depth', color=color2)
    ax2_twin.tick_params(axis='y', labelcolor=color2)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.set_title('Temporal Evolution of Duration and Depth')
    ax3 = fig.add_subplot(gs[1, 0])
    valid_data = data.dropna(subset=['depth', 'duration_hours'])
    sns.scatterplot(data=valid_data, 
                   x='depth',
                   y='duration_hours',
                   hue='season',
                   alpha=0.5,
                   ax=ax3)
    if len(valid_data) > 1:
        z = np.polyfit(valid_data['depth'], valid_data['duration_hours'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(valid_data['depth'].min(), valid_data['depth'].max(), 100)
        ax3.plot(x_range, p(x_range), "r--", alpha=0.8,
                 label=f'Trend: {z[0]:.1f} hours/m')
        ax3.legend(title='Season')
    ax3.set_xlabel('Depth (m)')
    ax3.set_ylabel('Duration (hours)')
    ax3.set_title('Zero Curtain Duration vs Depth')
    ax4 = fig.add_subplot(gs[1, 1])
    seasonal_depth = data.groupby(['season', 'depth_zone']).agg({
        'duration_hours': ['mean', 'std']
    }).reset_index()
    seasonal_depth.columns = ['season', 'depth_zone', 'mean_duration', 'std_duration']
    sns.barplot(data=seasonal_depth,
               x='season',
               y='mean_duration',
               hue='depth_zone',
               ax=ax4)
    ax4.set_xlabel('Season')
    ax4.set_ylabel('Mean Duration (hours)')
    ax4.set_title('Seasonal Duration by Depth Zone')
    plt.xticks(rotation=45)
    stats_dict = {
        'temporal_trend': stats.linregress(annual_means['year'], 
                                               annual_means['duration_mean'])
        if len(annual_means) > 2 else None,
        'depth_correlation': stats.pearsonr(valid_data['depth'], 
                                                valid_data['duration_hours'])
        if len(valid_data) > 2 else None,
        'seasonal_variation': data.groupby('season')['duration_hours'].agg([
            'mean', 'std', 'count']).to_dict(),
        'spatial_summary': {
            'n_sites': len(location_trends),
            'sites_increasing': (location_trends['trend'] > 0).sum(),
            'sites_decreasing': (location_trends['trend'] < 0).sum()
        }
    }
    plt.tight_layout()
    plt.savefig(Path(output_path) / 'zc_alt_dynamics.png', 
                dpi=300, 
                bbox_inches='tight')
    return fig, stats_dict

def analyze_seasonal_evolution(data, output_path):
    fig = plt.figure(figsize=(20, 16))
    gs = plt.GridSpec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    seasonal_duration = data.groupby(['year', 'season'])['duration_hours'].mean().unstack()
    for season in seasonal_duration.columns:
        ax1.plot(seasonal_duration.index, seasonal_duration[season], 
                marker='o', label=season)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Mean Duration (hours)')
    ax1.set_title('Evolution of Zero Curtain Duration by Season')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2 = fig.add_subplot(gs[0, 1])
    early_year = data['year'].min()
    late_year = data['year'].max()
    mid_year = early_year + (late_year - early_year) // 2
    periods = {
        'Early': (early_year, mid_year),
        'Late': (mid_year + 1, late_year)
    }
    for period_name, (start_year, end_year) in periods.items():
        period_data = data[(data['year'] >= start_year) & 
                          (data['year'] <= end_year)]
        counts = period_data['season'].value_counts()
        ax2.bar(counts.index, counts.values, 
                alpha=0.5, label=f'{period_name} ({start_year}-{end_year})')
    ax2.set_ylabel('Number of Events')
    ax2.set_title('Distribution of Zero Curtain Events by Season')
    ax2.legend()
    ax3 = fig.add_subplot(gs[1, 0])
    seasonal_depth = data.groupby(['year', 'season'])['depth'].mean().unstack()
    for season in seasonal_depth.columns:
        ax3.plot(seasonal_depth.index, seasonal_depth[season], 
                marker='o', label=season)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Mean Depth (m)')
    ax3.set_title('Evolution of Zero Curtain Depth by Season')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax4 = fig.add_subplot(gs[1, 1])
    sns.boxplot(data=data, x='season', y='mean_temp', ax=ax4)
    ax4.set_xlabel('Season')
    ax4.set_ylabel('Mean Temperature (°C)')
    ax4.set_title('Temperature Distribution During Zero Curtain by Season')
    ax5 = fig.add_subplot(gs[2, :])
    for season in data['season'].unique():
        season_data = data[data['season'] == season]
        sns.regplot(data=season_data, 
                   x='depth', 
                   y='duration_hours',
                   label=season,
                   scatter_kws={'alpha': 0.3},
                   ax=ax5)
    ax5.set_xlabel('Depth (m)')
    ax5.set_ylabel('Duration (hours)')
    ax5.set_title('Zero Curtain Duration vs Depth by Season')
    ax5.legend()
    stats_dict = {}
    for season in data['season'].unique():
        season_data = data[data['season'] == season]
        season_trends = season_data.groupby('year')['duration_hours'].mean()
        if len(season_trends) > 2:
            trend = stats.linregress(season_trends.index, season_trends.values)
            stats_dict[season] = {
                'duration_trend': trend.slope,
                'trend_pvalue': trend.pvalue,
                'mean_duration': season_data['duration_hours'].mean(),
                'std_duration': season_data['duration_hours'].std(),
                'mean_depth': season_data['depth'].mean(),
                'depth_correlation': stats.pearsonr(
                    season_data['depth'].dropna(),
                    season_data['duration_hours'].dropna()
                )[0]
            }
    plt.tight_layout()
    plt.savefig(Path(output_path) / 'seasonal_evolution.png',
                dpi=300,
                bbox_inches='tight')
    return fig, stats_dict

sample_df = merged_df.sample(frac=0.001)

sample_results = run_full_analysis_pipeline_with_memory_management(
    sample_df,
    output_base_dir='sample_test',
    use_checkpoints=True,
    batch_size=50
)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LogNorm

def plot_event_distribution(df):
    """
    Simple polar stereographic plot with logarithmic color scale
    """
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    
    # Basic map setup
    ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, color='black', linewidth=0.75)
    ax.gridlines()
    
    # Get event counts
    events = df.groupby(['source', 'latitude', 'longitude']).size()
    coords = events.index.to_frame()
    
    # Create scatter plot with basic log norm
    scatter = ax.scatter(coords['longitude'], 
                        coords['latitude'],
                        transform=ccrs.PlateCarree(),
                        c=events.values,
                        s=40,
                        #cmap='gist_ncar',
                        cmap='jet',
                        norm=LogNorm(vmin=10, vmax=890),
                        alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter,extend='max')
    cbar.set_label('Number of Events (log scale)')
    
    plt.title('Spatial Distribution of Zero Curtain Events')

    plt.tight_layout()
    
    return fig

fig = plot_event_distribution(events)

fig.savefig('spatialdistzerocurtainplot.png',dpi=300)

def plot_duration_distribution(df):
    """
    Create polar stereographic plot of mean zero curtain duration
    """
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    
    # Set up the map
    ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, color='black', linewidth=0.5)
    ax.gridlines()
    
    # Calculate mean duration by site
    site_duration = df.groupby(['source', 'latitude', 'longitude'])['duration_hours'].mean().reset_index()
    
    # Create scatter plot
    scatter = ax.scatter(site_duration['longitude'], site_duration['latitude'], 
                        transform=ccrs.PlateCarree(),
                        c=site_duration['duration_hours'], 
                        cmap='plasma',
                        vmin=0,
                        vmax=140,
                        s=40)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, extend='max')
    cbar.set_label('Mean Duration (hours)')#, size=12, labelpad=10)
    #cbar.ax.tick_params(labelsize=10)
    
    plt.title('Mean Zero Curtain Duration by Site')#, size=14, pad=20)
    
    plt.tight_layout()
    
    return fig
    
fig = plot_duration_distribution(events)

fig.savefig('meanzerocurtaindurationplot.png',dpi=300)

def plot_depth_distribution(df):
    """
    Create polar stereographic plot of depth zone distribution
    """
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    
    # Set up the map
    ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, color='black', linewidth=0.5)
    ax.gridlines()
    
    # Prepare depth zone data
    depth_zone_counts = df.groupby(['soil_temp_depth_zone', 'latitude', 'longitude']).size().reset_index(name='count')
    depth_order = ['shallow', 'intermediate', 'deep', 'very_deep']
    #depth_colors = ['#90EE90', '#4682B4', '#800080', '#DC143C']
    #depth_colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']  # Enhanced color scheme
    #depth_colors = ['#d73027', '#fc8d59', '#4575b4', '#313695']
    #depth_colors = ['#ffeda0', '#feb24c', '#f03b20', '#bd0026']
    #depth_colors = ['#1a9850', '#91cf60', '#d73027', '#762a83']
    depth_colors = ['#003f5c', '#58508d', '#ff6361', '#ffa600']
    
    # Plot each depth zone
    legend_elements = []
    for depth, color in zip(depth_order, depth_colors):
        depth_data = depth_zone_counts[depth_zone_counts['soil_temp_depth_zone'] == depth]
        scatter = ax.scatter(depth_data['longitude'], depth_data['latitude'], 
                           transform=ccrs.PlateCarree(),
                           c=color, 
                           s=np.sqrt(depth_data['count'])*3,
                           alpha=0.7,
                           label=f"{depth.capitalize()} (n={len(depth_data)})")
    
    # Add legend
    ax.legend(title='Depth Zone',
             title_fontsize=12,
             fontsize=10,
             bbox_to_anchor=(0, 0.9875),
             loc='upper left',
             borderaxespad=0.5)
    
    plt.title('Spatial Distribution by Depth Zone')#, size=14, pad=20)
    
    plt.tight_layout()
    
    return fig
    
fig = plot_depth_distribution(events)

fig.savefig('depthzonezerocurtainplot.png',dpi=300)

def plot_circumpolar_duration(df):
    """
    Create circumpolar plot of zero curtain duration
    """
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    
    # Set up the map
    ax.set_extent([-180, 180, 45, 90], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, color='black', linewidth=0.75)
    ax.gridlines()
    
    # Create scatter plot
    scatter = ax.scatter(df['longitude'], df['latitude'], 
                        c=df['duration_hours'],
                        cmap='viridis',
                        #vmin=6,
                        #vmax=120,
                        norm=LogNorm(vmin=1, vmax=200),
                        #norm=LogNorm(vmin=10, vmax=100),
                        transform=ccrs.PlateCarree(), 
                        s=40, 
                        alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, extend='both')
    cbar.set_label('Duration (hours)')#, size=12)
    #cbar.ax.tick_params(labelsize=10)
    
    plt.title('Circumpolar Distribution of Zero Curtain Events')#, size=14, pad=20)
    
    plt.tight_layout()
    
    return fig
    
fig = plot_circumpolar_duration(events) #1-100

fig.savefig('circumarcticdurationevents_zerocurtainplot.png',dpi=300)

import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

def spatial_zero_curtain_analysis(df):
    """
    Comprehensive spatial analysis of zero curtain events
    """
    # 1. Basic Spatial Distribution
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig)
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )
    
    # Subplot 1: Total Events by Site
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax1.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax1.add_feature(cfeature.LAND, facecolor='white', alpha=0.3)
    ax1.add_feature(cfeature.OCEAN, alpha=0.3)
    
    site_events = gdf.groupby(['site_id', 'latitude', 'longitude']).size().reset_index(name='event_count')
    scatter1 = ax1.scatter(site_events['longitude'], site_events['latitude'], 
                c=site_events['event_count'], cmap='viridis', 
                s=site_events['event_count']*3,
                alpha=0.7)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Number of Events', size=12, labelpad=10)
    cbar1.ax.tick_params(labelsize=10)
    ax1.set_title('Spatial Distribution of Zero Curtain Events', size=14, pad=20)
    ax1.set_title('Spatial Distribution of Zero Curtain Events', size=14, pad=20)
    # Calculate valid extent bounds
    lon_min = np.nanmin(df.longitude[np.isfinite(df.longitude)])
    lon_max = np.nanmax(df.longitude[np.isfinite(df.longitude)])
    lat_min = np.nanmin(df.latitude[np.isfinite(df.latitude)])
    lat_max = np.nanmax(df.latitude[np.isfinite(df.latitude)])
    
    # Set buffer for extent
    lon_buffer = (lon_max - lon_min) * 0.1
    lat_buffer = (lat_max - lat_min) * 0.1
    
    ax1.set_extent([lon_min - lon_buffer, lon_max + lon_buffer,
                   lat_min - lat_buffer, lat_max + lat_buffer],
                   crs=ccrs.PlateCarree())
    gl1 = ax1.gridlines(draw_labels=True, alpha=0.3)
    gl1.xlabel_style = {'size': 10}
    gl1.ylabel_style = {'size': 10}
    
    #ax1.set_xlabel('Longitude', size=12, labelpad=10)
    #ax1.set_ylabel('Latitude', size=12, labelpad=10)
    #ax1.tick_params(labelsize=10)
    #ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Mean Duration by Site
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax2.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax2.add_feature(cfeature.LAND, facecolor='white', alpha=0.3)
    ax2.add_feature(cfeature.OCEAN, alpha=0.3)
    
    site_duration = gdf.groupby(['site_id', 'latitude', 'longitude'])['duration_hours'].mean().reset_index()
    
    duration_min = np.percentile(site_duration['duration_hours'], 5)
    duration_max = np.percentile(site_duration['duration_hours'], 95)
    
    scatter2 = ax2.scatter(site_duration['longitude'], site_duration['latitude'], 
                c=site_duration['duration_hours'], cmap='plasma', 
                           vmin=duration_min, vmax=duration_max, s=100)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Mean Duration (hours)', size=12, labelpad=10)
    cbar2.ax.tick_params(labelsize=10)
    ax2.set_title('Mean Zero Curtain Duration by Site', size=14, pad=20)
    ax2.set_extent([lon_min - lon_buffer, lon_max + lon_buffer,
                   lat_min - lat_buffer, lat_max + lat_buffer],
                   crs=ccrs.PlateCarree())
    gl2 = ax2.gridlines(draw_labels=True, alpha=0.3)
    gl2.xlabel_style = {'size': 10}
    gl2.ylabel_style = {'size': 10}
    
    #ax2.set_xlabel('Longitude', size=12, labelpad=10)
    #ax2.set_ylabel('Latitude', size=12, labelpad=10)
    #ax2.tick_params(labelsize=10)
    #ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Depth Zone Distribution
    ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax3.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax3.add_feature(cfeature.LAND, facecolor='white', alpha=0.3)
    ax3.add_feature(cfeature.OCEAN, alpha=0.3)
    
    depth_zone_counts = gdf.groupby(['depth_zone', 'latitude', 'longitude']).size().reset_index(name='count')
    depth_order = ['shallow', 'intermediate', 'deep', 'very_deep']
    depth_colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']  # Enhanced color scheme
    
    legend_elements = []
    
    for depth, color in zip(depth_order, depth_colors):
        depth_data = depth_zone_counts[depth_zone_counts['depth_zone'] == depth]
        scatter = ax3.scatter(depth_data['longitude'], depth_data['latitude'], 
                    c=color, 
                    label=f"{depth.capitalize()} (n={len(depth_data)})", 
                    s=np.sqrt(depth_data['count'])*50, 
                    alpha=0.7)
        legend_elements.append(plt.scatter([], [], c=color, 
                                        label=f"{depth.capitalize()} (n={len(depth_data)})",
                                        s=100, alpha=0.7))
    
    ax3.legend(handles=legend_elements,
              title='Depth Zone',
              title_fontsize=12,
              fontsize=10,
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              borderaxespad=0,
              alignment='left',
              handletextpad=1)
    
    ax3.set_title('Spatial Distribution by Depth Zone', size=14, pad=20)
    ax3.set_extent([lon_min - lon_buffer, lon_max + lon_buffer,
                   lat_min - lat_buffer, lat_max + lat_buffer],
                   crs=ccrs.PlateCarree())
    gl3 = ax3.gridlines(draw_labels=True, alpha=0.3)
    gl3.xlabel_style = {'size': 10}
    gl3.ylabel_style = {'size': 10}
    
    #ax3.set_xlabel('Longitude', size=12, labelpad=10)
    #ax3.set_ylabel('Latitude', size=12, labelpad=10)
    #ax3.tick_params(labelsize=10)
    #ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Cartographic Projection
    ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.NorthPolarStereo())
    ax4.set_extent([-180, 180, 45, 90], ccrs.PlateCarree())
    ax4.gridlines(draw_labels=True)
    ax4.add_feature(cfeature.LAND, facecolor='white', alpha=0.5)
    ax4.add_feature(cfeature.OCEAN, alpha=0.5)
    ax4.coastlines(resolution='50m', color='black', linewidth=0.5)

    duration_min_global = np.percentile(gdf['duration_hours'], 5)
    duration_max_global = np.percentile(gdf['duration_hours'], 95)
    
    scatter4 = ax4.scatter(gdf['longitude'], gdf['latitude'], 
                         c=gdf['duration_hours'], cmap='viridis', 
                           vmin=duration_min_global, vmax=duration_max_global, 
                           transform=ccrs.PlateCarree(), s=30, alpha=0.7)
    cbar4 = plt.colorbar(scatter4, ax=ax4, orientation='horizontal', pad=0.1)
    cbar4.set_label('Duration (hours)', size=12)
    cbar4.ax.tick_params(labelsize=10)
    ax4.set_title('Circumpolar Distribution of Zero Curtain Events', size=14, pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Detailed Spatial Statistics
    print("\nSpatial Analysis Summary:")
    
    # Number of unique sites
    unique_sites = gdf['site_id'].nunique()
    print(f"Total Unique Sites: {unique_sites}")
    
    # Spatial extent
    print("\nSpatial Extent:")
    print(f"Latitude Range: {gdf['latitude'].min():.2f} to {gdf['latitude'].max():.2f}")
    print(f"Longitude Range: {gdf['longitude'].min():.2f} to {gdf['longitude'].max():.2f}")
    
    # Top sites by event count
    top_sites = gdf.groupby('site_id').size().nlargest(5)
    print("\nTop 5 Sites by Event Count:")
    print(top_sites)
    
    # Depth zone distribution
    depth_distribution = gdf.groupby('depth_zone').size()
    print("\nDepth Zone Distribution:")
    print(depth_distribution)
    
    return fig, gdf

import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

def spatial_zero_curtain_analysis(df):
    """
    Comprehensive spatial analysis of zero curtain events
    """
    # 1. Basic Spatial Distribution
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig)
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

    def setup_polar_map(ax):
        """Helper function to set up polar stereographic map"""
        ax.set_extent([-180, 180, 45, 90], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, color='black', linewidth=0.5)
        ax.gridlines()
        return ax
    
    # Subplot 1: Events Distribution
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.NorthPolarStereo())
    setup_polar_map(ax1)
    
    site_events = gdf.groupby(['site_id', 'latitude', 'longitude']).size().reset_index(name='event_count')
    scatter1 = ax1.scatter(site_events['longitude'], site_events['latitude'], 
                transform=ccrs.PlateCarree(),
                c=site_events['event_count'], 
                cmap='viridis', 
                norm=LogNorm(vmin=1, vmax=12000),
                s=np.sqrt(site_events['event_count'])*3,
                alpha=0.7)
    
    cbar1 = plt.colorbar(scatter1, ax=ax1, extend='max')
    cbar1.set_label('Number of Events', size=10)
    cbar1.ax.tick_params(labelsize=8)
    ax1.set_title('Spatial Distribution of Zero Curtain Events', size=12, pad=10)
    
    
    # Subplot 2: Duration Distribution
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.NorthPolarStereo())
    setup_polar_map(ax2)
    
    site_duration = gdf.groupby(['site_id', 'latitude', 'longitude'])['duration_hours'].mean().reset_index()
    duration_min = np.percentile(site_duration['duration_hours'], 5)
    duration_max = np.percentile(site_duration['duration_hours'], 95)
    scatter2 = ax2.scatter(site_duration['longitude'], site_duration['latitude'], 
                transform=ccrs.PlateCarree(),
                c=site_duration['duration_hours'], cmap='plasma', vmin=duration_min, 
                           vmax=duration_max, s=50)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Mean Duration (hours)', size=12, labelpad=10)
    cbar2.ax.tick_params(labelsize=10)
    ax2.set_title('Mean Zero Curtain Duration by Site', size=14, pad=20)
    
    
    # Subplot 3: Depth Zones
    ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.NorthPolarStereo())
    setup_polar_map(ax3)
    
    depth_zone_counts = gdf.groupby(['depth_zone', 'latitude', 'longitude']).size().reset_index(name='count')
    depth_order = ['shallow', 'intermediate', 'deep', 'very_deep']
    depth_colors = ['#90EE90', '#4682B4', '#800080', '#DC143C']
    #depth_colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']  # Enhanced color scheme
    
    legend_elements = []
    for depth, color in zip(depth_order, depth_colors):
        depth_data = depth_zone_counts[depth_zone_counts['depth_zone'] == depth]
        scatter = ax3.scatter(depth_data['longitude'], depth_data['latitude'], 
                    transform=ccrs.PlateCarree(),
                    c=color, label=f"{depth.capitalize()} (n={len(depth_data)})", 
                    s=np.sqrt(depth_data['count'])*50,
                    alpha=0.7)
        n_sites = len(depth_data)
        legend_elements.append(plt.scatter([], [], c=color, 
                                        label=f"{depth.capitalize()} (n={n_sites})",
                                        s=100, alpha=0.7))
    
    ax3.legend(handles=legend_elements,
              title='Depth Zone',
              title_fontsize=12,
              fontsize=10,
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              borderaxespad=0,
              alignment='left',
              handletextpad=1)
              
    ax3.set_title('Spatial Distribution by Depth Zone', size=14, pad=20)
    
    
    # Subplot 4: Duration Distribution (Polar)
    ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.NorthPolarStereo())
    setup_polar_map(ax4)

    duration_min_global = np.percentile(gdf['duration_hours'], 5)
    duration_max_global = np.percentile(gdf['duration_hours'], 95)
    
    scatter4 = ax4.scatter(gdf['longitude'], gdf['latitude'], 
                         c=gdf['duration_hours'], cmap='viridis', 
                           vmin=duration_min_global, vmax=duration_max_global, 
                           transform=ccrs.PlateCarree(), s=30, alpha=0.7)
    
    cbar4 = plt.colorbar(scatter4, ax=ax4, pad=0.1)
    cbar4.set_label('Duration (hours)', size=12)
    cbar4.ax.tick_params(labelsize=10)
    ax4.set_title('Circumpolar Distribution of Zero Curtain Events', size=14, pad=20)
    
    plt.tight_layout()
    plt.show()

    
    # Detailed Spatial Statistics
    print("\nSpatial Analysis Summary:")
    
    # Number of unique sites
    unique_sites = gdf['site_id'].nunique()
    print(f"Total Unique Sites: {unique_sites}")
    
    # Spatial extent
    print("\nSpatial Extent:")
    print(f"Latitude Range: {gdf['latitude'].min():.2f} to {gdf['latitude'].max():.2f}")
    print(f"Longitude Range: {gdf['longitude'].min():.2f} to {gdf['longitude'].max():.2f}")
    
    # Top sites by event count
    top_sites = gdf.groupby('site_id').size().nlargest(5)
    print("\nTop 5 Sites by Event Count:")
    print(top_sites)
    
    # Depth zone distribution
    depth_distribution = gdf.groupby('depth_zone').size()
    print("\nDepth Zone Distribution:")
    print(depth_distribution)
    
    return fig, gdf

spatial_gdf = spatial_zero_curtain_analysis(processor2_df_with_coords)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LogNorm

def plot_event_distribution(df):
    """
    Simple polar stereographic plot with logarithmic color scale
    """
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    
    # Basic map setup
    ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, color='black', linewidth=0.75)
    ax.gridlines()
    
    # Get event counts
    events = df.groupby(['site_id', 'latitude', 'longitude']).size()
    coords = events.index.to_frame()
    
    # Create scatter plot with basic log norm
    scatter = ax.scatter(coords['longitude'], 
                        coords['latitude'],
                        transform=ccrs.PlateCarree(),
                        c=events.values,
                        s=40,
                        cmap='gist_ncar',
                        norm=LogNorm(vmin=10, vmax=12153),
                        alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter,extend='max')
    cbar.set_label('Number of Events (log scale)')
    
    plt.title('Spatial Distribution of Zero Curtain Events')

    plt.tight_layout()
    
    return fig
    
fig = plot_event_distribution(processor2_df_with_coords)

fig.savefig('spatialdistzerocurtainplot.png',dpi=300)

def plot_duration_distribution(df):
    """
    Create polar stereographic plot of mean zero curtain duration
    """
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    
    # Set up the map
    ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, color='black', linewidth=0.5)
    ax.gridlines()
    
    # Calculate mean duration by site
    site_duration = df.groupby(['site_id', 'latitude', 'longitude'])['duration_hours'].mean().reset_index()
    
    # Create scatter plot
    scatter = ax.scatter(site_duration['longitude'], site_duration['latitude'], 
                        transform=ccrs.PlateCarree(),
                        c=site_duration['duration_hours'], 
                        cmap='plasma',
                        vmin=0,
                        vmax=140,
                        s=40)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, extend='max')
    cbar.set_label('Mean Duration (hours)')#, size=12, labelpad=10)
    #cbar.ax.tick_params(labelsize=10)
    
    plt.title('Mean Zero Curtain Duration by Site')#, size=14, pad=20)
    
    plt.tight_layout()
    
    return fig
    
fig = plot_duration_distribution(processor2_df_with_coords)

fig.savefig('meanzerocurtaindurationplot.png',dpi=300)

def plot_depth_distribution(df):
    """
    Create polar stereographic plot of depth zone distribution
    """
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    
    # Set up the map
    ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, color='black', linewidth=0.5)
    ax.gridlines()
    
    # Prepare depth zone data
    depth_zone_counts = df.groupby(['depth_zone', 'latitude', 'longitude']).size().reset_index(name='count')
    depth_order = ['shallow', 'intermediate', 'deep', 'very_deep']
    #depth_colors = ['#90EE90', '#4682B4', '#800080', '#DC143C']
    #depth_colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']  # Enhanced color scheme
    #depth_colors = ['#d73027', '#fc8d59', '#4575b4', '#313695']
    #depth_colors = ['#ffeda0', '#feb24c', '#f03b20', '#bd0026']
    #depth_colors = ['#1a9850', '#91cf60', '#d73027', '#762a83']
    depth_colors = ['#003f5c', '#58508d', '#ff6361', '#ffa600']
    
    # Plot each depth zone
    legend_elements = []
    for depth, color in zip(depth_order, depth_colors):
        depth_data = depth_zone_counts[depth_zone_counts['depth_zone'] == depth]
        scatter = ax.scatter(depth_data['longitude'], depth_data['latitude'], 
                           transform=ccrs.PlateCarree(),
                           c=color, 
                           s=np.sqrt(depth_data['count'])*3,
                           alpha=0.7,
                           label=f"{depth.capitalize()} (n={len(depth_data)})")
    
    # Add legend
    ax.legend(title='Depth Zone',
             title_fontsize=12,
             fontsize=10,
             bbox_to_anchor=(0, 0.9875),
             loc='upper left',
             borderaxespad=0.5)
    
    plt.title('Spatial Distribution by Depth Zone')#, size=14, pad=20)
    
    plt.tight_layout()
    
    return fig

fig = plot_depth_distribution(processor2_df_with_coords)

fig.savefig('depthzonezerocurtainplot.png',dpi=300)

def plot_circumpolar_duration(df):
    """
    Create circumpolar plot of zero curtain duration
    """
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    
    # Set up the map
    ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, color='black', linewidth=0.75)
    ax.gridlines()
    
    # Create scatter plot
    scatter = ax.scatter(df['longitude'], df['latitude'], 
                        c=df['duration_hours'],
                        cmap='viridis',
                        #vmin=6,
                        #vmax=120,
                        norm=LogNorm(vmin=1, vmax=200),
                        #norm=LogNorm(vmin=10, vmax=100),
                        transform=ccrs.PlateCarree(), 
                        s=40, 
                        alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, extend='both')
    cbar.set_label('Duration (hours)')#, size=12)
    #cbar.ax.tick_params(labelsize=10)
    
    plt.title('Circumpolar Distribution of Zero Curtain Events')#, size=14, pad=20)
    
    plt.tight_layout()
    
    return fig
    
fig = plot_circumpolar_duration(processor2_df_with_coords) #1-100

fig.savefig('circumarcticdurationevents_zerocurtainplot.png',dpi=300)

def identify_exemplar_sites(data):
    """
    Identify exemplar sites based on multiple criteria:
    1. Multiple depth zones represented
    2. Good temporal coverage (multiple years/seasons)
    3. Sufficient number of zero curtain events
    4. Range of depths monitored
    """
    # Calculate site-level metrics
    site_metrics = data.groupby('site_id').agg({
        'depth': ['nunique', 'min', 'max'],
        'year': 'nunique',
        'duration_days': 'count',
        'season': 'nunique',
        'depth_zone': lambda x: len(x.unique()),
        'site_name': 'first',
        'latitude': 'first',
        'longitude': 'first'
    })
    
    # Flatten column names
    site_metrics.columns = ['n_depths', 'min_depth', 'max_depth', 'n_years',
                          'n_events', 'n_seasons', 'n_zones', 'site_name',
                          'latitude', 'longitude']
    
    # Calculate depth range
    site_metrics['depth_range'] = site_metrics['max_depth'] - site_metrics['min_depth']
    
    # Define minimum criteria for exemplar sites
    min_criteria = {
        'n_depths': 3,      # At least 3 different depths
        'n_years': 2,       # At least 2 years of data
        'n_events': 20,     # At least 20 zero curtain events
        'n_seasons': 2,     # At least 2 seasons represented
        'n_zones': 2        # At least 2 depth zones
    }
    
    # Filter for sites meeting all criteria
    exemplar_mask = (
        (site_metrics['n_depths'] >= min_criteria['n_depths']) &
        (site_metrics['n_years'] >= min_criteria['n_years']) &
        (site_metrics['n_events'] >= min_criteria['n_events']) &
        (site_metrics['n_seasons'] >= min_criteria['n_seasons']) &
        (site_metrics['n_zones'] >= min_criteria['n_zones'])
    )
    
    exemplar_sites = site_metrics[exemplar_mask].copy()
    
    # Calculate a composite score for ranking
    # Normalize each metric to 0-1 range and sum
    for col in ['n_depths', 'n_years', 'n_events', 'n_seasons', 'n_zones', 'depth_range']:
        min_val = site_metrics[col].min()
        max_val = site_metrics[col].max()
        exemplar_sites[f'{col}_score'] = (exemplar_sites[col] - min_val) / (max_val - min_val)
    
    exemplar_sites['total_score'] = exemplar_sites[[col + '_score' for col in 
        ['n_depths', 'n_years', 'n_events', 'n_seasons', 'n_zones', 'depth_range']]].sum(axis=1)
    
    # Sort by total score
    exemplar_sites = exemplar_sites.sort_values('total_score', ascending=False)
    
    # Print summary of findings
    print(f"\nFound {len(exemplar_sites)} exemplar sites meeting minimum criteria:")
    print(f"- At least {min_criteria['n_depths']} depths monitored")
    print(f"- At least {min_criteria['n_years']} years of data")
    print(f"- At least {min_criteria['n_events']} zero curtain events")
    print(f"- At least {min_criteria['n_seasons']} seasons represented")
    print(f"- At least {min_criteria['n_zones']} depth zones")
    
    if len(exemplar_sites) > 0:
        print("\nTop exemplar sites:")
        for idx, site in exemplar_sites.head().iterrows():
            print(f"\nSite ID: {idx}")
            print(f"Site Name: {site['site_name']}")
            print(f"Location: {site['latitude']:.3f}°N, {site['longitude']:.3f}°E")
            print(f"Depths: {site['n_depths']} (range: {site['depth_range']:.1f}m)")
            print(f"Coverage: {site['n_years']} years, {site['n_seasons']} seasons")
            print(f"Events: {site['n_events']}")
            print(f"Score: {site['total_score']:.2f}")
    
        return exemplar_sites.index.tolist()
    else:
        print("No sites met the minimum criteria for exemplar status.")
        return None

exemplar_site_ids = identify_exemplar_sites(processor2_df_with_coords)

processor2_df_with_coords[processor2_df_with_coords['site_id']==exemplar_site_ids[0]]

if exemplar_site_ids is not None:
    plot_multiple_sites(processor2_df_with_coords, exemplar_site_ids[:3])

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_site_metrics(data, site_id):
    """Calculate comprehensive metrics for a single site"""
    site_data = data[data['site_id'] == site_id].copy()
    
    # Basic coverage metrics
    metrics = {
        'site_id': site_id,
        'site_name': site_data['site_name'].iloc[0],
        'latitude': site_data['latitude'].iloc[0],
        'longitude': site_data['longitude'].iloc[0],
        'n_depths': site_data['depth'].nunique(),
        'depth_range': site_data['depth'].max() - site_data['depth'].min(),
        'n_years': site_data['year'].nunique(),
        'year_range': site_data['year'].max() - site_data['year'].min(),
        'n_seasons': site_data['season'].nunique(),
        'n_zones': site_data['depth_zone'].nunique(),
        'total_events': len(site_data)
    }
    
    # Temporal patterns
    yearly_stats = site_data.groupby('year').agg({
        'duration_days': ['mean', 'std', 'count'],
        'mean_temp': ['mean', 'std']
    }).reset_index()
    
    if len(yearly_stats) >= 3:
        # Calculate trends
        years = yearly_stats['year'].values
        durations = yearly_stats['duration_days']['mean'].values
        temps = yearly_stats['mean_temp']['mean'].values
        
        duration_trend = stats.linregress(years, durations)
        temp_trend = stats.linregress(years, temps)
        
        metrics.update({
            'duration_trend': duration_trend.slope,
            'duration_trend_p': duration_trend.pvalue,
            'temp_trend': temp_trend.slope,
            'temp_trend_p': temp_trend.pvalue
        })
    
    # Depth zone patterns
    zone_stats = site_data.groupby('depth_zone').agg({
        'duration_days': ['mean', 'std', 'count'],
        'mean_temp': ['mean', 'std']
    }).reset_index()
    
    # Calculate depth zone correlations
    if len(zone_stats) >= 2:
        zone_corr = site_data.pivot_table(
            index='year',
            columns='depth_zone',
            values='duration_days',
            aggfunc='mean'
        ).corr()
        
        metrics['zone_correlation'] = zone_corr.mean().mean()
    
    # Seasonal patterns
    season_stats = site_data.groupby('season').agg({
        'duration_days': ['mean', 'std', 'count'],
        'mean_temp': ['mean', 'std']
    }).reset_index()
    
    # Calculate seasonal strength
    if len(season_stats) >= 2:
        f_stat, p_value = stats.f_oneway(
            *[group['duration_days'].values 
              for name, group in site_data.groupby('season')]
        )
        metrics['seasonal_f_stat'] = f_stat
        metrics['seasonal_p_value'] = p_value
    
    return metrics

def identify_exemplar_sites(data, min_criteria=None):
    """
    Identify exemplar sites based on comprehensive criteria
    
    Parameters:
    -----------
    data : pd.DataFrame
        Zero curtain metrics dataset
    min_criteria : dict, optional
        Minimum criteria for site selection
        
    Returns:
    --------
    pd.DataFrame
        Detailed metrics for exemplar sites
    """
    if min_criteria is None:
        min_criteria = {
            'n_depths': 3,
            'n_years': 2,
            'n_seasons': 2,
            'n_zones': 2,
            'total_events': 20
        }
    
    # Calculate metrics for all sites
    site_metrics = []
    for site_id in data['site_id'].unique():
        metrics = calculate_site_metrics(data, site_id)
        site_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(site_metrics)
    
    # Filter sites meeting minimum criteria
    mask = (
        (metrics_df['n_depths'] >= min_criteria['n_depths']) &
        (metrics_df['n_years'] >= min_criteria['n_years']) &
        (metrics_df['n_seasons'] >= min_criteria['n_seasons']) &
        (metrics_df['n_zones'] >= min_criteria['n_zones']) &
        (metrics_df['total_events'] >= min_criteria['total_events'])
    )
    
    exemplar_sites = metrics_df[mask].copy()
    
    if len(exemplar_sites) == 0:
        print("No sites met the minimum criteria.")
        return None
    
    # Calculate composite scores for different aspects
    # Coverage score
    for col in ['n_depths', 'n_years', 'n_seasons', 'n_zones', 'total_events']:
        exemplar_sites[f'{col}_score'] = (
            exemplar_sites[col] - exemplar_sites[col].min()
        ) / (exemplar_sites[col].max() - exemplar_sites[col].min())
    
    exemplar_sites['coverage_score'] = exemplar_sites[[
        'n_depths_score', 'n_years_score', 'n_seasons_score',
        'n_zones_score', 'total_events_score'
    ]].mean(axis=1)
    
    # Trend significance score (if available)
    if 'duration_trend_p' in exemplar_sites.columns:
        exemplar_sites['trend_score'] = (
            (1 - exemplar_sites['duration_trend_p']) + 
            (1 - exemplar_sites['temp_trend_p'])
        ) / 2
    
    # Seasonal strength score (if available)
    if 'seasonal_p_value' in exemplar_sites.columns:
        exemplar_sites['seasonal_score'] = 1 - exemplar_sites['seasonal_p_value']
    
    # Calculate final composite score
    score_columns = ['coverage_score']
    if 'trend_score' in exemplar_sites.columns:
        score_columns.append('trend_score')
    if 'seasonal_score' in exemplar_sites.columns:
        score_columns.append('seasonal_score')
    
    exemplar_sites['total_score'] = exemplar_sites[score_columns].mean(axis=1)
    
    # Sort by total score
    exemplar_sites = exemplar_sites.sort_values('total_score', ascending=False)
    
    # Print summary
    print(f"\nFound {len(exemplar_sites)} exemplar sites meeting criteria:")
    print(f"Minimum criteria:")
    for criterion, value in min_criteria.items():
        print(f"- {criterion}: {value}")
    
    print("\nTop exemplar sites:")
    for idx, site in exemplar_sites.head().iterrows():
        print(f"\nSite: {site['site_id']}")
        print(f"Location: {site['latitude']:.3f}°N, {site['longitude']:.3f}°E")
        print(f"Coverage: {site['n_years']} years, {site['n_seasons']} seasons")
        print(f"Depths: {site['n_depths']} (range: {site['depth_range']:.1f}m)")
        
        if 'duration_trend' in site:
            print(f"Duration trend: {site['duration_trend']:.3f} days/year (p={site['duration_trend_p']:.3f})")
        if 'temp_trend' in site:
            print(f"Temperature trend: {site['temp_trend']:.3f}°C/year (p={site['temp_trend_p']:.3f})")
        if 'seasonal_p_value' in site:
            print(f"Seasonal strength: p={site['seasonal_p_value']:.3f}")
        
        print(f"Total score: {site['total_score']:.3f}")
    
    return exemplar_sites

def analyze_zero_curtain_patterns(data, exemplar_sites):
    """
    Perform detailed analysis of zero curtain patterns for exemplar sites
    
    Parameters:
    -----------
    data : pd.DataFrame
        Zero curtain metrics dataset
    exemplar_sites : pd.DataFrame
        Identified exemplar sites with metrics
        
    Returns:
    --------
    dict
        Detailed analysis results
    """
    results = {}
    
    for _, site in exemplar_sites.head().iterrows():
        site_id = site['site_id']
        site_data = data[data['site_id'] == site_id].copy()
        
        # Temporal analysis
        temporal = {}
        
        # Year-to-year persistence
        yearly_means = site_data.groupby('year')['duration_days'].mean()
        temporal['year_to_year_corr'] = yearly_means.autocorr()
        
        # Seasonal timing analysis
        season_timing = site_data.groupby(['year', 'season'])['duration_days'].mean()
        season_timing = season_timing.unstack()
        
        if not season_timing.empty:
            temporal['season_correlations'] = season_timing.corr()
        
        # Depth zone analysis
        spatial = {}
        
        # Depth zone relationships
        zone_means = site_data.groupby(['year', 'depth_zone'])['duration_days'].mean()
        zone_means = zone_means.unstack()
        
        if not zone_means.empty:
            spatial['zone_correlations'] = zone_means.corr()
        
        # Temperature-duration relationship
        temp_duration_corr = stats.pearsonr(
            site_data['mean_temp'],
            site_data['duration_days']
        )
        spatial['temp_duration_correlation'] = temp_duration_corr[0]
        spatial['temp_duration_p_value'] = temp_duration_corr[1]
        
        results[site_id] = {
            'temporal_patterns': temporal,
            'spatial_patterns': spatial
        }
    
    return results

# Example usage
def run_complete_analysis(data, min_criteria=None):
    """Run complete zero curtain analysis pipeline"""
    
    print("1. Identifying exemplar sites...")
    exemplar_sites = identify_exemplar_sites(data, min_criteria)
    
    if exemplar_sites is not None:
        print("\n2. Analyzing zero curtain patterns...")
        analysis_results = analyze_zero_curtain_patterns(data, exemplar_sites)
        
        print("\n3. Pattern Analysis Summary:")
        for site_id, results in analysis_results.items():
            site_name = exemplar_sites[exemplar_sites['site_id'] == site_id]['site_name'].iloc[0]
            print(f"\nSite: {site_id}")
            
            temporal = results['temporal_patterns']
            print("Temporal Patterns:")
            print(f"- Year-to-year correlation: {temporal.get('year_to_year_corr', 'N/A'):.3f}")
            
            spatial = results['spatial_patterns']
            print("Spatial Patterns:")
            print(f"- Temperature-duration correlation: {spatial['temp_duration_correlation']:.3f} (p={spatial['temp_duration_p_value']:.3f})")
        
        return exemplar_sites, analysis_results
    
    return None, None
    
exemplar_sites, analysis_results = run_complete_analysis(processor2_df_with_coords)

if exemplar_sites is not None:
    print("\n4. Creating visualizations...")
    for _, site in exemplar_sites.head(3).iterrows():
        filename = plot_site(processor2_df_with_coords, site['site_id'])
        print(f"Saved plot: {filename}")

from matplotlib.gridspec import GridSpec

def plot_site_evolution_detailed(data, site_id):
    """
    Create detailed visualization of zero curtain evolution for a site
    
    Parameters:
    -----------
    data : pd.DataFrame
        Full zero curtain metrics dataset
    site_id : str
        Site identifier
    """
    # Get site data
    site_data = data[data['site_id'] == site_id].copy()
    
    if len(site_data) == 0:
        print(f"No data found for site: {site_id}")
        return None
    
    # Convert dates and sort
    site_data['start_date'] = pd.to_datetime(site_data['start_date'])
    site_data['end_date'] = pd.to_datetime(site_data['end_date'])
    site_data = site_data.sort_values('start_date')
    
    # Get site location
    site_lat = site_data['latitude'].iloc[0]
    site_lon = site_data['longitude'].iloc[0]
    site_name = site_data['site_name'].iloc[0]
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(4, 1, height_ratios=[3, 2, 2, 2], hspace=0.3)
    
    # 1. Duration and temperature by depth
    ax1 = fig.add_subplot(gs[0])
    scatter = ax1.scatter(site_data['depth'], 
                         site_data['duration_days'],
                         c=site_data['mean_temp'],
                         cmap='RdBu_r',
                         s=100,
                         alpha=0.7)
    
    # Add season markers
    for season in site_data['season'].unique():
        season_data = site_data[site_data['season'] == season]
        ax1.scatter([], [], marker='o', label=season)
    
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
    cbar.set_label('Mean Temperature (°C)', fontsize=10)
    ax1.set_xlabel('Depth (m)')
    ax1.set_ylabel('Duration (days)')
    ax1.set_title(f'Zero Curtain Evolution\n{site_name}\n({site_lat:.3f}°N, {site_lon:.3f}°E)')
    ax1.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Annual occurrence by depth zone
    ax2 = fig.add_subplot(gs[1])
    yearly_counts = site_data.groupby(['year', 'depth_zone']).size().unstack(fill_value=0)
    yearly_counts.plot(kind='bar', ax=ax2)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Number of Events')
    ax2.set_title('Annual Zero Curtain Events by Depth Zone')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    ax2.legend(title='Depth Zone', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Seasonal timing by depth zone
    ax3 = fig.add_subplot(gs[2])
    site_data['month'] = pd.to_datetime(site_data['start_date']).dt.month
    sns.boxplot(data=site_data, x='depth_zone', y='month', ax=ax3)
    ax3.set_ylabel('Month')
    ax3.set_title('Seasonal Timing by Depth Zone')
    
    # 4. Duration trends
    ax4 = fig.add_subplot(gs[3])
    for zone in site_data['depth_zone'].unique():
        zone_data = site_data[site_data['depth_zone'] == zone]
        yearly_mean = zone_data.groupby('year')['duration_days'].mean()
        
        # Plot mean duration
        ax4.plot(yearly_mean.index, yearly_mean.values, 'o-', 
                label=zone.replace('_', ' ').title(),
                markersize=5)
        
        # Add trend line if enough points
        if len(yearly_mean) > 2:
            z = np.polyfit(yearly_mean.index, yearly_mean.values, 1)
            p = np.poly1d(z)
            ax4.plot(yearly_mean.index, 
                    p(yearly_mean.index), 
                    "--", 
                    alpha=0.5)
    
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Mean Duration (days)')
    ax4.set_title('Duration Trends by Depth Zone')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    ax4.legend(title='Depth Zone', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Calculate site statistics
    stats = {
        'record_length_years': site_data['year'].nunique(),
        'total_events': len(site_data),
        'n_depths': site_data['depth'].nunique(),
        'depth_range': site_data['depth'].max() - site_data['depth'].min(),
        'mean_duration': site_data['duration_days'].mean(),
        'depth_zones': site_data['depth_zone'].nunique(),
        'seasons': site_data['season'].nunique()
    }
    
    # Calculate trends for each depth zone
    trends = {}
    for zone in site_data['depth_zone'].unique():
        zone_data = site_data[site_data['depth_zone'] == zone]
        yearly_mean = zone_data.groupby('year')['duration_days'].mean()
        
        if len(yearly_mean) > 2:
            z = np.polyfit(yearly_mean.index, yearly_mean.values, 1)
            trends[zone] = z[0]
    
    print(f"\nSite Statistics: {site_id}")
    print(f"Location: {site_lat:.3f}°N, {site_lon:.3f}°E")
    print(f"Record length: {stats['record_length_years']} years")
    print(f"Total events: {stats['total_events']}")
    print(f"Number of depths: {stats['n_depths']} (range: {stats['depth_range']:.1f}m)")
    print(f"Mean duration: {stats['mean_duration']:.1f} days")
    print(f"Depth zones: {stats['depth_zones']}")
    print(f"Seasons: {stats['seasons']}")
    
    print("\nDuration trends (days/year):")
    for zone, trend in trends.items():
        print(f"{zone}: {trend:.3f}")
    
    plt.tight_layout()
    return fig, stats, trends

def create_detailed_site_analysis(data, exemplar_sites):
    """Run detailed analysis for exemplar sites"""
    output_dir = "detailed_site_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, site in exemplar_sites.head(3).iterrows():
        print(f"\nAnalyzing site: {site['site_id']}")
        fig, stats, trends = plot_site_evolution_detailed(data, site['site_id'])
        
        if fig is not None:
            filename = os.path.join(output_dir, f"detailed_{site['site_id'].split('-')[1][:30]}.png")
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved detailed analysis to: {filename}")

if exemplar_sites is not None:
    create_detailed_site_analysis(processor2_df_with_coords, exemplar_sites)

exemplar_sites.to_csv('exemplar_sites.csv',index=False)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import gridspec
from pathlib import Path

def analyze_zc_alt_dynamics(data, output_path):
    """
    Analyze zero curtain dynamics with improved handling of temporal data
    """
    # Calculate trends for locations with sufficient temporal variation
    def calc_location_trend(group):
        if len(group) > 2 and len(group['year'].unique()) > 1:
            try:
                return scipy_stats.linregress(group['year'], group['duration_hours']).slope
            except:
                return np.nan
        return np.nan
    
    # Calculate location trends
    location_trends = data.groupby(['latitude', 'longitude']).apply(
        calc_location_trend
    ).reset_index(name='trend')
    
    # Filter out NaN trends for visualization
    location_trends = location_trends.dropna(subset=['trend'])
    
    # Rest of the visualization code remains the same...
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # 1. Spatial Evolution Map
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.NorthPolarStereo())
    ax1.add_feature(cfeature.LAND, facecolor='white', alpha=0.3)
    ax1.add_feature(cfeature.OCEAN, alpha=0.3)
    ax1.add_feature(cfeature.COASTLINE, color='black')
    
    if len(location_trends) > 0:
        scatter = ax1.scatter(location_trends['longitude'],
                            location_trends['latitude'],
                            c=location_trends['trend'],
                            transform=ccrs.PlateCarree(),
                            cmap='RdBu_r',
                            s=100,
                            alpha=0.7)
        
        plt.colorbar(scatter, ax=ax1, label='Trend in Duration (hours/year)')
    
    ax1.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    ax1.set_title('Spatial Distribution of Zero Curtain Trends')
    
    # 2. Temporal Evolution (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    annual_means = data.groupby('year').agg({
        'duration_hours': ['mean', 'std'],
        'depth': ['mean', 'std']
    }).reset_index()
    annual_means.columns = ['year', 'duration_mean', 'duration_std', 
                          'depth_mean', 'depth_std']
    
    # Plot duration with error bars
    color1 = 'tab:blue'
    ax2.errorbar(annual_means['year'], 
                annual_means['duration_mean'],
                yerr=annual_means['duration_std'],
                color=color1, 
                marker='o',
                label='Duration')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Mean Duration (hours)', color=color1)
    ax2.tick_params(axis='y', labelcolor=color1)
    
    # Plot depth on secondary axis with error bars
    ax2_twin = ax2.twinx()
    color2 = 'tab:red'
    ax2_twin.errorbar(annual_means['year'],
                     annual_means['depth_mean'],
                     yerr=annual_means['depth_std'],
                     color=color2,
                     marker='s',
                     label='Depth')
    ax2_twin.set_ylabel('Mean Depth', color=color2)
    ax2_twin.tick_params(axis='y', labelcolor=color2)
    
    # Add legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax2.set_title('Temporal Evolution of Duration and Depth')
    
    # 3. Depth vs Duration Relationship
    ax3 = fig.add_subplot(gs[1, 0])
    valid_data = data.dropna(subset=['depth', 'duration_hours'])
    
    sns.scatterplot(data=valid_data, 
                   x='depth',
                   y='duration_hours',
                   hue='season',
                   alpha=0.5,
                   ax=ax3)
    
    if len(valid_data) > 1:
        z = np.polyfit(valid_data['depth'], valid_data['duration_hours'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(valid_data['depth'].min(), valid_data['depth'].max(), 100)
        ax3.plot(x_range, p(x_range), "r--", alpha=0.8,
                 label=f'Trend: {z[0]:.1f} hours/m')
        ax3.legend(title='Season')
    
    ax3.set_xlabel('Depth (m)')
    ax3.set_ylabel('Duration (hours)')
    ax3.set_title('Zero Curtain Duration vs Depth')
    
    # 4. Seasonal Patterns
    ax4 = fig.add_subplot(gs[1, 1])
    seasonal_depth = data.groupby(['season', 'depth_zone']).agg({
        'duration_hours': ['mean', 'std']
    }).reset_index()
    seasonal_depth.columns = ['season', 'depth_zone', 'mean_duration', 'std_duration']
    
    sns.barplot(data=seasonal_depth,
               x='season',
               y='mean_duration',
               hue='depth_zone',
               ax=ax4)
    
    ax4.set_xlabel('Season')
    ax4.set_ylabel('Mean Duration (hours)')
    ax4.set_title('Seasonal Duration by Depth Zone')
    plt.xticks(rotation=45)
    
    # Calculate statistics
    stats = {
        'temporal_trend': scipy_stats.linregress(annual_means['year'], 
                                               annual_means['duration_mean'])
        if len(annual_means) > 2 else None,
        'depth_correlation': scipy_stats.pearsonr(valid_data['depth'], 
                                                valid_data['duration_hours'])
        if len(valid_data) > 2 else None,
        'seasonal_variation': data.groupby('season')['duration_hours'].agg([
            'mean', 'std', 'count']).to_dict(),
        'spatial_summary': {
            'n_sites': len(location_trends),
            'sites_increasing': (location_trends['trend'] > 0).sum(),
            'sites_decreasing': (location_trends['trend'] < 0).sum()
        }
    }
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(Path(output_path) / 'zc_alt_dynamics.png', 
                dpi=300, 
                bbox_inches='tight')
    
    return fig, stats

fig, stats = analyze_zc_alt_dynamics(processor2_df_with_coords, '/Users/[USER]/Desktop/Research/Code')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_temporal_evolution(data, output_path):
    """
    Create a standalone temporal evolution plot of zero curtain dynamics
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing zero curtain events data
    output_path : Path
        Path for saving outputs
    """
    # Calculate annual statistics
    annual_means = data.groupby('year').agg({
        'duration_hours': ['mean', 'std'],
        'depth': ['mean', 'std']
    }).reset_index()
    annual_means.columns = ['year', 'duration_mean', 'duration_std', 
                          'depth_mean', 'depth_std']
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot duration with error bars
    color1 = '#1f77b4'  # A deeper blue
    line1 = ax1.errorbar(annual_means['year'], 
                        annual_means['duration_mean'],
                        yerr=annual_means['duration_std'],
                        color=color1,
                        marker='o',
                        markersize=6,
                        linewidth=2,
                        capsize=4,
                        capthick=1.5,
                        label='Duration')
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Mean Duration (hours)', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Plot depth on secondary axis with error bars
    ax2 = ax1.twinx()
    color2 = '#d62728'  # A deeper red
    line2 = ax2.errorbar(annual_means['year'],
                        annual_means['depth_mean'],
                        yerr=annual_means['depth_std'],
                        color=color2,
                        marker='s',
                        markersize=6,
                        linewidth=2,
                        capsize=4,
                        capthick=1.5,
                        label='Depth')
    
    ax2.set_ylabel('Mean Depth (m)', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add legend
    lines = [line1, line2]
    ax1.legend(lines, [l.get_label() for l in lines], 
              loc='upper right', 
              fontsize=10)
    
    # Set title
    plt.title('Temporal Evolution of Zero Curtain Duration and Depth', 
              fontsize=14, 
              pad=20)
    
    # Customize grid
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(Path(output_path) / 'temporal_evolution.png',
                dpi=300,
                bbox_inches='tight')
    
    return fig

# Create output directory
output_path = Path('/Users/[USER]/Desktop/Research/Code')
output_path.mkdir(exist_ok=True, parents=True)

# Generate the plot
fig = plot_temporal_evolution(processor2_df_with_coords, output_path)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

def analyze_seasonal_evolution(data, output_path):
    """
    Analyze the evolution of zero curtain characteristics across seasons
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing zero curtain events
    output_path : Path
        Path for saving outputs
    """
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    gs = plt.GridSpec(3, 2)
    
    # 1. Duration by Season Over Time (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    seasonal_duration = data.groupby(['year', 'season'])['duration_hours'].mean().unstack()
    
    for season in seasonal_duration.columns:
        ax1.plot(seasonal_duration.index, seasonal_duration[season], 
                marker='o', label=season)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Mean Duration (hours)')
    ax1.set_title('Evolution of Zero Curtain Duration by Season')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Seasonal Distribution of Events (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    early_year = data['year'].min()
    late_year = data['year'].max()
    mid_year = early_year + (late_year - early_year) // 2
    
    periods = {
        'Early': (early_year, mid_year),
        'Late': (mid_year + 1, late_year)
    }
    
    for period_name, (start_year, end_year) in periods.items():
        period_data = data[(data['year'] >= start_year) & 
                          (data['year'] <= end_year)]
        counts = period_data['season'].value_counts()
        ax2.bar(counts.index, counts.values, 
                alpha=0.5, label=f'{period_name} ({start_year}-{end_year})')
    
    ax2.set_ylabel('Number of Events')
    ax2.set_title('Distribution of Zero Curtain Events by Season')
    ax2.legend()
    
    # 3. Depth Distribution by Season Over Time (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    seasonal_depth = data.groupby(['year', 'season'])['depth'].mean().unstack()
    
    for season in seasonal_depth.columns:
        ax3.plot(seasonal_depth.index, seasonal_depth[season], 
                marker='o', label=season)
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Mean Depth (m)')
    ax3.set_title('Evolution of Zero Curtain Depth by Season')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Temperature Distribution by Season (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    sns.boxplot(data=data, x='season', y='mean_temp', ax=ax4)
    ax4.set_xlabel('Season')
    ax4.set_ylabel('Mean Temperature (°C)')
    ax4.set_title('Temperature Distribution During Zero Curtain by Season')
    
    # 5. Duration-Depth Relationship by Season (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    
    for season in data['season'].unique():
        season_data = data[data['season'] == season]
        sns.regplot(data=season_data, 
                   x='depth', 
                   y='duration_hours',
                   label=season,
                   scatter_kws={'alpha': 0.3},
                   ax=ax5)
    
    ax5.set_xlabel('Depth (m)')
    ax5.set_ylabel('Duration (hours)')
    ax5.set_title('Zero Curtain Duration vs Depth by Season')
    ax5.legend()
    
    # Calculate statistics
    stats_dict = {}
    
    # Trend analysis for each season
    for season in data['season'].unique():
        season_data = data[data['season'] == season]
        season_trends = season_data.groupby('year')['duration_hours'].mean()
        
        if len(season_trends) > 2:
            trend = stats.linregress(season_trends.index, season_trends.values)
            stats_dict[season] = {
                'duration_trend': trend.slope,
                'trend_pvalue': trend.pvalue,
                'mean_duration': season_data['duration_hours'].mean(),
                'std_duration': season_data['duration_hours'].std(),
                'mean_depth': season_data['depth'].mean(),
                'depth_correlation': stats.pearsonr(
                    season_data['depth'].dropna(),
                    season_data['duration_hours'].dropna()
                )[0]
            }
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(Path(output_path) / 'seasonal_evolution.png',
                dpi=300,
                bbox_inches='tight')
    
    return fig, stats_dict

output_path = Path('/Users/[USER]/Desktop/Research/Code')
output_path.mkdir(exist_ok=True, parents=True)
fig, stats = analyze_seasonal_evolution(processor2_df_with_coords, output_path)

print("\nSeasonal Analysis Summary:")
for season, stats_dict in stats.items():
    print(f"\n{season}:")
    print(f"Duration Trend: {stats_dict['duration_trend']:.2f} hours/year")
    print(f"Trend P-value: {stats_dict['trend_pvalue']:.3e}")
    print(f"Mean Duration: {stats_dict['mean_duration']:.1f} hours")
    print(f"Mean Depth: {stats_dict['mean_depth']:.2f} m")
    print(f"Depth-Duration Correlation: {stats_dict['depth_correlation']:.3f}")

def analyze_feature_importance(model, output_dir, X_file, train_indices, feature_names):
    """Analyze feature importance using permutation importance"""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance
    import os
    
    # Load a small subset of data for feature importance
    X_mmap = np.load(X_file, mmap_mode='r')
    
    # Use a manageable sample size (10,000 samples or fewer)
    sample_size = min(10000, len(train_indices))
    sample_indices = np.random.choice(train_indices, size=sample_size, replace=False)
    
    # Load sample data
    X_sample = np.array([X_mmap[idx] for idx in sample_indices])
    y_sample = np.array([y_mmap[idx] for idx in sample_indices])
    
    print(f"Computing permutation importance on {sample_size} samples...")
    
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_sample, y_sample, 
                                            n_repeats=5, random_state=42, 
                                            n_jobs=-1)
    
    # Sort features by importance
    feature_importance = perm_importance.importances_mean
    sorted_indices = np.argsort(feature_importance)[::-1]
    
    # Get feature names if provided, otherwise use indices
    if feature_names is None:
        if hasattr(X_sample, 'shape') and len(X_sample.shape) > 1:
            feature_names = [f"Feature {i}" for i in range(X_sample.shape[1])]
        else:
            feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh([feature_names[i] for i in sorted_indices], 
             feature_importance[sorted_indices])
    plt.xlabel('Permutation Importance')
    plt.title('Feature Importance Analysis')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.show()
    
    # Save feature importance to CSV
    import pandas as pd
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in sorted_indices],
        'Importance': feature_importance[sorted_indices]
    })
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    return importance_df

def configure_tensorflow_memory():
    """Configure TensorFlow to use memory growth and handle device allocation carefully"""
    import tensorflow as tf
    
    # Disable GPU if experiencing persistent issues
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Get available physical devices
    physical_devices = tf.config.list_physical_devices('GPU')
    
    try:
        # Attempt to set memory growth for all GPUs
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Memory growth enabled for {device}")
            except Exception as e:
                print(f"Could not set memory growth for {device}: {e}")
        
        # Explicit device placement settings
        tf.config.set_soft_device_placement(True)
        
        # Optional: Limit GPU memory to prevent out-of-memory errors
        for device in physical_devices:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    device,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 3)]  # 3GB
                )
                print(f"Memory limit set for {device}")
            except Exception as e:
                print(f"Could not set memory limit for {device}: {e}")
    
    except Exception as global_e:
        print(f"Error configuring GPU memory: {global_e}")
    
    # Log available devices
    print("Available devices:")
    print(tf.config.list_physical_devices())

def configure_tensorflow_memory():
    """Configure TensorFlow to use memory growth and limit GPU memory allocation"""
    import tensorflow as tf
    
    # Only configure GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                # Allow memory growth - prevents TF from allocating all GPU memory at once
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Memory growth enabled for {device}")
            except Exception as e:
                print(f"Error configuring GPU: {e}")
    
    # Set soft device placement if possible
    try:
        tf.config.set_soft_device_placement(True)
    except:
        pass

from tensorflow.keras.layers import Layer, BatchNormalization

# 1. Define the custom layer
class BatchNorm5D(Layer):
    def __init__(self, **kwargs):
        super(BatchNorm5D, self).__init__(**kwargs)
        self.bn = BatchNormalization()
        
    def call(self, inputs, training=None):
        # Get the dimensions
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        time_steps = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        channels = input_shape[4]
        
        # Reshape to 4D for BatchNorm by combining batch and time dimensions
        x_reshaped = tf.reshape(inputs, [-1, height, width, channels])
        
        # Apply BatchNorm
        x_bn = self.bn(x_reshaped, training=training)
        
        # Reshape back to 5D
        x_back = tf.reshape(x_bn, [batch_size, time_steps, height, width, channels])
        return x_back
    
    def get_config(self):
        config = super(BatchNorm5D, self).get_config()
        return config

# 2. Register the custom layer
tf.keras.utils.get_custom_objects().update({'BatchNorm5D': BatchNorm5D})

def build_improved_zero_curtain_model_fixed(input_shape, include_moisture=True):
    """
    Fixed model architecture that works with the ConvLSTM2D and BatchNormalization
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Dropout
    from tensorflow.keras.layers import Reshape, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, Lambda
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    
    # Create a proper custom layer instead of Lambda
    class BatchNorm5D(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(BatchNorm5D, self).__init__(**kwargs)
            self.bn = BatchNormalization()
            
        def call(self, inputs, training=None):
            # Get the dimensions
            input_shape = tf.shape(inputs)
            batch_size = input_shape[0]
            time_steps = input_shape[1]
            height = input_shape[2]
            width = input_shape[3]
            channels = input_shape[4]
            
            # Reshape to 4D for BatchNorm by combining batch and time dimensions
            x_reshaped = tf.reshape(inputs, [-1, height, width, channels])
            
            # Apply BatchNorm
            x_bn = self.bn(x_reshaped, training=training)
            
            # Reshape back to 5D
            x_back = tf.reshape(x_bn, [batch_size, time_steps, height, width, channels])
            return x_back
    
    # Reduced L2 regularization strength
    reg_strength = 0.00005
    # Slightly reduced dropout rate
    dropout_rate = 0.25
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Add BatchNormalization to input
    x = BatchNormalization()(inputs)
    
    # Reshape for ConvLSTM (add spatial dimension)
    x = Reshape((input_shape[0], 1, 1, input_shape[1]))(x)
    
    # ConvLSTM layer with regularization
    from tensorflow.keras.layers import ConvLSTM2D
    convlstm = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 1),
        padding='same',
        return_sequences=True,
        activation='tanh',
        recurrent_dropout=dropout_rate,
        kernel_regularizer=l2(reg_strength)
    )(x)
    
    # CRITICAL FIX: Use custom layer instead of Lambda
    convlstm = BatchNorm5D()(convlstm)
    
    # Reshape back to (sequence_length, features)
    convlstm = Reshape((input_shape[0], 64))(convlstm)
    
    # Add positional encoding for transformer
    def positional_encoding(length, depth):
        positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
        depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth
        
        angle_rates = 1 / tf.pow(10000.0, depths)
        angle_rads = positions * angle_rates
        
        # Only use sin to ensure output depth matches input depth
        pos_encoding = tf.sin(angle_rads)
        
        # Add batch dimension
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        
        return pos_encoding
    
    # Add positional encoding
    pos_encoding = positional_encoding(input_shape[0], 64)
    transformer_input = convlstm + pos_encoding
    
    # Add BatchNormalization before transformer
    transformer_input = BatchNormalization()(transformer_input)
    
    # Improved transformer encoder block
    def transformer_encoder(x, num_heads=8, key_dim=64, ff_dim=128):
        # Multi-head attention with regularization
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,
            kernel_regularizer=l2(reg_strength)
        )(x, x)
        
        # Skip connection 1 with dropout
        x1 = Add()([attention_output, x])
        x1 = LayerNormalization(epsilon=1e-6)(x1)
        # Add BatchNormalization after LayerNormalization
        x1 = BatchNormalization()(x1)
        x1 = Dropout(dropout_rate)(x1)
        
        # Feed-forward network with regularization
        ff_output = Dense(ff_dim, activation='relu', kernel_regularizer=l2(reg_strength))(x1)
        ff_output = BatchNormalization()(ff_output)  # Add BatchNorm
        ff_output = Dropout(dropout_rate)(ff_output)
        ff_output = Dense(64, kernel_regularizer=l2(reg_strength))(ff_output)
        ff_output = BatchNormalization()(ff_output)  # Add BatchNorm
        
        # Skip connection 2
        x2 = Add()([ff_output, x1])
        return LayerNormalization(epsilon=1e-6)(x2)
    
    # Apply transformer encoder
    transformer_output = transformer_encoder(transformer_input)
    
    # Parallel CNN paths for multi-scale feature extraction (with regularization)
    cnn_1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', 
                  kernel_regularizer=l2(reg_strength))(inputs)
    cnn_1 = BatchNormalization()(cnn_1)  
    cnn_1 = Dropout(dropout_rate/2)(cnn_1)
    
    cnn_2 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu',
                  kernel_regularizer=l2(reg_strength))(inputs)
    cnn_2 = BatchNormalization()(cnn_2)  
    cnn_2 = Dropout(dropout_rate/2)(cnn_2)
    
    cnn_3 = Conv1D(filters=32, kernel_size=7, padding='same', activation='relu',
                  kernel_regularizer=l2(reg_strength))(inputs)
    cnn_3 = BatchNormalization()(cnn_3)  
    cnn_3 = Dropout(dropout_rate/2)(cnn_3)
    
    # VAE components
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    # Global temporal features
    global_max = GlobalMaxPooling1D()(transformer_output)
    global_avg = GlobalAveragePooling1D()(transformer_output)
    
    # VAE encoding with reduced regularization
    z_concat = Concatenate()([global_max, global_avg])
    z_concat = BatchNormalization()(z_concat)  # Add BatchNorm
    
    z_mean = Dense(32, kernel_regularizer=l2(reg_strength))(z_concat)
    z_log_var = Dense(32, kernel_regularizer=l2(reg_strength))(z_concat)
    z = Lambda(sampling)([z_mean, z_log_var])
    
    # Combine all features
    merged_features = Concatenate()(
        [
            GlobalMaxPooling1D()(cnn_1),
            GlobalMaxPooling1D()(cnn_2),
            GlobalMaxPooling1D()(cnn_3),
            global_max,
            global_avg,
            z
        ]
    )
    
    # Add BatchNorm to merged features
    merged_features = BatchNormalization()(merged_features)
    
    # Final classification layers with regularization
    x = Dense(128, activation='relu', kernel_regularizer=l2(reg_strength))(merged_features)
    x = BatchNormalization()(x)  
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(reg_strength))(x)
    x = BatchNormalization()(x)  
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_strength))(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # CRITICAL FIX: Reduce VAE loss weight
    kl_loss = -0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
    )
    model.add_loss(0.0002 * kl_loss)  # Reduced from 0.0005 to 0.0002
    
    # Compile model with gradient clipping but simpler metrics to avoid errors
    model.compile(
        optimizer=Adam(
            learning_rate=0.0005,
            clipvalue=0.5,  # Reduced from 1.0 to 0.5 for better stability
            epsilon=1e-7  # Increased epsilon for better numerical stability
        ),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', num_thresholds=200),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

# Cyclical Learning Rate Callback
class CyclicLR(tf.keras.callbacks.Callback):
    """
    Cyclical learning rate callback for smoother training.
    """
    def __init__(
        self,
        base_lr=0.0001,
        max_lr=0.001,
        step_size=2000,
        mode='triangular2'
    ):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        
        if self.mode == 'triangular':
            self.scale_fn = lambda x: 1.
        elif self.mode == 'triangular2':
            self.scale_fn = lambda x: 1 / (2. ** (x - 1))
        elif self.mode == 'exp_range':
            self.scale_fn = lambda x: 0.9 ** x
        
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        
    def on_train_begin(self, logs=None):
        logs = logs or {}
        tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)
            
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())

# Enhanced callbacks setup
def get_enhanced_callbacks(output_dir, fold_idx=None):
    """
    Create enhanced callbacks with increased patience and cyclical learning rate.
    """
    sub_dir = f"fold_{fold_idx}" if fold_idx is not None else ""
    checkpoint_dir = os.path.join(output_dir, sub_dir, "checkpoints")
    tensorboard_dir = os.path.join(output_dir, sub_dir, "tensorboard")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    callbacks = [
        # Early stopping with increased patience
        tf.keras.callbacks.EarlyStopping(
            patience=20,  # Increased from 15 to 20
            restore_best_weights=True,
            monitor='val_auc',
            mode='max',
            min_delta=0.005
        ),
        # Reduced LR with increased patience
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=10,  # Increased from 7 to 10
            min_lr=1e-6,
            monitor='val_auc',
            mode='max'
        ),
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, "best_model.h5"),
            save_best_only=True,
            monitor='val_auc',
            mode='max'
        ),
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=False,
            update_freq='epoch'
        ),
        # Cyclical learning rate
        CyclicLR(
            base_lr=0.0001,
            max_lr=0.001,
            step_size=2000,
            mode='triangular2'
        ),
        # Memory cleanup
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect()
        )
    ]
    
    return callbacks

def process_chunk_with_batch_safety(model, chunk_X, chunk_y, val_data, batch_size=256, epochs=3, class_weight=None):
    """
    Process a chunk with batch-level safety to prevent stopping at specific batches.
    """
    import numpy as np
    
    history_list = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_history = {"loss": [], "accuracy": [], "auc": [], "precision": [], "recall": [], "lr": []}
        
        # Process in smaller mini-batches to isolate failures
        num_batches = len(chunk_X) // batch_size + (1 if len(chunk_X) % batch_size > 0 else 0)
        
        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, len(chunk_X))
            
            # Extract batch data
            batch_X = chunk_X[batch_start:batch_end]
            batch_y = chunk_y[batch_start:batch_end]
            
            try:
                # Train on single batch
                batch_history = model.fit(
                    batch_X, batch_y,
                    epochs=1,
                    batch_size=batch_end - batch_start,
                    class_weight=class_weight,
                    verbose=0  # Silent mode to reduce output
                )
                
                # Store metrics
                for key in epoch_history:
                    if key in batch_history.history:
                        epoch_history[key].append(batch_history.history[key][0])
            
            except Exception as e:
                print(f"Error processing batch {batch+1}/{num_batches}: {e}")
                import traceback
                traceback.print_exc()
                # Skip this batch and continue
                continue
            
            # Print progress every 10 batches
            if batch % 10 == 0 or batch == num_batches - 1:
                metrics_str = " - ".join([f"{k}: {np.mean(v):.4f}" for k, v in epoch_history.items() if v])
                print(f"Batch {batch+1}/{num_batches} - {metrics_str}")
        
        # Validate after each epoch
        try:
            val_X, val_y = val_data
            val_metrics = model.evaluate(val_X, val_y, verbose=0)
            val_dict = {f"val_{name}": value for name, value in zip(model.metrics_names, val_metrics)}
            
            # Add validation metrics to epoch history
            for k, v in val_dict.items():
                epoch_history[k] = [v]
                
            # Add current learning rate
            if hasattr(model.optimizer, 'lr'):
                import tensorflow as tf
                epoch_history['lr'] = [float(tf.keras.backend.get_value(model.optimizer.lr))]
                
            metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in val_dict.items()])
            print(f"Validation: {metrics_str}")
        except Exception as val_e:
            print(f"Error during validation: {val_e}")
            import traceback
            traceback.print_exc()
        
        # Add epoch history to overall history
        history_list.append(epoch_history)
    
    # Combine histories into a format similar to model.fit
    combined_history = {"history": {}}
    for key in ["loss", "accuracy", "auc", "precision", "recall", "val_loss", "val_accuracy", 
                "val_auc", "val_precision", "val_recall", "lr"]:
        combined_history["history"][key] = []
        for epoch_hist in history_list:
            if key in epoch_hist:
                if isinstance(epoch_hist[key], list):
                    combined_history["history"][key].extend(epoch_hist[key])
                else:
                    combined_history["history"][key].append(epoch_hist[key])
    
    return combined_history

def improved_resumable_training(model, X_file, y_file, train_indices, val_indices, test_indices,
                               output_dir, batch_size=256, chunk_size=20000, epochs_per_chunk=3, 
                               save_frequency=5, class_weight=None, start_chunk=0):
    """
    Enhanced training function with improved error handling.
    """
    import os
    import gc
    import json
    import numpy as np
    from datetime import datetime, timedelta
    import time
    import psutil
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tensorboard"), exist_ok=True)
    
    # Process in chunks
    num_chunks = int(np.ceil(len(train_indices) / chunk_size))
    print(f"Processing {len(train_indices)} samples in {num_chunks} chunks of {chunk_size}")
    print(f"Starting from chunk {start_chunk+1}")
    
    # Open data files with memory mapping
    X_mmap = np.load(X_file, mmap_mode='r')
    y_mmap = np.load(y_file, mmap_mode='r')
    
    # Create validation set once (limited size for memory efficiency)
    val_limit = min(1000, len(val_indices))
    val_indices_subset = val_indices[:val_limit]
    
    # Load validation data
    val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
    val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    print(f"Loaded {len(val_X)} validation samples")
    
    # Load existing history if resuming
    history_log = []
    history_path = os.path.join(output_dir, "training_history.json")
    if start_chunk > 0 and os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history_log = json.load(f)
        except Exception as e:
            print(f"Could not load existing history: {e}")
            # Try pickle format
            pickle_path = os.path.join(output_dir, "training_history.pkl")
            if os.path.exists(pickle_path):
                import pickle
                with open(pickle_path, "rb") as f:
                    history_log = pickle.load(f)
    
    # If resuming, load latest model
    if start_chunk > 0:
        # Find the most recent checkpoint before start_chunk
        checkpoint_indices = []
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        for filename in os.listdir(checkpoints_dir):
            if filename.startswith("model_chunk_") and filename.endswith(".h5"):
                try:
                    idx = int(filename.split("_")[-1].split(".")[0])
                    if idx < start_chunk:
                        checkpoint_indices.append(idx)
                except ValueError:
                    continue
        
        if checkpoint_indices:
            latest_idx = max(checkpoint_indices)
            model_path = os.path.join(checkpoints_dir, f"model_chunk_{latest_idx}.h5")
            if os.path.exists(model_path):
                print(f"Loading model from checkpoint {model_path}")
                model = tf.keras.models.load_model(model_path)
            else:
                print(f"Warning: Could not find model checkpoint for chunk {latest_idx}")
    
    # Setup callbacks
    callbacks = [
        # Early stopping with increased patience
        tf.keras.callbacks.EarlyStopping(
            patience=10, 
            restore_best_weights=True,
            monitor='val_auc',
            mode='max',
            min_delta=0.001
        ),
        # Reduced LR with increased patience
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            monitor='val_auc',
            mode='max'
        ),
        # Memory cleanup after each epoch
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect()
        )
    ]
    
    # Track metrics across chunks
    start_time = time.time()
    
    # For safe recovery
    recovery_file = os.path.join(output_dir, "last_completed_chunk.txt")
    
    # Process each chunk
    for chunk_idx in range(start_chunk, num_chunks):
        # Track progress in file for recovery purposes
        with open(os.path.join(output_dir, "current_progress.txt"), "w") as f:
            f.write(f"Processing chunk {chunk_idx+1}/{num_chunks}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Get chunk indices
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(train_indices))
        chunk_indices = train_indices[start_idx:end_idx]
        
        # Report memory
        memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print(f"\n{'='*50}")
        print(f"Processing chunk {chunk_idx+1}/{num_chunks} with {len(chunk_indices)} samples")
        print(f"Memory before: {memory_before:.1f} MB")
        
        # Force garbage collection before loading new data
        gc.collect()
        
        try:
            # Load chunk data in smaller batches
            chunk_X = []
            chunk_y = []
            
            # Use a smaller batch size for loading
            load_batch_size = 5000
            for i in range(0, len(chunk_indices), load_batch_size):
                end_i = min(i + load_batch_size, len(chunk_indices))
                print(f"  Loading batch {i}-{end_i} of {len(chunk_indices)}...")
                
                try:
                    # Load this batch
                    batch_indices = chunk_indices[i:end_i]
                    X_batch = np.array([X_mmap[idx] for idx in batch_indices])
                    y_batch = np.array([y_mmap[idx] for idx in batch_indices])
                    
                    # Append to list
                    chunk_X.append(X_batch)
                    chunk_y.append(y_batch)
                    
                    # Free memory
                    del X_batch, y_batch
                    gc.collect()
                except Exception as inner_e:
                    print(f"Warning: Error loading batch {i}-{end_i}: {inner_e}")
                    # Continue to next batch
            
            # Combine batches
            try:
                chunk_X = np.concatenate(chunk_X)
                chunk_y = np.concatenate(chunk_y)
                print(f"Data loaded. Memory: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.1f} MB")
            except Exception as concat_e:
                print(f"Error combining data batches: {concat_e}")
                # Skip this chunk
                continue
            
            # Track each epoch separately for better error handling
            chunk_history = {}
            for epoch in range(epochs_per_chunk):
                try:
                    print(f"Epoch {epoch+1}/{epochs_per_chunk}")
                    # Train for a single epoch
                    # Use batch-safe training instead of standard fit
                    epoch_history = process_chunk_with_batch_safety(
                        model,
                        chunk_X, 
                        chunk_y,
                        val_data=(val_X, val_y),
                        batch_size=batch_size,
                        epochs=1,
                        class_weight=class_weight
                    )
                    
                    # Store history
                    for k, v in epoch_history["history"].items():  # Changed from epoch_history.history
                        if k not in chunk_history:
                            chunk_history[k] = []
                        chunk_history[k].extend([float(val) for val in v])
                    
                    # Save after each epoch for safety
                    epoch_model_path = os.path.join(output_dir, "checkpoints", f"model_chunk_{chunk_idx+1}_epoch_{epoch+1}.h5")
                    model.save(epoch_model_path)
                    
                except Exception as epoch_e:
                    print(f"Error during epoch {epoch+1}: {epoch_e}")
                    import traceback
                    traceback.print_exc()
                    # Try to continue with next epoch
            
            # Store history from all completed epochs
            if chunk_history:
                history_log.append(chunk_history)
                
                # Save history
                try:
                    with open(history_path, "w") as f:
                        json.dump(history_log, f)
                except Exception as history_e:
                    print(f"Warning: Could not save history to JSON: {history_e}")
                    # Fallback - save as pickle
                    import pickle
                    with open(os.path.join(output_dir, "training_history.pkl"), "wb") as f:
                        pickle.dump(history_log, f)
            
            # Save model periodically
            if (chunk_idx + 1) % save_frequency == 0 or chunk_idx == num_chunks - 1:
                try:
                    model_path = os.path.join(output_dir, "checkpoints", f"model_chunk_{chunk_idx+1}.h5")
                    model.save(model_path)
                    print(f"Model saved to {model_path}")
                except Exception as save_e:
                    print(f"Error saving model: {save_e}")
                    # Try saving weights only
                    try:
                        weights_path = os.path.join(output_dir, "checkpoints", f"model_weights_{chunk_idx+1}.h5")
                        model.save_weights(weights_path)
                        print(f"Saved model weights to {weights_path}")
                    except:
                        print("Could not save model in any format")
            
            # Explicitly delete everything from memory
            del chunk_X, chunk_y
            
        except Exception as outer_e:
            print(f"Critical error processing chunk {chunk_idx+1}: {outer_e}")
            import traceback
            traceback.print_exc()
            
            # Write error to file for debugging
            with open(os.path.join(output_dir, f"error_chunk_{chunk_idx+1}.txt"), "w") as f:
                f.write(f"Error processing chunk {chunk_idx+1}:\n")
                f.write(traceback.format_exc())
                
            # Try saving model in error state
            try:
                error_model_path = os.path.join(output_dir, "checkpoints", f"error_recovery_{chunk_idx+1}.h5")
                model.save(error_model_path)
                print(f"Saved model in error state to {error_model_path}")
            except:
                print("Could not save model in error state")
                
            # Continue to next chunk
            continue
        
        # Force garbage collection
        gc.collect()
        
        # Report memory after cleanup
        memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print(f"Memory after: {memory_after:.1f} MB (Change: {memory_after - memory_before:.1f} MB)")
        
        # Write recovery file with last completed chunk
        with open(recovery_file, "w") as f:
            f.write(str(chunk_idx + 1))
        
        # Estimate time
        elapsed = time.time() - start_time
        avg_time_per_chunk = elapsed / (chunk_idx - start_chunk + 1) if chunk_idx > start_chunk else elapsed
        remaining = avg_time_per_chunk * (num_chunks - chunk_idx - 1)
        print(f"Estimated remaining time: {timedelta(seconds=int(remaining))}")
        
        # Reset TensorFlow session periodically
        if (chunk_idx + 1) % 20 == 0:
            print("Resetting TensorFlow session to prevent memory issues")
            try:
                temp_model_path = os.path.join(output_dir, "checkpoints", f"temp_reset_{chunk_idx+1}.h5")
                model.save(temp_model_path)
                
                # Clear session
                tf.keras.backend.clear_session()
                gc.collect()
                
                # Reload model
                model = tf.keras.models.load_model(temp_model_path)
                print("TensorFlow session reset complete")
            except Exception as reset_e:
                print(f"Error during TensorFlow reset: {reset_e}")
                # Continue anyway
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.h5")
    try:
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
    except Exception as final_save_e:
        print(f"Error saving final model: {final_save_e}")
        try:
            # Try alternative location
            alt_path = os.path.join(output_dir, "checkpoints", "final_model_backup.h5")
            model.save(alt_path)
            final_model_path = alt_path
            print(f"Saved final model to alternate location: {alt_path}")
        except:
            print("Could not save final model in any format")
    
    return model, final_model_path


def configure_tensorflow_memory():
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print(f"Memory growth enabled for {device}")

configure_tensorflow_memory()

def build_optimized_model(input_shape):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Dropout
    from tensorflow.keras.layers import Reshape, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, ConvLSTM2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    dropout_rate = 0.3
    reg_strength = 0.0001
    
    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)
    
    # Reshape for ConvLSTM (add spatial dimension)
    x = Reshape((input_shape[0], 1, 1, input_shape[1]))(x)
    
    # ConvLSTM layer
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(2, 1),  # Smaller kernel for efficiency
        padding='same',
        return_sequences=True,
        activation='tanh',
        recurrent_dropout=dropout_rate,
        kernel_regularizer=l2(reg_strength)
    )(x)

    # Reshape back to (sequence_length, features)
    x = Reshape((input_shape[0], 64))(x)
    x = BatchNormalization()(x)

    # Transformer Encoder Block
    attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = Add()([attention_output, x])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Multi-scale CNN Feature Extraction
    cnn_1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(reg_strength))(inputs)
    cnn_2 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(reg_strength))(inputs)
    cnn_3 = Conv1D(filters=32, kernel_size=7, padding='same', activation='relu', kernel_regularizer=l2(reg_strength))(inputs)

    # Global Pooling
    global_max = GlobalMaxPooling1D()(x)
    global_avg = GlobalAveragePooling1D()(x)

    # Concatenate features
    merged_features = Concatenate()([GlobalMaxPooling1D()(cnn_1), GlobalMaxPooling1D()(cnn_2), GlobalMaxPooling1D()(cnn_3), global_max, global_avg])
    merged_features = BatchNormalization()(merged_features)

    # Fully Connected Layers
    x = Dense(128, activation='relu', kernel_regularizer=l2(reg_strength))(merged_features)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(reg_strength))(x)
    x = Dropout(dropout_rate)(x)

    # Output Layer
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_strength))(x)

    # Compile Model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

def train_model(model, X_file, y_file, train_indices, val_indices, output_dir, batch_size=256, epochs=3):
    os.makedirs(output_dir, exist_ok=True)
    
    X_mmap = np.load(X_file, mmap_mode='r')
    y_mmap = np.load(y_file, mmap_mode='r')

    val_X = np.array([X_mmap[idx] for idx in val_indices[:1000]])  # Limit validation size
    val_y = np.array([y_mmap[idx] for idx in val_indices[:1000]])

    total_batches = len(train_indices) // batch_size + (1 if len(train_indices) % batch_size > 0 else 0)
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
        batch_indices = train_indices[start_idx:end_idx]

        try:
            batch_X = np.array([X_mmap[idx] for idx in batch_indices])
            batch_y = np.array([y_mmap[idx] for idx in batch_indices])

            model.fit(
                batch_X, batch_y,
                epochs=1,
                batch_size=batch_size,
                verbose=1
            )

        except tf.errors.ResourceExhaustedError:
            print("Memory issue, reducing batch size")
            batch_size = max(batch_size // 2, 16)
        
        # Save model checkpoint
        if (batch_idx + 1) % 100 == 0:
            model.save(os.path.join(output_dir, f"model_batch_{batch_idx+1}.h5"))

        gc.collect()
    
    model.save(os.path.join(output_dir, "final_model.h5"))
    return model

# Paths
output_dir = '/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/improved_model'
data_dir = '/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/ml_data'
X_file = os.path.join(data_dir, 'X_features.npy')
y_file = os.path.join(data_dir, 'y_labels.npy')

# Load train/validation indices
with open("zero_curtain_pipeline/modeling/checkpoints/spatiotemporal_split.pkl", "rb") as f:
    split_data = pickle.load(f)
train_indices = split_data["train_indices"]
val_indices = split_data["val_indices"]
test_indices = split_data["test_indices"]

# Load input shape
X_mmap = np.load(X_file, mmap_mode='r')
input_shape = X_mmap[train_indices[0]].shape

# Build and Train Model
model = build_optimized_model(input_shape)
trained_model = train_model(model, X_file, y_file, train_indices, val_indices, output_dir)

def evaluate_model_with_visualizations(model, X_file, y_file, test_indices, output_dir, metadata_file=None):
    """
    Comprehensive evaluation with detailed visualizations.
    """
    import os
    import gc
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_curve, 
        auc, precision_recall_curve, average_precision_score
    )
    import traceback
    
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, "evaluation_log.txt")
    def log_message(message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)
    
    log_message(f"Starting evaluation on {len(test_indices)} test samples")
    
    # Load metadata if available
    metadata = None
    if metadata_file and os.path.exists(metadata_file):
        try:
            import pickle
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
            log_message(f"Loaded metadata from {metadata_file}")
        except Exception as e:
            log_message(f"Error loading metadata: {str(e)}")
    
    # Load memory-mapped data
    X = np.load(X_file, mmap_mode='r')
    y = np.load(y_file, mmap_mode='r')
    
    # Process test data in batches
    batch_size = 500  # Smaller batch size for more reliable processing
    num_batches = int(np.ceil(len(test_indices) / batch_size))
    
    all_preds = []
    all_true = []
    all_meta = []
    
    log_message(f"Processing test data in {num_batches} batches of size {batch_size}")
    
    # Save test indices for reference
    np.save(os.path.join(output_dir, "test_indices.npy"), test_indices)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(test_indices))
        batch_indices = test_indices[start_idx:end_idx]
        
        log_message(f"Processing batch {batch_idx+1}/{num_batches} with {len(batch_indices)} samples")
        
        try:
            # Load batch data
            batch_X = np.array([X[idx] for idx in batch_indices])
            batch_y = np.array([y[idx] for idx in batch_indices])
            
            # Store metadata if available
            if metadata:
                try:
                    batch_meta = []
                    for idx in batch_indices:
                        if idx in metadata:
                            batch_meta.append(metadata[idx])
                        else:
                            # Create empty metadata if not found
                            batch_meta.append({})
                    all_meta.extend(batch_meta)
                except Exception as meta_e:
                    log_message(f"Error extracting metadata for batch {batch_idx}: {str(meta_e)}")
                    # Continue without metadata for this batch
            
            # Predict
            batch_preds = model.predict(batch_X, verbose=0)
            
            # Store results
            all_preds.extend(batch_preds.flatten())
            all_true.extend(batch_y)
            
            # Clean up
            del batch_X, batch_y, batch_preds
            gc.collect()
            
            log_message(f"Completed batch {batch_idx+1}/{num_batches}")
            
        except Exception as e:
            log_message(f"Error processing batch {batch_idx+1}: {str(e)}")
            log_message(traceback.format_exc())
            
            # Try to process the batch in smaller chunks
            try:
                log_message("Attempting to process batch in smaller chunks...")
                sub_batch_size = 50  # Much smaller batch
                sub_batches = int(np.ceil(len(batch_indices) / sub_batch_size))
                
                for sub_idx in range(sub_batches):
                    sub_start = sub_idx * sub_batch_size
                    sub_end = min((sub_idx + 1) * sub_batch_size, len(batch_indices))
                    sub_indices = batch_indices[sub_start:sub_end]
                    
                    # Load and process sub-batch
                    sub_X = np.array([X[idx] for idx in sub_indices])
                    sub_y = np.array([y[idx] for idx in sub_indices])
                    
                    # Add metadata if available
                    if metadata:
                        sub_meta = []
                        for idx in sub_indices:
                            if idx in metadata:
                                sub_meta.append(metadata[idx])
                            else:
                                sub_meta.append({})
                        all_meta.extend(sub_meta)
                    
                    # Predict
                    sub_preds = model.predict(sub_X, verbose=0)
                    
                    # Store results
                    all_preds.extend(sub_preds.flatten())
                    all_true.extend(sub_y)
                    
                    # Clean up
                    del sub_X, sub_y, sub_preds
                    gc.collect()
                
                log_message(f"Successfully processed batch {batch_idx+1} in {sub_batches} sub-batches")
            except Exception as sub_e:
                log_message(f"Error processing sub-batches: {str(sub_e)}")
                log_message(traceback.format_exc())
                log_message(f"Skipping batch {batch_idx+1}")
    
    log_message(f"Prediction complete. Total samples: {len(all_preds)}")
    
    # Save raw predictions for reference
    np.save(os.path.join(output_dir, "test_predictions.npy"), np.array(all_preds))
    np.save(os.path.join(output_dir, "test_true_labels.npy"), np.array(all_true))
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    try:
        # Create binary predictions
        all_preds_binary = (all_preds > 0.5).astype(int)
        
        # Calculate metrics
        report = classification_report(all_true, all_preds_binary, output_dict=True)
        report_str = classification_report(all_true, all_preds_binary)
        conf_matrix = confusion_matrix(all_true, all_preds_binary)
        
        log_message("\nClassification Report:")
        log_message(report_str)
        
        log_message("\nConfusion Matrix:")
        log_message(str(conf_matrix))
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(all_true, all_preds)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(all_true, all_preds)
        avg_precision = average_precision_score(all_true, all_preds)
        
        log_message(f"\nROC AUC: {roc_auc:.4f}")
        log_message(f"Average Precision: {avg_precision:.4f}")
        
        # Save results to file
        with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
            f.write("Classification Report:\n")
            f.write(report_str)
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(conf_matrix))
            f.write(f"\n\nROC AUC: {roc_auc:.4f}")
            f.write(f"\nAverage Precision: {avg_precision:.4f}")
        
        # Create visualizations
        try:
            # 1. Confusion Matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['Negative', 'Positive'],
                      yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "visualizations", "confusion_matrix.png"), dpi=300)
            plt.close()
            
            # 2. ROC Curve
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, "visualizations", "roc_curve.png"), dpi=300)
            plt.close()
            
            # 3. Precision-Recall Curve
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="upper right")
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, "visualizations", "precision_recall_curve.png"), dpi=300)
            plt.close()
            
            # 4. Score distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(all_preds[all_true == 0], bins=50, alpha=0.5, label='Negative Class', color='blue')
            sns.histplot(all_preds[all_true == 1], bins=50, alpha=0.5, label='Positive Class', color='red')
            plt.xlabel('Prediction Score')
            plt.ylabel('Count')
            plt.title('Distribution of Prediction Scores by Class')
            plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, "visualizations", "score_distribution.png"), dpi=300)
            plt.close()
            
            log_message("Saved all visualization plots")
        except Exception as plot_e:
            log_message(f"Error creating visualization plots: {str(plot_e)}")
            log_message(traceback.format_exc())
        
        # Save predictions with metadata
        if metadata and all_meta:
            try:
                # Create results DataFrame with metadata
                results_data = []
                
                for i in range(len(all_preds)):
                    if i < len(all_meta):
                        result = {
                            'prediction': float(all_preds[i]),
                            'true_label': int(all_true[i]),
                            'predicted_label': int(all_preds_binary[i]),
                            'correct': int(all_preds_binary[i] == all_true[i])
                        }
                        
                        # Add metadata
                        meta = all_meta[i]
                        for key, value in meta.items():
                            # Convert numpy types to Python native types
                            if hasattr(value, 'dtype'):
                                if np.issubdtype(value.dtype, np.integer):
                                    value = int(value)
                                elif np.issubdtype(value.dtype, np.floating):
                                    value = float(value)
                            result[key] = value
                        
                        results_data.append(result)
                
                # Create DataFrame
                results_df = pd.DataFrame(results_data)
                
                # Save to CSV
                results_df.to_csv(os.path.join(output_dir, "test_predictions_with_metadata.csv"), index=False)
                log_message(f"Saved predictions with metadata to CSV, shape: {results_df.shape}")
            except Exception as csv_e:
                log_message(f"Error saving predictions CSV: {str(csv_e)}")
                log_message(traceback.format_exc())
                
                # Try a simpler format
                try:
                    simple_df = pd.DataFrame({
                        'prediction': all_preds,
                        'true_label': all_true,
                        'predicted_label': all_preds_binary
                    })
                    simple_df.to_csv(os.path.join(output_dir, "simple_predictions.csv"), index=False)
                    log_message("Saved simplified predictions to CSV")
                except:
                    log_message("Failed to save predictions in any format")
        
        # Return metrics dictionary
        evaluation_metrics = {
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision),
            'accuracy': float(report['accuracy']),
            'precision': float(report['1']['precision']),
            'recall': float(report['1']['recall']),
            'f1_score': float(report['1']['f1-score']),
            'num_samples': len(all_true),
            'positive_rate': float(np.mean(all_true))
        }
        
        # Save metrics to JSON
        import json
        with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
            json.dump(evaluation_metrics, f, indent=4)
        
        log_message("Evaluation complete")
        return evaluation_metrics
    
    except Exception as metrics_e:
        log_message(f"Error calculating metrics: {str(metrics_e)}")
        log_message(traceback.format_exc())
        
        # Return basic metrics that we can calculate
        return {
            'num_samples': len(all_true),
            'positive_rate': float(np.mean(all_true)),
            'mean_prediction': float(np.mean(all_preds)),
            'error': str(metrics_e)
        }

# # Example usage
# def main():
#     # Configuration
#     input_shape = (24, 5)  # Adjust to match your data shape
#     output_dir = '/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/improved_model'
#     data_dir = '/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/ml_data'
#     X_file = os.path.join(data_dir, 'X_features.npy')
#     y_file = os.path.join(data_dir, 'y_labels.npy')
#     metadata_file = os.path.join(data_dir, 'metadata.pkl')
    
#     # Load split indices
#     with open("zero_curtain_pipeline/modeling/checkpoints/spatiotemporal_split.pkl", "rb") as f:
#         split_data = pickle.load(f)
#     train_indices = split_data["train_indices"]
#     val_indices = split_data["val_indices"]
#     test_indices = split_data["test_indices"]
    
#     # Load spatial weights
#     # Load spatial weights
#     with open("zero_curtain_pipeline/modeling/checkpoints/spatial_density.pkl", "rb") as f:
#         weights_data = pickle.load(f)
    
#     sample_weights = weights_data["weights"][train_indices]
#     sample_weights = sample_weights / np.mean(sample_weights) * len(sample_weights)
    
#     # Calculate class weights
#     y = np.load(y_file, mmap_mode='r')
#     train_y = y[train_indices]
#     pos_weight = (len(train_y) - np.sum(train_y)) / max(1, np.sum(train_y))
#     class_weight = {0: 1.0, 1: pos_weight}
#     print(f"Using class weight {pos_weight:.2f} for positive examples")
    
#     # Build improved model
#     X = np.load(X_file, mmap_mode='r')
#     input_shape = X[train_indices[0]].shape
#     model = build_improved_zero_curtain_model(input_shape)
    
#     # Train model with improved functions
#     model, model_path = improved_resumable_training(
#         model=model,
#         X_file=X_file,
#         y_file=y_file,
#         train_indices=train_indices,
#         val_indices=val_indices,
#         test_indices=test_indices,
#         output_dir=output_dir,
#         batch_size=256,
#         chunk_size=20000,  # Smaller chunks for better memory efficiency
#         epochs_per_chunk=3,  # More epochs per chunk for better learning
#         save_frequency=5,
#         class_weight=class_weight,
#         start_chunk=0  # Start from beginning or set to resume
#     )
    
#     # Evaluate the final model
#     results = evaluate_model_with_visualizations(
#         model=model,
#         X_file=X_file,
#         y_file=y_file,
#         test_indices=test_indices,
#         output_dir=output_dir,
#         metadata_file=metadata_file
#     )
    
#     print("\nFinal evaluation results:")
#     for metric, value in results.items():
#         print(f"  {metric}: {value:.4f}")

# if __name__ == "__main__":
#     main()

def analyze_spatial_performance(output_dir, metadata_file=None):
    """
    Analyze model performance across different geographical regions.
    
    Parameters:
    -----------
    output_dir : str
        Directory with test predictions
    metadata_file : str
        Path to metadata pickle file
        
    Returns:
    --------
    dict
        Spatial analysis results
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json
    import traceback
    
    # Setup logging
    log_file = os.path.join(output_dir, "spatial_analysis_log.txt")
    def log_message(message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)
    
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations", "spatial")
    os.makedirs(vis_dir, exist_ok=True)
    
    log_message("Starting spatial performance analysis")
    
    try:
        # Load predictions with metadata
        pred_file = os.path.join(output_dir, "test_predictions_with_metadata.csv")
        if os.path.exists(pred_file):
            log_message(f"Loading predictions with metadata from {pred_file}")
            predictions = pd.read_csv(pred_file)
        else:
            log_message("No predictions with metadata found, attempting to load separate files")
            # Load separate files and combine
            preds = np.load(os.path.join(output_dir, "test_predictions.npy"))
            true = np.load(os.path.join(output_dir, "test_true_labels.npy"))
            test_indices = np.load(os.path.join(output_dir, "test_indices.npy"))
            
            log_message(f"Loaded {len(preds)} predictions and {len(true)} true labels")
            
            # Create dataframe
            predictions = pd.DataFrame({
                'prediction': preds,
                'true_label': true,
                'predicted_label': (preds > 0.5).astype(int),
                'correct': ((preds > 0.5).astype(int) == true).astype(int)
            })
            
            # Load metadata
            if metadata_file and os.path.exists(metadata_file):
                try:
                    import pickle
                    with open(metadata_file, "rb") as f:
                        metadata = pickle.load(f)
                    
                    log_message(f"Loaded metadata with {len(metadata)} entries")
                    
                    # Add metadata
                    for i, idx in enumerate(test_indices):
                        if i < len(predictions) and idx in metadata:
                            meta = metadata[idx]
                            for key, value in meta.items():
                                if key not in predictions.columns:
                                    predictions[key] = None
                                predictions.loc[i, key] = value
                        
                    log_message(f"Added metadata to predictions, columns: {list(predictions.columns)}")
                except Exception as meta_e:
                    log_message(f"Error adding metadata: {str(meta_e)}")
                    log_message(traceback.format_exc())
        
        # Check if we have spatial data
        if 'latitude' not in predictions.columns or 'longitude' not in predictions.columns:
            log_message("No spatial data (latitude/longitude) found in predictions")
            return {'error': 'No spatial data available'}
        
        log_message(f"Predictions dataframe loaded with shape {predictions.shape}")
        
        # Define latitude bands
        def get_lat_band(lat):
            if pd.isna(lat):
                return "Unknown"
            lat = float(lat)
            if lat < 55:
                return "<55°N"
            elif lat < 60:
                return "55-60°N"
            elif lat < 66.5:
                return "60-66.5°N (Subarctic)"
            elif lat < 70:
                return "66.5-70°N (Arctic)"
            elif lat < 75:
                return "70-75°N (Arctic)"
            elif lat < 80:
                return "75-80°N (Arctic)"
            else:
                return ">80°N (Arctic)"
        
        # Add latitude band
        predictions['lat_band'] = predictions['latitude'].apply(get_lat_band)
        
        # Group by source (site) if available
        if 'source' in predictions.columns:
            log_message("Grouping metrics by source site")
            site_metrics = predictions.groupby('source').apply(lambda x: pd.Series({
                'count': len(x),
                'positive_count': x['true_label'].sum(),
                'positive_rate': x['true_label'].mean(),
                'accuracy': x['correct'].mean(),
                'precision': (x['predicted_label'] & x['true_label']).sum() / max(1, x['predicted_label'].sum()),
                'recall': (x['predicted_label'] & x['true_label']).sum() / max(1, x['true_label'].sum()),
                'f1': 2 * (x['predicted_label'] & x['true_label']).sum() / 
                      max(1, (x['predicted_label'] & x['true_label']).sum() + 
                          0.5 * (x['predicted_label'].sum() + x['true_label'].sum())),
                'latitude': x['latitude'].mean(),
                'longitude': x['longitude'].mean(),
                'lat_band': x['lat_band'].iloc[0] if not x['lat_band'].isna().all() else "Unknown"
            }))
            
            # Filter out sites with too few samples
            site_metrics = site_metrics[site_metrics['count'] >= 10]
            log_message(f"Found {len(site_metrics)} sites with at least 10 samples")
            
            # Save site metrics
            site_metrics.to_csv(os.path.join(vis_dir, "site_metrics.csv"))
        else:
            site_metrics = None
            log_message("No 'source' column found, skipping site-level analysis")
        
        # Group by latitude band
        log_message("Grouping metrics by latitude band")
        band_metrics = predictions.groupby('lat_band').apply(lambda x: pd.Series({
            'count': len(x),
            'positive_count': x['true_label'].sum(),
            'positive_rate': x['true_label'].mean(),
            'accuracy': x['correct'].mean(),
            'precision': (x['predicted_label'] & x['true_label']).sum() / max(1, x['predicted_label'].sum()),
            'recall': (x['predicted_label'] & x['true_label']).sum() / max(1, x['true_label'].sum()),
            'f1': 2 * (x['predicted_label'] & x['true_label']).sum() / 
                  max(1, (x['predicted_label'] & x['true_label']).sum() + 
                      0.5 * (x['predicted_label'].sum() + x['true_label'].sum()))
        }))
        
        # Save latitude band metrics
        band_metrics.to_csv(os.path.join(vis_dir, "latitude_band_metrics.csv"))
        log_message(f"Found {len(band_metrics)} latitude bands")
        
        # Create visualizations
        # 1. Performance by latitude band
        try:
            plt.figure(figsize=(14, 8))
            
            # Order bands by latitude
            ordered_bands = [
                '<55°N', '55-60°N', '60-66.5°N (Subarctic)', 
                '66.5-70°N (Arctic)', '70-75°N (Arctic)', 
                '75-80°N (Arctic)', '>80°N (Arctic)'
            ]
            ordered_bands = [b for b in ordered_bands if b in band_metrics.index]
            
            # Get metrics
            band_data = band_metrics.loc[ordered_bands]
            
            # Plot metrics
            x = np.arange(len(ordered_bands))
            width = 0.15
            
            plt.bar(x - 2*width, band_data['accuracy'], width, label='Accuracy', color='#3274A1')
            plt.bar(x - width, band_data['precision'], width, label='Precision', color='#E1812C')
            plt.bar(x, band_data['recall'], width, label='Recall', color='#3A923A')
            plt.bar(x + width, band_data['f1'], width, label='F1', color='#C03D3E')
            plt.bar(x + 2*width, band_data['positive_rate'], width, label='Positive Rate', color='#9372B2')
            
            plt.xlabel('Latitude Band')
            plt.ylabel('Score')
            plt.title('Model Performance by Latitude Band')
            plt.xticks(x, ordered_bands, rotation=45, ha='right')
            plt.legend(loc='lower right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "performance_by_latitude.png"), dpi=300)
            plt.close()
            log_message("Created latitude band performance visualization")
        except Exception as e:
            log_message(f"Error creating latitude band plot: {str(e)}")
        
        # 2. Map visualization if cartopy is available
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            
            if site_metrics is not None:
                plt.figure(figsize=(15, 12))
                
                # Set up the projection
                ax = plt.axes(projection=ccrs.NorthPolarStereo())
                ax.set_extent([-180, 180, 45, 90], ccrs.PlateCarree())
                
                # Add map features
                ax.add_feature(cfeature.LAND, facecolor='lightgray')
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                
                # Add gridlines
                gl = ax.gridlines(draw_labels=True)
                gl.top_labels = False
                gl.right_labels = False
                
                # Filter sites with valid coordinates
                valid_sites = site_metrics.dropna(subset=['latitude', 'longitude'])
                
                # Create scatter plot
                scatter = ax.scatter(
                    valid_sites['longitude'], 
                    valid_sites['latitude'],
                    transform=ccrs.PlateCarree(),
                    c=valid_sites['f1'],
                    s=valid_sites['count'] / 5,  # Size by sample count
                    cmap='RdYlGn',
                    vmin=0, vmax=1,
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5
                )
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
                cbar.set_label('F1 Score')
                
                # Add Arctic Circle
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0, 0], 90 - 66.5
                verts = np.vstack([radius*np.sin(theta), radius*np.cos(theta)]).T
                circle = plt.Line2D(verts[:, 0], verts[:, 1], color='blue', 
                                  linestyle='--', transform=ax.transData)
                ax.add_line(circle)
                
                plt.title('Spatial Distribution of Model Performance (F1 Score)', fontsize=14)
                plt.savefig(os.path.join(vis_dir, "spatial_performance_map.png"), dpi=300, bbox_inches='tight')
                plt.close()
                log_message("Created spatial performance map")
            else:
                log_message("Skipping spatial map due to missing site metrics")
        except ImportError:
            log_message("Cartopy not available for map visualization")
        except Exception as map_e:
            log_message(f"Error creating map visualization: {str(map_e)}")
        
        # 3. Performance vs. sample count (if site metrics available)
        if site_metrics is not None:
            try:
                plt.figure(figsize=(12, 8))
                plt.scatter(site_metrics['count'], site_metrics['f1'], 
                          c=site_metrics['positive_rate'], cmap='viridis', 
                          alpha=0.7, s=50, edgecolor='black', linewidth=1)
                plt.colorbar(label='Positive Rate')
                plt.xscale('log')
                plt.xlabel('Number of Samples')
                plt.ylabel('F1 Score')
                plt.title('F1 Score vs. Sample Count by Site')
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(vis_dir, "performance_vs_sample_count.png"), dpi=300)
                plt.close()
                log_message("Created performance vs sample count plot")
            except Exception as e:
                log_message(f"Error creating performance vs sample count plot: {str(e)}")
        
        # Return summary
        summary = {
            'latitude_bands': {
                'count': len(band_metrics),
                'bands': band_metrics.to_dict()
            }
        }
        
        if site_metrics is not None:
            summary['sites'] = {
                'count': len(site_metrics),
                'best_performing': site_metrics.nlargest(5, 'f1')[['f1', 'count', 'positive_rate']].to_dict(),
                'worst_performing': site_metrics.nsmallest(5, 'f1')[['f1', 'count', 'positive_rate']].to_dict()
            }
        
        # Save summary
        with open(os.path.join(vis_dir, "spatial_summary.json"), "w") as f:
            # Handle NumPy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.floating, np.ndarray)):
                        return float(obj)
                    return super(NumpyEncoder, self).default(obj)
            
            json.dump(summary, f, indent=4, cls=NumpyEncoder)
        
        log_message("Spatial analysis complete")
        return summary
    
    except Exception as e:
        log_message(f"Error in spatial analysis: {str(e)}")
        log_message(traceback.format_exc())
        return {'error': str(e)}

def analyze_feature_importance(model, X_file, y_file, test_indices, output_dir):
    """
    Analyze feature importance using permutation importance.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    X_file : str
        Path to features numpy file
    y_file : str
        Path to labels numpy file
    test_indices : numpy.ndarray
        Test set indices
    output_dir : str
        Output directory for results
        
    Returns:
    --------
    dict
        Feature importance results
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_auc_score
    import gc
    import traceback
    
    # Create output directory
    vis_dir = os.path.join(output_dir, "visualizations", "features")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, "feature_importance_log.txt")
    def log_message(message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)
    
    log_message("Starting feature importance analysis")
    
    try:
        # Load a subset of test data to save memory
        max_samples = min(5000, len(test_indices))  # Limit to 5000 samples for memory
        np.random.seed(42)
        if len(test_indices) > max_samples:
            test_indices_subset = np.random.choice(test_indices, max_samples, replace=False)
        else:
            test_indices_subset = test_indices
        
        log_message(f"Using {len(test_indices_subset)} samples for feature importance analysis")
        
        # Load data
        X = np.load(X_file, mmap_mode='r')
        y = np.load(y_file, mmap_mode='r')
        
        # Ensure the memory remains manageable
        try:
            X_test = np.array([X[idx] for idx in test_indices_subset])
            y_test = np.array([y[idx] for idx in test_indices_subset])
        except MemoryError:
            # If memory error, reduce sample size further
            max_samples = min(1000, len(test_indices))
            test_indices_subset = np.random.choice(test_indices, max_samples, replace=False)
            log_message(f"Memory error, reducing to {max_samples} samples")
            X_test = np.array([X[idx] for idx in test_indices_subset])
            y_test = np.array([y[idx] for idx in test_indices_subset])
        
        log_message(f"Loaded test data with shape {X_test.shape}")
        
        # Get feature names (modify as needed based on your data)
        feature_names = ['Temperature', 'Temperature Gradient', 'Depth', 'Moisture', 'Moisture Gradient']
        
        # Truncate to actual number of features
        feature_names = feature_names[:X_test.shape[2]]
        log_message(f"Using feature names: {feature_names}")
        
        # Get baseline performance
        baseline_preds = model.predict(X_test, verbose=0)
        baseline_auc = roc_auc_score(y_test, baseline_preds)
        log_message(f"Baseline AUC: {baseline_auc:.4f}")
        
        # Perform permutation importance analysis
        n_repeats = 3  # Reduced from 5 to save time
        importances = np.zeros((len(feature_names), n_repeats))
        
        for feature_idx in range(len(feature_names)):
            feature_name = feature_names[feature_idx]
            log_message(f"Analyzing importance of feature: {feature_name}")
            
            for repeat in range(n_repeats):
                log_message(f"  Repeat {repeat+1}/{n_repeats}")
                
                # Create a copy of the test data
                X_permuted = X_test.copy()
                
                # Permute the feature across all time steps
                for time_step in range(X_test.shape[1]):
                    X_permuted[:, time_step, feature_idx] = np.random.permutation(X_test[:, time_step, feature_idx])
                
                # Predict with permuted feature
                perm_preds = model.predict(X_permuted, verbose=0)
                perm_auc = roc_auc_score(y_test, perm_preds)
                
                # Store importance (decrease in performance)
                importances[feature_idx, repeat] = baseline_auc - perm_auc
                log_message(f"    Decrease in AUC: {importances[feature_idx, repeat]:.4f}")
                
                # Clean up
                del X_permuted, perm_preds
                gc.collect()
        
        # Calculate mean and std of importance
        mean_importances = np.mean(importances, axis=1)
        std_importances = np.std(importances, axis=1)
        
        # Create DataFrame for results
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_importances,
            'Std': std_importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        log_message(f"Feature importance results:\n{importance_df}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df, 
                   xerr=importance_df['Std'], palette='viridis')
        plt.title('Feature Importance (Permutation Method)')
        plt.xlabel('Decrease in AUC when Feature is Permuted')
        plt.ylabel('Feature')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "feature_importance.png"), dpi=300)
        plt.close()
        
        # Save results
        importance_df.to_csv(os.path.join(vis_dir, "feature_importance.csv"), index=False)
        log_message("Saved feature importance results")
        
        # Analyze temporal patterns in feature importance
        log_message("Analyzing temporal importance patterns")
        
        # We'll analyze which time steps in the sequence are most important
        time_importances = np.zeros((X_test.shape[1], n_repeats))
        
        for time_idx in range(X_test.shape[1]):
            log_message(f"Analyzing importance of time step {time_idx+1}/{X_test.shape[1]}")
            
            for repeat in range(n_repeats):
                # Create a copy of the test data
                X_permuted = X_test.copy()
                
                # Permute all features at this time step
                X_permuted[:, time_idx, :] = np.random.permutation(X_test[:, time_idx, :])
                
                # Predict with permuted time step
                perm_preds = model.predict(X_permuted, verbose=0)
                perm_auc = roc_auc_score(y_test, perm_preds)
                
                # Store importance
                time_importances[time_idx, repeat] = baseline_auc - perm_auc
                
                # Clean up
                del X_permuted, perm_preds
                gc.collect()
        
        # Calculate mean and std
        mean_time_importances = np.mean(time_importances, axis=1)
        std_time_importances = np.std(time_importances, axis=1)
        
        # Create DataFrame
        time_importance_df = pd.DataFrame({
            'Time_Step': np.arange(X_test.shape[1]),
            'Importance': mean_time_importances,
            'Std': std_time_importances
        })
        
        log_message(f"Time step importance results:\n{time_importance_df}")
        
        # Plot time step importance
        plt.figure(figsize=(14, 8))
        plt.errorbar(time_importance_df['Time_Step'], time_importance_df['Importance'],
                    yerr=time_importance_df['Std'], fmt='o-', capsize=5, linewidth=2, markersize=8)
        plt.title('Importance of Time Steps in Sequence')
        plt.xlabel('Time Step Index')
        plt.ylabel('Decrease in AUC when Time Step is Permuted')
        plt.grid(alpha=0.3)
        plt.xticks(time_importance_df['Time_Step'])
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "time_step_importance.png"), dpi=300)
        plt.close()
        
        # Save time importance results
        time_importance_df.to_csv(os.path.join(vis_dir, "time_step_importance.csv"), index=False)
        log_message("Saved time step importance results")
        
        # Feature interaction heatmap (optional, if memory allows)
        try:
            # If we have few enough features, analyze pairwise interactions
            if len(feature_names) <= 5:
                log_message("Analyzing feature interactions")
                interactions = np.zeros((len(feature_names), len(feature_names)))
                
                for i in range(len(feature_names)):
                    for j in range(i+1, len(feature_names)):
                        # Skip diagonal
                        if i == j:
                            continue
                            
                        # Create permuted dataset
                        X_permuted = X_test.copy()
                        
                        # Permute both features
                        for time_step in range(X_test.shape[1]):
                            # Generate permutation indices
                            perm_idx = np.random.permutation(len(X_test))
                            X_permuted[:, time_step, i] = X_test[perm_idx, time_step, i]
                            X_permuted[:, time_step, j] = X_test[perm_idx, time_step, j]
                        
                        # Predict and calculate AUC
                        perm_preds = model.predict(X_permuted, verbose=0)
                        perm_auc = roc_auc_score(y_test, perm_preds)
                        
                        # Calculate interaction strength
                        # We compare permuting both features together vs....
                        interactions[i, j] = baseline_auc - perm_auc - (mean_importances[i] + mean_importances[j])
                        interactions[j, i] = interactions[i, j]  # Symmetric
                        
                        # Clean up
                        del X_permuted, perm_preds
                        gc.collect()
                
                # Plot interaction heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(interactions, annot=True, fmt=".3f", cmap="coolwarm",
                           xticklabels=feature_names, yticklabels=feature_names)
                plt.title("Feature Interaction Strength")
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, "feature_interactions.png"), dpi=300)
                plt.close()
                
                log_message("Saved feature interaction analysis")
        except Exception as inter_e:
            log_message(f"Error in feature interaction analysis: {str(inter_e)}")
        
        # Clean up
        del X_test, y_test
        gc.collect()
        
        log_message("Feature importance analysis complete")
        
        return {
            'feature_importance': importance_df.to_dict(orient='records'),
            'time_step_importance': time_importance_df.to_dict(orient='records'),
            'baseline_auc': float(baseline_auc)
        }
    
    except Exception as e:
        log_message(f"Error in feature importance analysis: {str(e)}")
        log_message(traceback.format_exc())
        return {'error': str(e)}

def failsafe_training_fixed(model, X_file, y_file, train_indices, val_indices, test_indices,
                           output_dir, batch_size=256, epochs=3, class_weight=None, start_batch=0):
    """
    Fixed robust training function that prevents stalling and handles TensorFlow errors gracefully,
    without changing the model architecture.
    """
    import os
    import numpy as np
    import tensorflow as tf
    import gc
    import time
    import psutil
    from datetime import datetime, timedelta
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, "logs", "training_log.txt")
    def log_message(message):
        with open(log_file, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
        print(message)
    
    log_message("Starting training with original model architecture")
    
    # Load data in memory-mapped mode
    X_mmap = np.load(X_file, mmap_mode='r')
    y_mmap = np.load(y_file, mmap_mode='r')
    
    # Configuration for validation
    # Use a subset of validation data to save memory
    val_limit = min(1000, len(val_indices))
    val_indices_subset = val_indices[:val_limit]
    val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
    val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    
    # Training configuration
    total_batches = len(train_indices) // batch_size + (1 if len(train_indices) % batch_size > 0 else 0)
    log_message(f"Total batches: {total_batches}")
    
    # Load a previous model if starting from a later batch
    if start_batch > 0:
        checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{start_batch}.h5")
        if os.path.exists(checkpoint_path):
            log_message(f"Loading model from checkpoint {checkpoint_path}")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            # Find the latest checkpoint before start_batch
            checkpoint_files = [f for f in os.listdir(os.path.join(output_dir, "checkpoints")) 
                               if f.startswith("model_batch_") and f.endswith(".h5")]
            batch_numbers = []
            for file in checkpoint_files:
                try:
                    batch_num = int(file.split("_")[-1].split(".")[0])
                    if batch_num < start_batch:
                        batch_numbers.append(batch_num)
                except:
                    continue
            
            if batch_numbers:
                latest_batch = max(batch_numbers)
                checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{latest_batch}.h5")
                log_message(f"Loading model from latest checkpoint {checkpoint_path}")
                model = tf.keras.models.load_model(checkpoint_path)
    
    # Store best validation metrics
    best_val_auc = 0
    best_model_path = None
    stall_counter = 0
    max_stalls = 3
    
    # Track progress
    start_time = time.time()
    
    # Main training loop - process batches
    for batch_idx in range(start_batch, total_batches):
        batch_start_time = time.time()
        
        # Calculate batch indices
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
        
        # Get current batch indices
        batch_indices = train_indices[start_idx:end_idx]
        
        # Report memory usage
        current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        log_message(f"Memory before batch load: {current_memory:.1f} MB")
        
        # Load batch data with chunking to avoid memory issues
        try:
            # Load in smaller chunks
            chunk_size = min(64, len(batch_indices))
            batch_X_chunks = []
            batch_y_chunks = []
            
            for i in range(0, len(batch_indices), chunk_size):
                end_i = min(i + chunk_size, len(batch_indices))
                chunk_indices = batch_indices[i:end_i]
                
                chunk_X = np.array([X_mmap[idx] for idx in chunk_indices])
                chunk_y = np.array([y_mmap[idx] for idx in chunk_indices])
                
                batch_X_chunks.append(chunk_X)
                batch_y_chunks.append(chunk_y)
                
                # Clean up chunks immediately
                gc.collect()
            
            # Combine chunks
            batch_X = np.concatenate(batch_X_chunks)
            batch_y = np.concatenate(batch_y_chunks)
            
            # Clean up chunk lists
            del batch_X_chunks, batch_y_chunks
            gc.collect()
            
        except Exception as e:
            log_message(f"Error loading batch data: {e}")
            continue
        
        # Train with progressive mini-batch size reduction if needed
        mini_batch_sizes = [min(batch_size, 128), 64, 32, 16]
        training_successful = False
        
        for mini_batch_size in mini_batch_sizes:
            if training_successful:
                break
                
            try:
                log_message(f"Trying mini-batch size {mini_batch_size}")
                
                # Train for one epoch
                history = model.fit(
                    batch_X, batch_y,
                    epochs=1,
                    verbose=1,
                    class_weight=class_weight,
                    batch_size=mini_batch_size
                )
                
                # Check for NaN loss
                if np.isnan(history.history['loss'][0]):
                    log_message(f"NaN loss detected with mini-batch size {mini_batch_size}")
                    continue
                
                # Check for AUC value
                if 'auc' in history.history and history.history['auc'][0] == 0:
                    stall_counter += 1
                    log_message(f"Warning: AUC is zero, stall detected. Counter: {stall_counter}/{max_stalls}")
                    
                    if stall_counter >= max_stalls:
                        log_message("Multiple stalls detected, resetting optimizer state")
                        
                        # Save weights
                        temp_weights_path = os.path.join(output_dir, "checkpoints", "temp_weights.h5")
                        model.save_weights(temp_weights_path)
                        
                        # Get current learning rate
                        if hasattr(model.optimizer, 'lr'):
                            current_lr = float(model.optimizer.lr.numpy())
                            # Reduce learning rate
                            new_lr = current_lr * 0.5
                            log_message(f"Reducing learning rate from {current_lr} to {new_lr}")
                        else:
                            new_lr = 0.0001
                            log_message(f"Setting learning rate to {new_lr}")
                        
                        # Recompile with fresh optimizer
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=new_lr),
                            loss='binary_crossentropy',
                            metrics=[
                                'accuracy',
                                tf.keras.metrics.AUC(name='auc'),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall')
                            ]
                        )
                        
                        # Load weights back
                        model.load_weights(temp_weights_path)
                        
                        # Reset counter
                        stall_counter = 0
                else:
                    # Reset stall counter if we have non-zero AUC
                    stall_counter = 0
                
                # Mark as successful
                training_successful = True
                
            except tf.errors.ResourceExhaustedError:
                log_message(f"Resource exhausted with mini-batch size {mini_batch_size}, trying smaller size")
                continue
            except Exception as e:
                log_message(f"Error during training with mini-batch size {mini_batch_size}: {e}")
                continue
        
        # If all mini-batch sizes failed, skip this batch
        if not training_successful:
            log_message(f"All mini-batch sizes failed for batch {batch_idx+1}, skipping")
            
            # If multiple consecutive batches fail, try resetting the session
            if batch_idx > start_batch:
                log_message("Resetting TensorFlow session to recover")
                
                # Save model
                temp_path = os.path.join(output_dir, "checkpoints", f"recovery_{batch_idx}.h5")
                model.save(temp_path)
                
                # Clear session
                tf.keras.backend.clear_session()
                gc.collect()
                
                # Reload model
                model = tf.keras.models.load_model(temp_path)
            
            continue
        
        # Clean up batch data
        del batch_X, batch_y
        gc.collect()
        
        # Evaluate on validation set every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches or stall_counter > 0:
            log_message(f"Evaluating on validation set at batch {batch_idx+1}/{total_batches}")
            
            try:
                val_metrics = model.evaluate(val_X, val_y, verbose=0)
                val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
                
                # Display metrics
                metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in val_metrics_dict.items()])
                log_message(f"Validation metrics at batch {batch_idx+1}/{total_batches}:")
                log_message(f"  {metrics_str}")
                
                # Check for better AUC
                val_auc = val_metrics_dict.get('auc', 0)
                if val_auc > best_val_auc:
                    log_message(f"New best model saved with val_auc: {val_auc:.4f}")
                    best_val_auc = val_auc
                    best_model_path = os.path.join(output_dir, "checkpoints", f"best_model_batch_{batch_idx+1}.h5")
                    model.save(best_model_path)
                
                # If validation AUC is also stuck at zero, this indicates a deeper problem
                if val_auc == 0 and stall_counter > 0:
                    log_message("Critical: Validation AUC is zero, attempting advanced recovery")
                    
                    # Save model state
                    temp_path = os.path.join(output_dir, "checkpoints", f"pre_recovery_{batch_idx+1}.h5")
                    model.save(temp_path)
                    
                    # Clear session
                    tf.keras.backend.clear_session()
                    gc.collect()
                    
                    # Reload model
                    model = tf.keras.models.load_model(temp_path)
                    
                    # More aggressive learning rate reduction
                    current_lr = 0.0001  # Default fallback
                    if hasattr(model.optimizer, 'lr'):
                        current_lr = float(model.optimizer.lr.numpy())
                    
                    new_lr = current_lr * 0.1  # 10x reduction
                    log_message(f"Emergency reduction of learning rate to {new_lr}")
                    
                    # Recompile with fresh optimizer and adjusted metrics
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(
                            learning_rate=new_lr,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-07,
                            amsgrad=True
                        ),
                        loss='binary_crossentropy',
                        metrics=[
                            'accuracy',
                            tf.keras.metrics.AUC(name='auc', num_thresholds=200),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall')
                        ]
                    )
                    
                    # Reset counter after emergency intervention
                    stall_counter = 0
                
            except Exception as eval_e:
                log_message(f"Error during validation: {eval_e}")
        
        # Save checkpoint every 100 batches
        if (batch_idx + 1) % 100 == 0:
            checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{batch_idx+1}.h5")
            model.save(checkpoint_path)
            log_message(f"Checkpoint saved to {checkpoint_path}")
        
        # Progress reporting
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time
        progress = (batch_idx - start_batch + 1) / (total_batches - start_batch)
        remaining = (elapsed / progress) * (1 - progress) if progress > 0 else 0
        
        log_message(f"Progress: {batch_idx+1}/{total_batches} batches ({progress*100:.1f}%)")
        log_message(f"Batch time: {batch_time:.1f}s")
        log_message(f"Elapsed: {timedelta(seconds=int(elapsed))}, Remaining: {timedelta(seconds=int(remaining))}")
        
        # Memory tracking
        current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        log_message(f"Memory usage: {current_memory:.1f} MB")
        
        # Reset TensorFlow session periodically to prevent memory growth
        if (batch_idx + 1) % 500 == 0 and batch_idx > 0:
            log_message("Resetting TensorFlow session")
            
            # Save model
            temp_path = os.path.join(output_dir, "checkpoints", f"temp_reset_{batch_idx+1}.h5")
            model.save(temp_path)
            
            # Clear session
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Reload model
            model = tf.keras.models.load_model(temp_path)
            log_message("TensorFlow session reset complete")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.h5")
    model.save(final_model_path)
    log_message(f"Training complete. Final model saved to {final_model_path}")
    
    # Return best model if available
    if best_model_path and os.path.exists(best_model_path):
        log_message(f"Loading best model from {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)
        return model, best_model_path
    
    return model, final_model_path

def fast_training(model, X_file, y_file, train_indices, val_indices, test_indices,
                 output_dir, batch_size=256, class_weight=None, start_batch=0):
    """
    Accelerated training function that focuses on speed and efficiency.
    """
    import os
    import numpy as np
    import tensorflow as tf
    import gc
    import time
    import psutil
    from datetime import datetime, timedelta
    import random
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, "logs", "training_log.txt")
    def log_message(message):
        with open(log_file, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
        print(message)
    
    log_message("Starting accelerated training")
    
    # Memory-mapped data access
    X_mmap = np.load(X_file, mmap_mode='r')
    y_mmap = np.load(y_file, mmap_mode='r')
    
    # Smaller validation set to save memory
    val_size = min(500, len(val_indices))
    val_indices_sample = val_indices[:val_size]
    val_X = np.array([X_mmap[idx] for idx in val_indices_sample])
    val_y = np.array([y_mmap[idx] for idx in val_indices_sample])
    
    # SPEED OPTIMIZATION 1: Use larger batches
    effective_batch_size = batch_size * 4
    log_message(f"Using accelerated batch size: {effective_batch_size}")
    
    # SPEED OPTIMIZATION 2: Skip most validation steps
    validation_frequency = 500  # Only validate every 500 batches
    
    # SPEED OPTIMIZATION 3: Limit total batches (use a sample of data)
    # This is critical to get results in a reasonable timeframe
    max_batches = 5000  # Limit to 5000 batches instead of 54371
    
    # SPEED OPTIMIZATION 4: Use a much higher learning rate
    if start_batch == 0:
        # Recompile with a higher learning rate
        log_message("Setting higher learning rate")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Much higher learning rate
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
    
    # Calculate approximate number of batches
    num_batches = min(max_batches, (len(train_indices) + effective_batch_size - 1) // effective_batch_size)
    log_message(f"Training on {num_batches} batches (out of original {len(train_indices) // batch_size})")
    
    # Load from checkpoint if resuming
    if start_batch > 0:
        checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{start_batch}.h5")
        if os.path.exists(checkpoint_path):
            log_message(f"Loading model from checkpoint {checkpoint_path}")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            log_message("Checkpoint not found, starting with fresh model")
    
    # Training tracking variables
    stall_counter = 0
    max_stalls = 3
    best_val_auc = 0
    best_model_path = None
    start_time = time.time()
    
    # Initial validation
    log_message("Performing initial validation")
    try:
        val_metrics = model.evaluate(val_X, val_y, verbose=0)
        val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in val_metrics_dict.items()])
        log_message(f"Initial validation: {metrics_str}")
        best_val_auc = val_metrics_dict.get('auc', 0)
    except Exception as e:
        log_message(f"Error during initial validation: {e}")
    
    # SPEED OPTIMIZATION 5: Use a subset of training data, with appropriate indices
    # We'll use stratified sampling to maintain class balance
    # First, separate indices by class
    pos_indices = []
    neg_indices = []
    
    # Sample 20% of data to determine class distribution
    sample_size = min(10000, len(train_indices))
    sample_indices = random.sample(list(train_indices), sample_size)
    
    for idx in sample_indices:
        if y_mmap[idx] > 0.5:
            pos_indices.append(idx)
        else:
            neg_indices.append(idx)
    
    # Calculate positive class ratio
    pos_ratio = len(pos_indices) / sample_size
    log_message(f"Positive class ratio in sample: {pos_ratio:.3f}")
    
    # Create a balanced subset of indices for faster training
    subset_size = min(num_batches * effective_batch_size, len(train_indices))
    subset_pos_size = int(subset_size * pos_ratio)
    subset_neg_size = subset_size - subset_pos_size
    
    # Sample from each class
    if len(pos_indices) > 0 and len(neg_indices) > 0:
        # For positive class
        if subset_pos_size <= len(pos_indices):
            subset_pos_indices = random.sample(pos_indices, subset_pos_size)
        else:
            subset_pos_indices = pos_indices
            
        # For negative class
        if subset_neg_size <= len(neg_indices):
            subset_neg_indices = random.sample(neg_indices, subset_neg_size)
        else:
            subset_neg_indices = neg_indices
            
        # Combine and shuffle
        training_subset = subset_pos_indices + subset_neg_indices
        random.shuffle(training_subset)
    else:
        # Fallback if stratification fails
        training_subset = random.sample(list(train_indices), min(subset_size, len(train_indices)))
    
    log_message(f"Created training subset with {len(training_subset)} samples")
    
    # Process each batch
    for batch_idx in range(start_batch, num_batches):
        batch_start_time = time.time()
        
        # Calculate batch indices (using our subset)
        start_idx = batch_idx * effective_batch_size
        end_idx = min((batch_idx + 1) * effective_batch_size, len(training_subset))
        
        # Skip if we've reached the end of our subset
        if start_idx >= len(training_subset):
            break
            
        batch_indices = training_subset[start_idx:end_idx]
        
        # Load batch data
        try:
            batch_X = np.array([X_mmap[idx] for idx in batch_indices])
            batch_y = np.array([y_mmap[idx] for idx in batch_indices])
        except Exception as e:
            log_message(f"Error loading batch data: {e}")
            continue
        
        # Train on this batch
        try:
            # SPEED OPTIMIZATION 6: More epochs per batch for faster convergence
            history = model.fit(
                batch_X, batch_y,
                epochs=3,  # Train for 3 epochs on each batch
                batch_size=min(len(batch_X), 128),
                class_weight=class_weight,
                verbose=1
            )
            
            # Check for NaN loss
            if np.isnan(history.history['loss'][-1]):
                log_message(f"NaN loss detected, skipping batch")
                continue
                
            # Check for stalls
            if 'auc' in history.history and history.history['auc'][-1] == 0:
                stall_counter += 1
                log_message(f"Warning: AUC is zero, stall detected. Counter: {stall_counter}/{max_stalls}")
                
                if stall_counter >= max_stalls:
                    log_message("Multiple stalls detected, resetting optimizer")
                    
                    # Get current learning rate
                    if hasattr(model.optimizer, 'lr'):
                        current_lr = float(model.optimizer.lr.numpy())
                        # But don't decrease too much
                        new_lr = max(0.0001, current_lr * 0.8)
                        log_message(f"Adjusting learning rate from {current_lr} to {new_lr}")
                    else:
                        new_lr = 0.0005
                        log_message(f"Setting learning rate to {new_lr}")
                    
                    # Recompile with fresh optimizer
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=new_lr),
                        loss='binary_crossentropy',
                        metrics=[
                            'accuracy',
                            tf.keras.metrics.AUC(name='auc'),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall')
                        ]
                    )
                    
                    # Reset counter
                    stall_counter = 0
            else:
                # Reset stall counter if no stall
                stall_counter = 0
                
        except tf.errors.ResourceExhaustedError:
            log_message(f"Resource exhausted, reducing batch size")
            
            # Try with smaller batch
            try:
                mini_batch_size = min(64, len(batch_X))
                history = model.fit(
                    batch_X, batch_y,
                    epochs=1,
                    batch_size=mini_batch_size,
                    class_weight=class_weight,
                    verbose=1
                )
            except Exception as inner_e:
                log_message(f"Error with reduced batch: {inner_e}")
                continue
                
        except Exception as e:
            log_message(f"Error during training: {e}")
            continue
        
        # Clean up
        del batch_X, batch_y
        gc.collect()
        
        # Validate at appropriate intervals
        if (batch_idx + 1) % validation_frequency == 0 or stall_counter > 0 or (batch_idx + 1) == num_batches:
            log_message(f"Validating at batch {batch_idx+1}/{num_batches}")
            
            try:
                val_metrics = model.evaluate(val_X, val_y, verbose=0)
                val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
                metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in val_metrics_dict.items()])
                log_message(f"Validation metrics at batch {batch_idx+1}/{num_batches}:")
                log_message(f"  {metrics_str}")
                
                # Save if better
                val_auc = val_metrics_dict.get('auc', 0)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_path = os.path.join(output_dir, "checkpoints", f"best_model_batch_{batch_idx+1}.h5")
                    model.save(best_model_path)
                    log_message(f"New best model saved with AUC {val_auc:.4f}")
            
            except Exception as e:
                log_message(f"Error during validation: {e}")
        
        # Save checkpoint regularly
        if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == num_batches:
            checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{batch_idx+1}.h5")
            model.save(checkpoint_path)
            log_message(f"Checkpoint saved to {checkpoint_path}")
        
        # Progress tracking
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time
        progress = (batch_idx - start_batch + 1) / (num_batches - start_batch)
        remaining = (elapsed / progress) * (1 - progress) if progress > 0 else 0
        
        log_message(f"Progress: {batch_idx+1}/{num_batches} batches ({progress*100:.1f}%)")
        log_message(f"Batch time: {batch_time:.1f}s")
        log_message(f"Elapsed: {timedelta(seconds=int(elapsed))}, Remaining: {timedelta(seconds=int(remaining))}")
        
        # Reset TensorFlow session periodically to prevent memory growth
        if (batch_idx + 1) % 500 == 0 and batch_idx > 0:
            log_message("Performing session reset")
            
            # Save model
            temp_path = os.path.join(output_dir, "checkpoints", f"temp_reset_{batch_idx+1}.h5")
            model.save(temp_path)
            
            # Clear session
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Reload model
            model = tf.keras.models.load_model(temp_path)
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.h5")
    model.save(final_model_path)
    log_message(f"Training complete. Final model saved to {final_model_path}")
    
    # Return best model if available
    if best_model_path and os.path.exists(best_model_path):
        log_message(f"Loading best model from {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)
        return model, best_model_path
    
    return model, final_model_path

def full_dataset_training(model, X_file, y_file, train_indices, val_indices, test_indices,
                         output_dir, batch_size=256, class_weight=None, start_batch=0):
    """
    Training function that uses the ENTIRE training set without any subsampling,
    while improving efficiency where possible.
    """
    import os
    import numpy as np
    import tensorflow as tf
    import gc
    import time
    import psutil
    from datetime import datetime, timedelta
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, "logs", "training_log.txt")
    def log_message(message):
        with open(log_file, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
        print(message)
    
    log_message("Starting training with FULL training dataset")
    
    # Load data in memory-mapped mode
    X_mmap = np.load(X_file, mmap_mode='r')
    y_mmap = np.load(y_file, mmap_mode='r')
    
    # Configuration for validation
    # Use a subset of validation data to save memory
    val_limit = min(1000, len(val_indices))
    val_indices_subset = val_indices[:val_limit]
    val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
    val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    
    # Improved batch size for better utilization
    # We still use the original batch size, but make sure it's efficient
    effective_batch_size = batch_size
    log_message(f"Using batch size: {effective_batch_size}")
    
    # Calculate total batches
    total_batches = len(train_indices) // effective_batch_size + (1 if len(train_indices) % effective_batch_size > 0 else 0)
    log_message(f"Training on all {total_batches} batches")
    
    # Validation frequency
    validation_frequency = 500  # Validate less frequently to save time
    
    # Try a higher learning rate if starting fresh
    if start_batch == 0:
        # Recompile with a higher learning rate
        log_message("Setting higher learning rate")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Higher learning rate
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
    
    # Load from checkpoint if resuming
    if start_batch > 0:
        checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{start_batch}.h5")
        if os.path.exists(checkpoint_path):
            log_message(f"Loading model from checkpoint {checkpoint_path}")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            log_message("Checkpoint not found, starting with fresh model")
    
    # Training tracking variables
    stall_counter = 0
    max_stalls = 3
    best_val_auc = 0
    best_model_path = None
    start_time = time.time()
    
    # Initial validation
    log_message("Performing initial validation")
    try:
        val_metrics = model.evaluate(val_X, val_y, verbose=0)
        val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in val_metrics_dict.items()])
        log_message(f"Initial validation: {metrics_str}")
        best_val_auc = val_metrics_dict.get('auc', 0)
    except Exception as e:
        log_message(f"Error during initial validation: {e}")
    
    # Main training loop
    for batch_idx in range(start_batch, total_batches):
        batch_start_time = time.time()
        
        # Calculate batch indices
        start_idx = batch_idx * effective_batch_size
        end_idx = min((batch_idx + 1) * effective_batch_size, len(train_indices))
        batch_indices = train_indices[start_idx:end_idx]
        
        # Report memory usage
        current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        log_message(f"Memory before batch load: {current_memory:.1f} MB")
        
        # Load batch data with chunking to avoid memory issues
        try:
            # Load in smaller chunks
            chunk_size = min(64, len(batch_indices))
            batch_X_chunks = []
            batch_y_chunks = []
            
            for i in range(0, len(batch_indices), chunk_size):
                end_i = min(i + chunk_size, len(batch_indices))
                chunk_indices = batch_indices[i:end_i]
                
                chunk_X = np.array([X_mmap[idx] for idx in chunk_indices])
                chunk_y = np.array([y_mmap[idx] for idx in chunk_indices])
                
                batch_X_chunks.append(chunk_X)
                batch_y_chunks.append(chunk_y)
                
                # Clean up chunks immediately
                gc.collect()
            
            # Combine chunks
            batch_X = np.concatenate(batch_X_chunks)
            batch_y = np.concatenate(batch_y_chunks)
            
            # Clean up chunk lists
            del batch_X_chunks, batch_y_chunks
            gc.collect()
            
        except Exception as e:
            log_message(f"Error loading batch data: {e}")
            continue
        
        # Train multiple epochs on this batch for better convergence
        # But limit to 2 epochs to keep it moving
        n_epochs = 2
        log_message(f"Training for {n_epochs} epochs on batch {batch_idx+1}/{total_batches}")
        
        # Try different mini-batch sizes if needed
        mini_batch_sizes = [min(effective_batch_size, 128), 64, 32]
        training_successful = False
        
        for mini_batch_size in mini_batch_sizes:
            if training_successful:
                break
                
            try:
                log_message(f"Trying mini-batch size {mini_batch_size}")
                
                # Train for multiple epochs
                history = model.fit(
                    batch_X, batch_y,
                    epochs=n_epochs,
                    verbose=1,
                    class_weight=class_weight,
                    batch_size=mini_batch_size
                )
                
                # Check for NaN loss
                if np.isnan(history.history['loss'][-1]):
                    log_message(f"NaN loss detected with mini-batch size {mini_batch_size}")
                    continue
                
                # Check for AUC value
                if 'auc' in history.history and history.history['auc'][-1] == 0:
                    stall_counter += 1
                    log_message(f"Warning: AUC is zero, stall detected. Counter: {stall_counter}/{max_stalls}")
                    
                    if stall_counter >= max_stalls:
                        log_message("Multiple stalls detected, resetting optimizer state")
                        
                        # Get current learning rate
                        if hasattr(model.optimizer, 'lr'):
                            current_lr = float(model.optimizer.lr.numpy())
                            # But don't decrease too much
                            new_lr = max(0.0001, current_lr * 0.5)
                            log_message(f"Adjusting learning rate from {current_lr} to {new_lr}")
                        else:
                            new_lr = 0.0005
                            log_message(f"Setting learning rate to {new_lr}")
                        
                        # Recompile with fresh optimizer
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=new_lr),
                            loss='binary_crossentropy',
                            metrics=[
                                'accuracy',
                                tf.keras.metrics.AUC(name='auc'),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall')
                            ]
                        )
                        
                        # Reset counter
                        stall_counter = 0
                else:
                    # Reset stall counter if no stall
                    stall_counter = 0
                
                # Mark as successful
                training_successful = True
                
            except tf.errors.ResourceExhaustedError:
                log_message(f"Resource exhausted with mini-batch size {mini_batch_size}, trying smaller size")
                continue
            except Exception as e:
                log_message(f"Error during training with mini-batch size {mini_batch_size}: {e}")
                continue
        
        # If all mini-batch sizes failed, skip this batch
        if not training_successful:
            log_message(f"All mini-batch sizes failed for batch {batch_idx+1}, skipping")
            continue
        
        # Clean up
        del batch_X, batch_y
        gc.collect()
        
        # Validate at appropriate intervals
        if (batch_idx + 1) % validation_frequency == 0 or stall_counter > 0 or (batch_idx + 1) == total_batches:
            log_message(f"Validating at batch {batch_idx+1}/{total_batches}")
            
            try:
                val_metrics = model.evaluate(val_X, val_y, verbose=0)
                val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
                metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in val_metrics_dict.items()])
                log_message(f"Validation metrics at batch {batch_idx+1}/{total_batches}:")
                log_message(f"  {metrics_str}")
                
                # Save if better
                val_auc = val_metrics_dict.get('auc', 0)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_path = os.path.join(output_dir, "checkpoints", f"best_model_batch_{batch_idx+1}.h5")
                    model.save(best_model_path)
                    log_message(f"New best model saved with AUC {val_auc:.4f}")
            
            except Exception as e:
                log_message(f"Error during validation: {e}")
        
        # Save checkpoint every 1000 batches
        if (batch_idx + 1) % 1000 == 0 or (batch_idx + 1) == total_batches:
            checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{batch_idx+1}.h5")
            model.save(checkpoint_path)
            log_message(f"Checkpoint saved to {checkpoint_path}")
        
        # Progress tracking
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time
        progress = (batch_idx - start_batch + 1) / (total_batches - start_batch)
        remaining = (elapsed / progress) * (1 - progress) if progress > 0 else 0
        
        log_message(f"Progress: {batch_idx+1}/{total_batches} batches ({progress*100:.1f}%)")
        log_message(f"Batch time: {batch_time:.1f}s")
        log_message(f"Elapsed: {timedelta(seconds=int(elapsed))}, Remaining: {timedelta(seconds=int(remaining))}")
        
        # Reset TensorFlow session periodically to prevent memory growth
        if (batch_idx + 1) % 500 == 0 and batch_idx > 0:
            log_message("Performing session reset")
            
            # Save model
            temp_path = os.path.join(output_dir, "checkpoints", f"temp_reset_{batch_idx+1}.h5")
            model.save(temp_path)
            
            # Clear session
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Reload model
            model = tf.keras.models.load_model(temp_path)
            log_message("Session reset complete")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.h5")
    model.save(final_model_path)
    log_message(f"Training complete. Final model saved to {final_model_path}")
    
    # Return best model if available
    if best_model_path and os.path.exists(best_model_path):
        log_message(f"Loading best model from {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)
        return model, best_model_path
    
    return model, final_model_path

def optimized_training(model, X_file, y_file, train_indices, val_indices, test_indices,
                     output_dir, batch_size=256, class_weight=None, start_batch=0):
    """
    Optimized training function focused solely on improving processing efficiency
    without changing the dataset or model architecture.
    """
    import os
    import numpy as np
    import tensorflow as tf
    import gc
    import time
    import psutil
    from datetime import datetime, timedelta
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Load data in memory-mapped mode
    X_mmap = np.load(X_file, mmap_mode='r')
    y_mmap = np.load(y_file, mmap_mode='r')
    
    # Prepare validation data
    val_limit = min(1000, len(val_indices))
    val_indices_subset = val_indices[:val_limit]
    val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
    val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    
    # Calculate total batches
    total_batches = len(train_indices) // batch_size + (1 if len(train_indices) % batch_size > 0 else 0)
    print(f"Training on all {total_batches} batches")
    
    # Load from checkpoint if resuming
    if start_batch > 0:
        checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{start_batch}.h5")
        if os.path.exists(checkpoint_path):
            print(f"Loading model from checkpoint {checkpoint_path}")
            model = tf.keras.models.load_model(checkpoint_path)
    
    # Training tracking variables
    best_val_auc = 0
    best_model_path = None
    start_time = time.time()
    
    # Use smaller individual batch sizes to avoid memory issues
    effective_batch_size = 64  # Keep this smaller for stability
    
    # Initial validation
    print("Performing initial validation")
    val_metrics = model.evaluate(val_X, val_y, verbose=0)
    val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
    metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in val_metrics_dict.items()])
    print(f"Initial validation: {metrics_str}")
    best_val_auc = val_metrics_dict.get('auc', 0)
    
    # Process each batch
    for batch_idx in range(start_batch, total_batches):
        batch_start_time = time.time()
        
        # Calculate batch indices
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
        batch_indices = train_indices[start_idx:end_idx]
        
        # Load batch data more efficiently
        batch_X = []
        batch_y = []
        
        for i in range(0, len(batch_indices), 64):
            end_i = min(i + 64, len(batch_indices))
            indices = batch_indices[i:end_i]
            X_chunk = np.array([X_mmap[idx] for idx in indices])
            y_chunk = np.array([y_mmap[idx] for idx in indices])
            batch_X.append(X_chunk)
            batch_y.append(y_chunk)
        
        batch_X = np.concatenate(batch_X)
        batch_y = np.concatenate(batch_y)
        
        # Train on this batch (using the smaller effective batch size)
        try:
            model.fit(
                batch_X, batch_y,
                epochs=1,
                batch_size=effective_batch_size,
                class_weight=class_weight,
                verbose=1
            )
        except Exception as e:
            print(f"Error training batch: {e}")
            # Try with even smaller batch size if there's an error
            try:
                model.fit(
                    batch_X, batch_y,
                    epochs=1,
                    batch_size=32,
                    class_weight=class_weight,
                    verbose=1
                )
            except:
                continue
        
        # Clean up batch data
        del batch_X, batch_y
        gc.collect()
        
        # Validate and save periodically
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
            # Evaluate on validation set
            val_metrics = model.evaluate(val_X, val_y, verbose=0)
            val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
            
            print(f"Validation at batch {batch_idx+1}/{total_batches}:")
            print(" - ".join([f"{k}: {v:.4f}" for k, v in val_metrics_dict.items()]))
            
            # Save if better
            val_auc = val_metrics_dict.get('auc', 0)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_path = os.path.join(output_dir, "checkpoints", f"best_model_{batch_idx+1}.h5")
                model.save(best_model_path)
                print(f"New best model: AUC = {val_auc:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{batch_idx+1}.h5")
            model.save(checkpoint_path)
        
        # Progress tracking
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time
        progress = (batch_idx - start_batch + 1) / (total_batches - start_batch)
        remaining = (elapsed / progress) * (1 - progress) if progress > 0 else 0
        
        print(f"Progress: {batch_idx+1}/{total_batches} batches ({progress*100:.1f}%)")
        print(f"Batch time: {batch_time:.1f}s")
        print(f"Elapsed: {timedelta(seconds=int(elapsed))}, Remaining: {timedelta(seconds=int(remaining))}")
        
        # Reset session periodically
        if (batch_idx + 1) % 500 == 0:
            print("Resetting TensorFlow session")
            temp_path = os.path.join(output_dir, "checkpoints", f"temp_{batch_idx+1}.h5")
            model.save(temp_path)
            tf.keras.backend.clear_session()
            gc.collect()
            model = tf.keras.models.load_model(temp_path)
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.h5")
    model.save(final_model_path)
    
    # Return best model if available
    if best_model_path and os.path.exists(best_model_path):
        model = tf.keras.models.load_model(best_model_path)
    
    return model, final_model_path

# def fixed_training_with_full_evaluation_verbose(model, X_file, y_file, train_indices, val_indices,...
#                                        output_dir, batch_size=256, class_weight=None, start_batch=...
#                                        metadata_file=None):
#     """
#     Comprehensive training function with stability fixes, integrated evaluation,
#     and extremely verbose forced logging.
#     """
#     import os
#     import numpy as np
#     import tensorflow as tf
#     import gc
#     import time
#     import psutil
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import pandas as pd
#     from sklearn.metrics import (
#         classification_report, confusion_matrix, roc_curve, 
#         auc, precision_recall_curve, average_precision_score, roc_auc_score
#     )
#     from datetime import datetime, timedelta
#     import sys
    
#     # Force print function to ensure output is displayed immediately
#     def force_print(message, also_log=True):
#         print(message, flush=True)
#         if also_log:
#             with open(os.path.join(output_dir, "verbose_log.txt"), "a") as f:
#                 timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 f.write(f"[{timestamp}] {message}\n")
    
#     force_print("="*80)
#     force_print("STARTING EMERGENCY VERBOSE TRAINING FUNCTION")
#     force_print("="*80)
    
#     # Create output directories
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
#     force_print("Output directories created successfully")
    
#     # Load data
#     force_print(f"Attempting to load data from {X_file} and {y_file}")
#     X_mmap = np.load(X_file, mmap_mode='r')
#     force_print(f"X data loaded: shape info available: {hasattr(X_mmap, 'shape')}")
#     y_mmap = np.load(y_file, mmap_mode='r')
#     force_print(f"Y data loaded: shape info available: {hasattr(y_mmap, 'shape')}")
    
#     # Prepare validation data - use FULL validation set
#     force_print(f"Loading validation data... ({len(val_indices)} samples)")
#     val_X = []
#     val_y = []
    
#     # Load validation data in chunks for stability
#     chunk_size = 100
#     for i in range(0, len(val_indices), chunk_size):
#         force_print(f"Loading validation chunk {i//chunk_size + 1}/{(len(val_indices) + chunk_size...
#         end_i = min(i + chunk_size, len(val_indices))
#         indices = val_indices[i:end_i]
#         X_chunk = np.array([X_mmap[idx] for idx in indices])
#         y_chunk = np.array([y_mmap[idx] for idx in indices])
#         val_X.append(X_chunk)
#         val_y.append(y_chunk)
#         force_print(f"  Chunk loaded: {len(X_chunk)} samples")
    
#     val_X = np.concatenate(val_X)
#     val_y = np.concatenate(val_y)
#     force_print(f"Validation data loaded: X shape={val_X.shape}, y shape={val_y.shape}")
    
#     # FIX: Recompile model with stability improvements
#     if start_batch == 0:
#         force_print("Recompiling model with stability improvements")
        
#         # Recompile with improved numeric stability
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(
#                 learning_rate=0.001,
#                 epsilon=1e-7,  # Increased epsilon for numeric stability
#                 clipnorm=1.0   # Gradient clipping for stability
#             ),
#             loss='binary_crossentropy',
#             metrics=[
#                 'accuracy',
# # FIX: Use numerically stable AUC implementation...
#                 tf.keras.metrics.AUC(name='auc', num_thresholds=200, from_logits=False),
#                 tf.keras.metrics.Precision(name='precision'),
#                 tf.keras.metrics.Recall(name='recall')
#             ]
#         )
#         force_print("Model recompilation complete")
    
#     # Calculate total batches - use FULL training set
# total_batches = len(train_indices) // batch_size + (1...
#     force_print(f"Training on all {total_batches} batches of full training set")
    
#     # Load checkpoint if resuming
#     if start_batch > 0:
#         checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{start_batch}.h5")
#         if os.path.exists(checkpoint_path):
#             force_print(f"Loading model from checkpoint {checkpoint_path}")
#             model = tf.keras.models.load_model(checkpoint_path)
#             force_print("Model loaded successfully")
#         else:
#             force_print(f"Checkpoint file not found: {checkpoint_path}")
    
#     # Training variables
#     best_val_auc = 0
#     best_model_path = None
#     start_time = time.time()
    
#     # Initial validation
#     force_print("Performing initial validation")
#     try:
#         val_metrics = model.evaluate(val_X, val_y, verbose=1)
#         val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
# metrics_str = " - ".join([f"{k}: {v:.4f}" for...
#         force_print(f"Initial validation: {metrics_str}")
#         best_val_auc = val_metrics_dict.get('auc', 0)
#     except Exception as e:
#         force_print(f"ERROR during initial validation: {str(e)}")
#         import traceback
#         force_print(traceback.format_exc())
    
#     # Main training loop - process FULL training set
#     for batch_idx in range(start_batch, total_batches):
#         batch_start_time = time.time()
#         force_print(f"="*50)
#         force_print(f"STARTING BATCH {batch_idx+1}/{total_batches}")
        
#         # Calculate batch indices
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
#         batch_indices = train_indices[start_idx:end_idx]
#         force_print(f"Batch indices: {start_idx} to {end_idx} (total: {len(batch_indices)})")
        
#         # Load batch data efficiently
#         force_print("Loading batch data...")
#         batch_X = []
#         batch_y = []
        
#         for i in range(0, len(batch_indices), 64):
#             end_i = min(i + 64, len(batch_indices))
#             indices = batch_indices[i:end_i]
#             force_print(f"  Loading mini-chunk {i}-{end_i} ({len(indices)} samples)")
#             X_chunk = np.array([X_mmap[idx] for idx in indices])
#             y_chunk = np.array([y_mmap[idx] for idx in indices])
#             batch_X.append(X_chunk)
#             batch_y.append(y_chunk)
#             force_print(f"  Mini-chunk loaded: X shape={X_chunk.shape}, y shape={y_chunk.shape}")
        
#         force_print("Concatenating batch chunks...")
#         batch_X = np.concatenate(batch_X)
#         batch_y = np.concatenate(batch_y)
#         force_print(f"Batch data loaded: X shape={batch_X.shape}, y shape={batch_y.shape}")
        
#         # FIX: Check class balance in batch
#         pos_ratio = np.mean(batch_y)
#         force_print(f"Batch class balance: {pos_ratio:.4f} positive, {1-pos_ratio:.4f} negative")
        
#         # Train on batch
#         force_print("Starting training on batch...")
#         try:
#             # FIX: Use more stable mini-batch sizes
#             mini_batch_size = 32  # Smaller, more stable mini-batches
#             force_print(f"Training with mini-batch size: {mini_batch_size}")
            
#             history = model.fit(
#                 batch_X, batch_y,
#                 epochs=1,
#                 batch_size=mini_batch_size,
#                 class_weight=class_weight,
#                 verbose=2  # More verbose output from fit
#             )
            
#             force_print("Training complete for this batch")
#             force_print(f"Training metrics: {history.history}")
            
#             # FIX: Check for zero AUC and immediately fix if detected
#             if 'auc' in history.history and history.history['auc'][-1] == 0:
#                 force_print("WARNING: Zero AUC detected, attempting recovery")
                
#                 # Retry with an even smaller batch size
#                 force_print("Retrying with smaller batch size (16)")
#                 model.fit(
#                     batch_X, batch_y,
#                     epochs=1,
#                     batch_size=16,
#                     class_weight=class_weight,
#                     verbose=2
#                 )
#                 force_print("Recovery attempt completed")
#         except Exception as e:
#             force_print(f"ERROR during training: {str(e)}")
#             import traceback
#             force_print(traceback.format_exc())
            
#             # Try with even smaller batch size
#             force_print("Attempting recovery with very small batch size (16)")
#             try:
#                 model.fit(
#                     batch_X, batch_y,
#                     epochs=1,
#                     batch_size=16,
#                     class_weight=class_weight,
#                     verbose=2
#                 )
#                 force_print("Recovery successful")
#             except Exception as e2:
#                 force_print(f"ERROR during recovery: {str(e2)}")
#                 force_print("Skipping this batch and continuing")
#                 # Skip this batch
#                 continue
        
#         # Clean up
#         force_print("Cleaning up batch data...")
#         del batch_X, batch_y
#         gc.collect()
#         force_print("Cleanup complete")
        
#         # Validate and save periodically
#         if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
#             force_print(f"Performing validation at batch {batch_idx+1}...")
#             # Evaluate on validation set
#             try:
#                 val_metrics = model.evaluate(val_X, val_y, verbose=1)
#                 val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
                
# metrics_str = " - ".join([f"{k}: {v:.4f}" for...
#                 force_print(f"Validation metrics at batch {batch_idx+1}/{total_batches}:")
#                 force_print(f"  {metrics_str}")
                
#                 # Save if better
#                 val_auc = val_metrics_dict.get('auc', 0)
#                 if val_auc > best_val_auc:
#                     best_val_auc = val_auc
#                     best_model_path = os.path.join(output_dir, "checkpoints", f"best_model_{batch_...
#                     force_print(f"New best model: AUC = {val_auc:.4f}, saving to {best_model_path}...
#                     model.save(best_model_path)
#                     force_print("Best model saved successfully")
                
#                 # Save checkpoint
#                 checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{batch_idx...
#                 force_print(f"Saving checkpoint to {checkpoint_path}")
#                 model.save(checkpoint_path)
#                 force_print("Checkpoint saved successfully")
#             except Exception as e:
#                 force_print(f"ERROR during validation: {str(e)}")
#                 import traceback
#                 force_print(traceback.format_exc())
        
#         # Progress tracking
#         batch_time = time.time() - batch_start_time
#         elapsed = time.time() - start_time
# progress = (batch_idx - start_batch + 1)...
# remaining = (elapsed / progress) * (1...
        
#         force_print(f"PROGRESS UPDATE:")
#         force_print(f"  Batch {batch_idx+1}/{total_batches} completed ({progress*100:.1f}%)")
#         force_print(f"  Batch processing time: {batch_time:.1f}s")
#         force_print(f"  Elapsed: {timedelta(seconds=int(elapsed))}")
#         force_print(f"  Estimated remaining: {timedelta(seconds=int(remaining))}")
#         force_print(f"  Memory usage: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 102...
        
#         # Reset session periodically
#         if (batch_idx + 1) % 500 == 0:
#             force_print("Resetting TensorFlow session to free memory")
#             temp_path = os.path.join(output_dir, "checkpoints", f"temp_{batch_idx+1}.h5")
#             force_print(f"Saving temporary model to {temp_path}")
#             model.save(temp_path)
#             force_print("Clearing session")
#             tf.keras.backend.clear_session()
#             gc.collect()
#             force_print("Loading model back")
#             model = tf.keras.models.load_model(temp_path)
#             force_print("Session reset complete")
        
#         force_print(f"BATCH {batch_idx+1} COMPLETED SUCCESSFULLY")
#         force_print("="*50)
    
#     # Save final model
#     force_print("Training complete! Saving final model...")
#     final_model_path = os.path.join(output_dir, "final_model.h5")
#     model.save(final_model_path)
#     force_print(f"Final model saved to {final_model_path}")
    
#     # Rest of the function continues with evaluation...
#     force_print("\n===== STARTING FULL EVALUATION ON ENTIRE TEST SET =====")
#     if best_model_path and os.path.exists(best_model_path):
#         force_print(f"Loading best model from {best_model_path}")
#         model = tf.keras.models.load_model(best_model_path)
#         force_print("Best model loaded successfully")
#     else:
#         force_print("Using final model for evaluation")
    
#     # Prepare test set evaluation
#     force_print(f"Evaluating model on ALL {len(test_indices)} test samples")
    
#     # Process test data in batches
#     batch_size = 1000
#     num_batches = int(np.ceil(len(test_indices) / batch_size))
    
#     all_preds = []
#     all_true = []
#     all_meta = []
    
#     # Load metadata if available
#     metadata = None
#     if metadata_file:
#         import pickle
#         try:
#             force_print(f"Loading metadata from {metadata_file}")
#             with open(metadata_file, "rb") as f:
#                 metadata = pickle.load(f)
#             force_print(f"Metadata loaded successfully - contains data for {len(metadata)} samples...
#         except Exception as e:
#             force_print(f"ERROR loading metadata: {str(e)}")
    
#     # Process test data - use COMPLETE test set
#     force_print(f"Processing test data in {num_batches} batches...")
#     for batch_idx in range(num_batches):
#         force_print(f"Processing test batch {batch_idx+1}/{num_batches}")
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(test_indices))
#         batch_indices = test_indices[start_idx:end_idx]
#         force_print(f"  Test batch indices: {start_idx} to {end_idx} (total: {len(batch_indices)})...
        
#         # Load batch data
#         force_print(f"  Loading test batch data...")
#         batch_X = np.array([X_mmap[idx] for idx in batch_indices])
#         batch_y = np.array([y_mmap[idx] for idx in batch_indices])
#         force_print(f"  Test batch loaded: X shape={batch_X.shape}, y shape={batch_y.shape}")
        
#         # Store metadata if available
#         if metadata:
#             try:
#                 force_print(f"  Processing metadata for this batch...")
#                 batch_meta = [metadata[idx] for idx in batch_indices]
#                 all_meta.extend(batch_meta)
#                 force_print(f"  Metadata processed: {len(batch_meta)} records added")
#             except Exception as e:
#                 force_print(f"  ERROR processing metadata: {str(e)}")
        
#         # Predict
#         force_print(f"  Making predictions...")
#         batch_preds = model.predict(batch_X, verbose=1)
#         force_print(f"  Predictions complete: shape={batch_preds.shape}")
        
#         # Store results
#         all_preds.extend(batch_preds.flatten())
#         all_true.extend(batch_y)
#         force_print(f"  Results stored: total predictions so far: {len(all_preds)}")
        
#         # Clean up
#         del batch_X, batch_y, batch_preds
#         gc.collect()
#         force_print(f"  Batch cleanup complete")
    
#     force_print("All test batches processed")
    
#     # Convert to numpy arrays
#     force_print("Converting results to numpy arrays...")
#     all_preds = np.array(all_preds)
#     all_true = np.array(all_true)
#     force_print(f"Arrays created: predictions={all_preds.shape}, ground truth={all_true.shape}")
    
#     # Create binary predictions
#     force_print("Creating binary predictions...")
#     all_preds_binary = (all_preds > 0.5).astype(int)
#     force_print(f"Binary predictions created: shape={all_preds_binary.shape}")
    
#     # Calculate metrics
#     force_print("Calculating evaluation metrics...")
#     report = classification_report(all_true, all_preds_binary, output_dict=True)
#     report_str = classification_report(all_true, all_preds_binary)
#     conf_matrix = confusion_matrix(all_true, all_preds_binary)
    
#     # Calculate ROC curve and AUC
#     force_print("Calculating ROC curve and AUC...")
#     fpr, tpr, _ = roc_curve(all_true, all_preds)
#     roc_auc = auc(fpr, tpr)
    
#     # Calculate Precision-Recall curve
#     force_print("Calculating Precision-Recall curve...")
#     precision, recall, _ = precision_recall_curve(all_true, all_preds)
#     avg_precision = average_precision_score(all_true, all_preds)
    
#     force_print("\nClassification Report:")
#     force_print(report_str)
    
#     force_print("\nConfusion Matrix:")
#     force_print(conf_matrix)
    
#     force_print(f"\nROC AUC: {roc_auc:.4f}")
#     force_print(f"Average Precision: {avg_precision:.4f}")
    
#     # Save results to file
#     force_print("Saving evaluation results to file...")
#     with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
#         f.write("Classification Report:\n")
#         f.write(report_str)
#         f.write("\n\nConfusion Matrix:\n")
#         f.write(str(conf_matrix))
#         f.write(f"\n\nROC AUC: {roc_auc:.4f}")
#         f.write(f"\nAverage Precision: {avg_precision:.4f}")
#     force_print("Evaluation results saved")
    
#     # Create visualizations
#     force_print("Creating visualizations...")
#     vis_dir = os.path.join(output_dir, "visualizations")
#     os.makedirs(vis_dir, exist_ok=True)
    
#     # 1. Confusion Matrix
#     force_print("Creating confusion matrix visualization...")
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#                xticklabels=['Negative', 'Positive'],
#                yticklabels=['Negative', 'Positive'])
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.title('Confusion Matrix')
#     plt.tight_layout()
#     plt.savefig(os.path.join(vis_dir, "confusion_matrix.png"), dpi=300)
#     plt.close()
#     force_print("Confusion matrix visualization saved")
    
#     # 2. ROC Curve
#     force_print("Creating ROC curve visualization...")
#     plt.figure(figsize=(10, 8))
#     plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     plt.grid(alpha=0.3)
#     plt.savefig(os.path.join(vis_dir, "roc_curve.png"), dpi=300)
#     plt.close()
#     force_print("ROC curve visualization saved")
    
#     # 3. Precision-Recall Curve
#     force_print("Creating Precision-Recall curve visualization...")
#     plt.figure(figsize=(10, 8))
#     plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend(loc="upper right")
#     plt.grid(alpha=0.3)
#     plt.savefig(os.path.join(vis_dir, "precision_recall_curve.png"), dpi=300)
#     plt.close()
#     force_print("Precision-Recall curve visualization saved")
    
#     # 4. Score distribution
#     force_print("Creating score distribution visualization...")
#     plt.figure(figsize=(12, 6))
#     sns.histplot(all_preds[all_true == 0], bins=50, alpha=0.5, label='Negative Class', color='blue...
#     sns.histplot(all_preds[all_true == 1], bins=50, alpha=0.5, label='Positive Class', color='red'...
#     plt.xlabel('Prediction Score')
#     plt.ylabel('Count')
#     plt.title('Distribution of Prediction Scores by Class')
#     plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.savefig(os.path.join(vis_dir, "score_distribution.png"), dpi=300)
#     plt.close()
#     force_print("Score distribution visualization saved")
    
#     # Save predictions
#     force_print("Saving predictions...")
#     if len(all_meta) > 0:
#         force_print("Creating results DataFrame with metadata...")
#         # Create results DataFrame with metadata
#         results_data = []
#         for i in range(len(all_preds)):
#             if i < len(all_meta):
#                 result = {
#                     'prediction': all_preds[i],
#                     'true_label': all_true[i],
#                     'predicted_label': all_preds_binary[i],
#                     'correct': all_preds_binary[i] == all_true[i]
#                 }
                
#                 # Add metadata
#                 meta = all_meta[i]
#                 for key, value in meta.items():
#                     if isinstance(value, (np.integer, np.floating)):
#                         value = float(value)
#                     result[key] = value
                
#                 results_data.append(result)
        
#         # Create DataFrame
#         results_df = pd.DataFrame(results_data)
        
#         # Save to CSV
#         force_print("Saving predictions with metadata to CSV...")
#         results_df.to_csv(os.path.join(output_dir, "test_predictions_with_metadata.csv"), index=Fa...
#         force_print("Predictions with metadata saved")
#     else:
#         # Save simpler predictions
#         force_print("Saving basic predictions...")
#         np.save(os.path.join(output_dir, "test_predictions.npy"), all_preds)
#         np.save(os.path.join(output_dir, "test_true_labels.npy"), all_true)
#         force_print("Basic predictions saved")
    
#     # Feature importance
#     force_print("\nPerforming feature importance analysis...")
    
#     # Process feature importance in batches
#     feature_importances = np.zeros(X_mmap[0].shape[1])
    
#     force_print(f"Feature shape: {X_mmap[0].shape}")
#     feature_names = ['Temperature', 'Temperature Gradient', 'Depth']
#     if X_mmap[0].shape[1] > len(feature_names):
#         for i in range(len(feature_names), X_mmap[0].shape[1]):
#             feature_names.append(f'Feature {i+1}')
#     feature_names = feature_names[:X_mmap[0].shape[1]]
#     force_print(f"Feature names: {feature_names}")
    
#     # Use one small batch for fast feature importance
#     force_print("Calculating quick feature importance on a small sample...")
#     test_subset = test_indices[:1000]  # Just use first 1000 samples
#     force_print(f"Loading {len(test_subset)} samples for feature importance...")
#     X_test_small = np.array([X_mmap[idx] for idx in test_subset])
#     y_test_small = np.array([y_mmap[idx] for idx in test_subset])
#     force_print(f"Test sample loaded: X shape={X_test_small.shape}, y shape={y_test_small.shape}")
    
#     # Get baseline
#     force_print("Getting baseline predictions...")
#     baseline_preds = model.predict(X_test_small, verbose=1)
#     baseline_auc = roc_auc_score(y_test_small, baseline_preds)
#     force_print(f"Baseline AUC: {baseline_auc:.4f}")
    
#     # Test each feature
#     for feature_idx in range(len(feature_names)):
#         force_print(f"Testing importance of feature '{feature_names[feature_idx]}'...")
#         # Permute feature
#         X_permuted = X_test_small.copy()
#         for time_step in range(X_test_small.shape[1]):
#             X_permuted[:, time_step, feature_idx] = np.random.permutation(X_test_small[:, time_ste...
        
#         # Get performance drop
#         force_print(f"  Getting predictions with permuted feature...")
#         perm_preds = model.predict(X_permuted, verbose=1)
#         perm_auc = roc_auc_score(y_test_small, perm_preds)
#         importance = baseline_auc - perm_auc
        
#         feature_importances[feature_idx] = importance
#         force_print(f"  Feature '{feature_names[feature_idx]}' importance: {importance:.4f}")
        
#         # Clean up
#         del X_permuted, perm_preds
#         gc.collect()
    
#     # Create feature importance dataframe
#     force_print("Creating feature importance dataframe...")
#     importance_df = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': feature_importances
#     })
#     importance_df = importance_df.sort_values('Importance', ascending=False)
#     force_print("Feature importance rankings:")
#     for idx, row in importance_df.iterrows():
#         force_print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
#     # Plot feature importance
#     force_print("Creating feature importance visualization...")
#     plt.figure(figsize=(10, 6))
#     plt.bar(importance_df['Feature'], importance_df['Importance'])
#     plt.xlabel('Feature')
#     plt.ylabel('Importance (decrease in AUC)')
#     plt.title('Feature Importance')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.savefig(os.path.join(vis_dir, "feature_importance.png"), dpi=300)
#     plt.close()
#     force_print("Feature importance visualization saved")
    
#     # Save importance results
#     force_print("Saving feature importance results...")
#     importance_df.to_csv(os.path.join(vis_dir, "feature_importance.csv"), index=False)
#     force_print("Feature importance results saved")
    
#     force_print("\n" + "="*30)
#     force_print("EVALUATION COMPLETE!")
#     force_print("="*30)
#     force_print(f"All results and visualizations saved to {output_dir}")
    
#     # Return results
#     final_results = {
#         'accuracy': report['accuracy'],
#         'precision': report['1']['precision'],
#         'recall': report['1']['recall'],
#         'f1': report['1']['f1-score'],
#         'auc': roc_auc,
#         'avg_precision': avg_precision
#     }
    
#     force_print(f"Final results: {final_results}")
#     force_print("FUNCTION EXECUTION COMPLETE")
    
#     return model, final_results

def fixed_training_with_full_evaluation_verbose(model, X_file, y_file, train_indices, val_indices, test_indices,
                                       output_dir, batch_size=256, class_weight=None, start_batch=0,
                                       metadata_file=None):
    """
    Comprehensive training function with stability fixes, integrated evaluation,
    and extremely verbose forced logging.
    """
    import os
    import numpy as np
    import tensorflow as tf
    import gc
    import time
    import psutil
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_curve, 
        auc, precision_recall_curve, average_precision_score, roc_auc_score
    )
    from datetime import datetime, timedelta
    import sys
    
    # Force print function to ensure output is displayed immediately
    def force_print(message, also_log=True):
        print(message, flush=True)
        if also_log:
            with open(os.path.join(output_dir, "verbose_log.txt"), "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
    
    force_print("="*80)
    force_print("STARTING EMERGENCY VERBOSE TRAINING FUNCTION")
    force_print("="*80)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    force_print("Output directories created successfully")
    
    # Load data
    force_print(f"Attempting to load data from {X_file} and {y_file}")
    X_mmap = np.load(X_file, mmap_mode='r')
    force_print(f"X data loaded: shape info available: {hasattr(X_mmap, 'shape')}")
    y_mmap = np.load(y_file, mmap_mode='r')
    force_print(f"Y data loaded: shape info available: {hasattr(y_mmap, 'shape')}")
    
    # Prepare validation data - limit to a reasonable size
    val_limit = min(5000, len(val_indices))
    val_indices_subset = val_indices[:val_limit]
    val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
    val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    force_print(f"Validation data loaded: X shape={val_X.shape}, y shape={val_y.shape}")
    
    # FIX: Recompile model with stability improvements
    if start_batch == 0:
        force_print("Recompiling model with stability improvements")
        
        # Recompile with improved numeric stability
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001,
                epsilon=1e-7,  # Increased epsilon for numeric stability
                clipnorm=1.0   # Gradient clipping for stability
            ),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                # FIX: Use numerically stable AUC implementation with more thresholds
                tf.keras.metrics.AUC(name='auc', num_thresholds=200, from_logits=False),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        force_print("Model recompilation complete")
    
    # Calculate total batches - use FULL training set
    total_batches = len(train_indices) // batch_size + (1 if len(train_indices) % batch_size > 0 else 0)
    force_print(f"Training on all {total_batches} batches of full training set")
    
    # Load checkpoint if resuming
    if start_batch > 0:
        checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{start_batch}.h5")
        if os.path.exists(checkpoint_path):
            force_print(f"Loading model from checkpoint {checkpoint_path}")
            model = tf.keras.models.load_model(checkpoint_path)
            force_print("Model loaded successfully")
        else:
            force_print(f"Checkpoint file not found: {checkpoint_path}")
    
    # Training variables
    best_val_auc = 0
    best_model_path = None
    start_time = time.time()
    validation_frequency = 500  # Validate every 500 batches
    
    # Main training loop - process FULL training set
    for batch_idx in range(start_batch, total_batches):
        batch_start_time = time.time()
        force_print(f"="*50)
        force_print(f"STARTING BATCH {batch_idx+1}/{total_batches}")
        
        # Calculate batch indices
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
        batch_indices = train_indices[start_idx:end_idx]
        force_print(f"Batch indices: {start_idx} to {end_idx} (total: {len(batch_indices)})")
        
        # Load batch data efficiently
        force_print("Loading batch data...")
        batch_X = []
        batch_y = []
        
        for i in range(0, len(batch_indices), 64):
            end_i = min(i + 64, len(batch_indices))
            indices = batch_indices[i:end_i]
            force_print(f"  Loading mini-chunk {i}-{end_i} ({len(indices)} samples)")
            X_chunk = np.array([X_mmap[idx] for idx in indices])
            y_chunk = np.array([y_mmap[idx] for idx in indices])
            batch_X.append(X_chunk)
            batch_y.append(y_chunk)
            force_print(f"  Mini-chunk loaded: X shape={X_chunk.shape}, y shape={y_chunk.shape}")
        
        force_print("Concatenating batch chunks...")
        batch_X = np.concatenate(batch_X)
        batch_y = np.concatenate(batch_y)
        force_print(f"Batch data loaded: X shape={batch_X.shape}, y shape={batch_y.shape}")
        
        # FIX: Check class balance in batch
        pos_ratio = np.mean(batch_y)
        force_print(f"Batch class balance: {pos_ratio:.4f} positive, {1-pos_ratio:.4f} negative")
        
        # Train on batch
        force_print("Starting training on batch...")
        try:
            # FIX: Use more stable mini-batch sizes
            mini_batch_size = 32  # Smaller, more stable mini-batches
            force_print(f"Training with mini-batch size: {mini_batch_size}")
            
            history = model.fit(
                batch_X, batch_y,
                epochs=1,
                batch_size=mini_batch_size,
                class_weight=class_weight,
                verbose=2  # More verbose output from fit
            )
            
            force_print("Training complete for this batch")
            force_print(f"Training metrics: {history.history}")
        except Exception as e:
            force_print(f"ERROR during training: {str(e)}")
            import traceback
            force_print(traceback.format_exc())
            continue
        
        # Clean up
        del batch_X, batch_y
        gc.collect()
        
        # Validate periodically - start only after a few batches
        if batch_idx >= 10 and (batch_idx + 1) % validation_frequency == 0:
            force_print(f"Performing validation at batch {batch_idx+1}...")
            try:
                val_metrics = model.evaluate(val_X, val_y, verbose=1)
                val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
                
                metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in val_metrics_dict.items()])
                force_print(f"Validation metrics at batch {batch_idx+1}/{total_batches}:")
                force_print(f"  {metrics_str}")
                
                # Save if better
                val_auc = val_metrics_dict.get('auc', 0)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_path = os.path.join(output_dir, "checkpoints", f"best_model_{batch_idx+1}.h5")
                    force_print(f"New best model: AUC = {val_auc:.4f}, saving to {best_model_path}")
                    model.save(best_model_path)
                    force_print("Best model saved successfully")
                
                # Save checkpoint
                checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{batch_idx+1}.h5")
                force_print(f"Saving checkpoint to {checkpoint_path}")
                model.save(checkpoint_path)
                force_print("Checkpoint saved successfully")
            except Exception as e:
                force_print(f"ERROR during validation: {str(e)}")
                import traceback
                force_print(traceback.format_exc())
        
        # Progress tracking
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time
        progress = (batch_idx - start_batch + 1) / (total_batches - start_batch)
        remaining = (elapsed / progress) * (1 - progress) if progress > 0 else 0
        
        force_print(f"PROGRESS UPDATE:")
        force_print(f"  Batch {batch_idx+1}/{total_batches} completed ({progress*100:.1f}%)")
        force_print(f"  Batch processing time: {batch_time:.1f}s")
        force_print(f"  Elapsed: {timedelta(seconds=int(elapsed))}")
        force_print(f"  Estimated remaining: {timedelta(seconds=int(remaining))}")
        force_print(f"  Memory usage: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.1f} MB")
        
        # Reset session periodically
        if (batch_idx + 1) % 500 == 0:
            force_print("Resetting TensorFlow session to free memory")
            temp_path = os.path.join(output_dir, "checkpoints", f"temp_{batch_idx+1}.h5")
            force_print(f"Saving temporary model to {temp_path}")
            model.save(temp_path)
            force_print("Clearing session")
            tf.keras.backend.clear_session()
            gc.collect()
            force_print("Loading model back")
            model = tf.keras.models.load_model(temp_path)
            force_print("Session reset complete")
        
        force_print(f"BATCH {batch_idx+1} COMPLETED SUCCESSFULLY")
        force_print("="*50)
    
    # Save final model
    force_print("Training complete! Saving final model...")
    final_model_path = os.path.join(output_dir, "final_model.h5")
    model.save(final_model_path)
    force_print(f"Final model saved to {final_model_path}")
    
    # Rest of the function continues with evaluation...
    force_print("\n===== STARTING FULL EVALUATION ON ENTIRE TEST SET =====")
    if best_model_path and os.path.exists(best_model_path):
        force_print(f"Loading best model from {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)
        force_print("Best model loaded successfully")
    else:
        force_print("Using final model for evaluation")
    
    # Prepare test set evaluation
    force_print(f"Evaluating model on ALL {len(test_indices)} test samples")
    
    # Process test data in batches
    batch_size = 1000
    num_batches = int(np.ceil(len(test_indices) / batch_size))
    
    all_preds = []
    all_true = []
    all_meta = []
    
    # Load metadata if available
    metadata = None
    if metadata_file:
        import pickle
        try:
            force_print(f"Loading metadata from {metadata_file}")
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
            force_print(f"Metadata loaded successfully - contains data for {len(metadata)} samples")
        except Exception as e:
            force_print(f"ERROR loading metadata: {str(e)}")
    
    # Process test data - use COMPLETE test set
    force_print(f"Processing test data in {num_batches} batches...")
    for batch_idx in range(num_batches):
        force_print(f"Processing test batch {batch_idx+1}/{num_batches}")
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(test_indices))
        batch_indices = test_indices[start_idx:end_idx]
        force_print(f"  Test batch indices: {start_idx} to {end_idx} (total: {len(batch_indices)})")
        
        # Load batch data
        force_print(f"  Loading test batch data...")
        batch_X = np.array([X_mmap[idx] for idx in batch_indices])
        batch_y = np.array([y_mmap[idx] for idx in batch_indices])
        force_print(f"  Test batch loaded: X shape={batch_X.shape}, y shape={batch_y.shape}")
        
        # Store metadata if available
        if metadata:
            try:
                force_print(f"  Processing metadata for this batch...")
                batch_meta = [metadata[idx] for idx in batch_indices]
                all_meta.extend(batch_meta)
                force_print(f"  Metadata processed: {len(batch_meta)} records added")
            except Exception as e:
                force_print(f"  ERROR processing metadata: {str(e)}")
        
        # Predict
        force_print(f"  Making predictions...")
        batch_preds = model.predict(batch_X, verbose=1)
        force_print(f"  Predictions complete: shape={batch_preds.shape}")
        
        # Store results
        all_preds.extend(batch_preds.flatten())
        all_true.extend(batch_y)
        force_print(f"  Results stored: total predictions so far: {len(all_preds)}")
        
        # Clean up
        del batch_X, batch_y, batch_preds
        gc.collect()
        force_print(f"  Batch cleanup complete")
    
    force_print("All test batches processed")
    
    # Convert to numpy arrays
    force_print("Converting results to numpy arrays...")
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    force_print(f"Arrays created: predictions={all_preds.shape}, ground truth={all_true.shape}")
    
    # Create binary predictions
    force_print("Creating binary predictions...")
    all_preds_binary = (all_preds > 0.5).astype(int)
    force_print(f"Binary predictions created: shape={all_preds_binary.shape}")
    
    # Calculate metrics
    force_print("Calculating evaluation metrics...")
    report = classification_report(all_true, all_preds_binary, output_dict=True)
    report_str = classification_report(all_true, all_preds_binary)
    conf_matrix = confusion_matrix(all_true, all_preds_binary)
    
    # Calculate ROC curve and AUC
    force_print("Calculating ROC curve and AUC...")
    fpr, tpr, _ = roc_curve(all_true, all_preds)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Precision-Recall curve
    force_print("Calculating Precision-Recall curve...")
    precision, recall, _ = precision_recall_curve(all_true, all_preds)
    avg_precision = average_precision_score(all_true, all_preds)
    
    force_print("\nClassification Report:")
    force_print(report_str)
    
    force_print("\nConfusion Matrix:")
    force_print(conf_matrix)
    
    force_print(f"\nROC AUC: {roc_auc:.4f}")
    force_print(f"Average Precision: {avg_precision:.4f}")
    
    # Save results to file
    force_print("Saving evaluation results to file...")
    with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
        f.write("Classification Report:\n")
        f.write(report_str)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write(f"\n\nROC AUC: {roc_auc:.4f}")
        f.write(f"\nAverage Precision: {avg_precision:.4f}")
    force_print("Evaluation results saved")
    
    # Create visualizations
    force_print("Creating visualizations...")
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    force_print("Creating confusion matrix visualization...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    force_print("Confusion matrix visualization saved")
    
    # 2. ROC Curve
    force_print("Creating ROC curve visualization...")
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "roc_curve.png"), dpi=300)
    plt.close()
    force_print("ROC curve visualization saved")
    
    # 3. Precision-Recall Curve
    force_print("Creating Precision-Recall curve visualization...")
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "precision_recall_curve.png"), dpi=300)
    plt.close()
    force_print("Precision-Recall curve visualization saved")
    
    # 4. Score distribution
    force_print("Creating score distribution visualization...")
    plt.figure(figsize=(12, 6))
    sns.histplot(all_preds[all_true == 0], bins=50, alpha=0.5, label='Negative Class', color='blue')
    sns.histplot(all_preds[all_true == 1], bins=50, alpha=0.5, label='Positive Class', color='red')
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Scores by Class')
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "score_distribution.png"), dpi=300)
    plt.close()
    force_print("Score distribution visualization saved")
    
    # Save predictions
    force_print("Saving predictions...")
    if len(all_meta) > 0:
        force_print("Creating results DataFrame with metadata...")
        # Create results DataFrame with metadata
        results_data = []
        for i in range(len(all_preds)):
            if i < len(all_meta):
                result = {
                    'prediction': all_preds[i],
                    'true_label': all_true[i],
                    'predicted_label': all_preds_binary[i],
                    'correct': all_preds_binary[i] == all_true[i]
                }
                
                # Add metadata
                meta = all_meta[i]
                for key, value in meta.items():
                    if isinstance(value, (np.integer, np.floating)):
                        value = float(value)
                    result[key] = value
                
                results_data.append(result)
        
        # Create DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        force_print("Saving predictions with metadata to CSV...")
        results_df.to_csv(os.path.join(output_dir, "test_predictions_with_metadata.csv"), index=False)
        force_print("Predictions with metadata saved")
    else:
        # Save simpler predictions
        force_print("Saving basic predictions...")
        np.save(os.path.join(output_dir, "test_predictions.npy"), all_preds)
        np.save(os.path.join(output_dir, "test_true_labels.npy"), all_true)
        force_print("Basic predictions saved")
    
    # Feature importance
    force_print("\nPerforming feature importance analysis...")
    
    # Process feature importance in batches
    feature_importances = np.zeros(X_mmap[0].shape[1])
    
    force_print(f"Feature shape: {X_mmap[0].shape}")
    feature_names = ['Temperature', 'Temperature Gradient', 'Depth']
    if X_mmap[0].shape[1] > len(feature_names):
        for i in range(len(feature_names), X_mmap[0].shape[1]):
            feature_names.append(f'Feature {i+1}')
    feature_names = feature_names[:X_mmap[0].shape[1]]
    force_print(f"Feature names: {feature_names}")
    
    # Use one small batch for fast feature importance
    force_print("Calculating quick feature importance on a small sample...")
    test_subset = test_indices[:1000]  # Just use first 1000 samples
    force_print(f"Loading {len(test_subset)} samples for feature importance...")
    X_test_small = np.array([X_mmap[idx] for idx in test_subset])
    y_test_small = np.array([y_mmap[idx] for idx in test_subset])
    force_print(f"Test sample loaded: X shape={X_test_small.shape}, y shape={y_test_small.shape}")
    
    # Get baseline
    force_print("Getting baseline predictions...")
    baseline_preds = model.predict(X_test_small, verbose=1)
    baseline_auc = roc_auc_score(y_test_small, baseline_preds)
    force_print(f"Baseline AUC: {baseline_auc:.4f}")
    
    # Test each feature
    for feature_idx in range(len(feature_names)):
        force_print(f"Testing importance of feature '{feature_names[feature_idx]}'...")
        # Permute feature
        X_permuted = X_test_small.copy()
        for time_step in range(X_test_small.shape[1]):
            X_permuted[:, time_step, feature_idx] = np.random.permutation(X_test_small[:, time_step, feature_idx])
        
        # Get performance drop
        force_print(f"  Getting predictions with permuted feature...")
        perm_preds = model.predict(X_permuted, verbose=1)
        perm_auc = roc_auc_score(y_test_small, perm_preds)
        importance = baseline_auc - perm_auc
        
        feature_importances[feature_idx] = importance
        force_print(f"  Feature '{feature_names[feature_idx]}' importance: {importance:.4f}")
        
        # Clean up
        del X_permuted, perm_preds
        gc.collect()
    
    # Create feature importance dataframe
    force_print("Creating feature importance dataframe...")
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    force_print("Feature importance rankings:")
    for idx, row in importance_df.iterrows():
        force_print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Plot feature importance
    force_print("Creating feature importance visualization...")
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Feature')
    plt.ylabel('Importance (decrease in AUC)')
    plt.title('Feature Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "feature_importance.png"), dpi=300)
    plt.close()
    force_print("Feature importance visualization saved")
    
    # Save importance results
    force_print("Saving feature importance results...")
    importance_df.to_csv(os.path.join(vis_dir, "feature_importance.csv"), index=False)
    force_print("Feature importance results saved")
    
    force_print("\n" + "="*30)
    force_print("EVALUATION COMPLETE!")
    force_print("="*30)
    force_print(f"All results and visualizations saved to {output_dir}")
    
    # Return results
    final_results = {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score'],
        'auc': roc_auc,
        'avg_precision': avg_precision
    }
    
    force_print(f"Final results: {final_results}")
    force_print("FUNCTION EXECUTION COMPLETE")
    
    return model, final_results

def ultra_fast_training(model, X_file, y_file, train_indices, val_indices, test_indices,
                       output_dir, batch_size=512, class_weight=None, start_batch=0):
    """
    Ultrafast training function with minimal overhead and maximum efficiency
    """
    import os
    import numpy as np
    import tensorflow as tf
    import gc
    import time
    import psutil
    from datetime import datetime, timedelta
    
    # Disable verbose logging
    tf.get_logger().setLevel('ERROR')
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Memory-mapped data access
    X_mmap = np.load(X_file, mmap_mode='r')
    y_mmap = np.load(y_file, mmap_mode='r')
    
    # Efficient validation data loading
    val_limit = min(2000, len(val_indices))
    val_indices_subset = val_indices[:val_limit]
    val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
    val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    
    # Efficient model compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001, 
            clipnorm=1.0  # Gradient clipping
        ),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    # Calculate total batches
    total_batches = len(train_indices) // batch_size + (1 if len(train_indices) % batch_size > 0 else 0)
    
    # Tracking variables
    best_val_auc = 0
    best_model_path = None
    start_time = time.time()
    
    # Efficient batch processing
    for batch_idx in range(start_batch, total_batches):
        # Calculate batch indices
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
        batch_indices = train_indices[start_idx:end_idx]
        
        # Direct batch loading - avoid chunking
        try:
            batch_X = np.array([X_mmap[idx] for idx in batch_indices])
            batch_y = np.array([y_mmap[idx] for idx in batch_indices])
        except Exception as e:
            print(f"Error loading batch {batch_idx}: {e}")
            continue
        
        # Train on batch
        try:
            history = model.fit(
                batch_X, batch_y,
                epochs=1,
                batch_size=min(len(batch_X), 128),  # Adaptive batch size
                class_weight=class_weight,
                verbose=0  # Minimal logging
            )
        except Exception as e:
            print(f"Training error on batch {batch_idx}: {e}")
            continue
        
        # Periodic validation and checkpointing
        if batch_idx % 100 == 0:
            try:
                # Validate
                val_metrics = model.evaluate(val_X, val_y, verbose=0)
                val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
                val_auc = val_metrics_dict.get('auc', 0)
                
                # Save best model
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_path = os.path.join(output_dir, f"best_model_batch_{batch_idx}.h5")
                    model.save(best_model_path)
                
                # Save periodic checkpoint
                checkpoint_path = os.path.join(output_dir, f"checkpoint_batch_{batch_idx}.h5")
                model.save(checkpoint_path)
            except Exception as e:
                print(f"Validation error at batch {batch_idx}: {e}")
        
        # Memory and progress management
        del batch_X, batch_y
        gc.collect()
        
        # Progress tracking (minimal output)
        if batch_idx % 50 == 0:
            elapsed = time.time() - start_time
            progress = (batch_idx - start_batch + 1) / (total_batches - start_batch)
            remaining = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            print(f"Batch {batch_idx+1}/{total_batches} - Progress: {progress*100:.1f}% - Elapsed: {timedelta(seconds=int(elapsed))}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.h5")
    model.save(final_model_path)
    
    # Return best or final model
    if best_model_path and os.path.exists(best_model_path):
        return tf.keras.models.load_model(best_model_path), best_model_path
    
    return model, final_model_path

def hyper_fast_training(model, X_file, y_file, train_indices, val_indices, test_indices,
                       output_dir, batch_size=1024, class_weight=None, start_batch=0):
    """
    Hyper-optimized training function with absolute minimal overhead and device handling
    """
    import os
    import numpy as np
    import tensorflow as tf
    import gc
    import time
    from datetime import datetime, timedelta
    
    # Extreme logging suppression
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Logging function with minimal overhead
    def quick_log(message):
        print(message, flush=True)
        with open(os.path.join(output_dir, "training_log.txt"), "a") as f:
            f.write(f"{datetime.now()}: {message}\n")
    
    # Extremely efficient data loading
    try:
        X_mmap = np.load(X_file, mmap_mode='r')
        y_mmap = np.load(y_file, mmap_mode='r')
    except Exception as e:
        quick_log(f"CRITICAL DATA LOADING ERROR: {e}")
        return model, None
    
    # Efficient validation subset
    val_indices_subset = val_indices[:min(2000, len(val_indices))]
    
    # Use explicit device context
    with tf.device('/GPU:0'):
        val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
        val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    
    # Total batches calculation
    total_batches = len(train_indices) // batch_size + (1 if len(train_indices) % batch_size > 0 else 0)
    quick_log(f"Total batches: {total_batches}, Batch size: {batch_size}")
    
    # Performance tracking
    start_time = time.time()
    best_val_auc = 0
    best_model_path = None
    
    # Main training loop with extreme efficiency
    for batch_idx in range(start_batch, total_batches):
        # Batch indices
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
        batch_indices = train_indices[start_idx:end_idx]
        
        # Direct, fast data loading with device context
        with tf.device('/GPU:0'):
            try:
                batch_X = np.array([X_mmap[idx] for idx in batch_indices])
                batch_y = np.array([y_mmap[idx] for idx in batch_indices])
            except Exception as e:
                quick_log(f"Error loading batch {batch_idx}: {e}")
                continue
            
            # Train with minimal overhead
            try:
                history = model.fit(
                    batch_X, batch_y,
                    epochs=1,
                    batch_size=min(len(batch_X), 256),  # Adaptive batch size
                    class_weight=class_weight,
                    verbose=0  # Absolute minimal logging
                )
            except Exception as e:
                quick_log(f"Training error on batch {batch_idx}: {e}")
                continue
        
        # Periodic validation (every 100 batches)
        if batch_idx % 100 == 0:
            try:
                with tf.device('/GPU:0'):
                    val_metrics = model.evaluate(val_X, val_y, verbose=0)
                    val_auc = val_metrics[model.metrics_names.index('auc')]
                
                # Model saving
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_path = os.path.join(output_dir, f"best_model_batch_{batch_idx}.h5")
                    model.save(best_model_path)
            except Exception as e:
                quick_log(f"Validation error at batch {batch_idx}: {e}")
        
        # Progress reporting (minimal)
        if batch_idx % 50 == 0:
            elapsed = time.time() - start_time
            progress = (batch_idx - start_batch + 1) / (total_batches - start_batch)
            remaining = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            quick_log(f"Batch {batch_idx+1}/{total_batches} - Progress: {progress*100:.1f}% - Elapsed: {timedelta(seconds=int(elapsed))}")
        
        # Aggressive memory management
        del batch_X, batch_y
        gc.collect()
    
    # Final model save
    final_model_path = os.path.join(output_dir, "final_model.h5")
    model.save(final_model_path)
    
    # Return best or final model
    if best_model_path and os.path.exists(best_model_path):
        return tf.keras.models.load_model(best_model_path), best_model_path
    
    return model, final_model_path

def ultra_robust_training(model, X_file, y_file, train_indices, val_indices, test_indices,
                          output_dir, batch_size=512, max_epochs=100, class_weight=None):
    """
    ULTRA-ROBUST TRAINING FUNCTION WITH MAXIMUM ERROR RECOVERY AND DIAGNOSTICS
    """
    import os
    import numpy as np
    import tensorflow as tf
    import gc
    import time
    import psutil
    import json
    from datetime import datetime, timedelta
    import traceback
    import sys
    
    # Comprehensive logging setup
    log_file = os.path.join(output_dir, "ultra_robust_training_log.txt")
    
    def robust_log(message, level="INFO"):
        """
        Logging function with multiple output channels
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        
        # Print to console
        print(log_message, flush=True)
        
        # Write to log file
        try:
            with open(log_file, "a") as f:
                f.write(log_message + "\n")
        except Exception as log_error:
            print(f"ERROR LOGGING: {log_error}")
    
    # Critical error handling wrapper
    def critical_error_handler(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                robust_log(f"CRITICAL ERROR IN {func.__name__}: {e}", "CRITICAL")
                robust_log(traceback.format_exc(), "CRITICAL")
                
                # Create error snapshot
                error_snapshot = {
                    "timestamp": datetime.now().isoformat(),
                    "function": func.__name__,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                
                # Save error snapshot
                try:
                    with open(os.path.join(output_dir, f"error_snapshot_{int(time.time())}.json"), "w") as f:
                        json.dump(error_snapshot, f, indent=4)
                except:
                    pass
                
                # Re-raise the exception
                raise
        return wrapper
    
    # Comprehensive memory and device management
    @critical_error_handler
    def configure_tensorflow_environment():
        """
        Ultra-detailed TensorFlow and GPU configuration
        """
        # Disable verbose logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')
        
        # Identify and configure devices
        physical_devices = tf.config.list_physical_devices()
        robust_log(f"Available Physical Devices: {physical_devices}")
        
        gpu_devices = tf.config.list_physical_devices('GPU')
        for device in gpu_devices:
            try:
                # Enable memory growth
                tf.config.experimental.set_memory_growth(device, True)
                robust_log(f"Memory growth enabled for {device}")
                
                # Optional: Set virtual device
                tf.config.experimental.set_virtual_device_configuration(
                    device,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4)]  # 4GB
                )
            except Exception as e:
                robust_log(f"Could not configure {device}: {e}", "WARNING")
        
        # Enable soft device placement
        tf.config.set_soft_device_placement(True)
    
    # Memory-safe data loading
    @critical_error_handler
    def load_data_safely(X_file, y_file, train_indices, val_indices):
        """
        Safe data loading with comprehensive error handling
        """
        # Memory-mapped data access
        X_mmap = np.load(X_file, mmap_mode='r')
        y_mmap = np.load(y_file, mmap_mode='r')
        
        # Safe validation subset selection
        val_limit = min(2000, len(val_indices))
        val_indices_subset = val_indices[:val_limit]
        
        # Load validation data
        val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
        val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
        
        robust_log(f"Validation data shape: X {val_X.shape}, y {val_y.shape}")
        
        return X_mmap, y_mmap, val_X, val_y
    
    # Comprehensive training loop
    @critical_error_handler
    def execute_training_loop(model, X_mmap, y_mmap, train_indices, val_X, val_y, output_dir):
        """
        Advanced training loop with maximum recovery capabilities
        """
        # Training configuration
        total_batches = len(train_indices) // batch_size + (1 if len(train_indices) % batch_size > 0 else 0)
        robust_log(f"Total batches: {total_batches}, Batch size: {batch_size}")
        
        # Performance tracking
        best_val_auc = 0
        best_model_path = None
        training_history = []
        start_time = time.time()
        
        # Callbacks for advanced monitoring
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', 
                patience=15, 
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', 
                factor=0.5, 
                patience=7
            )
        ]
        
        # Multi-epoch training strategy
        for epoch in range(max_epochs):
            robust_log(f"Starting Epoch {epoch + 1}/{max_epochs}")
            
            # Shuffle training indices to prevent overfitting
            np.random.shuffle(train_indices)
            
            # Process batches
            epoch_losses = []
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
                batch_indices = train_indices[start_idx:end_idx]
                
                # Safe batch loading
                try:
                    batch_X = np.array([X_mmap[idx] for idx in batch_indices])
                    batch_y = np.array([y_mmap[idx] for idx in batch_indices])
                except Exception as batch_load_error:
                    robust_log(f"Batch loading error: {batch_load_error}", "WARNING")
                    continue
                
                # Train on batch
                try:
                    history = model.fit(
                        batch_X, batch_y,
                        validation_data=(val_X, val_y),
                        epochs=1,
                        batch_size=min(len(batch_X), 128),
                        verbose=0,
                        callbacks=callbacks
                    )
                    
                    # Track batch performance
                    batch_loss = history.history.get('loss', [np.nan])[0]
                    epoch_losses.append(batch_loss)
                    
                    # Periodic validation and checkpointing
                    if batch_idx % 50 == 0:
                        val_metrics = model.evaluate(val_X, val_y, verbose=0)
                        val_auc = val_metrics[model.metrics_names.index('auc')]
                        
                        if val_auc > best_val_auc:
                            best_val_auc = val_auc
                            best_model_path = os.path.join(output_dir, f"best_model_epoch_{epoch+1}_batch_{batch_idx}.h5")
                            model.save(best_model_path)
                            robust_log(f"New best model saved with AUC: {best_val_auc}")
                
                except Exception as train_error:
                    robust_log(f"Training error in batch {batch_idx}: {train_error}", "ERROR")
                    continue
                
                # Memory management
                del batch_X, batch_y
                gc.collect()
            
            # Epoch summary
            mean_loss = np.mean(epoch_losses)
            robust_log(f"Epoch {epoch + 1} Summary: Mean Loss = {mean_loss}")
            training_history.append({
                "epoch": epoch + 1,
                "mean_loss": mean_loss,
                "best_val_auc": best_val_auc
            })
            
            # Early stopping condition
            if len(training_history) > 10 and all(
                history['mean_loss'] > mean_loss * 1.1 
                for history in training_history[-10:]
            ):
                robust_log("Potential training stagnation detected. Stopping.")
                break
        
        return model, best_model_path, training_history
    
    # MAIN EXECUTION PIPELINE
    try:
        # Configure TensorFlow
        configure_tensorflow_environment()
        
        # Load data
        X_mmap, y_mmap, val_X, val_y = load_data_safely(X_file, y_file, train_indices, val_indices)
        
        # Execute training
        trained_model, best_model_path, training_log = execute_training_loop(
            model, X_mmap, y_mmap, train_indices, val_X, val_y, output_dir
        )
        
        # Save training history
        with open(os.path.join(output_dir, "training_history.json"), "w") as f:
            json.dump(training_log, f, indent=4)
        
        return trained_model, best_model_path
    
    except Exception as pipeline_error:
        robust_log(f"PIPELINE EXECUTION FAILED: {pipeline_error}", "CRITICAL")
        robust_log(traceback.format_exc(), "CRITICAL")
        raise

# from tensorflow.keras.models import load_model

# # Load your current model and continue with fixed training
# checkpoint_path = "/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/improved_model...
# model = load_model(checkpoint_path)

import tensorflow as tf
import numpy as np

def build_robust_device_aware_model(input_shape, include_moisture=True):
    """
    ConvLSTM2D model with explicit device handling and placement strategies
    """
    with tf.device('/GPU:0'):  # Explicitly use GPU
        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Reshape for ConvLSTM2D
        x = tf.keras.layers.Reshape((input_shape[0], 1, 1, input_shape[1]))(inputs)
        
        # ConvLSTM2D with careful device placement
        def create_convlstm_layer():
            return tf.keras.layers.ConvLSTM2D(
                filters=64,
                kernel_size=(3, 1),
                padding='same',
                return_sequences=True,
                activation='tanh',
                recurrent_dropout=0.2,
                kernel_regularizer=tf.keras.regularizers.l2(0.0001)
            )
        
        # Use strategy to handle potential device conflicts
        try:
            convlstm = create_convlstm_layer()(x)
        except Exception as e:
            print(f"ConvLSTM2D placement error: {e}")
            # Fallback strategy
            with tf.device('/CPU:0'):
                convlstm = create_convlstm_layer()(x)
        
        # Reshape back
        convlstm = tf.keras.layers.Reshape((input_shape[0], 64))(convlstm)
        
        # Advanced feature extraction
        # Parallel convolutional paths
        cnn_branches = []
        for kernel_size in [3, 5, 7]:
            branch = tf.keras.layers.Conv1D(
                filters=32, 
                kernel_size=kernel_size, 
                padding='same', 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.0001)
            )(inputs)
            cnn_branches.append(branch)
        
        # Combine CNN branches
        x = tf.keras.layers.Concatenate()(cnn_branches + [convlstm])
        
        # Global temporal features
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        
        # Regularization and classification layers
        x = tf.keras.layers.Dense(
            128, 
            activation='relu', 
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Dense(
            64, 
            activation='relu', 
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            1, 
            activation='sigmoid', 
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )(x)
        
        # Create model with explicit device handling
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Advanced compilation with device-aware settings
        def compile_with_device_handling(m):
            try:
                with tf.device('/GPU:0'):
                    m.compile(
                        optimizer=tf.keras.optimizers.Adam(
                            learning_rate=0.001,
                            clipnorm=1.0,  # Gradient clipping
                            amsgrad=True   # Advanced optimizer
                        ),
                        loss='binary_crossentropy',
                        metrics=[
                            'accuracy',
                            tf.keras.metrics.AUC(name='auc', num_thresholds=200),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall')
                        ]
                    )
            except Exception as compilation_error:
                print(f"GPU compilation error: {compilation_error}")
                # Fallback to CPU compilation
                with tf.device('/CPU:0'):
                    m.compile(
                        optimizer=tf.keras.optimizers.Adam(
                            learning_rate=0.001,
                            clipnorm=1.0,
                            amsgrad=True
                        ),
                        loss='binary_crossentropy',
                        metrics=[
                            'accuracy',
                            tf.keras.metrics.AUC(name='auc', num_thresholds=200),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall')
                        ]
                    )
            return m
        
        # Apply compilation
        model = compile_with_device_handling(model)
        
        return model

def configure_tensorflow_environment():
    """
    Comprehensive TensorFlow device and memory configuration
    """
    # Disable verbose logging
    tf.get_logger().setLevel('ERROR')
    
    # List available devices
    physical_devices = tf.config.list_physical_devices()
    print("Available Physical Devices:", physical_devices)
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("Available GPUs:", gpu_devices)
    
    try:
        # Attempt to use first GPU with memory growth
        if gpu_devices:
            tf.config.experimental.set_memory_growth(gpu_devices[0], True)
            print(f"Memory growth enabled for {gpu_devices[0]}")
        
        # Set virtual device configuration
        if gpu_devices:
            tf.config.experimental.set_virtual_device_configuration(
                gpu_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4)]  # 4GB
            )
            print("Virtual GPU configuration applied")
        
        # Enable soft device placement
        tf.config.set_soft_device_placement(True)
        print("Soft device placement enabled")
    
    except Exception as e:
        print(f"Device configuration error: {e}")
    
    return physical_devices

# Patch for potential device placement issues
def patch_device_placement_errors():
    """
    Advanced patch to handle TensorFlow device placement conflicts
    """
    def patch_conv_lstm_device_placement():
        # Monkey patch to handle ConvLSTM2D device conflicts
        original_call = tf.keras.layers.ConvLSTM2D.__call__
        
        def patched_call(self, inputs, *args, **kwargs):
            try:
                return original_call(self, inputs, *args, **kwargs)
            except Exception as e:
                print(f"ConvLSTM2D placement error: {e}")
                # Attempt fallback strategies
                with tf.device('/CPU:0'):
                    return original_call(self, inputs, *args, **kwargs)
        
        tf.keras.layers.ConvLSTM2D.__call__ = patched_call
    
    # Apply patches
    patch_conv_lstm_device_placement()
    print("Device placement error patches applied")

# Comprehensive initialization
def initialize_robust_training_environment():
    """
    Initialize a robust training environment with advanced error handling
    """
    # Configure TensorFlow environment
    devices = configure_tensorflow_environment()
    
    # Apply device placement patches
    patch_device_placement_errors()
    
    return devices

def ultra_robust_training_fixed(model, X_file, y_file, train_indices, val_indices, test_indices,
                          output_dir, batch_size=512, max_epochs=100, class_weight=None):
    """
    Fixed ULTRA-ROBUST TRAINING FUNCTION that prevents batch stalling issues
    """
    import os
    import numpy as np
    import tensorflow as tf
    import gc
    import time
    import psutil
    import json
    from datetime import datetime, timedelta
    import traceback
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Logging setup
    log_file = os.path.join(output_dir, "ultra_robust_training_log.txt")
    
    def robust_log(message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message, flush=True)
        with open(log_file, "a") as f:
            f.write(log_message + "\n")
    
    # CRITICAL FIX: Reset TensorFlow's internal state
    def reset_tensorflow():
        robust_log("Performing complete TensorFlow reset")
        tf.keras.backend.clear_session()
        gc.collect()
        
        # CRITICAL FIX: Force GPU memory cleanup
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.reset_memory_stats(gpu)
                robust_log(f"Reset memory stats for {gpu}")
            except:
                robust_log(f"Could not reset memory stats for {gpu}", "WARNING")
        return

    # Configure TensorFlow
    robust_log("Configuring TensorFlow environment")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    # NEW FIX: Set memory growth at the beginning
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            robust_log(f"Memory growth enabled for {gpu}")
        except:
            robust_log(f"Could not set memory growth for {gpu}", "WARNING")
    
    # CRITICAL FIX: Set seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Load data safely
    robust_log("Loading data files")
    X_mmap = np.load(X_file, mmap_mode='r')
    y_mmap = np.load(y_file, mmap_mode='r')
    
    # Smaller validation set
    val_limit = min(1000, len(val_indices))
    val_indices_subset = val_indices[:val_limit]
    val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
    val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    robust_log(f"Validation data loaded with shape X:{val_X.shape}, y:{val_y.shape}")
    
    # NEW FIX: Recompile model with numeric stability improvements
    robust_log("Recompiling model with stability improvements")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            epsilon=1e-7,
            clipnorm=1.0,
            clipvalue=0.5  # NEW: Clip individual gradients
        ),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', num_thresholds=200),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    # Main training configuration
    total_batches = len(train_indices) // batch_size + (1 if len(train_indices) % batch_size > 0 else 0)
    robust_log(f"Training configuration: {total_batches} batches with size {batch_size}")
    
    # Performance tracking
    best_val_auc = 0
    best_model_path = None
    training_history = []
    start_time = time.time()
    
    # CRITICAL FIX: State tracking variables
    stall_counter = 0
    reset_interval = 200  # Reset every 200 batches regardless of status
    last_improvement_batch = 0
    
    # Main training loop with CRITICAL ANTI-STALL MEASURES
    for epoch in range(max_epochs):
        robust_log(f"Starting Epoch {epoch + 1}/{max_epochs}")
        
        # Shuffle training indices
        np.random.shuffle(train_indices)
        epoch_losses = []
        batch_counter = 0
        
        # NEW FIX: Batch processing with guaranteed progress
        batch_idx = 0
        while batch_idx < total_batches:
            batch_counter += 1
            
            # CRITICAL FIX: Force reset at regular intervals
            if batch_counter % reset_interval == 0:
                robust_log(f"Scheduled TensorFlow reset at batch {batch_idx}")
                # Save model state
                temp_model_path = os.path.join(output_dir, "checkpoints", f"temp_epoch_{epoch+1}_batch_{batch_idx}.h5")
                model.save(temp_model_path)
                
                # Full reset
                reset_tensorflow()
                
                # Reload model
                model = tf.keras.models.load_model(temp_model_path)
                robust_log("Model reloaded after reset")
            
            # NEW FIX: Monitor memory before batch
            mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            robust_log(f"Memory before batch {batch_idx}: {mem_before:.1f} MB")
            
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
            batch_indices = train_indices[start_idx:end_idx]
            
            # NEW FIX: Try progressively smaller batch sizes if needed
            for attempt, mini_batch_size in enumerate([128, 64, 32, 16]):
                try:
                    # Load batch data
                    robust_log(f"Loading batch {batch_idx+1}/{total_batches} with {len(batch_indices)} samples")
                    batch_X = np.array([X_mmap[idx] for idx in batch_indices])
                    batch_y = np.array([y_mmap[idx] for idx in batch_indices])
                    
                    # CRITICAL FIX: Check for NaN/Inf in input data
                    if np.isnan(batch_X).any() or np.isinf(batch_X).any():
                        robust_log(f"Found NaN/Inf in batch data, cleaning...", "WARNING")
                        batch_X = np.nan_to_num(batch_X, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Train on batch with smaller mini-batch size
                    robust_log(f"Training with mini-batch size {mini_batch_size}")
                    history = model.fit(
                        batch_X, batch_y,
                        epochs=1,
                        batch_size=mini_batch_size,
                        class_weight=class_weight,
                        verbose=0
                    )
                    
                    # Check for NaN loss (training failure)
                    if np.isnan(history.history['loss'][0]):
                        robust_log(f"NaN loss detected with batch size {mini_batch_size}", "WARNING")
                        if attempt == 3:  # Last attempt
                            robust_log("All batch sizes failed, skipping batch", "WARNING")
                            stall_counter += 1
                            break
                        continue  # Try smaller batch size
                    
                    # Success - record loss and break
                    epoch_losses.append(history.history['loss'][0])
                    stall_counter = 0
                    break
                    
                except Exception as e:
                    robust_log(f"Error training batch {batch_idx} with size {mini_batch_size}: {e}", "ERROR")
                    if attempt == 3:  # Last attempt
                        robust_log("All batch sizes failed, skipping batch", "WARNING")
                        stall_counter += 1
                        break
            
            # Clean up batch data regardless of success
            try:
                del batch_X, batch_y
            except:
                pass
            gc.collect()
            
            # NEW FIX: Monitor memory after batch
            mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            robust_log(f"Memory after batch {batch_idx}: {mem_after:.1f} MB (delta: {mem_after-mem_before:.1f} MB)")
            
            # Validate periodically
            if batch_idx % 50 == 0 or stall_counter > 0:
                try:
                    robust_log(f"Evaluating on validation set at batch {batch_idx+1}")
                    val_metrics = model.evaluate(val_X, val_y, verbose=0)
                    val_auc = val_metrics[model.metrics_names.index('auc')]
                    robust_log(f"Validation AUC: {val_auc:.4f}")
                    
                    # Save if better
                    if val_auc > best_val_auc:
                        improvement = val_auc - best_val_auc
                        best_val_auc = val_auc
                        last_improvement_batch = batch_idx
                        best_model_path = os.path.join(output_dir, "checkpoints", f"best_model_epoch_{epoch+1}_batch_{batch_idx}.h5")
                        model.save(best_model_path)
                        robust_log(f"New best model saved with AUC improvement of {improvement:.4f}")
                        
                        # CRITICAL FIX: Reset stall counter after improvement
                        stall_counter = 0
                    else:
                        # CRITICAL FIX: Recalibrate learning rate if no improvement
                        if batch_idx - last_improvement_batch > 300:
                            robust_log("No improvement for 300 batches, recalibrating learning rate", "WARNING")
                            current_lr = float(tf.keras.backend.get_value(model.optimizer.lr))
                            new_lr = current_lr * 0.5
                            tf.keras.backend.set_value(model.optimizer.lr, new_lr)
                            robust_log(f"Learning rate decreased from {current_lr} to {new_lr}")
                            last_improvement_batch = batch_idx  # Reset counter
                except Exception as val_error:
                    robust_log(f"Validation error: {val_error}", "ERROR")
            
            # Save checkpoint periodically
            if batch_idx % 100 == 0:
                checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_epoch_{epoch+1}_batch_{batch_idx}.h5")
                model.save(checkpoint_path)
                robust_log(f"Checkpoint saved to {checkpoint_path}")
            
            # CRITICAL FIX: Handle stalling
            if stall_counter >= 5:
                robust_log("Training appears to be stalled, performing emergency recovery", "WARNING")
                
                # Save current state
                emergency_path = os.path.join(output_dir, "checkpoints", f"emergency_epoch_{epoch+1}_batch_{batch_idx}.h5")
                model.save(emergency_path)
                
                # Full reset
                reset_tensorflow()
                
                # Reload with completely fresh optimizer
                model = tf.keras.models.load_model(emergency_path)
                
                # Recompile with lower learning rate
                current_lr = float(tf.keras.backend.get_value(model.optimizer.lr))
                new_lr = max(current_lr * 0.1, 1e-6)  # Dramatic reduction
                
                robust_log(f"Recompiling with emergency learning rate {new_lr}", "WARNING")
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=new_lr,
                        epsilon=1e-8,
                        clipnorm=0.5
                    ),
                    loss='binary_crossentropy',
                    metrics=[
                        'accuracy',
                        tf.keras.metrics.AUC(name='auc', num_thresholds=200),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')
                    ]
                )
                
                # Reset stall counter
                stall_counter = 0
                robust_log("Emergency recovery complete")
            
            # Progress reporting
            if batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                progress = batch_idx / total_batches
                remaining = (elapsed / max(0.001, progress)) * (1 - progress)
                
                robust_log(f"Progress: {batch_idx+1}/{total_batches} batches ({progress*100:.1f}%)")
                robust_log(f"Elapsed: {timedelta(seconds=int(elapsed))}, Remaining: {timedelta(seconds=int(remaining))}")
            
            # Move to next batch
            batch_idx += 1
        
        # Epoch complete - summarize
        if epoch_losses:
            mean_loss = np.mean(epoch_losses)
            robust_log(f"Epoch {epoch+1} complete. Mean loss: {mean_loss:.6f}")
            
            # Check for early stopping
            if epoch >= 5 and mean_loss > 0.65:  # High loss after several epochs
                robust_log("Training not improving, stopping early")
                break
        else:
            robust_log(f"Epoch {epoch+1} had no successful batches", "WARNING")
            # Skip to next epoch rather than stopping

    # Training complete
    final_model_path = os.path.join(output_dir, "final_model.h5")
    model.save(final_model_path)
    robust_log(f"Training complete. Final model saved to {final_model_path}")
    
    # Return best model if available
    if best_model_path and os.path.exists(best_model_path):
        robust_log(f"Loading best model from {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)
        return model, best_model_path
    
    return model, final_model_path

def compile_model_with_device_handling(model):
    """Compile model with explicit device handling"""
    with tf.device('/GPU:0'):  # Force GPU usage
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001, 
                clipnorm=1.0,
                amsgrad=True
            ),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
            ]
        )
    return model

def ultra_robust_training_fixed(model, X_file, y_file, train_indices, val_indices, test_indices,
                          output_dir, batch_size=128, max_epochs=100, class_weight=None):
    """
    Fixed ULTRA-ROBUST TRAINING FUNCTION that prevents batch stalling issues
    """
    import os
    import numpy as np
    import tensorflow as tf
    import gc
    import time
    import psutil
    import json
    from datetime import datetime, timedelta
    import traceback
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Logging setup
    log_file = os.path.join(output_dir, "ultra_robust_training_log.txt")
    
    def robust_log(message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message, flush=True)
        with open(log_file, "a") as f:
            f.write(log_message + "\n")
    
    # Checkpoint cleanup function
    def cleanup_old_checkpoints(directory, pattern, keep=5):
        """Remove old checkpoints, keeping only the most recent ones"""
        files = [f for f in os.listdir(directory) if f.startswith(pattern) and f.endswith('.h5')]
        # Sort files by modification time (newest last)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))
        # Delete older files, keeping 'keep' most recent ones
        if len(files) > keep:
            for f in files[:-keep]:
                try:
                    os.remove(os.path.join(directory, f))
                    robust_log(f"Removed old checkpoint: {f}")
                except Exception as e:
                    robust_log(f"Error removing checkpoint {f}: {e}", "WARNING")
    
    # CRITICAL FIX: Reset TensorFlow's internal state
    def reset_tensorflow():
        robust_log("Performing complete TensorFlow reset")
        tf.keras.backend.clear_session()
        gc.collect()
        
        # CRITICAL FIX: Force GPU memory cleanup
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.reset_memory_stats(gpu)
                robust_log(f"Reset memory stats for {gpu}")
            except:
                robust_log(f"Could not reset memory stats for {gpu}", "WARNING")
        return

    # Configure TensorFlow
    robust_log("Configuring TensorFlow environment")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    # NEW FIX: Set memory growth at the beginning
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            robust_log(f"Memory growth enabled for {gpu}")
        except:
            robust_log(f"Could not set memory growth for {gpu}", "WARNING")
    
    # CRITICAL FIX: Set seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Load data safely
    robust_log("Loading data files")
    X_mmap = np.load(X_file, mmap_mode='r')
    y_mmap = np.load(y_file, mmap_mode='r')
    
    # Smaller validation set
    val_limit = min(1000, len(val_indices))
    val_indices_subset = val_indices[:val_limit]
    val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
    val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    robust_log(f"Validation data loaded with shape X:{val_X.shape}, y:{val_y.shape}")
    
    # NEW FIX: Recompile model with stability improvements
    robust_log("Recompiling model with stability improvements")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            epsilon=1e-7,
            clipnorm=1.0,
            clipvalue=0.5  # NEW: Clip individual gradients
        ),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            # FIX: Use numerically stable AUC implementation with more thresholds
            tf.keras.metrics.AUC(name='auc', num_thresholds=200, from_logits=False),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    # Main training configuration
    total_batches = len(train_indices) // batch_size + (1 if len(train_indices) % batch_size > 0 else 0)
    robust_log(f"Training configuration: {total_batches} batches with size {batch_size}")
    
    # Performance tracking
    best_val_auc = 0
    best_model_path = None
    training_history = []
    start_time = time.time()
    
    # CRITICAL FIX: State tracking variables
    stall_counter = 0
    reset_interval = 200  # Reset every 200 batches regardless of status
    last_improvement_batch = 0
    
    # Main training loop with CRITICAL ANTI-STALL MEASURES
    for epoch in range(max_epochs):
        robust_log(f"Starting Epoch {epoch + 1}/{max_epochs}")
        
        # Shuffle training indices
        np.random.shuffle(train_indices)
        epoch_losses = []
        batch_counter = 0
        
        # NEW FIX: Batch processing with guaranteed progress
        batch_idx = 0
        while batch_idx < total_batches:
            batch_counter += 1
            
            # CRITICAL FIX: Force reset at regular intervals
            if batch_counter % reset_interval == 0:
                robust_log(f"Scheduled TensorFlow reset at batch {batch_idx}")
                # Save model state
                temp_model_path = os.path.join(output_dir, "checkpoints", f"temp_epoch_{epoch+1}_batch_{batch_idx}.h5")
                
                # Save model with custom objects
                tf.keras.models.save_model(
                    model,
                    temp_model_path,
                    save_format='h5',
                    save_traces=False  # Important to prevent serialization issues
                )
                
                # Full reset
                reset_tensorflow()
                
                # Reload model with custom objects
                model = tf.keras.models.load_model(
                    temp_model_path,
                    custom_objects={'BatchNorm5D': BatchNorm5D}
                )
                robust_log("Model reloaded after reset")
                
                # Clean up temporary checkpoints
                cleanup_old_checkpoints(os.path.join(output_dir, "checkpoints"), "temp_epoch_", keep=2)
            
            # NEW FIX: Monitor memory before batch
            mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            robust_log(f"Memory before batch {batch_idx}: {mem_before:.1f} MB")
            
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
            batch_indices = train_indices[start_idx:end_idx]
            
            # NEW FIX: Try progressively smaller batch sizes if needed
            for attempt, mini_batch_size in enumerate([128, 64, 32, 16]):
                try:
                    # Load batch data
                    robust_log(f"Loading batch {batch_idx+1}/{total_batches} with {len(batch_indices)} samples")
                    batch_X = np.array([X_mmap[idx] for idx in batch_indices])
                    batch_y = np.array([y_mmap[idx] for idx in batch_indices])
                    
                    # CRITICAL FIX: Check for NaN/Inf in input data
                    if np.isnan(batch_X).any() or np.isinf(batch_X).any():
                        robust_log(f"Found NaN/Inf in batch data, cleaning...", "WARNING")
                        batch_X = np.nan_to_num(batch_X, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Train on batch with smaller mini-batch size
                    robust_log(f"Training with mini-batch size {mini_batch_size}")
                    history = model.fit(
                        batch_X, batch_y,
                        epochs=1,
                        batch_size=mini_batch_size,
                        class_weight=class_weight,
                        verbose=0
                    )
                    
                    # Check for NaN loss (training failure)
                    if np.isnan(history.history['loss'][0]):
                        robust_log(f"NaN loss detected with batch size {mini_batch_size}", "WARNING")
                        if attempt == 3:  # Last attempt
                            robust_log("All batch sizes failed, skipping batch", "WARNING")
                            stall_counter += 1
                            break
                        continue  # Try smaller batch size
                    
                    # Success - record loss and break
                    epoch_losses.append(history.history['loss'][0])
                    stall_counter = 0
                    break
                    
                except Exception as e:
                    robust_log(f"Error training batch {batch_idx} with size {mini_batch_size}: {e}", "ERROR")
                    if attempt == 3:  # Last attempt
                        robust_log("All batch sizes failed, skipping batch", "WARNING")
                        stall_counter += 1
                        break
            
            # Clean up batch data regardless of success
            try:
                del batch_X, batch_y
            except:
                pass
            gc.collect()
            
            # NEW FIX: Monitor memory after batch
            mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            robust_log(f"Memory after batch {batch_idx}: {mem_after:.1f} MB (delta: {mem_after-mem_before:.1f} MB)")
            
            # Validate periodically
            if batch_idx % 500 == 0 or stall_counter > 0:
                try:
                    robust_log(f"Evaluating on validation set at batch {batch_idx+1}")
                    val_metrics = model.evaluate(val_X, val_y, verbose=0)
                    val_auc = val_metrics[model.metrics_names.index('auc')]
                    robust_log(f"Validation AUC: {val_auc:.4f}")
                    
                    # Save if better
                    if val_auc > best_val_auc:
                        improvement = val_auc - best_val_auc
                        best_val_auc = val_auc
                        last_improvement_batch = batch_idx
                        best_model_path = os.path.join(output_dir, "checkpoints", f"best_model_epoch_{epoch+1}_batch_{batch_idx}.h5")
                        
                        # Save with proper format
                        tf.keras.models.save_model(
                            model,
                            best_model_path,
                            save_format='h5',
                            save_traces=False  # Important to prevent serialization issues
                        )
                        robust_log(f"New best model saved with AUC improvement of {improvement:.4f}")
                        
                        # CRITICAL FIX: Reset stall counter after improvement
                        stall_counter = 0
                        
                        # Clean up best model checkpoints (keeping only most recent)
                        cleanup_old_checkpoints(os.path.join(output_dir, "checkpoints"), "best_model_", keep=3)
                    else:
                        # CRITICAL FIX: Recalibrate learning rate if no improvement
                        if batch_idx - last_improvement_batch > 300:
                            robust_log("No improvement for 300 batches, recalibrating learning rate", "WARNING")
                            current_lr = float(tf.keras.backend.get_value(model.optimizer.lr))
                            new_lr = current_lr * 0.5
                            tf.keras.backend.set_value(model.optimizer.lr, new_lr)
                            robust_log(f"Learning rate decreased from {current_lr} to {new_lr}")
                            last_improvement_batch = batch_idx  # Reset counter
                except Exception as val_error:
                    robust_log(f"Validation error: {val_error}", "ERROR")
            
            # Save checkpoint periodically
            if batch_idx % 500 == 0:
                checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_epoch_{epoch+1}_batch_{batch_idx}.h5")
                
                # Save with proper format
                tf.keras.models.save_model(
                    model,
                    checkpoint_path,
                    save_format='h5',
                    save_traces=False  # Important to prevent serialization issues
                )
                robust_log(f"Checkpoint saved to {checkpoint_path}")
                
                # Clean up regular checkpoints
                cleanup_old_checkpoints(os.path.join(output_dir, "checkpoints"), "model_epoch_", keep=3)
            
            # CRITICAL FIX: Handle stalling
            if stall_counter >= 5:
                robust_log("Training appears to be stalled, performing emergency recovery", "WARNING")
                
                # Save current state
                emergency_path = os.path.join(output_dir, "checkpoints", f"emergency_epoch_{epoch+1}_batch_{batch_idx}.h5")
                
                # Save with proper format
                tf.keras.models.save_model(
                    model,
                    emergency_path,
                    save_format='h5',
                    save_traces=False  # Important to prevent serialization issues
                )
                
                # Full reset
                reset_tensorflow()
                
                # Reload with completely fresh optimizer
                model = tf.keras.models.load_model(
                    emergency_path,
                    custom_objects={'BatchNorm5D': BatchNorm5D}
                )
                
                # Recompile with lower learning rate
                current_lr = float(tf.keras.backend.get_value(model.optimizer.lr))
                new_lr = max(current_lr * 0.1, 1e-6)  # Dramatic reduction
                
                robust_log(f"Recompiling with emergency learning rate {new_lr}", "WARNING")
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=new_lr,
                        epsilon=1e-8,
                        clipnorm=0.5
                    ),
                    loss='binary_crossentropy',
                    metrics=[
                        'accuracy',
                        tf.keras.metrics.AUC(name='auc', num_thresholds=200),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')
                    ]
                )
                
                # Reset stall counter
                stall_counter = 0
                robust_log("Emergency recovery complete")
                
                # Clean up emergency checkpoints
                cleanup_old_checkpoints(os.path.join(output_dir, "checkpoints"), "emergency_epoch_", keep=2)
            
            # Progress reporting
            if batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                progress = batch_idx / total_batches
                remaining = (elapsed / max(0.001, progress)) * (1 - progress)
                
                robust_log(f"Progress: {batch_idx+1}/{total_batches} batches ({progress*100:.1f}%)")
                robust_log(f"Elapsed: {timedelta(seconds=int(elapsed))}, Remaining: {timedelta(seconds=int(remaining))}")
            
            # Move to next batch
            batch_idx += 1
        
        # Epoch complete - summarize
        if epoch_losses:
            mean_loss = np.mean(epoch_losses)
            robust_log(f"Epoch {epoch+1} complete. Mean loss: {mean_loss:.6f}")
            
            # Check for early stopping
            if epoch >= 5 and mean_loss > 0.65:  # High loss after several epochs
                robust_log("Training not improving, stopping early")
                break
        else:
            robust_log(f"Epoch {epoch+1} had no successful batches", "WARNING")
            # Skip to next epoch rather than stopping

    # Training complete
    final_model_path = os.path.join(output_dir, "final_model.h5")
    
    # Save with proper format
    tf.keras.models.save_model(
        model,
        final_model_path,
        save_format='h5',
        save_traces=False  # Important to prevent serialization issues
    )
    robust_log(f"Training complete. Final model saved to {final_model_path}")
    
    # Final cleanup of all checkpoints except the best ones
    robust_log("Performing final checkpoint cleanup")
    cleanup_old_checkpoints(os.path.join(output_dir, "checkpoints"), "model_epoch_", keep=1)
    cleanup_old_checkpoints(os.path.join(output_dir, "checkpoints"), "temp_epoch_", keep=0)
    cleanup_old_checkpoints(os.path.join(output_dir, "checkpoints"), "emergency_epoch_", keep=0)
    
    # Return best model if available
    if best_model_path and os.path.exists(best_model_path):
        robust_log(f"Loading best model from {best_model_path}")
        model = tf.keras.models.load_model(
            best_model_path,
            custom_objects={'BatchNorm5D': BatchNorm5D}
        )
        return model, best_model_path
    
    return model, final_model_path

def run_improved_pipeline():
    """
    Run the improved pipeline with stall-proof training.
    """
    # Configure TensorFlow
    configure_tensorflow_memory()

    # Suppress TensorFlow logging
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
    import warnings
    warnings.filterwarnings('ignore')  

    # For clearing Jupyter output
    from IPython.display import clear_output
    
    # Paths and configuration
    output_dir = '/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/improved_model'
    data_dir = '/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/ml_data'
    X_file = os.path.join(data_dir, 'X_features.npy')
    y_file = os.path.join(data_dir, 'y_labels.npy')
    metadata_file = os.path.join(data_dir, 'metadata.pkl')
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # Load split indices
    with open("zero_curtain_pipeline/modeling/checkpoints/spatiotemporal_split.pkl", "rb") as f:
        split_data = pickle.load(f)
    train_indices = split_data["train_indices"]
    val_indices = split_data["val_indices"]
    test_indices = split_data["test_indices"]
    
    # Load spatial weights
    try:
        with open("zero_curtain_pipeline/modeling/checkpoints/spatial_density.pkl", "rb") as f:
            weights_data = pickle.load(f)
        sample_weights = weights_data["weights"][train_indices]
        sample_weights = sample_weights / np.mean(sample_weights) * len(sample_weights)
    except:
        print("No spatial weights found, using uniform weights")
        sample_weights = None
    
    # Calculate class weights
    y = np.load(y_file, mmap_mode='r')
    train_y = y[train_indices]
    pos_weight = (len(train_y) - np.sum(train_y)) / max(1, np.sum(train_y))
    class_weight = {0: 1.0, 1: pos_weight}
    print(f"Using class weight {pos_weight:.2f} for positive examples")
    
    # Get sample shape
    X = np.load(X_file, mmap_mode='r')
    input_shape = X[train_indices[0]].shape
    
    # Build fixed model
    print("Building fixed model architecture...")
    model = build_improved_zero_curtain_model_fixed(input_shape)
    
    # Print model summary
    model.summary()
    
    # Determine start batch by finding the most recent checkpoint
    start_batch = 0
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                           if f.startswith("model_batch_") and f.endswith(".h5")]
        
        if checkpoint_files:
            batch_nums = []
            for f in checkpoint_files:
                try:
                    batch_num = int(f.replace("model_batch_", "").replace(".h5", ""))
                    batch_nums.append(batch_num)
                except ValueError:
                    continue
            
            if batch_nums:
                start_batch = max(batch_nums)
                print(f"Found existing checkpoint at batch {start_batch}")
    
    # Train model with failsafe function
    print("\nTraining model with fixed architecture...")
    model, best_model_path = failsafe_training_fixed(
        model=model,
        X_file=X_file,
        y_file=y_file,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        output_dir=output_dir,
        batch_size=256,
        epochs=3,
        class_weight=class_weight,
        start_batch=0
    )
    
    clear_output(wait=True)
    print("Training completed. Starting evaluation...")
    
    # Evaluate model
    print("\nEvaluating model...")
    eval_results = evaluate_model_with_visualizations(
        model=model,
        X_file=X_file,
        y_file=y_file,
        test_indices=test_indices,
        output_dir=output_dir,
        metadata_file=metadata_file
    )
    
    clear_output(wait=True)
    print("Evaluation completed. Analyzing spatial performance...")
    
    # Analyze spatial performance
    print("\nAnalyzing spatial performance patterns...")
    spatial_results = analyze_spatial_performance(
        output_dir=output_dir,
        metadata_file=metadata_file
    )
    
    clear_output(wait=True)
    print("Spatial analysis completed. Analyzing feature importance...")
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    feature_results = analyze_feature_importance(
        model=model,
        X_file=X_file,
        y_file=y_file,
        test_indices=test_indices,
        output_dir=output_dir
    )
    
    # Save all results
    all_results = {
        'evaluation': eval_results,
        'spatial_analysis': spatial_results,
        'feature_importance': feature_results
    }
    
    with open(os.path.join(output_dir, "complete_analysis.json"), "w") as f:
        import json
        
        # Convert numpy values to Python native types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                    return float(obj)
                return json.JSONEncoder.default(self, obj)
        
        json.dump(all_results, f, indent=4, cls=NumpyEncoder)
    
    print("\nImproved pipeline complete!")
    print(f"Results saved to {output_dir}")
    
    return all_results

if __name__ == "__main__":
    run_improved_pipeline()


# # WHOLE CODE FOR [AUTOMATED_TOOL]

# import os,sys
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import cmocean
# import gc
# import glob
# import json
# import logging
# import matplotlib.gridspec as gridspec
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pandas as pd
# import pathlib
# import pickle
# import psutil
# import re
# import gc
# import dask
# import dask.dataframe as dd
# import scipy.interpolate as interpolate
# import scipy.stats as stats
# import seaborn as sns
# from pyproj import Proj
# import sklearn.experimental
# import sklearn.impute
# import sklearn.linear_model
# import sklearn.preprocessing
# import tqdm
# import xarray as xr
# import warnings
# warnings.filterwarnings('ignore')

# from osgeo import gdal, osr
# from matplotlib.colors import LinearSegmentedColormap
# from concurrent.futures import ThreadPoolExecutor
# from datetime import datetime, timedelta
# from pathlib import Path
# from scipy.spatial import cKDTree
# from tqdm import tqdm
# from tqdm.notebook import tqdm

# import tensorflow as tf
# try:
#     physical_devices = tf.config.list_physical_devices('GPU')
#     for device in physical_devices:
#         tf.config.experimental.set_memory_growth(device, True)
# except:
#     pass

# print("TensorFlow version:", tf.__version__)

# import keras_tuner as kt
# from keras_tuner.tuners import BayesianOptimization
# #os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# #os.environ["DEVICE_COUNT_GPU"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # DISABLE GPU
# import tensorflow as tf
# import json
# import glob
# import keras
# from keras import layers
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report

# from packaging import version

# print("TensorFlow version: ", tf.__version__)
# assert version.parse(tf.__version__).release[0] >= 2, \
#     "This notebook requires TensorFlow 2.0 or above."

# print("==========================")

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print(sys.getrecursionlimit())
# sys.setrecursionlimit(1000000000)
# print(sys.getrecursionlimit())

# def configure_tensorflow_memory():
# """Configure TensorFlow to use memory growth and...
#     import tensorflow as tf
    
#     # Disable GPU if experiencing persistent issues
#     # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
#     # Get available physical devices
#     physical_devices = tf.config.list_physical_devices('GPU')
    
#     try:
#         # Attempt to set memory growth for all GPUs
#         for device in physical_devices:
#             try:
#                 tf.config.experimental.set_memory_growth(device, True)
#                 print(f"Memory growth enabled for {device}")
#             except Exception as e:
#                 print(f"Could not set memory growth for {device}: {e}")
        
#         # Explicit device placement settings
#         tf.config.set_soft_device_placement(True)
        
#         # Optional: Limit GPU memory to prevent out-of-memory errors
#         for device in physical_devices:
#             try:
#                 tf.config.experimental.set_virtual_device_configuration(
#                     device,
#                     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 3)]  # ...
#                 )
#                 print(f"Memory limit set for {device}")
#             except Exception as e:
#                 print(f"Could not set memory limit for {device}: {e}")
    
#     except Exception as global_e:
#         print(f"Error configuring GPU memory: {global_e}")
    
#     # Log available devices
#     print("Available devices:")
#     print(tf.config.list_physical_devices())

# def build_improved_zero_curtain_model(input_shape, include_moisture=True):
#     """
#     Advanced model architecture with increased regularization and gradient clipping.
#     Fixed to avoid TensorFlow operation conflicts.
#     """
#     from tensorflow.keras.models import Model
#     from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Dropout
#     from tensorflow.keras.layers import Reshape, Concatenate, GlobalAveragePooling1D, GlobalMaxPoo...
#     from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
#     from tensorflow.keras.optimizers import Adam
#     from tensorflow.keras.regularizers import l2
    
#     # L2 regularization strength
#     reg_strength = 0.0001
#     # Increased dropout rate
#     dropout_rate = 0.3
    
#     # Input layer
#     inputs = Input(shape=input_shape)
    
#     # Reshape for ConvLSTM (add spatial dimension)
#     x = tf.keras.layers.Reshape((input_shape[0], 1, 1, input_shape[1]))(inputs)
    
#     # ConvLSTM layer with regularization
#     from tensorflow.keras.layers import ConvLSTM2D
#     convlstm = ConvLSTM2D(
#         filters=64,
#         kernel_size=(3, 1),
#         padding='same',
#         return_sequences=True,
#         activation='tanh',
#         recurrent_dropout=dropout_rate,
#         kernel_regularizer=l2(reg_strength)
#     )(x)
    
#     # Reshape back to (sequence_length, features)
#     convlstm = Reshape((input_shape[0], 64))(convlstm)
    
#     # Add positional encoding for transformer (existing function)
#     def positional_encoding(length, depth):
#         positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
#         depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth
        
#         angle_rates = 1 / tf.pow(10000.0, depths)
#         angle_rads = positions * angle_rates
        
#         # Only use sin to ensure output depth matches input depth
#         pos_encoding = tf.sin(angle_rads)
        
#         # Add batch dimension
#         pos_encoding = tf.expand_dims(pos_encoding, 0)
        
#         return pos_encoding
    
#     # Add positional encoding
#     pos_encoding = positional_encoding(input_shape[0], 64)
#     transformer_input = convlstm + pos_encoding
    
#     # Improved transformer encoder block
#     def transformer_encoder(x, num_heads=8, key_dim=64, ff_dim=128):
#         # Multi-head attention with regularization
#         attention_output = MultiHeadAttention(
#             num_heads=num_heads, key_dim=key_dim,
#             kernel_regularizer=l2(reg_strength)
#         )(x, x)
        
#         # Skip connection 1 with dropout
#         x1 = Add()([attention_output, x])
#         x1 = LayerNormalization(epsilon=1e-6)(x1)
#         x1 = Dropout(dropout_rate)(x1)  # Added dropout after norm
        
#         # Feed-forward network with regularization
#         ff_output = Dense(ff_dim, activation='relu', kernel_regularizer=l2(reg_strength))(x1)
#         ff_output = Dropout(dropout_rate)(ff_output)
#         ff_output = Dense(64, kernel_regularizer=l2(reg_strength))(ff_output)
        
#         # Skip connection 2
#         x2 = Add()([ff_output, x1])
#         return LayerNormalization(epsilon=1e-6)(x2)
    
#     # Apply transformer encoder
#     transformer_output = transformer_encoder(transformer_input)
    
# # Parallel CNN paths for multi-scale feature...
#     cnn_1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', 
#                   kernel_regularizer=l2(reg_strength))(inputs)
#     cnn_1 = BatchNormalization()(cnn_1)
#     cnn_1 = Dropout(dropout_rate/2)(cnn_1)  # Lighter dropout for CNNs
    
#     cnn_2 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu',
#                   kernel_regularizer=l2(reg_strength))(inputs)
#     cnn_2 = BatchNormalization()(cnn_2)
#     cnn_2 = Dropout(dropout_rate/2)(cnn_2)
    
#     cnn_3 = Conv1D(filters=32, kernel_size=7, padding='same', activation='relu',
#                   kernel_regularizer=l2(reg_strength))(inputs)
#     cnn_3 = BatchNormalization()(cnn_3)
#     cnn_3 = Dropout(dropout_rate/2)(cnn_3)
    
#     # VAE components (existing code with regularization)
#     def sampling(args):
#         z_mean, z_log_var = args
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
#     # Global temporal features
#     global_max = GlobalMaxPooling1D()(transformer_output)
#     global_avg = GlobalAveragePooling1D()(transformer_output)
    
#     # VAE encoding with regularization
#     z_mean = Dense(32, kernel_regularizer=l2(reg_strength))(Concatenate()([global_max, global_avg]...
#     z_log_var = Dense(32, kernel_regularizer=l2(reg_strength))(Concatenate()([global_max, global_a...
#     z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
    
#     # Combine all features
#     merged_features = Concatenate()(
#         [
#             GlobalMaxPooling1D()(cnn_1),
#             GlobalMaxPooling1D()(cnn_2),
#             GlobalMaxPooling1D()(cnn_3),
#             global_max,
#             global_avg,
#             z
#         ]
#     )
    
#     # Final classification layers with regularization
#     x = Dense(128, activation='relu', kernel_regularizer=l2(reg_strength))(merged_features)
#     x = Dropout(dropout_rate)(x)
#     x = BatchNormalization()(x)
#     x = Dense(64, activation='relu', kernel_regularizer=l2(reg_strength))(x)
#     x = Dropout(dropout_rate)(x)
    
#     # Output layer
#     outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_strength))(x)
    
#     # Create model
#     model = Model(inputs=inputs, outputs=outputs)
    
#     # Add VAE loss with reduced weight
#     kl_loss = -0.5 * tf.reduce_mean(
#         z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
#     )
#     model.add_loss(0.0005 * kl_loss)  # Reduced from 0.001 to 0.0005
    
#     # Compile model with gradient clipping but simpler metrics to avoid errors
#     model.compile(
#         optimizer=Adam(
#             learning_rate=0.0005,  # Reduced learning rate
#             clipvalue=1.0  # Add gradient clipping
#         ),
#         loss='binary_crossentropy',
#         metrics=[
#             'accuracy',
#             tf.keras.metrics.AUC(name='auc'),
#             tf.keras.metrics.Precision(name='precision'),
#             tf.keras.metrics.Recall(name='recall')
#             # Removed custom F1 implementation that was causing errors
#         ]
#     )
    
#     return model

# # Cyclical Learning Rate Callback
# class CyclicLR(tf.keras.callbacks.Callback):
#     """
#     Cyclical learning rate callback for smoother training.
#     """
#     def __init__(
#         self,
#         base_lr=0.0001,
#         max_lr=0.001,
#         step_size=2000,
#         mode='triangular2'
#     ):
#         super(CyclicLR, self).__init__()
#         self.base_lr = base_lr
#         self.max_lr = max_lr
#         self.step_size = step_size
#         self.mode = mode
        
#         if self.mode == 'triangular':
#             self.scale_fn = lambda x: 1.
#         elif self.mode == 'triangular2':
#             self.scale_fn = lambda x: 1 / (2. ** (x - 1))
#         elif self.mode == 'exp_range':
#             self.scale_fn = lambda x: 0.9 ** x
        
#         self.clr_iterations = 0.
#         self.trn_iterations = 0.
#         self.history = {}

#     def clr(self):
#         cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
#         x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
# return self.base_lr + (self.max_lr - self.base_lr) *...
        
#     def on_train_begin(self, logs=None):
#         logs = logs or {}
#         tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)
            
#     def on_batch_end(self, batch, logs=None):
#         logs = logs or {}
#         self.trn_iterations += 1
#         self.clr_iterations += 1
#         tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())

# # Enhanced callbacks setup
# def get_enhanced_callbacks(output_dir, fold_idx=None):
#     """
# Create enhanced callbacks with increased patience and...
#     """
#     sub_dir = f"fold_{fold_idx}" if fold_idx is not None else ""
#     checkpoint_dir = os.path.join(output_dir, sub_dir, "checkpoints")
#     tensorboard_dir = os.path.join(output_dir, sub_dir, "tensorboard")
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     os.makedirs(tensorboard_dir, exist_ok=True)
    
#     callbacks = [
#         # Early stopping with increased patience
#         tf.keras.callbacks.EarlyStopping(
#             patience=20,  # Increased from 15 to 20
#             restore_best_weights=True,
#             monitor='val_auc',
#             mode='max',
#             min_delta=0.005
#         ),
#         # Reduced LR with increased patience
#         tf.keras.callbacks.ReduceLROnPlateau(
#             factor=0.5,
#             patience=10,  # Increased from 7 to 10
#             min_lr=1e-6,
#             monitor='val_auc',
#             mode='max'
#         ),
#         # Model checkpoint
#         tf.keras.callbacks.ModelCheckpoint(
#             os.path.join(checkpoint_dir, "best_model.h5"),
#             save_best_only=True,
#             monitor='val_auc',
#             mode='max'
#         ),
#         # TensorBoard
#         tf.keras.callbacks.TensorBoard(
#             log_dir=tensorboard_dir,
#             histogram_freq=1,
#             write_graph=False,
#             update_freq='epoch'
#         ),
#         # Cyclical learning rate
#         CyclicLR(
#             base_lr=0.0001,
#             max_lr=0.001,
#             step_size=2000,
#             mode='triangular2'
#         ),
#         # Memory cleanup
#         tf.keras.callbacks.LambdaCallback(
#             on_epoch_end=lambda epoch, logs: gc.collect()
#         )
#     ]
    
#     return callbacks

# def process_chunk_with_batch_safety(model, chunk_X, chunk_y, val_data, batch_size=256, epochs=3, c...
#     """
# Process a chunk with batch-level safety to...
#     """
#     import numpy as np
    
#     history_list = []
    
#     for epoch in range(epochs):
#         print(f"Epoch {epoch+1}/{epochs}")
# epoch_history = {"loss": [], "accuracy": [], "auc":...
        
#         # Process in smaller mini-batches to isolate failures
# num_batches = len(chunk_X) // batch_size + (1...
        
#         for batch in range(num_batches):
#             batch_start = batch * batch_size
#             batch_end = min((batch + 1) * batch_size, len(chunk_X))
            
#             # Extract batch data
#             batch_X = chunk_X[batch_start:batch_end]
#             batch_y = chunk_y[batch_start:batch_end]
            
#             try:
#                 # Train on single batch
#                 batch_history = model.fit(
#                     batch_X, batch_y,
#                     epochs=1,
#                     batch_size=batch_end - batch_start,
#                     class_weight=class_weight,
#                     verbose=0  # Silent mode to reduce output
#                 )
                
#                 # Store metrics
#                 for key in epoch_history:
#                     if key in batch_history.history:
#                         epoch_history[key].append(batch_history.history[key][0])
            
#             except Exception as e:
#                 print(f"Error processing batch {batch+1}/{num_batches}: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 # Skip this batch and continue
#                 continue
            
#             # Print progress every 10 batches
#             if batch % 10 == 0 or batch == num_batches - 1:
# metrics_str = " - ".join([f"{k}: {np.mean(v):.4f}" for...
#                 print(f"Batch {batch+1}/{num_batches} - {metrics_str}")
        
#         # Validate after each epoch
#         try:
#             val_X, val_y = val_data
#             val_metrics = model.evaluate(val_X, val_y, verbose=0)
# val_dict = {f"val_{name}": value for name, value...
            
#             # Add validation metrics to epoch history
#             for k, v in val_dict.items():
#                 epoch_history[k] = [v]
                
#             # Add current learning rate
#             if hasattr(model.optimizer, 'lr'):
#                 import tensorflow as tf
#                 epoch_history['lr'] = [float(tf.keras.backend.get_value(model.optimizer.lr))]
                
# metrics_str = " - ".join([f"{k}: {v:.4f}" for...
#             print(f"Validation: {metrics_str}")
#         except Exception as val_e:
#             print(f"Error during validation: {val_e}")
#             import traceback
#             traceback.print_exc()
        
#         # Add epoch history to overall history
#         history_list.append(epoch_history)
    
#     # Combine histories into a format similar to model.fit
#     combined_history = {"history": {}}
# for key in ["loss", "accuracy", "auc", "precision",...
#                 "val_auc", "val_precision", "val_recall", "lr"]:
#         combined_history["history"][key] = []
#         for epoch_hist in history_list:
#             if key in epoch_hist:
#                 if isinstance(epoch_hist[key], list):
#                     combined_history["history"][key].extend(epoch_hist[key])
#                 else:
#                     combined_history["history"][key].append(epoch_hist[key])
    
#     return combined_history

# def improved_resumable_training(model, X_file, y_file, train_indices, val_indices, test_indices,
#                                output_dir, batch_size=256, chunk_size=20000, epochs_per_chunk=3, 
#                                save_frequency=5, class_weight=None, start_chunk=0):
#     """
#     Enhanced training function with improved error handling.
#     """
#     import os
#     import gc
#     import json
#     import numpy as np
#     import tensorflow as tf
#     from datetime import datetime, timedelta
#     import time
#     import psutil
    
#     # Create output directories
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "tensorboard"), exist_ok=True)
    
#     # Process in chunks
#     num_chunks = int(np.ceil(len(train_indices) / chunk_size))
#     print(f"Processing {len(train_indices)} samples in {num_chunks} chunks of {chunk_size}")
#     print(f"Starting from chunk {start_chunk+1}")
    
#     # Open data files with memory mapping
#     X_mmap = np.load(X_file, mmap_mode='r')
#     y_mmap = np.load(y_file, mmap_mode='r')
    
#     # Create validation set once (limited size for memory efficiency)
#     val_limit = min(1000, len(val_indices))
#     val_indices_subset = val_indices[:val_limit]
    
#     # Load validation data
#     val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
#     val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
#     print(f"Loaded {len(val_X)} validation samples")
    
#     # Load existing history if resuming
#     history_log = []
#     history_path = os.path.join(output_dir, "training_history.json")
#     if start_chunk > 0 and os.path.exists(history_path):
#         try:
#             with open(history_path, "r") as f:
#                 history_log = json.load(f)
#         except Exception as e:
#             print(f"Could not load existing history: {e}")
#             # Try pickle format
#             pickle_path = os.path.join(output_dir, "training_history.pkl")
#             if os.path.exists(pickle_path):
#                 import pickle
#                 with open(pickle_path, "rb") as f:
#                     history_log = pickle.load(f)
    
#     # If resuming, load latest model
#     if start_chunk > 0:
#         # Find the most recent checkpoint before start_chunk
#         checkpoint_indices = []
#         checkpoints_dir = os.path.join(output_dir, "checkpoints")
#         for filename in os.listdir(checkpoints_dir):
#             if filename.startswith("model_chunk_") and filename.endswith(".h5"):
#                 try:
#                     idx = int(filename.split("_")[-1].split(".")[0])
#                     if idx < start_chunk:
#                         checkpoint_indices.append(idx)
#                 except ValueError:
#                     continue
        
#         if checkpoint_indices:
#             latest_idx = max(checkpoint_indices)
#             model_path = os.path.join(checkpoints_dir, f"model_chunk_{latest_idx}.h5")
#             if os.path.exists(model_path):
#                 print(f"Loading model from checkpoint {model_path}")
#                 model = tf.keras.models.load_model(model_path)
#             else:
#                 print(f"Warning: Could not find model checkpoint for chunk {latest_idx}")
    
#     # Setup callbacks
#     callbacks = [
#         # Early stopping with increased patience
#         tf.keras.callbacks.EarlyStopping(
#             patience=10, 
#             restore_best_weights=True,
#             monitor='val_auc',
#             mode='max',
#             min_delta=0.001
#         ),
#         # Reduced LR with increased patience
#         tf.keras.callbacks.ReduceLROnPlateau(
#             factor=0.5,
#             patience=5,
#             min_lr=1e-6,
#             monitor='val_auc',
#             mode='max'
#         ),
#         # Memory cleanup after each epoch
#         tf.keras.callbacks.LambdaCallback(
#             on_epoch_end=lambda epoch, logs: gc.collect()
#         )
#     ]
    
#     # Track metrics across chunks
#     start_time = time.time()
    
#     # For safe recovery
#     recovery_file = os.path.join(output_dir, "last_completed_chunk.txt")
    
#     # Process each chunk
#     for chunk_idx in range(start_chunk, num_chunks):
#         # Track progress in file for recovery purposes
#         with open(os.path.join(output_dir, "current_progress.txt"), "w") as f:
#             f.write(f"Processing chunk {chunk_idx+1}/{num_chunks}\n")
#             f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
#         # Get chunk indices
#         start_idx = chunk_idx * chunk_size
#         end_idx = min((chunk_idx + 1) * chunk_size, len(train_indices))
#         chunk_indices = train_indices[start_idx:end_idx]
        
#         # Report memory
#         memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
#         print(f"\n{'='*50}")
#         print(f"Processing chunk {chunk_idx+1}/{num_chunks} with {len(chunk_indices)} samples")
#         print(f"Memory before: {memory_before:.1f} MB")
        
#         # Force garbage collection before loading new data
#         gc.collect()
        
#         try:
#             # Load chunk data in smaller batches
#             chunk_X = []
#             chunk_y = []
            
#             # Use a smaller batch size for loading
#             load_batch_size = 5000
#             for i in range(0, len(chunk_indices), load_batch_size):
#                 end_i = min(i + load_batch_size, len(chunk_indices))
#                 print(f"  Loading batch {i}-{end_i} of {len(chunk_indices)}...")
                
#                 try:
#                     # Load this batch
#                     batch_indices = chunk_indices[i:end_i]
#                     X_batch = np.array([X_mmap[idx] for idx in batch_indices])
#                     y_batch = np.array([y_mmap[idx] for idx in batch_indices])
                    
#                     # Append to list
#                     chunk_X.append(X_batch)
#                     chunk_y.append(y_batch)
                    
#                     # Free memory
#                     del X_batch, y_batch
#                     gc.collect()
#                 except Exception as inner_e:
#                     print(f"Warning: Error loading batch {i}-{end_i}: {inner_e}")
#                     # Continue to next batch
            
#             # Combine batches
#             try:
#                 chunk_X = np.concatenate(chunk_X)
#                 chunk_y = np.concatenate(chunk_y)
#                 print(f"Data loaded. Memory: {psutil.Process(os.getpid()).memory_info().rss / (102...
#             except Exception as concat_e:
#                 print(f"Error combining data batches: {concat_e}")
#                 # Skip this chunk
#                 continue
            
#             # Track each epoch separately for better error handling
#             chunk_history = {}
#             for epoch in range(epochs_per_chunk):
#                 try:
#                     print(f"Epoch {epoch+1}/{epochs_per_chunk}")
#                     # Train for a single epoch
#                     # Use batch-safe training instead of standard fit
#                     epoch_history = process_chunk_with_batch_safety(
#                         model,
#                         chunk_X, 
#                         chunk_y,
#                         val_data=(val_X, val_y),
#                         batch_size=batch_size,
#                         epochs=1,
#                         class_weight=class_weight
#                     )
                    
#                     # Store history
#                     for k, v in epoch_history["history"].items():  # Changed from epoch_history.hi...
#                         if k not in chunk_history:
#                             chunk_history[k] = []
#                         chunk_history[k].extend([float(val) for val in v])
                    
#                     # Save after each epoch for safety
#                     epoch_model_path = os.path.join(output_dir, "checkpoints", f"model_chunk_{chun...
#                     model.save(epoch_model_path)
                    
#                 except Exception as epoch_e:
#                     print(f"Error during epoch {epoch+1}: {epoch_e}")
#                     import traceback
#                     traceback.print_exc()
#                     # Try to continue with next epoch
            
#             # Store history from all completed epochs
#             if chunk_history:
#                 history_log.append(chunk_history)
                
#                 # Save history
#                 try:
#                     with open(history_path, "w") as f:
#                         json.dump(history_log, f)
#                 except Exception as history_e:
#                     print(f"Warning: Could not save history to JSON: {history_e}")
#                     # Fallback - save as pickle
#                     import pickle
#                     with open(os.path.join(output_dir, "training_history.pkl"), "wb") as f:
#                         pickle.dump(history_log, f)
            
#             # Save model periodically
# if (chunk_idx + 1) % save_frequency ==...
#                 try:
#                     model_path = os.path.join(output_dir, "checkpoints", f"model_chunk_{chunk_idx+...
#                     model.save(model_path)
#                     print(f"Model saved to {model_path}")
#                 except Exception as save_e:
#                     print(f"Error saving model: {save_e}")
#                     # Try saving weights only
#                     try:
#                         weights_path = os.path.join(output_dir, "checkpoints", f"model_weights_{ch...
#                         model.save_weights(weights_path)
#                         print(f"Saved model weights to {weights_path}")
#                     except:
#                         print("Could not save model in any format")
            
#             # Explicitly delete everything from memory
#             del chunk_X, chunk_y
            
#         except Exception as outer_e:
#             print(f"Critical error processing chunk {chunk_idx+1}: {outer_e}")
#             import traceback
#             traceback.print_exc()
            
#             # Write error to file for debugging
#             with open(os.path.join(output_dir, f"error_chunk_{chunk_idx+1}.txt"), "w") as f:
#                 f.write(f"Error processing chunk {chunk_idx+1}:\n")
#                 f.write(traceback.format_exc())
                
#             # Try saving model in error state
#             try:
#                 error_model_path = os.path.join(output_dir, "checkpoints", f"error_recovery_{chunk...
#                 model.save(error_model_path)
#                 print(f"Saved model in error state to {error_model_path}")
#             except:
#                 print("Could not save model in error state")
                
#             # Continue to next chunk
#             continue
        
#         # Force garbage collection
#         gc.collect()
        
#         # Report memory after cleanup
#         memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
#         print(f"Memory after: {memory_after:.1f} MB (Change: {memory_after - memory_before:.1f} MB...
        
#         # Write recovery file with last completed chunk
#         with open(recovery_file, "w") as f:
#             f.write(str(chunk_idx + 1))
        
#         # Estimate time
#         elapsed = time.time() - start_time
# avg_time_per_chunk = elapsed / (chunk_idx - start_chunk...
#         remaining = avg_time_per_chunk * (num_chunks - chunk_idx - 1)
#         print(f"Estimated remaining time: {timedelta(seconds=int(remaining))}")
        
#         # Reset TensorFlow session periodically
#         if (chunk_idx + 1) % 20 == 0:
#             print("Resetting TensorFlow session to prevent memory issues")
#             try:
#                 temp_model_path = os.path.join(output_dir, "checkpoints", f"temp_reset_{chunk_idx+...
#                 model.save(temp_model_path)
                
#                 # Clear session
#                 tf.keras.backend.clear_session()
#                 gc.collect()
                
#                 # Reload model
#                 model = tf.keras.models.load_model(temp_model_path)
#                 print("TensorFlow session reset complete")
#             except Exception as reset_e:
#                 print(f"Error during TensorFlow reset: {reset_e}")
#                 # Continue anyway
    
#     # Save final model
#     final_model_path = os.path.join(output_dir, "final_model.h5")
#     try:
#         model.save(final_model_path)
#         print(f"Final model saved to {final_model_path}")
#     except Exception as final_save_e:
#         print(f"Error saving final model: {final_save_e}")
#         try:
#             # Try alternative location
#             alt_path = os.path.join(output_dir, "checkpoints", "final_model_backup.h5")
#             model.save(alt_path)
#             final_model_path = alt_path
#             print(f"Saved final model to alternate location: {alt_path}")
#         except:
#             print("Could not save final model in any format")
    
#     return model, final_model_path

# def evaluate_model_with_visualizations(model, X_file, y_file, test_indices, output_dir, metadata_f...
#     """
#     Comprehensive evaluation with detailed visualizations.
#     """
#     import os
#     import gc
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from sklearn.metrics import (
#         classification_report, confusion_matrix, roc_curve, 
#         auc, precision_recall_curve, average_precision_score
#     )
#     import traceback
    
#     os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
#     # Setup logging
#     log_file = os.path.join(output_dir, "evaluation_log.txt")
#     def log_message(message):
#         from datetime import datetime
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(log_file, "a") as f:
#             f.write(f"[{timestamp}] {message}\n")
#         print(message)
    
#     log_message(f"Starting evaluation on {len(test_indices)} test samples")
    
#     # Load metadata if available
#     metadata = None
#     if metadata_file and os.path.exists(metadata_file):
#         try:
#             import pickle
#             with open(metadata_file, "rb") as f:
#                 metadata = pickle.load(f)
#             log_message(f"Loaded metadata from {metadata_file}")
#         except Exception as e:
#             log_message(f"Error loading metadata: {str(e)}")
    
#     # Load memory-mapped data
#     X = np.load(X_file, mmap_mode='r')
#     y = np.load(y_file, mmap_mode='r')
    
#     # Process test data in batches
#     batch_size = 500  # Smaller batch size for more reliable processing
#     num_batches = int(np.ceil(len(test_indices) / batch_size))
    
#     all_preds = []
#     all_true = []
#     all_meta = []
    
#     log_message(f"Processing test data in {num_batches} batches of size {batch_size}")
    
#     # Save test indices for reference
#     np.save(os.path.join(output_dir, "test_indices.npy"), test_indices)
    
#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(test_indices))
#         batch_indices = test_indices[start_idx:end_idx]
        
#         log_message(f"Processing batch {batch_idx+1}/{num_batches} with {len(batch_indices)} sampl...
        
#         try:
#             # Load batch data
#             batch_X = np.array([X[idx] for idx in batch_indices])
#             batch_y = np.array([y[idx] for idx in batch_indices])
            
#             # Store metadata if available
#             if metadata:
#                 try:
#                     batch_meta = []
#                     for idx in batch_indices:
#                         if idx in metadata:
#                             batch_meta.append(metadata[idx])
#                         else:
#                             # Create empty metadata if not found
#                             batch_meta.append({})
#                     all_meta.extend(batch_meta)
#                 except Exception as meta_e:
#                     log_message(f"Error extracting metadata for batch {batch_idx}: {str(meta_e)}")
#                     # Continue without metadata for this batch
            
#             # Predict
#             batch_preds = model.predict(batch_X, verbose=0)
            
#             # Store results
#             all_preds.extend(batch_preds.flatten())
#             all_true.extend(batch_y)
            
#             # Clean up
#             del batch_X, batch_y, batch_preds
#             gc.collect()
            
#             log_message(f"Completed batch {batch_idx+1}/{num_batches}")
            
#         except Exception as e:
#             log_message(f"Error processing batch {batch_idx+1}: {str(e)}")
#             log_message(traceback.format_exc())
            
#             # Try to process the batch in smaller chunks
#             try:
#                 log_message("Attempting to process batch in smaller chunks...")
#                 sub_batch_size = 50  # Much smaller batch
#                 sub_batches = int(np.ceil(len(batch_indices) / sub_batch_size))
                
#                 for sub_idx in range(sub_batches):
#                     sub_start = sub_idx * sub_batch_size
#                     sub_end = min((sub_idx + 1) * sub_batch_size, len(batch_indices))
#                     sub_indices = batch_indices[sub_start:sub_end]
                    
#                     # Load and process sub-batch
#                     sub_X = np.array([X[idx] for idx in sub_indices])
#                     sub_y = np.array([y[idx] for idx in sub_indices])
                    
#                     # Add metadata if available
#                     if metadata:
#                         sub_meta = []
#                         for idx in sub_indices:
#                             if idx in metadata:
#                                 sub_meta.append(metadata[idx])
#                             else:
#                                 sub_meta.append({})
#                         all_meta.extend(sub_meta)
                    
#                     # Predict
#                     sub_preds = model.predict(sub_X, verbose=0)
                    
#                     # Store results
#                     all_preds.extend(sub_preds.flatten())
#                     all_true.extend(sub_y)
                    
#                     # Clean up
#                     del sub_X, sub_y, sub_preds
#                     gc.collect()
                
#                 log_message(f"Successfully processed batch {batch_idx+1} in {sub_batches} sub-batc...
#             except Exception as sub_e:
#                 log_message(f"Error processing sub-batches: {str(sub_e)}")
#                 log_message(traceback.format_exc())
#                 log_message(f"Skipping batch {batch_idx+1}")
    
#     log_message(f"Prediction complete. Total samples: {len(all_preds)}")
    
#     # Save raw predictions for reference
#     np.save(os.path.join(output_dir, "test_predictions.npy"), np.array(all_preds))
#     np.save(os.path.join(output_dir, "test_true_labels.npy"), np.array(all_true))
    
#     # Convert to numpy arrays
#     all_preds = np.array(all_preds)
#     all_true = np.array(all_true)
    
#     try:
#         # Create binary predictions
#         all_preds_binary = (all_preds > 0.5).astype(int)
        
#         # Calculate metrics
#         report = classification_report(all_true, all_preds_binary, output_dict=True)
#         report_str = classification_report(all_true, all_preds_binary)
#         conf_matrix = confusion_matrix(all_true, all_preds_binary)
        
#         log_message("\nClassification Report:")
#         log_message(report_str)
        
#         log_message("\nConfusion Matrix:")
#         log_message(str(conf_matrix))
        
#         # Calculate ROC curve and AUC
#         fpr, tpr, _ = roc_curve(all_true, all_preds)
#         roc_auc = auc(fpr, tpr)
        
#         # Calculate Precision-Recall curve
#         precision, recall, _ = precision_recall_curve(all_true, all_preds)
#         avg_precision = average_precision_score(all_true, all_preds)
        
#         log_message(f"\nROC AUC: {roc_auc:.4f}")
#         log_message(f"Average Precision: {avg_precision:.4f}")
        
#         # Save results to file
#         with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
#             f.write("Classification Report:\n")
#             f.write(report_str)
#             f.write("\n\nConfusion Matrix:\n")
#             f.write(str(conf_matrix))
#             f.write(f"\n\nROC AUC: {roc_auc:.4f}")
#             f.write(f"\nAverage Precision: {avg_precision:.4f}")
        
#         # Create visualizations
#         try:
#             # 1. Confusion Matrix
#             plt.figure(figsize=(10, 8))
#             sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#                       xticklabels=['Negative', 'Positive'],
#                       yticklabels=['Negative', 'Positive'])
#             plt.xlabel('Predicted Label')
#             plt.ylabel('True Label')
#             plt.title('Confusion Matrix')
#             plt.tight_layout()
#             plt.savefig(os.path.join(output_dir, "visualizations", "confusion_matrix.png"), dpi=30...
#             plt.close()
            
#             # 2. ROC Curve
#             plt.figure(figsize=(10, 8))
#             plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
#             plt.plot([0, 1], [0, 1], 'k--')
#             plt.xlabel('False Positive Rate')
#             plt.ylabel('True Positive Rate')
#             plt.title('Receiver Operating Characteristic (ROC) Curve')
#             plt.legend(loc="lower right")
#             plt.grid(alpha=0.3)
#             plt.savefig(os.path.join(output_dir, "visualizations", "roc_curve.png"), dpi=300)
#             plt.close()
            
#             # 3. Precision-Recall Curve
#             plt.figure(figsize=(10, 8))
#             plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.3f})'...
#             plt.xlabel('Recall')
#             plt.ylabel('Precision')
#             plt.title('Precision-Recall Curve')
#             plt.legend(loc="upper right")
#             plt.grid(alpha=0.3)
#             plt.savefig(os.path.join(output_dir, "visualizations", "precision_recall_curve.png"), ...
#             plt.close()
            
#             # 4. Score distribution
#             plt.figure(figsize=(12, 6))
#             sns.histplot(all_preds[all_true == 0], bins=50, alpha=0.5, label='Negative Class', col...
#             sns.histplot(all_preds[all_true == 1], bins=50, alpha=0.5, label='Positive Class', col...
#             plt.xlabel('Prediction Score')
#             plt.ylabel('Count')
#             plt.title('Distribution of Prediction Scores by Class')
#             plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
#             plt.legend()
#             plt.grid(alpha=0.3)
#             plt.savefig(os.path.join(output_dir, "visualizations", "score_distribution.png"), dpi=...
#             plt.close()
            
#             log_message("Saved all visualization plots")
#         except Exception as plot_e:
#             log_message(f"Error creating visualization plots: {str(plot_e)}")
#             log_message(traceback.format_exc())
        
#         # Save predictions with metadata
#         if metadata and all_meta:
#             try:
#                 # Create results DataFrame with metadata
#                 results_data = []
                
#                 for i in range(len(all_preds)):
#                     if i < len(all_meta):
#                         result = {
#                             'prediction': float(all_preds[i]),
#                             'true_label': int(all_true[i]),
#                             'predicted_label': int(all_preds_binary[i]),
#                             'correct': int(all_preds_binary[i] == all_true[i])
#                         }
                        
#                         # Add metadata
#                         meta = all_meta[i]
#                         for key, value in meta.items():
#                             # Convert numpy types to Python native types
#                             if hasattr(value, 'dtype'):
#                                 if np.issubdtype(value.dtype, np.integer):
#                                     value = int(value)
#                                 elif np.issubdtype(value.dtype, np.floating):
#                                     value = float(value)
#                             result[key] = value
                        
#                         results_data.append(result)
                
#                 # Create DataFrame
#                 results_df = pd.DataFrame(results_data)
                
#                 # Save to CSV
#                 results_df.to_csv(os.path.join(output_dir, "test_predictions_with_metadata.csv"), ...
#                 log_message(f"Saved predictions with metadata to CSV, shape: {results_df.shape}")
#             except Exception as csv_e:
#                 log_message(f"Error saving predictions CSV: {str(csv_e)}")
#                 log_message(traceback.format_exc())
                
#                 # Try a simpler format
#                 try:
#                     simple_df = pd.DataFrame({
#                         'prediction': all_preds,
#                         'true_label': all_true,
#                         'predicted_label': all_preds_binary
#                     })
#                     simple_df.to_csv(os.path.join(output_dir, "simple_predictions.csv"), index=Fal...
#                     log_message("Saved simplified predictions to CSV")
#                 except:
#                     log_message("Failed to save predictions in any format")
        
#         # Return metrics dictionary
#         evaluation_metrics = {
#             'roc_auc': float(roc_auc),
#             'avg_precision': float(avg_precision),
#             'accuracy': float(report['accuracy']),
#             'precision': float(report['1']['precision']),
#             'recall': float(report['1']['recall']),
#             'f1_score': float(report['1']['f1-score']),
#             'num_samples': len(all_true),
#             'positive_rate': float(np.mean(all_true))
#         }
        
#         # Save metrics to JSON
#         import json
#         with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
#             json.dump(evaluation_metrics, f, indent=4)
        
#         log_message("Evaluation complete")
#         return evaluation_metrics
    
#     except Exception as metrics_e:
#         log_message(f"Error calculating metrics: {str(metrics_e)}")
#         log_message(traceback.format_exc())
        
#         # Return basic metrics that we can calculate
#         return {
#             'num_samples': len(all_true),
#             'positive_rate': float(np.mean(all_true)),
#             'mean_prediction': float(np.mean(all_preds)),
#             'error': str(metrics_e)
#         }

# def evaluate_model_with_visualizations(model, X_file, y_file, test_indices, output_dir, metadata_f...
#     """
#     Comprehensive evaluation with detailed visualizations.
#     """
#     import os
#     import gc
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from sklearn.metrics import (
#         classification_report, confusion_matrix, roc_curve, 
#         auc, precision_recall_curve, average_precision_score
#     )
#     import traceback
    
#     os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
#     # Setup logging
#     log_file = os.path.join(output_dir, "evaluation_log.txt")
#     def log_message(message):
#         from datetime import datetime
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(log_file, "a") as f:
#             f.write(f"[{timestamp}] {message}\n")
#         print(message)
    
#     log_message(f"Starting evaluation on {len(test_indices)} test samples")
    
#     # Load metadata if available
#     metadata = None
#     if metadata_file and os.path.exists(metadata_file):
#         try:
#             import pickle
#             with open(metadata_file, "rb") as f:
#                 metadata = pickle.load(f)
#             log_message(f"Loaded metadata from {metadata_file}")
#         except Exception as e:
#             log_message(f"Error loading metadata: {str(e)}")
    
#     # Load memory-mapped data
#     X = np.load(X_file, mmap_mode='r')
#     y = np.load(y_file, mmap_mode='r')
    
#     # Process test data in batches
#     batch_size = 500  # Smaller batch size for more reliable processing
#     num_batches = int(np.ceil(len(test_indices) / batch_size))
    
#     all_preds = []
#     all_true = []
#     all_meta = []
    
#     log_message(f"Processing test data in {num_batches} batches of size {batch_size}")
    
#     # Save test indices for reference
#     np.save(os.path.join(output_dir, "test_indices.npy"), test_indices)
    
#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(test_indices))
#         batch_indices = test_indices[start_idx:end_idx]
        
#         log_message(f"Processing batch {batch_idx+1}/{num_batches} with {len(batch_indices)} sampl...
        
#         try:
#             # Load batch data
#             batch_X = np.array([X[idx] for idx in batch_indices])
#             batch_y = np.array([y[idx] for idx in batch_indices])
            
#             # Store metadata if available
#             if metadata:
#                 try:
#                     batch_meta = []
#                     for idx in batch_indices:
#                         if idx in metadata:
#                             batch_meta.append(metadata[idx])
#                         else:
#                             # Create empty metadata if not found
#                             batch_meta.append({})
#                     all_meta.extend(batch_meta)
#                 except Exception as meta_e:
#                     log_message(f"Error extracting metadata for batch {batch_idx}: {str(meta_e)}")
#                     # Continue without metadata for this batch
            
#             # Predict
#             batch_preds = model.predict(batch_X, verbose=0)
            
#             # Store results
#             all_preds.extend(batch_preds.flatten())
#             all_true.extend(batch_y)
            
#             # Clean up
#             del batch_X, batch_y, batch_preds
#             gc.collect()
            
#             log_message(f"Completed batch {batch_idx+1}/{num_batches}")
            
#         except Exception as e:
#             log_message(f"Error processing batch {batch_idx+1}: {str(e)}")
#             log_message(traceback.format_exc())
            
#             # Try to process the batch in smaller chunks
#             try:
#                 log_message("Attempting to process batch in smaller chunks...")
#                 sub_batch_size = 50  # Much smaller batch
#                 sub_batches = int(np.ceil(len(batch_indices) / sub_batch_size))
                
#                 for sub_idx in range(sub_batches):
#                     sub_start = sub_idx * sub_batch_size
#                     sub_end = min((sub_idx + 1) * sub_batch_size, len(batch_indices))
#                     sub_indices = batch_indices[sub_start:sub_end]
                    
#                     # Load and process sub-batch
#                     sub_X = np.array([X[idx] for idx in sub_indices])
#                     sub_y = np.array([y[idx] for idx in sub_indices])
                    
#                     # Add metadata if available
#                     if metadata:
#                         sub_meta = []
#                         for idx in sub_indices:
#                             if idx in metadata:
#                                 sub_meta.append(metadata[idx])
#                             else:
#                                 sub_meta.append({})
#                         all_meta.extend(sub_meta)
                    
#                     # Predict
#                     sub_preds = model.predict(sub_X, verbose=0)
                    
#                     # Store results
#                     all_preds.extend(sub_preds.flatten())
#                     all_true.extend(sub_y)
                    
#                     # Clean up
#                     del sub_X, sub_y, sub_preds
#                     gc.collect()
                
#                 log_message(f"Successfully processed batch {batch_idx+1} in {sub_batches} sub-batc...
#             except Exception as sub_e:
#                 log_message(f"Error processing sub-batches: {str(sub_e)}")
#                 log_message(traceback.format_exc())
#                 log_message(f"Skipping batch {batch_idx+1}")
    
#     log_message(f"Prediction complete. Total samples: {len(all_preds)}")
    
#     # Save raw predictions for reference
#     np.save(os.path.join(output_dir, "test_predictions.npy"), np.array(all_preds))
#     np.save(os.path.join(output_dir, "test_true_labels.npy"), np.array(all_true))
    
#     # Convert to numpy arrays
#     all_preds = np.array(all_preds)
#     all_true = np.array(all_true)
    
#     try:
#         # Create binary predictions
#         all_preds_binary = (all_preds > 0.5).astype(int)
        
#         # Calculate metrics
#         report = classification_report(all_true, all_preds_binary, output_dict=True)
#         report_str = classification_report(all_true, all_preds_binary)
#         conf_matrix = confusion_matrix(all_true, all_preds_binary)
        
#         log_message("\nClassification Report:")
#         log_message(report_str)
        
#         log_message("\nConfusion Matrix:")
#         log_message(str(conf_matrix))
        
#         # Calculate ROC curve and AUC
#         fpr, tpr, _ = roc_curve(all_true, all_preds)
#         roc_auc = auc(fpr, tpr)
        
#         # Calculate Precision-Recall curve
#         precision, recall, _ = precision_recall_curve(all_true, all_preds)
#         avg_precision = average_precision_score(all_true, all_preds)
        
#         log_message(f"\nROC AUC: {roc_auc:.4f}")
#         log_message(f"Average Precision: {avg_precision:.4f}")
        
#         # Save results to file
#         with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
#             f.write("Classification Report:\n")
#             f.write(report_str)
#             f.write("\n\nConfusion Matrix:\n")
#             f.write(str(conf_matrix))
#             f.write(f"\n\nROC AUC: {roc_auc:.4f}")
#             f.write(f"\nAverage Precision: {avg_precision:.4f}")
        
#         # Create visualizations
#         try:
#             # 1. Confusion Matrix
#             plt.figure(figsize=(10, 8))
#             sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#                       xticklabels=['Negative', 'Positive'],
#                       yticklabels=['Negative', 'Positive'])
#             plt.xlabel('Predicted Label')
#             plt.ylabel('True Label')
#             plt.title('Confusion Matrix')
#             plt.tight_layout()
#             plt.savefig(os.path.join(output_dir, "visualizations", "confusion_matrix.png"), dpi=30...
#             plt.close()
            
#             # 2. ROC Curve
#             plt.figure(figsize=(10, 8))
#             plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
#             plt.plot([0, 1], [0, 1], 'k--')
#             plt.xlabel('False Positive Rate')
#             plt.ylabel('True Positive Rate')
#             plt.title('Receiver Operating Characteristic (ROC) Curve')
#             plt.legend(loc="lower right")
#             plt.grid(alpha=0.3)
#             plt.savefig(os.path.join(output_dir, "visualizations", "roc_curve.png"), dpi=300)
#             plt.close()
            
#             # 3. Precision-Recall Curve
#             plt.figure(figsize=(10, 8))
#             plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.3f})'...
#             plt.xlabel('Recall')
#             plt.ylabel('Precision')
#             plt.title('Precision-Recall Curve')
#             plt.legend(loc="upper right")
#             plt.grid(alpha=0.3)
#             plt.savefig(os.path.join(output_dir, "visualizations", "precision_recall_curve.png"), ...
#             plt.close()
            
#             # 4. Score distribution
#             plt.figure(figsize=(12, 6))
#             sns.histplot(all_preds[all_true == 0], bins=50, alpha=0.5, label='Negative Class', col...
#             sns.histplot(all_preds[all_true == 1], bins=50, alpha=0.5, label='Positive Class', col...
#             plt.xlabel('Prediction Score')
#             plt.ylabel('Count')
#             plt.title('Distribution of Prediction Scores by Class')
#             plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
#             plt.legend()
#             plt.grid(alpha=0.3)
#             plt.savefig(os.path.join(output_dir, "visualizations", "score_distribution.png"), dpi=...
#             plt.close()
            
#             log_message("Saved all visualization plots")
#         except Exception as plot_e:
#             log_message(f"Error creating visualization plots: {str(plot_e)}")
#             log_message(traceback.format_exc())
        
#         # Save predictions with metadata
#         if metadata and all_meta:
#             try:
#                 # Create results DataFrame with metadata
#                 results_data = []
                
#                 for i in range(len(all_preds)):
#                     if i < len(all_meta):
#                         result = {
#                             'prediction': float(all_preds[i]),
#                             'true_label': int(all_true[i]),
#                             'predicted_label': int(all_preds_binary[i]),
#                             'correct': int(all_preds_binary[i] == all_true[i])
#                         }
                        
#                         # Add metadata
#                         meta = all_meta[i]
#                         for key, value in meta.items():
#                             # Convert numpy types to Python native types
#                             if hasattr(value, 'dtype'):
#                                 if np.issubdtype(value.dtype, np.integer):
#                                     value = int(value)
#                                 elif np.issubdtype(value.dtype, np.floating):
#                                     value = float(value)
#                             result[key] = value
                        
#                         results_data.append(result)
                
#                 # Create DataFrame
#                 results_df = pd.DataFrame(results_data)
                
#                 # Save to CSV
#                 results_df.to_csv(os.path.join(output_dir, "test_predictions_with_metadata.csv"), ...
#                 log_message(f"Saved predictions with metadata to CSV, shape: {results_df.shape}")
#             except Exception as csv_e:
#                 log_message(f"Error saving predictions CSV: {str(csv_e)}")
#                 log_message(traceback.format_exc())
                
#                 # Try a simpler format
#                 try:
#                     simple_df = pd.DataFrame({
#                         'prediction': all_preds,
#                         'true_label': all_true,
#                         'predicted_label': all_preds_binary
#                     })
#                     simple_df.to_csv(os.path.join(output_dir, "simple_predictions.csv"), index=Fal...
#                     log_message("Saved simplified predictions to CSV")
#                 except:
#                     log_message("Failed to save predictions in any format")
        
#         # Return metrics dictionary
#         evaluation_metrics = {
#             'roc_auc': float(roc_auc),
#             'avg_precision': float(avg_precision),
#             'accuracy': float(report['accuracy']),
#             'precision': float(report['1']['precision']),
#             'recall': float(report['1']['recall']),
#             'f1_score': float(report['1']['f1-score']),
#             'num_samples': len(all_true),
#             'positive_rate': float(np.mean(all_true))
#         }
        
#         # Save metrics to JSON
#         import json
#         with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
#             json.dump(evaluation_metrics, f, indent=4)
        
#         log_message("Evaluation complete")
#         return evaluation_metrics
    
#     except Exception as metrics_e:
#         log_message(f"Error calculating metrics: {str(metrics_e)}")
#         log_message(traceback.format_exc())
        
#         # Return basic metrics that we can calculate
#         return {
#             'num_samples': len(all_true),
#             'positive_rate': float(np.mean(all_true)),
#             'mean_prediction': float(np.mean(all_preds)),
#             'error': str(metrics_e)
#         }


# def analyze_spatial_performance(output_dir, metadata_file=None):
#     """
#     Analyze model performance across different geographical regions.
    
#     Parameters:
#     -----------
#     output_dir : str
#         Directory with test predictions
#     metadata_file : str
#         Path to metadata pickle file
        
#     Returns:
#     --------
#     dict
#         Spatial analysis results
#     """
#     import os
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import json
#     import traceback
    
#     # Setup logging
#     log_file = os.path.join(output_dir, "spatial_analysis_log.txt")
#     def log_message(message):
#         from datetime import datetime
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(log_file, "a") as f:
#             f.write(f"[{timestamp}] {message}\n")
#         print(message)
    
#     # Create visualizations directory
#     vis_dir = os.path.join(output_dir, "visualizations", "spatial")
#     os.makedirs(vis_dir, exist_ok=True)
    
#     log_message("Starting spatial performance analysis")
    
#     try:
#         # Load predictions with metadata
#         pred_file = os.path.join(output_dir, "test_predictions_with_metadata.csv")
#         if os.path.exists(pred_file):
#             log_message(f"Loading predictions with metadata from {pred_file}")
#             predictions = pd.read_csv(pred_file)
#         else:
# log_message("No predictions with metadata found, attempting to...
#             # Load separate files and combine
#             preds = np.load(os.path.join(output_dir, "test_predictions.npy"))
#             true = np.load(os.path.join(output_dir, "test_true_labels.npy"))
#             test_indices = np.load(os.path.join(output_dir, "test_indices.npy"))
            
#             log_message(f"Loaded {len(preds)} predictions and {len(true)} true labels")
            
#             # Create dataframe
#             predictions = pd.DataFrame({
#                 'prediction': preds,
#                 'true_label': true,
#                 'predicted_label': (preds > 0.5).astype(int),
#                 'correct': ((preds > 0.5).astype(int) == true).astype(int)
#             })
            
#             # Load metadata
#             if metadata_file and os.path.exists(metadata_file):
#                 try:
#                     import pickle
#                     with open(metadata_file, "rb") as f:
#                         metadata = pickle.load(f)
                    
#                     log_message(f"Loaded metadata with {len(metadata)} entries")
                    
#                     # Add metadata
#                     for i, idx in enumerate(test_indices):
#                         if i < len(predictions) and idx in metadata:
#                             meta = metadata[idx]
#                             for key, value in meta.items():
#                                 if key not in predictions.columns:
#                                     predictions[key] = None
#                                 predictions.loc[i, key] = value
                        
#                     log_message(f"Added metadata to predictions, columns: {list(predictions.column...
#                 except Exception as meta_e:
#                     log_message(f"Error adding metadata: {str(meta_e)}")
#                     log_message(traceback.format_exc())
        
#         # Check if we have spatial data
# if 'latitude' not in predictions.columns or 'longitude'...
#             log_message("No spatial data (latitude/longitude) found in predictions")
#             return {'error': 'No spatial data available'}
        
#         log_message(f"Predictions dataframe loaded with shape {predictions.shape}")
        
#         # Define latitude bands
#         def get_lat_band(lat):
#             if pd.isna(lat):
#                 return "Unknown"
#             lat = float(lat)
#             if lat < 55:
#                 return "<55°N"
#             elif lat < 60:
#                 return "55-60°N"
#             elif lat < 66.5:
#                 return "60-66.5°N (Subarctic)"
#             elif lat < 70:
#                 return "66.5-70°N (Arctic)"
#             elif lat < 75:
#                 return "70-75°N (Arctic)"
#             elif lat < 80:
#                 return "75-80°N (Arctic)"
#             else:
#                 return ">80°N (Arctic)"
        
#         # Add latitude band
#         predictions['lat_band'] = predictions['latitude'].apply(get_lat_band)
        
#         # Group by source (site) if available
#         if 'source' in predictions.columns:
#             log_message("Grouping metrics by source site")
#             site_metrics = predictions.groupby('source').apply(lambda x: pd.Series({
#                 'count': len(x),
#                 'positive_count': x['true_label'].sum(),
#                 'positive_rate': x['true_label'].mean(),
#                 'accuracy': x['correct'].mean(),
#                 'precision': (x['predicted_label'] & x['true_label']).sum() / max(1, x['predicted_...
#                 'recall': (x['predicted_label'] & x['true_label']).sum() / max(1, x['true_label']....
#                 'f1': 2 * (x['predicted_label'] & x['true_label']).sum() / 
#                       max(1, (x['predicted_label'] & x['true_label']).sum() + 
#                           0.5 * (x['predicted_label'].sum() + x['true_label'].sum())),
#                 'latitude': x['latitude'].mean(),
#                 'longitude': x['longitude'].mean(),
#                 'lat_band': x['lat_band'].iloc[0] if not x['lat_band'].isna().all() else "Unknown"
#             }))
            
#             # Filter out sites with too few samples
#             site_metrics = site_metrics[site_metrics['count'] >= 10]
#             log_message(f"Found {len(site_metrics)} sites with at least 10 samples")
            
#             # Save site metrics
#             site_metrics.to_csv(os.path.join(vis_dir, "site_metrics.csv"))
#         else:
#             site_metrics = None
#             log_message("No 'source' column found, skipping site-level analysis")
        
#         # Group by latitude band
#         log_message("Grouping metrics by latitude band")
#         band_metrics = predictions.groupby('lat_band').apply(lambda x: pd.Series({
#             'count': len(x),
#             'positive_count': x['true_label'].sum(),
#             'positive_rate': x['true_label'].mean(),
#             'accuracy': x['correct'].mean(),
#             'precision': (x['predicted_label'] & x['true_label']).sum() / max(1, x['predicted_labe...
#             'recall': (x['predicted_label'] & x['true_label']).sum() / max(1, x['true_label'].sum(...
#             'f1': 2 * (x['predicted_label'] & x['true_label']).sum() / 
#                   max(1, (x['predicted_label'] & x['true_label']).sum() + 
#                       0.5 * (x['predicted_label'].sum() + x['true_label'].sum()))
#         }))
        
#         # Save latitude band metrics
#         band_metrics.to_csv(os.path.join(vis_dir, "latitude_band_metrics.csv"))
#         log_message(f"Found {len(band_metrics)} latitude bands")
        
#         # Create visualizations
#         # 1. Performance by latitude band
#         try:
#             plt.figure(figsize=(14, 8))
            
#             # Order bands by latitude
#             ordered_bands = [
#                 '<55°N', '55-60°N', '60-66.5°N (Subarctic)', 
#                 '66.5-70°N (Arctic)', '70-75°N (Arctic)', 
#                 '75-80°N (Arctic)', '>80°N (Arctic)'
#             ]
# ordered_bands = [b for b in ordered_bands...
            
#             # Get metrics
#             band_data = band_metrics.loc[ordered_bands]
            
#             # Plot metrics
#             x = np.arange(len(ordered_bands))
#             width = 0.15
            
#             plt.bar(x - 2*width, band_data['accuracy'], width, label='Accuracy', color='#3274A1')
#             plt.bar(x - width, band_data['precision'], width, label='Precision', color='#E1812C')
#             plt.bar(x, band_data['recall'], width, label='Recall', color='#3A923A')
#             plt.bar(x + width, band_data['f1'], width, label='F1', color='#C03D3E')
#             plt.bar(x + 2*width, band_data['positive_rate'], width, label='Positive Rate', color='...
            
#             plt.xlabel('Latitude Band')
#             plt.ylabel('Score')
#             plt.title('Model Performance by Latitude Band')
#             plt.xticks(x, ordered_bands, rotation=45, ha='right')
#             plt.legend(loc='lower right')
#             plt.grid(axis='y', alpha=0.3)
#             plt.tight_layout()
#             plt.savefig(os.path.join(vis_dir, "performance_by_latitude.png"), dpi=300)
#             plt.close()
#             log_message("Created latitude band performance visualization")
#         except Exception as e:
#             log_message(f"Error creating latitude band plot: {str(e)}")
        
#         # 2. Map visualization if cartopy is available
#         try:
#             import cartopy.crs as ccrs
#             import cartopy.feature as cfeature
            
#             if site_metrics is not None:
#                 plt.figure(figsize=(15, 12))
                
#                 # Set up the projection
#                 ax = plt.axes(projection=ccrs.NorthPolarStereo())
#                 ax.set_extent([-180, 180, 45, 90], ccrs.PlateCarree())
                
#                 # Add map features
#                 ax.add_feature(cfeature.LAND, facecolor='lightgray')
#                 ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
#                 ax.add_feature(cfeature.COASTLINE)
#                 ax.add_feature(cfeature.BORDERS, linestyle=':')
                
#                 # Add gridlines
#                 gl = ax.gridlines(draw_labels=True)
#                 gl.top_labels = False
#                 gl.right_labels = False
                
#                 # Filter sites with valid coordinates
#                 valid_sites = site_metrics.dropna(subset=['latitude', 'longitude'])
                
#                 # Create scatter plot
#                 scatter = ax.scatter(
#                     valid_sites['longitude'], 
#                     valid_sites['latitude'],
#                     transform=ccrs.PlateCarree(),
#                     c=valid_sites['f1'],
#                     s=valid_sites['count'] / 5,  # Size by sample count
#                     cmap='RdYlGn',
#                     vmin=0, vmax=1,
#                     alpha=0.8,
#                     edgecolor='black',
#                     linewidth=0.5
#                 )
                
#                 # Add colorbar
#                 cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
#                 cbar.set_label('F1 Score')
                
#                 # Add Arctic Circle
#                 theta = np.linspace(0, 2*np.pi, 100)
#                 center, radius = [0, 0], 90 - 66.5
#                 verts = np.vstack([radius*np.sin(theta), radius*np.cos(theta)]).T
#                 circle = plt.Line2D(verts[:, 0], verts[:, 1], color='blue', 
#                                   linestyle='--', transform=ax.transData)
#                 ax.add_line(circle)
                
#                 plt.title('Spatial Distribution of Model Performance (F1 Score)', fontsize=14)
#                 plt.savefig(os.path.join(vis_dir, "spatial_performance_map.png"), dpi=300, bbox_in...
#                 plt.close()
#                 log_message("Created spatial performance map")
#             else:
#                 log_message("Skipping spatial map due to missing site metrics")
#         except ImportError:
#             log_message("Cartopy not available for map visualization")
#         except Exception as map_e:
#             log_message(f"Error creating map visualization: {str(map_e)}")
        
#         # 3. Performance vs. sample count (if site metrics available)
#         if site_metrics is not None:
#             try:
#                 plt.figure(figsize=(12, 8))
#                 plt.scatter(site_metrics['count'], site_metrics['f1'], 
#                           c=site_metrics['positive_rate'], cmap='viridis', 
#                           alpha=0.7, s=50, edgecolor='black', linewidth=1)
#                 plt.colorbar(label='Positive Rate')
#                 plt.xscale('log')
#                 plt.xlabel('Number of Samples')
#                 plt.ylabel('F1 Score')
#                 plt.title('F1 Score vs. Sample Count by Site')
#                 plt.grid(alpha=0.3)
#                 plt.savefig(os.path.join(vis_dir, "performance_vs_sample_count.png"), dpi=300)
#                 plt.close()
#                 log_message("Created performance vs sample count plot")
#             except Exception as e:
#                 log_message(f"Error creating performance vs sample count plot: {str(e)}")
        
#         # Return summary
#         summary = {
#             'latitude_bands': {
#                 'count': len(band_metrics),
#                 'bands': band_metrics.to_dict()
#             }
#         }
        
#         if site_metrics is not None:
#             summary['sites'] = {
#                 'count': len(site_metrics),
#                 'best_performing': site_metrics.nlargest(5, 'f1')[['f1', 'count', 'positive_rate']...
#                 'worst_performing': site_metrics.nsmallest(5, 'f1')[['f1', 'count', 'positive_rate...
#             }
        
#         # Save summary
#         with open(os.path.join(vis_dir, "spatial_summary.json"), "w") as f:
#             # Handle NumPy types
#             class NumpyEncoder(json.JSONEncoder):
#                 def default(self, obj):
#                     if isinstance(obj, (np.integer, np.floating, np.ndarray)):
#                         return float(obj)
#                     return super(NumpyEncoder, self).default(obj)
            
#             json.dump(summary, f, indent=4, cls=NumpyEncoder)
        
#         log_message("Spatial analysis complete")
#         return summary
    
#     except Exception as e:
#         log_message(f"Error in spatial analysis: {str(e)}")
#         log_message(traceback.format_exc())
#         return {'error': str(e)}

# def analyze_feature_importance(model, X_file, y_file, test_indices, output_dir):
#     """
#     Analyze feature importance using permutation importance.
    
#     Parameters:
#     -----------
#     model : tf.keras.Model
#         Trained model
#     X_file : str
#         Path to features numpy file
#     y_file : str
#         Path to labels numpy file
#     test_indices : numpy.ndarray
#         Test set indices
#     output_dir : str
#         Output directory for results
        
#     Returns:
#     --------
#     dict
#         Feature importance results
#     """
#     import os
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from sklearn.metrics import roc_auc_score
#     import gc
#     import traceback
    
#     # Create output directory
#     vis_dir = os.path.join(output_dir, "visualizations", "features")
#     os.makedirs(vis_dir, exist_ok=True)
    
#     # Setup logging
#     log_file = os.path.join(output_dir, "feature_importance_log.txt")
#     def log_message(message):
#         from datetime import datetime
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(log_file, "a") as f:
#             f.write(f"[{timestamp}] {message}\n")
#         print(message)
    
#     log_message("Starting feature importance analysis")
    
#     try:
#         # Load a subset of test data to save memory
# max_samples = min(5000, len(test_indices)) # Limit to...
#         np.random.seed(42)
#         if len(test_indices) > max_samples:
#             test_indices_subset = np.random.choice(test_indices, max_samples, replace=False)
#         else:
#             test_indices_subset = test_indices
        
#         log_message(f"Using {len(test_indices_subset)} samples for feature importance analysis")
        
#         # Load data
#         X = np.load(X_file, mmap_mode='r')
#         y = np.load(y_file, mmap_mode='r')
        
#         # Ensure the memory remains manageable
#         try:
#             X_test = np.array([X[idx] for idx in test_indices_subset])
#             y_test = np.array([y[idx] for idx in test_indices_subset])
#         except MemoryError:
#             # If memory error, reduce sample size further
#             max_samples = min(1000, len(test_indices))
#             test_indices_subset = np.random.choice(test_indices, max_samples, replace=False)
#             log_message(f"Memory error, reducing to {max_samples} samples")
#             X_test = np.array([X[idx] for idx in test_indices_subset])
#             y_test = np.array([y[idx] for idx in test_indices_subset])
        
#         log_message(f"Loaded test data with shape {X_test.shape}")
        
#         # Get feature names (modify as needed based on your data)
#         feature_names = ['Temperature', 'Temperature Gradient', 'Depth', 'Moisture', 'Moisture Gra...
        
#         # Truncate to actual number of features
#         feature_names = feature_names[:X_test.shape[2]]
#         log_message(f"Using feature names: {feature_names}")
        
#         # Get baseline performance
#         baseline_preds = model.predict(X_test, verbose=0)
#         baseline_auc = roc_auc_score(y_test, baseline_preds)
#         log_message(f"Baseline AUC: {baseline_auc:.4f}")
        
#         # Perform permutation importance analysis
#         n_repeats = 3  # Reduced from 5 to save time
#         importances = np.zeros((len(feature_names), n_repeats))
        
#         for feature_idx in range(len(feature_names)):
#             feature_name = feature_names[feature_idx]
#             log_message(f"Analyzing importance of feature: {feature_name}")
            
#             for repeat in range(n_repeats):
#                 log_message(f"  Repeat {repeat+1}/{n_repeats}")
                
#                 # Create a copy of the test data
#                 X_permuted = X_test.copy()
                
#                 # Permute the feature across all time steps
#                 for time_step in range(X_test.shape[1]):
#                     X_permuted[:, time_step, feature_idx] = np.random.permutation(X_test[:, time_s...
                
#                 # Predict with permuted feature
#                 perm_preds = model.predict(X_permuted, verbose=0)
#                 perm_auc = roc_auc_score(y_test, perm_preds)
                
#                 # Store importance (decrease in performance)
#                 importances[feature_idx, repeat] = baseline_auc - perm_auc
#                 log_message(f"    Decrease in AUC: {importances[feature_idx, repeat]:.4f}")
                
#                 # Clean up
#                 del X_permuted, perm_preds
#                 gc.collect()
        
#         # Calculate mean and std of importance
#         mean_importances = np.mean(importances, axis=1)
#         std_importances = np.std(importances, axis=1)
        
#         # Create DataFrame for results
#         importance_df = pd.DataFrame({
#             'Feature': feature_names,
#             'Importance': mean_importances,
#             'Std': std_importances
#         })
        
#         # Sort by importance
#         importance_df = importance_df.sort_values('Importance', ascending=False)
        
#         log_message(f"Feature importance results:\n{importance_df}")
        
#         # Plot feature importance
#         plt.figure(figsize=(12, 8))
#         sns.barplot(x='Importance', y='Feature', data=importance_df, 
#                    xerr=importance_df['Std'], palette='viridis')
#         plt.title('Feature Importance (Permutation Method)')
#         plt.xlabel('Decrease in AUC when Feature is Permuted')
#         plt.ylabel('Feature')
#         plt.grid(axis='x', alpha=0.3)
#         plt.tight_layout()
#         plt.savefig(os.path.join(vis_dir, "feature_importance.png"), dpi=300)
#         plt.close()
        
#         # Save results
#         importance_df.to_csv(os.path.join(vis_dir, "feature_importance.csv"), index=False)
#         log_message("Saved feature importance results")
        
#         # Analyze temporal patterns in feature importance
#         log_message("Analyzing temporal importance patterns")
        
#         # We'll analyze which time steps in the sequence are most important
#         time_importances = np.zeros((X_test.shape[1], n_repeats))
        
#         for time_idx in range(X_test.shape[1]):
#             log_message(f"Analyzing importance of time step {time_idx+1}/{X_test.shape[1]}")
            
#             for repeat in range(n_repeats):
#                 # Create a copy of the test data
#                 X_permuted = X_test.copy()
                
#                 # Permute all features at this time step
#                 X_permuted[:, time_idx, :] = np.random.permutation(X_test[:, time_idx, :])
                
#                 # Predict with permuted time step
#                 perm_preds = model.predict(X_permuted, verbose=0)
#                 perm_auc = roc_auc_score(y_test, perm_preds)
                
#                 # Store importance
#                 time_importances[time_idx, repeat] = baseline_auc - perm_auc
                
#                 # Clean up
#                 del X_permuted, perm_preds
#                 gc.collect()
        
#         # Calculate mean and std
#         mean_time_importances = np.mean(time_importances, axis=1)
#         std_time_importances = np.std(time_importances, axis=1)
        
#         # Create DataFrame
#         time_importance_df = pd.DataFrame({
#             'Time_Step': np.arange(X_test.shape[1]),
#             'Importance': mean_time_importances,
#             'Std': std_time_importances
#         })
        
#         log_message(f"Time step importance results:\n{time_importance_df}")
        
#         # Plot time step importance
#         plt.figure(figsize=(14, 8))
#         plt.errorbar(time_importance_df['Time_Step'], time_importance_df['Importance'],
#                     yerr=time_importance_df['Std'], fmt='o-', capsize=5, linewidth=2, markersize=8...
#         plt.title('Importance of Time Steps in Sequence')
#         plt.xlabel('Time Step Index')
#         plt.ylabel('Decrease in AUC when Time Step is Permuted')
#         plt.grid(alpha=0.3)
#         plt.xticks(time_importance_df['Time_Step'])
#         plt.tight_layout()
#         plt.savefig(os.path.join(vis_dir, "time_step_importance.png"), dpi=300)
#         plt.close()
        
#         # Save time importance results
#         time_importance_df.to_csv(os.path.join(vis_dir, "time_step_importance.csv"), index=False)
#         log_message("Saved time step importance results")
        
#         # Feature interaction heatmap (optional, if memory allows)
#         try:
#             # If we have few enough features, analyze pairwise interactions
#             if len(feature_names) <= 5:
#                 log_message("Analyzing feature interactions")
#                 interactions = np.zeros((len(feature_names), len(feature_names)))
                
#                 for i in range(len(feature_names)):
#                     for j in range(i+1, len(feature_names)):
#                         # Skip diagonal
#                         if i == j:
#                             continue
                            
#                         # Create permuted dataset
#                         X_permuted = X_test.copy()
                        
#                         # Permute both features
#                         for time_step in range(X_test.shape[1]):
#                             # Generate permutation indices
#                             perm_idx = np.random.permutation(len(X_test))
#                             X_permuted[:, time_step, i] = X_test[perm_idx, time_step, i]
#                             X_permuted[:, time_step, j] = X_test[perm_idx, time_step, j]
                        
#                         # Predict and calculate AUC
#                         perm_preds = model.predict(X_permuted, verbose=0)
#                         perm_auc = roc_auc_score(y_test, perm_preds)
                        
#                         # Calculate interaction strength
# # We compare permuting both features together...
# interactions[i, j] = baseline_auc - perm_auc -...
#                         interactions[j, i] = interactions[i, j]  # Symmetric
                        
#                         # Clean up
#                         del X_permuted, perm_preds
#                         gc.collect()
                
#                 # Plot interaction heatmap
#                 plt.figure(figsize=(10, 8))
#                 sns.heatmap(interactions, annot=True, fmt=".3f", cmap="coolwarm",
#                            xticklabels=feature_names, yticklabels=feature_names)
#                 plt.title("Feature Interaction Strength")
#                 plt.tight_layout()
#                 plt.savefig(os.path.join(vis_dir, "feature_interactions.png"), dpi=300)
#                 plt.close()
                
#                 log_message("Saved feature interaction analysis")
#         except Exception as inter_e:
#             log_message(f"Error in feature interaction analysis: {str(inter_e)}")
        
#         # Clean up
#         del X_test, y_test
#         gc.collect()
        
#         log_message("Feature importance analysis complete")
        
#         return {
#             'feature_importance': importance_df.to_dict(orient='records'),
#             'time_step_importance': time_importance_df.to_dict(orient='records'),
#             'baseline_auc': float(baseline_auc)
#         }
    
#     except Exception as e:
#         log_message(f"Error in feature importance analysis: {str(e)}")
#         log_message(traceback.format_exc())
#         return {'error': str(e)}

# def failsafe_training_fixed(model, X_file, y_file, train_indices, val_indices, test_indices,
#                            output_dir, batch_size=256, epochs=3, class_weight=None, start_batch=0)...
#     """
# Fixed robust training function that prevents stalling...
#     without changing the model architecture.
#     """
#     import os
#     import numpy as np
#     import tensorflow as tf
#     import gc
#     import time
#     import psutil
#     from datetime import datetime, timedelta
    
#     # Create output directories
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
#     # Setup logging
#     log_file = os.path.join(output_dir, "logs", "training_log.txt")
#     def log_message(message):
#         with open(log_file, "a") as f:
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             f.write(f"[{timestamp}] {message}\n")
#         print(message)
    
#     log_message("Starting training with original model architecture")
    
#     # Load data in memory-mapped mode
#     X_mmap = np.load(X_file, mmap_mode='r')
#     y_mmap = np.load(y_file, mmap_mode='r')
    
#     # Configuration for validation
#     # Use a subset of validation data to save memory
#     val_limit = min(1000, len(val_indices))
#     val_indices_subset = val_indices[:val_limit]
#     val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
#     val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    
#     # Training configuration
# total_batches = len(train_indices) // batch_size + (1...
#     log_message(f"Total batches: {total_batches}")
    
#     # Load a previous model if starting from a later batch
#     if start_batch > 0:
#         checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{start_batch}.h5")
#         if os.path.exists(checkpoint_path):
#             log_message(f"Loading model from checkpoint {checkpoint_path}")
#             model = tf.keras.models.load_model(checkpoint_path)
#         else:
#             # Find the latest checkpoint before start_batch
#             checkpoint_files = [f for f in os.listdir(os.path.join(output_dir, "checkpoints")) 
#                                if f.startswith("model_batch_") and f.endswith(".h5")]
#             batch_numbers = []
#             for file in checkpoint_files:
#                 try:
#                     batch_num = int(file.split("_")[-1].split(".")[0])
#                     if batch_num < start_batch:
#                         batch_numbers.append(batch_num)
#                 except:
#                     continue
            
#             if batch_numbers:
#                 latest_batch = max(batch_numbers)
#                 checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{latest_ba...
#                 log_message(f"Loading model from latest checkpoint {checkpoint_path}")
#                 model = tf.keras.models.load_model(checkpoint_path)
    
#     # Store best validation metrics
#     best_val_auc = 0
#     best_model_path = None
#     stall_counter = 0
#     max_stalls = 3
    
#     # Track progress
#     start_time = time.time()
    
#     # Main training loop - process batches
#     for batch_idx in range(start_batch, total_batches):
#         batch_start_time = time.time()
        
#         # Calculate batch indices
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
        
#         # Get current batch indices
#         batch_indices = train_indices[start_idx:end_idx]
        
#         # Report memory usage
#         current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
#         log_message(f"Memory before batch load: {current_memory:.1f} MB")
        
#         # Load batch data with chunking to avoid memory issues
#         try:
#             # Load in smaller chunks
#             chunk_size = min(64, len(batch_indices))
#             batch_X_chunks = []
#             batch_y_chunks = []
            
#             for i in range(0, len(batch_indices), chunk_size):
#                 end_i = min(i + chunk_size, len(batch_indices))
#                 chunk_indices = batch_indices[i:end_i]
                
#                 chunk_X = np.array([X_mmap[idx] for idx in chunk_indices])
#                 chunk_y = np.array([y_mmap[idx] for idx in chunk_indices])
                
#                 batch_X_chunks.append(chunk_X)
#                 batch_y_chunks.append(chunk_y)
                
#                 # Clean up chunks immediately
#                 gc.collect()
            
#             # Combine chunks
#             batch_X = np.concatenate(batch_X_chunks)
#             batch_y = np.concatenate(batch_y_chunks)
            
#             # Clean up chunk lists
#             del batch_X_chunks, batch_y_chunks
#             gc.collect()
            
#         except Exception as e:
#             log_message(f"Error loading batch data: {e}")
#             continue
        
#         # Train with progressive mini-batch size reduction if needed
#         mini_batch_sizes = [min(batch_size, 128), 64, 32, 16]
#         training_successful = False
        
#         for mini_batch_size in mini_batch_sizes:
#             if training_successful:
#                 break
                
#             try:
#                 log_message(f"Trying mini-batch size {mini_batch_size}")
                
#                 # Train for one epoch
#                 history = model.fit(
#                     batch_X, batch_y,
#                     epochs=1,
#                     verbose=1,
#                     class_weight=class_weight,
#                     batch_size=mini_batch_size
#                 )
                
#                 # Check for NaN loss
#                 if np.isnan(history.history['loss'][0]):
#                     log_message(f"NaN loss detected with mini-batch size {mini_batch_size}")
#                     continue
                
#                 # Check for AUC value
#                 if 'auc' in history.history and history.history['auc'][0] == 0:
#                     stall_counter += 1
#                     log_message(f"Warning: AUC is zero, stall detected. Counter: {stall_counter}/{...
                    
#                     if stall_counter >= max_stalls:
#                         log_message("Multiple stalls detected, resetting optimizer state")
                        
#                         # Save weights
#                         temp_weights_path = os.path.join(output_dir, "checkpoints", "temp_weights....
#                         model.save_weights(temp_weights_path)
                        
#                         # Get current learning rate
#                         if hasattr(model.optimizer, 'lr'):
#                             current_lr = float(model.optimizer.lr.numpy())
#                             # Reduce learning rate
#                             new_lr = current_lr * 0.5
#                             log_message(f"Reducing learning rate from {current_lr} to {new_lr}")
#                         else:
#                             new_lr = 0.0001
#                             log_message(f"Setting learning rate to {new_lr}")
                        
#                         # Recompile with fresh optimizer
#                         model.compile(
#                             optimizer=tf.keras.optimizers.Adam(learning_rate=new_lr),
#                             loss='binary_crossentropy',
#                             metrics=[
#                                 'accuracy',
#                                 tf.keras.metrics.AUC(name='auc'),
#                                 tf.keras.metrics.Precision(name='precision'),
#                                 tf.keras.metrics.Recall(name='recall')
#                             ]
#                         )
                        
#                         # Load weights back
#                         model.load_weights(temp_weights_path)
                        
#                         # Reset counter
#                         stall_counter = 0
#                 else:
#                     # Reset stall counter if we have non-zero AUC
#                     stall_counter = 0
                
#                 # Mark as successful
#                 training_successful = True
                
#             except tf.errors.ResourceExhaustedError:
#                 log_message(f"Resource exhausted with mini-batch size {mini_batch_size}, trying sm...
#                 continue
#             except Exception as e:
#                 log_message(f"Error during training with mini-batch size {mini_batch_size}: {e}")
#                 continue
        
#         # If all mini-batch sizes failed, skip this batch
#         if not training_successful:
#             log_message(f"All mini-batch sizes failed for batch {batch_idx+1}, skipping")
            
#             # If multiple consecutive batches fail, try resetting the session
#             if batch_idx > start_batch:
#                 log_message("Resetting TensorFlow session to recover")
                
#                 # Save model
#                 temp_path = os.path.join(output_dir, "checkpoints", f"recovery_{batch_idx}.h5")
#                 model.save(temp_path)
                
#                 # Clear session
#                 tf.keras.backend.clear_session()
#                 gc.collect()
                
#                 # Reload model
#                 model = tf.keras.models.load_model(temp_path)
            
#             continue
        
#         # Clean up batch data
#         del batch_X, batch_y
#         gc.collect()
        
#         # Evaluate on validation set every 10 batches
# if (batch_idx + 1) % 10 ==...
#             log_message(f"Evaluating on validation set at batch {batch_idx+1}/{total_batches}")
            
#             try:
#                 val_metrics = model.evaluate(val_X, val_y, verbose=0)
#                 val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
                
#                 # Display metrics
# metrics_str = " - ".join([f"{k}: {v:.4f}" for...
#                 log_message(f"Validation metrics at batch {batch_idx+1}/{total_batches}:")
#                 log_message(f"  {metrics_str}")
                
#                 # Check for better AUC
#                 val_auc = val_metrics_dict.get('auc', 0)
#                 if val_auc > best_val_auc:
#                     log_message(f"New best model saved with val_auc: {val_auc:.4f}")
#                     best_val_auc = val_auc
#                     best_model_path = os.path.join(output_dir, "checkpoints", f"best_model_batch_{...
#                     model.save(best_model_path)
                
# # If validation AUC is also stuck...
#                 if val_auc == 0 and stall_counter > 0:
#                     log_message("Critical: Validation AUC is zero, attempting advanced recovery")
                    
#                     # Save model state
#                     temp_path = os.path.join(output_dir, "checkpoints", f"pre_recovery_{batch_idx+...
#                     model.save(temp_path)
                    
#                     # Clear session
#                     tf.keras.backend.clear_session()
#                     gc.collect()
                    
#                     # Reload model
#                     model = tf.keras.models.load_model(temp_path)
                    
#                     # More aggressive learning rate reduction
#                     current_lr = 0.0001  # Default fallback
#                     if hasattr(model.optimizer, 'lr'):
#                         current_lr = float(model.optimizer.lr.numpy())
                    
#                     new_lr = current_lr * 0.1  # 10x reduction
#                     log_message(f"Emergency reduction of learning rate to {new_lr}")
                    
#                     # Recompile with fresh optimizer and adjusted metrics
#                     model.compile(
#                         optimizer=tf.keras.optimizers.Adam(
#                             learning_rate=new_lr,
#                             beta_1=0.9,
#                             beta_2=0.999,
#                             epsilon=1e-07,
#                             amsgrad=True
#                         ),
#                         loss='binary_crossentropy',
#                         metrics=[
#                             'accuracy',
#                             tf.keras.metrics.AUC(name='auc', num_thresholds=200),
#                             tf.keras.metrics.Precision(name='precision'),
#                             tf.keras.metrics.Recall(name='recall')
#                         ]
#                     )
                    
#                     # Reset counter after emergency intervention
#                     stall_counter = 0
                
#             except Exception as eval_e:
#                 log_message(f"Error during validation: {eval_e}")
        
#         # Save checkpoint every 100 batches
#         if (batch_idx + 1) % 100 == 0:
#             checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{batch_idx+1}....
#             model.save(checkpoint_path)
#             log_message(f"Checkpoint saved to {checkpoint_path}")
        
#         # Progress reporting
#         batch_time = time.time() - batch_start_time
#         elapsed = time.time() - start_time
# progress = (batch_idx - start_batch + 1)...
# remaining = (elapsed / progress) * (1...
        
#         log_message(f"Progress: {batch_idx+1}/{total_batches} batches ({progress*100:.1f}%)")
#         log_message(f"Batch time: {batch_time:.1f}s")
#         log_message(f"Elapsed: {timedelta(seconds=int(elapsed))}, Remaining: {timedelta(seconds=in...
        
#         # Memory tracking
#         current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
#         log_message(f"Memory usage: {current_memory:.1f} MB")
        
#         # Reset TensorFlow session periodically to prevent memory growth
#         if (batch_idx + 1) % 500 == 0 and batch_idx > 0:
#             log_message("Resetting TensorFlow session")
            
#             # Save model
#             temp_path = os.path.join(output_dir, "checkpoints", f"temp_reset_{batch_idx+1}.h5")
#             model.save(temp_path)
            
#             # Clear session
#             tf.keras.backend.clear_session()
#             gc.collect()
            
#             # Reload model
#             model = tf.keras.models.load_model(temp_path)
#             log_message("TensorFlow session reset complete")
    
#     # Save final model
#     final_model_path = os.path.join(output_dir, "final_model.h5")
#     model.save(final_model_path)
#     log_message(f"Training complete. Final model saved to {final_model_path}")
    
#     # Return best model if available
#     if best_model_path and os.path.exists(best_model_path):
#         log_message(f"Loading best model from {best_model_path}")
#         model = tf.keras.models.load_model(best_model_path)
#         return model, best_model_path
    
#     return model, final_model_path

# def fast_training(model, X_file, y_file, train_indices, val_indices, test_indices,
#                  output_dir, batch_size=256, class_weight=None, start_batch=0):
#     """
#     Accelerated training function that focuses on speed and efficiency.
#     """
#     import os
#     import numpy as np
#     import tensorflow as tf
#     import gc
#     import time
#     import psutil
#     from datetime import datetime, timedelta
#     import random
    
#     # Create output directories
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
#     # Setup logging
#     log_file = os.path.join(output_dir, "logs", "training_log.txt")
#     def log_message(message):
#         with open(log_file, "a") as f:
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             f.write(f"[{timestamp}] {message}\n")
#         print(message)
    
#     log_message("Starting accelerated training")
    
#     # Memory-mapped data access
#     X_mmap = np.load(X_file, mmap_mode='r')
#     y_mmap = np.load(y_file, mmap_mode='r')
    
#     # Smaller validation set to save memory
#     val_size = min(500, len(val_indices))
#     val_indices_sample = val_indices[:val_size]
#     val_X = np.array([X_mmap[idx] for idx in val_indices_sample])
#     val_y = np.array([y_mmap[idx] for idx in val_indices_sample])
    
#     # SPEED OPTIMIZATION 1: Use larger batches
#     effective_batch_size = batch_size * 4
#     log_message(f"Using accelerated batch size: {effective_batch_size}")
    
#     # SPEED OPTIMIZATION 2: Skip most validation steps
#     validation_frequency = 500  # Only validate every 500 batches
    
#     # SPEED OPTIMIZATION 3: Limit total batches (use a sample of data)
#     # This is critical to get results in a reasonable timeframe
#     max_batches = 5000  # Limit to 5000 batches instead of 54371
    
#     # SPEED OPTIMIZATION 4: Use a much higher learning rate
#     if start_batch == 0:
#         # Recompile with a higher learning rate
#         log_message("Setting higher learning rate")
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Much higher learning rate
#             loss='binary_crossentropy',
#             metrics=[
#                 'accuracy',
#                 tf.keras.metrics.AUC(name='auc'),
#                 tf.keras.metrics.Precision(name='precision'),
#                 tf.keras.metrics.Recall(name='recall')
#             ]
#         )
    
#     # Calculate approximate number of batches
# num_batches = min(max_batches, (len(train_indices) + effective_batch_size -...
# log_message(f"Training on {num_batches} batches (out of original...
    
#     # Load from checkpoint if resuming
#     if start_batch > 0:
#         checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{start_batch}.h5")
#         if os.path.exists(checkpoint_path):
#             log_message(f"Loading model from checkpoint {checkpoint_path}")
#             model = tf.keras.models.load_model(checkpoint_path)
#         else:
#             log_message("Checkpoint not found, starting with fresh model")
    
#     # Training tracking variables
#     stall_counter = 0
#     max_stalls = 3
#     best_val_auc = 0
#     best_model_path = None
#     start_time = time.time()
    
#     # Initial validation
#     log_message("Performing initial validation")
#     try:
#         val_metrics = model.evaluate(val_X, val_y, verbose=0)
#         val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
# metrics_str = " - ".join([f"{k}: {v:.4f}" for...
#         log_message(f"Initial validation: {metrics_str}")
#         best_val_auc = val_metrics_dict.get('auc', 0)
#     except Exception as e:
#         log_message(f"Error during initial validation: {e}")
    
# # SPEED OPTIMIZATION 5: Use a subset...
#     # We'll use stratified sampling to maintain class balance
#     # First, separate indices by class
#     pos_indices = []
#     neg_indices = []
    
#     # Sample 20% of data to determine class distribution
#     sample_size = min(10000, len(train_indices))
#     sample_indices = random.sample(list(train_indices), sample_size)
    
#     for idx in sample_indices:
#         if y_mmap[idx] > 0.5:
#             pos_indices.append(idx)
#         else:
#             neg_indices.append(idx)
    
#     # Calculate positive class ratio
#     pos_ratio = len(pos_indices) / sample_size
#     log_message(f"Positive class ratio in sample: {pos_ratio:.3f}")
    
#     # Create a balanced subset of indices for faster training
#     subset_size = min(num_batches * effective_batch_size, len(train_indices))
#     subset_pos_size = int(subset_size * pos_ratio)
#     subset_neg_size = subset_size - subset_pos_size
    
#     # Sample from each class
#     if len(pos_indices) > 0 and len(neg_indices) > 0:
#         # For positive class
#         if subset_pos_size <= len(pos_indices):
#             subset_pos_indices = random.sample(pos_indices, subset_pos_size)
#         else:
#             subset_pos_indices = pos_indices
            
#         # For negative class
#         if subset_neg_size <= len(neg_indices):
#             subset_neg_indices = random.sample(neg_indices, subset_neg_size)
#         else:
#             subset_neg_indices = neg_indices
            
#         # Combine and shuffle
#         training_subset = subset_pos_indices + subset_neg_indices
#         random.shuffle(training_subset)
#     else:
#         # Fallback if stratification fails
#         training_subset = random.sample(list(train_indices), min(subset_size, len(train_indices)))
    
#     log_message(f"Created training subset with {len(training_subset)} samples")
    
#     # Process each batch
#     for batch_idx in range(start_batch, num_batches):
#         batch_start_time = time.time()
        
#         # Calculate batch indices (using our subset)
#         start_idx = batch_idx * effective_batch_size
#         end_idx = min((batch_idx + 1) * effective_batch_size, len(training_subset))
        
#         # Skip if we've reached the end of our subset
#         if start_idx >= len(training_subset):
#             break
            
#         batch_indices = training_subset[start_idx:end_idx]
        
#         # Load batch data
#         try:
#             batch_X = np.array([X_mmap[idx] for idx in batch_indices])
#             batch_y = np.array([y_mmap[idx] for idx in batch_indices])
#         except Exception as e:
#             log_message(f"Error loading batch data: {e}")
#             continue
        
#         # Train on this batch
#         try:
# # SPEED OPTIMIZATION 6: More epochs per...
#             history = model.fit(
#                 batch_X, batch_y,
#                 epochs=3,  # Train for 3 epochs on each batch
#                 batch_size=min(len(batch_X), 128),
#                 class_weight=class_weight,
#                 verbose=1
#             )
            
#             # Check for NaN loss
#             if np.isnan(history.history['loss'][-1]):
#                 log_message(f"NaN loss detected, skipping batch")
#                 continue
                
#             # Check for stalls
#             if 'auc' in history.history and history.history['auc'][-1] == 0:
#                 stall_counter += 1
#                 log_message(f"Warning: AUC is zero, stall detected. Counter: {stall_counter}/{max_...
                
#                 if stall_counter >= max_stalls:
#                     log_message("Multiple stalls detected, resetting optimizer")
                    
#                     # Get current learning rate
#                     if hasattr(model.optimizer, 'lr'):
#                         current_lr = float(model.optimizer.lr.numpy())
#                         # But don't decrease too much
#                         new_lr = max(0.0001, current_lr * 0.8)
#                         log_message(f"Adjusting learning rate from {current_lr} to {new_lr}")
#                     else:
#                         new_lr = 0.0005
#                         log_message(f"Setting learning rate to {new_lr}")
                    
#                     # Recompile with fresh optimizer
#                     model.compile(
#                         optimizer=tf.keras.optimizers.Adam(learning_rate=new_lr),
#                         loss='binary_crossentropy',
#                         metrics=[
#                             'accuracy',
#                             tf.keras.metrics.AUC(name='auc'),
#                             tf.keras.metrics.Precision(name='precision'),
#                             tf.keras.metrics.Recall(name='recall')
#                         ]
#                     )
                    
#                     # Reset counter
#                     stall_counter = 0
#             else:
#                 # Reset stall counter if no stall
#                 stall_counter = 0
                
#         except tf.errors.ResourceExhaustedError:
#             log_message(f"Resource exhausted, reducing batch size")
            
#             # Try with smaller batch
#             try:
#                 mini_batch_size = min(64, len(batch_X))
#                 history = model.fit(
#                     batch_X, batch_y,
#                     epochs=1,
#                     batch_size=mini_batch_size,
#                     class_weight=class_weight,
#                     verbose=1
#                 )
#             except Exception as inner_e:
#                 log_message(f"Error with reduced batch: {inner_e}")
#                 continue
                
#         except Exception as e:
#             log_message(f"Error during training: {e}")
#             continue
        
#         # Clean up
#         del batch_X, batch_y
#         gc.collect()
        
#         # Validate at appropriate intervals
# if (batch_idx + 1) % validation_frequency ==...
#             log_message(f"Validating at batch {batch_idx+1}/{num_batches}")
            
#             try:
#                 val_metrics = model.evaluate(val_X, val_y, verbose=0)
#                 val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
# metrics_str = " - ".join([f"{k}: {v:.4f}" for...
#                 log_message(f"Validation metrics at batch {batch_idx+1}/{num_batches}:")
#                 log_message(f"  {metrics_str}")
                
#                 # Save if better
#                 val_auc = val_metrics_dict.get('auc', 0)
#                 if val_auc > best_val_auc:
#                     best_val_auc = val_auc
#                     best_model_path = os.path.join(output_dir, "checkpoints", f"best_model_batch_{...
#                     model.save(best_model_path)
#                     log_message(f"New best model saved with AUC {val_auc:.4f}")
            
#             except Exception as e:
#                 log_message(f"Error during validation: {e}")
        
#         # Save checkpoint regularly
#         if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == num_batches:
#             checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{batch_idx+1}....
#             model.save(checkpoint_path)
#             log_message(f"Checkpoint saved to {checkpoint_path}")
        
#         # Progress tracking
#         batch_time = time.time() - batch_start_time
#         elapsed = time.time() - start_time
#         progress = (batch_idx - start_batch + 1) / (num_batches - start_batch)
# remaining = (elapsed / progress) * (1...
        
#         log_message(f"Progress: {batch_idx+1}/{num_batches} batches ({progress*100:.1f}%)")
#         log_message(f"Batch time: {batch_time:.1f}s")
#         log_message(f"Elapsed: {timedelta(seconds=int(elapsed))}, Remaining: {timedelta(seconds=in...
        
#         # Reset TensorFlow session periodically to prevent memory growth
#         if (batch_idx + 1) % 500 == 0 and batch_idx > 0:
#             log_message("Performing session reset")
            
#             # Save model
#             temp_path = os.path.join(output_dir, "checkpoints", f"temp_reset_{batch_idx+1}.h5")
#             model.save(temp_path)
            
#             # Clear session
#             tf.keras.backend.clear_session()
#             gc.collect()
            
#             # Reload model
#             model = tf.keras.models.load_model(temp_path)
    
#     # Save final model
#     final_model_path = os.path.join(output_dir, "final_model.h5")
#     model.save(final_model_path)
#     log_message(f"Training complete. Final model saved to {final_model_path}")
    
#     # Return best model if available
#     if best_model_path and os.path.exists(best_model_path):
#         log_message(f"Loading best model from {best_model_path}")
#         model = tf.keras.models.load_model(best_model_path)
#         return model, best_model_path
    
#     return model, final_model_path

# def full_dataset_training(model, X_file, y_file, train_indices, val_indices, test_indices,
#                          output_dir, batch_size=256, class_weight=None, start_batch=0):
#     """
# Training function that uses the ENTIRE training...
#     while improving efficiency where possible.
#     """
#     import os
#     import numpy as np
#     import tensorflow as tf
#     import gc
#     import time
#     import psutil
#     from datetime import datetime, timedelta
    
#     # Create output directories
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
#     # Setup logging
#     log_file = os.path.join(output_dir, "logs", "training_log.txt")
#     def log_message(message):
#         with open(log_file, "a") as f:
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             f.write(f"[{timestamp}] {message}\n")
#         print(message)
    
#     log_message("Starting training with FULL training dataset")
    
#     # Load data in memory-mapped mode
#     X_mmap = np.load(X_file, mmap_mode='r')
#     y_mmap = np.load(y_file, mmap_mode='r')
    
#     # Configuration for validation
#     # Use a subset of validation data to save memory
#     val_limit = min(1000, len(val_indices))
#     val_indices_subset = val_indices[:val_limit]
#     val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
#     val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    
#     # Improved batch size for better utilization
#     # We still use the original batch size, but make sure it's efficient
#     effective_batch_size = batch_size
#     log_message(f"Using batch size: {effective_batch_size}")
    
#     # Calculate total batches
# total_batches = len(train_indices) // effective_batch_size + (1...
#     log_message(f"Training on all {total_batches} batches")
    
#     # Validation frequency
#     validation_frequency = 500  # Validate less frequently to save time
    
#     # Try a higher learning rate if starting fresh
#     if start_batch == 0:
#         # Recompile with a higher learning rate
#         log_message("Setting higher learning rate")
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Higher learning rate
#             loss='binary_crossentropy',
#             metrics=[
#                 'accuracy',
#                 tf.keras.metrics.AUC(name='auc'),
#                 tf.keras.metrics.Precision(name='precision'),
#                 tf.keras.metrics.Recall(name='recall')
#             ]
#         )
    
#     # Load from checkpoint if resuming
#     if start_batch > 0:
#         checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{start_batch}.h5")
#         if os.path.exists(checkpoint_path):
#             log_message(f"Loading model from checkpoint {checkpoint_path}")
#             model = tf.keras.models.load_model(checkpoint_path)
#         else:
#             log_message("Checkpoint not found, starting with fresh model")
    
#     # Training tracking variables
#     stall_counter = 0
#     max_stalls = 3
#     best_val_auc = 0
#     best_model_path = None
#     start_time = time.time()
    
#     # Initial validation
#     log_message("Performing initial validation")
#     try:
#         val_metrics = model.evaluate(val_X, val_y, verbose=0)
#         val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
# metrics_str = " - ".join([f"{k}: {v:.4f}" for...
#         log_message(f"Initial validation: {metrics_str}")
#         best_val_auc = val_metrics_dict.get('auc', 0)
#     except Exception as e:
#         log_message(f"Error during initial validation: {e}")
    
#     # Main training loop
#     for batch_idx in range(start_batch, total_batches):
#         batch_start_time = time.time()
        
#         # Calculate batch indices
#         start_idx = batch_idx * effective_batch_size
#         end_idx = min((batch_idx + 1) * effective_batch_size, len(train_indices))
#         batch_indices = train_indices[start_idx:end_idx]
        
#         # Report memory usage
#         current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
#         log_message(f"Memory before batch load: {current_memory:.1f} MB")
        
#         # Load batch data with chunking to avoid memory issues
#         try:
#             # Load in smaller chunks
#             chunk_size = min(64, len(batch_indices))
#             batch_X_chunks = []
#             batch_y_chunks = []
            
#             for i in range(0, len(batch_indices), chunk_size):
#                 end_i = min(i + chunk_size, len(batch_indices))
#                 chunk_indices = batch_indices[i:end_i]
                
#                 chunk_X = np.array([X_mmap[idx] for idx in chunk_indices])
#                 chunk_y = np.array([y_mmap[idx] for idx in chunk_indices])
                
#                 batch_X_chunks.append(chunk_X)
#                 batch_y_chunks.append(chunk_y)
                
#                 # Clean up chunks immediately
#                 gc.collect()
            
#             # Combine chunks
#             batch_X = np.concatenate(batch_X_chunks)
#             batch_y = np.concatenate(batch_y_chunks)
            
#             # Clean up chunk lists
#             del batch_X_chunks, batch_y_chunks
#             gc.collect()
            
#         except Exception as e:
#             log_message(f"Error loading batch data: {e}")
#             continue
        
#         # Train multiple epochs on this batch for better convergence
#         # But limit to 2 epochs to keep it moving
#         n_epochs = 2
#         log_message(f"Training for {n_epochs} epochs on batch {batch_idx+1}/{total_batches}")
        
#         # Try different mini-batch sizes if needed
#         mini_batch_sizes = [min(effective_batch_size, 128), 64, 32]
#         training_successful = False
        
#         for mini_batch_size in mini_batch_sizes:
#             if training_successful:
#                 break
                
#             try:
#                 log_message(f"Trying mini-batch size {mini_batch_size}")
                
#                 # Train for multiple epochs
#                 history = model.fit(
#                     batch_X, batch_y,
#                     epochs=n_epochs,
#                     verbose=1,
#                     class_weight=class_weight,
#                     batch_size=mini_batch_size
#                 )
                
#                 # Check for NaN loss
#                 if np.isnan(history.history['loss'][-1]):
#                     log_message(f"NaN loss detected with mini-batch size {mini_batch_size}")
#                     continue
                
#                 # Check for AUC value
#                 if 'auc' in history.history and history.history['auc'][-1] == 0:
#                     stall_counter += 1
#                     log_message(f"Warning: AUC is zero, stall detected. Counter: {stall_counter}/{...
                    
#                     if stall_counter >= max_stalls:
#                         log_message("Multiple stalls detected, resetting optimizer state")
                        
#                         # Get current learning rate
#                         if hasattr(model.optimizer, 'lr'):
#                             current_lr = float(model.optimizer.lr.numpy())
#                             # But don't decrease too much
#                             new_lr = max(0.0001, current_lr * 0.5)
#                             log_message(f"Adjusting learning rate from {current_lr} to {new_lr}")
#                         else:
#                             new_lr = 0.0005
#                             log_message(f"Setting learning rate to {new_lr}")
                        
#                         # Recompile with fresh optimizer
#                         model.compile(
#                             optimizer=tf.keras.optimizers.Adam(learning_rate=new_lr),
#                             loss='binary_crossentropy',
#                             metrics=[
#                                 'accuracy',
#                                 tf.keras.metrics.AUC(name='auc'),
#                                 tf.keras.metrics.Precision(name='precision'),
#                                 tf.keras.metrics.Recall(name='recall')
#                             ]
#                         )
                        
#                         # Reset counter
#                         stall_counter = 0
#                 else:
#                     # Reset stall counter if no stall
#                     stall_counter = 0
                
#                 # Mark as successful
#                 training_successful = True
                
#             except tf.errors.ResourceExhaustedError:
#                 log_message(f"Resource exhausted with mini-batch size {mini_batch_size}, trying sm...
#                 continue
#             except Exception as e:
#                 log_message(f"Error during training with mini-batch size {mini_batch_size}: {e}")
#                 continue
        
#         # If all mini-batch sizes failed, skip this batch
#         if not training_successful:
#             log_message(f"All mini-batch sizes failed for batch {batch_idx+1}, skipping")
#             continue
        
#         # Clean up
#         del batch_X, batch_y
#         gc.collect()
        
#         # Validate at appropriate intervals
# if (batch_idx + 1) % validation_frequency ==...
#             log_message(f"Validating at batch {batch_idx+1}/{total_batches}")
            
#             try:
#                 val_metrics = model.evaluate(val_X, val_y, verbose=0)
#                 val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
# metrics_str = " - ".join([f"{k}: {v:.4f}" for...
#                 log_message(f"Validation metrics at batch {batch_idx+1}/{total_batches}:")
#                 log_message(f"  {metrics_str}")
                
#                 # Save if better
#                 val_auc = val_metrics_dict.get('auc', 0)
#                 if val_auc > best_val_auc:
#                     best_val_auc = val_auc
#                     best_model_path = os.path.join(output_dir, "checkpoints", f"best_model_batch_{...
#                     model.save(best_model_path)
#                     log_message(f"New best model saved with AUC {val_auc:.4f}")
            
#             except Exception as e:
#                 log_message(f"Error during validation: {e}")
        
#         # Save checkpoint every 1000 batches
#         if (batch_idx + 1) % 1000 == 0 or (batch_idx + 1) == total_batches:
#             checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{batch_idx+1}....
#             model.save(checkpoint_path)
#             log_message(f"Checkpoint saved to {checkpoint_path}")
        
#         # Progress tracking
#         batch_time = time.time() - batch_start_time
#         elapsed = time.time() - start_time
# progress = (batch_idx - start_batch + 1)...
# remaining = (elapsed / progress) * (1...
        
#         log_message(f"Progress: {batch_idx+1}/{total_batches} batches ({progress*100:.1f}%)")
#         log_message(f"Batch time: {batch_time:.1f}s")
#         log_message(f"Elapsed: {timedelta(seconds=int(elapsed))}, Remaining: {timedelta(seconds=in...
        
#         # Reset TensorFlow session periodically to prevent memory growth
#         if (batch_idx + 1) % 500 == 0 and batch_idx > 0:
#             log_message("Performing session reset")
            
#             # Save model
#             temp_path = os.path.join(output_dir, "checkpoints", f"temp_reset_{batch_idx+1}.h5")
#             model.save(temp_path)
            
#             # Clear session
#             tf.keras.backend.clear_session()
#             gc.collect()
            
#             # Reload model
#             model = tf.keras.models.load_model(temp_path)
#             log_message("Session reset complete")
    
#     # Save final model
#     final_model_path = os.path.join(output_dir, "final_model.h5")
#     model.save(final_model_path)
#     log_message(f"Training complete. Final model saved to {final_model_path}")
    
#     # Return best model if available
#     if best_model_path and os.path.exists(best_model_path):
#         log_message(f"Loading best model from {best_model_path}")
#         model = tf.keras.models.load_model(best_model_path)
#         return model, best_model_path
    
#     return model, final_model_path

# def optimized_training(model, X_file, y_file, train_indices, val_indices, test_indices,
#                      output_dir, batch_size=256, class_weight=None, start_batch=0):
#     """
#     Optimized training function focused solely on improving processing efficiency
#     without changing the dataset or model architecture.
#     """
#     import os
#     import numpy as np
#     import tensorflow as tf
#     import gc
#     import time
#     import psutil
#     from datetime import datetime, timedelta
    
#     # Create output directories
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
#     # Load data in memory-mapped mode
#     X_mmap = np.load(X_file, mmap_mode='r')
#     y_mmap = np.load(y_file, mmap_mode='r')
    
#     # Prepare validation data
#     val_limit = min(1000, len(val_indices))
#     val_indices_subset = val_indices[:val_limit]
#     val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
#     val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    
#     # Calculate total batches
# total_batches = len(train_indices) // batch_size + (1...
#     print(f"Training on all {total_batches} batches")
    
#     # Load from checkpoint if resuming
#     if start_batch > 0:
#         checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{start_batch}.h5")
#         if os.path.exists(checkpoint_path):
#             print(f"Loading model from checkpoint {checkpoint_path}")
#             model = tf.keras.models.load_model(checkpoint_path)
    
#     # Training tracking variables
#     best_val_auc = 0
#     best_model_path = None
#     start_time = time.time()
    
#     # Use smaller individual batch sizes to avoid memory issues
#     effective_batch_size = 64  # Keep this smaller for stability
    
#     # Initial validation
#     print("Performing initial validation")
#     val_metrics = model.evaluate(val_X, val_y, verbose=0)
#     val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
# metrics_str = " - ".join([f"{k}: {v:.4f}" for...
#     print(f"Initial validation: {metrics_str}")
#     best_val_auc = val_metrics_dict.get('auc', 0)
    
#     # Process each batch
#     for batch_idx in range(start_batch, total_batches):
#         batch_start_time = time.time()
        
#         # Calculate batch indices
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
#         batch_indices = train_indices[start_idx:end_idx]
        
#         # Load batch data more efficiently
#         batch_X = []
#         batch_y = []
        
#         for i in range(0, len(batch_indices), 64):
#             end_i = min(i + 64, len(batch_indices))
#             indices = batch_indices[i:end_i]
#             X_chunk = np.array([X_mmap[idx] for idx in indices])
#             y_chunk = np.array([y_mmap[idx] for idx in indices])
#             batch_X.append(X_chunk)
#             batch_y.append(y_chunk)
        
#         batch_X = np.concatenate(batch_X)
#         batch_y = np.concatenate(batch_y)
        
#         # Train on this batch (using the smaller effective batch size)
#         try:
#             model.fit(
#                 batch_X, batch_y,
#                 epochs=1,
#                 batch_size=effective_batch_size,
#                 class_weight=class_weight,
#                 verbose=1
#             )
#         except Exception as e:
#             print(f"Error training batch: {e}")
#             # Try with even smaller batch size if there's an error
#             try:
#                 model.fit(
#                     batch_X, batch_y,
#                     epochs=1,
#                     batch_size=32,
#                     class_weight=class_weight,
#                     verbose=1
#                 )
#             except:
#                 continue
        
#         # Clean up batch data
#         del batch_X, batch_y
#         gc.collect()
        
#         # Validate and save periodically
#         if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
#             # Evaluate on validation set
#             val_metrics = model.evaluate(val_X, val_y, verbose=0)
#             val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
            
#             print(f"Validation at batch {batch_idx+1}/{total_batches}:")
#             print(" - ".join([f"{k}: {v:.4f}" for k, v in val_metrics_dict.items()]))
            
#             # Save if better
#             val_auc = val_metrics_dict.get('auc', 0)
#             if val_auc > best_val_auc:
#                 best_val_auc = val_auc
#                 best_model_path = os.path.join(output_dir, "checkpoints", f"best_model_{batch_idx+...
#                 model.save(best_model_path)
#                 print(f"New best model: AUC = {val_auc:.4f}")
            
#             # Save checkpoint
#             checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{batch_idx+1}....
#             model.save(checkpoint_path)
        
#         # Progress tracking
#         batch_time = time.time() - batch_start_time
#         elapsed = time.time() - start_time
# progress = (batch_idx - start_batch + 1)...
# remaining = (elapsed / progress) * (1...
        
#         print(f"Progress: {batch_idx+1}/{total_batches} batches ({progress*100:.1f}%)")
#         print(f"Batch time: {batch_time:.1f}s")
#         print(f"Elapsed: {timedelta(seconds=int(elapsed))}, Remaining: {timedelta(seconds=int(rema...
        
#         # Reset session periodically
#         if (batch_idx + 1) % 500 == 0:
#             print("Resetting TensorFlow session")
#             temp_path = os.path.join(output_dir, "checkpoints", f"temp_{batch_idx+1}.h5")
#             model.save(temp_path)
#             tf.keras.backend.clear_session()
#             gc.collect()
#             model = tf.keras.models.load_model(temp_path)
    
#     # Save final model
#     final_model_path = os.path.join(output_dir, "final_model.h5")
#     model.save(final_model_path)
    
#     # Return best model if available
#     if best_model_path and os.path.exists(best_model_path):
#         model = tf.keras.models.load_model(best_model_path)
    
#     return model, final_model_path

# def fixed_training_with_full_evaluation_verbose(model, X_file, y_file, train_indices, val_indices,...
#                                        output_dir, batch_size=256, class_weight=None, start_batch=...
#                                        metadata_file=None):
#     """
#     Comprehensive training function with stability fixes, integrated evaluation,
#     and extremely verbose forced logging.
#     """
#     import os
#     import numpy as np
#     import tensorflow as tf
#     import gc
#     import time
#     import psutil
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import pandas as pd
#     from sklearn.metrics import (
#         classification_report, confusion_matrix, roc_curve, 
#         auc, precision_recall_curve, average_precision_score, roc_auc_score
#     )
#     from datetime import datetime, timedelta
#     import sys
    
#     # Force print function to ensure output is displayed immediately
#     def force_print(message, also_log=True):
#         print(message, flush=True)
#         if also_log:
#             with open(os.path.join(output_dir, "verbose_log.txt"), "a") as f:
#                 timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 f.write(f"[{timestamp}] {message}\n")
    
#     force_print("="*80)
#     force_print("STARTING EMERGENCY VERBOSE TRAINING FUNCTION")
#     force_print("="*80)
    
#     # Create output directories
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
#     force_print("Output directories created successfully")
    
#     # Load data
#     force_print(f"Attempting to load data from {X_file} and {y_file}")
#     X_mmap = np.load(X_file, mmap_mode='r')
#     force_print(f"X data loaded: shape info available: {hasattr(X_mmap, 'shape')}")
#     y_mmap = np.load(y_file, mmap_mode='r')
#     force_print(f"Y data loaded: shape info available: {hasattr(y_mmap, 'shape')}")
    
#     # Prepare validation data - limit to a reasonable size
#     val_limit = min(5000, len(val_indices))
#     val_indices_subset = val_indices[:val_limit]
#     val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
#     val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
#     force_print(f"Validation data loaded: X shape={val_X.shape}, y shape={val_y.shape}")
    
#     # FIX: Recompile model with stability improvements
#     if start_batch == 0:
#         force_print("Recompiling model with stability improvements")
        
#         # Recompile with improved numeric stability
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(
#                 learning_rate=0.001,
#                 epsilon=1e-7,  # Increased epsilon for numeric stability
#                 clipnorm=1.0   # Gradient clipping for stability
#             ),
#             loss='binary_crossentropy',
#             metrics=[
#                 'accuracy',
# # FIX: Use numerically stable AUC implementation...
#                 tf.keras.metrics.AUC(name='auc', num_thresholds=200, from_logits=False),
#                 tf.keras.metrics.Precision(name='precision'),
#                 tf.keras.metrics.Recall(name='recall')
#             ]
#         )
#         force_print("Model recompilation complete")
    
#     # Calculate total batches - use FULL training set
# total_batches = len(train_indices) // batch_size + (1...
#     force_print(f"Training on all {total_batches} batches of full training set")
    
#     # Load checkpoint if resuming
#     if start_batch > 0:
#         checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{start_batch}.h5")
#         if os.path.exists(checkpoint_path):
#             force_print(f"Loading model from checkpoint {checkpoint_path}")
#             model = tf.keras.models.load_model(checkpoint_path)
#             force_print("Model loaded successfully")
#         else:
#             force_print(f"Checkpoint file not found: {checkpoint_path}")
    
#     # Training variables
#     best_val_auc = 0
#     best_model_path = None
#     start_time = time.time()
#     validation_frequency = 50  # Validate every 50 batches
    
#     # Main training loop - process FULL training set
#     for batch_idx in range(start_batch, total_batches):
#         batch_start_time = time.time()
#         force_print(f"="*50)
#         force_print(f"STARTING BATCH {batch_idx+1}/{total_batches}")
        
#         # Calculate batch indices
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
#         batch_indices = train_indices[start_idx:end_idx]
#         force_print(f"Batch indices: {start_idx} to {end_idx} (total: {len(batch_indices)})")
        
#         # Load batch data efficiently
#         force_print("Loading batch data...")
#         batch_X = []
#         batch_y = []
        
#         for i in range(0, len(batch_indices), 64):
#             end_i = min(i + 64, len(batch_indices))
#             indices = batch_indices[i:end_i]
#             force_print(f"  Loading mini-chunk {i}-{end_i} ({len(indices)} samples)")
#             X_chunk = np.array([X_mmap[idx] for idx in indices])
#             y_chunk = np.array([y_mmap[idx] for idx in indices])
#             batch_X.append(X_chunk)
#             batch_y.append(y_chunk)
#             force_print(f"  Mini-chunk loaded: X shape={X_chunk.shape}, y shape={y_chunk.shape}")
        
#         force_print("Concatenating batch chunks...")
#         batch_X = np.concatenate(batch_X)
#         batch_y = np.concatenate(batch_y)
#         force_print(f"Batch data loaded: X shape={batch_X.shape}, y shape={batch_y.shape}")
        
#         # FIX: Check class balance in batch
#         pos_ratio = np.mean(batch_y)
#         force_print(f"Batch class balance: {pos_ratio:.4f} positive, {1-pos_ratio:.4f} negative")
        
#         # Train on batch
#         force_print("Starting training on batch...")
#         try:
#             # FIX: Use more stable mini-batch sizes
#             mini_batch_size = 32  # Smaller, more stable mini-batches
#             force_print(f"Training with mini-batch size: {mini_batch_size}")
            
#             history = model.fit(
#                 batch_X, batch_y,
#                 epochs=1,
#                 batch_size=mini_batch_size,
#                 class_weight=class_weight,
#                 verbose=2  # More verbose output from fit
#             )
            
#             force_print("Training complete for this batch")
#             force_print(f"Training metrics: {history.history}")
#         except Exception as e:
#             force_print(f"ERROR during training: {str(e)}")
#             import traceback
#             force_print(traceback.format_exc())
#             continue
        
#         # Clean up
#         del batch_X, batch_y
#         gc.collect()
        
#         # Validate periodically - start only after a few batches
#         if batch_idx >= 10 and (batch_idx + 1) % validation_frequency == 0:
#             force_print(f"Performing validation at batch {batch_idx+1}...")
#             try:
#                 val_metrics = model.evaluate(val_X, val_y, verbose=1)
#                 val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
                
# metrics_str = " - ".join([f"{k}: {v:.4f}" for...
#                 force_print(f"Validation metrics at batch {batch_idx+1}/{total_batches}:")
#                 force_print(f"  {metrics_str}")
                
#                 # Save if better
#                 val_auc = val_metrics_dict.get('auc', 0)
#                 if val_auc > best_val_auc:
#                     best_val_auc = val_auc
#                     best_model_path = os.path.join(output_dir, "checkpoints", f"best_model_{batch_...
#                     force_print(f"New best model: AUC = {val_auc:.4f}, saving to {best_model_path}...
#                     model.save(best_model_path)
#                     force_print("Best model saved successfully")
                
#                 # Save checkpoint
#                 checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_batch_{batch_idx...
#                 force_print(f"Saving checkpoint to {checkpoint_path}")
#                 model.save(checkpoint_path)
#                 force_print("Checkpoint saved successfully")
#             except Exception as e:
#                 force_print(f"ERROR during validation: {str(e)}")
#                 import traceback
#                 force_print(traceback.format_exc())
        
#         # Progress tracking
#         batch_time = time.time() - batch_start_time
#         elapsed = time.time() - start_time
# progress = (batch_idx - start_batch + 1)...
# remaining = (elapsed / progress) * (1...
        
#         force_print(f"PROGRESS UPDATE:")
#         force_print(f"  Batch {batch_idx+1}/{total_batches} completed ({progress*100:.1f}%)")
#         force_print(f"  Batch processing time: {batch_time:.1f}s")
#         force_print(f"  Elapsed: {timedelta(seconds=int(elapsed))}")
#         force_print(f"  Estimated remaining: {timedelta(seconds=int(remaining))}")
#         force_print(f"  Memory usage: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 102...
        
#         # Reset session periodically
#         if (batch_idx + 1) % 500 == 0:
#             force_print("Resetting TensorFlow session to free memory")
#             temp_path = os.path.join(output_dir, "checkpoints", f"temp_{batch_idx+1}.h5")
#             force_print(f"Saving temporary model to {temp_path}")
#             model.save(temp_path)
#             force_print("Clearing session")
#             tf.keras.backend.clear_session()
#             gc.collect()
#             force_print("Loading model back")
#             model = tf.keras.models.load_model(temp_path)
#             force_print("Session reset complete")
        
#         force_print(f"BATCH {batch_idx+1} COMPLETED SUCCESSFULLY")
#         force_print("="*50)
    
#     # Save final model
#     force_print("Training complete! Saving final model...")
#     final_model_path = os.path.join(output_dir, "final_model.h5")
#     model.save(final_model_path)
#     force_print(f"Final model saved to {final_model_path}")
    
#     # Rest of the function continues with evaluation...
#     force_print("\n===== STARTING FULL EVALUATION ON ENTIRE TEST SET =====")
#     if best_model_path and os.path.exists(best_model_path):
#         force_print(f"Loading best model from {best_model_path}")
#         model = tf.keras.models.load_model(best_model_path)
#         force_print("Best model loaded successfully")
#     else:
#         force_print("Using final model for evaluation")
    
#     # Prepare test set evaluation
#     force_print(f"Evaluating model on ALL {len(test_indices)} test samples")
    
#     # Process test data in batches
#     batch_size = 1000
#     num_batches = int(np.ceil(len(test_indices) / batch_size))
    
#     all_preds = []
#     all_true = []
#     all_meta = []
    
#     # Load metadata if available
#     metadata = None
#     if metadata_file:
#         import pickle
#         try:
#             force_print(f"Loading metadata from {metadata_file}")
#             with open(metadata_file, "rb") as f:
#                 metadata = pickle.load(f)
#             force_print(f"Metadata loaded successfully - contains data for {len(metadata)} samples...
#         except Exception as e:
#             force_print(f"ERROR loading metadata: {str(e)}")
    
#     # Process test data - use COMPLETE test set
#     force_print(f"Processing test data in {num_batches} batches...")
#     for batch_idx in range(num_batches):
#         force_print(f"Processing test batch {batch_idx+1}/{num_batches}")
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(test_indices))
#         batch_indices = test_indices[start_idx:end_idx]
#         force_print(f"  Test batch indices: {start_idx} to {end_idx} (total: {len(batch_indices)})...
        
#         # Load batch data
#         force_print(f"  Loading test batch data...")
#         batch_X = np.array([X_mmap[idx] for idx in batch_indices])
#         batch_y = np.array([y_mmap[idx] for idx in batch_indices])
#         force_print(f"  Test batch loaded: X shape={batch_X.shape}, y shape={batch_y.shape}")
        
#         # Store metadata if available
#         if metadata:
#             try:
#                 force_print(f"  Processing metadata for this batch...")
#                 batch_meta = [metadata[idx] for idx in batch_indices]
#                 all_meta.extend(batch_meta)
#                 force_print(f"  Metadata processed: {len(batch_meta)} records added")
#             except Exception as e:
#                 force_print(f"  ERROR processing metadata: {str(e)}")
        
#         # Predict
#         force_print(f"  Making predictions...")
#         batch_preds = model.predict(batch_X, verbose=1)
#         force_print(f"  Predictions complete: shape={batch_preds.shape}")
        
#         # Store results
#         all_preds.extend(batch_preds.flatten())
#         all_true.extend(batch_y)
#         force_print(f"  Results stored: total predictions so far: {len(all_preds)}")
        
#         # Clean up
#         del batch_X, batch_y, batch_preds
#         gc.collect()
#         force_print(f"  Batch cleanup complete")
    
#     force_print("All test batches processed")
    
#     # Convert to numpy arrays
#     force_print("Converting results to numpy arrays...")
#     all_preds = np.array(all_preds)
#     all_true = np.array(all_true)
#     force_print(f"Arrays created: predictions={all_preds.shape}, ground truth={all_true.shape}")
    
#     # Create binary predictions
#     force_print("Creating binary predictions...")
#     all_preds_binary = (all_preds > 0.5).astype(int)
#     force_print(f"Binary predictions created: shape={all_preds_binary.shape}")
    
#     # Calculate metrics
#     force_print("Calculating evaluation metrics...")
#     report = classification_report(all_true, all_preds_binary, output_dict=True)
#     report_str = classification_report(all_true, all_preds_binary)
#     conf_matrix = confusion_matrix(all_true, all_preds_binary)
    
#     # Calculate ROC curve and AUC
#     force_print("Calculating ROC curve and AUC...")
#     fpr, tpr, _ = roc_curve(all_true, all_preds)
#     roc_auc = auc(fpr, tpr)
    
#     # Calculate Precision-Recall curve
#     force_print("Calculating Precision-Recall curve...")
#     precision, recall, _ = precision_recall_curve(all_true, all_preds)
#     avg_precision = average_precision_score(all_true, all_preds)
    
#     force_print("\nClassification Report:")
#     force_print(report_str)
    
#     force_print("\nConfusion Matrix:")
#     force_print(conf_matrix)
    
#     force_print(f"\nROC AUC: {roc_auc:.4f}")
#     force_print(f"Average Precision: {avg_precision:.4f}")
    
#     # Save results to file
#     force_print("Saving evaluation results to file...")
#     with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
#         f.write("Classification Report:\n")
#         f.write(report_str)
#         f.write("\n\nConfusion Matrix:\n")
#         f.write(str(conf_matrix))
#         f.write(f"\n\nROC AUC: {roc_auc:.4f}")
#         f.write(f"\nAverage Precision: {avg_precision:.4f}")
#     force_print("Evaluation results saved")
    
#     # Create visualizations
#     force_print("Creating visualizations...")
#     vis_dir = os.path.join(output_dir, "visualizations")
#     os.makedirs(vis_dir, exist_ok=True)
    
#     # 1. Confusion Matrix
#     force_print("Creating confusion matrix visualization...")
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#                xticklabels=['Negative', 'Positive'],
#                yticklabels=['Negative', 'Positive'])
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.title('Confusion Matrix')
#     plt.tight_layout()
#     plt.savefig(os.path.join(vis_dir, "confusion_matrix.png"), dpi=300)
#     plt.close()
#     force_print("Confusion matrix visualization saved")
    
#     # 2. ROC Curve
#     force_print("Creating ROC curve visualization...")
#     plt.figure(figsize=(10, 8))
#     plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     plt.grid(alpha=0.3)
#     plt.savefig(os.path.join(vis_dir, "roc_curve.png"), dpi=300)
#     plt.close()
#     force_print("ROC curve visualization saved")
    
#     # 3. Precision-Recall Curve
#     force_print("Creating Precision-Recall curve visualization...")
#     plt.figure(figsize=(10, 8))
#     plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend(loc="upper right")
#     plt.grid(alpha=0.3)
#     plt.savefig(os.path.join(vis_dir, "precision_recall_curve.png"), dpi=300)
#     plt.close()
#     force_print("Precision-Recall curve visualization saved")
    
#     # 4. Score distribution
#     force_print("Creating score distribution visualization...")
#     plt.figure(figsize=(12, 6))
#     sns.histplot(all_preds[all_true == 0], bins=50, alpha=0.5, label='Negative Class', color='blue...
#     sns.histplot(all_preds[all_true == 1], bins=50, alpha=0.5, label='Positive Class', color='red'...
#     plt.xlabel('Prediction Score')
#     plt.ylabel('Count')
#     plt.title('Distribution of Prediction Scores by Class')
#     plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.savefig(os.path.join(vis_dir, "score_distribution.png"), dpi=300)
#     plt.close()
#     force_print("Score distribution visualization saved")
    
#     # Save predictions
#     force_print("Saving predictions...")
#     if len(all_meta) > 0:
#         force_print("Creating results DataFrame with metadata...")
#         # Create results DataFrame with metadata
#         results_data = []
#         for i in range(len(all_preds)):
#             if i < len(all_meta):
#                 result = {
#                     'prediction': all_preds[i],
#                     'true_label': all_true[i],
#                     'predicted_label': all_preds_binary[i],
#                     'correct': all_preds_binary[i] == all_true[i]
#                 }
                
#                 # Add metadata
#                 meta = all_meta[i]
#                 for key, value in meta.items():
#                     if isinstance(value, (np.integer, np.floating)):
#                         value = float(value)
#                     result[key] = value
                
#                 results_data.append(result)
        
#         # Create DataFrame
#         results_df = pd.DataFrame(results_data)
        
#         # Save to CSV
#         force_print("Saving predictions with metadata to CSV...")
#         results_df.to_csv(os.path.join(output_dir, "test_predictions_with_metadata.csv"), index=Fa...
#         force_print("Predictions with metadata saved")
#     else:
#         # Save simpler predictions
#         force_print("Saving basic predictions...")
#         np.save(os.path.join(output_dir, "test_predictions.npy"), all_preds)
#         np.save(os.path.join(output_dir, "test_true_labels.npy"), all_true)
#         force_print("Basic predictions saved")
    
#     # Feature importance
#     force_print("\nPerforming feature importance analysis...")
    
#     # Process feature importance in batches
#     feature_importances = np.zeros(X_mmap[0].shape[1])
    
#     force_print(f"Feature shape: {X_mmap[0].shape}")
#     feature_names = ['Temperature', 'Temperature Gradient', 'Depth']
#     if X_mmap[0].shape[1] > len(feature_names):
#         for i in range(len(feature_names), X_mmap[0].shape[1]):
#             feature_names.append(f'Feature {i+1}')
#     feature_names = feature_names[:X_mmap[0].shape[1]]
#     force_print(f"Feature names: {feature_names}")
    
#     # Use one small batch for fast feature importance
#     force_print("Calculating quick feature importance on a small sample...")
#     test_subset = test_indices[:1000]  # Just use first 1000 samples
#     force_print(f"Loading {len(test_subset)} samples for feature importance...")
#     X_test_small = np.array([X_mmap[idx] for idx in test_subset])
#     y_test_small = np.array([y_mmap[idx] for idx in test_subset])
#     force_print(f"Test sample loaded: X shape={X_test_small.shape}, y shape={y_test_small.shape}")
    
#     # Get baseline
#     force_print("Getting baseline predictions...")
#     baseline_preds = model.predict(X_test_small, verbose=1)
#     baseline_auc = roc_auc_score(y_test_small, baseline_preds)
#     force_print(f"Baseline AUC: {baseline_auc:.4f}")
    
#     # Test each feature
#     for feature_idx in range(len(feature_names)):
#         force_print(f"Testing importance of feature '{feature_names[feature_idx]}'...")
#         # Permute feature
#         X_permuted = X_test_small.copy()
#         for time_step in range(X_test_small.shape[1]):
#             X_permuted[:, time_step, feature_idx] = np.random.permutation(X_test_small[:, time_ste...
        
#         # Get performance drop
#         force_print(f"  Getting predictions with permuted feature...")
#         perm_preds = model.predict(X_permuted, verbose=1)
#         perm_auc = roc_auc_score(y_test_small, perm_preds)
#         importance = baseline_auc - perm_auc
        
#         feature_importances[feature_idx] = importance
#         force_print(f"  Feature '{feature_names[feature_idx]}' importance: {importance:.4f}")
        
#         # Clean up
#         del X_permuted, perm_preds
#         gc.collect()
    
#     # Create feature importance dataframe
#     force_print("Creating feature importance dataframe...")
#     importance_df = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': feature_importances
#     })
#     importance_df = importance_df.sort_values('Importance', ascending=False)
#     force_print("Feature importance rankings:")
#     for idx, row in importance_df.iterrows():
#         force_print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
#     # Plot feature importance
#     force_print("Creating feature importance visualization...")
#     plt.figure(figsize=(10, 6))
#     plt.bar(importance_df['Feature'], importance_df['Importance'])
#     plt.xlabel('Feature')
#     plt.ylabel('Importance (decrease in AUC)')
#     plt.title('Feature Importance')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.savefig(os.path.join(vis_dir, "feature_importance.png"), dpi=300)
#     plt.close()
#     force_print("Feature importance visualization saved")
    
#     # Save importance results
#     force_print("Saving feature importance results...")
#     importance_df.to_csv(os.path.join(vis_dir, "feature_importance.csv"), index=False)
#     force_print("Feature importance results saved")
    
#     force_print("\n" + "="*30)
#     force_print("EVALUATION COMPLETE!")
#     force_print("="*30)
#     force_print(f"All results and visualizations saved to {output_dir}")
    
#     # Return results
#     final_results = {
#         'accuracy': report['accuracy'],
#         'precision': report['1']['precision'],
#         'recall': report['1']['recall'],
#         'f1': report['1']['f1-score'],
#         'auc': roc_auc,
#         'avg_precision': avg_precision
#     }
    
#     force_print(f"Final results: {final_results}")
#     force_print("FUNCTION EXECUTION COMPLETE")
    
#     return model, final_results

# def ultra_fast_training(model, X_file, y_file, train_indices, val_indices, test_indices,
#                        output_dir, batch_size=512, class_weight=None, start_batch=0):
#     """
#     Ultrafast training function with minimal overhead and maximum efficiency
#     """
#     import os
#     import numpy as np
#     import tensorflow as tf
#     import gc
#     import time
#     import psutil
#     from datetime import datetime, timedelta
    
#     # Disable verbose logging
#     tf.get_logger().setLevel('ERROR')
    
#     # Create output directories
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
#     # Memory-mapped data access
#     X_mmap = np.load(X_file, mmap_mode='r')
#     y_mmap = np.load(y_file, mmap_mode='r')
    
#     # Efficient validation data loading
#     val_limit = min(2000, len(val_indices))
#     val_indices_subset = val_indices[:val_limit]
#     val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
#     val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    
#     # Efficient model compilation
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(
#             learning_rate=0.001, 
#             clipnorm=1.0  # Gradient clipping
#         ),
#         loss='binary_crossentropy',
#         metrics=[
#             'accuracy',
#             tf.keras.metrics.AUC(name='auc'),
#             tf.keras.metrics.Precision(name='precision'),
#             tf.keras.metrics.Recall(name='recall')
#         ]
#     )
    
#     # Calculate total batches
# total_batches = len(train_indices) // batch_size + (1...
    
#     # Tracking variables
#     best_val_auc = 0
#     best_model_path = None
#     start_time = time.time()
    
#     # Efficient batch processing
#     for batch_idx in range(start_batch, total_batches):
#         # Calculate batch indices
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
#         batch_indices = train_indices[start_idx:end_idx]
        
#         # Direct batch loading - avoid chunking
#         try:
#             batch_X = np.array([X_mmap[idx] for idx in batch_indices])
#             batch_y = np.array([y_mmap[idx] for idx in batch_indices])
#         except Exception as e:
#             print(f"Error loading batch {batch_idx}: {e}")
#             continue
        
#         # Train on batch
#         try:
#             history = model.fit(
#                 batch_X, batch_y,
#                 epochs=1,
#                 batch_size=min(len(batch_X), 128),  # Adaptive batch size
#                 class_weight=class_weight,
#                 verbose=0  # Minimal logging
#             )
#         except Exception as e:
#             print(f"Training error on batch {batch_idx}: {e}")
#             continue
        
#         # Periodic validation and checkpointing
#         if batch_idx % 100 == 0:
#             try:
#                 # Validate
#                 val_metrics = model.evaluate(val_X, val_y, verbose=0)
#                 val_metrics_dict = dict(zip(model.metrics_names, val_metrics))
#                 val_auc = val_metrics_dict.get('auc', 0)
                
#                 # Save best model
#                 if val_auc > best_val_auc:
#                     best_val_auc = val_auc
#                     best_model_path = os.path.join(output_dir, f"best_model_batch_{batch_idx}.h5")
#                     model.save(best_model_path)
                
#                 # Save periodic checkpoint
#                 checkpoint_path = os.path.join(output_dir, f"checkpoint_batch_{batch_idx}.h5")
#                 model.save(checkpoint_path)
#             except Exception as e:
#                 print(f"Validation error at batch {batch_idx}: {e}")
        
#         # Memory and progress management
#         del batch_X, batch_y
#         gc.collect()
        
#         # Progress tracking (minimal output)
#         if batch_idx % 50 == 0:
#             elapsed = time.time() - start_time
# progress = (batch_idx - start_batch + 1)...
# remaining = (elapsed / progress) * (1...
#             print(f"Batch {batch_idx+1}/{total_batches} - Progress: {progress*100:.1f}% - Elapsed:...
    
#     # Save final model
#     final_model_path = os.path.join(output_dir, "final_model.h5")
#     model.save(final_model_path)
    
#     # Return best or final model
#     if best_model_path and os.path.exists(best_model_path):
#         return tf.keras.models.load_model(best_model_path), best_model_path
    
#     return model, final_model_path

# def compile_model_with_device_handling(model):
#     """Compile model with explicit device handling"""
#     with tf.device('/GPU:0'):  # Force GPU usage
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(
#                 learning_rate=0.001, 
#                 clipnorm=1.0,
#                 amsgrad=True
#             ),
#             loss='binary_crossentropy',
#             metrics=[
#                 'accuracy',
#                 tf.keras.metrics.AUC(name='auc'),
#             ]
#         )
#     return model

# def hyper_fast_training(model, X_file, y_file, train_indices, val_indices, test_indices,
#                        output_dir, batch_size=1024, class_weight=None, start_batch=0):
#     """
# Hyper-optimized training function with absolute minimal overhead...
#     """
#     import os
#     import numpy as np
#     import tensorflow as tf
#     import gc
#     import time
#     from datetime import datetime, timedelta
    
#     # Extreme logging suppression
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#     tf.get_logger().setLevel('ERROR')
    
#     # Create output directories
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
#     # Logging function with minimal overhead
#     def quick_log(message):
#         print(message, flush=True)
#         with open(os.path.join(output_dir, "training_log.txt"), "a") as f:
#             f.write(f"{datetime.now()}: {message}\n")
    
#     # Extremely efficient data loading
#     try:
#         X_mmap = np.load(X_file, mmap_mode='r')
#         y_mmap = np.load(y_file, mmap_mode='r')
#     except Exception as e:
#         quick_log(f"CRITICAL DATA LOADING ERROR: {e}")
#         return model, None
    
#     # Efficient validation subset
#     val_indices_subset = val_indices[:min(2000, len(val_indices))]
    
#     # Use explicit device context
#     with tf.device('/GPU:0'):
#         val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
#         val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    
#     # Total batches calculation
# total_batches = len(train_indices) // batch_size + (1...
#     quick_log(f"Total batches: {total_batches}, Batch size: {batch_size}")
    
#     # Performance tracking
#     start_time = time.time()
#     best_val_auc = 0
#     best_model_path = None
    
#     # Main training loop with extreme efficiency
#     for batch_idx in range(start_batch, total_batches):
#         # Batch indices
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
#         batch_indices = train_indices[start_idx:end_idx]
        
#         # Direct, fast data loading with device context
#         with tf.device('/GPU:0'):
#             try:
#                 batch_X = np.array([X_mmap[idx] for idx in batch_indices])
#                 batch_y = np.array([y_mmap[idx] for idx in batch_indices])
#             except Exception as e:
#                 quick_log(f"Error loading batch {batch_idx}: {e}")
#                 continue
            
#             # Train with minimal overhead
#             try:
#                 history = model.fit(
#                     batch_X, batch_y,
#                     epochs=1,
#                     batch_size=min(len(batch_X), 256),  # Adaptive batch size
#                     class_weight=class_weight,
#                     verbose=0  # Absolute minimal logging
#                 )
#             except Exception as e:
#                 quick_log(f"Training error on batch {batch_idx}: {e}")
#                 continue
        
#         # Periodic validation (every 100 batches)
#         if batch_idx % 100 == 0:
#             try:
#                 with tf.device('/GPU:0'):
#                     val_metrics = model.evaluate(val_X, val_y, verbose=0)
#                     val_auc = val_metrics[model.metrics_names.index('auc')]
                
#                 # Model saving
#                 if val_auc > best_val_auc:
#                     best_val_auc = val_auc
#                     best_model_path = os.path.join(output_dir, f"best_model_batch_{batch_idx}.h5")
#                     model.save(best_model_path)
#             except Exception as e:
#                 quick_log(f"Validation error at batch {batch_idx}: {e}")
        
#         # Progress reporting (minimal)
#         if batch_idx % 50 == 0:
#             elapsed = time.time() - start_time
# progress = (batch_idx - start_batch + 1)...
# remaining = (elapsed / progress) * (1...
#             quick_log(f"Batch {batch_idx+1}/{total_batches} - Progress: {progress*100:.1f}% - Elap...
        
#         # Aggressive memory management
#         del batch_X, batch_y
#         gc.collect()
    
#     # Final model save
#     final_model_path = os.path.join(output_dir, "final_model.h5")
#     model.save(final_model_path)
    
#     # Return best or final model
#     if best_model_path and os.path.exists(best_model_path):
#         return tf.keras.models.load_model(best_model_path), best_model_path
    
#     return model, final_model_path

# def ultra_robust_training(model, X_file, y_file, train_indices, val_indices, test_indices,
#                           output_dir, batch_size=512, max_epochs=100, class_weight=None):
#     """
#     ULTRA-ROBUST TRAINING FUNCTION WITH MAXIMUM ERROR RECOVERY AND DIAGNOSTICS
#     """
#     import os
#     import numpy as np
#     import tensorflow as tf
#     import gc
#     import time
#     import psutil
#     import json
#     from datetime import datetime, timedelta
#     import traceback
#     import sys
    
#     # Comprehensive logging setup
#     log_file = os.path.join(output_dir, "ultra_robust_training_log.txt")
    
#     def robust_log(message, level="INFO"):
#         """
#         Logging function with multiple output channels
#         """
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         log_message = f"[{timestamp}] [{level}] {message}"
        
#         # Print to console
#         print(log_message, flush=True)
        
#         # Write to log file
#         try:
#             with open(log_file, "a") as f:
#                 f.write(log_message + "\n")
#         except Exception as log_error:
#             print(f"ERROR LOGGING: {log_error}")
    
#     # Critical error handling wrapper
#     def critical_error_handler(func):
#         def wrapper(*args, **kwargs):
#             try:
#                 return func(*args, **kwargs)
#             except Exception as e:
#                 robust_log(f"CRITICAL ERROR IN {func.__name__}: {e}", "CRITICAL")
#                 robust_log(traceback.format_exc(), "CRITICAL")
                
#                 # Create error snapshot
#                 error_snapshot = {
#                     "timestamp": datetime.now().isoformat(),
#                     "function": func.__name__,
#                     "error_type": type(e).__name__,
#                     "error_message": str(e),
#                     "traceback": traceback.format_exc()
#                 }
                
#                 # Save error snapshot
#                 try:
#                     with open(os.path.join(output_dir, f"error_snapshot_{int(time.time())}.json"),...
#                         json.dump(error_snapshot, f, indent=4)
#                 except:
#                     pass
                
#                 # Re-raise the exception
#                 raise
#         return wrapper
    
#     # Comprehensive memory and device management
#     @critical_error_handler
#     def configure_tensorflow_environment():
#         """
#         Ultra-detailed TensorFlow and GPU configuration
#         """
#         # Disable verbose logging
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#         tf.get_logger().setLevel('ERROR')
        
#         # Identify and configure devices
#         physical_devices = tf.config.list_physical_devices()
#         robust_log(f"Available Physical Devices: {physical_devices}")
        
#         gpu_devices = tf.config.list_physical_devices('GPU')
#         for device in gpu_devices:
#             try:
#                 # Enable memory growth
#                 tf.config.experimental.set_memory_growth(device, True)
#                 robust_log(f"Memory growth enabled for {device}")
                
#                 # Optional: Set virtual device
#                 tf.config.experimental.set_virtual_device_configuration(
#                     device,
#                     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4)]  # ...
#                 )
#             except Exception as e:
#                 robust_log(f"Could not configure {device}: {e}", "WARNING")
        
#         # Enable soft device placement
#         tf.config.set_soft_device_placement(True)
    
#     # Memory-safe data loading
#     @critical_error_handler
#     def load_data_safely(X_file, y_file, train_indices, val_indices):
#         """
#         Safe data loading with comprehensive error handling
#         """
#         # Memory-mapped data access
#         X_mmap = np.load(X_file, mmap_mode='r')
#         y_mmap = np.load(y_file, mmap_mode='r')
        
#         # Safe validation subset selection
#         val_limit = min(2000, len(val_indices))
#         val_indices_subset = val_indices[:val_limit]
        
#         # Load validation data
#         val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
#         val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
        
#         robust_log(f"Validation data shape: X {val_X.shape}, y {val_y.shape}")
        
#         return X_mmap, y_mmap, val_X, val_y
    
#     # Comprehensive training loop
#     @critical_error_handler
#     def execute_training_loop(model, X_mmap, y_mmap, train_indices, val_X, val_y, output_dir):
#         """
#         Advanced training loop with maximum recovery capabilities
#         """
#         # Training configuration
# total_batches = len(train_indices) // batch_size + (1...
#         robust_log(f"Total batches: {total_batches}, Batch size: {batch_size}")
        
#         # Performance tracking
#         best_val_auc = 0
#         best_model_path = None
#         training_history = []
#         start_time = time.time()
        
#         # Callbacks for advanced monitoring
#         callbacks = [
#             tf.keras.callbacks.EarlyStopping(
#                 monitor='val_auc', 
#                 patience=15, 
#                 restore_best_weights=True
#             ),
#             tf.keras.callbacks.ReduceLROnPlateau(
#                 monitor='val_auc', 
#                 factor=0.5, 
#                 patience=7
#             )
#         ]
        
#         # Multi-epoch training strategy
#         for epoch in range(max_epochs):
#             robust_log(f"Starting Epoch {epoch + 1}/{max_epochs}")
            
#             # Shuffle training indices to prevent overfitting
#             np.random.shuffle(train_indices)
            
#             # Process batches
#             epoch_losses = []
#             for batch_idx in range(total_batches):
#                 start_idx = batch_idx * batch_size
#                 end_idx = min((batch_idx + 1) * batch_size, len(train_indices))
#                 batch_indices = train_indices[start_idx:end_idx]
                
#                 # Safe batch loading
#                 try:
#                     batch_X = np.array([X_mmap[idx] for idx in batch_indices])
#                     batch_y = np.array([y_mmap[idx] for idx in batch_indices])
#                 except Exception as batch_load_error:
#                     robust_log(f"Batch loading error: {batch_load_error}", "WARNING")
#                     continue
                
#                 # Train on batch
#                 try:
#                     history = model.fit(
#                         batch_X, batch_y,
#                         validation_data=(val_X, val_y),
#                         epochs=1,
#                         batch_size=min(len(batch_X), 128),
#                         verbose=0,
#                         callbacks=callbacks
#                     )
                    
#                     # Track batch performance
#                     batch_loss = history.history.get('loss', [np.nan])[0]
#                     epoch_losses.append(batch_loss)
                    
#                     # Periodic validation and checkpointing
#                     if batch_idx % 50 == 0:
#                         val_metrics = model.evaluate(val_X, val_y, verbose=0)
#                         val_auc = val_metrics[model.metrics_names.index('auc')]
                        
#                         if val_auc > best_val_auc:
#                             best_val_auc = val_auc
#                             best_model_path = os.path.join(output_dir, f"best_model_epoch_{epoch+1...
#                             model.save(best_model_path)
#                             robust_log(f"New best model saved with AUC: {best_val_auc}")
                
#                 except Exception as train_error:
#                     robust_log(f"Training error in batch {batch_idx}: {train_error}", "ERROR")
#                     continue
                
#                 # Memory management
#                 del batch_X, batch_y
#                 gc.collect()
            
#             # Epoch summary
#             mean_loss = np.mean(epoch_losses)
#             robust_log(f"Epoch {epoch + 1} Summary: Mean Loss = {mean_loss}")
#             training_history.append({
#                 "epoch": epoch + 1,
#                 "mean_loss": mean_loss,
#                 "best_val_auc": best_val_auc
#             })
            
#             # Early stopping condition
#             if len(training_history) > 10 and all(
#                 history['mean_loss'] > mean_loss * 1.1 
#                 for history in training_history[-10:]
#             ):
#                 robust_log("Potential training stagnation detected. Stopping.")
#                 break
        
#         return model, best_model_path, training_history
    
#     # MAIN EXECUTION PIPELINE
#     try:
#         # Configure TensorFlow
#         configure_tensorflow_environment()
        
#         # Load data
#         X_mmap, y_mmap, val_X, val_y = load_data_safely(X_file, y_file, train_indices, val_indices...
        
#         # Execute training
#         trained_model, best_model_path, training_log = execute_training_loop(
#             model, X_mmap, y_mmap, train_indices, val_X, val_y, output_dir
#         )
        
#         # Save training history
#         with open(os.path.join(output_dir, "training_history.json"), "w") as f:
#             json.dump(training_log, f, indent=4)
        
#         return trained_model, best_model_path
    
#     except Exception as pipeline_error:
#         robust_log(f"PIPELINE EXECUTION FAILED: {pipeline_error}", "CRITICAL")
#         robust_log(traceback.format_exc(), "CRITICAL")
#         raise

# def run_improved_pipeline():
#     """
#     Run the improved pipeline with stall-proof training.
#     """
#     # Configure TensorFlow
#     configure_tensorflow_memory()
    
#     # Paths and configuration
#     output_dir = '/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/improved_model'
#     data_dir = '/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/ml_data'
#     X_file = os.path.join(data_dir, 'X_features.npy')
#     y_file = os.path.join(data_dir, 'y_labels.npy')
#     metadata_file = os.path.join(data_dir, 'metadata.pkl')
    
#     # Create directories
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
#     # Load split indices
#     with open("zero_curtain_pipeline/modeling/checkpoints/spatiotemporal_split.pkl", "rb") as f:
#         split_data = pickle.load(f)
#     train_indices = split_data["train_indices"]
#     val_indices = split_data["val_indices"]
#     test_indices = split_data["test_indices"]
    
#     # Load spatial weights
#     try:
#         with open("zero_curtain_pipeline/modeling/checkpoints/spatial_density.pkl", "rb") as f:
#             weights_data = pickle.load(f)
#         sample_weights = weights_data["weights"][train_indices]
#         sample_weights = sample_weights / np.mean(sample_weights) * len(sample_weights)
#     except:
#         print("No spatial weights found, using uniform weights")
#         sample_weights = None
    
#     # Calculate class weights
#     y = np.load(y_file, mmap_mode='r')
#     train_y = y[train_indices]
#     pos_weight = (len(train_y) - np.sum(train_y)) / max(1, np.sum(train_y))
#     class_weight = {0: 1.0, 1: pos_weight}
#     print(f"Using class weight {pos_weight:.2f} for positive examples")
    
#     # Get sample shape
#     X = np.load(X_file, mmap_mode='r')
#     input_shape = X[train_indices[0]].shape
    
#     # Build improved model
#     print("Building improved model...")
#     model = build_improved_zero_curtain_model(input_shape)
    
#     # Print model summary
#     model.summary()
    
#     # Determine start batch by finding the most recent checkpoint
#     start_batch = 0
#     checkpoint_dir = os.path.join(output_dir, "checkpoints")
#     if os.path.exists(checkpoint_dir):
#         checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
#                            if f.startswith("model_batch_") and f.endswith(".h5")]
        
#         if checkpoint_files:
#             batch_nums = []
#             for f in checkpoint_files:
#                 try:
#                     batch_num = int(f.replace("model_batch_", "").replace(".h5", ""))
#                     batch_nums.append(batch_num)
#                 except ValueError:
#                     continue
            
#             if batch_nums:
#                 start_batch = max(batch_nums)
#                 print(f"Found existing checkpoint at batch {start_batch}")

#     # Train model with failsafe function
#     print("\nTraining model with stall-proof techniques...")
#     model, best_model_path = ultra_robust_training(
#         model=model,
#         X_file=X_file,
#         y_file=y_file,
#         train_indices=train_indices,
#         val_indices=val_indices,
#         test_indices=test_indices,
#         output_dir=output_dir,
#         batch_size=1024,
#         max_epochs=100,  # More epochs with early stopping
#         class_weight=class_weight
#     )
    
#     # Evaluate model
#     print("\nEvaluating model...")
#     eval_results = evaluate_model_with_visualizations(
#         model=model,
#         X_file=X_file,
#         y_file=y_file,
#         test_indices=test_indices,
#         output_dir=output_dir,
#         metadata_file=metadata_file
#     )
    
#     # Analyze spatial performance
#     print("\nAnalyzing spatial performance patterns...")
#     spatial_results = analyze_spatial_performance(
#         output_dir=output_dir,
#         metadata_file=metadata_file
#     )
    
#     # Analyze feature importance
#     print("\nAnalyzing feature importance...")
#     feature_results = analyze_feature_importance(
#         model=model,
#         X_file=X_file,
#         y_file=y_file,
#         test_indices=test_indices,
#         output_dir=output_dir
#     )
    
#     # Save all results
#     all_results = {
#         'evaluation': eval_results,
#         'spatial_analysis': spatial_results,
#         'feature_importance': feature_results
#     }
    
#     with open(os.path.join(output_dir, "complete_analysis.json"), "w") as f:
#         import json
        
#         # Convert numpy values to Python native types
#         class NumpyEncoder(json.JSONEncoder):
#             def default(self, obj):
#                 if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
#                     return float(obj)
#                 return json.JSONEncoder.default(self, obj)
        
#         json.dump(all_results, f, indent=4, cls=NumpyEncoder)
    
#     print("\nImproved pipeline complete!")
#     print(f"Results saved to {output_dir}")
    
#     return all_results

# if __name__ == "__main__":
#     run_improved_pipeline()


# Make sure this is at the very...
def configure_tensorflow_memory():
    """Configure TensorFlow to use memory growth and limit GPU memory allocation"""
    import tensorflow as tf
    import os
    
    # Reset TensorFlow session and clear any previous configurations
    tf.keras.backend.clear_session()
    
    # Set environment variables to control TensorFlow behavior
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    
    # Configure GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                # Allow memory growth - prevents TF from allocating all GPU memory at once
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Memory growth enabled for {device}")
            except Exception as e:
                print(f"Error configuring GPU: {e}")
    
    # Configure threading before initializing the TensorFlow context
    try:
        # These settings need to be applied before TF operations, so we reset first
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception as e:
        print(f"Unable to configure threading: {e}")
    
    # Set soft device placement
    tf.config.set_soft_device_placement(True)

def f1_score(y_true, y_pred):
    """
    Compatible F1 score implementation
    # """
    # This function will be passed directly to the metrics parameter
    precision = tf.keras.metrics.Precision()(y_true, y_pred)
    recall = tf.keras.metrics.Recall()(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

def build_improved_zero_curtain_model(input_shape, include_moisture=True):
    """
    Advanced model architecture with increased regularization and gradient clipping.
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Dropout
    from tensorflow.keras.layers import Reshape, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    
    # L2 regularization strength
    reg_strength = 0.0001
    # Increased dropout rate
    dropout_rate = 0.3
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Reshape for ConvLSTM (add spatial dimension)
    x = tf.keras.layers.Reshape((input_shape[0], 1, 1, input_shape[1]))(inputs)
    
    # ConvLSTM layer with regularization
    from tensorflow.keras.layers import ConvLSTM2D
    convlstm = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 1),
        padding='same',
        return_sequences=True,
        activation='tanh',
        recurrent_dropout=dropout_rate,
        kernel_regularizer=l2(reg_strength)
    )(x)
    
    # Reshape back to (sequence_length, features)
    convlstm = Reshape((input_shape[0], 64))(convlstm)
    
    # Add positional encoding for transformer (existing function)
    def positional_encoding(length, depth):
        positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
        depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth
        
        angle_rates = 1 / tf.pow(10000.0, depths)
        angle_rads = positions * angle_rates
        
        # Only use sin to ensure output depth matches input depth
        pos_encoding = tf.sin(angle_rads)
        
        # Add batch dimension
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        
        return pos_encoding
    
    # Add positional encoding
    pos_encoding = positional_encoding(input_shape[0], 64)
    transformer_input = convlstm + pos_encoding
    
    # Improved transformer encoder block
    def transformer_encoder(x, num_heads=8, key_dim=64, ff_dim=128):
        # Multi-head attention with regularization
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,
            kernel_regularizer=l2(reg_strength)
        )(x, x)
        
        # Skip connection 1 with dropout
        x1 = Add()([attention_output, x])
        x1 = LayerNormalization(epsilon=1e-6)(x1)
        x1 = Dropout(dropout_rate)(x1)  # Added dropout after norm
        
        # Feed-forward network with regularization
        ff_output = Dense(ff_dim, activation='relu', kernel_regularizer=l2(reg_strength))(x1)
        ff_output = Dropout(dropout_rate)(ff_output)
        ff_output = Dense(64, kernel_regularizer=l2(reg_strength))(ff_output)
        
        # Skip connection 2
        x2 = Add()([ff_output, x1])
        return LayerNormalization(epsilon=1e-6)(x2)
    
    # Apply transformer encoder
    transformer_output = transformer_encoder(transformer_input)
    
    # Parallel CNN paths for multi-scale feature extraction (with regularization)
    cnn_1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', 
                  kernel_regularizer=l2(reg_strength))(inputs)
    cnn_1 = BatchNormalization()(cnn_1)
    cnn_1 = Dropout(dropout_rate/2)(cnn_1)  # Lighter dropout for CNNs
    
    cnn_2 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu',
                  kernel_regularizer=l2(reg_strength))(inputs)
    cnn_2 = BatchNormalization()(cnn_2)
    cnn_2 = Dropout(dropout_rate/2)(cnn_2)
    
    cnn_3 = Conv1D(filters=32, kernel_size=7, padding='same', activation='relu',
                  kernel_regularizer=l2(reg_strength))(inputs)
    cnn_3 = BatchNormalization()(cnn_3)
    cnn_3 = Dropout(dropout_rate/2)(cnn_3)
    
    # VAE components (existing code with regularization)
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    # Global temporal features
    global_max = GlobalMaxPooling1D()(transformer_output)
    global_avg = GlobalAveragePooling1D()(transformer_output)
    
    # VAE encoding with regularization
    z_mean = Dense(32, kernel_regularizer=l2(reg_strength))(Concatenate()([global_max, global_avg]))
    z_log_var = Dense(32, kernel_regularizer=l2(reg_strength))(Concatenate()([global_max, global_avg]))
    z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
    
    # Combine all features
    merged_features = Concatenate()(
        [
            GlobalMaxPooling1D()(cnn_1),
            GlobalMaxPooling1D()(cnn_2),
            GlobalMaxPooling1D()(cnn_3),
            global_max,
            global_avg,
            z
        ]
    )
    
    # Final classification layers with regularization
    x = Dense(128, activation='relu', kernel_regularizer=l2(reg_strength))(merged_features)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(reg_strength))(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_strength))(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Add VAE loss with reduced weight
    kl_loss = -0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
    )
    model.add_loss(0.0005 * kl_loss)  # Reduced from 0.001 to 0.0005
    
    # Compile model with gradient clipping but without the custom F1 metric
    # that's causing the ZerosLike error
    # model.compile(
    #     optimizer=Adam(
    #         learning_rate=0.0005,  # Reduced learning rate
    #         clipvalue=1.0  # Add gradient clipping
    #     ),
    #     loss='binary_crossentropy',
    #     metrics=[
    #         'accuracy',
    #         tf.keras.metrics.AUC(name='auc'),
    #         tf.keras.metrics.Precision(name='precision'),
    #         tf.keras.metrics.Recall(name='recall')
    #         # Removed F1Score metric that was causing errors
    #     ]
    # )
    model.compile(
        optimizer=Adam(
            learning_rate=0.0005,
            clipvalue=1.0
        ),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            f1_score  # Use the function directly
        ]
    )
    
    return model

# def build_improved_zero_curtain_model(input_shape, include_moisture=True):
#     """
#     Advanced model architecture with increased regularization and gradient clipping.
#     """
#     from tensorflow.keras.models import Model
#     from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Dropout
#     from tensorflow.keras.layers import Reshape, Concatenate, GlobalAveragePooling1D, GlobalMaxPoo...
#     from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
#     from tensorflow.keras.optimizers import Adam
#     from tensorflow.keras.regularizers import l2
    
#     # Create custom F1 metric since F1Score isn't available
#     # class F1Score(tf.keras.metrics.Metric):
#     #     def __init__(self, name='f1', **kwargs):
#     #         super().__init__(name=name, **kwargs)
#     #         self.precision = tf.keras.metrics.Precision()
#     #         self.recall = tf.keras.metrics.Recall()
            
#     #     def update_state(self, y_true, y_pred, sample_weight=None):
#     #         y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
#     #         self.precision.update_state(y_true, y_pred, sample_weight)
#     #         self.recall.update_state(y_true, y_pred, sample_weight)
            
#     #     def result(self):
#     #         p = self.precision.result()
#     #         r = self.recall.result()
#     #         # Calculate F1
#     #         return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())
            
#     #     def reset_state(self):
#     #         self.precision.reset_state()
#     #         self.recall.reset_state()
    
#     # L2 regularization strength
#     reg_strength = 0.0001
#     # Increased dropout rate
#     dropout_rate = 0.3
    
#     # Input layer
#     inputs = Input(shape=input_shape)
    
#     # Reshape for ConvLSTM (add spatial dimension)
#     x = tf.keras.layers.Reshape((input_shape[0], 1, 1, input_shape[1]))(inputs)
    
#     # ConvLSTM layer with regularization
#     from tensorflow.keras.layers import ConvLSTM2D
#     convlstm = ConvLSTM2D(
#         filters=64,
#         kernel_size=(3, 1),
#         padding='same',
#         return_sequences=True,
#         activation='tanh',
#         recurrent_dropout=dropout_rate,
#         kernel_regularizer=l2(reg_strength)
#     )(x)
    
#     # Reshape back to (sequence_length, features)
#     convlstm = Reshape((input_shape[0], 64))(convlstm)
    
#     # Add positional encoding for transformer (existing function)
#     def positional_encoding(length, depth):
#         positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
#         depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth
        
#         angle_rates = 1 / tf.pow(10000.0, depths)
#         angle_rads = positions * angle_rates
        
#         # Only use sin to ensure output depth matches input depth
#         pos_encoding = tf.sin(angle_rads)
        
#         # Add batch dimension
#         pos_encoding = tf.expand_dims(pos_encoding, 0)
        
#         return pos_encoding
    
#     # Add positional encoding
#     pos_encoding = positional_encoding(input_shape[0], 64)
#     transformer_input = convlstm + pos_encoding
    
#     # Improved transformer encoder block
#     def transformer_encoder(x, num_heads=8, key_dim=64, ff_dim=128):
#         # Multi-head attention with regularization
#         attention_output = MultiHeadAttention(
#             num_heads=num_heads, key_dim=key_dim,
#             kernel_regularizer=l2(reg_strength)
#         )(x, x)
        
#         # Skip connection 1 with dropout
#         x1 = Add()([attention_output, x])
#         x1 = LayerNormalization(epsilon=1e-6)(x1)
#         x1 = Dropout(dropout_rate)(x1)  # Added dropout after norm
        
#         # Feed-forward network with regularization
#         ff_output = Dense(ff_dim, activation='relu', kernel_regularizer=l2(reg_strength))(x1)
#         ff_output = Dropout(dropout_rate)(ff_output)
#         ff_output = Dense(64, kernel_regularizer=l2(reg_strength))(ff_output)
        
#         # Skip connection 2
#         x2 = Add()([ff_output, x1])
#         return LayerNormalization(epsilon=1e-6)(x2)
    
#     # Apply transformer encoder
#     transformer_output = transformer_encoder(transformer_input)
    
# # Parallel CNN paths for multi-scale feature...
#     cnn_1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', 
#                   kernel_regularizer=l2(reg_strength))(inputs)
#     cnn_1 = BatchNormalization()(cnn_1)
#     cnn_1 = Dropout(dropout_rate/2)(cnn_1)  # Lighter dropout for CNNs
    
#     cnn_2 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu',
#                   kernel_regularizer=l2(reg_strength))(inputs)
#     cnn_2 = BatchNormalization()(cnn_2)
#     cnn_2 = Dropout(dropout_rate/2)(cnn_2)
    
#     cnn_3 = Conv1D(filters=32, kernel_size=7, padding='same', activation='relu',
#                   kernel_regularizer=l2(reg_strength))(inputs)
#     cnn_3 = BatchNormalization()(cnn_3)
#     cnn_3 = Dropout(dropout_rate/2)(cnn_3)
    
#     # VAE components (existing code with regularization)
#     def sampling(args):
#         z_mean, z_log_var = args
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
#     # Global temporal features
#     global_max = GlobalMaxPooling1D()(transformer_output)
#     global_avg = GlobalAveragePooling1D()(transformer_output)
    
#     # VAE encoding with regularization
#     z_mean = Dense(32, kernel_regularizer=l2(reg_strength))(Concatenate()([global_max, global_avg]...
#     z_log_var = Dense(32, kernel_regularizer=l2(reg_strength))(Concatenate()([global_max, global_a...
#     z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
    
#     # Combine all features
#     merged_features = Concatenate()(
#         [
#             GlobalMaxPooling1D()(cnn_1),
#             GlobalMaxPooling1D()(cnn_2),
#             GlobalMaxPooling1D()(cnn_3),
#             global_max,
#             global_avg,
#             z
#         ]
#     )
    
#     # Final classification layers with regularization
#     x = Dense(128, activation='relu', kernel_regularizer=l2(reg_strength))(merged_features)
#     x = Dropout(dropout_rate)(x)
#     x = BatchNormalization()(x)
#     x = Dense(64, activation='relu', kernel_regularizer=l2(reg_strength))(x)
#     x = Dropout(dropout_rate)(x)
    
#     # Output layer
#     outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_strength))(x)
    
#     # Create model
#     model = Model(inputs=inputs, outputs=outputs)
    
#     # Add VAE loss with reduced weight
#     kl_loss = -0.5 * tf.reduce_mean(
#         z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
#     )
#     model.add_loss(0.0005 * kl_loss)  # Reduced from 0.001 to 0.0005

#     # In your build_improved_zero_curtain_model function
#     model.compile(
#         optimizer=Adam(
#             learning_rate=0.0005,
#             clipvalue=1.0
#         ),
#         loss='binary_crossentropy',
#         metrics=[
#             'accuracy',
#             tf.keras.metrics.AUC(name='auc'),
#             tf.keras.metrics.Precision(name='precision'),
#             tf.keras.metrics.Recall(name='recall'),
#             f1_score  # Use the function directly
#         ]
#     )
    
#     # Compile model with gradient clipping
#     # model.compile(
#     #     optimizer=Adam(
#     #         learning_rate=0.0005,  # Reduced learning rate
#     #         clipvalue=1.0  # Add gradient clipping
#     #     ),
#     #     loss='binary_crossentropy',
#     #     metrics=[
#     #         'accuracy',
#     #         tf.keras.metrics.AUC(name='auc'),
#     #         tf.keras.metrics.Precision(name='precision'),
#     #         tf.keras.metrics.Recall(name='recall'),
#     #         F1Score(name='f1')  # Using our custom F1 implementation
#     #     ]
#     # )
    
#     return model

# Cyclical Learning Rate Callback
class CyclicLR(tf.keras.callbacks.Callback):
    """
    Cyclical learning rate callback for smoother training.
    """
    def __init__(
        self,
        base_lr=0.0001,
        max_lr=0.001,
        step_size=2000,
        mode='triangular2'
    ):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        
        if self.mode == 'triangular':
            self.scale_fn = lambda x: 1.
        elif self.mode == 'triangular2':
            self.scale_fn = lambda x: 1 / (2. ** (x - 1))
        elif self.mode == 'exp_range':
            self.scale_fn = lambda x: 0.9 ** x
        
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        
    def on_train_begin(self, logs=None):
        logs = logs or {}
        tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)
            
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())

# Enhanced callbacks setup
def get_enhanced_callbacks(output_dir, fold_idx=None):
    """
    Create enhanced callbacks with increased patience and cyclical learning rate.
    """
    sub_dir = f"fold_{fold_idx}" if fold_idx is not None else ""
    checkpoint_dir = os.path.join(output_dir, sub_dir, "checkpoints")
    tensorboard_dir = os.path.join(output_dir, sub_dir, "tensorboard")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    callbacks = [
        # Early stopping with increased patience
        tf.keras.callbacks.EarlyStopping(
            patience=20,  # Increased from 15 to 20
            restore_best_weights=True,
            monitor='val_auc',
            mode='max',
            min_delta=0.005
        ),
        # Reduced LR with increased patience
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=10,  # Increased from 7 to 10
            min_lr=1e-6,
            monitor='val_auc',
            mode='max'
        ),
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, "best_model.h5"),
            save_best_only=True,
            monitor='val_auc',
            mode='max'
        ),
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=False,
            update_freq='epoch'
        ),
        # Cyclical learning rate
        CyclicLR(
            base_lr=0.0001,
            max_lr=0.001,
            step_size=2000,
            mode='triangular2'
        ),
        # Memory cleanup
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect()
        )
    ]
    
    return callbacks

def improved_resumable_training(model, X_file, y_file, train_indices, val_indices, test_indices,
                               output_dir, batch_size=256, chunk_size=20000, epochs_per_chunk=3, 
                               save_frequency=5, class_weight=None, start_chunk=0):
    """
    Enhanced training function with improved memory management and validation.
    """
    import os
    import gc
    import json
    import numpy as np
    import tensorflow as tf
    from datetime import datetime, timedelta
    import time
    import psutil
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tensorboard"), exist_ok=True)
    
    # Process in chunks
    num_chunks = int(np.ceil(len(train_indices) / chunk_size))
    print(f"Processing {len(train_indices)} samples in {num_chunks} chunks of {chunk_size}")
    print(f"Starting from chunk {start_chunk+1}")
    
    # Create validation set once (limited size for memory efficiency)
    val_limit = min(2000, len(val_indices))
    val_indices_subset = val_indices[:val_limit]
    
    # Open data files with memory mapping
    X_mmap = np.load(X_file, mmap_mode='r')
    y_mmap = np.load(y_file, mmap_mode='r')
    
    # Load validation data once
    val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
    val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    print(f"Loaded {len(val_X)} validation samples")
    
    # Load existing history if resuming
    history_log = []
    history_path = os.path.join(output_dir, "training_history.json")
    if start_chunk > 0 and os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history_log = json.load(f)
        except Exception as e:
            print(f"Could not load existing history: {e}")
            # Try pickle format
            pickle_path = os.path.join(output_dir, "training_history.pkl")
            if os.path.exists(pickle_path):
                import pickle
                with open(pickle_path, "rb") as f:
                    history_log = pickle.load(f)
    
    # If resuming, load latest model
    if start_chunk > 0:
        # Find the most recent checkpoint before start_chunk
        checkpoint_indices = []
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        for filename in os.listdir(checkpoints_dir):
            if filename.startswith("model_chunk_") and filename.endswith(".h5"):
                try:
                    idx = int(filename.split("_")[-1].split(".")[0])
                    if idx < start_chunk:
                        checkpoint_indices.append(idx)
                except ValueError:
                    continue
        
        if checkpoint_indices:
            latest_idx = max(checkpoint_indices)
            model_path = os.path.join(checkpoints_dir, f"model_chunk_{latest_idx}.h5")
            if os.path.exists(model_path):
                print(f"Loading model from checkpoint {model_path}")
                model = tf.keras.models.load_model(model_path)
            else:
                print(f"Warning: Could not find model checkpoint for chunk {latest_idx}")
    
    # Setup enhanced callbacks
    callbacks = get_enhanced_callbacks(output_dir)
    
    # Track metrics across chunks
    start_time = time.time()
    
    # For safe recovery
    recovery_file = os.path.join(output_dir, "last_completed_chunk.txt")
    
    # Process each chunk
    for chunk_idx in range(start_chunk, num_chunks):
        # Get chunk indices
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(train_indices))
        chunk_indices = train_indices[start_idx:end_idx]
        
        # Report memory
        memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print(f"\n{'='*50}")
        print(f"Processing chunk {chunk_idx+1}/{num_chunks} with {len(chunk_indices)} samples")
        print(f"Memory before: {memory_before:.1f} MB")
        
        # Force garbage collection before loading new data
        gc.collect()
        
        # Load chunk data
        chunk_X = np.array([X_mmap[idx] for idx in chunk_indices])
        chunk_y = np.array([y_mmap[idx] for idx in chunk_indices])

        chunk_X = np.reshape(chunk_X, (chunk_X.shape[0], chunk_X.shape[1], 1, 1, chunk_X.shape[2]))
        
        print(f"Data loaded. Memory: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.1f} MB")

        print(f"Chunk_X shape before training: {chunk_X.shape}")
        print(f"Validation X shape: {val_X.shape}")
        
        # Train on chunk
        print(f"Training for {epochs_per_chunk} epochs...")
        try:
            history = model.fit(
                chunk_X, chunk_y,
                validation_data=(val_X, val_y),
                epochs=epochs_per_chunk,
                batch_size=batch_size,
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=1
            )
            
            # Store serializable metrics
            chunk_metrics = {}
            for k, v in history.history.items():
                chunk_metrics[k] = [float(val) for val in v]
            history_log.append(chunk_metrics)
            
            # Save history
            try:
                with open(history_path, "w") as f:
                    json.dump(history_log, f)
            except Exception as e:
                print(f"Warning: Could not save history to JSON: {e}")
                # Fallback - save as pickle
                import pickle
                with open(os.path.join(output_dir, "training_history.pkl"), "wb") as f:
                    pickle.dump(history_log, f)
                    
        except Exception as e:
            print(f"Error during training: {e}")
            # Save model to avoid complete loss
            error_model_path = os.path.join(output_dir, "checkpoints", f"error_recovery_chunk_{chunk_idx}.h5")
            model.save(error_model_path)
            print(f"Saved model for error recovery to {error_model_path}")
            continue
        
        # Save model periodically
        if (chunk_idx + 1) % save_frequency == 0 or chunk_idx == num_chunks - 1:
            model_path = os.path.join(output_dir, "checkpoints", f"model_chunk_{chunk_idx+1}.h5")
            model.save(model_path)
            print(f"Model saved to {model_path}")
        
        # Generate predictions periodically
        if (chunk_idx + 1) % save_frequency == 0 or chunk_idx == num_chunks - 1:
            chunk_preds = model.predict(chunk_X, batch_size=batch_size)
            np.save(os.path.join(output_dir, "predictions", f"chunk_{chunk_idx+1}_predictions.npy"), chunk_preds)
            np.save(os.path.join(output_dir, "predictions", f"chunk_{chunk_idx+1}_indices.npy"), chunk_indices)
            del chunk_preds
        
        # Explicitly delete everything from memory
        del chunk_X, chunk_y
        
        # Force garbage collection
        gc.collect()
        
        # Report memory after cleanup
        memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print(f"Memory after: {memory_after:.1f} MB (Change: {memory_after - memory_before:.1f} MB)")
        
        # Write recovery file with last completed chunk
        with open(recovery_file, "w") as f:
            f.write(str(chunk_idx + 1))
        
        # Estimate time
        elapsed = time.time() - start_time
        avg_time_per_chunk = elapsed / (chunk_idx - start_chunk + 1) if chunk_idx > start_chunk else elapsed
        remaining = avg_time_per_chunk * (num_chunks - chunk_idx - 1)
        print(f"Estimated remaining time: {timedelta(seconds=int(remaining))}")
        
        # Reset TensorFlow session periodically
        if (chunk_idx + 1) % 40 == 0:
            print("Approaching potential memory issue - resetting TensorFlow session")
            temp_model_path = os.path.join(output_dir, "checkpoints", f"temp_reset_point_{chunk_idx+1}.h5")
            model.save(temp_model_path)
            
            # Clear session
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Reload model
            model = tf.keras.models.load_model(temp_model_path)
            print("TensorFlow session reset complete")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.h5")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return model, final_model_path

def evaluate_model_with_visualizations(model, X_file, y_file, test_indices, output_dir, metadata_file=None):
    """
    Comprehensive evaluation with detailed visualizations.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_curve, 
        auc, precision_recall_curve, average_precision_score
    )
    import os
    import gc
    
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Load metadata if available
    metadata = None
    if metadata_file:
        import pickle
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
    
    # Load memory-mapped data
    X = np.load(X_file, mmap_mode='r')
    y = np.load(y_file, mmap_mode='r')
    
    # Process test data in batches
    batch_size = 1000
    num_batches = int(np.ceil(len(test_indices) / batch_size))
    
    all_preds = []
    all_true = []
    all_meta = []
    
    print(f"Evaluating model on {len(test_indices)} test samples in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(test_indices))
        batch_indices = test_indices[start_idx:end_idx]
        
        # Load batch data
        batch_X = np.array([X[idx] for idx in batch_indices])
        batch_y = np.array([y[idx] for idx in batch_indices])
        
        # Store metadata if available
        if metadata:
            batch_meta = [metadata[idx] for idx in batch_indices]
            all_meta.extend(batch_meta)
        
        # Predict
        batch_preds = model.predict(batch_X, verbose=0)
        
        # Store results
        all_preds.extend(batch_preds.flatten())
        all_true.extend(batch_y)
        
        # Clean up
        del batch_X, batch_y, batch_preds
        gc.collect()
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    # Create binary predictions
    all_preds_binary = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    report = classification_report(all_true, all_preds_binary, output_dict=True)
    report_str = classification_report(all_true, all_preds_binary)
    conf_matrix = confusion_matrix(all_true, all_preds_binary)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_true, all_preds)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_true, all_preds)
    avg_precision = average_precision_score(all_true, all_preds)
    
    print("\nClassification Report:")
    print(report_str)
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    # Save results to file
    with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
        f.write("Classification Report:\n")
        f.write(report_str)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write(f"\n\nROC AUC: {roc_auc:.4f}")
        f.write(f"\nAverage Precision: {avg_precision:.4f}")
    
    # Create visualizations
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "visualizations", "confusion_matrix.png"), dpi=300)
    plt.close()
    
    # 2. ROC Curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "visualizations", "roc_curve.png"), dpi=300)
    plt.close()
    
    # 3. Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "visualizations", "precision_recall_curve.png"), dpi=300)
    plt.close()
    
    # 4. Score distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(all_preds[all_true == 0], bins=50, alpha=0.5, label='Negative Class', color='blue')
    sns.histplot(all_preds[all_true == 1], bins=50, alpha=0.5, label='Positive Class', color='red')
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Scores by Class')
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "visualizations", "score_distribution.png"), dpi=300)
    plt.close()
    
    # Save predictions
    if metadata:
        # Create results DataFrame with metadata
        results_data = []
        for i in range(len(all_preds)):
            result = {
                'prediction': all_preds[i],
                'true_label': all_true[i],
                'predicted_label': all_preds_binary[i],
                'correct': all_preds_binary[i] == all_true[i]
            }
            
            # Add metadata
            meta = all_meta[i]
            for key, value in meta.items():
                if isinstance(value, (np.integer, np.floating)):
                    value = float(value)
                result[key] = value
            
            results_data.append(result)
        
        # Create DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        results_df.to_csv(os.path.join(output_dir, "test_predictions_with_metadata.csv"), index=False)
    else:
        # Save simpler predictions
        np.save(os.path.join(output_dir, "test_predictions.npy"), all_preds)
        np.save(os.path.join(output_dir, "test_true_labels.npy"), all_true)
    
    return {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score']
    }

# # Example usage
# def main():
#     # Configuration
#     input_shape = (24, 5)  # Adjust to match your data shape
#     output_dir = '/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/improved_model'
#     data_dir = '/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/ml_data'
#     X_file = os.path.join(data_dir, 'X_features.npy')
#     y_file = os.path.join(data_dir, 'y_labels.npy')
#     metadata_file = os.path.join(data_dir, 'metadata.pkl')
    
#     # Load split indices
#     with open("zero_curtain_pipeline/modeling/checkpoints/spatiotemporal_split.pkl", "rb") as f:
#         split_data = pickle.load(f)
#     train_indices = split_data["train_indices"]
#     val_indices = split_data["val_indices"]
#     test_indices = split_data["test_indices"]
    
#     # Load spatial weights
#     with open("zero_curtain_pipeline/modeling/checkpoints/spatial_density.pkl", "rb") as f:
#         weights_data = pickle.load(f)
    
#     sample_weights = weights_data["weights"][train_indices]
#     sample_weights = sample_weights / np.mean(sample_weights) * len(sample_weights)
    
#     # Calculate class weights
#     y = np.load(y_file, mmap_mode='r')
#     train_y = y[train_indices]
#     pos_weight = (len(train_y) - np.sum(train_y)) / max(1, np.sum(train_y))
#     class_weight = {0: 1.0, 1: pos_weight}
#     print(f"Using class weight {pos_weight:.2f} for positive examples")
    
#     # Build improved model
#     X = np.load(X_file, mmap_mode='r')
#     input_shape = X[train_indices[0]].shape
#     model = build_improved_zero_curtain_model(input_shape)
    
#     # Train model with improved functions
#     model, model_path = improved_resumable_training(
#         model=model,
#         X_file=X_file,
#         y_file=y_file,
#         train_indices=train_indices,
#         val_indices=val_indices,
#         test_indices=test_indices,
#         output_dir=output_dir,
#         batch_size=256,
#         chunk_size=20000,  # Smaller chunks for better memory efficiency
#         epochs_per_chunk=3,  # More epochs per chunk for better learning
#         save_frequency=5,
#         class_weight=class_weight,
#         start_chunk=0  # Start from beginning or set to resume
#     )
    
#     # Evaluate the final model
#     results = evaluate_model_with_visualizations(
#         model=model,
#         X_file=X_file,
#         y_file=y_file,
#         test_indices=test_indices,
#         output_dir=output_dir,
#         metadata_file=metadata_file
#     )
    
#     print("\nFinal evaluation results:")
#     for metric, value in results.items():
#         print(f"  {metric}: {value:.4f}")

# if __name__ == "__main__":
#     main()

def analyze_spatial_performance(output_dir, metadata_file):
    """
    Analyze model performance across different geographical regions.
    
    Parameters:
    -----------
    output_dir : str
        Directory with test predictions
    metadata_file : str
        Path to metadata pickle file
        
    Returns:
    --------
    dict
        Spatial analysis results
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json
    
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations", "spatial")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load predictions with metadata
    pred_file = os.path.join(output_dir, "test_predictions_with_metadata.csv")
    if os.path.exists(pred_file):
        predictions = pd.read_csv(pred_file)
    else:
        # Load separate files and combine
        preds = np.load(os.path.join(output_dir, "test_predictions.npy"))
        true = np.load(os.path.join(output_dir, "test_true_labels.npy"))
        
        # Load metadata
        import pickle
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        
        # Create DataFrame
        test_indices = np.load(os.path.join(output_dir, "test_indices.npy"))
        predictions = pd.DataFrame({
            'prediction': preds,
            'true_label': true,
            'predicted_label': (preds > 0.5).astype(int),
            'correct': ((preds > 0.5).astype(int) == true).astype(int)
        })
        
        # Add metadata
        for i, idx in enumerate(test_indices):
            meta = metadata[idx]
            for key, value in meta.items():
                if key not in predictions.columns:
                    predictions[key] = None
                predictions.loc[i, key] = value
    
    # Define latitude bands
    def get_lat_band(lat):
        if pd.isna(lat):
            return "Unknown"
        lat = float(lat)
        if lat < 49:
            return "<49°N"
        elif lat < 55:
            return "49-55°N (Boreal)"
        elif lat < 60:
            return "55-60°N (Boreal)"
        elif lat < 66.5:
            return "60-66.5°N (Subarctic)"
        elif lat < 70:
            return "66.5-70°N (Arctic)"
        elif lat < 75:
            return "70-75°N (Arctic)"
        elif lat < 80:
            return "75-80°N (Arctic)"
        else:
            return ">80°N (Arctic)"
    
    # Add latitude band
    predictions['lat_band'] = predictions['latitude'].apply(get_lat_band)
    
    # Group by source (site)
    site_metrics = predictions.groupby('source').apply(lambda x: pd.Series({
        'count': len(x),
        'positive_count': x['true_label'].sum(),
        'positive_rate': x['true_label'].mean(),
        'accuracy': x['correct'].mean(),
        'precision': (x['predicted_label'] & x['true_label']).sum() / max(1, x['predicted_label'].sum()),
        'recall': (x['predicted_label'] & x['true_label']).sum() / max(1, x['true_label'].sum()),
        'f1': 2 * (x['predicted_label'] & x['true_label']).sum() / 
              max(1, (x['predicted_label'] & x['true_label']).sum() + 
                  0.5 * (x['predicted_label'].sum() + x['true_label'].sum())),
        'latitude': x['latitude'].mean(),
        'longitude': x['longitude'].mean(),
        'lat_band': x['lat_band'].iloc[0] if not x['lat_band'].isna().all() else "Unknown"
    }))
    
    # Filter out sites with too few samples
    site_metrics = site_metrics[site_metrics['count'] >= 10]
    
    # Group by latitude band
    band_metrics = predictions.groupby('lat_band').apply(lambda x: pd.Series({
        'count': len(x),
        'positive_count': x['true_label'].sum(),
        'positive_rate': x['true_label'].mean(),
        'accuracy': x['correct'].mean(),
        'precision': (x['predicted_label'] & x['true_label']).sum() / max(1, x['predicted_label'].sum()),
        'recall': (x['predicted_label'] & x['true_label']).sum() / max(1, x['true_label'].sum()),
        'f1': 2 * (x['predicted_label'] & x['true_label']).sum() / 
              max(1, (x['predicted_label'] & x['true_label']).sum() + 
                  0.5 * (x['predicted_label'].sum() + x['true_label'].sum()))
    }))
    
    # Create visualizations
    # 1. Performance by latitude band
    plt.figure(figsize=(14, 8))
    
    # Order bands by latitude
    ordered_bands = [
        '<55°N', '55-60°N', '60-66.5°N (Subarctic)', 
        '66.5-70°N (Arctic)', '70-75°N (Arctic)', 
        '75-80°N (Arctic)', '>80°N (Arctic)'
    ]
    ordered_bands = [b for b in ordered_bands if b in band_metrics.index]
    
    # Get metrics
    band_data = band_metrics.loc[ordered_bands]
    
    # Plot metrics
    x = np.arange(len(ordered_bands))
    width = 0.15
    
    plt.bar(x - 2*width, band_data['accuracy'], width, label='Accuracy', color='#3274A1')
    plt.bar(x - width, band_data['precision'], width, label='Precision', color='#E1812C')
    plt.bar(x, band_data['recall'], width, label='Recall', color='#3A923A')
    plt.bar(x + width, band_data['f1'], width, label='F1', color='#C03D3E')
    plt.bar(x + 2*width, band_data['positive_rate'], width, label='Positive Rate', color='#9372B2')
    
    plt.xlabel('Latitude Band')
    plt.ylabel('Score')
    plt.title('Model Performance by Latitude Band')
    plt.xticks(x, ordered_bands, rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "performance_by_latitude.png"), dpi=300)
    plt.close()
    
    # 2. Map visualization if cartopy is available
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        plt.figure(figsize=(15, 12))
        
        # Set up the projection
        ax = plt.axes(projection=ccrs.NorthPolarStereo())
        ax.set_extent([-180, 180, 45, 90], ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        
        # Filter sites with valid coordinates
        valid_sites = site_metrics.dropna(subset=['latitude', 'longitude'])
        
        # Create scatter plot
        scatter = ax.scatter(
            valid_sites['longitude'], 
            valid_sites['latitude'],
            transform=ccrs.PlateCarree(),
            c=valid_sites['f1'],
            s=valid_sites['count'] / 5,  # Size by sample count
            cmap='RdYlGn',
            vmin=0, vmax=1,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label('F1 Score')
        
        # Add Arctic Circle
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0, 0], 90 - 66.5
        verts = np.vstack([radius*np.sin(theta), radius*np.cos(theta)]).T
        circle = plt.Line2D(verts[:, 0], verts[:, 1], color='blue', 
                          linestyle='--', transform=ax.transData)
        ax.add_line(circle)
        
        plt.title('Spatial Distribution of Model Performance (F1 Score)', fontsize=14)
        plt.savefig(os.path.join(vis_dir, "spatial_performance_map.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except ImportError:
        print("Cartopy not available for map visualization")
    
    # 3. Performance vs. sample count
    plt.figure(figsize=(12, 8))
    plt.scatter(site_metrics['count'], site_metrics['f1'], 
               c=site_metrics['positive_rate'], cmap='viridis', 
               alpha=0.7, s=50, edgecolor='black', linewidth=1)
    plt.colorbar(label='Positive Rate')
    plt.xscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Sample Count by Site')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "performance_vs_sample_count.png"), dpi=300)
    plt.close()
    
    # Save results
    site_metrics.to_csv(os.path.join(vis_dir, "site_metrics.csv"))
    band_metrics.to_csv(os.path.join(vis_dir, "latitude_band_metrics.csv"))
    
    # Return summary
    return {
        'site_count': len(site_metrics),
        'latitude_bands': band_metrics.to_dict(),
        'best_performing_sites': site_metrics.nlargest(5, 'f1')[['f1', 'count', 'positive_rate']].to_dict(),
        'worst_performing_sites': site_metrics.nsmallest(5, 'f1')[['f1', 'count', 'positive_rate']].to_dict()
    }

def analyze_feature_importance(model, X_file, y_file, test_indices, output_dir):
    """
    Analyze feature importance using permutation importance.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    X_file : str
        Path to features numpy file
    y_file : str
        Path to labels numpy file
    test_indices : numpy.ndarray
        Test set indices
    output_dir : str
        Output directory for results
        
    Returns:
    --------
    dict
        Feature importance results
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_auc_score
    import gc
    
    # Create output directory
    vis_dir = os.path.join(output_dir, "visualizations", "features")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load a subset of test data to save memory
    max_samples = 5000
    if len(test_indices) > max_samples:
        np.random.seed(42)
        test_indices_subset = np.random.choice(test_indices, max_samples, replace=False)
    else:
        test_indices_subset = test_indices
    
    # Load data
    X = np.load(X_file, mmap_mode='r')
    y = np.load(y_file, mmap_mode='r')
    
    X_test = np.array([X[idx] for idx in test_indices_subset])
    y_test = np.array([y[idx] for idx in test_indices_subset])
    
    # Get feature names (modify as needed based on your data)
    feature_names = ['Temperature', 'Temperature Gradient', 'Depth', 'Moisture', 'Moisture Gradient']
    
    # Truncate to actual number of features
    feature_names = feature_names[:X_test.shape[2]]
    
    # Get baseline performance
    baseline_preds = model.predict(X_test)
    baseline_auc = roc_auc_score(y_test, baseline_preds)
    
    # Perform permutation importance analysis
    n_repeats = 5
    importances = np.zeros((len(feature_names), n_repeats))
    
    for feature_idx in range(len(feature_names)):
        for repeat in range(n_repeats):
            # Create a copy of the test data
            X_permuted = X_test.copy()
            
            # Permute the feature across all time steps
            for time_step in range(X_test.shape[1]):
                X_permuted[:, time_step, feature_idx] = np.random.permutation(X_test[:, time_step, feature_idx])
            
            # Predict with permuted feature
            perm_preds = model.predict(X_permuted)
            perm_auc = roc_auc_score(y_test, perm_preds)
            
            # Store importance (decrease in performance)
            importances[feature_idx, repeat] = baseline_auc - perm_auc
            
            # Clean up
            del X_permuted, perm_preds
            gc.collect()
    
    # Calculate mean and std of importance
    mean_importances = np.mean(importances, axis=1)
    std_importances = np.std(importances, axis=1)
    
    # Create DataFrame for results
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_importances,
        'Std': std_importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, 
               xerr=importance_df['Std'], palette='viridis')
    plt.title('Feature Importance (Permutation Method)')
    plt.xlabel('Decrease in AUC when Feature is Permuted')
    plt.ylabel('Feature')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "feature_importance.png"), dpi=300)
    plt.close()
    
    # Save results
    importance_df.to_csv(os.path.join(vis_dir, "feature_importance.csv"), index=False)
    
    # Analyze temporal patterns in feature importance
    # We'll analyze which time steps in the sequence are most important
    time_importances = np.zeros((X_test.shape[1], n_repeats))
    
    for time_idx in range(X_test.shape[1]):
        for repeat in range(n_repeats):
            # Create a copy of the test data
            X_permuted = X_test.copy()
            
            # Permute all features at this time step
            X_permuted[:, time_idx, :] = np.random.permutation(X_test[:, time_idx, :])
            
            # Predict with permuted time step
            perm_preds = model.predict(X_permuted)
            perm_auc = roc_auc_score(y_test, perm_preds)
            
            # Store importance
            time_importances[time_idx, repeat] = baseline_auc - perm_auc
            
            # Clean up
            del X_permuted, perm_preds
            gc.collect()
    
    # Calculate mean and std
    mean_time_importances = np.mean(time_importances, axis=1)
    std_time_importances = np.std(time_importances, axis=1)
    
    # Create DataFrame
    time_importance_df = pd.DataFrame({
        'Time Step': np.arange(X_test.shape[1]),
        'Importance': mean_time_importances,
        'Std': std_time_importances
    })
    
    # Plot time step importance
    plt.figure(figsize=(14, 8))
    plt.errorbar(time_importance_df['Time Step'], time_importance_df['Importance'],
                yerr=time_importance_df['Std'], fmt='o-', capsize=5, linewidth=2, markersize=8)
    plt.title('Importance of Time Steps in Sequence')
    plt.xlabel('Time Step Index')
    plt.ylabel('Decrease in AUC when Time Step is Permuted')
    plt.grid(alpha=0.3)
    plt.xticks(time_importance_df['Time Step'])
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "time_step_importance.png"), dpi=300)
    plt.close()
    
    # Save time importance results
    time_importance_df.to_csv(os.path.join(vis_dir, "time_step_importance.csv"), index=False)
    
    # Clean up
    del X_test, y_test
    gc.collect()
    
    return {
        'feature_importance': importance_df.to_dict(orient='records'),
        'time_step_importance': time_importance_df.to_dict(orient='records'),
        'baseline_auc': float(baseline_auc)
    }

tf.config.run_functions_eagerly(True)

def run_improved_pipeline():
    """
    Run the full improved pipeline with all enhancements.
    """
    # Configure TensorFlow
    #configure_tensorflow_memory()
    
    # Paths and configuration
    output_dir = '/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/improved_model'
    data_dir = '/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/ml_data'
    X_file = os.path.join(data_dir, 'X_features.npy')
    y_file = os.path.join(data_dir, 'y_labels.npy')
    metadata_file = os.path.join(data_dir, 'metadata.pkl')
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Load split indices
    with open("zero_curtain_pipeline/modeling/checkpoints/spatiotemporal_split.pkl", "rb") as f:
        split_data = pickle.load(f)
    train_indices = split_data["train_indices"]
    val_indices = split_data["val_indices"]
    test_indices = split_data["test_indices"]
    
    # Load spatial weights
    try:
        with open("zero_curtain_pipeline/modeling/checkpoints/spatial_density.pkl", "rb") as f:
            weights_data = pickle.load(f)
        sample_weights = weights_data["weights"][train_indices]
        sample_weights = sample_weights / np.mean(sample_weights) * len(sample_weights)
    except:
        print("No spatial weights found, using uniform weights")
        sample_weights = None
    
    # Calculate class weights
    y = np.load(y_file, mmap_mode='r')
    train_y = y[train_indices]
    pos_weight = (len(train_y) - np.sum(train_y)) / max(1, np.sum(train_y))
    class_weight = {0: 1.0, 1: pos_weight}
    print(f"Using class weight {pos_weight:.2f} for positive examples")
    
    # Get sample shape
    X = np.load(X_file, mmap_mode='r')
    input_shape = X[train_indices[0]].shape
    
    # Build improved model
    print("Building improved model...")
    model = build_improved_zero_curtain_model(input_shape)
    
    # Print model summary
    model.summary()
    
    # Train model
    print("\nTraining model with improved techniques...")
    model, model_path = improved_resumable_training(
        model=model,
        X_file=X_file,
        y_file=y_file,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        output_dir=output_dir,
        batch_size=256,
        chunk_size=20000,
        epochs_per_chunk=3,
        save_frequency=5,
        class_weight=class_weight,
        start_chunk=0
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    eval_results = evaluate_model_with_visualizations(
        model=model,
        X_file=X_file,
        y_file=y_file,
        test_indices=test_indices,
        output_dir=output_dir,
        metadata_file=metadata_file
    )
    
    # Analyze spatial performance
    print("\nAnalyzing spatial performance patterns...")
    spatial_results = analyze_spatial_performance(
        output_dir=output_dir,
        metadata_file=metadata_file
    )
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    feature_results = analyze_feature_importance(
        model=model,
        X_file=X_file,
        y_file=y_file,
        test_indices=test_indices,
        output_dir=output_dir
    )
    
    # Save all results
    all_results = {
        'evaluation': eval_results,
        'spatial_analysis': spatial_results,
        'feature_importance': feature_results
    }
    
    with open(os.path.join(output_dir, "complete_analysis.json"), "w") as f:
        import json
        
        # Convert numpy values to Python native types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                    return float(obj)
                return json.JSONEncoder.default(self, obj)
        
        json.dump(all_results, f, indent=4, cls=NumpyEncoder)
    
    print("\nImproved pipeline complete!")
    print(f"Results saved to {output_dir}")
    
    return all_results

if __name__ == "__main__":
    run_improved_pipeline()

def analyze_spatial_performance_fixed(output_dir, metadata_file=None):
    """
    Analyze model performance across different geographical regions with robust error handling.
    
    Parameters:
    -----------
    output_dir : str
        Directory with test predictions
    metadata_file : str
        Path to metadata pickle file
        
    Returns:
    --------
    dict
        Spatial analysis results
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json
    import traceback
    
    # Setup logging
    log_file = os.path.join(output_dir, "spatial_analysis_log.txt")
    def log_message(message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)
    
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations", "spatial")
    os.makedirs(vis_dir, exist_ok=True)
    
    log_message("Starting spatial performance analysis")
    
    try:
        # Load predictions with metadata
        pred_file = os.path.join(output_dir, "test_predictions_with_metadata.csv")
        if os.path.exists(pred_file):
            log_message(f"Loading predictions with metadata from {pred_file}")
            predictions = pd.read_csv(pred_file)
        else:
            log_message("No predictions with metadata found, attempting to load separate files")
            # Load separate files and combine
            try:
                preds = np.load(os.path.join(output_dir, "test_predictions.npy"))
                true = np.load(os.path.join(output_dir, "test_true_labels.npy"))
                test_indices = np.load(os.path.join(output_dir, "test_indices.npy"))
                
                log_message(f"Loaded {len(preds)} predictions and {len(true)} true labels")
                
                # Create dataframe
                predictions = pd.DataFrame({
                    'prediction': preds,
                    'true_label': true,
                    'predicted_label': (preds > 0.5).astype(int),
                    'correct': ((preds > 0.5).astype(int) == true).astype(int)
                })
                
                # Load metadata if provided
                if metadata_file and os.path.exists(metadata_file):
                    try:
                        import pickle
                        with open(metadata_file, "rb") as f:
                            metadata = pickle.load(f)
                        
                        log_message(f"Loaded metadata with {len(metadata)} entries")
                        
                        # Add metadata fields
                        for i, idx in enumerate(test_indices):
                            if i < len(predictions) and idx in metadata:
                                meta = metadata[idx]
                                for key, value in meta.items():
                                    if key not in predictions.columns:
                                        predictions[key] = None
                                    predictions.loc[i, key] = value
                        
                        log_message(f"Added metadata to predictions, columns: {list(predictions.columns)}")
                    except Exception as meta_e:
                        log_message(f"Error adding metadata: {str(meta_e)}")
                        log_message(traceback.format_exc())
            except Exception as file_e:
                log_message(f"Error loading prediction files: {str(file_e)}")
                return {'error': 'Failed to load prediction data'}
        
        # Check if we have spatial data
        if 'latitude' not in predictions.columns or 'longitude' not in predictions.columns:
            log_message("No spatial data (latitude/longitude) found in predictions")
            return {'error': 'No spatial data available'}
        
        log_message(f"Predictions dataframe loaded with shape {predictions.shape}")
        
        # Define latitude bands
        def get_lat_band(lat):
            if pd.isna(lat):
                return "Unknown"
            try:
                lat = float(lat)
                if lat < 55:
                    return "<55°N"
                elif lat < 60:
                    return "55-60°N"
                elif lat < 66.5:
                    return "60-66.5°N (Subarctic)"
                elif lat < 70:
                    return "66.5-70°N (Arctic)"
                elif lat < 75:
                    return "70-75°N (Arctic)"
                elif lat < 80:
                    return "75-80°N (Arctic)"
                else:
                    return ">80°N (Arctic)"
            except (ValueError, TypeError):
                return "Unknown"
        
        # Add latitude band
        predictions['lat_band'] = predictions['latitude'].apply(get_lat_band)
        
        # Group by source (site) if available
        site_metrics = None
        if 'source' in predictions.columns:
            log_message("Grouping metrics by source site")
            
            # Define safe aggregation function
            def safe_site_aggregation(x):
                # Make sure we have at least one sample
                if len(x) == 0:
                    return pd.Series({
                        'count': 0,
                        'positive_count': 0,
                        'positive_rate': np.nan,
                        'accuracy': np.nan,
                        'precision': np.nan,
                        'recall': np.nan,
                        'f1': np.nan,
                        'latitude': np.nan,
                        'longitude': np.nan,
                        'lat_band': "Unknown"
                    })
                
                # Calculate metrics safely
                try:
                    # Get valid coordinates if possible
                    lat = pd.to_numeric(x['latitude'], errors='coerce').mean()
                    lon = pd.to_numeric(x['longitude'], errors='coerce').mean()
                    
                    # Handle division by zero
                    pred_sum = x['predicted_label'].sum()
                    true_sum = x['true_label'].sum()
                    
                    # Safe precision calculation
                    if pred_sum > 0:
                        precision = (x['predicted_label'] & x['true_label']).sum() / pred_sum
                    else:
                        precision = np.nan
                    
                    # Safe recall calculation
                    if true_sum > 0:
                        recall = (x['predicted_label'] & x['true_label']).sum() / true_sum
                    else:
                        recall = np.nan
                    
                    # Safe F1 calculation
                    if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = np.nan
                    
                    # Get latitude band if available
                    if 'lat_band' in x.columns and not x['lat_band'].isna().all():
                        lat_band = x['lat_band'].iloc[0]
                    else:
                        lat_band = "Unknown"
                    
                    return pd.Series({
                        'count': len(x),
                        'positive_count': true_sum,
                        'positive_rate': true_sum / len(x),
                        'accuracy': x['correct'].mean(),
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'latitude': lat,
                        'longitude': lon,
                        'lat_band': lat_band
                    })
                except Exception as agg_e:
                    log_message(f"Error during site aggregation: {str(agg_e)}")
                    return pd.Series({
                        'count': len(x),
                        'positive_count': np.nan,
                        'positive_rate': np.nan,
                        'accuracy': np.nan,
                        'precision': np.nan,
                        'recall': np.nan,
                        'f1': np.nan,
                        'latitude': np.nan,
                        'longitude': np.nan,
                        'lat_band': "Unknown"
                    })
            
            # Apply safe aggregation
            try:
                site_metrics = predictions.groupby('source').apply(safe_site_aggregation)
                
                # Filter out sites with too few samples
                site_metrics = site_metrics[site_metrics['count'] >= 10]
                log_message(f"Found {len(site_metrics)} sites with at least 10 samples")
                
                # Save site metrics
                site_metrics.to_csv(os.path.join(vis_dir, "site_metrics.csv"))
            except Exception as site_e:
                log_message(f"Error calculating site metrics: {str(site_e)}")
                log_message(traceback.format_exc())
        else:
            log_message("No 'source' column found, skipping site-level analysis")
        
        # Group by latitude band
        log_message("Grouping metrics by latitude band")
        
        # Define safe band aggregation function
        def safe_band_aggregation(x):
            # Make sure we have at least one sample
            if len(x) == 0:
                return pd.Series({
                    'count': 0,
                    'positive_count': 0,
                    'positive_rate': np.nan,
                    'accuracy': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1': np.nan
                })
            
            # Calculate metrics safely
            try:
                # Handle division by zero
                pred_sum = x['predicted_label'].sum()
                true_sum = x['true_label'].sum()
                
                # Safe precision calculation
                if pred_sum > 0:
                    precision = (x['predicted_label'] & x['true_label']).sum() / pred_sum
                else:
                    precision = np.nan
                
                # Safe recall calculation  
                if true_sum > 0:
                    recall = (x['predicted_label'] & x['true_label']).sum() / true_sum
                else:
                    recall = np.nan
                
                # Safe F1 calculation
                if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = np.nan
                
                return pd.Series({
                    'count': len(x),
                    'positive_count': true_sum,
                    'positive_rate': true_sum / len(x),
                    'accuracy': x['correct'].mean(),
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
            except Exception as agg_e:
                log_message(f"Error during band aggregation: {str(agg_e)}")
                return pd.Series({
                    'count': len(x),
                    'positive_count': np.nan,
                    'positive_rate': np.nan,
                    'accuracy': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1': np.nan
                })
        
        # Apply safe aggregation
        try:
            band_metrics = predictions.groupby('lat_band').apply(safe_band_aggregation)
            
            # Save latitude band metrics
            band_metrics.to_csv(os.path.join(vis_dir, "latitude_band_metrics.csv"))
            log_message(f"Found {len(band_metrics)} latitude bands")
        except Exception as band_e:
            log_message(f"Error calculating latitude band metrics: {str(band_e)}")
            log_message(traceback.format_exc())
            # Create empty DataFrame
            band_metrics = pd.DataFrame(columns=['count', 'positive_count', 'positive_rate', 'accuracy', 
                                                'precision', 'recall', 'f1'])
        
        # Create visualizations
        # 1. Performance by latitude band
        try:
            plt.figure(figsize=(14, 8))
            
            # Order bands by latitude
            ordered_bands = [
                '<55°N', '55-60°N', '60-66.5°N (Subarctic)', 
                '66.5-70°N (Arctic)', '70-75°N (Arctic)', 
                '75-80°N (Arctic)', '>80°N (Arctic)'
            ]
            available_bands = [b for b in ordered_bands if b in band_metrics.index]
            
            if available_bands:
                # Get metrics
                band_data = band_metrics.loc[available_bands]
                
                # Plot metrics
                x = np.arange(len(available_bands))
                width = 0.15
                
                plt.bar(x - 2*width, band_data['accuracy'], width, label='Accuracy', color='#3274A1')
                plt.bar(x - width, band_data['precision'], width, label='Precision', color='#E1812C')
                plt.bar(x, band_data['recall'], width, label='Recall', color='#3A923A')
                plt.bar(x + width, band_data['f1'], width, label='F1', color='#C03D3E')
                plt.bar(x + 2*width, band_data['positive_rate'], width, label='Positive Rate', color='#9372B2')
                
                plt.xlabel('Latitude Band')
                plt.ylabel('Score')
                plt.title('Model Performance by Latitude Band')
                plt.xticks(x, available_bands, rotation=45, ha='right')
                plt.legend(loc='lower right')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, "performance_by_latitude.png"), dpi=300)
            else:
                log_message("No valid latitude bands available for plotting")
            plt.close()
            log_message("Created latitude band performance visualization")
        except Exception as e:
            log_message(f"Error creating latitude band plot: {str(e)}")
            log_message(traceback.format_exc())
        
        # 2. Map visualization if cartopy is available
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            
            if site_metrics is not None and len(site_metrics) > 0:
                plt.figure(figsize=(15, 12))
                
                # Set up the projection
                ax = plt.axes(projection=ccrs.NorthPolarStereo())
                ax.set_extent([-180, 180, 45, 90], ccrs.PlateCarree())
                
                # Add map features
                ax.add_feature(cfeature.LAND, facecolor='lightgray')
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                
                # Add gridlines
                gl = ax.gridlines(draw_labels=True)
                gl.top_labels = False
                gl.right_labels = False
                
                # Filter sites with valid coordinates
                valid_sites = site_metrics.dropna(subset=['latitude', 'longitude'])
                
                # Create scatter plot with safe colormap handling
                try:
                    scatter = ax.scatter(
                        valid_sites['longitude'], 
                        valid_sites['latitude'],
                        transform=ccrs.PlateCarree(),
                        c=valid_sites['f1'].fillna(0),  # Replace NaN with 0 for coloring
                        s=valid_sites['count'] / 5,     # Size by sample count
                        cmap='RdYlGn',
                        vmin=0, vmax=1,
                        alpha=0.8,
                        edgecolor='black',
                        linewidth=0.5
                    )
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
                    cbar.set_label('F1 Score')
                    
                    # Add Arctic Circle
                    theta = np.linspace(0, 2*np.pi, 100)
                    center, radius = [0, 0], 90 - 66.5
                    verts = np.vstack([radius*np.sin(theta), radius*np.cos(theta)]).T
                    circle = plt.Line2D(verts[:, 0], verts[:, 1], color='blue', 
                                      linestyle='--', transform=ax.transData)
                    ax.add_line(circle)
                    
                    plt.title('Spatial Distribution of Model Performance (F1 Score)', fontsize=14)
                    plt.savefig(os.path.join(vis_dir, "spatial_performance_map.png"), dpi=300, bbox_inches='tight')
                except Exception as scatter_e:
                    log_message(f"Error creating scatter plot: {str(scatter_e)}")
                plt.close()
                log_message("Created spatial performance map")
            else:
                log_message("Skipping spatial map due to missing site metrics or no valid sites")
        except ImportError:
            log_message("Cartopy not available for map visualization")
        except Exception as map_e:
            log_message(f"Error creating map visualization: {str(map_e)}")
            log_message(traceback.format_exc())
        
        # 3. Performance vs. sample count (if site metrics available)
        if site_metrics is not None and len(site_metrics) > 0:
            try:
                plt.figure(figsize=(12, 8))
                
                # Filter out NaN values
                valid_sites = site_metrics.dropna(subset=['f1', 'positive_rate'])
                
                if len(valid_sites) > 0:
                    plt.scatter(valid_sites['count'], valid_sites['f1'], 
                              c=valid_sites['positive_rate'], cmap='viridis', 
                              alpha=0.7, s=50, edgecolor='black', linewidth=1)
                    plt.colorbar(label='Positive Rate')
                    plt.xscale('log')
                    plt.xlabel('Number of Samples')
                    plt.ylabel('F1 Score')
                    plt.title('F1 Score vs. Sample Count by Site')
                    plt.grid(alpha=0.3)
                    plt.savefig(os.path.join(vis_dir, "performance_vs_sample_count.png"), dpi=300)
                else:
                    log_message("No valid sites with F1 scores for performance vs sample count plot")
                plt.close()
                log_message("Created performance vs sample count plot")
            except Exception as e:
                log_message(f"Error creating performance vs sample count plot: {str(e)}")
                log_message(traceback.format_exc())
        
        # Return summary
        try:
            summary = {
                'latitude_bands': {
                    'count': len(band_metrics),
                    'bands': band_metrics.to_dict()
                }
            }
            
            if site_metrics is not None and len(site_metrics) > 0:
                try:
                    # Get top/bottom sites safely with error handling
                    best_sites = site_metrics.dropna(subset=['f1']).nlargest(5, 'f1')
                    worst_sites = site_metrics.dropna(subset=['f1']).nsmallest(5, 'f1')
                    
                    summary['sites'] = {
                        'count': len(site_metrics),
                        'best_performing': best_sites[['f1', 'count', 'positive_rate']].to_dict(),
                        'worst_performing': worst_sites[['f1', 'count', 'positive_rate']].to_dict()
                    }
                except Exception as site_summary_e:
                    log_message(f"Error creating site summary: {str(site_summary_e)}")
                    summary['sites'] = {'count': len(site_metrics), 'error': str(site_summary_e)}
            
            # Save summary to file with proper JSON serialization
            with open(os.path.join(vis_dir, "spatial_summary.json"), "w") as f:
                # Handle NumPy types and NaN values
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, (np.integer, np.floating, np.ndarray)):
                            if np.isnan(obj):
                                return None
                            return float(obj)
                        return super(NumpyEncoder, self).default(obj)
                
                json.dump(summary, f, indent=4, cls=NumpyEncoder)
            
            log_message("Spatial analysis complete")
            return summary
        
        except Exception as summary_e:
            log_message(f"Error creating summary: {str(summary_e)}")
            log_message(traceback.format_exc())
            return {'error': 'Error creating summary', 'message': str(summary_e)}
    
    except Exception as e:
        log_message(f"Error in spatial analysis: {str(e)}")
        log_message(traceback.format_exc())
        return {'error': str(e)}

def analyze_feature_importance_fixed(model, X_file, y_file, test_indices, output_dir):
    """
    Analyze feature importance using permutation importance with improved error handling.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    X_file : str
        Path to features numpy file
    y_file : str
        Path to labels numpy file
    test_indices : numpy.ndarray
        Test set indices
    output_dir : str
        Output directory for results
        
    Returns:
    --------
    dict
        Feature importance results
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_auc_score
    import gc
    import traceback
    
    # Create output directory
    vis_dir = os.path.join(output_dir, "visualizations", "features")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, "feature_importance_log.txt")
    def log_message(message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)
    
    log_message("Starting feature importance analysis")
    
    try:
        # Load a smaller subset of test data to save memory
        max_samples = 1000  # Reduced from 5000 to 1000 for better stability
        np.random.seed(42)
        if len(test_indices) > max_samples:
            test_indices_subset = np.random.choice(test_indices, max_samples, replace=False)
        else:
            test_indices_subset = test_indices
        
        log_message(f"Using {len(test_indices_subset)} samples for feature importance analysis")
        
        # Load data
        X = np.load(X_file, mmap_mode='r')
        y = np.load(y_file, mmap_mode='r')
        
        # Load test data with error handling
        try:
            X_test = np.array([X[idx] for idx in test_indices_subset])
            y_test = np.array([y[idx] for idx in test_indices_subset])
        except MemoryError:
            # If memory error, reduce sample size further
            max_samples = 500
            test_indices_subset = np.random.choice(test_indices, max_samples, replace=False)
            log_message(f"Memory error, reducing to {max_samples} samples")
            X_test = np.array([X[idx] for idx in test_indices_subset])
            y_test = np.array([y[idx] for idx in test_indices_subset])
        
        log_message(f"Loaded test data with shape {X_test.shape}")
        
        # Get feature names (modify as needed based on your data)
        feature_names = ['Temperature', 'Temperature Gradient', 'Depth', 'Moisture', 'Moisture Gradient']
        
        # Truncate to actual number of features
        feature_names = feature_names[:X_test.shape[2]]
        log_message(f"Using feature names: {feature_names}")
        
        # Get baseline performance with error handling
        try:
            baseline_preds = model.predict(X_test, verbose=0)
            baseline_auc = roc_auc_score(y_test, baseline_preds)
            log_message(f"Baseline AUC: {baseline_auc:.4f}")
        except Exception as baseline_e:
            log_message(f"Error calculating baseline performance: {str(baseline_e)}")
            log_message(traceback.format_exc())
            return {'error': 'Failed to calculate baseline performance'}
        
        # Perform permutation importance analysis with reduced repeats
        n_repeats = 2  # Reduced from 3 to 2 to save time
        importances = np.zeros((len(feature_names), n_repeats))
        
        for feature_idx in range(len(feature_names)):
            feature_name = feature_names[feature_idx]
            log_message(f"Analyzing importance of feature: {feature_name}")
            
            for repeat in range(n_repeats):
                log_message(f"  Repeat {repeat+1}/{n_repeats}")
                
                try:
                    # Create a copy of the test data
                    X_permuted = X_test.copy()
                    
                    # Permute the feature across all time steps
                    for time_step in range(X_test.shape[1]):
                        X_permuted[:, time_step, feature_idx] = np.random.permutation(X_test[:, time_step, feature_idx])
                    
                    # Predict with permuted feature
                    perm_preds = model.predict(X_permuted, verbose=0)
                    perm_auc = roc_auc_score(y_test, perm_preds)
                    
                    # Store importance (decrease in performance)
                    importances[feature_idx, repeat] = baseline_auc - perm_auc
                    log_message(f"    Decrease in AUC: {importances[feature_idx, repeat]:.4f}")
                    
                    # Clean up
                    del X_permuted, perm_preds
                    gc.collect()
                except Exception as perm_e:
                    log_message(f"    Error during permutation: {str(perm_e)}")
                    # In case of error, assign NaN
                    importances[feature_idx, repeat] = np.nan
        
        # Calculate mean and std of importance, handling NaN values
        mean_importances = np.nanmean(importances, axis=1)
        std_importances = np.nanstd(importances, axis=1)
        
        # Create DataFrame for results
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_importances,
            'Std': std_importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        log_message(f"Feature importance results:\n{importance_df}")
        
        # Plot feature importance
        try:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df, 
                      xerr=importance_df['Std'], palette='viridis')
            plt.title('Feature Importance (Permutation Method)')
            plt.xlabel('Decrease in AUC when Feature is Permuted')
            plt.ylabel('Feature')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "feature_importance.png"), dpi=300)
            plt.close()
            
            # Save results
            importance_df.to_csv(os.path.join(vis_dir, "feature_importance.csv"), index=False)
            log_message("Saved feature importance results")
        except Exception as plot_e:
            log_message(f"Error creating feature importance plot: {str(plot_e)}")
            log_message(traceback.format_exc())
        
        # Analyze temporal patterns in feature importance only...
        if X_test.shape[1] > 1:
            log_message("Analyzing temporal importance patterns")
            
            # We'll analyze which time steps in the sequence are most important
            time_importances = np.zeros((X_test.shape[1], n_repeats))
            
            for time_idx in range(X_test.shape[1]):
                log_message(f"Analyzing importance of time step {time_idx+1}/{X_test.shape[1]}")
                
                for repeat in range(n_repeats):
                    try:
                        # Create a copy of the test data
                        X_permuted = X_test.copy()
                        
                        # Permute all features at this time step
                        X_permuted[:, time_idx, :] = np.random.permutation(X_test[:, time_idx, :])
                        
                        # Predict with permuted time step
                        perm_preds = model.predict(X_permuted, verbose=0)
                        perm_auc = roc_auc_score(y_test, perm_preds)
                        
                        # Store importance
                        time_importances[time_idx, repeat] = baseline_auc - perm_auc
                        
                        # Clean up
                        del X_permuted, perm_preds
                        gc.collect()
                    except Exception as time_e:
                        log_message(f"    Error during time step permutation: {str(time_e)}")
                        time_importances[time_idx, repeat] = np.nan
            
            # Calculate mean and std
            mean_time_importances = np.nanmean(time_importances, axis=1)
            std_time_importances = np.nanstd(time_importances, axis=1)
            
            # Create DataFrame
            time_importance_df = pd.DataFrame({
                'Time_Step': np.arange(X_test.shape[1]),
                'Importance': mean_time_importances,
                'Std': std_time_importances
            })
            
            log_message(f"Time step importance results:\n{time_importance_df}")
            
            # Plot time step importance
            try:
                plt.figure(figsize=(14, 8))
                plt.errorbar(time_importance_df['Time_Step'], time_importance_df['Importance'],
                            yerr=time_importance_df['Std'], fmt='o-', capsize=5, linewidth=2, markersize=8)
                plt.title('Importance of Time Steps in Sequence')
                plt.xlabel('Time Step Index')
                plt.ylabel('Decrease in AUC when Time Step is Permuted')
                plt.grid(alpha=0.3)
                plt.xticks(time_importance_df['Time_Step'])
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, "time_step_importance.png"), dpi=300)
                plt.close()
                
                # Save time importance results
                time_importance_df.to_csv(os.path.join(vis_dir, "time_step_importance.csv"), index=False)
                log_message("Saved time step importance results")
            except Exception as time_plot_e:
                log_message(f"Error creating time step importance plot: {str(time_plot_e)}")
                log_message(traceback.format_exc())
        else:
            log_message("Skipping temporal analysis - insufficient time steps")
            time_importance_df = None
        
        # Clean up
        del X_test, y_test
        gc.collect()
        
        log_message("Feature importance analysis complete")
        
        # Return results dictionary
        result = {
            'feature_importance': importance_df.to_dict(orient='records'),
            'baseline_auc': float(baseline_auc)
        }
        
        # Add time importance if available
        if time_importance_df is not None:
            result['time_step_importance'] = time_importance_df.to_dict(orient='records')
        
        return result
    
    except Exception as e:
        log_message(f"Error in feature importance analysis: {str(e)}")
        log_message(traceback.format_exc())
        return {'error': str(e)}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero Curtain Model - Improved Implementation

A comprehensive model for zero curtain detection in permafrost monitoring,
with specific optimizations for numerical stability and memory management
to address training stalls.

"""

import tensorflow as tf
import numpy as np
import os
import gc
import json
import pickle
import time
import psutil
from datetime import datetime, timedelta
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Dropout
from tensorflow.keras.layers import Reshape, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, Lambda
from tensorflow.keras.layers import ConvLSTM2D, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

# ============================================================================
# PART 1: CORE CLASSES AND UTILITIES
# ============================================================================

class Logger:
    """Custom logger for consistent logging throughout the pipeline"""
    
    def __init__(self, log_dir, log_filename="pipeline_log.txt"):
        """
        Initialize the logger
        
        Parameters:
        -----------
        log_dir : str
            Directory for log files
        log_filename : str
            Name of the log file
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, log_filename)
        
        # Initialize log file with header
        with open(self.log_file, "w") as f:
            f.write(f"=== Zero Curtain Pipeline Log ===\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
    
    def log(self, message, print_to_console=True, level="INFO"):
        """
        Log a message with timestamp
        
        Parameters:
        -----------
        message : str
            Message to log
        print_to_console : bool
            Whether to also print to console
        level : str
            Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
            
        if print_to_console:
            print(log_entry)
            
    def section(self, section_name):
        """Create a visual section separator in the log"""
        self.log("\n" + "="*50)
        self.log(f"SECTION: {section_name}")
        self.log("="*50 + "\n")


class BatchNorm5D(Layer):
    """
    Fixed implementation of BatchNorm5D that properly handles 5D inputs from ConvLSTM2D.
    This implementation correctly manages shapes and serialization.
    """
    def __init__(self, **kwargs):
        super(BatchNorm5D, self).__init__(**kwargs)
        self.bn = BatchNormalization()
        
    def call(self, inputs, training=None):
        # Get the dimensions
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        time_steps = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        channels = input_shape[4]
        
        # Reshape to 4D for BatchNorm by combining batch and time dimensions
        x_reshaped = tf.reshape(inputs, [-1, height, width, channels])
        
        # Apply BatchNorm
        x_bn = self.bn(x_reshaped, training=training)
        
        # Reshape back to 5D
        x_back = tf.reshape(x_bn, [batch_size, time_steps, height, width, channels])
        return x_back
    
    def get_config(self):
        config = super(BatchNorm5D, self).get_config()
        return config


class CyclicLR(tf.keras.callbacks.Callback):
    """
    Cyclical Learning Rate callback for better convergence.
    
    This implementation cycles the learning rate between two boundaries with
    a constant frequency, which can help escape local minima.
    """
    def __init__(
        self,
        base_lr=0.0001,
        max_lr=0.001,
        step_size=2000,
        mode='triangular2',
        gamma=1.0
    ):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        
        if self.mode == 'triangular':
            self.scale_fn = lambda x: 1.
        elif self.mode == 'triangular2':
            self.scale_fn = lambda x: 1 / (2. ** (x - 1))
        elif self.mode == 'exp_range':
            self.scale_fn = lambda x: gamma ** (x)
        else:
            raise ValueError(f"Mode {self.mode} not supported.")
        
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

    def clr(self):
        """Calculate the current learning rate"""
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        
    def on_train_begin(self, logs=None):
        """Initialize learning rate to base_lr at the start of training"""
        logs = logs or {}
        tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)
            
    def on_batch_end(self, batch, logs=None):
        """Update learning rate after each batch"""
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        # Set the learning rate
        lr = self.clr()
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


def configure_tensorflow():
    """
    Configure TensorFlow for optimal performance and memory management
    """
    # Configure memory growth to prevent TF from allocating all GPU memory at once
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
    except Exception as e:
        print(f"Error configuring GPU: {e}")
    
    # Allow TensorFlow to run operations on the CPU if needed
    tf.config.set_soft_device_placement(True)
    
    # Use a more conservative inter/intra op parallelism
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    
    # Manually register custom layer
    tf.keras.utils.get_custom_objects().update({'BatchNorm5D': BatchNorm5D})
    
    # Log available devices
    print("Available devices:")
    print(tf.config.list_physical_devices())


#def f1_score(y_true, y_pred):
#    """Compute F1 score with numerical stability safeguards"""
#    precision = tf.keras.metrics.Precision()(y_true, y_pred)
#    recall = tf.keras.metrics.Recall()(y_true, y_pred)
#    # Add epsilon to avoid division by zero
# return 2 * ((precision * recall) /...

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        # Initialize metrics that will be used to calculate F1
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update the internal metrics
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        
    def result(self):
        # Calculate F1 score from precision and recall
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    def reset_state(self):
        # Reset internal states
        self.precision.reset_state()
        self.recall.reset_state()

def print_memory_usage(logger=None):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    
    message = f"Memory usage: {memory_mb:.1f} MB"
    if logger:
        logger.log(message)
    else:
        print(message)
    
    return memory_mb


def safe_dump_json(data, filepath):
    """
    Safely save data to JSON, handling NumPy types
    
    Parameters:
    -----------
    data : dict
        Data to save
    filepath : str
        Output file path
    """
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif np.isnan(obj):
                return None
            return super(NumpyEncoder, self).default(obj)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=4)
        return True
    except Exception as e:
        print(f"Error saving JSON: {e}")
        try:
            # Try again with pickle
            with open(filepath + '.pkl', 'wb') as f:
                pickle.dump(data, f)
            return True
        except:
            return False

# ============================================================================
# PART 2: MODEL ARCHITECTURE
# ============================================================================

def build_zero_curtain_model(input_shape, logger=None):
    """
    Build the improved zero curtain detection model with stability enhancements
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (time_steps, features)
    logger : Logger, optional
        Logger instance for output
        
    Returns:
    --------
    tf.keras.Model
        Compiled model
    """
    if logger:
        logger.log(f"Building model with input shape: {input_shape}")
    
    # Hyperparameters - carefully tuned for stability
    reg_strength = 0.00001  # Extremely reduced to prevent vanishing gradients
    dropout_rate = 0.2  # Reduced to prevent training stalls
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Add BatchNormalization to normalize input data
    x = BatchNormalization()(inputs)
    
    # Reshape for ConvLSTM (add spatial dimension)
    x = Reshape((input_shape[0], 1, 1, input_shape[1]))(x)
    
    # ConvLSTM layer with reduced regularization
    convlstm = ConvLSTM2D(
        filters=32,  # Reduced from 64 to decrease memory usage
        kernel_size=(3, 1),
        padding='same',
        return_sequences=True,
        activation='tanh',
        recurrent_dropout=dropout_rate,
        kernel_regularizer=l2(reg_strength)
    )(x)
    
    # Use BatchNorm5D for the ConvLSTM output
    convlstm = BatchNorm5D()(convlstm)
    
    # Reshape back to (sequence_length, features)
    convlstm = Reshape((input_shape[0], 32))(convlstm)
    
    # Add positional encoding for transformer
    def positional_encoding(length, depth):
        positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
        depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth
        
        angle_rates = 1 / tf.pow(10000.0, depths)
        angle_rads = positions * angle_rates
        
        # Only use sin to ensure output depth matches input depth
        pos_encoding = tf.sin(angle_rads)
        
        # Add batch dimension
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        
        return pos_encoding
    
    # Add positional encoding
    pos_encoding = positional_encoding(input_shape[0], 32)
    transformer_input = convlstm + pos_encoding
    
    # Add BatchNormalization before transformer
    transformer_input = BatchNormalization()(transformer_input)
    
    # Simplified transformer encoder block with reduced complexity
    def transformer_encoder(x, num_heads=4, key_dim=32, ff_dim=64):
        # Multi-head attention with regularization
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,
            dropout=0.1,  # Add dropout in attention
            kernel_regularizer=l2(reg_strength)
        )(x, x)
        
        # Skip connection 1 with dropout
        x1 = Add()([attention_output, x])
        x1 = LayerNormalization(epsilon=1e-6)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(dropout_rate)(x1)
        
        # Feed-forward network with regularization
        ff_output = Dense(ff_dim, activation='relu', kernel_regularizer=l2(reg_strength))(x1)
        ff_output = BatchNormalization()(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        ff_output = Dense(32, kernel_regularizer=l2(reg_strength))(ff_output)
        ff_output = BatchNormalization()(ff_output)
        
        # Skip connection 2
        x2 = Add()([ff_output, x1])
        return LayerNormalization(epsilon=1e-6)(x2)
    
    # Apply transformer encoder
    transformer_output = transformer_encoder(transformer_input)
    
    # Parallel CNN paths for multi-scale feature extraction (with regularization)
    cnn_1 = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu',
                  kernel_regularizer=l2(reg_strength))(inputs)
    cnn_1 = BatchNormalization()(cnn_1)
    cnn_1 = Dropout(dropout_rate/2)(cnn_1)
    
    cnn_2 = Conv1D(filters=16, kernel_size=5, padding='same', activation='relu',
                  kernel_regularizer=l2(reg_strength))(inputs)
    cnn_2 = BatchNormalization()(cnn_2)
    cnn_2 = Dropout(dropout_rate/2)(cnn_2)
    
    # Simplified VAE components
#    def sampling(args):
#        z_mean, z_log_var = args
#        batch = tf.shape(z_mean)[0]
#        dim = tf.shape(z_mean)[1]
#        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def sampling(args):
        """
        Reparameterization trick by sampling from an isotropic unit Gaussian.
        
        Parameters:
        -----------
        args : list
            List containing mean and log of variance of the latent space [z_mean, z_log_var]
            
        Returns:
        --------
        z : tf.Tensor
            Sampled latent vector
        """
        z_mean, z_log_var = args
        
        # By default, random_normal has mean = 0 and std = 1.0
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        
        # Reparameterization trick
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    # Global temporal features
    global_max = GlobalMaxPooling1D()(transformer_output)
    global_avg = GlobalAveragePooling1D()(transformer_output)
    
    # VAE encoding with extremely reduced regularization
    z_concat = Concatenate()([global_max, global_avg])
    z_concat = BatchNormalization()(z_concat)
    
    #z_mean = Dense(16, kernel_regularizer=l2(reg_strength))(z_concat)
    #z_log_var = Dense(16, kernel_regularizer=l2(reg_strength))(z_concat)
    #z = Lambda(sampling, output_shape=(16,))([z_mean, z_log_var])
    #z = Lambda(sampling)([z_mean, z_log_var])
    z_mean = Dense(16, kernel_regularizer=l2(reg_strength))(z_concat)
    z_log_var = Dense(16, kernel_regularizer=l2(reg_strength))(z_concat)
    z = Lambda(sampling)([z_mean, z_log_var])
    
    # Combine all features
    merged_features = Concatenate()(
        [
            GlobalMaxPooling1D()(cnn_1),
            GlobalMaxPooling1D()(cnn_2),
            global_max,
            global_avg,
            z
        ]
    )
    
    # Add BatchNorm to merged features
    merged_features = BatchNormalization()(merged_features)
    
    # Final classification layers with reduced complexity
    x = Dense(64, activation='relu', kernel_regularizer=l2(reg_strength))(merged_features)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(reg_strength))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_strength))(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # CRITICAL: Drastically reduce VAE loss weight to prevent training instability
    kl_loss = -0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
    )
    model.add_loss(0.00005 * kl_loss)  # Extreme reduction of KL loss weight
    
    # Compile model with safeguards
#    model.compile(
#        optimizer=Adam(
#            learning_rate=0.0001,  # Reduced learning rate
#            clipvalue=0.5,  # Gradient clipping
#            epsilon=1e-7  # Numerical stability
#        ),
#        loss='binary_crossentropy',  # Standard loss for binary classification
#        metrics=[
#            'accuracy',
#            tf.keras.metrics.AUC(name='auc', num_thresholds=200),
#            tf.keras.metrics.Precision(name='precision'),
#            tf.keras.metrics.Recall(name='recall'),
#            f1_score  # Custom F1 score implementation
#        ]
#    )

    model.compile(
        optimizer=Adam(
            learning_rate=0.0001,
            clipvalue=0.5,
            epsilon=1e-7
        ),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', num_thresholds=200),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    if logger:
        # Log model summary
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        for line in summary_lines:
            logger.log(line, print_to_console=False)
        
        logger.log(f"Model built with {model.count_params():,} parameters")
    
    return model


def get_training_callbacks(output_dir, monitor='val_auc', mode='max', logger=None):
    """
    Create a set of callbacks for training with safeguards
    
    Parameters:
    -----------
    output_dir : str
        Directory to save checkpoints
    monitor : str
        Metric to monitor for early stopping/model saving
    mode : str
        'max' or 'min' for the monitor metric
    logger : Logger, optional
        Logger instance
        
    Returns:
    --------
    list
        List of Keras callbacks
    """
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if logger:
        logger.log(f"Setting up callbacks with checkpoint dir: {checkpoint_dir}")
    
    callbacks = [
        # Early stopping with longer patience
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=15,
            restore_best_weights=True,
            mode=mode,
            min_delta=0.001,
            verbose=1
        ),
        
        # Reduce learning rate with shorter patience
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode=mode,
            verbose=1
        ),
        
        # Save model checkpoints
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.h5"),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=True,  # Only save weights to avoid serialization issues
            verbose=1
        ),
        
        # Cyclical learning rate for better convergence
        CyclicLR(
            base_lr=0.00005,  # Lower base LR
            max_lr=0.0005,    # Lower max LR
            step_size=1000,
            mode='triangular2'
        ),
        
        # Terminate on NaN to prevent wasting time on diverged training
        tf.keras.callbacks.TerminateOnNaN(),
        
        # Clear memory after each epoch
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect()
        )
    ]
    
    return callbacks
    
# ============================================================================
# PART 3: TRAINING FUNCTIONS
# ============================================================================

def load_data_batch(X_mmap, y_mmap, indices, start_idx, batch_size, logger=None):
    """
    Safely load a batch of data from memory-mapped arrays
    
    Parameters:
    -----------
    X_mmap : numpy.memmap
        Memory-mapped features array
    y_mmap : numpy.memmap
        Memory-mapped labels array
    indices : list or numpy.ndarray
        Indices to sample from
    start_idx : int
        Starting index in the indices list
    batch_size : int
        Number of samples to load
    logger : Logger, optional
        Logger instance
        
    Returns:
    --------
    tuple
        (X_batch, y_batch) - batch of features and labels
    """
    end_idx = min(start_idx + batch_size, len(indices))
    batch_indices = indices[start_idx:end_idx]
    
    try:
        X_batch = np.array([X_mmap[idx] for idx in batch_indices])
        y_batch = np.array([y_mmap[idx] for idx in batch_indices])
        
        if logger:
            logger.log(f"Loaded batch {start_idx}-{end_idx} with shape X={X_batch.shape}, y={y_batch.shape}",
                       print_to_console=False)
        
        return X_batch, y_batch
    except Exception as e:
        if logger:
            logger.log(f"Error loading batch {start_idx}-{end_idx}: {e}", level="ERROR")
        
        # Try again with smaller batch size
        if batch_size > 10:
            half_batch = batch_size // 2
            X1, y1 = load_data_batch(X_mmap, y_mmap, indices, start_idx, half_batch, logger)
            X2, y2 = load_data_batch(X_mmap, y_mmap, indices, start_idx + half_batch,
                                    batch_size - half_batch, logger)
            return np.concatenate([X1, X2]), np.concatenate([y1, y2])
        else:
            # Small batch still failed - load one by one
            X_list = []
            y_list = []
            for idx in batch_indices:
                try:
                    X_list.append(X_mmap[idx])
                    y_list.append(y_mmap[idx])
                except Exception as e2:
                    if logger:
                        logger.log(f"Error loading sample {idx}: {e2}", level="ERROR")
            
            if not X_list:
                raise ValueError(f"Could not load any samples from batch {start_idx}-{end_idx}")
                
            return np.array(X_list), np.array(y_list)


def train_on_chunk(model, X_mmap, y_mmap, train_indices, val_data, chunk_idx, chunk_size,
                  output_dir, epochs=3, batch_size=64, logger=None):
    """
    Train model on a single chunk of data with robust error handling
    
    Parameters:
    -----------
    model : tf.keras.Model
        Model to train
    X_mmap : numpy.memmap
        Memory-mapped features
    y_mmap : numpy.memmap
        Memory-mapped labels
    train_indices : numpy.ndarray
        Indices for training set
    val_data : tuple
        (X_val, y_val) validation data
    chunk_idx : int
        Current chunk index
    chunk_size : int
        Size of each chunk
    output_dir : str
        Output directory for saving model
    epochs : int
        Number of epochs to train per chunk
    batch_size : int
        Batch size for training
    logger : Logger
        Logger instance
        
    Returns:
    --------
    dict
        Training history
    """
    # Get chunk indices
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, len(train_indices))
    chunk_indices = train_indices[start_idx:end_idx]
    
    if logger:
        logger.log(f"\n{'='*50}")
        logger.log(f"Processing chunk {chunk_idx+1} with {len(chunk_indices)} samples")
        memory_before = print_memory_usage(logger)
    
    # Force garbage collection before loading new data
    gc.collect()
    
    # Load data in manageable batches
    X_chunks = []
    y_chunks = []
    
    # Use relatively small batch size for loading to prevent memory spikes
    load_batch_size = 1000
    
    for i in range(0, len(chunk_indices), load_batch_size):
        try:
            X_batch, y_batch = load_data_batch(X_mmap, y_mmap, chunk_indices, i, load_batch_size, logger)
            X_chunks.append(X_batch)
            y_chunks.append(y_batch)
            del X_batch, y_batch
            gc.collect()
        except Exception as e:
            if logger:
                logger.log(f"Warning: Skipping batch {i}-{i+load_batch_size} due to error: {e}", level="WARNING")
    
    if not X_chunks:
        if logger:
            logger.log(f"Error: Could not load any data for chunk {chunk_idx+1}", level="ERROR")
        return None
    
    # Combine batches
    try:
        X_data = np.concatenate(X_chunks)
        y_data = np.concatenate(y_chunks)
        
        if logger:
            logger.log(f"Loaded chunk data with shape X={X_data.shape}, y={y_data.shape}")
            memory_after_load = print_memory_usage(logger)
    except Exception as e:
        if logger:
            logger.log(f"Error combining data batches: {e}", level="ERROR")
        return None
    
    # Get callbacks with checkpoints specific to this chunk
    callbacks = get_training_callbacks(
        output_dir=os.path.join(output_dir, f"chunk_{chunk_idx+1}"),
        logger=logger
    )
    
    # Add custom TensorBoard callback only for this chunk
    log_dir = os.path.join(output_dir, "logs", f"chunk_{chunk_idx+1}")
    os.makedirs(log_dir, exist_ok=True)
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    )
    callbacks.append(tb_callback)
    
    # Train for a few epochs
    history = None
    
    # Determine effective batch size based on available memory
    effective_batch_size = min(batch_size, 64)  # Cap at 64 for stability
    
    try:
        if logger:
            logger.log(f"Training for {epochs} epochs with batch size {effective_batch_size}")
            
        # Training with safeguards
        history = model.fit(
            X_data, y_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=effective_batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save weights after training
        weights_path = os.path.join(output_dir, "checkpoints", f"model_weights_chunk_{chunk_idx+1}")
        model.save_weights(weights_path)
        
        if logger:
            logger.log(f"Saved model weights to {weights_path}")
            memory_after_train = print_memory_usage(logger)
    except Exception as e:
        if logger:
            logger.log(f"Error during training: {e}", level="ERROR")
            
        # Try to save model in error state
        try:
            error_weights_path = os.path.join(output_dir, "checkpoints", f"error_weights_chunk_{chunk_idx+1}")
            model.save_weights(error_weights_path)
            if logger:
                logger.log(f"Saved weights in error state to {error_weights_path}")
        except:
            if logger:
                logger.log("Could not save weights in error state", level="ERROR")
    
    # Clean up
    del X_chunks, y_chunks, X_data, y_data
    gc.collect()
    
    return history

def progressive_training(X_file, y_file, train_indices, val_indices, output_dir,
                        chunk_size=5000, rebuild_interval=5, logger=None):
    """
    Train model with regular rebuilds to prevent memory issues and stalls
    
    Parameters:
    -----------
    X_file : str
        Path to features file
    y_file : str
        Path to labels file
    train_indices : numpy.ndarray
        Indices for training
    val_indices : numpy.ndarray
        Indices for validation
    output_dir : str
        Output directory
    chunk_size : int
        Size of each data chunk
    rebuild_interval : int
        Rebuild model every N chunks
    logger : Logger
        Logger instance
        
    Returns:
    --------
    str
        Path to best model weights
    """
    # Total chunks to process
    num_chunks = int(np.ceil(len(train_indices) / chunk_size))
    
    if logger:
        logger.log(f"Starting progressive training with {num_chunks} chunks")
        logger.log(f"Will rebuild model every {rebuild_interval} chunks")
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create progress tracking file
    progress_file = os.path.join(output_dir, "training_progress.txt")
    
    # Check for existing progress
    start_chunk = 0
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                start_chunk = int(f.read().strip())
                if logger:
                    logger.log(f"Resuming training from chunk {start_chunk+1}")
        except:
            if logger:
                logger.log("Could not read progress file, starting from beginning")
    
    # Open data files in read-only mode
    X_mmap = np.load(X_file, mmap_mode='r')
    y_mmap = np.load(y_file, mmap_mode='r')
    
    # Load a sample to get input shape
    sample_shape = X_mmap[train_indices[0]].shape
    if logger:
        logger.log(f"Sample shape: {sample_shape}")
    
    # Prepare validation data (small subset for memory efficiency)
    val_size = min(1000, len(val_indices))
    val_indices_subset = val_indices[:val_size]
    X_val, y_val = load_data_batch(X_mmap, y_mmap, val_indices_subset, 0, val_size, logger)
    if logger:
        logger.log(f"Loaded validation data with shape X={X_val.shape}, y={y_val.shape}")
    
    # Initialize history
    all_history = []
    
    # Main training loop with rebuild checkpoints
    start_time = time.time()
    best_weights_path = None
    
    # Always build the model at the start, before training begins
    # This fixes the "referenced before assignment" error
    if logger:
        logger.log(f"Building initial model")
    
    # Build fresh model
    model = build_zero_curtain_model(sample_shape, logger)
    
    # Load weights if resuming
    if start_chunk > 0:
        # Find most recent weights
        prev_chunk = start_chunk - 1
        while prev_chunk >= 0:
            weights_path = os.path.join(checkpoint_dir, f"model_weights_chunk_{prev_chunk+1}")
            if os.path.exists(weights_path) or os.path.exists(weights_path + ".index"):
                try:
                    model.load_weights(weights_path)
                    if logger:
                        logger.log(f"Loaded weights from chunk {prev_chunk+1}")
                    break
                except:
                    if logger:
                        logger.log(f"Failed to load weights from {weights_path}", level="WARNING")
            prev_chunk -= 1
    
    for chunk_idx in range(start_chunk, num_chunks):
        # Rebuild model every N chunks to prevent memory issues
        if chunk_idx > start_chunk and chunk_idx % rebuild_interval == 0:
            if logger:
                logger.log(f"Rebuilding model at chunk {chunk_idx+1}", level="INFO")
            
            # Clear TensorFlow session and memory
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Build fresh model
            model = build_zero_curtain_model(sample_shape, logger)
            
            # Load weights from previous chunk
            prev_chunk = chunk_idx - 1
            weights_path = os.path.join(checkpoint_dir, f"model_weights_chunk_{prev_chunk+1}")
            if os.path.exists(weights_path) or os.path.exists(weights_path + ".index"):
                try:
                    model.load_weights(weights_path)
                    if logger:
                        logger.log(f"Loaded weights from chunk {prev_chunk+1}")
                except Exception as e:
                    if logger:
                        logger.log(f"Failed to load weights: {e}", level="WARNING")
        
        # Train on current chunk
        chunk_history = train_on_chunk(
            model=model,
            X_mmap=X_mmap,
            y_mmap=y_mmap,
            train_indices=train_indices,
            val_data=(X_val, y_val),
            chunk_idx=chunk_idx,
            chunk_size=chunk_size,
            output_dir=output_dir,
            epochs=3,  # Reduced epochs per chunk
            batch_size=64,
            logger=logger
        )
        
        # Store history
        if chunk_history is not None:
            all_history.append({
                'chunk': chunk_idx + 1,
                'history': chunk_history.history
            })
            
            # Save history to file
            try:
                with open(os.path.join(output_dir, "training_history.json"), "w") as f:
                    json.dump(all_history, f, indent=4, cls=NumpyEncoder)
            except Exception as e:
                if logger:
                    logger.log(f"Error saving history: {e}", level="WARNING")
                
                # Try pickle as fallback
                try:
                    with open(os.path.join(output_dir, "training_history.pkl"), "wb") as f:
                        pickle.dump(all_history, f)
                except:
                    pass
        
        # Update progress file
        with open(progress_file, "w") as f:
            f.write(str(chunk_idx + 1))
        
        # Update best weights path
        weights_path = os.path.join(checkpoint_dir, f"model_weights_chunk_{chunk_idx+1}")
        if os.path.exists(weights_path) or os.path.exists(weights_path + ".index"):
            best_weights_path = weights_path
        
        # Report progress
        elapsed = time.time() - start_time
        avg_time_per_chunk = elapsed / (chunk_idx - start_chunk + 1) if chunk_idx > start_chunk else elapsed
        remaining = avg_time_per_chunk * (num_chunks - chunk_idx - 1)
        
        if logger:
            logger.log(f"Completed chunk {chunk_idx+1}/{num_chunks}")
            logger.log(f"Estimated remaining time: {timedelta(seconds=int(remaining))}")
    
    # Save final model
    try:
        final_weights_path = os.path.join(checkpoint_dir, "final_model_weights")
        model.save_weights(final_weights_path)
        if logger:
            logger.log(f"Saved final model weights to {final_weights_path}")
        best_weights_path = final_weights_path
    except Exception as e:
        if logger:
            logger.log(f"Error saving final model: {e}", level="ERROR")
    
    # Clean up
    del X_val, y_val
    tf.keras.backend.clear_session()
    gc.collect()
    
    if logger:
        logger.log("Training complete!")
    
    return best_weights_path

#def progressive_training(X_file, y_file, train_indices, val_indices, output_dir,
#                        chunk_size=5000, rebuild_interval=5, logger=None):
#    """
#    Train model with regular rebuilds to prevent memory issues and stalls
#    
#    Parameters:
#    -----------
#    X_file : str
#        Path to features file
#    y_file : str
#        Path to labels file
#    train_indices : numpy.ndarray
#        Indices for training
#    val_indices : numpy.ndarray
#        Indices for validation
#    output_dir : str
#        Output directory
#    chunk_size : int
#        Size of each data chunk
#    rebuild_interval : int
#        Rebuild model every N chunks
#    logger : Logger
#        Logger instance
#        
#    Returns:
#    --------
#    str
#        Path to best model weights
#    """
#    # Total chunks to process
#    num_chunks = int(np.ceil(len(train_indices) / chunk_size))
#    
#    if logger:
#        logger.log(f"Starting progressive training with {num_chunks} chunks")
#        logger.log(f"Will rebuild model every {rebuild_interval} chunks")
#    
#    # Create checkpoint directory
#    checkpoint_dir = os.path.join(output_dir, "checkpoints")
#    os.makedirs(checkpoint_dir, exist_ok=True)
#    
#    # Create progress tracking file
#    progress_file = os.path.join(output_dir, "training_progress.txt")
#    
#    # Check for existing progress
#    start_chunk = 0
#    if os.path.exists(progress_file):
#        try:
#            with open(progress_file, "r") as f:
#                start_chunk = int(f.read().strip())
#                if logger:
#                    logger.log(f"Resuming training from chunk {start_chunk+1}")
#        except:
#            if logger:
#                logger.log("Could not read progress file, starting from beginning")
#    
#    # Open data files in read-only mode
#    X_mmap = np.load(X_file, mmap_mode='r')
#    y_mmap = np.load(y_file, mmap_mode='r')
#    
#    # Load a sample to get input shape
#    sample_shape = X_mmap[train_indices[0]].shape
#    if logger:
#        logger.log(f"Sample shape: {sample_shape}")
#    
#    # Prepare validation data (small subset for memory efficiency)
#    val_size = min(1000, len(val_indices))
#    val_indices_subset = val_indices[:val_size]
#    X_val, y_val = load_data_batch(X_mmap, y_mmap, val_indices_subset, 0, val_size, logger)
#    if logger:
#        logger.log(f"Loaded validation data with shape X={X_val.shape}, y={y_val.shape}")
#    
#    # Initialize history
#    all_history = []
#    
#    # Main training loop with rebuild checkpoints
#    start_time = time.time()
#    best_weights_path = None
#    
#    for chunk_idx in range(start_chunk, num_chunks):
#        # Rebuild model every N chunks to prevent memory issues
#        if chunk_idx % rebuild_interval == 0:
#            if logger:
#                logger.log(f"Rebuilding model at chunk {chunk_idx+1}", level="INFO")
#            
#            # Clear TensorFlow session and memory
#            tf.keras.backend.clear_session()
#            gc.collect()
#            
#            # Build fresh model
#            model = build_zero_curtain_model(sample_shape, logger)
#            
#            # Load weights if not at the beginning
#            if chunk_idx > 0:
#                # Find most recent weights
#                prev_chunk = chunk_idx - 1
#                while prev_chunk >= 0:
#                    weights_path = os.path.join(checkpoint_dir, f"model_weights_chunk_{prev_chunk+1...
#                    if os.path.exists(weights_path) or os.path.exists(weights_path + ".index"):
#                        try:
#                            model.load_weights(weights_path)
#                            if logger:
#                                logger.log(f"Loaded weights from chunk {prev_chunk+1}")
#                            break
#                        except:
#                            if logger:
#                                logger.log(f"Failed to load weights from {weights_path}", level="WA...
#                    prev_chunk -= 1
#        
#        # Train on current chunk
#        chunk_history = train_on_chunk(
#            model=model,
#            X_mmap=X_mmap,
#            y_mmap=y_mmap,
#            train_indices=train_indices,
#            val_data=(X_val, y_val),
#            chunk_idx=chunk_idx,
#            chunk_size=chunk_size,
#            output_dir=output_dir,
#            epochs=3,  # Reduced epochs per chunk
#            batch_size=64,
#            logger=logger
#        )
#        
#        # Store history
#        if chunk_history is not None:
#            all_history.append({
#                'chunk': chunk_idx + 1,
#                'history': chunk_history.history
#            })
#            
#            # Save history to file
#            try:
#                with open(os.path.join(output_dir, "training_history.json"), "w") as f:
#                    json.dump(all_history, f, indent=4, cls=NumpyEncoder)
#            except Exception as e:
#                if logger:
#                    logger.log(f"Error saving history: {e}", level="WARNING")
#                
#                # Try pickle as fallback
#                try:
#                    with open(os.path.join(output_dir, "training_history.pkl"), "wb") as f:
#                        pickle.dump(all_history, f)
#                except:
#                    pass
#        
#        # Update progress file
#        with open(progress_file, "w") as f:
#            f.write(str(chunk_idx + 1))
#        
#        # Update best weights path
#        weights_path = os.path.join(checkpoint_dir, f"model_weights_chunk_{chunk_idx+1}")
#        if os.path.exists(weights_path) or os.path.exists(weights_path + ".index"):
#            best_weights_path = weights_path
#        
#        # Report progress
#        elapsed = time.time() - start_time
# avg_time_per_chunk = elapsed / (chunk_idx - start_chunk...
#        remaining = avg_time_per_chunk * (num_chunks - chunk_idx - 1)
#        
#        if logger:
#            logger.log(f"Completed chunk {chunk_idx+1}/{num_chunks}")
#            logger.log(f"Estimated remaining time: {timedelta(seconds=int(remaining))}")
#    
#    # Save final model
#    try:
#        final_weights_path = os.path.join(checkpoint_dir, "final_model_weights")
#        model.save_weights(final_weights_path)
#        if logger:
#            logger.log(f"Saved final model weights to {final_weights_path}")
#        best_weights_path = final_weights_path
#    except Exception as e:
#        if logger:
#            logger.log(f"Error saving final model: {e}", level="ERROR")
#    
#    # Clean up
#    del X_val, y_val
#    tf.keras.backend.clear_session()
#    gc.collect()
#    
#    if logger:
#        logger.log("Training complete!")
#    
#    return best_weights_path


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif np.isnan(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

# ============================================================================
# PART 4: EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, X_file, y_file, test_indices, output_dir, metadata_file=None, logger=None):
    """
    Comprehensive model evaluation with visualizations and error handling
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    X_file : str
        Path to features file
    y_file : str
        Path to labels file
    test_indices : numpy.ndarray
        Test set indices
    output_dir : str
        Output directory for results
    metadata_file : str, optional
        Path to metadata file
    logger : Logger, optional
        Logger instance
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Create output directories
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    if logger:
        logger.log(f"Starting evaluation on {len(test_indices)} test samples")
    
    # Load metadata if available
    metadata = None
    if metadata_file and os.path.exists(metadata_file):
        try:
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
            if logger:
                logger.log(f"Loaded metadata with {len(metadata)} entries")
        except Exception as e:
            if logger:
                logger.log(f"Error loading metadata: {e}", level="WARNING")
    
    # Load memory-mapped data
    X = np.load(X_file, mmap_mode='r')
    y = np.load(y_file, mmap_mode='r')
    
    # Process test data in small batches for memory efficiency
    batch_size = 50  # Very small batch size for stable processing
    num_batches = int(np.ceil(len(test_indices) / batch_size))
    
    if logger:
        logger.log(f"Processing test data in {num_batches} batches of size {batch_size}")
    
    all_preds = []
    all_true = []
    all_meta = []
    
    # Save test indices for reference
    np.save(os.path.join(output_dir, "test_indices.npy"), test_indices)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(test_indices))
        batch_indices = test_indices[start_idx:end_idx]
        
        if logger and batch_idx % 10 == 0:
            logger.log(f"Processing test batch {batch_idx+1}/{num_batches}")
        
        try:
            # Load batch safely
            X_batch, y_batch = load_data_batch(X, y, batch_indices, 0, len(batch_indices), logger)
            
            # Add metadata if available
            if metadata:
                batch_meta = []
                for idx in batch_indices:
                    if idx in metadata:
                        batch_meta.append(metadata[idx])
                    else:
                        batch_meta.append({})
                all_meta.extend(batch_meta)
            
            # Predict with error handling
            try:
                batch_preds = model.predict(X_batch, verbose=0)
            except Exception as e:
                if logger:
                    logger.log(f"Error in prediction for batch {batch_idx+1}: {e}", level="ERROR")
                
                # Try with even smaller sub-batches
                batch_preds = []
                sub_size = 10
                for i in range(0, len(X_batch), sub_size):
                    end_i = min(i + sub_size, len(X_batch))
                    sub_preds = model.predict(X_batch[i:end_i], verbose=0)
                    batch_preds.append(sub_preds)
                
                batch_preds = np.concatenate(batch_preds)
            
            # Store results
            all_preds.extend(batch_preds.flatten())
            all_true.extend(y_batch)
            
            # Clean up
            del X_batch, y_batch, batch_preds
            gc.collect()
            
        except Exception as e:
            if logger:
                logger.log(f"Error processing batch {batch_idx+1}: {e}", level="ERROR")
                
            # Skip this batch and continue
            continue
    
    if logger:
        logger.log(f"Completed predictions for {len(all_preds)} test samples")
    
    # Save raw predictions
    np.save(os.path.join(output_dir, "test_predictions.npy"), np.array(all_preds))
    np.save(os.path.join(output_dir, "test_true_labels.npy"), np.array(all_true))
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    # Calculate metrics
    try:
        # Binary predictions
        all_preds_binary = (all_preds > 0.5).astype(int)
        
        # Classification report
        report = classification_report(all_true, all_preds_binary, output_dict=True)
        report_str = classification_report(all_true, all_preds_binary)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_true, all_preds_binary)
        
        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(all_true, all_preds)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(all_true, all_preds)
        avg_precision = average_precision_score(all_true, all_preds)
        
        if logger:
            logger.log("\nClassification Report:")
            logger.log(report_str)
            logger.log(f"\nROC AUC: {roc_auc:.4f}")
            logger.log(f"Average Precision: {avg_precision:.4f}")
        
        # Save results as text
        with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
            f.write("Classification Report:\n")
            f.write(report_str)
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(conf_matrix))
            f.write(f"\n\nROC AUC: {roc_auc:.4f}")
            f.write(f"\nAverage Precision: {avg_precision:.4f}")
        
        # Create visualizations
        try:
            # 1. Confusion Matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['Negative', 'Positive'],
                      yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "confusion_matrix.png"), dpi=300)
            plt.close()
            
            # 2. ROC Curve
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(vis_dir, "roc_curve.png"), dpi=300)
            plt.close()
            
            # 3. Precision-Recall Curve
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="upper right")
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(vis_dir, "precision_recall_curve.png"), dpi=300)
            plt.close()
            
            # 4. Score distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(all_preds[all_true == 0], bins=50, alpha=0.5, label='Negative Class', color='blue')
            sns.histplot(all_preds[all_true == 1], bins=50, alpha=0.5, label='Positive Class', color='red')
            plt.xlabel('Prediction Score')
            plt.ylabel('Count')
            plt.title('Distribution of Prediction Scores by Class')
            plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(vis_dir, "score_distribution.png"), dpi=300)
            plt.close()
            
            if logger:
                logger.log("Created all visualization plots")
        except Exception as e:
            if logger:
                logger.log(f"Error creating visualization plots: {e}", level="ERROR")
        
        # Save predictions with metadata
        if metadata and all_meta:
            try:
                # Create DataFrame with metadata
                results_data = []
                
                for i in range(len(all_preds)):
                    if i < len(all_meta):
                        result = {
                            'prediction': float(all_preds[i]),
                            'true_label': int(all_true[i]),
                            'predicted_label': int(all_preds_binary[i]),
                            'correct': int(all_preds_binary[i] == all_true[i])
                        }
                        
                        # Add metadata fields
                        meta = all_meta[i]
                        for key, value in meta.items():
                            # Convert NumPy types to native Python types
                            if hasattr(value, 'dtype'):
                                if np.issubdtype(value.dtype, np.integer):
                                    value = int(value)
                                elif np.issubdtype(value.dtype, np.floating):
                                    value = float(value)
                            result[key] = value
                        
                        results_data.append(result)
                
                # Create DataFrame
                results_df = pd.DataFrame(results_data)
                
                # Save to CSV
                results_df.to_csv(os.path.join(output_dir, "test_predictions_with_metadata.csv"), index=False)
                
                if logger:
                    logger.log(f"Saved predictions with metadata to CSV, shape: {results_df.shape}")
            except Exception as e:
                if logger:
                    logger.log(f"Error saving predictions with metadata: {e}", level="ERROR")
                
                # Try simpler format
                try:
                    simple_df = pd.DataFrame({
                        'prediction': all_preds,
                        'true_label': all_true,
                        'predicted_label': all_preds_binary
                    })
                    simple_df.to_csv(os.path.join(output_dir, "simple_predictions.csv"), index=False)
                    
                    if logger:
                        logger.log("Saved simplified predictions to CSV")
                except:
                    pass
        
        # Return metrics
        metrics = {
            'accuracy': float(report['accuracy']),
            'precision': float(report['1']['precision']) if '1' in report else 0,
            'recall': float(report['1']['recall']) if '1' in report else 0,
            'f1_score': float(report['1']['f1-score']) if '1' in report else 0,
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision),
            'num_samples': len(all_true),
            'positive_rate': float(np.mean(all_true))
        }
        
        # Save metrics to JSON
        safe_dump_json(metrics, os.path.join(output_dir, "evaluation_metrics.json"))
        
        return metrics
    
    except Exception as e:
        if logger:
            logger.log(f"Error in evaluation: {e}", level="ERROR")
        
        # Return basic metrics
        basic_metrics = {
            'num_samples': len(all_true),
            'positive_rate': float(np.mean(all_true)) if len(all_true) > 0 else 0,
            'mean_prediction': float(np.mean(all_preds)) if len(all_preds) > 0 else 0,
            'error': str(e)
        }
        
        # Save basic metrics
        safe_dump_json(basic_metrics, os.path.join(output_dir, "basic_metrics.json"))
        
        return basic_metrics


def analyze_feature_importance(model, X_file, y_file, test_indices, output_dir,
                             feature_names=None, logger=None):
    """
    Analyze feature importance using permutation method
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    X_file : str
        Path to features file
    y_file : str
        Path to labels file
    test_indices : numpy.ndarray
        Test set indices
    output_dir : str
        Output directory
    feature_names : list, optional
        Names of features
    logger : Logger, optional
        Logger instance
        
    Returns:
    --------
    dict
        Feature importance results
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    
    # Create output directory
    vis_dir = os.path.join(output_dir, "visualizations", "features")
    os.makedirs(vis_dir, exist_ok=True)
    
    if logger:
        logger.log(f"Starting feature importance analysis on {len(test_indices)} test samples")
    
    # Use a smaller subset for feature importance to save memory
    max_samples = 500
    np.random.seed(42)
    
    if len(test_indices) > max_samples:
        test_indices_subset = np.random.choice(test_indices, max_samples, replace=False)
    else:
        test_indices_subset = test_indices
    
    if logger:
        logger.log(f"Using {len(test_indices_subset)} samples for feature importance analysis")
    
    # Load data
    X = np.load(X_file, mmap_mode='r')
    y = np.load(y_file, mmap_mode='r')
    
    # Safely load test data
    try:
        X_test, y_test = load_data_batch(X, y, test_indices_subset, 0, len(test_indices_subset), logger)
    except Exception as e:
        if logger:
            logger.log(f"Error loading test data: {e}", level="ERROR")
        return {'error': f"Failed to load test data: {str(e)}"}
    
    if logger:
        logger.log(f"Test data loaded with shape X={X_test.shape}, y={y_test.shape}")
    
    # Get feature count from first sample
    n_features = X_test.shape[2]
    
    # Use default feature names if not provided
    if feature_names is None or len(feature_names) != n_features:
        feature_names = [f"Feature_{i+1}" for i in range(n_features)]
    
    if logger:
        logger.log(f"Analyzing importance for features: {feature_names}")
    
    # Calculate baseline performance
    try:
        baseline_preds = model.predict(X_test, verbose=0)
        baseline_auc = roc_auc_score(y_test, baseline_preds)
        
        if logger:
            logger.log(f"Baseline AUC: {baseline_auc:.4f}")
    except Exception as e:
        if logger:
            logger.log(f"Error calculating baseline performance: {e}", level="ERROR")
        return {'error': f"Failed to calculate baseline performance: {str(e)}"}
    
    # Perform permutation importance analysis
    n_repeats = 3
    importances = np.zeros((n_features, n_repeats))
    
    for feature_idx in range(n_features):
        feature_name = feature_names[feature_idx]
        
        if logger:
            logger.log(f"Analyzing importance of feature: {feature_name}")
        
        for repeat in range(n_repeats):
            try:
                # Create a copy of the test data
                X_permuted = X_test.copy()
                
                # Permute the feature across all time steps
                for time_step in range(X_test.shape[1]):
                    X_permuted[:, time_step, feature_idx] = np.random.permutation(X_test[:, time_step, feature_idx])
                
                # Predict with permuted feature
                perm_preds = model.predict(X_permuted, verbose=0)
                perm_auc = roc_auc_score(y_test, perm_preds)
                
                # Store importance (decrease in performance)
                importances[feature_idx, repeat] = baseline_auc - perm_auc
                
                # Clean up
                del X_permuted, perm_preds
                gc.collect()
                
                if logger and repeat == n_repeats - 1:
                    avg_importance = np.mean(importances[feature_idx, :])
                    logger.log(f"  Average importance: {avg_importance:.4f}")
                
            except Exception as e:
                if logger:
                    logger.log(f"  Error in permutation {repeat+1} for feature {feature_name}: {e}", level="ERROR")
                importances[feature_idx, repeat] = np.nan
    
    # Calculate mean and std of importance
    mean_importances = np.nanmean(importances, axis=1)
    std_importances = np.nanstd(importances, axis=1)
    
    # Create DataFrame for results
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_importances,
        'Std': std_importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    if logger:
        logger.log(f"\nFeature importance results:\n{importance_df}")
    
    # Plot feature importance
    try:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df,
                  xerr=importance_df['Std'], palette='viridis')
        plt.title('Feature Importance (Permutation Method)')
        plt.xlabel('Decrease in AUC when Feature is Permuted')
        plt.ylabel('Feature')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "feature_importance.png"), dpi=300)
        plt.close()
        
        # Save results
        importance_df.to_csv(os.path.join(vis_dir, "feature_importance.csv"), index=False)
        
        if logger:
            logger.log("Created feature importance plot and saved results")
            
    except Exception as e:
        if logger:
            logger.log(f"Error creating feature importance plot: {e}", level="ERROR")
    
    # Clean up
    del X_test, y_test
    gc.collect()
    
    # Return results
    return {
        'feature_importance': importance_df.to_dict(orient='records'),
        'baseline_auc': float(baseline_auc) if 'baseline_auc' in locals() else None
    }


def analyze_spatial_performance(output_dir, metadata_file=None, logger=None):
    """
    Analyze model performance across different geographical regions
    
    Parameters:
    -----------
    output_dir : str
        Directory with test predictions
    metadata_file : str, optional
        Path to metadata pickle file
    logger : Logger, optional
        Logger instance
        
    Returns:
    --------
    dict
        Spatial analysis results
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations", "spatial")
    os.makedirs(vis_dir, exist_ok=True)
    
    if logger:
        logger.log("Starting spatial performance analysis")
    
    # Try to load predictions with metadata
    pred_file = os.path.join(output_dir, "test_predictions_with_metadata.csv")
    
    if os.path.exists(pred_file):
        if logger:
            logger.log(f"Loading predictions with metadata from {pred_file}")
        
        try:
            predictions = pd.read_csv(pred_file)
        except Exception as e:
            if logger:
                logger.log(f"Error loading predictions CSV: {e}", level="ERROR")
            return {'error': f"Failed to load predictions: {str(e)}"}
    else:
        if logger:
            logger.log("No predictions with metadata found, attempting to reconstruct")
        
        try:
            # Load separate files
            preds = np.load(os.path.join(output_dir, "test_predictions.npy"))
            true = np.load(os.path.join(output_dir, "test_true_labels.npy"))
            test_indices = np.load(os.path.join(output_dir, "test_indices.npy"))
            
            # Create DataFrame
            predictions = pd.DataFrame({
                'prediction': preds,
                'true_label': true,
                'predicted_label': (preds > 0.5).astype(int),
                'correct': ((preds > 0.5).astype(int) == true).astype(int)
            })
            
            # Add metadata if available
            if metadata_file and os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "rb") as f:
                        metadata = pickle.load(f)
                    
                    if logger:
                        logger.log(f"Loaded metadata with {len(metadata)} entries")
                    
                    # Add metadata fields
                    for i, idx in enumerate(test_indices):
                        if i < len(predictions) and idx in metadata:
                            meta = metadata[idx]
                            for key, value in meta.items():
                                if key not in predictions.columns:
                                    predictions[key] = None
                                predictions.loc[i, key] = value
                except Exception as e:
                    if logger:
                        logger.log(f"Error adding metadata: {e}", level="ERROR")
        except Exception as e:
            if logger:
                logger.log(f"Error reconstructing predictions: {e}", level="ERROR")
            return {'error': f"Failed to reconstruct predictions: {str(e)}"}
    
    # Check if spatial data is available
    if 'latitude' not in predictions.columns or 'longitude' not in predictions.columns:
        if logger:
            logger.log("No spatial data (latitude/longitude) found in predictions")
        return {'error': 'No spatial data available'}
    
    # Define latitude bands
    def get_lat_band(lat):
        if pd.isna(lat):
            return "Unknown"
        
        try:
            lat = float(lat)
            if lat < 49:
                return "<49°N"
            elif lat < 55:
                return "49-55°N"
            elif lat < 60:
                return "55-60°N"
            elif lat < 66.5:
                return "60-66.5°N (Subarctic)"
            elif lat < 70:
                return "66.5-70°N (Arctic)"
            elif lat < 75:
                return "70-75°N (Arctic)"
            elif lat < 80:
                return "75-80°N (Arctic)"
            else:
                return ">80°N (Arctic)"
        except (ValueError, TypeError):
            return "Unknown"
    
    # Add latitude band
    predictions['lat_band'] = predictions['latitude'].apply(get_lat_band)
    
    # Group by latitude band
    try:
        # Safe aggregation function
        def safe_band_aggregation(x):
            if len(x) == 0:
                return pd.Series({
                    'count': 0,
                    'positive_count': 0,
                    'positive_rate': np.nan,
                    'accuracy': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1': np.nan
                })
            
            try:
                # Calculate metrics safely
                pred_sum = x['predicted_label'].sum()
                true_sum = x['true_label'].sum()
                
                # Safe precision calculation
                if pred_sum > 0:
                    precision = (x['predicted_label'] & x['true_label']).sum() / pred_sum
                else:
                    precision = np.nan
                
                # Safe recall calculation
                if true_sum > 0:
                    recall = (x['predicted_label'] & x['true_label']).sum() / true_sum
                else:
                    recall = np.nan
                
                # Safe F1 calculation
                if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = np.nan
                
                return pd.Series({
                    'count': len(x),
                    'positive_count': true_sum,
                    'positive_rate': true_sum / len(x),
                    'accuracy': x['correct'].mean(),
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
            except Exception as e:
                if logger:
                    logger.log(f"Error in band aggregation: {e}", level="ERROR")
                
                return pd.Series({
                    'count': len(x),
                    'positive_count': np.nan,
                    'positive_rate': np.nan,
                    'accuracy': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1': np.nan
                })
        
        # Apply aggregation
        band_metrics = predictions.groupby('lat_band').apply(safe_band_aggregation)
        band_metrics.to_csv(os.path.join(vis_dir, "latitude_band_metrics.csv"))
        
        if logger:
            logger.log(f"Analyzed performance across {len(band_metrics)} latitude bands")
        
        # Create visualization of performance by latitude band
        try:
            plt.figure(figsize=(14, 8))
            
            # Order bands by latitude
            ordered_bands = [
                '<55°N', '55-60°N', '60-66.5°N (Subarctic)',
                '66.5-70°N (Arctic)', '70-75°N (Arctic)',
                '75-80°N (Arctic)', '>80°N (Arctic)'
            ]
            
            available_bands = [b for b in ordered_bands if b in band_metrics.index]
            
            if available_bands:
                # Get data
                band_data = band_metrics.loc[available_bands]
                
                # Create plot
                x = np.arange(len(available_bands))
                width = 0.15
                
                plt.bar(x - 2*width, band_data['accuracy'], width, label='Accuracy', color='#3274A1')
                plt.bar(x - width, band_data['precision'], width, label='Precision', color='#E1812C')
                plt.bar(x, band_data['recall'], width, label='Recall', color='#3A923A')
                plt.bar(x + width, band_data['f1'], width, label='F1', color='#C03D3E')
                plt.bar(x + 2*width, band_data['positive_rate'], width, label='Positive Rate', color='#9372B2')
                
                plt.xlabel('Latitude Band')
                plt.ylabel('Score')
                plt.title('Model Performance by Latitude Band')
                plt.xticks(x, available_bands, rotation=45, ha='right')
                plt.legend(loc='lower right')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, "performance_by_latitude.png"), dpi=300)
                plt.close()
                
                if logger:
                    logger.log("Created latitude band performance visualization")
            else:
                if logger:
                    logger.log("No valid latitude bands for visualization")
                
        except Exception as e:
            if logger:
                logger.log(f"Error creating latitude band plot: {e}", level="ERROR")
        
        # Create map visualization if cartopy is available
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            
            # Group by site/source if available
            if 'source' in predictions.columns:
                site_metrics = predictions.groupby('source').apply(safe_band_aggregation)
                site_metrics = site_metrics[site_metrics['count'] >= 10]  # Filter small sites
                site_metrics.to_csv(os.path.join(vis_dir, "site_metrics.csv"))
                
                if len(site_metrics) > 0:
                    plt.figure(figsize=(15, 12))
                    
                    # Set up projection
                    ax = plt.axes(projection=ccrs.NorthPolarStereo())
                    ax.set_extent([-180, 180, 45, 90], ccrs.PlateCarree())
                    
                    # Add map features
                    ax.add_feature(cfeature.LAND, facecolor='lightgray')
                    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.BORDERS, linestyle=':')
                    
                    # Add gridlines
                    gl = ax.gridlines(draw_labels=True)
                    gl.top_labels = False
                    gl.right_labels = False
                    
                    # Get site coordinates
                    site_coords = predictions.groupby('source').agg({
                        'latitude': 'mean',
                        'longitude': 'mean'
                    })
                    
                    # Merge with metrics
                    site_data = site_metrics.merge(site_coords, left_index=True, right_index=True)
                    site_data = site_data
                    # Filter out rows with NaN coordinates
                    site_data = site_data.dropna(subset=['latitude', 'longitude'])
                    
                    # Create scatter plot
                    scatter = ax.scatter(
                        site_data['longitude'],
                        site_data['latitude'],
                        transform=ccrs.PlateCarree(),
                        c=site_data['f1'].fillna(0),
                        s=site_data['count'] / 5,
                        cmap='RdYlGn',
                        vmin=0, vmax=1,
                        alpha=0.8,
                        edgecolor='black',
                        linewidth=0.5
                    )
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
                    cbar.set_label('F1 Score')
                    
                    # Add Arctic Circle
                    theta = np.linspace(0, 2*np.pi, 100)
                    center, radius = [0, 0], 90 - 66.5
                    verts = np.vstack([radius*np.sin(theta), radius*np.cos(theta)]).T
                    circle = plt.Line2D(verts[:, 0], verts[:, 1], color='blue',
                                      linestyle='--', transform=ax.transData)
                    ax.add_line(circle)
                    
                    plt.title('Spatial Distribution of Model Performance (F1 Score)', fontsize=14)
                    plt.savefig(os.path.join(vis_dir, "spatial_performance_map.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    if logger:
                        logger.log("Created spatial performance map")
        except ImportError:
            if logger:
                logger.log("Cartopy not available for map visualization", level="WARNING")
        except Exception as e:
            if logger:
                logger.log(f"Error creating map visualization: {e}", level="ERROR")
        
        # Return results
        summary = {
            'latitude_bands': {
                'count': len(band_metrics),
                'bands': band_metrics.to_dict()
            }
        }
        
        # Save summary
        safe_dump_json(summary, os.path.join(vis_dir, "spatial_summary.json"))
        
        return summary
    
    except Exception as e:
        if logger:
            logger.log(f"Error in spatial analysis: {e}", level="ERROR")
        return {'error': str(e)}

# ============================================================================
# PART 5: MAIN EXECUTION SCRIPT
# ============================================================================

def run_pipeline(data_dir, output_dir, metadata_file=None, resume=True, chunk_size=5000):
    """
    Run the full Zero Curtain model pipeline with safe execution
    
    Parameters:
    -----------
    data_dir : str
        Directory containing input data files
    output_dir : str
        Directory for output
    metadata_file : str, optional
        Path to metadata file
    resume : bool
        Whether to resume training from previous checkpoint
    chunk_size : int
        Size of training chunks
        
    Returns:
    --------
    dict
        Pipeline results and metrics
    """
    # Start timing
    start_time = time.time()
    
    # Set up output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize logger
    logger = Logger(output_dir)
    logger.section("PIPELINE INITIALIZATION")
    logger.log(f"Starting Zero Curtain pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Data directory: {data_dir}")
    logger.log(f"Output directory: {output_dir}")
    logger.log(f"Metadata file: {metadata_file if metadata_file else 'None'}")
    
    # Configure TensorFlow
    logger.log("Configuring TensorFlow")
    configure_tensorflow()
    
    # Define file paths
    X_file = os.path.join(data_dir, 'ml_data/X_features.npy')
    y_file = os.path.join(data_dir, 'ml_data/y_labels.npy')
    
    if not os.path.exists(X_file) or not os.path.exists(y_file):
        logger.log(f"Error: Data files not found. X_file: {X_file}, y_file: {y_file}", level="ERROR")
        return {'error': 'Data files not found'}
    
    # Step 1: Load or create data split
    logger.section("DATA PREPARATION")
    
    split_file = os.path.join(data_dir, "checkpoints/spatiotemporal_split.pkl")
    
    if resume and os.path.exists(split_file):
        # Load existing split
        logger.log(f"Loading existing data split from {split_file}")
        with open(split_file, "rb") as f:
            split_data = pickle.load(f)
            
        train_indices = split_data["train_indices"]
        val_indices = split_data["val_indices"]
        test_indices = split_data["test_indices"]
    else:
        # Create new data split
        logger.log("Creating new train/val/test split")
        
        try:
            # Load data to get total samples
            X = np.load(X_file, mmap_mode='r')
            total_samples = len(X)
            
            # Create stratified split
            y = np.load(y_file, mmap_mode='r')
            
            # Use smaller test and validation sets (10% each)
            from sklearn.model_selection import train_test_split
            
            # First split off test set
            train_val_indices, test_indices = train_test_split(
                np.arange(total_samples),
                test_size=0.1,
                random_state=42,
                stratify=y
            )
            
            # Then split train/val
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=0.11,  # ~10% of original data
                random_state=42,
                stratify=y[train_val_indices]
            )
            
            # Save split
            split_data = {
                "train_indices": train_indices,
                "val_indices": val_indices,
                "test_indices": test_indices
            }
            
            with open(split_file, "wb") as f:
                pickle.dump(split_data, f)
                
            logger.log(f"Created and saved new data split with {len(train_indices)} train, "
                      f"{len(val_indices)} validation, and {len(test_indices)} test samples")
        except Exception as e:
            logger.log(f"Error creating data split: {e}", level="ERROR")
            return {'error': f'Failed to create data split: {str(e)}'}
    
    # Step 2: Training
    logger.section("MODEL TRAINING")
    
    # Training with progressive rebuilding to avoid memory issues
    try:
        # First ensure the custom layer is registered
        tf.keras.utils.get_custom_objects().update({'BatchNorm5D': BatchNorm5D})
        
        best_weights_path = progressive_training(
            X_file=X_file,
            y_file=y_file,
            train_indices=train_indices,
            val_indices=val_indices,
            output_dir=output_dir,
            chunk_size=chunk_size,
            rebuild_interval=5,  # Rebuild every 5 chunks
            logger=logger
        )
        
        logger.log(f"Training completed with best weights at: {best_weights_path}")
    except Exception as e:
        logger.log(f"Error during training: {e}", level="ERROR")
        import traceback
        logger.log(traceback.format_exc(), level="ERROR")
        return {'error': f'Training failed: {str(e)}'}
    
    # Step 3: Evaluation
    logger.section("MODEL EVALUATION")
    
    try:
        # Clear session before evaluation
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Load a fresh model for evaluation
        X = np.load(X_file, mmap_mode='r')
        sample_shape = X[train_indices[0]].shape
        
        eval_model = build_zero_curtain_model(sample_shape, logger)
        
        # Load best weights
        eval_model.load_weights(best_weights_path)
        logger.log(f"Loaded evaluation model with weights from {best_weights_path}")
        
        # Evaluate
        eval_results = evaluate_model(
            model=eval_model,
            X_file=X_file,
            y_file=y_file,
            test_indices=test_indices,
            output_dir=output_dir,
            metadata_file=metadata_file,
            logger=logger
        )
        
        logger.log("Evaluation completed")
        logger.log(f"Results: {eval_results}")
    except Exception as e:
        logger.log(f"Error during evaluation: {e}", level="ERROR")
        import traceback
        logger.log(traceback.format_exc(), level="ERROR")
        eval_results = {'error': f'Evaluation failed: {str(e)}'}
    
    # Step 4: Feature Importance Analysis
    logger.section("FEATURE IMPORTANCE ANALYSIS")
    
    try:
        # Define feature names for better interpretability
        feature_names = ['Temperature', 'Temperature Gradient', 'Depth']
        
        # Analyze feature importance
        feature_results = analyze_feature_importance(
            model=eval_model,
            X_file=X_file,
            y_file=y_file,
            test_indices=test_indices,
            output_dir=output_dir,
            feature_names=feature_names,
            logger=logger
        )
        
        logger.log("Feature importance analysis completed")
    except Exception as e:
        logger.log(f"Error during feature importance analysis: {e}", level="ERROR")
        import traceback
        logger.log(traceback.format_exc(), level="ERROR")
        feature_results = {'error': f'Feature analysis failed: {str(e)}'}
    
    # Step 5: Spatial Analysis
    logger.section("SPATIAL ANALYSIS")
    
    try:
        # Analyze spatial performance
        spatial_results = analyze_spatial_performance(
            output_dir=output_dir,
            metadata_file=metadata_file,
            logger=logger
        )
        
        logger.log("Spatial analysis completed")
    except Exception as e:
        logger.log(f"Error during spatial analysis: {e}", level="ERROR")
        import traceback
        logger.log(traceback.format_exc(), level="ERROR")
        spatial_results = {'error': f'Spatial analysis failed: {str(e)}'}
    
    # Compile all results
    end_time = time.time()
    execution_time = end_time - start_time
    
    logger.section("PIPELINE SUMMARY")
    logger.log(f"Pipeline completed in {timedelta(seconds=int(execution_time))}")
    
    # Compile results
    results = {
        'execution_time': execution_time,
        'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'),
        'evaluation': eval_results,
        'feature_importance': feature_results,
        'spatial_analysis': spatial_results
    }
    
    # Save final results
    results_file = os.path.join(output_dir, "pipeline_results.json")
    safe_dump_json(results, results_file)
    
    logger.log(f"Results saved to {results_file}")
    logger.log("Pipeline execution complete!")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Zero Curtain Detection Pipeline")
    parser.add_argument("--data_dir", required=True, help="Directory containing input data files")
    parser.add_argument("--output_dir", required=True, help="Directory for output")
    parser.add_argument("--metadata", help="Path to metadata file (optional)")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Size of training chunks")
    parser.add_argument("--no_resume", action="store_true", help="Start training from scratch")
    
    args = parser.parse_args()
    
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        metadata_file=args.metadata,
        resume=not args.no_resume,
        chunk_size=args.chunk_size
    )


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.colors import LogNorm, SymLogNorm
import os

class QuantumEnvironmentalAnalyzer:
    """Analyze relationships between quantum parameters and environmental drivers"""
    
    def __init__(self, quantum_data, environmental_data):
        """
        Initialize analyzer with quantum and environmental data
        
        Args:
            quantum_data: dict containing thermal_wavelength, tunneling_probability
            environmental_data: dict containing temp, alt, and other environmental variables
        """
        # Quantum parameters
        self.thermal_wavelength = quantum_data['thermal_wavelength']
        self.tunneling_probability = quantum_data['tunneling_probability']
        
        # Environmental parameters
        self.temperature = environmental_data['temp']
        self.alt = environmental_data['alt']
        
        # Optional environmental data
        self.additional_data = {
            k: v for k, v in environmental_data.items() 
            if k not in ['temp', 'alt']
        }
    
    def analyze_phase_transitions(self):
        """Analyze quantum behavior during phase transitions"""
        # Define phase transition threshold
        transition_zone = np.abs(self.temperature) <= 0.1  # ±0.1°C
        
        # Analyze quantum parameters in transition zones
        wavelength_transition = self.thermal_wavelength[transition_zone]
        tunneling_transition = self.tunneling_probability[transition_zone]
        
        return {
            'wavelength_stats': {
                'mean': np.nanmean(wavelength_transition),
                'std': np.nanstd(wavelength_transition),
                'percentiles': np.nanpercentile(wavelength_transition, [25, 50, 75])
            },
            'tunneling_stats': {
                'mean': np.nanmean(tunneling_transition),
                'std': np.nanstd(tunneling_transition),
                'percentiles': np.nanpercentile(tunneling_transition, [25, 50, 75])
            },
            'transition_frequency': np.sum(transition_zone) / transition_zone.size
        }
    
    def compute_environmental_correlations(self):
        """Compute correlations between quantum and environmental parameters"""
        correlations = {}
        
        # Prepare quantum parameters
        quantum_params = {
            'thermal_wavelength': self.thermal_wavelength.flatten(),
            'tunneling_probability': self.tunneling_probability.flatten()
        }
        
        # Prepare environmental parameters
        env_params = {
            'temperature': self.temperature.flatten(),
            'alt': self.alt.flatten()
        }
        env_params.update({
            k: v.flatten() for k, v in self.additional_data.items()
        })
        
        # Compute correlations
        for qp_name, qp_values in quantum_params.items():
            correlations[qp_name] = {}
            for ep_name, ep_values in env_params.items():
                # Remove NaN values
                mask = ~(np.isnan(qp_values) | np.isnan(ep_values))
                if np.sum(mask) > 0:
                    corr, p_value = stats.pearsonr(
                        qp_values[mask],
                        ep_values[mask]
                    )
                    correlations[qp_name][ep_name] = {
                        'correlation': corr,
                        'p_value': p_value
                    }
        
        return correlations
    
    def analyze_seasonal_patterns(self, time_grid):
        """Analyze seasonal patterns in quantum parameters"""
        months = pd.DatetimeIndex(time_grid).month
        seasonal_patterns = {}
        
        for month in range(1, 13):
            month_mask = months == month
            seasonal_patterns[month] = {
                'thermal_wavelength': {
                    'mean': np.nanmean(self.thermal_wavelength[month_mask]),
                    'std': np.nanstd(self.thermal_wavelength[month_mask]),
                    'active_fraction': np.sum(~np.isnan(self.thermal_wavelength[month_mask])) / 
                                     self.thermal_wavelength[month_mask].size
                },
                'tunneling_probability': {
                    'mean': np.nanmean(self.tunneling_probability[month_mask]),
                    'std': np.nanstd(self.tunneling_probability[month_mask]),
                    'active_fraction': np.sum(~np.isnan(self.tunneling_probability[month_mask])) / 
                                     self.tunneling_probability[month_mask].size
                }
            }
        
        return seasonal_patterns
    
    def identify_hotspots(self, lat_grid, lon_grid):
        """Identify regions of enhanced quantum activity"""
        # Define hotspot thresholds
        wavelength_threshold = np.nanmean(self.thermal_wavelength) + np.nanstd(self.thermal_wavelength)
        tunneling_threshold = np.nanmean(self.tunneling_probability) + np.nanstd(self.tunneling_probability)
        
        # Identify hotspots
        wavelength_hotspots = np.nanmean(self.thermal_wavelength, axis=0) > wavelength_threshold
        tunneling_hotspots = np.nanmean(self.tunneling_probability, axis=0) > tunneling_threshold
        
        # Get hotspot coordinates
        hotspots = {
            'wavelength': {
                'latitude': lat_grid[wavelength_hotspots],
                'longitude': lon_grid[wavelength_hotspots],
                'intensity': np.nanmean(self.thermal_wavelength, axis=0)[wavelength_hotspots]
            },
            'tunneling': {
                'latitude': lat_grid[tunneling_hotspots],
                'longitude': lon_grid[tunneling_hotspots],
                'intensity': np.nanmean(self.tunneling_probability, axis=0)[tunneling_hotspots]
            }
        }
        
        return hotspots
    
    def visualize_relationships(self, output_dir='quantum_analysis'):
        """Generate visualizations of quantum-environmental relationships"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Phase transition analysis
        self._plot_phase_transition_analysis(output_dir)
        
        # 2. Environmental correlations
        self._plot_correlation_matrix(output_dir)
        
        # 3. Seasonal patterns
        self._plot_seasonal_patterns(output_dir)
        
        # 4. Hotspot visualization
        self._plot_hotspots(output_dir)
    
    def _plot_phase_transition_analysis(self, output_dir):
        """Plot phase transition analysis results"""
        plt.figure(figsize=(15, 10))
        
        # Implementation here
        
        plt.savefig(f'{output_dir}/phase_transitions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_matrix(self, output_dir):
        """Plot correlation matrix"""
        plt.figure(figsize=(12, 8))
        
        # Implementation here
        
        plt.savefig(f'{output_dir}/correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_seasonal_patterns(self, output_dir):
        """Plot seasonal patterns"""
        plt.figure(figsize=(15, 10))
        
        # Implementation here
        
        plt.savefig(f'{output_dir}/seasonal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_hotspots(self, output_dir):
        """Plot quantum activity hotspots"""
        plt.figure(figsize=(15, 10))
        
        # Implementation here
        
        plt.savefig(f'{output_dir}/hotspots.png', dpi=300, bbox_inches='tight')
        plt.close()

class MultiSourceIntegrator:
    """Framework for integrating multiple data sources with quantum parameters"""
    
    def __init__(self):
        self.data_sources = {}
        self.transformations = {}
        self.scale_factors = {}
    
    def add_remote_sensing_source(self, name, data, metadata):
        """
        Add remote sensing data source
        
        Parameters:
        -----------
        name : str
            Name of data source (e.g., 'MODIS_LST', 'SMAP_SM')
        data : numpy.ndarray
            Data array
        metadata : dict
            Metadata including resolution, timestamps, etc.
        """
        self.data_sources[name] = {
            'data': data,
            'metadata': metadata,
            'type': 'remote_sensing'
        }
    
    def add_in_situ_source(self, name, data, metadata):
        """Add in-situ measurement source"""
        self.data_sources[name] = {
            'data': data,
            'metadata': metadata,
            'type': 'in_situ'
        }
    
    def register_transformation(self, source_name, transform_func):
        """Register data transformation function"""
        self.transformations[source_name] = transform_func
    
    def set_scale_factor(self, source_name, scale_factor):
        """Set scale factor for data source"""
        self.scale_factors[source_name] = scale_factor
    
    def harmonize_data(self, target_grid, target_times):
        """
        Harmonize all data sources to common spatiotemporal grid
        
        Parameters:
        -----------
        target_grid : tuple
            (lat_grid, lon_grid) for target resolution
        target_times : array-like
            Target timestamps
        """
        harmonized_data = {}
        
        for name, source in self.data_sources.items():
            # Apply transformation if registered
            if name in self.transformations:
                data = self.transformations[name](source['data'])
            else:
                data = source['data']
            
            # Apply scale factor if registered
            if name in self.scale_factors:
                data *= self.scale_factors[name]
            
            # Interpolate to target grid
            harmonized_data[name] = self._interpolate_to_target(
                data,
                source['metadata'],
                target_grid,
                target_times
            )
        
        return harmonized_data
    
    def _interpolate_to_target(self, data, metadata, target_grid, target_times):
        """Interpolate data to target grid"""
        # Implementation depends on data type and structure
        pass
    
    def analyze_relationships(self, harmonized_data):
        """Analyze relationships between different data sources"""
        # Implementation for relationship analysis
        pass
    
    def generate_integrated_visualization(self, harmonized_data, output_dir):
        """Generate visualizations of integrated data"""
        # Implementation for visualization
        pass

class RemoteSensingProcessor:
    """Process specific remote sensing data types"""
    
    @staticmethod
    def process_modis_lst(data):
        """Process MODIS land surface temperature data"""
        # Implementation for MODIS LST processing
        pass
    
    @staticmethod
    def process_smap_sm(data):
        """Process SMAP soil moisture data"""
        # Implementation for SMAP processing
        pass
    
    @staticmethod
    def process_insar(data):
        """Process InSAR data"""
        # Implementation for InSAR processing
        pass

class DataValidator:
    """Validate integrated data quality"""
    
    @staticmethod
    def validate_coverage(data):
        """Validate spatial and temporal coverage"""
        # Implementation for coverage validation
        pass
    
    @staticmethod
    def validate_consistency(data):
        """Validate data consistency"""
        # Implementation for consistency validation
        pass
    
    @staticmethod
    def generate_quality_report(validation_results):
        """Generate data quality report"""
        # Implementation for quality reporting
        pass

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from datetime import datetime, timedelta

# def analyze_dataset(temp_df, zc_df):
#     """Analyze the temperature and zero curtain datasets"""
#     print("Dataset Overview:")
#     print(f"Temperature measurements: {len(temp_df):,}")
#     print(f"Zero curtain events: {len(zc_df):,}")
    
#     # Temporal coverage
#     temp_range = pd.to_datetime(temp_df['datetime'], format='mixed')
#     print("\nTemporal Coverage:")
#     print(f"Start: {temp_range.min()}")
#     print(f"End: {temp_range.max()}")
    
#     # Depth distribution
#     depths = temp_df['depth'].unique()
#     print(f"\nUnique measurement depths: {len(depths)}")
#     print(f"Depth range: {depths.min():.1f}m to {depths.max():.1f}m")
    
#     # Zero curtain statistics
#     print("\nZero Curtain Statistics:")
#     print(f"Mean duration: {zc_df['duration_hours'].mean():.1f} hours")
#     print(f"Median duration: {zc_df['duration_hours'].median():.1f} hours")
    
#     return {
#         'depths': np.sort(depths),
#         'temp_range': temp_range,
#         'spatial_extent': {
#             'lat_range': (temp_df['latitude'].min(), temp_df['latitude'].max()),
#             'lon_range': (temp_df['longitude'].min(), temp_df['longitude'].max())
#         }
#     }

# def preprocess_for_training(temp_df, zc_df, window_size=30, min_depths=4):
#     """Preprocess the data for model training with duplicate handling"""
#     # Convert to datetime
#     temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], format='mixed')
#     zc_df['start_date'] = pd.to_datetime(zc_df['start_date'], format='mixed')
#     zc_df['end_date'] = pd.to_datetime(zc_df['end_date'], format='mixed')
    
#     # Handle duplicate measurements by averaging
#     temp_df = temp_df.groupby(['site_id', 'datetime', 'depth']).agg({
#         'temperature': 'mean',
#         'latitude': 'first',
#         'longitude': 'first'
#     }).reset_index()
    
#     # Select common depths across sites
#     depth_counts = temp_df.groupby('depth').site_id.nunique()
#     common_depths = depth_counts[depth_counts >= min_depths].index.sort_values()
    
#     # Filter for common depths
#     temp_df = temp_df[temp_df['depth'].isin(common_depths)]
    
#     # Process each site
#     site_groups = temp_df.groupby('site_id')
#     processed_data = []
    
#     for site_id, group in site_groups:
#         # Create pivot table with cleaned data
#         pivot = group.pivot_table(
#             index='datetime',
#             columns='depth',
#             values='temperature',
#             aggfunc='mean'  # Handle any remaining duplicates
#         ).sort_index()
        
#         # Ensure regular time intervals
#         pivot = pivot.resample('D').mean()  # Daily resampling
        
#         # Skip if insufficient data
#         if len(pivot.columns) < min_depths:
#             continue
            
#         # Create sliding windows
#         dates = pivot.index
#         for i in range(len(dates) - window_size + 1):
#             window = pivot.iloc[i:i+window_size]
            
#             # Skip windows with too many missing values
#             if window.isnull().sum().sum() / (window.shape[0] * window.shape[1]) > 0.3:
#                 continue
                
#             # Check for zero curtain events
#             window_start = dates[i]
#             window_end = dates[i + window_size - 1]
            
#             zc_events = zc_df[
#                 (zc_df['site_id'] == site_id) &
#                 (zc_df['start_date'] >= window_start) &
#                 (zc_df['end_date'] <= window_end)
#             ]
            
#             # Fill missing values using interpolation
#             window_filled = window.interpolate(method='linear', axis=0)
#             window_filled = window_filled.fillna(method='ffill').fillna(method='bfill')
            
#             if not window_filled.isnull().any().any():
#                 processed_data.append({
#                     'site_id': site_id,
#                     'window_data': window_filled.values,
#                     'window_start': window_start,
#                     'window_end': window_end,
#                     'latitude': group['latitude'].iloc[0],
#                     'longitude': group['longitude'].iloc[0],
#                     'depths': window_filled.columns.values,
#                     'has_zero_curtain': 1 if not zc_events.empty else 0,
#                     'zc_duration': zc_events['duration_hours'].iloc[0] if not zc_events.empty else...
#                 })
#     print(f"\nProcessed {len(processed_data)} valid windows from {len(site_groups)} sites")
#     return processed_data

# def create_physics_features(processed_data):
#     """Create additional physics-based features"""
#     for data in processed_data:
#         temp_profile = data['window_data']
#         depths = data['depths']
        
#         # Calculate thermal gradients
#         depth_diff = np.diff(depths)
#         vertical_gradients = np.zeros_like(temp_profile)
#         for i in range(len(depths)-1):
# vertical_gradients[:, i] = (temp_profile[:, i+1] - temp_profile[:,...
        
#         temporal_gradients = np.gradient(temp_profile, axis=0)
        
#         # Estimate thermal diffusivity (simplified)
#         thermal_diffusivity = np.abs(vertical_gradients) / (np.abs(temporal_gradients) + 1e-6)
        
#         # Add to features
#         data['thermal_gradients'] = vertical_gradients
#         data['thermal_diffusivity'] = thermal_diffusivity
#         data['temp_gradients_temporal'] = temporal_gradients
        
#         # Phase state indicator (near 0°C)
#         data['phase_state'] = np.abs(temp_profile) < 0.1
        
#         # Calculate potential freezing front velocity
#         data['freezing_front_velocity'] = np.zeros_like(vertical_gradients)
#         mask = data['phase_state'][:, :-1]
#         data['freezing_front_velocity'][mask] = vertical_gradients[mask] * thermal_diffusivity[mas...
    
#     return processed_data

# def prepare_training_batches(processed_data, batch_size=32, val_split=0.2):
#     """Prepare data batches for training with validation split"""
#     # Convert to numpy arrays
#     X_temp = np.array([d['window_data'] for d in processed_data])
#     X_gradients = np.array([d['thermal_gradients'] for d in processed_data])
#     X_diffusivity = np.array([d['thermal_diffusivity'] for d in processed_data])
#     X_phase = np.array([d['phase_state'] for d in processed_data]).astype(float)
#     X_velocity = np.array([d['freezing_front_velocity'] for d in processed_data])
    
#     # Spatial coordinates
#     X_spatial = np.array([[d['latitude'], d['longitude']] for d in processed_data])
    
#     # Normalize features
#     scaler = StandardScaler()
#     X_spatial_norm = scaler.fit_transform(X_spatial)
    
#     # Combine features
#     X = np.concatenate([
#         X_temp.reshape(len(X_temp), -1),
#         X_gradients.reshape(len(X_gradients), -1),
#         X_diffusivity.reshape(len(X_diffusivity), -1),
#         X_phase.reshape(len(X_phase), -1),
#         X_velocity.reshape(len(X_velocity), -1),
#         X_spatial_norm
#     ], axis=1)
    
#     # Labels
#     y = np.array([d['has_zero_curtain'] for d in processed_data])
    
#     # Split indices
#     n_val = int(len(X) * val_split)
#     indices = np.random.permutation(len(X))
#     train_idx, val_idx = indices[n_val:], indices[:n_val]
    
#     # Create TensorFlow datasets
#     train_dataset = tf.data.Dataset.from_tensor_slices(
#         (X[train_idx], y[train_idx])
#     ).shuffle(buffer_size=len(train_idx)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
#     val_dataset = tf.data.Dataset.from_tensor_slices(
#         (X[val_idx], y[val_idx])
#     ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
#     return train_dataset, val_dataset

