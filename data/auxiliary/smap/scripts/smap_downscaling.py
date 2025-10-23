#!/usr/bin/env python3
"""
SMAP Downscaling Pipeline: 9km → 30m
Uses terrain-aware Kriging with ArcticDEM and Landsat corrections
"""

import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from scipy.ndimage import zoom
import gc
from tqdm import tqdm
from pyproj import Transformer

# Directories
BASE_DIR = Path.home() / "arctic_zero_curtain_pipeline"
SMAP_DIR = BASE_DIR / "data" / "auxiliary" / "smap" / "consolidated"
DOWNSCALED_DIR = BASE_DIR / "data" / "auxiliary" / "smap" / "downscaled"
ARCTICDEM_DIR = BASE_DIR / "data" / "auxiliary" / "arcticdem"
LANDSAT_DIR = BASE_DIR / "data" / "auxiliary" / "landsat"
CHECKPOINT_DIR = BASE_DIR / "data" / "auxiliary" / "smap" / "checkpoints"

# Create output directory
DOWNSCALED_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Processing parameters
TARGET_RESOLUTION = 30  # meters
SMAP_RESOLUTION = 9000  # meters
CHUNK_SIZE = 27000  # 3 SMAP cells
TEMPORAL_WINDOW = 16  # days for Landsat matching


class SMAPDownscaler:
    """Downscale SMAP data from 9km to 30m using auxiliary data"""
    
    def __init__(self, year, start_date=None, end_date=None):
        self.year = year
        self.start_date = start_date or datetime(year, 1, 1)
        self.end_date = end_date or datetime(year, 12, 31)
        
        # Load SMAP data for year
        self.smap_file = SMAP_DIR / f"smap_{year}.parquet"
        if not self.smap_file.exists():
            raise FileNotFoundError(f"SMAP file not found: {self.smap_file}")
        
        # Load auxiliary data
        self.load_arcticdem()
        self.load_landsat()
        
        # Coordinate transformer
        self.transformer = Transformer.from_crs(3413, 4326, always_xy=True)
        
        # Checkpoint file
        self.checkpoint_file = CHECKPOINT_DIR / f"downscaling_{year}.txt"
        self.completed_dates = self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load completed dates from checkpoint"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                dates = set(line.strip() for line in f if line.strip())
            print(f"[CHECKPOINT] Loaded {len(dates)} completed dates")
            return dates
        return set()
    
    def mark_complete(self, date):
        """Mark date as completed"""
        with open(self.checkpoint_file, 'a') as f:
            f.write(f"{date.strftime('%Y-%m-%d')}\n")
    
    def load_arcticdem(self):
        """Load ArcticDEM data"""
        print("[LOAD] Loading ArcticDEM...")
        dem_file = ARCTICDEM_DIR / "merged_resumable.parquet"
        
        if not dem_file.exists():
            print("[WARNING] ArcticDEM not found - using placeholder")
            self.dem_df = None
            return
        
        try:
            # Load with memory mapping for efficiency
            self.dem_df = pd.read_parquet(
                dem_file,
                columns=['latitude', 'longitude', 'elevation'],
                engine='pyarrow'
            )
            
            # Convert to projected coordinates
            transformer = Transformer.from_crs(4326, 3413, always_xy=True)
            x, y = transformer.transform(
                self.dem_df['longitude'].values,
                self.dem_df['latitude'].values
            )
            self.dem_df['x'] = x
            self.dem_df['y'] = y
            
            print(f"[OK] Loaded {len(self.dem_df):,} DEM points")
        except Exception as e:
            print(f"[ERROR] Failed to load ArcticDEM: {e}")
            self.dem_df = None
    
    def load_landsat(self):
        """Load Landsat data"""
        print("[LOAD] Loading Landsat...")
        landsat_file = LANDSAT_DIR / "landsat_arctic_data_complete.parquet"
        
        if not landsat_file.exists():
            print("[WARNING] Landsat not found - using placeholder")
            self.landsat_df = None
            return
        
        try:
            # Load relevant columns only
            self.landsat_df = pd.read_parquet(
                landsat_file,
                columns=['acquisition_date', 'latitude', 'longitude', 'B2', 'B3', 'B4', 'B10', 'cloud_cover'],
                engine='pyarrow'
            )
            
            # Convert date
            self.landsat_df['acquisition_date'] = pd.to_datetime(
                self.landsat_df['acquisition_date']
            )
            
            # Filter by year and cloud cover
            self.landsat_df = self.landsat_df[
                (self.landsat_df['acquisition_date'].dt.year == self.year) &
                (self.landsat_df['cloud_cover'] <= 20)
            ]
            
            # Convert to projected coordinates
            transformer = Transformer.from_crs(4326, 3413, always_xy=True)
            x, y = transformer.transform(
                self.landsat_df['longitude'].values,
                self.landsat_df['latitude'].values
            )
            self.landsat_df['x'] = x
            self.landsat_df['y'] = y
            
            print(f"[OK] Loaded {len(self.landsat_df):,} Landsat observations for {self.year}")
        except Exception as e:
            print(f"[ERROR] Failed to load Landsat: {e}")
            self.landsat_df = None
    
    def get_landsat_for_date(self, date, bounds):
        """Get Landsat data for specific date and bounds"""
        if self.landsat_df is None:
            return None
        
        x_min, x_max, y_min, y_max = bounds
        
        # Temporal window
        date_min = date - timedelta(days=TEMPORAL_WINDOW)
        date_max = date + timedelta(days=TEMPORAL_WINDOW)
        
        # Filter
        mask = (
            (self.landsat_df['acquisition_date'] >= date_min) &
            (self.landsat_df['acquisition_date'] <= date_max) &
            (self.landsat_df['x'] >= x_min - 10000) &
            (self.landsat_df['x'] <= x_max + 10000) &
            (self.landsat_df['y'] >= y_min - 10000) &
            (self.landsat_df['y'] <= y_max + 10000)
        )
        
        result = self.landsat_df[mask]
        
        if len(result) == 0:
            return None
        
        # Calculate NDVI
        result = result.copy()
        result['ndvi'] = (result['B4'] - result['B2']) / (result['B4'] + result['B2'] + 1e-8)
        result['ndvi'] = result['ndvi'].clip(-1, 1)
        
        return result
    
    def get_dem_for_bounds(self, bounds):
        """Get DEM data for specific bounds"""
        if self.dem_df is None:
            return None
        
        x_min, x_max, y_min, y_max = bounds
        
        mask = (
            (self.dem_df['x'] >= x_min) &
            (self.dem_df['x'] <= x_max) &
            (self.dem_df['y'] >= y_min) &
            (self.dem_df['y'] <= y_max)
        )
        
        return self.dem_df[mask]
    
    def load_smap_for_date(self, date):
        """Load SMAP data for specific date"""
        parquet_file = pq.ParquetFile(self.smap_file)
        
        # Scan row groups for date
        date_str = date.date()
        collected = []
        
        for i in range(parquet_file.metadata.num_row_groups):
            # Read datetime column only
            rg = parquet_file.read_row_group(i, columns=['datetime'])
            df_dates = rg.to_pandas()
            
            if (df_dates['datetime'].dt.date == date_str).any():
                # Read full row group
                rg_full = parquet_file.read_row_group(i)
                df_full = rg_full.to_pandas()
                
                # Filter to date
                mask = df_full['datetime'].dt.date == date_str
                collected.append(df_full[mask])
        
        if collected:
            return pd.concat(collected, ignore_index=True)
        
        return pd.DataFrame()
    
    def downscale_chunk(self, smap_chunk, bounds, date):
        """Downscale single chunk from 9km to 30m"""
        x_min, x_max, y_min, y_max = bounds
        
        # Check if we have data
        if len(smap_chunk) == 0:
            return None
        
        # Create 30m grid
        width = int((x_max - x_min) / TARGET_RESOLUTION)
        height = int((y_max - y_min) / TARGET_RESOLUTION)
        
        x_fine = np.linspace(x_min, x_max, width)
        y_fine = np.linspace(y_max, y_min, height)
        xx_fine, yy_fine = np.meshgrid(x_fine, y_fine)
        
        # Get auxiliary data
        landsat_data = self.get_landsat_for_date(date, bounds)
        dem_data = self.get_dem_for_bounds(bounds)
        
        # Downscale each variable
        results = {}
        
        for var in ['sm_surface', 'sm_rootzone', 'soil_temp_layer1', 'soil_temp_layer2',
                    'soil_temp_layer3', 'soil_temp_layer4', 'soil_temp_layer5', 'soil_temp_layer6']:
            
            if var not in smap_chunk.columns:
                results[var] = np.full((height, width), np.nan, dtype=np.float32)
                continue
            
            # Get valid SMAP points
            valid_mask = smap_chunk[var].notna()
            if not valid_mask.any():
                results[var] = np.full((height, width), np.nan, dtype=np.float32)
                continue
            
            smap_valid = smap_chunk[valid_mask]
            
            # Interpolate to fine grid
            try:
                fine_grid = griddata(
                    (smap_valid['x'].values, smap_valid['y'].values),
                    smap_valid[var].values,
                    (xx_fine, yy_fine),
                    method='linear',
                    fill_value=np.nan
                )
            except Exception as e:
                print(f"[WARNING] Interpolation failed for {var}: {e}")
                results[var] = np.full((height, width), np.nan, dtype=np.float32)
                continue
            
            # Apply corrections if auxiliary data available
            if dem_data is not None and len(dem_data) > 0:
                fine_grid = self.apply_terrain_correction(
                    fine_grid, dem_data, xx_fine, yy_fine, var
                )
            
            if landsat_data is not None and len(landsat_data) > 0:
                fine_grid = self.apply_landsat_correction(
                    fine_grid, landsat_data, xx_fine, yy_fine, var
                )
            
            results[var] = fine_grid.astype(np.float32)
        
        return results
    
    def apply_terrain_correction(self, grid, dem_data, xx, yy, var):
        """Apply terrain-based corrections"""
        try:
            # Interpolate elevation to fine grid
            elevation = griddata(
                (dem_data['x'].values, dem_data['y'].values),
                dem_data['elevation'].values,
                (xx, yy),
                method='linear',
                fill_value=np.nan
            )
            
            # Calculate elevation correction
            elev_mean = np.nanmean(elevation)
            elev_diff = elevation - elev_mean
            
            if 'soil_temp' in var:
                # Temperature lapse rate: -6.5°C per 1000m
                lapse_rate = -6.5 / 1000
                correction = lapse_rate * elev_diff
                grid = grid + correction
            
            elif 'sm_' in var:
                # Moisture-elevation relationship
                # Higher elevation = typically drier
                correction = -0.0001 * elev_diff
                grid = grid * (1 + correction)
                grid = np.clip(grid, 0, 0.5)
        
        except Exception as e:
            print(f"[WARNING] Terrain correction failed: {e}")
        
        return grid
    
    def apply_landsat_correction(self, grid, landsat_data, xx, yy, var):
        """Apply Landsat-based corrections"""
        try:
            # Interpolate NDVI to fine grid
            ndvi = griddata(
                (landsat_data['x'].values, landsat_data['y'].values),
                landsat_data['ndvi'].values,
                (xx, yy),
                method='linear',
                fill_value=0.2
            )
            
            if 'sm_' in var:
                # Vegetation increases moisture retention
                ndvi_factor = 1.0 + 0.2 * ndvi
                grid = grid * ndvi_factor
                grid = np.clip(grid, 0, 0.5)
            
            # Could also use B10 (thermal) for temperature corrections
            if 'soil_temp' in var and 'B10' in landsat_data.columns:
                lst = griddata(
                    (landsat_data['x'].values, landsat_data['y'].values),
                    landsat_data['B10'].values,
                    (xx, yy),
                    method='linear',
                    fill_value=273.15
                )
                
                # LST influences soil temperature
                lst_mean = np.nanmean(lst)
                lst_influence = 0.1 * (lst - lst_mean)
                grid = grid + lst_influence
        
        except Exception as e:
            print(f"[WARNING] Landsat correction failed: {e}")
        
        return grid
    
    def get_circumarctic_bounds(self):
        """Get bounds for circumarctic domain"""
        transformer = Transformer.from_crs(4326, 3413, always_xy=True)
        
        corners = [
            (-180, 49),
            (180, 49),
            (180, 85.044),
            (-180, 85.044)
        ]
        
        x_coords, y_coords = [], []
        for lon, lat in corners:
            x, y = transformer.transform(lon, lat)
            x_coords.append(x)
            y_coords.append(y)
        
        return {
            'west': min(x_coords),
            'east': max(x_coords),
            'south': min(y_coords),
            'north': max(y_coords)
        }
    
    def create_processing_chunks(self):
        """Create spatial chunks for processing"""
        bounds = self.get_circumarctic_bounds()
        
        chunks = []
        x = bounds['west']
        while x < bounds['east']:
            y = bounds['south']
            while y < bounds['north']:
                chunks.append({
                    'bounds': (x, min(x + CHUNK_SIZE, bounds['east']),
                              y, min(y + CHUNK_SIZE, bounds['north'])),
                    'center': (x + CHUNK_SIZE/2, y + CHUNK_SIZE/2)
                })
                y += CHUNK_SIZE
            x += CHUNK_SIZE
        
        print(f"[INFO] Created {len(chunks):,} processing chunks")
        return chunks
    
    def process_date(self, date):
        """Process single date"""
        date_str = date.strftime('%Y-%m-%d')
        
        # Check if already completed
        if date_str in self.completed_dates:
            print(f"[SKIP] {date_str} - already completed")
            return True
        
        print(f"\n[DATE] Processing {date_str}")
        
        # Load SMAP data for date
        smap_data = self.load_smap_for_date(date)
        
        if len(smap_data) == 0:
            print(f"[SKIP] {date_str} - no SMAP data")
            return False
        
        print(f"[DATA] Loaded {len(smap_data):,} SMAP points")
        
        # Create output file
        output_file = DOWNSCALED_DIR / f"smap_downscaled_{date.strftime('%Y%m%d')}.parquet"
        
        # Process in chunks
        chunks = self.create_processing_chunks()
        all_results = []
        
        for chunk_info in tqdm(chunks, desc=f"Processing {date_str}"):
            bounds = chunk_info['bounds']
            x_min, x_max, y_min, y_max = bounds
            
            # Filter SMAP data for chunk
            mask = (
                (smap_data['x'] >= x_min) &
                (smap_data['x'] <= x_max) &
                (smap_data['y'] >= y_min) &
                (smap_data['y'] <= y_max)
            )
            
            chunk_data = smap_data[mask]
            
            if len(chunk_data) == 0:
                continue
            
            # Downscale chunk
            results = self.downscale_chunk(chunk_data, bounds, date)
            
            if results is not None:
                # Convert to DataFrame
                width = results['sm_surface'].shape[1]
                height = results['sm_surface'].shape[0]
                
                x_coords = np.linspace(x_min, x_max, width)
                y_coords = np.linspace(y_max, y_min, height)
                
                xx, yy = np.meshgrid(x_coords, y_coords)
                
                # Flatten and create DataFrame
                chunk_df = pd.DataFrame({
                    'datetime': date,
                    'x': xx.ravel(),
                    'y': yy.ravel()
                })
                
                # Add variables
                for var, data in results.items():
                    chunk_df[var] = data.ravel()
                
                # Remove NaN rows
                chunk_df = chunk_df.dropna(subset=['sm_surface', 'sm_rootzone'], how='all')
                
                if len(chunk_df) > 0:
                    all_results.append(chunk_df)
        
        # Combine and save
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            
            # Convert coordinates to lat/lon
            lon, lat = self.transformer.transform(final_df['x'].values, final_df['y'].values)
            final_df['latitude'] = lat
            final_df['longitude'] = lon
            
            # Save to parquet
            final_df.to_parquet(output_file, compression='zstd', index=False)
            
            print(f"[OK] Saved {len(final_df):,} points to {output_file.name}")
            
            # Mark complete
            self.mark_complete(date)
            
            # Clean up
            del final_df, all_results
            gc.collect()
            
            return True
        else:
            print(f"[WARNING] No valid data for {date_str}")
            return False
    
    def run(self):
        """Run downscaling for all dates in range"""
        print("=" * 80)
        print(f"SMAP Downscaling: {self.year}")
        print("=" * 80)
        print(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Target resolution: {TARGET_RESOLUTION}m")
        print(f"Chunk size: {CHUNK_SIZE}m")
        
        # Generate date range
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        
        print(f"\nProcessing {len(dates)} dates...")
        
        success_count = 0
        fail_count = 0
        
        for date in dates:
            try:
                success = self.process_date(date)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except KeyboardInterrupt:
                print("\n[INTERRUPTED] Stopping...")
                break
            except Exception as e:
                print(f"[ERROR] {date.date()}: {e}")
                fail_count += 1
                continue
        
        print("\n" + "=" * 80)
        print(f"Complete: {success_count} success, {fail_count} failed")
        print("=" * 80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SMAP Downscaling Pipeline')
    parser.add_argument('year', type=int, help='Year to process')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Parse dates if provided
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    
    try:
        downscaler = SMAPDownscaler(args.year, start_date, end_date)
        downscaler.run()
        return 0
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        return 130
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())