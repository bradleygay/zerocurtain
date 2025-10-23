#!/usr/bin/env python3
"""
ArcticDEM Consolidation Module

Consolidates processed ArcticDEM tiles into a unified parquet dataframe
optimized for spatial queries. Uses vectorized operations and batch
processing for computational efficiency.

"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import rasterio
import pyproj
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import gc

# Processing configuration
MAX_WORKERS = 8
BATCH_SIZE = 50
CHECKPOINT_INTERVAL = 10

class CheckpointManager:
    """Manages consolidation checkpoints"""
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    
    def load(self) -> Dict:
        """Load checkpoint data"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading checkpoint: {e}")
        return {"processed_files": [], "batch_num": 0}
    
    def save(self, processed_files: List[str], batch_num: int):
        """Save checkpoint data"""
        checkpoint = {
            "processed_files": processed_files,
            "batch_num": batch_num,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)

class CoordinateTransformer:
    """Handles coordinate transformations"""
    
    def __init__(self, source_crs: str = 'EPSG:3413', target_crs: str = 'EPSG:4326'):
        self.transformer = pyproj.Transformer.from_crs(
            source_crs, target_crs, always_xy=True
        )
    
    def transform_coordinates(self, xs: np.ndarray, ys: np.ndarray) -> tuple:
        """
        Transform coordinates from source to target CRS
        
        Parameters:
        -----------
        xs : np.ndarray
            X coordinates in source CRS
        ys : np.ndarray
            Y coordinates in source CRS
        
        Returns:
        --------
        tuple
            (longitudes, latitudes) in target CRS
        """
        lons, lats = self.transformer.transform(xs, ys)
        return lons, lats

class DEMTileProcessor:
    """Processes individual DEM tiles"""
    
    def __init__(self, transformer: CoordinateTransformer):
        self.transformer = transformer
    
    def process_tile(self, tiff_path: Path) -> Optional[Dict]:
        """
        Extract and transform data from single DEM tile
        
        Parameters:
        -----------
        tiff_path : Path
            Path to GeoTIFF file
        
        Returns:
        --------
        Optional[Dict]
            Dictionary with datetime, coordinates, and elevations
        """
        try:
            with rasterio.open(tiff_path) as src:
                # Read elevation data
                elevation = src.read(1, masked=True)
                transform = src.transform
                
                # Get valid data mask
                valid_mask = ~elevation.mask & (elevation.data != -9999)
                if not np.any(valid_mask):
                    return None
                
                # Extract valid data indices
                rows, cols = np.where(valid_mask)
                valid_elevations = elevation.data[valid_mask]
                
                # Vectorized coordinate calculation
                xs = transform.a * cols + transform.b * rows + transform.c
                ys = transform.d * cols + transform.e * rows + transform.f
                
                # Transform to geographic coordinates
                lons, lats = self.transformer.transform_coordinates(xs, ys)
                
                # Get file modification time as datetime
                file_time = datetime.fromtimestamp(tiff_path.stat().st_mtime)
                
                return {
                    'datetime': file_time,
                    'latitude': lats,
                    'longitude': lons,
                    'elevation': valid_elevations
                }
                
        except Exception as e:
            logging.error(f"Error processing {tiff_path.name}: {e}")
            return None

class BatchWriter:
    """Handles batch writing to parquet"""
    
    def __init__(self, batch_dir: Path):
        self.batch_dir = batch_dir
        self.batch_dir.mkdir(parents=True, exist_ok=True)
    
    def write_batch(self, batch_data: List[Dict], batch_num: int) -> Path:
        """
        Write batch data to parquet file
        
        Parameters:
        -----------
        batch_data : List[Dict]
            List of tile data dictionaries
        batch_num : int
            Batch number for file naming
        
        Returns:
        --------
        Path
            Path to written batch file
        """
        # Pre-allocate arrays
        total_points = sum(len(data['latitude']) for data in batch_data)
        
        datetimes = np.empty(total_points, dtype='datetime64[ns]')
        latitudes = np.empty(total_points, dtype=np.float64)
        longitudes = np.empty(total_points, dtype=np.float64)
        elevations = np.empty(total_points, dtype=np.float32)
        
        # Fill arrays efficiently
        idx = 0
        for data in batch_data:
            n_points = len(data['latitude'])
            end_idx = idx + n_points
            
            datetimes[idx:end_idx] = data['datetime']
            latitudes[idx:end_idx] = data['latitude']
            longitudes[idx:end_idx] = data['longitude']
            elevations[idx:end_idx] = data['elevation']
            
            idx = end_idx
        
        # Create DataFrame
        df = pd.DataFrame({
            'datetime': datetimes,
            'latitude': latitudes,
            'longitude': longitudes,
            'elevation': elevations
        })
        
        # Write to parquet
        output_file = self.batch_dir / f"batch_{batch_num:05d}.parquet"
        df.to_parquet(
            output_file,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        return output_file

class ArcticDEMConsolidator:
    """Main consolidation orchestrator"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.checkpoint_mgr = CheckpointManager(Path(config['checkpoint_file']))
        self.transformer = CoordinateTransformer()
        self.tile_processor = DEMTileProcessor(self.transformer)
        self.batch_writer = BatchWriter(Path(config['batch_dir']))
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        log_dir = Path(self.config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"arcticdem_consolidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def get_remaining_files(self) -> tuple:
        """Get list of files not yet processed"""
        input_dir = Path(self.config['input_dir'])
        all_files = sorted(input_dir.glob('*_30m_terrain.tif'))
        
        # Load checkpoint
        checkpoint = self.checkpoint_mgr.load()
        processed_files = set(checkpoint["processed_files"])
        
        # Filter remaining
        remaining = [f for f in all_files if f.name not in processed_files]
        
        self.logger.info(f"Total files: {len(all_files)}")
        self.logger.info(f"Already processed: {len(processed_files)}")
        self.logger.info(f"Remaining: {len(remaining)}")
        
        return remaining, checkpoint["batch_num"], list(processed_files)
    
    def consolidate_tiles(self):
        """Execute consolidation pipeline"""
        self.logger.info("=== ArcticDEM Consolidation Started ===")
        self.logger.info(f"Using {MAX_WORKERS} workers")
        
        # Get remaining files
        remaining_files, batch_num, processed_files = self.get_remaining_files()
        
        if not remaining_files:
            self.logger.info("All files processed. Proceeding to merge.")
            self.merge_final()
            return
        
        # Process files in batches
        batch_results = []
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all remaining files
            futures = {
                executor.submit(self.tile_processor.process_tile, f): f
                for f in remaining_files
            }
            
            # Process as completed
            with tqdm(total=len(futures), desc="Processing tiles") as pbar:
                for future in as_completed(futures):
                    filename = futures[future]
                    result = future.result()
                    
                    if result is not None:
                        batch_results.append(result)
                        processed_files.append(filename.name)
                        
                        # Write batch when ready
                        if len(batch_results) >= BATCH_SIZE:
                            self.batch_writer.write_batch(batch_results, batch_num)
                            batch_num += 1
                            
                            # Save checkpoint
                            if batch_num % CHECKPOINT_INTERVAL == 0:
                                self.checkpoint_mgr.save(processed_files, batch_num)
                                self.logger.info(
                                    f"Checkpoint: {len(processed_files)} files processed"
                                )
                            
                            batch_results = []
                            gc.collect()
                    
                    pbar.update(1)
            
            # Write final batch
            if batch_results:
                self.batch_writer.write_batch(batch_results, batch_num)
                self.checkpoint_mgr.save(processed_files, batch_num + 1)
        
        self.logger.info("Tile processing complete")
        self.merge_final()
    
    def merge_final(self):
        """Merge all batch files into final parquet"""
        self.logger.info("=== Starting Final Merge ===")
        
        batch_files = sorted(self.batch_writer.batch_dir.glob('batch_*.parquet'))
        self.logger.info(f"Merging {len(batch_files)} batches")
        
        if not batch_files:
            self.logger.warning("No batch files found to merge")
            return
        
        # Read all batches
        dfs = []
        for batch_file in tqdm(batch_files, desc="Reading batches"):
            try:
                df = pd.read_parquet(batch_file)
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"Error reading {batch_file}: {e}")
        
        if not dfs:
            self.logger.error("No valid batch data to merge")
            return
        
        # Concatenate
        self.logger.info("Concatenating dataframes")
        final_df = pd.concat(dfs, ignore_index=True)
        
        # Write final output
        output_file = Path(self.config['output_file'])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Writing final output: {output_file}")
        final_df.to_parquet(
            output_file,
            engine='pyarrow',
            compression='snappy',
            index=False,
            row_group_size=100000
        )
        
        self.logger.info(f"Complete! {len(final_df)} records saved")
        
        # Cleanup
        self.logger.info("Cleaning up batch files")
        for batch_file in batch_files:
            try:
                batch_file.unlink()
            except Exception as e:
                self.logger.warning(f"Could not remove {batch_file}: {e}")
        
        # Remove checkpoint
        try:
            self.checkpoint_mgr.checkpoint_file.unlink()
        except Exception as e:
            self.logger.warning(f"Could not remove checkpoint: {e}")

def main():
    """Main entry point"""
    config = {
        'input_dir': "data/auxiliary/arcticdem/processed",
        'batch_dir': "data/auxiliary/arcticdem/batches",
        'output_file': "data/auxiliary/arcticdem/arcticdem_consolidated.parquet",
        'checkpoint_file': "checkpoints/arcticdem/consolidation_progress.json",
        'log_dir': "logs/arcticdem"
    }
    
    consolidator = ArcticDEMConsolidator(config)
    consolidator.consolidate_tiles()

if __name__ == "__main__":
    main()
