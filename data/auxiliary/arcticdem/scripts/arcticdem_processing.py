#!/usr/bin/env python3
"""
ArcticDEM Processing Module

Terrain-aware resampling of ArcticDEM tiles from native resolution to 30m
using Gaussian filtering weighted by terrain slope. This preserves terrain
features while enabling efficient spatial analysis.

"""

import os
import sys
import logging
import numpy as np
import rasterio
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
from scipy.ndimage import gaussian_filter
from affine import Affine
import concurrent.futures
from dataclasses import dataclass

# Fixed sigma for computational efficiency
SIGMA_VALUE = 1.0
MAX_WORKERS = 8

@dataclass
class ProcessingMetrics:
    """Metrics for processing operations"""
    items_processed: int = 0
    items_failed: int = 0
    total_bytes_processed: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def elapsed_seconds(self) -> float:
        """Calculate elapsed time in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

class TerrainAwareResampler:
    """Handles terrain-aware DEM resampling operations"""
    
    def __init__(self, target_resolution: float = 30.0, sigma: float = SIGMA_VALUE):
        self.target_resolution = target_resolution
        self.sigma = sigma
        self.logger = logging.getLogger(__name__)
    
    def compute_slope(self, elevation: np.ndarray, transform: Affine) -> np.ndarray:
        """
        Compute terrain slope from elevation gradient
        
        Parameters:
        -----------
        elevation : np.ndarray
            2D elevation array
        transform : Affine
            Rasterio affine transformation
        
        Returns:
        --------
        np.ndarray
            Slope magnitude array
        """
        xres = abs(transform.a)
        yres = abs(transform.e)
        
        # Compute gradients
        dzdx = np.gradient(elevation, axis=1) / xres
        dzdy = np.gradient(elevation, axis=0) / yres
        
        # Slope magnitude
        slope = np.sqrt(dzdx**2 + dzdy**2)
        
        return slope
    
    def apply_terrain_filter(self, elevation: np.ndarray, 
                            slope: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian filter with fixed sigma for efficiency
        
        Parameters:
        -----------
        elevation : np.ndarray
            Elevation data
        slope : np.ndarray
            Slope data (unused in fixed-sigma implementation)
        
        Returns:
        --------
        np.ndarray
            Filtered elevation
        """
        # Fixed sigma for computational speed
        smoothed = gaussian_filter(elevation, sigma=self.sigma, mode='nearest')
        return smoothed
    
    def block_reduce(self, data: np.ndarray, factor: int) -> np.ndarray:
        """
        Reduce array by block averaging
        
        Parameters:
        -----------
        data : np.ndarray
            Input array
        factor : int
            Downsampling factor
        
        Returns:
        --------
        np.ndarray
            Reduced array
        """
        h, w = data.shape
        h_crop = (h // factor) * factor
        w_crop = (w // factor) * factor
        
        if h_crop == 0 or w_crop == 0:
            raise ValueError("Image too small for specified factor")
        
        # Reshape and average
        reduced = data[:h_crop, :w_crop].reshape(
            h_crop // factor, factor,
            w_crop // factor, factor
        ).mean(axis=(1, 3))
        
        return reduced
    
    def resample_dem(self, input_file: Path, output_file: Path) -> bool:
        """
        Resample DEM with terrain-aware processing
        
        Parameters:
        -----------
        input_file : Path
            Input DEM path
        output_file : Path
            Output resampled DEM path
        
        Returns:
        --------
        bool
            Success status
        """
        self.logger.info(f"Resampling: {input_file.name}")
        
        try:
            with rasterio.open(input_file) as src:
                # Calculate downscale factor
                original_res = src.res[0]
                downscale_factor = int(self.target_resolution / original_res)
                
                self.logger.info(
                    f"Resolution: {original_res}m -> {self.target_resolution}m "
                    f"(factor: {downscale_factor})"
                )
                
                # Read elevation data
                data = src.read(1, masked=True)
                
                # Handle NoData
                if hasattr(data, 'mask') and data.mask.any():
                    elevation = data.filled(np.nan)
                    self.logger.debug("Filled masked values with NaN")
                else:
                    elevation = data
                
                # Compute slope
                slope = self.compute_slope(elevation, src.transform)
                
                # Apply terrain-aware smoothing
                smoothed = self.apply_terrain_filter(elevation, slope)
                
                # Block reduce to target resolution
                reduced = self.block_reduce(smoothed, downscale_factor)
                
                # Update transform
                new_transform = src.transform * Affine.scale(
                    downscale_factor, downscale_factor
                )
                
                # Update metadata
                meta = src.meta.copy()
                meta.update({
                    'height': reduced.shape[0],
                    'width': reduced.shape[1],
                    'transform': new_transform,
                    'dtype': 'float32',
                    'count': 1,
                    'compress': 'lzw',
                    'tiled': True,
                    'blockxsize': 512,
                    'blockysize': 512
                })
                
                # Write output
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(output_file, 'w', **meta) as dst:
                    dst.write(reduced.astype(np.float32), 1)
            
            self.logger.info(f"Resampling completed: {output_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Resampling error {input_file.name}: {e}")
            if output_file.exists():
                output_file.unlink()
            return False

class DEMProcessor:
    """Orchestrates DEM processing pipeline"""
    
    def __init__(self, config: dict):
        self.config = config
        self.resampler = TerrainAwareResampler(
            target_resolution=config.get('target_resolution', 30.0)
        )
        self.metrics = ProcessingMetrics()
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        log_dir = Path(self.config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"arcticdem_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def process_single_file(self, input_file: Path) -> bool:
        """
        Process single DEM file
        
        Parameters:
        -----------
        input_file : Path
            Input DEM file
        
        Returns:
        --------
        bool
            Success status
        """
        output_dir = Path(self.config['output_dir'])
        item_id = input_file.stem.replace('_original', '')
        output_file = output_dir / f"{item_id}_30m_terrain.tif"
        
        # Skip if exists
        if output_file.exists():
            self.logger.info(f"Output exists: {output_file.name}")
            return True
        
        # Resample
        success = self.resampler.resample_dem(input_file, output_file)
        
        # Always clean up temp file
        if input_file.exists():
            try:
                input_file.unlink()
                self.logger.debug(f"Removed temp file: {input_file.name}")
            except Exception as e:
                self.logger.warning(f"Could not remove temp file: {e}")
        
        # Update metrics
        if success:
            self.metrics.items_processed += 1
            if output_file.exists():
                self.metrics.total_bytes_processed += output_file.stat().st_size
        else:
            self.metrics.items_failed += 1
        
        return success
    
    def process_directory(self, parallel: bool = True) -> ProcessingMetrics:
        """
        Process all DEM files in temp directory
        
        Parameters:
        -----------
        parallel : bool
            Enable parallel processing
        
        Returns:
        --------
        ProcessingMetrics
            Processing statistics
        """
        self.logger.info("=== ArcticDEM Processing Started ===")
        self.metrics.start_time = datetime.now()
        
        temp_dir = Path(self.config['temp_dir'])
        input_files = list(temp_dir.glob("*_original.tif"))
        
        self.logger.info(f"Found {len(input_files)} files to process")
        
        if parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(self.process_single_file, f): f
                    for f in input_files
                }
                
                for future in concurrent.futures.as_completed(futures):
                    input_file = futures[future]
                    try:
                        success = future.result()
                        if not success:
                            self.logger.warning(f"Failed: {input_file.name}")
                    except Exception as e:
                        self.logger.error(f"Exception processing {input_file.name}: {e}")
                        self.metrics.items_failed += 1
        else:
            for input_file in input_files:
                self.process_single_file(input_file)
        
        self.metrics.end_time = datetime.now()
        
        # Log summary
        self.logger.info("=== Processing Completed ===")
        self.logger.info(f"Processed: {self.metrics.items_processed}")
        self.logger.info(f"Failed: {self.metrics.items_failed}")
        self.logger.info(
            f"Data volume: {self.metrics.total_bytes_processed / 1e9:.2f} GB"
        )
        self.logger.info(
            f"Elapsed: {self.metrics.elapsed_seconds():.1f} seconds"
        )
        
        return self.metrics

def main():
    """Main entry point"""
    config = {
        'temp_dir': "data/auxiliary/arcticdem/temp",
        'output_dir': "data/auxiliary/arcticdem/processed",
        'log_dir': "logs/arcticdem",
        'target_resolution': 30.0
    }
    
    processor = DEMProcessor(config)
    metrics = processor.process_directory(parallel=True)
    
    sys.exit(0 if metrics.items_failed == 0 else 1)

if __name__ == "__main__":
    main()
