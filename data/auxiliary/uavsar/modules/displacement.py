#!/usr/bin/env python3
"""
modules/displacement.py
Displacement Extraction and Resampling Module - COMPLETE IMPLEMENTATION


Extracts displacement from unwrapped phase, applies quality control,
and resamples to 30m resolution with proper georeferencing.
"""

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import Affine
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging


class DisplacementExtractor:
    """
    Displacement extraction with quality control and 30m resampling
    """
    
    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 target_resolution: int = 30,
                 target_crs: str = 'EPSG:4326',
                 quality_control: Optional['DisplacementQualityControl'] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize displacement extractor
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_resolution = target_resolution
        self.target_crs = target_crs
        self.quality_control = quality_control
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'maps_generated': 0,
            'qc_passed': 0,
            'qc_failed': 0
        }
    
    def process_all_unwrapped_phases(self) -> Dict:
        """
        Process all unwrapped phase files
        """
        self.logger.info("Processing displacement maps...")
        
        # Find all unwrapped phase files
        unwrapped_files = list(self.input_dir.rglob("*_unwrapped.tif"))
        
        self.logger.info(f"Found {len(unwrapped_files)} unwrapped phase files")
        
        for unw_file in unwrapped_files:
            try:
                self._process_unwrapped_file(unw_file)
            except Exception as e:
                self.logger.error(f"Failed to process {unw_file}: {e}")
        
        return self.stats
    
    def _process_unwrapped_file(self, unw_file: Path):
        """Process a single unwrapped phase file"""
        
        # Find corresponding coherence file
        coh_file = unw_file.parent / unw_file.name.replace('_unwrapped.tif', '_coherence.tif')
        
        if not coh_file.exists():
            self.logger.warning(f"Coherence file not found for {unw_file.name}")
            coherence = None
        else:
            with rasterio.open(coh_file) as src:
                coherence = src.read(1, masked=True)
        
        # Read unwrapped phase
        with rasterio.open(unw_file) as src:
            unwrapped_phase = src.read(1, masked=True)
            src_crs = src.crs
            src_transform = src.transform
            src_profile = src.profile
        
        # Convert phase to displacement
        # displacement (meters) = (unwrapped_phase * wavelength) / (4 * pi)
        wavelength = 0.238  # L-band wavelength in meters
        displacement = (unwrapped_phase * wavelength) / (4 * np.pi)
        
        # Apply quality control if available
        if self.quality_control and coherence is not None:
            passed, qc_mask = self.quality_control.validate(displacement, coherence)
            
            if not passed:
                self.logger.warning(f"Quality control failed: {unw_file.name}")
                self.stats['qc_failed'] += 1
                return
            
            # Apply QC mask
            displacement = np.where(qc_mask, displacement, np.nan)
            self.stats['qc_passed'] += 1
        
        # Resample to 30m
        displacement_30m, transform_30m, crs_30m = self._resample_to_30m(
            displacement,
            src_transform,
            src_crs
        )
        
        # Save displacement map
        output_filename = unw_file.name.replace('_unwrapped.tif', '_displacement_30m.tif')
        output_path = self.output_dir / unw_file.parent.name / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._save_displacement(displacement_30m, output_path, transform_30m, crs_30m)
        
        self.stats['maps_generated'] += 1
        self.logger.info(f"Generated: {output_filename}")
    
    def _resample_to_30m(self, 
                        displacement: np.ndarray,
                        src_transform: Affine,
                        src_crs: str) -> Tuple[np.ndarray, Affine, str]:
        """
        Resample displacement to 30m resolution with CRS transformation
        """
        from pyproj import Transformer
        
        # Calculate bounds
        height, width = displacement.shape
        bounds = rasterio.transform.array_bounds(height, width, src_transform)
        
        # Calculate target transform
        if self.target_crs == 'EPSG:4326':
            # For WGS84, calculate degree resolution equivalent to 30m
            target_res = 0.0003  # degrees (conservative estimate)
        else:
            target_res = self.target_resolution
        
        target_transform, target_width, target_height = calculate_default_transform(
            src_crs,
            self.target_crs,
            width,
            height,
            *bounds,
            resolution=target_res
        )
        
        # Initialize output array
        resampled = np.zeros((target_height, target_width), dtype=np.float32)
        
        # Reproject
        reproject(
            source=displacement,
            destination=resampled,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=self.target_crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan
        )
        
        return resampled, target_transform, self.target_crs
    
    def _save_displacement(self, 
                          displacement: np.ndarray,
                          output_path: Path,
                          transform: Affine,
                          crs: str):
        """Save displacement as GeoTIFF"""
        
        height, width = displacement.shape
        
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'float32',
            'crs': crs,
            'transform': transform,
            'compress': 'DEFLATE',
            'predictor': 3,
            'tiled': True,
            'nodata': np.nan
        }
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(displacement.astype(np.float32), 1)
