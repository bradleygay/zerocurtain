#!/usr/bin/env python3
"""
modules/uavsar_processing.py
UAVSAR H5 Processing - CORRECTED geolocation from metadata

"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime
import rasterio
from rasterio.transform import from_bounds


class UAVSARProcessor:
    """Process UAVSAR H5 files to displacement GeoTIFFs"""
    
    def __init__(self, output_dir: str, logger: Optional[logging.Logger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
    
    def process_h5_pair_to_displacement(self,
                                       h5_file1: Path,
                                       h5_file2: Path,
                                       polarization: str = 'HHHV') -> Optional[Path]:
        """
        Process H5 pair directly to displacement GeoTIFF
        CORRECTED: Uses geolocation from H5 metadata attributes
        """
        
        try:
            # Read SLC data
            with h5py.File(h5_file1, 'r') as f1, h5py.File(h5_file2, 'r') as f2:
                slc1 = f1[polarization][:]
                slc2 = f2[polarization][:]
                
                # CRITICAL: Read geolocation from metadata attributes
                nw_lat = f1.attrs['northwest latitude']
                nw_lon = f1.attrs['northwest longitude']
                lat_spacing = f1.attrs['latitude spacing']
                lon_spacing = f1.attrs['longitude spacing']
                
                self.logger.info(f"    Geolocation: NW corner ({nw_lat:.4f}°N, {nw_lon:.4f}°W)")
                self.logger.info(f"    Spacing: {lat_spacing:.6f}° lat, {lon_spacing:.6f}° lon")
            
            self.logger.info(f"    Loaded SLCs: shape={slc1.shape}")
            
            # Compute interferogram
            interferogram = slc1 * np.conj(slc2)
            
            # Extract phase (displacement proxy)
            phase = np.angle(interferogram)
            
            # Convert to displacement (simplified)
            wavelength = 0.238  # L-band wavelength in meters
            displacement = (phase / (4 * np.pi)) * wavelength
            
            # Mask invalid values
            coherence = np.abs(interferogram) / (np.abs(slc1) * np.abs(slc2) + 1e-10)
            mask = (coherence > 0.3) & np.isfinite(displacement)
            
            displacement_masked = np.where(mask, displacement, np.nan)
            
            valid_percent = (np.sum(mask) / mask.size) * 100
            self.logger.info(f"    Valid pixels: {valid_percent:.1f}%")
            
            # Create output filename
            output_filename = self._create_output_filename(h5_file1, h5_file2, polarization)
            output_path = self.output_dir / output_filename
            
            # CRITICAL: Create correct geotransform from metadata
            # Calculate bounds: NW corner + (rows * lat_spacing, cols * lon_spacing)
            rows, cols = displacement_masked.shape
            
            # Note: lat_spacing is NEGATIVE (north to south)
            west = nw_lon
            east = nw_lon + (cols * lon_spacing)
            north = nw_lat
            south = nw_lat + (rows * lat_spacing)  # lat_spacing is negative
            
            self.logger.info(f"    Bounds: N={north:.4f}, S={south:.4f}, W={west:.4f}, E={east:.4f}")
            
            # Create transform
            transform = from_bounds(west, south, east, north, cols, rows)
            
            # Save as GeoTIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=rows,
                width=cols,
                count=1,
                dtype=displacement_masked.dtype,
                crs='EPSG:4326',
                transform=transform,
                compress='lzw',
                nodata=np.nan
            ) as dst:
                dst.write(displacement_masked, 1)
            
            self.logger.info(f"     Generated: {output_filename}")
            
            return output_path
        
        except Exception as e:
            self.logger.error(f"     Failed: {e}")
            return None
    
    def _create_output_filename(self, h5_file1: Path, h5_file2: Path, polarization: str) -> str:
        """Create output filename from input H5 files"""
        base1 = h5_file1.stem
        base2 = h5_file2.stem
        return f"{base1}_{base2}_{polarization}_displacement_30m.tif"


class InMemoryProcessor:
    """Process H5 files to displacement without intermediate storage"""
    
    def __init__(self, h5_dir: str, output_dir: str, logger: Optional[logging.Logger] = None):
        self.h5_dir = Path(h5_dir)
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.processor = UAVSARProcessor(output_dir, logger)
        
        self.stats = {
            'pairs_found': 0,
            'pairs_processed': 0,
            'pairs_failed': 0
        }
    
    def process_all_sites(self) -> Dict:
        """Process all H5 files to displacement GeoTIFFs"""
        
        self.logger.info("="*80)
        self.logger.info("IN-MEMORY INTERFEROMETRIC PROCESSING")
        self.logger.info("H5 → Displacement GeoTIFFs (no intermediate storage)")
        self.logger.info("="*80)
        
        # Find all H5 files
        h5_files = list(self.h5_dir.rglob("*.h5"))
        
        if not h5_files:
            self.logger.warning(f"No H5 files found in {self.h5_dir}")
            return self.stats
        
        self.logger.info(f"Found {len(h5_files)} total H5 files")
        
        # Group by site
        sites = self._group_by_site(h5_files)
        
        self.logger.info(f"Grouped into {len(sites)} flight lines")
        
        # Process each site
        for site_idx, (site_name, site_files) in enumerate(sites.items(), 1):
            self.logger.info(f"\n[{site_idx}/{len(sites)}] Flight line: {site_name}")
            self.logger.info(f"  H5 files: {len(site_files)}")
            
            # Find temporal pairs
            pairs = self._find_temporal_pairs(site_files)
            
            self.logger.info(f"  Found {len(pairs)} pairs")
            self.stats['pairs_found'] += len(pairs)
            
            # Process each pair
            for pair_idx, (file1, file2) in enumerate(pairs, 1):
                self.logger.info(f"  [{pair_idx}/{len(pairs)}] {file1.name} + {file2.name}")
                
                # Determine polarization
                polarization = self._determine_polarization(file1)
                self.logger.info(f"    Polarization: {polarization}")
                
                # Process to displacement
                result = self.processor.process_h5_pair_to_displacement(
                    file1, file2, polarization
                )
                
                if result:
                    self.stats['pairs_processed'] += 1
                else:
                    self.stats['pairs_failed'] += 1
        
        # Summary
        self.logger.info("\n" + "="*80)
        self.logger.info("PROCESSING COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Flight lines processed: {len(sites)}")
        self.logger.info(f"Pairs found: {self.stats['pairs_found']}")
        self.logger.info(f"Displacement maps generated: {self.stats['pairs_processed']}")
        self.logger.info(f"Pairs failed: {self.stats['pairs_failed']}")
        self.logger.info("="*80)
        
        return self.stats
    
    def _group_by_site(self, h5_files: list) -> Dict[str, list]:
        """Group H5 files by site name"""
        sites = {}
        
        for h5_file in h5_files:
            # Extract site from filename (first part before underscore)
            site_name = h5_file.stem.split('_')[0]
            
            if site_name not in sites:
                sites[site_name] = []
            
            sites[site_name].append(h5_file)
        
        # Sort files within each site
        for site_name in sites:
            sites[site_name] = sorted(sites[site_name])
            self.logger.info(f"  {site_name}: {len(sites[site_name])} files")
        
        return sites
    
    def _find_temporal_pairs(self, h5_files: list) -> list:
        """Find temporal pairs from sorted H5 files"""
        pairs = []
        
        # Simple pairing: consecutive files
        for i in range(len(h5_files) - 1):
            pairs.append((h5_files[i], h5_files[i + 1]))
        
        return pairs
    
    def _determine_polarization(self, h5_file: Path) -> str:
        """Determine best polarization from H5 file"""
        
        # Priority order
        polarizations = ['HHHV', 'HHVV', 'HVVV', 'HHHH', 'HVHV', 'VVVV']
        
        try:
            with h5py.File(h5_file, 'r') as f:
                for pol in polarizations:
                    if pol in f:
                        return pol
        except Exception as e:
            self.logger.warning(f"Error reading polarizations from {h5_file}: {e}")
        
        return 'HHHV'  # Default
