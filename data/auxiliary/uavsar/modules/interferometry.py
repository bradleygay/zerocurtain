#!/usr/bin/env python3
"""
modules/interferometry.py
In-Memory Interferometric Processing - CORRECTED GEOLOCATION

"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging
import rasterio
from rasterio.transform import from_bounds


class InMemoryInterferometricProcessor:
    """Process H5 files directly to displacement GeoTIFFs"""
    
    def __init__(self, 
                 base_dir: str,
                 output_dir: str,
                 target_resolution: int = 30,
                 target_crs: str = 'EPSG:4326',
                 coherence_threshold: float = 0.3,
                 skip_existing: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize processor
        Args:
            base_dir: Base directory containing H5 files
            output_dir: Output directory for displacement GeoTIFFs
            target_resolution: Target resolution in meters
            target_crs: Target coordinate reference system
            coherence_threshold: Minimum coherence for valid pixels
            skip_existing: Skip processing if output exists
            logger: Logger instance
        """
        self.h5_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_resolution = target_resolution
        self.target_crs = target_crs
        self.coherence_threshold = coherence_threshold
        self.skip_existing = skip_existing
        self.logger = logger or logging.getLogger(__name__)
        
        self.stats = {
            'pairs_found': 0,
            'pairs_processed': 0,
            'pairs_failed': 0,
            'sites_processed': 0,
            'displacement_maps_generated': 0
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
            self.stats['sites_processed'] = len(sites)
            self.logger.info(f"\n[{site_idx}/{len(sites)}] Flight line: {site_name}")
            self.logger.info(f"  H5 files: {len(site_files)}")
            
            # Find temporal pairs
            pairs = self._find_temporal_pairs(site_files)
            self.logger.info(f"  Found {len(pairs)} pairs")
            self.stats['pairs_found'] += len(pairs)
            
            # Process each pair
            for pair_idx, (file1, file2) in enumerate(pairs, 1):
                # Check if output already exists
                output_filename = self._create_output_filename(file1, file2, 'HHHV')
                output_path = self.output_dir / output_filename
                
                if self.skip_existing and output_path.exists():
                    self.logger.info(f"  [{pair_idx}/{len(pairs)}] Skipping (checkpoint)")
                    continue
                
                self.logger.info(f"  [{pair_idx}/{len(pairs)}] {file1.name} + {file2.name}")
                
                # Determine polarization
                polarization = self._determine_polarization(file1)
                self.logger.info(f"    Polarization: {polarization}")
                
                # Process to displacement
                result = self._process_pair_to_displacement(file1, file2, polarization)
                
                if result:
                    self.stats['pairs_processed'] += 1
                    self.stats['displacement_maps_generated'] += 1
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
    
    def _process_pair_to_displacement(self, h5_file1: Path, h5_file2: Path, polarization: str) -> Optional[Path]:
        """Process H5 pair to displacement - CORRECTED GEOLOCATION"""
        
        try:
            # Read SLC data and metadata
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
            
            # Extract phase
            phase = np.angle(interferogram)
            
            # Convert to displacement
            wavelength = 0.238  # L-band meters
            displacement = (phase / (4 * np.pi)) * wavelength
            
            # Mask invalid values using configured threshold
            coherence = np.abs(interferogram) / (np.abs(slc1) * np.abs(slc2) + 1e-10)
            mask = (coherence > self.coherence_threshold) & np.isfinite(displacement)
            displacement_masked = np.where(mask, displacement, np.nan)
            
            valid_percent = (np.sum(mask) / mask.size) * 100
            self.logger.info(f"    Valid pixels: {valid_percent:.1f}%")
            
            # Create output path
            output_filename = self._create_output_filename(h5_file1, h5_file2, polarization)
            output_path = self.output_dir / output_filename
            
            # CRITICAL: Create correct geotransform from H5 metadata
            rows, cols = displacement_masked.shape
            
            west = nw_lon
            east = nw_lon + (cols * lon_spacing)
            north = nw_lat
            south = nw_lat + (rows * lat_spacing)  # lat_spacing is negative
            
            self.logger.info(f"    Bounds: N={north:.4f}, S={south:.4f}, W={west:.4f}, E={east:.4f}")
            
            transform = from_bounds(west, south, east, north, cols, rows)
            
            # Save GeoTIFF with configured CRS
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=rows,
                width=cols,
                count=1,
                dtype=displacement_masked.dtype,
                crs=self.target_crs,
                transform=transform,
                compress='lzw',
                nodata=np.nan
            ) as dst:
                dst.write(displacement_masked, 1)
            
            self.logger.info(f"     Generated: {output_filename}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"     Failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _group_by_site(self, h5_files: list) -> Dict[str, list]:
        """Group H5 files by site"""
        sites = {}
        
        for h5_file in h5_files:
            site_name = h5_file.stem.split('_')[0]
            if site_name not in sites:
                sites[site_name] = []
            sites[site_name].append(h5_file)
        
        for site_name in sites:
            sites[site_name] = sorted(sites[site_name])
            self.logger.info(f"  {site_name}: {len(sites[site_name])} files")
        
        return sites
    
    def _find_temporal_pairs(self, h5_files: list) -> list:
        """Find temporal pairs"""
        pairs = []
        for i in range(len(h5_files) - 1):
            pairs.append((h5_files[i], h5_files[i + 1]))
        return pairs
    
    def _determine_polarization(self, h5_file: Path) -> str:
        """Determine best polarization"""
        polarizations = ['HHHV', 'HHVV', 'HVVV', 'HHHH', 'HVHV', 'VVVV']
        
        try:
            with h5py.File(h5_file, 'r') as f:
                for pol in polarizations:
                    if pol in f:
                        return pol
        except:
            pass
        
        return 'HHHV'
    
    def _create_output_filename(self, h5_file1: Path, h5_file2: Path, polarization: str) -> str:
        """Create output filename"""
        return f"{h5_file1.stem}_{h5_file2.stem}_{polarization}_displacement_{self.target_resolution}m.tif"
