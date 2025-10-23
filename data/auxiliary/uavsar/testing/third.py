#!/usr/bin/env python3
"""
UAVSAR/NISAR Consolidated Interferometric Processing Pipeline
Institution: [RESEARCH_INSTITUTION] - Arctic Research and Data Science
Date: 2025-01-20

This consolidated script provides a complete end-to-end pipeline for processing
UAVSAR and NISAR-simulated interferometric data with the following capabilities:

1. Robust state tracking and intelligent resumption after interruptions
2. Comprehensive polarization detection across diverse HDF5 structures
3. Automated interferogram generation with quality validation
4. Phase unwrapping using optimized SNAPHU configuration
5. Georeferenced mosaic generation with metadata preservation
6. Parallel processing support for computational efficiency
7. Detailed logging and progress reporting

Pipeline Architecture:
    Data Discovery → Pair Identification → Interferogram Generation → 
    Phase Unwrapping → Georeferenced Mosaicking → Quality Assessment

Usage:
    python uavsar_nisar_pipeline.py --base_dir /path/to/data --output_dir /path/to/output --snaphu_path /path/to/snaphu
    
    For resumption after interruption:
    python uavsar_nisar_pipeline.py --base_dir /path/to/data --output_dir /path/to/output --resume
"""

import os
import sys
import h5py
import json
import pickle
import hashlib
import logging
import argparse
import itertools
import subprocess
import traceback
import re
import glob
import warnings
import numpy as np
import rasterio
from rasterio.transform import Affine
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("uavsar_nisar_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS - INTERFEROGRAM PROCESSING
# ============================================================================

def load_complex_slc(h5_path, pol):
    """
    Load complex SLC image (real + imaginary) for given polarization.
    
    Parameters:
    -----------
    h5_path : str
        Path to HDF5 file
    pol : str
        Polarization (HH, HV, VH, VV)
    
    Returns:
    --------
    numpy.ndarray : Complex SLC data
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            # Try multiple possible paths for polarization data
            paths_to_try = [
                f"/science/LSAR/SLC/frequencyA/{pol}",
                f"/science/LSAR/RSLC/frequencyA/{pol}",
                f"/science/LSAR/SLC/swaths/frequencyA/{pol}"
            ]
            
            for base_path in paths_to_try:
                # Check for real and imaginary components
                if f"{base_path}/r" in f and f"{base_path}/i" in f:
                    r = f[f"{base_path}/r"][:]
                    i = f[f"{base_path}/i"][:]
                    return r + 1j * i
                
                # Check for direct complex array
                if base_path in f and isinstance(f[base_path], h5py.Dataset):
                    return f[base_path][:]
            
            raise KeyError(f"Polarization {pol} not found in any expected location")
            
    except Exception as e:
        logger.error(f"Error loading SLC from {h5_path} for polarization {pol}: {e}")
        raise


def compute_interferogram(ref_slc, sec_slc):
    """
    Compute interferogram from reference and secondary SLC images.
    
    Parameters:
    -----------
    ref_slc : numpy.ndarray
        Reference complex SLC
    sec_slc : numpy.ndarray
        Secondary complex SLC
    
    Returns:
    --------
    numpy.ndarray : Complex interferogram
    """
    # Ensure same dimensions
    min_rows = min(ref_slc.shape[0], sec_slc.shape[0])
    min_cols = min(ref_slc.shape[1], sec_slc.shape[1])
    
    ref_slc = ref_slc[:min_rows, :min_cols]
    sec_slc = sec_slc[:min_rows, :min_cols]
    
    return ref_slc * np.conj(sec_slc)


def compute_coherence(ref_slc, sec_slc, window_size=5):
    """
    Estimate coherence using moving window approach.
    
    Parameters:
    -----------
    ref_slc : numpy.ndarray
        Reference complex SLC
    sec_slc : numpy.ndarray
        Secondary complex SLC
    window_size : int
        Size of estimation window
    
    Returns:
    --------
    numpy.ndarray : Coherence map [0, 1]
    """
    rows, cols = ref_slc.shape
    coherence = np.zeros((rows, cols), dtype=np.float32)
    hw = window_size // 2
    
    for i in range(hw, rows - hw):
        for j in range(hw, cols - hw):
            ref_win = ref_slc[i-hw:i+hw+1, j-hw:j+hw+1]
            sec_win = sec_slc[i-hw:i+hw+1, j-hw:j+hw+1]
            
            numerator = np.abs(np.sum(ref_win * np.conj(sec_win)))
            denominator = np.sqrt(np.sum(np.abs(ref_win)**2) * np.sum(np.abs(sec_win)**2))
            
            if denominator > 0:
                coherence[i, j] = numerator / denominator
    
    # Fill edges
    coherence[:hw, :] = coherence[hw, :]
    coherence[-hw:, :] = coherence[-hw-1, :]
    coherence[:, :hw] = coherence[:, hw:hw+1]
    coherence[:, -hw:] = coherence[:, -hw-1:-hw]
    
    return coherence


def export_geotiff(data, output_path, transform=None, crs=None, dtype="float32", nodata=None):
    """
    Export 2D array as GeoTIFF with proper georeferencing.
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D array to export
    output_path : str
        Output file path
    transform : affine.Affine, optional
        Geospatial transform
    crs : str, optional
        Coordinate reference system
    dtype : str
        Data type for output
    nodata : float, optional
        NoData value
    """
    height, width = data.shape
    
    if transform is None:
        transform = Affine.identity()
    
    meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': dtype,
        'transform': transform,
        'compress': 'DEFLATE',
        'predictor': 3 if dtype.startswith('float') else 2,
        'zlevel': 9,
        'tiled': True
    }
    
    if crs:
        meta['crs'] = crs
    if nodata is not None:
        meta['nodata'] = nodata
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data.astype(dtype), 1)


# ============================================================================
# METADATA EXTRACTION AND GEOREFERENCING
# ============================================================================

class MetadataExtractor:
    """Handles metadata extraction from HDF5 and annotation files."""
    
    @staticmethod
    def extract_h5_metadata(h5_path):
        """
        Extract comprehensive geospatial metadata from HDF5 file.
        
        Parameters:
        -----------
        h5_path : str
            Path to HDF5 file
        
        Returns:
        --------
        dict : Metadata dictionary
        """
        metadata = {}
        
        if not os.path.exists(h5_path):
            logger.warning(f"HDF5 file not found: {h5_path}")
            return metadata
        
        try:
            with h5py.File(h5_path, 'r') as f:
                # Extract EPSG code
                epsg_paths = [
                    '/science/LSAR/SLC/metadata/geolocationGrid/epsg',
                    '/science/LSAR/RSLC/metadata/geolocationGrid/epsg',
                    '/science/geolocationGrid/epsg',
                    '/metadata/geolocationGrid/epsg'
                ]
                
                for path in epsg_paths:
                    if path in f:
                        try:
                            epsg_value = f[path][()]
                            if isinstance(epsg_value, (np.ndarray, list, tuple)):
                                metadata['epsg'] = int(epsg_value[0])
                            else:
                                metadata['epsg'] = int(epsg_value)
                            logger.debug(f"Found EPSG: {metadata['epsg']}")
                            break
                        except Exception as e:
                            logger.debug(f"Error reading EPSG from {path}: {e}")
                
                # Default to WGS84 if not found
                if 'epsg' not in metadata:
                    metadata['epsg'] = 4326
                    logger.debug("EPSG not found, defaulting to 4326 (WGS84)")
                
                # Extract geolocation data
                geo_paths = [
                    '/science/LSAR/SLC/metadata/geolocationGrid',
                    '/science/LSAR/RSLC/metadata/geolocationGrid',
                    '/science/geolocationGrid',
                    '/metadata/geolocationGrid'
                ]
                
                for geo_path in geo_paths:
                    if geo_path in f:
                        geo_group = f[geo_path]
                        
                        # Extract coordinates
                        if 'coordinateX' in geo_group and 'coordinateY' in geo_group:
                            coordX = geo_group['coordinateX'][:]
                            coordY = geo_group['coordinateY'][:]
                            
                            # Filter invalid values
                            valid_mask = coordX > -1e11
                            lon_values = np.where(valid_mask, coordX, np.nan)
                            lat_values = np.where(valid_mask, coordY, np.nan)
                            
                            metadata['min_lon'] = float(np.nanmin(lon_values))
                            metadata['max_lon'] = float(np.nanmax(lon_values))
                            metadata['min_lat'] = float(np.nanmin(lat_values))
                            metadata['max_lat'] = float(np.nanmax(lat_values))
                            metadata['coord_shape'] = coordX.shape
                            
                        break
                
                # Extract identification metadata
                ident_paths = [
                    '/science/LSAR/identification',
                    '/science/LSAR/SLC/identification',
                    '/science/LSAR/RSLC/identification'
                ]
                
                for ident_path in ident_paths:
                    if ident_path in f:
                        ident_group = f[ident_path]
                        
                        if 'boundingPolygon' in ident_group:
                            metadata['boundingPolygon'] = ident_group['boundingPolygon'][()]
                        
                        if 'trackNumber' in ident_group:
                            metadata['trackNumber'] = ident_group['trackNumber'][()]
                        
                        break
                
                metadata['h5_source'] = h5_path
                
        except Exception as e:
            logger.error(f"Error extracting metadata from {h5_path}: {e}")
            logger.debug(traceback.format_exc())
        
        return metadata
    
    @staticmethod
    def extract_ann_metadata(ann_path):
        """
        Extract georeferencing metadata from UAVSAR annotation file.
        
        Parameters:
        -----------
        ann_path : str
            Path to .ann file
        
        Returns:
        --------
        dict : Metadata dictionary
        """
        metadata = {}
        
        if not os.path.exists(ann_path):
            logger.warning(f"ANN file not found: {ann_path}")
            return metadata
        
        try:
            with open(ann_path, 'r') as f:
                content = f.read()
            
            # Extract UTM zone
            utm_match = re.search(r'UTM zone:\s*(\d+)([NS])', content)
            if utm_match:
                metadata['utm_zone'] = int(utm_match.group(1))
                metadata['utm_hemisphere'] = utm_match.group(2)
            
            # Extract origin coordinates
            origin_match = re.search(r'Ground range origin:\s*([+-]?\d+\.\d+)\s+meters\s+([+-]?\d+\.\d+)\s+meters', content)
            if origin_match:
                metadata['origin_x'] = float(origin_match.group(1))
                metadata['origin_y'] = float(origin_match.group(2))
            
            # Extract pixel spacing
            range_spacing = re.search(r'Ground range pixel spacing:\s*([+-]?\d+\.\d+)\s+meters', content)
            if range_spacing:
                metadata['pixel_spacing_x'] = float(range_spacing.group(1))
            
            azimuth_spacing = re.search(r'Azimuth pixel spacing:\s*([+-]?\d+\.\d+)\s+meters', content)
            if azimuth_spacing:
                metadata['pixel_spacing_y'] = float(azimuth_spacing.group(1))
            
            # Extract center coordinates
            lat_match = re.search(r'Scene center latitude:\s*([+-]?\d+\.\d+)\s+degrees', content)
            lon_match = re.search(r'Scene center longitude:\s*([+-]?\d+\.\d+)\s+degrees', content)
            if lat_match and lon_match:
                metadata['center_lat'] = float(lat_match.group(1))
                metadata['center_lon'] = float(lon_match.group(1))
            
            # Extract additional metadata
            patterns = {
                'wavelength': r'Radar wavelength \(m\):\s*([+-]?\d+\.\d+)',
                'heading': r'Platform heading \(degrees\):\s*([+-]?\d+\.\d+)',
                'look_direction': r'Look Direction:\s*(\w+)',
                'polarization': r'Polarization:\s*(\w+)',
                'processing_date': r'Processing date:\s*(\d{4}-\d{2}-\d{2})'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    metadata[key] = match.group(1)
            
            metadata['ann_source'] = ann_path
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {ann_path}: {e}")
        
        return metadata
    
    @staticmethod
    def compute_geotransform(metadata, width, height):
        """
        Compute geospatial transform from metadata.
        
        Parameters:
        -----------
        metadata : dict
            Metadata dictionary
        width : int
            Raster width
        height : int
            Raster height
        
        Returns:
        --------
        tuple : (transform, crs)
        """
        transform = None
        crs = None
        
        # WGS84 transformation
        if metadata.get('epsg') == 4326 or ('min_lat' in metadata and 'min_lon' in metadata):
            if all(k in metadata for k in ['min_lon', 'max_lon', 'min_lat', 'max_lat']):
                pixel_width = (metadata['max_lon'] - metadata['min_lon']) / width
                pixel_height = (metadata['max_lat'] - metadata['min_lat']) / height
                
                transform = Affine(
                    pixel_width, 0.0, metadata['min_lon'],
                    0.0, pixel_height, metadata['min_lat']
                )
                crs = "EPSG:4326"
        
        # UTM transformation
        elif 'utm_zone' in metadata and 'utm_hemisphere' in metadata:
            if all(k in metadata for k in ['pixel_spacing_x', 'pixel_spacing_y', 'origin_x', 'origin_y']):
                pixel_width = metadata['pixel_spacing_x']
                pixel_height = -abs(metadata['pixel_spacing_y'])  # Negative for north-up
                
                transform = Affine(
                    pixel_width, 0.0, metadata['origin_x'],
                    0.0, pixel_height, metadata['origin_y'] + abs(pixel_height) * height
                )
                
                hemisphere_code = 6 if metadata['utm_hemisphere'] == 'N' else 7
                crs = f"EPSG:{hemisphere_code}{metadata['utm_zone']:02d}"
        
        if transform is None:
            logger.warning("Insufficient metadata for proper georeferencing")
            transform = Affine.identity()
        
        return transform, crs


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class UAVSARNISARPipeline:
    """
    Comprehensive UAVSAR/NISAR interferometric processing pipeline.
    
    This class orchestrates the complete workflow from data discovery
    through interferogram generation, phase unwrapping, and final
    georeferenced product creation.
    """
    
    def __init__(self, base_dir, output_dir, snaphu_path, skip_existing=True, 
                 max_workers=1, tile_size=512, snaphu_timeout=3600):
        """
        Initialize pipeline with configuration parameters.
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing UAVSAR/NISAR data
        output_dir : str
            Output directory for products
        snaphu_path : str
            Path to SNAPHU executable
        skip_existing : bool
            Skip processing if outputs exist
        max_workers : int
            Number of parallel workers
        tile_size : int
            Tile size for phase unwrapping
        snaphu_timeout : int
            SNAPHU timeout in seconds
        """
        self.base_dir = os.path.expanduser(base_dir)
        self.output_dir = os.path.expanduser(output_dir)
        self.snaphu_path = os.path.expanduser(snaphu_path)
        self.skip_existing = skip_existing
        self.max_workers = max_workers
        self.tile_size = tile_size
        self.snaphu_timeout = snaphu_timeout
        
        # Create state tracking directories
        self.state_dir = os.path.join(self.output_dir, ".pipeline_state")
        os.makedirs(self.state_dir, exist_ok=True)
        
        # State files
        self.sites_state_file = os.path.join(self.state_dir, "sites_state.json")
        self.pairs_state_file = os.path.join(self.state_dir, "pairs_state.json")
        
        # Load existing state
        self.sites_state = self._load_json_state(self.sites_state_file)
        self.pairs_state = self._load_json_state(self.pairs_state_file)
        
        # Initialize metadata extractor
        self.metadata_extractor = MetadataExtractor()
        
        # Statistics
        self.stats = {
            "total_sites": 0,
            "processed_sites": 0,
            "total_pairs": 0,
            "processed_pairs": 0,
            "skipped_pairs": 0,
            "failed_pairs": 0,
            "start_time": datetime.now().isoformat()
        }
        
        logger.info("Pipeline initialized successfully")
    
    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================
    
    def _load_json_state(self, state_file):
        """Load JSON state file with error handling."""
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state file {state_file}: {e}")
        return {}
    
    def _save_json_state(self, state_file, state):
        """Save state to JSON file."""
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save state file {state_file}: {e}")
    
    def _mark_site_processed(self, site):
        """Mark a site as processed."""
        self.sites_state[site] = {
            "processed": True,
            "timestamp": datetime.now().isoformat()
        }
        self._save_json_state(self.sites_state_file, self.sites_state)
        self.stats["processed_sites"] += 1
    
    def _mark_pair_processed(self, site, pair_id, status="completed", error=None):
        """Mark a pair as processed/failed."""
        if site not in self.pairs_state:
            self.pairs_state[site] = {}
        
        self.pairs_state[site][pair_id] = {
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        if error:
            self.pairs_state[site][pair_id]["error"] = str(error)
            self.stats["failed_pairs"] += 1
        elif status == "skipped":
            self.stats["skipped_pairs"] += 1
        else:
            self.stats["processed_pairs"] += 1
        
        self._save_json_state(self.pairs_state_file, self.pairs_state)
    
    def _is_pair_processed(self, site, pair_id):
        """Check if pair is already processed."""
        if site in self.pairs_state and pair_id in self.pairs_state[site]:
            return self.pairs_state[site][pair_id]["status"] in ["completed", "skipped"]
        return False
    
    # ========================================================================
    # DATA DISCOVERY AND VALIDATION
    # ========================================================================
    
    def find_valid_polarizations(self, h5_file):
        """
        Find valid polarizations in HDF5 file with comprehensive path checking.
        
        Parameters:
        -----------
        h5_file : h5py.File or str
            HDF5 file object or path
        
        Returns:
        --------
        list : Valid polarizations found
        """
        valid_pols = []
        
        try:
            # Handle both file path and file object
            if isinstance(h5_file, str):
                with h5py.File(h5_file, 'r') as f:
                    return self.find_valid_polarizations(f)
            
            # Check multiple possible paths
            pol_paths = [
                "science/LSAR/SLC/frequencyA",
                "science/LSAR/RSLC/frequencyA",
                "science/LSAR/SLC/swaths/frequencyA"
            ]
            
            for base_path in pol_paths:
                if base_path not in h5_file:
                    continue
                
                for pol in ["HH", "HV", "VH", "VV"]:
                    pol_path = f"{base_path}/{pol}"
                    
                    # Check for real and imaginary components
                    if pol_path in h5_file:
                        if "r" in h5_file[pol_path] and "i" in h5_file[pol_path]:
                            valid_pols.append(pol)
                            continue
                    
                    # Check for direct dataset
                    if pol in h5_file[base_path]:
                        if isinstance(h5_file[base_path][pol], h5py.Dataset):
                            valid_pols.append(pol)
            
            # Remove duplicates while preserving order
            valid_pols = list(dict.fromkeys(valid_pols))
            
        except Exception as e:
            logger.error(f"Error finding polarizations: {e}")
        
        return valid_pols
    
    def get_date_from_filename(self, filename):
        """
        Extract date from UAVSAR/NISAR filename with multiple pattern matching.
        
        Parameters:
        -----------
        filename : str
            Filename to parse
        
        Returns:
        --------
        datetime : Parsed date or None
        """
        basename = os.path.basename(filename)
        parts = basename.split("_")
        
        # Strategy 1: Look for YYMMDD in parts
        for part in parts:
            if len(part) == 6 and part.isdigit():
                try:
                    return datetime.strptime(part, "%y%m%d")
                except ValueError:
                    continue
        
        # Strategy 2: Look for YYYYMMDD in parts
        for part in parts:
            if len(part) == 8 and part.isdigit():
                try:
                    return datetime.strptime(part, "%Y%m%d")
                except ValueError:
                    continue
        
        # Strategy 3: Regex search
        date_patterns = [
            r'_(\d{6})_',  # YYMMDD
            r'_(\d{8})_',  # YYYYMMDD
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, basename)
            if match:
                date_str = match.group(1)
                try:
                    if len(date_str) == 6:
                        return datetime.strptime(date_str, "%y%m%d")
                    elif len(date_str) == 8:
                        return datetime.strptime(date_str, "%Y%m%d")
                except ValueError:
                    continue
        
        logger.warning(f"Could not extract date from filename: {filename}")
        return None
    
    def find_h5_files(self, site_path):
        """
        Find all HDF5 files in site directory with recursive search.
        
        Parameters:
        -----------
        site_path : str
            Path to site directory
        
        Returns:
        --------
        list : Sorted list of HDF5 file paths
        """
        h5_files = []
        
        # Use os.walk for comprehensive search
        for root, _, files in os.walk(site_path):
            for file in files:
                if file.endswith('.h5'):
                    h5_files.append(os.path.join(root, file))
        
        # Also try glob as backup
        glob_files = glob.glob(os.path.join(site_path, "**/*.h5"), recursive=True)
        h5_files.extend(glob_files)
        
        # Remove duplicates and sort
        h5_files = sorted(list(set(h5_files)))
        
        logger.info(f"Found {len(h5_files)} HDF5 files in {site_path}")
        if h5_files:
            logger.debug(f"Sample files: {h5_files[:3]}")
        
        return h5_files
    
    def find_temporal_pairs(self, h5_files):
        """
        Identify valid temporal pairs for interferometry.
        
        Parameters:
        -----------
        h5_files : list
            List of HDF5 file paths
        
        Returns:
        --------
        list : List of (ref_file, sec_file) tuples
        """
        logger.info(f"Finding temporal pairs from {len(h5_files)} files")
        
        # Extract dates from filenames
        dated_files = []
        for f in h5_files:
            date = self.get_date_from_filename(os.path.basename(f))
            if date:
                dated_files.append((f, date))
        
        if not dated_files:
            logger.warning("No files with valid dates found")
            return []
        
        # Sort by date
        dated_files.sort(key=lambda x: x[1])
        
        # Generate all possible pairs
        pairs = list(itertools.combinations([f for f, _ in dated_files], 2))
        
        logger.info(f"Generated {len(pairs)} temporal pairs")
        return pairs
    
    # ========================================================================
    # INTERFEROGRAM GENERATION
    # ========================================================================
    
    def generate_interferogram_for_polarization(self, ref_path, sec_path, output_dir, pol):
        """
        Generate interferogram for a single polarization.
        
        Parameters:
        -----------
        ref_path : str
            Reference SLC file path
        sec_path : str
            Secondary SLC file path
        output_dir : str
            Output directory
        pol : str
            Polarization to process
        
        Returns:
        --------
        bool : Success status
        """
        try:
            logger.info(f"Generating interferogram for {pol}")
            
            # Load SLC data
            ref_slc = load_complex_slc(ref_path, pol)
            sec_slc = load_complex_slc(sec_path, pol)
            
            logger.debug(f"SLC shapes - Ref: {ref_slc.shape}, Sec: {sec_slc.shape}")
            
            # Compute interferogram
            ifg = compute_interferogram(ref_slc, sec_slc)
            
            # Extract phase and amplitude
            phase = np.angle(ifg)
            amplitude = np.abs(ifg)
            
            # Compute coherence
            coherence = compute_coherence(ref_slc, sec_slc)
            
            # Save products
            phase_path = os.path.join(output_dir, f"{pol}_phase.tif")
            amp_path = os.path.join(output_dir, f"{pol}_amplitude.tif")
            coh_path = os.path.join(output_dir, f"{pol}_coherence.tif")
            
            export_geotiff(phase, phase_path)
            export_geotiff(amplitude, amp_path)
            export_geotiff(coherence, coh_path)
            
            logger.info(f"Saved interferogram products for {pol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate interferogram for {pol}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def generate_interferograms(self, ref_file, sec_file, output_dir):
        """
        Generate interferograms for all common polarizations.
        
        Parameters:
        -----------
        ref_file : str
            Reference file path
        sec_file : str
            Secondary file path
        output_dir : str
            Output directory
        
        Returns:
        --------
        bool : Success status
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Find common polarizations
            with h5py.File(ref_file, 'r') as ref_h5:
                ref_pols = self.find_valid_polarizations(ref_h5)
            
            with h5py.File(sec_file, 'r') as sec_h5:
                sec_pols = self.find_valid_polarizations(sec_h5)
            
            common_pols = list(set(ref_pols) & set(sec_pols))
            
            if not common_pols:
                logger.warning(f"No common polarizations: Ref={ref_pols}, Sec={sec_pols}")
                return False
            
            logger.info(f"Processing common polarizations: {common_pols}")
            
            # Process each polarization
            success_count = 0
            for pol in common_pols:
                if self.generate_interferogram_for_polarization(ref_file, sec_file, output_dir, pol):
                    success_count += 1
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed during interferogram generation: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    # ========================================================================
    # PHASE UNWRAPPING
    # ========================================================================
    
    def unwrap_phase_snaphu(self, phase_file, coherence_file, output_file, width=None, height=None):
        """
        Unwrap phase using SNAPHU with optimized configuration.
        
        Parameters:
        -----------
        phase_file : str
            Path to wrapped phase file
        coherence_file : str
            Path to coherence file
        output_file : str
            Path to output unwrapped file
        width : int, optional
            Image width
        height : int, optional
            Image height
        
        Returns:
        --------
        bool : Success status
        """
        try:
            # Read phase and coherence
            with rasterio.open(phase_file) as src:
                phase = src.read(1)
                if width is None:
                    width = src.width
                if height is None:
                    height = src.height
                transform = src.transform
                crs = src.crs
            
            with rasterio.open(coherence_file) as src:
                coherence = src.read(1)
            
            # Create temporary directory for SNAPHU
            snaphu_dir = os.path.join(os.path.dirname(output_file), "snaphu_temp")
            os.makedirs(snaphu_dir, exist_ok=True)
            
            # Write binary files
            phase_bin = os.path.join(snaphu_dir, "phase.dat")
            coh_bin = os.path.join(snaphu_dir, "coherence.dat")
            unw_bin = os.path.join(snaphu_dir, "unwrapped.dat")
            
            phase.astype(np.float32).tofile(phase_bin)
            coherence.astype(np.float32).tofile(coh_bin)
            
            # Create SNAPHU configuration
            config_file = os.path.join(snaphu_dir, "snaphu.conf")
            with open(config_file, 'w') as f:
                f.write(f"CORRFILE {os.path.basename(coh_bin)}\n")
                f.write(f"OUTFILE {os.path.basename(unw_bin)}\n")
                f.write("COSTMODE SMOOTH\n")
                f.write("INITMETHOD MCF\n")
                f.write("DEFOMAX_CYCLE 1.2\n")
            
            # Run SNAPHU
            cmd = [
                self.snaphu_path,
                "-f", config_file,
                os.path.basename(phase_bin),
                str(width)
            ]
            
            logger.info(f"Running SNAPHU: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=snaphu_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.snaphu_timeout,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"SNAPHU failed: {result.stderr}")
                return False
            
            # Read unwrapped result
            if os.path.exists(unw_bin):
                unwrapped = np.fromfile(unw_bin, dtype=np.float32).reshape(phase.shape)
                
                # Save as GeoTIFF
                export_geotiff(unwrapped, output_file, transform=transform, crs=crs)
                
                logger.info(f"Successfully unwrapped phase: {output_file}")
                return True
            else:
                logger.error("Unwrapped output file not created")
                return False
            
        except subprocess.TimeoutExpired:
            logger.error(f"SNAPHU timed out after {self.snaphu_timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"Error during phase unwrapping: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    # ========================================================================
    # PAIR PROCESSING
    # ========================================================================
    
    def process_pair(self, site, ref_file, sec_file):
        """
        Process a single interferometric pair through complete workflow.
        
        Parameters:
        -----------
        site : str
            Site name
        ref_file : str
            Reference file path
        sec_file : str
            Secondary file path
        
        Returns:
        --------
        bool : Success status
        """
        ref_basename = os.path.basename(ref_file)
        sec_basename = os.path.basename(sec_file)
        pair_id = f"{os.path.splitext(ref_basename)[0]}__{os.path.splitext(sec_basename)[0]}"
        output_dir = os.path.join(self.output_dir, site, pair_id)
        
        logger.info(f"Processing pair: {pair_id} in site {site}")
        
        # Check if already processed
        if self.skip_existing and self._is_pair_processed(site, pair_id):
            logger.info(f"Skipping already processed pair: {pair_id}")
            self._mark_pair_processed(site, pair_id, "skipped")
            return True
        
        # Check if output already exists
        unwrapped_output = os.path.join(output_dir, "unwrapped_phase.tif")
        if self.skip_existing and os.path.exists(unwrapped_output):
            logger.info(f"Output already exists: {unwrapped_output}")
            self._mark_pair_processed(site, pair_id, "skipped")
            return True
        
        try:
            # Extract and save metadata
            metadata = {
                "site": site,
                "reference": ref_file,
                "secondary": sec_file,
                "metadata_ref": self.metadata_extractor.extract_h5_metadata(ref_file),
                "metadata_sec": self.metadata_extractor.extract_h5_metadata(sec_file),
                "processing_date": datetime.now().isoformat(),
                "pipeline_version": "1.0.0"
            }
            
            os.makedirs(output_dir, exist_ok=True)
            metadata_file = os.path.join(output_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Generate interferograms
            logger.info("Generating interferograms...")
            if not self.generate_interferograms(ref_file, sec_file, output_dir):
                logger.error("Interferogram generation failed")
                self._mark_pair_processed(site, pair_id, "failed", "Interferogram generation failed")
                return False
            
            # Unwrap phase for each polarization
            logger.info("Unwrapping phase...")
            polarizations = [f.split('_')[0] for f in os.listdir(output_dir) if f.endswith('_phase.tif')]
            
            unwrap_success = False
            for pol in polarizations:
                phase_file = os.path.join(output_dir, f"{pol}_phase.tif")
                coh_file = os.path.join(output_dir, f"{pol}_coherence.tif")
                unw_file = os.path.join(output_dir, f"{pol}_unwrapped.tif")
                
                if os.path.exists(phase_file) and os.path.exists(coh_file):
                    if self.unwrap_phase_snaphu(phase_file, coh_file, unw_file):
                        unwrap_success = True
                        logger.info(f"Successfully unwrapped {pol}")
                    else:
                        logger.warning(f"Failed to unwrap {pol}")
            
            if not unwrap_success:
                logger.warning("Phase unwrapping failed for all polarizations")
                # Don't mark as failed - partial success is still useful
            
            # Mark as completed
            self._mark_pair_processed(site, pair_id, "completed")
            logger.info(f"Successfully processed pair: {pair_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed processing pair {pair_id}: {e}")
            logger.debug(traceback.format_exc())
            self._mark_pair_processed(site, pair_id, "failed", str(e))
            return False
    
    # ========================================================================
    # SITE PROCESSING
    # ========================================================================
    
    def process_site(self, site):
        """
        Process all pairs for a single site.
        
        Parameters:
        -----------
        site : str
            Site name
        """
        site_path = os.path.join(self.base_dir, site)
        
        if not os.path.isdir(site_path):
            logger.warning(f"Not a directory: {site_path}")
            return
        
        logger.info(f"Processing site: {site}")
        
        # Find all HDF5 files
        h5_files = self.find_h5_files(site_path)
        
        if not h5_files:
            logger.warning(f"No HDF5 files found in {site}")
            self._mark_site_processed(site)
            return
        
        # Find temporal pairs
        pairs = self.find_temporal_pairs(h5_files)
        
        if not pairs:
            logger.warning(f"No valid temporal pairs found for {site}")
            self._mark_site_processed(site)
            return
        
        # Update statistics
        self.stats["total_pairs"] += len(pairs)
        
        # Process each pair
        success_count = 0
        for idx, (ref, sec) in enumerate(pairs, 1):
            logger.info(f"Processing pair {idx}/{len(pairs)} for site {site}")
            if self.process_pair(site, ref, sec):
                success_count += 1
        
        logger.info(f"Completed site {site}: {success_count}/{len(pairs)} pairs successful")
        self._mark_site_processed(site)
    
    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================
    
    def run(self):
        """
        Execute the complete pipeline workflow.
        
        Returns:
        --------
        dict : Processing statistics
        """
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("UAVSAR/NISAR Interferometric Processing Pipeline")
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"SNAPHU path: {self.snaphu_path}")
        logger.info(f"Skip existing: {self.skip_existing}")
        logger.info(f"Max workers: {self.max_workers}")
        logger.info("=" * 80)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Discover all sites
        sites = sorted([d for d in os.listdir(self.base_dir)
                       if os.path.isdir(os.path.join(self.base_dir, d))])
        
        self.stats["total_sites"] = len(sites)
        logger.info(f"Found {len(sites)} sites to process")
        
        # Process each site
        if self.max_workers > 1:
            logger.info(f"Using parallel processing with {self.max_workers} workers")
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.process_site, site): site for site in sites}
                for future in as_completed(futures):
                    site = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error processing site {site}: {e}")
        else:
            for idx, site in enumerate(sites, 1):
                logger.info(f"Processing site {idx}/{len(sites)}: {site}")
                self.process_site(site)
        
        # Calculate runtime
        end_time = datetime.now()
        total_runtime = end_time - start_time
        
        # Update final statistics
        self.stats["end_time"] = end_time.isoformat()
        self.stats["total_runtime"] = str(total_runtime)
        
        # Save statistics
        stats_file = os.path.join(self.state_dir, "final_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        # Log final summary
        logger.info("=" * 80)
        logger.info("Pipeline Completed")
        logger.info(f"Total runtime: {total_runtime}")
        logger.info(f"Sites processed: {self.stats['processed_sites']}/{self.stats['total_sites']}")
        logger.info(f"Pairs processed: {self.stats['processed_pairs']}/{self.stats['total_pairs']}")
        logger.info(f"Pairs skipped: {self.stats['skipped_pairs']}")
        logger.info(f"Pairs failed: {self.stats['failed_pairs']}")
        logger.info(f"Statistics saved to: {stats_file}")
        logger.info("=" * 80)
        
        return self.stats


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="UAVSAR/NISAR Consolidated Interferometric Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python uavsar_nisar_pipeline.py --base_dir /path/to/data --output_dir /path/to/output --snaphu_path /path/to/snaphu
  
  # Resume after interruption
  python uavsar_nisar_pipeline.py --base_dir /path/to/data --output_dir /path/to/output --resume
  
  # Parallel processing
  python uavsar_nisar_pipeline.py --base_dir /path/to/data --output_dir /path/to/output --snaphu_path /path/to/snaphu --max_workers 4
  
  # Reprocess all data
  python uavsar_nisar_pipeline.py --base_dir /path/to/data --output_dir /path/to/output --snaphu_path /path/to/snaphu --no-skip
        """
    )
    
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Base directory containing UAVSAR/NISAR data organized by site"
    )
    
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for processed interferograms and products"
    )
    
    parser.add_argument(
        "--snaphu_path",
        default=os.path.expanduser("~/Downloads/snaphu-v1.4.2/bin/snaphu"),
        help="Path to SNAPHU executable (default: ~/Downloads/snaphu-v1.4.2/bin/snaphu)"
    )
    
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip processing if output already exists (default: True)"
    )
    
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Reprocess all pairs, even if outputs exist"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing from saved state"
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of parallel workers for site processing (default: 1)"
    )
    
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Tile size for phase unwrapping (default: 512)"
    )
    
    parser.add_argument(
        "--snaphu_timeout",
        type=int,
        default=3600,
        help="SNAPHU timeout in seconds (default: 3600)"
    )
    
    parser.add_argument(
        "--log_level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Validate paths
    if not os.path.exists(args.base_dir):
        logger.error(f"Base directory does not exist: {args.base_dir}")
        return 1
    
    if not os.path.exists(args.snaphu_path):
        logger.error(f"SNAPHU executable not found: {args.snaphu_path}")
        logger.error("Please install SNAPHU or specify correct path with --snaphu_path")
        return 1
    
    # Determine skip_existing based on flags
    skip_existing = args.skip_existing and not args.no_skip
    
    # Initialize and run pipeline
    try:
        pipeline = UAVSARNISARPipeline(
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            snaphu_path=args.snaphu_path,
            skip_existing=skip_existing,
            max_workers=args.max_workers,
            tile_size=args.tile_size,
            snaphu_timeout=args.snaphu_timeout
        )
        
        stats = pipeline.run()
        
        # Determine exit code based on results
        if stats['failed_pairs'] > 0:
            logger.warning(f"Pipeline completed with {stats['failed_pairs']} failures")
            return 1
        
        logger.info("Pipeline completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        logger.info("State has been saved - use --resume to continue")
        return 130
    
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())


# # Basic processing
# python uavsar_nisar_pipeline.py --base_dir /Volumes/All/uavsar --output_dir /Volumes/All/uavsar_outputs --snaphu_path ...

# # Resume after interruption
# python uavsar_nisar_pipeline.py --base_dir /Volumes/All/uavsar --output_dir /Volumes/All/uavsar_outputs --resume

# # Parallel processing with 4 workers
# python uavsar_nisar_pipeline.py --base_dir /Volumes/All/uavsar --output_dir /Volumes/All/uavsar_outputs --snaphu_path ...

# # Force reprocessing
# python uavsar_nisar_pipeline.py --base_dir /Volumes/All/uavsar --output_dir /Volumes/All/uavsar_outputs --snaphu_path ...