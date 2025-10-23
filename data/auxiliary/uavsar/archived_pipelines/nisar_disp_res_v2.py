#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from rasterio.transform import Affine
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import multiprocessing as mp
from functools import partial

# Constants
WAVELENGTH_LBAND = 0.238403545  # meters
TARGET_RES = 30  # meters
UNWRAPPED_SUFFIX = '_unw.tif'
PROCESSED_DIR = '/Volumes/All/nisarsims_processed/'
SOURCE_DIR = '/Volumes/All/nisar/nisarsims/'
ANN_DIR = '/Volumes/All/nisar/nisarsims_source'

def extract_acquisition_ids(unw_path, debug=False):
    """
    Extract acquisition IDs from interferogram filename
    with fallback for test files
    """
    basename = os.path.basename(unw_path)
    
    # Special case for test files
    if basename == "unwrapped.tif" or not "__" in basename:
        if debug:
            print(f"[DEBUG] Using placeholder IDs for test file: {basename}")
        # Use the directory name as a placeholder site name
        dir_name = os.path.basename(os.path.dirname(unw_path))
        # Create generic acquisition IDs
        return f"{dir_name}_test_acq1", f"{dir_name}_test_acq2"
    
    # Standard case - Split by double underscore to get the two acquisition parts
    parts = basename.split('__')
    
    if len(parts) < 2 or '_frequency' not in parts[1]:
        raise ValueError(f"Could not extract pair IDs from {basename}")
    
    acq1 = parts[0]
    acq2 = parts[1].split('_frequency')[0]
    
    return acq1, acq2

def find_ann_file(acquisition_id, debug=False):
    """
    Find annotation file by matching the initial characters with files in the annotation directory
    with special handling for CD/CX pattern variations
    """
    if debug:
        print(f"[DEBUG] Searching for annotation file for {acquisition_id}")
    
    # Extract the site name (first part of the acquisition ID before underscore)
    site_match = re.match(r'^([a-zA-Z]+)', acquisition_id)
    if not site_match:
        raise ValueError(f"Could not extract site name from {acquisition_id}")
    
    site_name = site_match.group(1)
    
    # Handle the CD/CX pattern variation
    # If acquisition_id contains 'CD', create an alternative ID with 'CX'
    alternative_ids = [acquisition_id]
    if '_CD_' in acquisition_id:
        alternative_ids.append(acquisition_id.replace('_CD_', '_CX_'))
    elif '_CX_' in acquisition_id:
        alternative_ids.append(acquisition_id.replace('_CX_', '_CD_'))
    
    # Get base IDs (everything before _C[X|D]_)
    base_ids = []
    for aid in alternative_ids:
        parts = re.split(r'_C[XD]_', aid)
        if len(parts) > 1:
            base_ids.append(parts[0])
    
    if not base_ids:
        base_ids = [acquisition_id.split('_CX_')[0] if '_CX_' in acquisition_id
                    else acquisition_id.split('_CD_')[0] if '_CD_' in acquisition_id
                    else acquisition_id]
    
    if debug:
        print(f"[DEBUG] Base IDs to search for: {base_ids}")
    
    # Look for annotation files in the site's directory
    site_dir = os.path.join(ANN_DIR, site_name)
    
    if debug:
        print(f"[DEBUG] Searching in directory: {site_dir}")
    
    # Try all possible base IDs and search patterns
    for base_id in base_ids:
        # Search for directories containing the base_id
        for root, dirs, files in os.walk(site_dir):
            # Look for .ann files in the 'ann' subdirectory
            ann_dir = os.path.join(root, 'ann')
            if os.path.exists(ann_dir):
                ann_files = glob.glob(os.path.join(ann_dir, '*.ann'))
                
                if debug:
                    print(f"[DEBUG] Found {len(ann_files)} .ann files in {ann_dir}")
                
                # Try to find the best match based on filename similarity
                best_match = None
                best_score = 0
                
                for ann_file in ann_files:
                    ann_basename = os.path.basename(ann_file)
                    
                    # Check if the file starts with the base ID
                    if ann_basename.startswith(base_id):
                        if debug:
                            print(f"[DEBUG] Found direct match: {ann_basename}")
                        return ann_file
                    
                    # Calculate similarity for all parts of the ID except for CX/CD variation
                    score = filename_similarity(ann_basename, base_id)
                    if score > best_score:
                        best_score = score
                        best_match = ann_file
                
                # Return the best match if it's a good match
                if best_match and best_score > 0.8:
                    if debug:
                        print(f"[DEBUG] Found best match: {os.path.basename(best_match)} with score {best_score}")
                    return best_match
    
    # If still no match, try a more aggressive search with just the core ID parts
    # Extract core parts: site_name, track number, frame number, orbit number, date
    core_parts_match = re.match(r'^([a-zA-Z]+)_(\d+)_(\d+)_(\d+)_(\d+)', acquisition_id)
    if core_parts_match:
        core_id = '_'.join(core_parts_match.groups())
        
        if debug:
            print(f"[DEBUG] Trying core ID search with: {core_id}")
        
        # Look for any .ann file containing these core parts
        for root, dirs, files in os.walk(site_dir):
            for filename in files:
                if filename.endswith('.ann') and core_id in filename:
                    if debug:
                        print(f"[DEBUG] Found match using core ID: {filename}")
                    return os.path.join(root, filename)
    
    # Special handling for scotty dataset with CD/CX mismatch
    if 'scotty' in acquisition_id and '_CD_' in acquisition_id:
        # Try to find a corresponding file with similar date/orbit information but different CX/CD pattern
        alternative_pattern = acquisition_id.replace('_CD_', '_CX_')
        scotty_files = glob.glob(os.path.join(ANN_DIR, 'scotty', '*', 'ann', '*.ann'))
        
        if debug:
            print(f"[DEBUG] Special scotty handling: looking for pattern similar to {alternative_pattern}")
            print(f"[DEBUG] Found {len(scotty_files)} potential scotty annotation files")
        
        for ann_file in scotty_files:
            ann_basename = os.path.basename(ann_file)
            # Check if key parts match (ignore the CX/CD difference)
            parts_to_match = acquisition_id.split('_CD_')[0].split('_')[1:4]  # Track, frame, orbit
            if all(part in ann_basename for part in parts_to_match):
                print(f"[INFO] Found alternative match for {acquisition_id}: {ann_basename}")
                return ann_file
    
    if debug:
        print(f"[DEBUG] No annotation file found for {acquisition_id}")
    
    return None

def filename_similarity(filename, base_id):
    """
    Calculate similarity between filename and base_id by comparing parts
    """
    # Split the filenames into parts by underscores
    file_parts = filename.split('_')
    base_parts = base_id.split('_')
    
    # Count matching parts
    matches = sum(1 for fp, bp in zip(file_parts, base_parts) if fp == bp)
    
    # Calculate similarity score
    total_parts = max(len(file_parts), len(base_parts))
    return matches / total_parts if total_parts > 0 else 0

def parse_incidence_angle(ann_path, debug=False):
    """
    Extract incidence angle information from annotation file
    """
    if not ann_path:
        if debug:
            print("[DEBUG] No annotation file provided, using default angle")
        return 35.0  # Default mid-range incidence angle
    
    try:
        with open(ann_path, 'r') as f:
            text = f.read()
    except Exception as e:
        if debug:
            print(f"[DEBUG] Error reading annotation file: {e}")
        return 35.0  # Default if file can't be read
    
    if debug:
        print(f"[DEBUG] Successfully read annotation file: {ann_path}")
    
    # Updated regex patterns to match exactly the format observed in the files
    near_pattern = r'Simulated NISAR Near Range Incidence Angle\s+\(deg\)\s+=\s+([0-9.]+)'
    far_pattern = r'Simulated NISAR Far Range Incidence Angle\s+\(deg\)\s+=\s+([0-9.]+)'
    
    near_match = re.search(near_pattern, text)
    far_match = re.search(far_pattern, text)
    
    if near_match and far_match:
        near_angle = float(near_match.group(1))
        far_angle = float(far_match.group(1))
        mean_angle = (near_angle + far_angle) / 2.0
        print(f"[INFO] Found incidence angles in {os.path.basename(ann_path)}: Near={near_angle:.2f}°, Far={far_angle:.2f}°, Mean={mean_angle:.2f}°")
        return mean_angle
    
    # Try alternative patterns if the first ones don't match
    alternative_patterns = [
        (r'Near Range Incidence Angle\s+\(deg\)\s+=\s+([0-9.]+)',
         r'Far Range Incidence Angle\s+\(deg\)\s+=\s+([0-9.]+)'),
        (r'Near_Incidence_Angle\s*=\s*([0-9.]+)',
         r'Far_Incidence_Angle\s*=\s*([0-9.]+)'),
    ]
    
    for near_pat, far_pat in alternative_patterns:
        near_match = re.search(near_pat, text)
        far_match = re.search(far_pat, text)
        
        if near_match and far_match:
            near_angle = float(near_match.group(1))
            far_angle = float(far_match.group(1))
            mean_angle = (near_angle + far_angle) / 2.0
            print(f"[INFO] Found incidence angles using alternative pattern in {os.path.basename(ann_path)}: Near={near_angle:.2f}°, Far={far_angle:.2f}°, Mean={mean_angle:.2f}°")
            return mean_angle
    
    if debug:
        # If we still can't find the angles, print a portion of the file to help diagnose
        print(f"[DEBUG] Content sample from {ann_path}:")
        lines = text.splitlines()
        angle_related_lines = []
        for i, line in enumerate(lines):
            if "incidence" in line.lower() or "angle" in line.lower():
                angle_related_lines.append(f"Line {i}: {line}")
                
        if angle_related_lines:
            for line in angle_related_lines[:10]:  # Show up to 10 angle-related lines
                print(line)
        else:
            print("[DEBUG] No lines containing 'incidence' or 'angle' found in annotation file")
    
    # Default to typical L-band SAR incidence angles if not found
    print(f"[WARNING] Could not find incidence angles in {os.path.basename(ann_path)}, using default values")
    return 35.0  # Default mid-range incidence angle
    
def find_coherence_file(unw_path):
    """Find the corresponding coherence file for an unwrapped interferogram"""
    # Get the directory and base filename without extension
    directory = os.path.dirname(unw_path)
    base_filename = os.path.basename(unw_path).replace('_unw.tif', '')
    
    # Check for .cor.tif version
    cor_path = os.path.join(directory, f"{base_filename}.cor.tif")
    if os.path.exists(cor_path):
        return cor_path
        
    # Check for _coherence.tif version
    coherence_path = os.path.join(directory, f"{base_filename}_coherence.tif")
    if os.path.exists(coherence_path):
        return coherence_path
    
    # Not found
    return None

def extract_coordinates_from_annotation(ann_file):
    """Extract corner coordinates from annotation file using the exact format from the files"""
    if not ann_file or not os.path.exists(ann_file):
        return None
    
    try:
        with open(ann_file, 'r') as f:
            content = f.read()
        
        # Look for the specific coordinate format from the files
        patterns = {
            'ul_lat': r'Approximate Upper Left Latitude\s*\(deg\)\s*=\s*([-+]?\d*\.\d+)',
            'ul_lon': r'Approximate Upper Left Longitude\s*\(deg\)\s*=\s*([-+]?\d*\.\d+)',
            'ur_lat': r'Approximate Upper Right Latitude\s*\(deg\)\s*=\s*([-+]?\d*\.\d+)',
            'ur_lon': r'Approximate Upper Right Longitude\s*\(deg\)\s*=\s*([-+]?\d*\.\d+)',
            'll_lat': r'Approximate Lower Left Latitude\s*\(deg\)\s*=\s*([-+]?\d*\.\d+)',
            'll_lon': r'Approximate Lower Left Longitude\s*\(deg\)\s*=\s*([-+]?\d*\.\d+)',
            'lr_lat': r'Approximate Lower Right Latitude\s*\(deg\)\s*=\s*([-+]?\d*\.\d+)',
            'lr_lon': r'Approximate Lower Right Longitude\s*\(deg\)\s*=\s*([-+]?\d*\.\d+)'
        }
        
        coords = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                coords[key] = float(match.group(1))
        
        # Check if we have all the coordinates
        if len(coords) == 8:
            print(f"[INFO] Found all coordinates in {os.path.basename(ann_file)}")
            return coords
        else:
            print(f"[WARNING] Only found {len(coords)}/8 coordinates in {os.path.basename(ann_file)}")
            for key in patterns.keys():
                if key not in coords:
                    print(f"[DEBUG] Missing: {key}")
            return coords if len(coords) >= 4 else None  # Return partial coords if we have at least some
            
    except Exception as e:
        print(f"[ERROR] Failed to extract coordinates from {ann_file}: {e}")
        return None
        
#def process_single_file(unw_path, coherence_path=None, output_dir=None,
#                        coherence_threshold=0.4, filter_strength=0.5, debug=True, test_mode=False):
#    """
#    Process a single interferogram with coherence masking and filtering
#    """
#    if output_dir is None:
#        output_dir = PROCESSED_DIR
#    
#    os.makedirs(output_dir, exist_ok=True)
#    
#    print(f"{'='*80}\n[INFO] Processing single file: {unw_path}\n{'='*80}")
#    
#    try:
#        # Extract acquisition IDs (modified for test files)
#        acq1, acq2 = extract_acquisition_ids(unw_path, debug)
#        print(f"[INFO] Processing pair: {acq1} and {acq2}")
#        
#        # In test mode, use default incidence angle and skip annotation file lookup
#        if test_mode:
#            print("[INFO] Running in test mode - using default incidence angle")
#            mean_theta = 35.0
#            ann1 = None
#            ann2 = None
#        else:
#            # Find annotation files
#            ann1 = find_ann_file(acq1, debug)
#            ann2 = find_ann_file(acq2, debug)
#            
#            if ann1:
#                print(f"[INFO] Found annotation file for acq1: {os.path.basename(ann1)}")
#            else:
#                print(f"[WARNING] No annotation file found for {acq1}, will use default values")
#                
#            if ann2:
#                print(f"[INFO] Found annotation file for acq2: {os.path.basename(ann2)}")
#            else:
#                print(f"[WARNING] No annotation file found for {acq2}, will use default values")
#            
#            # Get incidence angles
#            theta1 = parse_incidence_angle(ann1, debug)
#            theta2 = parse_incidence_angle(ann2, debug)
#            mean_theta = (theta1 + theta2) / 2.0
#        
#        print(f"[INFO] Using incidence angle: {mean_theta:.2f} degrees")
#        
#        # Automatically look for coherence file if not provided
#        if not coherence_path:
#            coherence_path = find_coherence_file(unw_path)
#            if coherence_path:
#                print(f"[INFO] Found coherence file: {os.path.basename(coherence_path)}")
#        
#        # Read unwrapped phase data
#        print(f"[INFO] Reading unwrapped phase data from {unw_path}")
#        try:
#            with rasterio.open(unw_path) as src:
#                phase = src.read(1)
#                profile = src.profile.copy()
#                transform = src.transform
#                
#                # Extract geospatial info
#                crs = src.crs
#                bounds = src.bounds
#                
#                # Add diagnostics for phase data
#                valid_phase = phase[np.isfinite(phase)]
#                if len(valid_phase) > 0:
#                    print(f"[INFO] Phase values range: {np.min(valid_phase):.6f} to {np.max(valid_phase):.6f}")
#                    print(f"[INFO] Phase mean: {np.mean(valid_phase):.6f}")
#                    print(f"[INFO] Phase non-zero: {np.sum(valid_phase != 0)} out of {len(valid_phase)}")
#                
#                if crs is not None:
#                    print(f"[INFO] Input CRS detected: {crs.to_string()}")
#                else:
#                    print(f"[WARNING] No CRS information found in input file")
#                
#                # Check if the transform is meaningful (not just pixel coordinates)
#                has_valid_transform = (transform[0] != 1.0 or transform[4] != -1.0 or
#                                      transform[1] != 0.0 or transform[2] != 0.0 or
#                                      transform[3] != 0.0 or transform[5] != 0.0)
#                
#                if not has_valid_transform:
#                    print("[WARNING] Input data appears to be in pixel coordinates rather than geographic coordinates")
#                    
#                    # Try to extract coordinates from annotation files using the new function
#                    coords = None
#                    if ann1:
#                        coords = extract_coordinates_from_annotation(ann1)
#                    
#                    if not coords and ann2:
#                        coords = extract_coordinates_from_annotation(ann2)
#                    
#                    if coords and len(coords) >= 4:
#                        # Calculate min/max lat/lon based on available coordinates
#                        lat_keys = [k for k in coords.keys() if 'lat' in k]
#                        lon_keys = [k for k in coords.keys() if 'lon' in k]
#                        
#                        if lat_keys and lon_keys:
#                            min_lat = min(coords[k] for k in lat_keys)
#                            max_lat = max(coords[k] for k in lat_keys)
#                            min_lon = min(coords[k] for k in lon_keys)
#                            max_lon = max(coords[k] for k in lon_keys)
#                            
#                            # Calculate transform
#                            width, height = profile['width'], profile['height']
#                            x_res = (max_lon - min_lon) / width
#                            y_res = (max_lat - min_lat) / height
#                            
#                            # Create a proper geotransform
#                            # IMPORTANT: Use full precision for these values!
#                            transform = Affine(x_res, 0.0, min_lon,
#                                             0.0, -y_res, max_lat)
#                            
#                            # Update CRS to ensure it's properly set
#                            if not crs:
#                                crs = rasterio.crs.CRS.from_epsg(4326)
#                            
#                            print(f"[INFO] Created geotransform from corner coordinates")
#                            print(f"[INFO] Lat range: [{min_lat:.6f}, {max_lat:.6f}], Lon range: [{min_lon:.6f}, {max_l...
#                            print(f"[INFO] Resolution: x={x_res:.8f}, y={y_res:.8f} degrees")
#                            print(f"[INFO] Transform: {transform}")
#                            
#                            # Add extra debug info
#                            print(f"[DEBUG] Transform values: a={transform.a:.10f}, b={transform.b:.10f}, c={transform....
#                        else:
#                            print("[WARNING] Insufficient coordinate information extracted")
#                    else:
#                        print("[WARNING] Could not find coordinate information in annotation files")
#                else:
#                    print(f"[INFO] Using existing geotransform from input file: {transform}")
#        except Exception as e:
#            print(f"[ERROR] Failed to read file {unw_path}: {e}")
#            return False
#        
#        # Read coherence data if available
#        coherence = None
#        if coherence_path and os.path.exists(coherence_path):
#            print(f"[INFO] Reading coherence data from {coherence_path}")
#            try:
#                with rasterio.open(coherence_path) as src:
#                    coherence = src.read(1)
#                    if coherence.shape != phase.shape:
#                        print(f"[WARNING] Coherence shape {coherence.shape} doesn't match phase shape {phase.shape}")
#                        coherence = None
#                    else:
#                        print(f"[INFO] Coherence data loaded successfully")
#                        # Print basic stats about coherence
#                        valid_coh = coherence[np.isfinite(coherence)]
#                        if len(valid_coh) > 0:
#                            print(f"[INFO] Coherence range: {np.min(valid_coh):.4f} to {np.max(valid_coh):.4f}")
#                            print(f"[INFO] Coherence mean: {np.mean(valid_coh):.4f}")
#                            print(f"[INFO] Coherence values above threshold ({coherence_threshold}): {np.sum(valid_coh ...
#            except Exception as e:
#                print(f"[WARNING] Failed to read coherence file {coherence_path}: {e}")
#                coherence = None
#        
#        # If no coherence file is available, create a dummy coherence of 1.0 (no masking)
#        if coherence is None:
#            print("[INFO] No coherence data found, using all pixels (no coherence masking)")
#            coherence = np.ones_like(phase)
#        
#        # Compute filtered displacement
#        print(f"[INFO] Computing displacement with coherence filtering (threshold={coherence_threshold}, filter_strengt...
#        disp = compute_displacement_with_filtering(
#            phase, coherence, WAVELENGTH_LBAND, mean_theta,
#            coherence_threshold, filter_strength, debug
#        )
#        
#        if debug:
#            # Visualize displacement data if in debug mode
#            debug_dir = os.path.join(output_dir, 'debug')
#            os.makedirs(debug_dir, exist_ok=True)
#            disp_viz_path = os.path.join(debug_dir, os.path.basename(unw_path).replace('.tif', '_disp_viz.png'))
#            visualize_data(disp, f"Displacement (m) - {os.path.basename(unw_path)}", disp_viz_path)
#        
#        # Save the original displacement data
#        orig_disp_filename = os.path.basename(unw_path).replace(UNWRAPPED_SUFFIX, '_vertical_disp_m_orig.tif')
#        orig_disp_path = os.path.join(output_dir, orig_disp_filename)
#
#        # Create a copy of the profile for the original resolution file
#        orig_profile = profile.copy()
#        orig_profile.update({
#            'dtype': 'float32',
#            'count': 1,
#            'compress': 'lzw',
#            'nodata': np.nan,
#        })
#        
#        # Ensure the profile has CRS information
#        if crs is not None:
#            orig_profile.update({'crs': crs})
#        
#        # Make sure transform is set in the profile
#        orig_profile.update({'transform': transform})
#
#        print(f"[INFO] Writing original resolution displacement file: {orig_disp_path}")
#        with rasterio.open(orig_disp_path, 'w', **orig_profile) as dst:
#            dst.write(disp.reshape(1, disp.shape[0], disp.shape[1]))
#        
#        # Check if transform appears to be in pixel coordinates
#        pixel_coords = (abs(transform[0]) == 1.0 and abs(transform[4]) == 1.0 and
#                   transform[1] == 0.0 and transform[2] == 0.0 and
#                   transform[3] == 0.0 and transform[5] == 0.0)
#    
#        if pixel_coords and debug:
#            print("[WARNING] Input data appears to be in pixel coordinates rather than geographic coordinates")
#            print("[INFO] Creating synthetic geographic transform based on target resolution")
#            
#        # For resampling with pixel coordinates, create a synthetic transform
#        if pixel_coords:
#            # Create synthetic transforms that preserve the data values
#            src_transform = transform
#            dst_transform = Affine(TARGET_RES, 0.0, 0.0,
#                                  0.0, -TARGET_RES, 0.0)
#            
#            # Create appropriately sized output array
#            width = max(1, int(profile['width'] * abs(src_transform[0]) / TARGET_RES))
#            height = max(1, int(profile['height'] * abs(src_transform[4]) / TARGET_RES))
#            
#            if debug:
#                print(f"[DEBUG] Using synthetic transforms for resampling")
#                print(f"[DEBUG] Calculated dimensions: {width} x {height}")
#        else:
#            # Use the standard method for georeferenced data
#            dst_transform, width, height = calculate_default_transform(
#                profile['crs'], profile['crs'],
#                profile['width'], profile['height'],
#                *src.bounds,
#                resolution=TARGET_RES
#            )
#            
#            if debug:
#                print(f"[DEBUG] Original dimensions: {profile['width']} x {profile['height']}")
#                print(f"[DEBUG] Resampled dimensions: {width} x {height}")
#                print(f"[DEBUG] Original transform: {transform}")
#                print(f"[DEBUG] Resampled transform: {dst_transform}")
#        
#        # Create resampled array with appropriate dimensions
#        resampled_disp = np.empty((1, height, width), dtype=np.float32)
#        
#        # Perform resampling
#        print(f"[INFO] Resampling displacement to {TARGET_RES}m resolution")
#        
#        # For pixel-based data, use zoom for proper resampling
#        if pixel_coords:
#            print("[DEBUG] Using pixel-based resampling approach")
#            
#            # For pixel-based data, we need to preserve the values during resampling
#            x_scale = profile['width'] / width
#            y_scale = profile['height'] / height
#            
#            # Modified approach:
#            # 1. Create a mask of valid (non-NaN) pixels
#            valid_mask = np.isfinite(disp)
#            
#            # 2. Replace NaNs with a specific value for resampling
#            temp_disp = np.copy(disp)
#            temp_disp[~valid_mask] = 0.0
#            
#            # 3. Resample both the data and the mask
#            resampled_disp[0] = zoom(temp_disp, (1/y_scale, 1/x_scale), order=1)
#            resampled_mask = zoom(valid_mask.astype(float), (1/y_scale, 1/x_scale), order=0) > 0.5
#            
#            # 4. Apply the resampled mask to the resampled data
#            resampled_disp[0][~resampled_mask] = np.nan
#            
#            print(f"[DEBUG] Rescaled using scipy.ndimage.zoom with factors: ({1/y_scale}, {1/x_scale})")
#            print(f"[DEBUG] Original valid pixels: {np.sum(valid_mask)} ({np.sum(valid_mask)/disp.size*100:.1f}%)")
#            print(f"[DEBUG] Resampled valid pixels: {np.sum(resampled_mask)} ({np.sum(resampled_mask)/resampled_mask.si...
#        else:
#            # Standard georeferenced approach with improved handling of NaN values
#            # 1. Create a mask of valid (non-NaN) pixels
#            valid_mask = np.isfinite(disp)
#            
#            # 2. Replace NaNs with zeros before resampling
#            temp_disp = np.copy(disp)
#            temp_disp[~valid_mask] = 0.0
#            
#            # Print additional debug info
#            print(f"[DEBUG] Original transform details: {transform}")
#            print(f"[DEBUG] Destination transform details: {dst_transform}")
#            print(f"[DEBUG] Valid pixels before resampling: {np.sum(valid_mask)} ({np.sum(valid_mask)/disp.size*100:.1f...
#            
#            # 3. Create a mask array and set it to 1 where data is valid
#            mask_array = np.zeros_like(disp, dtype=np.uint8)
#            mask_array[valid_mask] = 1
#            
#            # 4. Resample both the data and mask
#            try:
#                # Try with bilinear interpolation first
#                reproject(
#                    source=np.expand_dims(temp_disp, 0),
#                    destination=resampled_disp,
#                    src_transform=transform,
#                    src_crs=profile['crs'],
#                    dst_transform=dst_transform,
#                    dst_crs=profile['crs'],
#                    resampling=Resampling.bilinear
#                )
#                
#                # Resample the mask (0s and 1s)
#                resampled_mask_array = np.empty((1, height, width), dtype=np.uint8)
#                reproject(
#                    source=np.expand_dims(mask_array, 0),
#                    destination=resampled_mask_array,
#                    src_transform=transform,
#                    src_crs=profile['crs'],
#                    dst_transform=dst_transform,
#                    dst_crs=profile['crs'],
#                    resampling=Resampling.nearest
#                )
#                
#                # 5. Apply the mask to the resampled data
#                resampled_disp[0][resampled_mask_array[0] == 0] = np.nan
#                
#                # Check if we have any valid pixels after resampling
#                valid_resampled = np.isfinite(resampled_disp[0])
#                print(f"[DEBUG] Valid pixels after resampling: {np.sum(valid_resampled)} ({np.sum(valid_resampled)/resa...
#                
#                # If we've lost all valid pixels, fall back to nearest-neighbor resampling
#                if np.sum(valid_resampled) == 0:
#                    print("[WARNING] Bilinear resampling resulted in all NaNs, trying nearest neighbor resampling inste...
#                    
#                    # Try nearest neighbor resampling instead
#                    reproject(
#                        source=np.expand_dims(temp_disp, 0),
#                        destination=resampled_disp,
#                        src_transform=transform,
#                        src_crs=profile['crs'],
#                        dst_transform=dst_transform,
#                        dst_crs=profile['crs'],
#                        resampling=Resampling.nearest
#                    )
#                    
#                    # Apply mask again
#                    resampled_disp[0][resampled_mask_array[0] == 0] = np.nan
#                    
#                    valid_resampled = np.isfinite(resampled_disp[0])
#                    print(f"[DEBUG] Valid pixels after nearest resampling: {np.sum(valid_resampled)} ({np.sum(valid_res...
#                
#                # If still no valid pixels, fall back to zoom-based resampling like for pixel coordinates
#                if np.sum(valid_resampled) == 0:
#                    print("[WARNING] All resampling methods failed with standard approach, falling back to zoom-based r...
#                    
#                    # Fall back to zoom-based resampling
#                    x_scale = profile['width'] / width
#                    y_scale = profile['height'] / height
#                    
#                    # Resample using zoom
#                    resampled_disp[0] = zoom(temp_disp, (1/y_scale, 1/x_scale), order=1)
#                    resampled_mask = zoom(valid_mask.astype(float), (1/y_scale, 1/x_scale), order=0) > 0.5
#                    
#                    # Apply the resampled mask
#                    resampled_disp[0][~resampled_mask] = np.nan
#                    
#                    print(f"[DEBUG] Fallback: rescaled using scipy.ndimage.zoom with factors: ({1/y_scale}, {1/x_scale}...
#                    print(f"[DEBUG] Fallback: valid pixels: {np.sum(resampled_mask)} ({np.sum(resampled_mask)/resampled...
#                
#            except Exception as e:
#                print(f"[WARNING] Error during resampling: {e}")
#                print("[WARNING] Falling back to zoom-based resampling")
#                
#                # Fall back to zoom-based resampling
#                x_scale = profile['width'] / width
#                y_scale = profile['height'] / height
#                
#                # Resample using zoom
#                resampled_disp[0] = zoom(temp_disp, (1/y_scale, 1/x_scale), order=1)
#                resampled_mask = zoom(valid_mask.astype(float), (1/y_scale, 1/x_scale), order=0) > 0.5
#                
#                # Apply the resampled mask
#                resampled_disp[0][~resampled_mask] = np.nan
#                
#                print(f"[DEBUG] Fallback: rescaled using scipy.ndimage.zoom with factors: ({1/y_scale}, {1/x_scale})")
#                print(f"[DEBUG] Fallback: valid pixels: {np.sum(resampled_mask)} ({np.sum(resampled_mask)/resampled_mas...
#            
#            if debug:
#                print(f"[DEBUG] Original valid pixels: {np.sum(valid_mask)} ({np.sum(valid_mask)/disp.size*100:.1f}%)")
#                print(f"[DEBUG] Resampled valid pixels: {np.sum(resampled_mask_array[0])} ({np.sum(resampled_mask_array...
#            
#        if 'crs' in profile and profile['crs'] is not None:
#            print(f"[INFO] CRS preserved during resampling: {profile['crs'].to_string()}")
#        else:
#            print(f"[WARNING] CRS information may have been lost during resampling")
#            
#        if debug:
#            # Visualize resampled data
#            debug_dir = os.path.join(output_dir, 'debug')
#            os.makedirs(debug_dir, exist_ok=True)
#            resampled_viz_path = os.path.join(debug_dir, os.path.basename(unw_path).replace('.tif', '_resampled_viz.png...
#            visualize_data(resampled_disp[0], f"Resampled Displacement (m) - {os.path.basename(unw_path)}", resampled_v...
#            
#            # Print statistics about resampled data
#            valid_resampled = np.isfinite(resampled_disp[0])
#            print(f"[DEBUG] Resampled valid pixels: {np.sum(valid_resampled)} ({np.sum(valid_resampled)/resampled_disp[...
#            
#            if np.any(valid_resampled):
#                valid_data = resampled_disp[0][valid_resampled]
#                print(f"[DEBUG] Resampled value range: {np.min(valid_data):.4f} to {np.max(valid_data):.4f}")
#                print(f"[DEBUG] Resampled mean value: {np.mean(valid_data):.4f}")
#        
#        # Update profile for output file
#        profile.update({
#            'dtype': 'float32',
#            'height': height,
#            'width': width,
#            'transform': dst_transform,
#            'count': 1,
#            'compress': 'lzw',
#            'nodata': np.nan,
#        })
#        
#        # Ensure CRS is preserved
#        if crs is not None:
#            profile.update({'crs': crs})
#        
#        # Define output filename and path
#        out_filename = os.path.basename(unw_path).replace(UNWRAPPED_SUFFIX, f'_vertical_disp_m_{TARGET_RES}m.tif')
#        out_path = os.path.join(output_dir, out_filename)
#
#        # Write output file
#        print(f"[INFO] Writing output file: {out_path}")
#        try:
#            # Make sure the profile has all important CRS information
#            with rasterio.open(unw_path) as src_check:
#                # If the profile doesn't have CRS but the source does, use the source CRS
#                if ('crs' not in profile or profile['crs'] is None) and src_check.crs is not None:
#                    print(f"[INFO] Copying CRS from source file to output")
#                    profile['crs'] = src_check.crs
#                
#            # Add nodata value to ensure NaNs are handled properly
#            profile.update({'nodata': np.nan})
#            
#            # Write with complete profile
#            with rasterio.open(out_path, 'w', **profile) as dst:
#                dst.write(resampled_disp)
#            
#            print(f"[INFO] Successfully wrote output with preserved geospatial information")
#        except Exception as e:
#            print(f"[WARNING] Error preserving full CRS information: {e}")
#            print(f"[INFO] Falling back to basic output")
#            
#            # Basic fallback method if the above fails
#            with rasterio.open(out_path, 'w', **profile) as dst:
#                dst.write(resampled_disp)
#        
#        # Verify the output file
#        verify_result = verify_output_file(out_path)
#        if verify_result:
#            print(f"[SUCCESS] Successfully processed and verified {out_filename}")
#            return True
#        else:
#            print(f"[ERROR] Failed to produce valid output for {out_filename}")
#            return False
#        
#        try:
#            with rasterio.open(out_path) as verification:
#                if verification.crs is not None:
#                    print(f"[INFO] Output file has valid CRS: {verification.crs.to_string()}")
#                    print(f"[INFO] Output transform: {verification.transform}")
#                    return True
#                else:
#                    print(f"[WARNING] Output file is missing CRS information")
#                    return verify_result
#        except Exception as e:
#            print(f"[WARNING] Failed to verify CRS in output: {e}")
#            return verify_result
#            
#    except Exception as e:
#        print(f"[ERROR] Failed to process {unw_path}: {e}")
#        import traceback
#        traceback.print_exc()
#        return False
def process_single_file(unw_path, coherence_path=None, output_dir=None,
                        coherence_threshold=0.4, filter_strength=0.5, debug=True, test_mode=False):
    """
    Process a single interferogram with coherence masking and filtering
    """
    if output_dir is None:
        output_dir = PROCESSED_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{'='*80}\n[INFO] Processing single file: {unw_path}\n{'='*80}")
    
    try:
        # Extract acquisition IDs (modified for test files)
        acq1, acq2 = extract_acquisition_ids(unw_path, debug)
        print(f"[INFO] Processing pair: {acq1} and {acq2}")
        
        # In test mode, use default incidence angle and skip annotation file lookup
        if test_mode:
            print("[INFO] Running in test mode - using default incidence angle")
            mean_theta = 35.0
            ann1 = None
            ann2 = None
        else:
            # Find annotation files
            ann1 = find_ann_file(acq1, debug)
            ann2 = find_ann_file(acq2, debug)
            
            if ann1:
                print(f"[INFO] Found annotation file for acq1: {os.path.basename(ann1)}")
            else:
                print(f"[WARNING] No annotation file found for {acq1}, will use default values")
                
            if ann2:
                print(f"[INFO] Found annotation file for acq2: {os.path.basename(ann2)}")
            else:
                print(f"[WARNING] No annotation file found for {acq2}, will use default values")
            
            # Get incidence angles
            theta1 = parse_incidence_angle(ann1, debug)
            theta2 = parse_incidence_angle(ann2, debug)
            mean_theta = (theta1 + theta2) / 2.0
        
        print(f"[INFO] Using incidence angle: {mean_theta:.2f} degrees")
        
        # Automatically look for coherence file if not provided
        if not coherence_path:
            coherence_path = find_coherence_file(unw_path)
            if coherence_path:
                print(f"[INFO] Found coherence file: {os.path.basename(coherence_path)}")
        
        # Read unwrapped phase data
        print(f"[INFO] Reading unwrapped phase data from {unw_path}")
        try:
            with rasterio.open(unw_path) as src:
                phase = src.read(1)
                profile = src.profile.copy()
                transform = src.transform
                
                # Extract geospatial info
                crs = src.crs
                bounds = src.bounds
                
                # Add diagnostics for phase data
                valid_phase = phase[np.isfinite(phase)]
                if len(valid_phase) > 0:
                    print(f"[INFO] Phase values range: {np.min(valid_phase):.6f} to {np.max(valid_phase):.6f}")
                    print(f"[INFO] Phase mean: {np.mean(valid_phase):.6f}")
                    print(f"[INFO] Phase non-zero: {np.sum(valid_phase != 0)} out of {len(valid_phase)}")
                
                if crs is not None:
                    print(f"[INFO] Input CRS detected: {crs.to_string()}")
                else:
                    print(f"[WARNING] No CRS information found in input file")
                
                # Check if the transform is meaningful (not just pixel coordinates)
                has_valid_transform = (transform[0] != 1.0 or transform[4] != -1.0 or
                                      transform[1] != 0.0 or transform[2] != 0.0 or
                                      transform[3] != 0.0 or transform[5] != 0.0)
                
                if not has_valid_transform:
                    print("[WARNING] Input data appears to be in pixel coordinates rather than geographic coordinates")
                    
                    # Try to extract coordinates from annotation files using the new function
                    coords = None
                    if ann1:
                        coords = extract_coordinates_from_annotation(ann1)
                    
                    if not coords and ann2:
                        coords = extract_coordinates_from_annotation(ann2)
                    
                    if coords and len(coords) >= 4:
                        # Calculate min/max lat/lon based on available coordinates
                        lat_keys = [k for k in coords.keys() if 'lat' in k]
                        lon_keys = [k for k in coords.keys() if 'lon' in k]
                        
                        if lat_keys and lon_keys:
                            min_lat = min(coords[k] for k in lat_keys)
                            max_lat = max(coords[k] for k in lat_keys)
                            min_lon = min(coords[k] for k in lon_keys)
                            max_lon = max(coords[k] for k in lon_keys)
                            
                            # Calculate transform
                            width, height = profile['width'], profile['height']
                            x_res = (max_lon - min_lon) / width
                            y_res = (max_lat - min_lat) / height
                            
                            # Create a proper geotransform
                            # IMPORTANT: Use full precision for these values!
                            transform = Affine(x_res, 0.0, min_lon,
                                             0.0, -y_res, max_lat)
                            
                            # Update CRS to ensure it's properly set
                            if not crs:
                                crs = rasterio.crs.CRS.from_epsg(4326)
                            
                            print(f"[INFO] Created geotransform from corner coordinates")
                            print(f"[INFO] Lat range: [{min_lat:.6f}, {max_lat:.6f}], Lon range: [{min_lon:.6f}, {max_lon:.6f}]")
                            print(f"[INFO] Resolution: x={x_res:.8f}, y={y_res:.8f} degrees")
                            print(f"[INFO] Transform: {transform}")
                            
                            # Add extra debug info
                            print(f"[DEBUG] Transform values: a={transform.a:.10f}, b={transform.b:.10f}, c={transform.c:.10f}, d={transform.d:.10f}, e={transform.e:.10f}, f={transform.f:.10f}")
                        else:
                            print("[WARNING] Insufficient coordinate information extracted")
                    else:
                        print("[WARNING] Could not find coordinate information in annotation files")
                else:
                    print(f"[INFO] Using existing geotransform from input file: {transform}")
        except Exception as e:
            print(f"[ERROR] Failed to read file {unw_path}: {e}")
            return False
        
        # Read coherence data if available
        coherence = None
        if coherence_path and os.path.exists(coherence_path):
            print(f"[INFO] Reading coherence data from {coherence_path}")
            try:
                with rasterio.open(coherence_path) as src:
                    coherence = src.read(1)
                    if coherence.shape != phase.shape:
                        print(f"[WARNING] Coherence shape {coherence.shape} doesn't match phase shape {phase.shape}")
                        coherence = None
                    else:
                        print(f"[INFO] Coherence data loaded successfully")
                        # Print basic stats about coherence
                        valid_coh = coherence[np.isfinite(coherence)]
                        if len(valid_coh) > 0:
                            print(f"[INFO] Coherence range: {np.min(valid_coh):.4f} to {np.max(valid_coh):.4f}")
                            print(f"[INFO] Coherence mean: {np.mean(valid_coh):.4f}")
                            print(f"[INFO] Coherence values above threshold ({coherence_threshold}): {np.sum(valid_coh >= coherence_threshold)} out of {len(valid_coh)} ({100*np.sum(valid_coh >= coherence_threshold)/len(valid_coh):.1f}%)")
            except Exception as e:
                print(f"[WARNING] Failed to read coherence file {coherence_path}: {e}")
                coherence = None
        
        # If no coherence file is available, create a dummy coherence of 1.0 (no masking)
        if coherence is None:
            print("[INFO] No coherence data found, using all pixels (no coherence masking)")
            coherence = np.ones_like(phase)
        
        # Compute filtered displacement
        print(f"[INFO] Computing displacement with coherence filtering (threshold={coherence_threshold}, filter_strength={filter_strength})")
        disp = compute_displacement_with_filtering(
            phase, coherence, WAVELENGTH_LBAND, mean_theta,
            coherence_threshold, filter_strength, debug
        )
        
        if debug:
            # Visualize displacement data if in debug mode
            debug_dir = os.path.join(output_dir, 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            disp_viz_path = os.path.join(debug_dir, os.path.basename(unw_path).replace('.tif', '_disp_viz.png'))
            visualize_data(disp, f"Displacement (m) - {os.path.basename(unw_path)}", disp_viz_path)
        
        # Save the original displacement data
        orig_disp_filename = os.path.basename(unw_path).replace(UNWRAPPED_SUFFIX, '_vertical_disp_m_orig.tif')
        orig_disp_path = os.path.join(output_dir, orig_disp_filename)

        # Create a copy of the profile for the original resolution file
        orig_profile = profile.copy()
        orig_profile.update({
            'dtype': 'float32',
            'count': 1,
            'compress': 'lzw',
            'nodata': np.nan,
        })
        
        # Ensure the profile has CRS information
        if crs is not None:
            orig_profile.update({'crs': crs})
        
        # Make sure transform is set in the profile
        orig_profile.update({'transform': transform})

        print(f"[INFO] Writing original resolution displacement file: {orig_disp_path}")
        with rasterio.open(orig_disp_path, 'w', **orig_profile) as dst:
            dst.write(disp.reshape(1, disp.shape[0], disp.shape[1]))
        
        # ==== CRITICAL FIX: Use proper geographic calculations for resampling ====
        
        # Check if we have a valid transform for resampling
        if transform.is_identity or (transform.a == 1.0 and transform.e == -1.0):
            print("[ERROR] Cannot resample with missing or identity transform")
            return False
            
        # Calculate target resolution and dimensions for the resampled output
        if crs is not None and crs.to_string().upper() == 'EPSG:4326':
            # We're working with geographic coordinates (lat/lon)
            # Need to calculate proper resolution using haversine
            
            # Find the midpoint latitude for accurate distance calculations
            if bounds:
                mid_lat = (bounds.bottom + bounds.top) / 2
            else:
                # Estimate midpoint from transform
                height = profile['height']
                mid_lat = transform.f + (transform.e * height / 2)
            
            # For very high latitudes, cap the adjustment factor to avoid numerical instability
            lat_adjust_factor = min(np.cos(np.radians(mid_lat)), 0.1)
            
            # Use haversine formula to calculate proper resolution in degrees for 30m
            earth_radius = 6371000  # meters
            target_lat_res = np.degrees(TARGET_RES / earth_radius)
            
            # For longitude, adjust for latitude using cos(latitude)
            # Apply minimum cap to handle high latitudes
            target_lon_res = target_lat_res / lat_adjust_factor if lat_adjust_factor > 0.01 else target_lat_res * 10
            
            # Safety check - if we get absurd values, fall back to a reasonable approach
            if target_lon_res > 0.1:  # More than 0.1 degrees is clearly wrong
                print(f"[WARNING] Calculated longitude resolution ({target_lon_res:.8f}°) is unreasonable")
                print(f"[WARNING] Using safe fallback resolution of 0.001° (approximately 100m at equator)")
                target_lon_res = 0.001  # Fallback to approximately 100m at equator
            
            # Get bounds from transform
            west = transform.c
            north = transform.f
            east = west + (transform.a * profile['width'])
            south = north + (transform.e * profile['height'])
            
            # Calculate target dimensions with safety checks
            new_width = max(50, min(10000, int(np.abs((east - west) / target_lon_res))))
            new_height = max(50, min(10000, int(np.abs((south - north) / target_lat_res))))
            
            # Create the destination transform
            dst_transform = Affine(target_lon_res, 0.0, west,
                                   0.0, -target_lat_res, north)
                                   
            print(f"[INFO] Geographic resampling: {transform.a:.8f}° to {target_lon_res:.8f}° lon, {np.abs(transform.e):.8f}° to {target_lat_res:.8f}° lat")
                                   
            print(f"[INFO] Geographic resampling: {transform.a:.8f}° to {target_lon_res:.8f}° lon, {np.abs(transform.e):.8f}° to {target_lat_res:.8f}° lat")
        else:
            # For projected coordinates or unknown CRS, use rasterio's calculation
            dst_transform, new_width, new_height = calculate_default_transform(
                crs if crs else rasterio.crs.CRS.from_epsg(4326),  # Default to WGS84 if no CRS
                crs if crs else rasterio.crs.CRS.from_epsg(4326),
                profile['width'], profile['height'],
                transform.c, transform.f + (transform.e * profile['height']),
                transform.c + (transform.a * profile['width']), transform.f,
                resolution=TARGET_RES
            )
            
        print(f"[INFO] Resampling displacement to {TARGET_RES}m resolution")
        print(f"[DEBUG] Original dimensions: {profile['width']} x {profile['height']}")
        print(f"[DEBUG] Resampled dimensions: {new_width} x {new_height}")
        print(f"[DEBUG] Original transform details: {transform}")
        print(f"[DEBUG] Destination transform details: {dst_transform}")
        
        # Create resampled array with appropriate dimensions
        resampled_disp = np.empty((1, new_height, new_width), dtype=np.float32)
        
        # For handling NaN values properly during resampling
        # 1. Create a mask of valid (non-NaN) pixels
        valid_mask = np.isfinite(disp)
        print(f"[DEBUG] Valid pixels before resampling: {np.sum(valid_mask)} ({np.sum(valid_mask)/disp.size*100:.1f}%)")
        
        # 2. Replace NaNs with zeros before resampling
        temp_disp = np.copy(disp)
        temp_disp[~valid_mask] = 0.0
        
        # 3. Create a mask array and set it to 1 where data is valid
        mask_array = np.zeros_like(disp, dtype=np.uint8)
        mask_array[valid_mask] = 1
        
        try:
            # Try with bilinear interpolation first
            reproject(
                source=np.expand_dims(temp_disp, 0),
                destination=resampled_disp,
                src_transform=transform,
                src_crs=crs,
                dst_transform=dst_transform,
                dst_crs=crs,
                resampling=Resampling.bilinear
            )
            
            # Resample the mask (0s and 1s)
            resampled_mask_array = np.empty((1, new_height, new_width), dtype=np.uint8)
            reproject(
                source=np.expand_dims(mask_array, 0),
                destination=resampled_mask_array,
                src_transform=transform,
                src_crs=crs,
                dst_transform=dst_transform,
                dst_crs=crs,
                resampling=Resampling.nearest
            )
            
            # Apply the mask to the resampled data
            resampled_disp[0][resampled_mask_array[0] == 0] = np.nan
            
            # Check if we have any valid pixels after resampling
            valid_resampled = np.isfinite(resampled_disp[0])
            print(f"[DEBUG] Valid pixels after resampling: {np.sum(valid_resampled)} ({np.sum(valid_resampled)/resampled_disp[0].size*100:.1f}%)")
            
            # If we've lost all valid pixels, fall back to nearest-neighbor resampling
            if np.sum(valid_resampled) == 0:
                print("[WARNING] Bilinear resampling resulted in all NaNs, trying nearest neighbor resampling instead")
                
                # Try nearest neighbor resampling instead
                reproject(
                    source=np.expand_dims(temp_disp, 0),
                    destination=resampled_disp,
                    src_transform=transform,
                    src_crs=crs,
                    dst_transform=dst_transform,
                    dst_crs=crs,
                    resampling=Resampling.nearest
                )
                
                # Apply mask again
                resampled_disp[0][resampled_mask_array[0] == 0] = np.nan
                
                valid_resampled = np.isfinite(resampled_disp[0])
                print(f"[DEBUG] Valid pixels after nearest resampling: {np.sum(valid_resampled)} ({np.sum(valid_resampled)/resampled_disp[0].size*100:.1f}%)")
            
            # If still no valid pixels, fall back to zoom-based resampling
            if np.sum(valid_resampled) == 0:
                print("[WARNING] All resampling methods failed with standard approach, falling back to zoom-based resampling")
                
                # Calculate zoom factors based on dimension ratios
                y_scale = new_height / disp.shape[0]
                x_scale = new_width / disp.shape[1]
                
                # Resample using zoom
                resampled_disp[0] = zoom(temp_disp, (y_scale, x_scale), order=1)
                resampled_mask = zoom(valid_mask.astype(float), (y_scale, x_scale), order=0) > 0.5
                
                # Apply the resampled mask
                resampled_disp[0][~resampled_mask] = np.nan
                
                print(f"[DEBUG] Fallback: rescaled using scipy.ndimage.zoom with factors: ({y_scale}, {x_scale})")
                print(f"[DEBUG] Fallback: valid pixels: {np.sum(resampled_mask)} ({np.sum(resampled_mask)/resampled_mask.size*100:.1f}%)")
            
        except Exception as e:
            print(f"[WARNING] Error during resampling: {e}")
            print("[WARNING] Falling back to zoom-based resampling")
            
            # Calculate zoom factors based on dimension ratios
            y_scale = new_height / disp.shape[0]
            x_scale = new_width / disp.shape[1]
            
            # Resample using zoom
            resampled_disp[0] = zoom(temp_disp, (y_scale, x_scale), order=1)
            resampled_mask = zoom(valid_mask.astype(float), (y_scale, x_scale), order=0) > 0.5
            
            # Apply the resampled mask
            resampled_disp[0][~resampled_mask] = np.nan
            
            print(f"[DEBUG] Fallback: rescaled using scipy.ndimage.zoom with factors: ({y_scale}, {x_scale})")
            print(f"[DEBUG] Fallback: valid pixels: {np.sum(resampled_mask)} ({np.sum(resampled_mask)/resampled_mask.size*100:.1f}%)")
        
        if 'crs' in profile and profile['crs'] is not None:
            print(f"[INFO] CRS preserved during resampling: {profile['crs'].to_string()}")
        else:
            print(f"[WARNING] CRS information may have been lost during resampling")
            
        if debug:
            # Visualize resampled data
            debug_dir = os.path.join(output_dir, 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            resampled_viz_path = os.path.join(debug_dir, os.path.basename(unw_path).replace('.tif', '_resampled_viz.png'))
            visualize_data(resampled_disp[0], f"Resampled Displacement (m) - {os.path.basename(unw_path)}", resampled_viz_path)
        
        # Update profile for output file
        profile.update({
            'dtype': 'float32',
            'height': new_height,
            'width': new_width,
            'transform': dst_transform,
            'count': 1,
            'compress': 'lzw',
            'nodata': np.nan,
        })
        
        # Ensure CRS is preserved
        if crs is not None:
            profile.update({'crs': crs})
        
        # Define output filename and path
        out_filename = os.path.basename(unw_path).replace(UNWRAPPED_SUFFIX, f'_vertical_disp_m_{TARGET_RES}m.tif')
        out_path = os.path.join(output_dir, out_filename)

        # Write output file
        print(f"[INFO] Writing output file: {out_path}")
        try:
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(resampled_disp)
            
            print(f"[INFO] Successfully wrote output with preserved geospatial information")
        except Exception as e:
            print(f"[ERROR] Failed to write output: {e}")
            return False
        
        # Verify the output file
        verify_result = verify_output_file(out_path)
        if verify_result:
            print(f"[SUCCESS] Successfully processed and verified {out_filename}")
            
            # Double-check the geospatial information
            try:
                with rasterio.open(out_path) as verification:
                    output_crs = verification.crs
                    output_transform = verification.transform
                    
                    if output_crs is None:
                        print(f"[ERROR] Output CRS is missing!")
                        return False
                        
                    if (output_transform.c == 0.0 and output_transform.f == 0.0):
                        print(f"[ERROR] Output transform has origin at (0,0)!")
                        return False
                        
                    print(f"[INFO] Verified output has proper CRS: {output_crs.to_string()}")
                    print(f"[INFO] Verified output has proper transform with origin: ({output_transform.c}, {output_transform.f})")
                    
                    return True
            except Exception as e:
                print(f"[ERROR] Failed during verification: {e}")
                return False
        else:
            print(f"[ERROR] Failed to produce valid output for {out_filename}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to process {unw_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

# Add the haversine function for proper distance calculations
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r

def compute_displacement(phase, wavelength, incidence_deg, debug=False):
    """
    Convert unwrapped phase to vertical displacement with proper error handling
    """
    if debug:
        # Log phase statistics before processing
        phase_stats = {
            'min': np.nanmin(phase) if not np.all(np.isnan(phase)) else "all NaN",
            'max': np.nanmax(phase) if not np.all(np.isnan(phase)) else "all NaN",
            'mean': np.nanmean(phase) if not np.all(np.isnan(phase)) else "all NaN",
            'zeros': np.sum(phase == 0),
            'nans': np.sum(np.isnan(phase)),
            'infs': np.sum(np.isinf(phase)),
            'total_pixels': phase.size
        }
        print(f"[DEBUG] Phase statistics before displacement calculation: {phase_stats}")
    
    # Pre-mask invalid values
    valid_mask = np.isfinite(phase)
    result = np.full_like(phase, np.nan, dtype=np.float32)
    
    # Only process valid pixels
    if np.any(valid_mask):
        incidence_rad = np.deg2rad(incidence_deg)
        result[valid_mask] = -1.0 * wavelength * phase[valid_mask] / (4 * np.pi * np.cos(incidence_rad))
        
        # Verify displacement values are reasonable (e.g., -5m to +5m for most applications)
        valid_disp = result[valid_mask]
        if debug:
            disp_stats = {
                'min': np.nanmin(valid_disp) if len(valid_disp) > 0 else "none",
                'max': np.nanmax(valid_disp) if len(valid_disp) > 0 else "none",
                'mean': np.nanmean(valid_disp) if len(valid_disp) > 0 else "none",
                'valid_pixels': np.sum(valid_mask),
                'extreme_values': np.sum(np.logical_or(valid_disp < -5.0, valid_disp > 5.0)) if len(valid_disp) > 0 else "none"
            }
            print(f"[DEBUG] Displacement statistics: {disp_stats}")
        
        extreme_values = np.logical_or(result < -5.0, result > 5.0)
        if np.any(extreme_values & valid_mask):
            print(f"[WARNING] Extreme displacement values detected: range [{np.nanmin(result):.2f}, {np.nanmax(result):.2f}] meters")
    else:
        print(f"[ERROR] No valid phase values found in input data")
    
    return result
    
def compute_displacement_with_filtering(phase, coherence, wavelength, incidence_deg,
                                        coherence_threshold=0.4, filter_strength=0.5, debug=False):
    """
    Convert unwrapped phase to vertical displacement with coherence masking and filtering
    
    Parameters:
    -----------
    phase : numpy.ndarray
        Unwrapped phase data
    coherence : numpy.ndarray
        Interferometric coherence (same shape as phase)
    wavelength : float
        Radar wavelength in meters
    incidence_deg : float
        Incidence angle in degrees
    coherence_threshold : float
        Threshold for coherence masking (0.0-1.0)
    filter_strength : float
        Strength of the filtering (0.0-1.0)
    debug : bool
        Enable debug output
        
    Returns:
    --------
    numpy.ndarray
        Filtered displacement map with low coherence areas masked (NaN)
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter
    
    if debug:
        # Log phase statistics before processing
        phase_stats = {
            'min': np.nanmin(phase) if not np.all(np.isnan(phase)) else "all NaN",
            'max': np.nanmax(phase) if not np.all(np.isnan(phase)) else "all NaN",
            'mean': np.nanmean(phase) if not np.all(np.isnan(phase)) else "all NaN",
            'zeros': np.sum(phase == 0),
            'nans': np.sum(np.isnan(phase)),
            'infs': np.sum(np.isinf(phase)),
            'total_pixels': phase.size
        }
        print(f"[DEBUG] Phase statistics before filtering: {phase_stats}")
        
        # Log coherence statistics
        coherence_stats = {
            'min': np.nanmin(coherence) if not np.all(np.isnan(coherence)) else "all NaN",
            'max': np.nanmax(coherence) if not np.all(np.isnan(coherence)) else "all NaN",
            'mean': np.nanmean(coherence) if not np.all(np.isnan(coherence)) else "all NaN",
            'below_threshold': np.sum(coherence < coherence_threshold) / coherence.size * 100
        }
        print(f"[DEBUG] Coherence statistics: {coherence_stats}")
    
    # Create coherence mask
    coherence_mask = coherence >= coherence_threshold
    
    # Apply adaptive filtering - stronger in low coherence areas, weaker in high coherence areas
    sigma_base = 1.0 * filter_strength
    
    # Create filtered phase array
    filtered_phase = np.copy(phase)
    
    # Apply filtering only to valid phase values
    valid_mask = np.isfinite(phase)
    if np.any(valid_mask):
        # Create a copy of the phase array for filtering
        phase_for_filtering = np.copy(phase)
        
        # Replace non-finite values with zeros for filtering
        phase_for_filtering[~valid_mask] = 0
        
        # Apply Gaussian filter to the entire array
        filtered_data = gaussian_filter(phase_for_filtering, sigma=sigma_base)
        
        # Only update valid pixels in the filtered result
        filtered_phase[valid_mask] = filtered_data[valid_mask]
    
    mask = coherence_mask & valid_mask
    if np.sum(mask) == 0:
        print(f"[WARNING] No valid pixels after applying coherence threshold {coherence_threshold}")
        return np.full_like(phase, np.nan, dtype=np.float32)

    # Before calculating the displacement values, print filtered phase stats:
    filtered_valid = filtered_phase[mask]
    if len(filtered_valid) > 0:
        print(f"[INFO] Filtered phase range: {np.min(filtered_valid):.6f} to {np.max(filtered_valid):.6f}")
        print(f"[INFO] Filtered phase mean: {np.mean(filtered_valid):.6f}")
    
    # Compute displacement only for areas with sufficient coherence
    result = np.full_like(phase, np.nan, dtype=np.float32)
    
    mask = coherence_mask & valid_mask
    if np.any(mask):
        incidence_rad = np.deg2rad(incidence_deg)
        result[mask] = -1.0 * wavelength * filtered_phase[mask] / (4 * np.pi * np.cos(incidence_rad))
        
        # Verify displacement values are reasonable
        valid_disp = result[mask]
        
        if len(valid_disp) > 0:
            print(f"[INFO] Displacement range: {np.min(valid_disp):.6f} to {np.max(valid_disp):.6f} meters")
            
            # Check if values are extremely small
            if np.max(np.abs(valid_disp)) < 1e-5:
                print(f"[WARNING] Displacement values very small, might need scaling")
                # Option 1: Apply scaling directly (comment out if not needed)
                # result[mask] *= 1000.0  # Scale by 1000x to convert to mm or amplify signal
                # print(f"[INFO] After scaling: {np.min(result[mask]):.6f} to {np.max(result[mask]):.6f} meters")
                
        if debug:
            disp_stats = {
                'min': np.nanmin(valid_disp) if len(valid_disp) > 0 else "none",
                'max': np.nanmax(valid_disp) if len(valid_disp) > 0 else "none",
                'mean': np.nanmean(valid_disp) if len(valid_disp) > 0 else "none",
                'valid_pixels': np.sum(mask),
                'valid_percentage': np.sum(mask) / phase.size * 100,
                'extreme_values': np.sum(np.logical_or(valid_disp < -5.0, valid_disp > 5.0)) if len(valid_disp) > 0 else "none"
            }
            print(f"[DEBUG] Filtered displacement statistics: {disp_stats}")
        
        # Flag extreme values
        extreme_values = np.logical_or(result < -5.0, result > 5.0)
        if np.any(extreme_values & mask):
            if debug:
                print(f"[WARNING] Extreme displacement values detected: range [{np.nanmin(result[mask]):.2f}, {np.nanmax(result[mask]):.2f}] meters")
    else:
        print(f"[WARNING] No valid pixels remain after coherence masking with threshold {coherence_threshold}")
    
    # Try alternative scaling to see if it gives more reasonable results
    # This is for diagnostic purposes only
    if np.any(mask):
        # Try assuming phase is in 100ths of radians
        test_scale = 100.0
        test_result = -1.0 * wavelength * (filtered_phase[mask] * test_scale) / (4 * np.pi * np.cos(incidence_rad))
        if len(test_result) > 0:
            print(f"[DIAGNOSTIC] With phase * {test_scale}: {np.min(test_result):.6f} to {np.max(test_result):.6f} meters")
        
        # Try assuming phase needs wavelength scaling adjustment
        test_scale2 = 1000.0
        test_result2 = -1.0 * (wavelength * test_scale2) * filtered_phase[mask] / (4 * np.pi * np.cos(incidence_rad))
        if len(test_result2) > 0:
            print(f"[DIAGNOSTIC] With wavelength * {test_scale2}: {np.min(test_result2):.6f} to {np.max(test_result2):.6f} meters")
    
    return result

def verify_output_file(file_path):
    """
    Verify the output file contains valid data
    """
    print(f"[INFO] Verifying output file: {file_path}")
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            
            # Check for all-zero or all-NaN
            all_zeros = np.all(data == 0)
            all_nans = np.all(np.isnan(data))
            
            if all_zeros:
                print(f"[ERROR] Output file {file_path} contains all zeros!")
                return False
            
            if all_nans:
                print(f"[ERROR] Output file {file_path} contains all NaNs!")
                return False
                
            # Check for reasonable displacement range
            valid_data = data[np.isfinite(data) & (data != 0)]
            if len(valid_data) > 0:
                data_range = (np.min(valid_data), np.max(valid_data))
                print(f"[INFO] Valid displacement range: {data_range[0]:.4f} to {data_range[1]:.4f} meters")
                print(f"[INFO] Valid pixels: {len(valid_data)} ({len(valid_data)/data.size*100:.1f}% of total)")
                return True
            else:
                print(f"[ERROR] No valid data (non-zero, non-NaN) in {file_path}")
                return False
    except Exception as e:
        print(f"[ERROR] Failed to verify {file_path}: {e}")
        return False

def visualize_data(data, title, save_path=None):
    """
    Visualize data for debugging purposes
    """
    plt.figure(figsize=(10, 8))
    
    # Handle NaN values for better visualization
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        print("[WARNING] No valid data to visualize")
        plt.text(0.5, 0.5, "No valid data", ha='center', va='center')
    else:
        # Choose a good color range for the data
        vmin, vmax = np.percentile(valid_data, [2, 98]) if len(valid_data) > 100 else (np.min(valid_data), np.max(valid_data))
        
        # Plot the data
        im = plt.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label='Value')
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def process_all_interferograms(debug=False, max_files=None):
    """
    Main function to process all interferograms
    """
    # Create output directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Find all unwrapped interferograms
    unw_paths = glob.glob(os.path.join(SOURCE_DIR, '**', f'*{UNWRAPPED_SUFFIX}'), recursive=True)
    print(f"[INFO] Found {len(unw_paths)} interferograms.")
    
    # Option to process only a subset for testing
    if max_files is not None:
        print(f"[INFO] Processing only first {max_files} files as requested")
        unw_paths = unw_paths[:max_files]
    
    processed_count = 0
    error_count = 0
    
    # Process each interferogram
    for i, unw_path in enumerate(unw_paths):
        print(f"[INFO] Processing file {i+1} of {len(unw_paths)}: {os.path.basename(unw_path)}")
        #success = process_single_file(unw_path, PROCESSED_DIR, debug)
        success = process_single_file(unw_path, coherence_path=None, output_dir=PROCESSED_DIR, debug=debug)
        if success:
            processed_count += 1
        else:
            error_count += 1
    
    print(f"[SUMMARY] Processed: {processed_count}, Errors: {error_count}, Total: {len(unw_paths)}")
    
def process_file_wrapper(unw_path, output_dir, debug=False, test_mode=False):
    """Wrapper function for multiprocessing"""
    try:
        #success = process_single_file(unw_path, output_dir, debug, test_mode)
        success = process_single_file(unw_path, coherence_path=None, output_dir=PROCESSED_DIR, debug=debug)
        return (unw_path, success)
    except Exception as e:
        print(f"[ERROR] Failed processing {unw_path}: {e}")
        return (unw_path, False)

def process_all_interferograms_parallel(debug=False, max_files=None, num_workers=None):
    """Process all interferograms in parallel"""
    # Create output directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Find all unwrapped interferograms
    unw_paths = glob.glob(os.path.join(SOURCE_DIR, '**', f'*{UNWRAPPED_SUFFIX}'), recursive=True)
    print(f"[INFO] Found {len(unw_paths)} interferograms.")
    
    # Option to process only a subset for testing
    if max_files is not None:
        print(f"[INFO] Processing only first {max_files} files as requested")
        unw_paths = unw_paths[:max_files]
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # Leave one CPU for system
    
    print(f"[INFO] Starting parallel processing with {num_workers} workers")
    
    # Create a pool of workers
    pool = mp.Pool(processes=num_workers)
    
    # Process files in parallel
    func = partial(process_file_wrapper, output_dir=PROCESSED_DIR, debug=debug, test_mode=False)
    results = pool.map(func, unw_paths)
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Count successes and failures
    successes = [r for r in results if r[1]]
    failures = [r for r in results if not r[1]]
    
    print(f"[SUMMARY] Successfully processed: {len(successes)}, Failed: {len(failures)}, Total: {len(unw_paths)}")
    
    # Print failed files for reference
    if failures:
        print("[INFO] Failed files:")
        for path, _ in failures:
            print(f"  - {os.path.basename(path)}")
    
    return len(successes), len(failures)
    
def process_remaining_interferograms(debug=False, start_index=None, end_index=None):
    """
    Process remaining interferograms starting from a specific index
    """
    # Create output directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Find all unwrapped interferograms
    unw_paths = glob.glob(os.path.join(SOURCE_DIR, '**', f'*{UNWRAPPED_SUFFIX}'), recursive=True)
    print(f"[INFO] Found {len(unw_paths)} interferograms total.")
    
    # Sort paths to ensure consistent ordering
    unw_paths.sort()
    
    # Determine which files have already been processed
    processed_files = set()
    for file in glob.glob(os.path.join(PROCESSED_DIR, '*_vertical_disp_m_30m.tif')):
        processed_files.add(os.path.basename(file).replace('_vertical_disp_m_30m.tif', UNWRAPPED_SUFFIX))
    
    # Filter out already processed files
    remaining_paths = []
    for path in unw_paths:
        if os.path.basename(path) not in processed_files:
            remaining_paths.append(path)
    
    print(f"[INFO] Already processed: {len(unw_paths) - len(remaining_paths)} files")
    print(f"[INFO] Remaining to process: {len(remaining_paths)} files")
    
    # Apply start and end indices if provided
    if start_index is not None:
        remaining_paths = remaining_paths[start_index:]
        print(f"[INFO] Starting from index {start_index}")
    
    if end_index is not None:
        remaining_paths = remaining_paths[:end_index-start_index if start_index else end_index]
        print(f"[INFO] Processing up to index {end_index}")
    
    # Process remaining files
    processed_count = 0
    error_count = 0
    
    for i, unw_path in enumerate(remaining_paths):
        print(f"[INFO] Processing remaining file {i+1} of {len(remaining_paths)}: {os.path.basename(unw_path)}")
        success = process_single_file(unw_path, coherence_path=None, output_dir=PROCESSED_DIR, debug=debug)
        if success:
            processed_count += 1
        else:
            error_count += 1
            print(f"[ERROR] Failed on {os.path.basename(unw_path)}")
    
    print(f"[SUMMARY] Processed: {processed_count}, Errors: {error_count}, Total: {len(remaining_paths)}")
    return processed_count, error_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process NISAR interferograms')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--single-file', type=str, help='Process a single file (provide path)')
    parser.add_argument('--coherence-file', type=str, help='Path to coherence file (.cor)')
    parser.add_argument('--coherence-threshold', type=float, default=0.4, help='Coherence threshold (0.0-1.0)')
    parser.add_argument('--filter-strength', type=float, default=0.5, help='Filtering strength (0.0-1.0)')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode (bypass annotation lookup)')
    parser.add_argument('--output-dir', type=str, default=PROCESSED_DIR, help='Output directory for processed files')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    parser.add_argument('--resume', action='store_true', help='Resume processing from where it left off')
    parser.add_argument('--start-index', type=int, help='Start processing from this index')
    parser.add_argument('--end-index', type=int, help='Process until this index')
    
    args = parser.parse_args()
    
    if args.single_file:
        process_single_file(
            args.single_file,
            args.coherence_file,
            args.output_dir,
            args.coherence_threshold,
            args.filter_strength,
            args.debug,
            args.test_mode
        )
    elif args.resume or args.start_index is not None:
        process_remaining_interferograms(args.debug, args.start_index, args.end_index)
    else:
        process_all_interferograms(args.debug, args.max_files)
