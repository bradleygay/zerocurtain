#!/usr/bin/env python3
"""
Enhanced UAVSAR Interferogram Generation Script
Date: 2025-04-13

This script generates interferograms from NISAR/UAVSAR SLC files with
improved polarization detection and error handling.
"""

import os
import sys
import h5py
import numpy as np
import argparse
import logging
import rasterio
from rasterio.transform import from_origin
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("interferogram_generation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def find_valid_polarizations(h5_path, debug=True):
    """
    Returns list of valid polarizations present in the HDF5 file with enhanced detection.
    Searches for valid polarizations across multiple possible locations.
    
    Parameters:
    -----------
    h5_path : str
        Path to the input HDF5 file
    debug : bool
        If True, print debugging information
        
    Returns:
    --------
    list
        List of valid polarization names found
    """
    standard_pols = ["HH", "HV", "VH", "VV"]
    found_pols = []
    pol_paths = {}
    
    if debug:
        logger.info(f"Searching for polarizations in: {h5_path}")
        
    try:
        with h5py.File(h5_path, 'r') as f:
            # Check if this is a valid NISAR/UAVSAR file
            if "science" not in f:
                logger.warning(f"Not a valid NISAR/UAVSAR file: {h5_path}")
                if debug:
                    logger.info(f"Keys at root level: {list(f.keys())}")
                return []
                
            # List of possible paths where polarization data might be found
            possible_paths = [
                "science/LSAR/SLC/frequencyA",         # Standard NISAR path
                "science/LSAR/RSLC/frequencyA",        # Resampled SLC
                "science/LSAR/SLC/swaths/frequencyA",  # Alternative path
                "science/LSAR/SLC",                    # Direct under SLC
                "science/LSAR"                         # Directly under LSAR
            ]
            
            # Try each path
            for base_path in possible_paths:
                if base_path not in f:
                    continue
                    
                if debug:
                    logger.info(f"Found base path: {base_path}")
                    
                # Check if polarizations exist directly under this path
                for pol in standard_pols:
                    pol_path = f"{base_path}/{pol}"
                    
                    # Case 1: Standard r/i components
                    if pol_path in f and "r" in f[pol_path] and "i" in f[pol_path]:
                        if debug:
                            logger.info(f"Found polarization {pol} at {pol_path} with r/i components")
                            logger.info(f"  r shape: {f[f'{pol_path}/r'].shape}")
                            logger.info(f"  i shape: {f[f'{pol_path}/i'].shape}")
                        
                        found_pols.append(pol)
                        pol_paths[pol] = pol_path
                        continue
                        
                    # Case 2: Direct complex data
                    elif pol_path in f and isinstance(f[pol_path], h5py.Dataset):
                        if debug:
                            logger.info(f"Found polarization {pol} at {pol_path} as direct dataset")
                            logger.info(f"  shape: {f[pol_path].shape}")
                            logger.info(f"  dtype: {f[pol_path].dtype}")
                        
                        # Check if it's likely to be complex data
                        if 'complex' in str(f[pol_path].dtype) or \
                           (len(f[pol_path].shape) > 0 and f[pol_path].shape[-1] == 2):
                            found_pols.append(pol)
                            pol_paths[pol] = pol_path
                
            # If no polarizations found yet, try a manual search through the entire file
            if not found_pols and debug:
                logger.info("No polarizations found in standard paths, performing manual search")
                
                # List of groups/datasets to check
                def find_potential_pols(name, obj):
                    if isinstance(obj, h5py.Group) and os.path.basename(name) in standard_pols:
                        # Check if this group has 'r' and 'i' datasets
                        if "r" in obj and "i" in obj:
                            pol = os.path.basename(name)
                            if pol not in found_pols:
                                logger.info(f"Found polarization {pol} at {name} in manual search")
                                found_pols.append(pol)
                                pol_paths[pol] = name
                                
                # Visit all items in the file
                f.visititems(find_potential_pols)
                    
        if debug:
            logger.info(f"Final polarizations found: {found_pols}")
            
        return found_pols
    except Exception as e:
        logger.error(f"Error finding polarizations in {h5_path}: {e}")
        logger.debug(traceback.format_exc())
        return []


def load_complex_slc(h5_path, pol, debug=True):
    """
    Load complex SLC data for the given polarization with enhanced robustness.
    
    Parameters:
    -----------
    h5_path : str
        Path to the input HDF5 file
    pol : str
        Polarization to load (e.g., "HH", "HV", etc.)
    debug : bool
        If True, print debugging information
        
    Returns:
    --------
    numpy.ndarray
        Complex SLC data (r + i*j), or None if loading fails
    """
    if debug:
        logger.info(f"Loading polarization {pol} from {h5_path}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # List of possible paths where polarization data might be found
            possible_paths = [
                f"science/LSAR/SLC/frequencyA/{pol}",
                f"science/LSAR/RSLC/frequencyA/{pol}",
                f"science/LSAR/SLC/swaths/frequencyA/{pol}",
                f"science/LSAR/SLC/{pol}",
                f"science/LSAR/{pol}"
            ]
            
            # Try each path
            for path in possible_paths:
                if path not in f:
                    continue
                    
                # Check if this path has 'r' and 'i' components
                if "r" in f[path] and "i" in f[path]:
                    if debug:
                        logger.info(f"Found r/i components at {path}")
                        logger.info(f"  r shape: {f[f'{path}/r'].shape}")
                        logger.info(f"  i shape: {f[f'{path}/i'].shape}")
                    
                    # Load the components
                    r = f[f"{path}/r"][:]
                    i = f[f"{path}/i"][:]
                    
                    # Check for NaN or Inf values
                    if np.isnan(r).any() or np.isnan(i).any() or \
                       np.isinf(r).any() or np.isinf(i).any():
                        logger.warning(f"NaN or Inf values found in {pol} data")
                    
                    return r + 1j * i
                
                # Check if this is a direct complex dataset
                elif isinstance(f[path], h5py.Dataset):
                    if debug:
                        logger.info(f"Found direct dataset at {path}")
                        logger.info(f"  shape: {f[path].shape}")
                        logger.info(f"  dtype: {f[path].dtype}")
                    
                    data = f[path][:]
                    
                    # Check if it's complex data or has a last dimension of 2
                    if 'complex' in str(data.dtype):
                        return data
                    elif len(data.shape) > 0 and data.shape[-1] == 2:
                        return data[..., 0] + 1j * data[..., 1]
            
            # If we get here, we couldn't find the data
            logger.error(f"Could not find polarization {pol} in standard paths")
            
            # Try a manual search through the entire file
            if debug:
                logger.info("Performing manual search")
                
                def find_complex_data(name, obj):
                    base_name = os.path.basename(name)
                    parent_name = os.path.basename(os.path.dirname(name))
                    
                    # Check if this is an 'r' or 'i' dataset under the requested polarization
                    if parent_name == pol and base_name in ['r', 'i']:
                        logger.info(f"Found component {base_name} at {name}")
                
                # Visit all items in the file
                f.visititems(find_complex_data)
                
            return None
    except Exception as e:
        logger.error(f"Error loading polarization {pol} from {h5_path}: {e}")
        logger.debug(traceback.format_exc())
        return None


def compute_interferogram(ref_slc, sec_slc, multi_look=None):
    """
    Compute interferogram from two complex SLC images.
    
    Parameters:
    -----------
    ref_slc : ndarray
        Reference complex SLC image
    sec_slc : ndarray
        Secondary complex SLC image
    multi_look : tuple, optional
        (range_looks, azimuth_looks) for multi-looking
        
    Returns:
    --------
    dict
        Dictionary containing phase, amplitude, and coherence
    """
    if ref_slc is None or sec_slc is None:
        logger.error("Cannot compute interferogram: input data is None")
        return None
        
    if ref_slc.shape != sec_slc.shape:
        logger.error(f"Shape mismatch: reference {ref_slc.shape}, secondary {sec_slc.shape}")
        return None
    
    try:
        # Calculate raw interferogram
        ifg = ref_slc * np.conj(sec_slc)
        
        # Apply multi-looking if requested
        if multi_look and len(multi_look) == 2:
            range_looks, azimuth_looks = multi_look
            
            # Function to apply multi-looking
            def apply_multilook(data, rg_looks, az_looks):
                """Apply multi-looking to a 2D array"""
                rows, cols = data.shape
                # Calculate new dimensions
                new_rows = rows // az_looks
                new_cols = cols // rg_looks
                
                # Reshape and average
                reshaped = data[:new_rows*az_looks, :new_cols*rg_looks]
                reshaped = reshaped.reshape(new_rows, az_looks, new_cols, rg_looks)
                return np.mean(reshaped, axis=(1, 3))
            
            # Apply multi-looking
            ifg_ml = apply_multilook(ifg, range_looks, azimuth_looks)
            
            # Calculate multi-looked intensity images
            ref_int_ml = apply_multilook(np.abs(ref_slc)**2, range_looks, azimuth_looks)
            sec_int_ml = apply_multilook(np.abs(sec_slc)**2, range_looks, azimuth_looks)
            
            # Calculate coherence
            numerator = np.abs(ifg_ml)
            denominator = np.sqrt(ref_int_ml * sec_int_ml)
            coherence = np.zeros_like(numerator)
            valid = denominator > 0
            coherence[valid] = numerator[valid] / denominator[valid]
            
            # Extract phase and amplitude
            phase = np.angle(ifg_ml)
            amplitude = np.abs(ifg_ml)
            
        else:
            # No multi-looking
            phase = np.angle(ifg)
            amplitude = np.abs(ifg)
            
            # Calculate coherence (local estimation in 5x5 window)
            from scipy.ndimage import uniform_filter
            window_size = 5
            
            # Moving window coherence estimation
            smooth_ifg = uniform_filter(ifg, size=window_size)
            smooth_ref_pow = uniform_filter(np.abs(ref_slc)**2, size=window_size)
            smooth_sec_pow = uniform_filter(np.abs(sec_slc)**2, size=window_size)
            
            numerator = np.abs(smooth_ifg)
            denominator = np.sqrt(smooth_ref_pow * smooth_sec_pow)
            coherence = np.zeros_like(numerator)
            valid = denominator > 0
            coherence[valid] = numerator[valid] / denominator[valid]
        
        return {
            "phase": phase,
            "amplitude": amplitude,
            "coherence": coherence
        }
        
    except Exception as e:
        logger.error(f"Error computing interferogram: {e}")
        logger.debug(traceback.format_exc())
        return None


def export_geotiff(data, output_path, metadata=None, dtype="float32"):
    """
    Export 2D array as GeoTIFF with enhanced metadata handling.
    
    Parameters:
    -----------
    data : ndarray
        2D array to export
    output_path : str
        Path for output GeoTIFF
    metadata : dict, optional
        Metadata for georeference (if None, uses identity transform)
    dtype : str
        Data type for output file
    """
    try:
        height, width = data.shape
        
        # Default transform if no metadata provided
        transform = from_origin(0, 0, 1, 1)
        crs = None
        
        # Use metadata if provided
        if metadata:
            if "transform" in metadata:
                transform = metadata["transform"]
            if "crs" in metadata:
                crs = metadata["crs"]
        
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Export the data
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform
        ) as dst:
            # Handle NaN and Inf values
            valid_data = data.copy()
            if np.isnan(valid_data).any() or np.isinf(valid_data).any():
                logger.warning(f"NaN or Inf values found in data for {output_path}")
                # Replace with zeros
                valid_data[np.isnan(valid_data) | np.isinf(valid_data)] = 0
                
            dst.write(valid_data.astype(dtype), 1)
            
        logger.info(f"Exported {output_path}")
    except Exception as e:
        logger.error(f"Error exporting GeoTIFF {output_path}: {e}")
        logger.debug(traceback.format_exc())


def generate_ifg_for_polarization(reference_path, secondary_path, output_dir, pol, unwrap=False, debug=True):
    """
    Generate interferogram for a single polarization.
    
    Parameters:
    -----------
    reference_path : str
        Path to reference HDF5 file
    secondary_path : str
        Path to secondary HDF5 file
    output_dir : str
        Directory for output files
    pol : str
        Polarization to process
    unwrap : bool
        Flag indicating whether to unwrap phase
    debug : bool
        Flag for verbose debugging output
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        if debug:
            logger.info(f"Processing polarization {pol}")
            logger.info(f"Reference: {reference_path}")
            logger.info(f"Secondary: {secondary_path}")
            logger.info(f"Output dir: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load SLC data
        logger.info(f"Loading SLC data for polarization {pol}...")
        ref_slc = load_complex_slc(reference_path, pol, debug)
        if ref_slc is None:
            logger.error(f"Failed to load reference SLC for {pol}")
            return False
            
        sec_slc = load_complex_slc(secondary_path, pol, debug)
        if sec_slc is None:
            logger.error(f"Failed to load secondary SLC for {pol}")
            return False
        
        # Check if shapes match
        if ref_slc.shape != sec_slc.shape:
            logger.error(f"Shape mismatch for {pol}: ref {ref_slc.shape} vs sec {sec_slc.shape}")
            return False
        
        logger.info(f"Computing interferogram for {pol}...")
        # Generate interferogram
        ifg_result = compute_interferogram(ref_slc, sec_slc)
        if ifg_result is None:
            logger.error(f"Failed to compute interferogram for {pol}")
            return False
        
        # Get components
        phase = ifg_result["phase"]
        amplitude = ifg_result["amplitude"]
        coherence = ifg_result["coherence"]
        
        # Unwrap phase if requested
        if unwrap:
            logger.info(f"Unwrapping phase for {pol}...")
            try:
                from insar_utils import apply_goldstein_filter, dummy_unwrap
                # Filter phase
                filtered_phase = apply_goldstein_filter(phase, coherence)
                # Unwrap
                unwrapped_phase = dummy_unwrap(filtered_phase)
                
                # Export unwrapped phase
                unwrapped_out = os.path.join(output_dir, f"{pol}_unwrapped_phase.tif")
                export_geotiff(unwrapped_phase, unwrapped_out)
            except Exception as e:
                logger.error(f"Phase unwrapping failed for {pol}: {e}")
                logger.debug(traceback.format_exc())
                # Continue with wrapped phase
        
        # Define output file paths
        phase_out = os.path.join(output_dir, f"{pol}_phase.tif")
        amp_out = os.path.join(output_dir, f"{pol}_amplitude.tif")
        coh_out = os.path.join(output_dir, f"{pol}_coherence.tif")
        
        # Export results
        logger.info(f"Saving results for {pol}...")
        export_geotiff(phase, phase_out)
        export_geotiff(amplitude, amp_out)
        export_geotiff(coherence, coh_out)
        
        # Export a copy in the root directory for compatibility with existing code
        if output_dir:
            base_phase_out = os.path.join(output_dir, "phase.tif")
            base_coh_out = os.path.join(output_dir, "coherence.tif")
            
            # If we processed HH or first polarization, use it for base files
            if pol == "HH" or pol == list(ifg_result.keys())[0]:
                export_geotiff(phase, base_phase_out)
                export_geotiff(coherence, base_coh_out)
                logger.info("Saved base phase and coherence files")
        
        return True
    except Exception as e:
        logger.error(f"Error generating interferogram for {pol}: {e}")
        logger.debug(traceback.format_exc())
        return False


def main():
    """Main function to handle command-line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate interferograms from NISAR SLC HDF5 files with enhanced polarization detection."
    )
    parser.add_argument(
        '--reference', type=str, required=True,
        help='Path to reference HDF5 SLC file'
    )
    parser.add_argument(
        '--secondary', type=str, required=True,
        help='Path to secondary HDF5 SLC file'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--pol', type=str, nargs='+', default=None,
        help='Optional list of polarizations to process'
    )
    parser.add_argument(
        '--unwrap', action='store_true',
        help='Enable phase unwrapping'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable verbose debugging output'
    )
    parser.add_argument(
        '--force-pol', type=str, nargs='+', default=None,
        help='Force using these polarizations even if not found in files'
    )
    
    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.reference):
        logger.error(f"Reference file not found: {args.reference}")
        return 1
    if not os.path.exists(args.secondary):
        logger.error(f"Secondary file not found: {args.secondary}")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Determine polarizations
    if args.pol:
        pols_to_use = args.pol
        logger.info(f"Using user-specified polarizations: {pols_to_use}")
    elif args.force_pol:
        pols_to_use = args.force_pol
        logger.info(f"Forcing polarizations: {pols_to_use}")
    else:
        # Auto-detect polarizations
        ref_pols = find_valid_polarizations(args.reference, args.debug)
        sec_pols = find_valid_polarizations(args.secondary, args.debug)
        
        # Find common polarizations
        pols_to_use = list(set(ref_pols) & set(sec_pols))
        
        if not pols_to_use:
            # If no common polarizations found, try forcing standard polarizations
            if args.debug:
                logger.warning("No common polarizations found, trying standard polarizations")
            pols_to_use = ["HH", "HV", "VH", "VV"]
            
        logger.info(f"Auto-detected valid polarizations: {pols_to_use}")
    
    success_count = 0
    
    # Process each polarization
    for pol in pols_to_use:
        logger.info(f"Processing polarization: {pol}")
        try:
            if generate_ifg_for_polarization(
                reference_path=args.reference,
                secondary_path=args.secondary,
                output_dir=args.output,
                pol=pol,
                unwrap=args.unwrap,
                debug=args.debug
            ):
                success_count += 1
        except Exception as e:
            logger.error(f"Failed to process polarization {pol}: {e}")
            logger.debug(traceback.format_exc())
            continue
    
    # If we processed at least one polarization successfully, consider it a success
    if success_count > 0:
        logger.info(f"[DONE] Interferogram generation completed successfully for {success_count}/{len(pols_to_use)} polarizations")
        return 0
    else:
        logger.error("[FAILED] No polarizations were processed successfully")
        return 1


if __name__ == "__main__":
    sys.exit(main())
