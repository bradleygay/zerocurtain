#!/usr/bin/env python3
"""
Polarization-Aware HDF5 File Handler with Checkpointing
Date: 2025-04-13

This module provides specialized utilities for:
1. Robust HDF5 polarization detection in NISAR/UAVSAR files
2. Enhanced complex SLC data extraction
3. Metadata handling with redundancy checks
4. State persistence for checkpoint-based resumption
"""

import os
import h5py
import json
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import hashlib

# Configure logger
logger = logging.getLogger(__name__)


class H5CheckpointManager:
    """Manages checkpoints and state persistence for UAVSAR processing"""
    
    def __init__(self, checkpoint_dir):
        """Initialize with checkpoint directory"""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.file_registry = self._load_registry()
    
    def _load_registry(self):
        """Load the file registry from disk"""
        registry_file = self.checkpoint_dir / "file_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load file registry: {e}")
        return {}
    
    def _save_registry(self):
        """Save the file registry to disk"""
        registry_file = self.checkpoint_dir / "file_registry.json"
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.file_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save file registry: {e}")
    
    def get_file_hash(self, filepath):
        """Create a hash for a file based on path and modification time"""
        stat = os.stat(filepath)
        hash_input = f"{filepath}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def register_file(self, filepath, metadata=None):
        """Register a file and its metadata in the registry"""
        file_hash = self.get_file_hash(filepath)
        self.file_registry[filepath] = {
            "hash": file_hash,
            "last_accessed": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self._save_registry()
        return file_hash
    
    def is_file_modified(self, filepath):
        """Check if a file has been modified since last registered"""
        if filepath not in self.file_registry:
            return True
        
        current_hash = self.get_file_hash(filepath)
        registered_hash = self.file_registry[filepath]["hash"]
        return current_hash != registered_hash
    
    def get_pair_checkpoint_path(self, ref_path, sec_path):
        """Get the checkpoint file path for a pair of files"""
        ref_hash = self.get_file_hash(ref_path)
        sec_hash = self.get_file_hash(sec_path)
        pair_id = f"{ref_hash[:8]}_{sec_hash[:8]}"
        return self.checkpoint_dir / f"pair_{pair_id}.json"
    
    def save_pair_checkpoint(self, ref_path, sec_path, status, output_dir=None, metadata=None):
        """Save a checkpoint for a processed pair"""
        checkpoint_path = self.get_pair_checkpoint_path(ref_path, sec_path)
        
        checkpoint_data = {
            "reference": ref_path,
            "secondary": sec_path,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "output_dir": output_dir,
            "metadata": metadata or {}
        }
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save pair checkpoint: {e}")
            return False
    
    def get_pair_checkpoint(self, ref_path, sec_path):
        """Get the checkpoint data for a pair if it exists"""
        checkpoint_path = self.get_pair_checkpoint_path(ref_path, sec_path)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load pair checkpoint: {e}")
            return None
    
    def should_process_pair(self, ref_path, sec_path, force=False):
        """Determine if a pair should be processed based on checkpoint status"""
        if force:
            return True
            
        checkpoint = self.get_pair_checkpoint(ref_path, sec_path)
        
        if not checkpoint:
            return True
            
        # If files haven't changed and status is completed or skipped, don't process
        if (not self.is_file_modified(ref_path) and 
            not self.is_file_modified(sec_path) and
            checkpoint["status"] in ["completed", "skipped"]):
            return False
            
        return True


class UAVSARPolarizationHandler:
    """Specialized handler for UAVSAR/NISAR polarization detection and processing"""
    
    def __init__(self):
        """Initialize the handler"""
        # Known polarization path patterns to check, in order of likelihood
        self.polarization_paths = [
            "science/LSAR/SLC/frequencyA",
            "science/LSAR/RSLC/frequencyA",
            "science/LSAR/SLC/swaths/frequencyA"
        ]
        
    def find_valid_polarizations(self, h5_path):
        """Robustly detect valid polarizations in a NISAR/UAVSAR H5 file"""
        valid_pols = []
        
        try:
            with h5py.File(h5_path, 'r') as f:
                # Try each known path pattern
                for path in self.polarization_paths:
                    if path not in f:
                        continue
                    
                    # Check each polarization
                    for pol in ["HH", "HV", "VH", "VV"]:
                        pol_path = f"{path}/{pol}"
                        
                        # Case 1: Direct r/i components under polarization
                        if pol_path in f and "r" in f[pol_path] and "i" in f[pol_path]:
                            if pol not in valid_pols:
                                valid_pols.append(pol)
                            continue
                        
                        # Case 2: Group with nested r/i data
                        if pol in f[path]:
                            if "r" in f[path][pol] and "i" in f[path][pol]:
                                if pol not in valid_pols:
                                    valid_pols.append(pol)
                            
                            # Case 3: Direct dataset with complex data
                            elif isinstance(f[path][pol], h5py.Dataset):
                                if pol not in valid_pols:
                                    valid_pols.append(pol)
        
        except Exception as e:
            logger.error(f"Error detecting polarizations in {h5_path}: {e}")
        
        return valid_pols
    
    def load_complex_slc(self, h5_path, pol):
        """Load complex SLC data for given polarization with enhanced robustness"""
        try:
            with h5py.File(h5_path, 'r') as f:
                # Try each path pattern
                for path in self.polarization_paths:
                    if path not in f:
                        continue
                    
                    pol_path = f"{path}/{pol}"
                    
                    # Case 1: Standard r/i components
                    if pol_path in f and "r" in f[pol_path] and "i" in f[pol_path]:
                        r = f[f"{pol_path}/r"][:]
                        i = f[f"{pol_path}/i"][:]
                        return r + 1j * i
                    
                    # Case 2: Nested group structure
                    if pol in f[path]:
                        if "r" in f[path][pol] and "i" in f[path][pol]:
                            r = f[f"{path}/{pol}/r"][:]
                            i = f[f"{path}/{pol}/i"][:]
                            return r + 1j * i
                        
                        # Case 3: Direct complex dataset
                        elif isinstance(f[path][pol], h5py.Dataset):
                            data = f[f"{path}/{pol}"][:]
                            if data.dtype == np.complex64 or data.dtype == np.complex128:
                                return data
                            elif data.shape[-1] == 2:  # Real/imag pairs along last dimension
                                return data[..., 0] + 1j * data[..., 1]
            
            # If we got here, we couldn't find the data
            logger.error(f"Could not load polarization {pol} from {h5_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading SLC data from {h5_path} for pol {pol}: {e}")
            return None
    
    def extract_polarization_metadata(self, h5_path, pol):
        """Extract metadata specific to a polarization"""
        metadata = {}
        
        try:
            with h5py.File(h5_path, 'r') as f:
                # Try each path pattern
                for path in self.polarization_paths:
                    if path not in f:
                        continue
                    
                    pol_path = f"{path}/{pol}"
                    
                    if pol_path in f:
                        # Extract common metadata fields if they exist
                        for field in ["startingRange", "xSpacing", "ySpacing", "startingAzimuth"]:
                            if field in f[pol_path]:
                                metadata[field] = f[pol_path][field][()]
                
                # Look for frequency information
                for freq_path in ["science/LSAR/SLC/processedCenterFrequency", 
                                  "science/LSAR/identification/processedCenterFrequency"]:
                    if freq_path in f:
                        metadata["centerFrequency"] = f[freq_path][()]
                        break
                
                # Look for bandwidth information
                for bw_path in ["science/LSAR/SLC/processedRangeBandwidth",
                               "science/LSAR/identification/processedRangeBandwidth"]:
                    if bw_path in f:
                        metadata["rangeBandwidth"] = f[bw_path][()]
                        break
        
        except Exception as e:
            logger.error(f"Error extracting polarization metadata from {h5_path}: {e}")
        
        return metadata


def compute_interferogram(ref_slc, sec_slc, multi_look=None):
    """
    Compute interferogram from two complex SLC images
    
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
        logger.error("Null input to compute_interferogram")
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
        return None


def apply_goldstein_filter(phase, coherence, alpha=0.5, window_size=32):
    """
    Apply Goldstein filter to enhance phase before unwrapping.
    
    Parameters:
    -----------
    phase : ndarray
        Input wrapped phase
    coherence : ndarray
        Coherence map
    alpha : float
        Filter parameter (0 = no filtering, 1 = maximum filtering)
    window_size : int
        Window size for FFT
    """
    from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
    import numpy as np
    
    # Create output array
    filtered = np.zeros_like(phase)
    
    # Apply filter in overlapping windows
    for i in range(0, phase.shape[0] - window_size, window_size // 2):
        for j in range(0, phase.shape[1] - window_size, window_size // 2):
            # Extract window
            window = phase[i:i+window_size, j:j+window_size]
            
            # Apply Hann window to reduce edge effects
            hann = np.outer(np.hanning(window_size), np.hanning(window_size))
            window = window * hann
            
            # Calculate local coherence
            local_coherence = np.mean(coherence[i:i+window_size, j:j+window_size])
            # Adaptive alpha based on coherence
            adaptive_alpha = alpha * (1 - local_coherence)
            
            # FFT
            spectrum = fftshift(fft2(np.exp(1j * window)))
            
            # Calculate power spectrum
            power = np.abs(spectrum)
            
            # Goldstein filter
            filtered_spectrum = spectrum * (power**adaptive_alpha)
            
            # Inverse FFT
            filtered_window = np.angle(ifft2(ifftshift(filtered_spectrum)))
            
            # Add to output with blending
            blend = np.outer(np.hanning(window_size), np.hanning(window_size))
            filtered[i:i+window_size, j:j+window_size] += filtered_window * blend
    
    return filtered