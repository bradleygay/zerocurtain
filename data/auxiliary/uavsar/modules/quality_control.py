#!/usr/bin/env python3
"""
modules/quality_control.py
Quality Control Module for Displacement Maps


Implements comprehensive quality control checks for displacement measurements.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional


class DisplacementQualityControl:
    """
    Quality control for displacement maps
    """
    
    def __init__(self,
                 displacement_min: float = -10.0,
                 displacement_max: float = 10.0,
                 coherence_threshold: float = 0.4,
                 valid_pixel_fraction: float = 0.2,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize quality control
        
        Parameters:
        -----------
        displacement_min : float
            Minimum allowed displacement (meters)
        displacement_max : float
            Maximum allowed displacement (meters)
        coherence_threshold : float
            Minimum coherence for valid pixels
        valid_pixel_fraction : float
            Minimum fraction of valid pixels required
        logger : logging.Logger, optional
            Logger instance
        """
        self.displacement_min = displacement_min
        self.displacement_max = displacement_max
        self.coherence_threshold = coherence_threshold
        self.valid_pixel_fraction = valid_pixel_fraction
        self.logger = logger or logging.getLogger(__name__)
    
    def validate(self, 
                displacement: np.ndarray,
                coherence: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Validate displacement map
        
        Parameters:
        -----------
        displacement : np.ndarray
            Displacement values
        coherence : np.ndarray
            Coherence values
        
        Returns:
        --------
        tuple : (passed, quality_mask)
            passed: bool indicating if QC passed
            quality_mask: boolean array of valid pixels
        """
        # Create quality mask
        quality_mask = np.ones_like(displacement, dtype=bool)
        
        # Check 1: Physical range
        range_mask = (displacement >= self.displacement_min) & (displacement <= self.displacement_max)
        quality_mask &= range_mask
        
        range_violations = np.sum(~range_mask)
        if range_violations > 0:
            self.logger.debug(f"Range violations: {range_violations}")
        
        # Check 2: Coherence threshold
        coherence_mask = coherence >= self.coherence_threshold
        quality_mask &= coherence_mask
        
        low_coherence = np.sum(~coherence_mask)
        if low_coherence > 0:
            self.logger.debug(f"Low coherence pixels: {low_coherence}")
        
        # Check 3: Valid pixel fraction
        valid_fraction = np.sum(quality_mask) / quality_mask.size
        
        if valid_fraction < self.valid_pixel_fraction:
            self.logger.warning(f"Insufficient valid pixels: {valid_fraction:.2%}")
            return False, quality_mask
        
        return True, quality_mask
    
    def compute_statistics(self, 
                          displacement: np.ndarray,
                          quality_mask: np.ndarray) -> Dict:
        """Compute quality statistics"""
        
        valid_displacement = displacement[quality_mask]
        
        if len(valid_displacement) == 0:
            return {}
        
        return {
            'mean': float(np.nanmean(valid_displacement)),
            'std': float(np.nanstd(valid_displacement)),
            'median': float(np.nanmedian(valid_displacement)),
            'min': float(np.nanmin(valid_displacement)),
            'max': float(np.nanmax(valid_displacement)),
            'valid_pixels': int(np.sum(quality_mask)),
            'total_pixels': int(quality_mask.size),
            'valid_fraction': float(np.sum(quality_mask) / quality_mask.size)
        }
