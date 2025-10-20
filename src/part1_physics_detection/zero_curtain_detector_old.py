"""Minimal zero-curtain detector - guaranteed to work."""

import numpy as np
import pandas as pd
from typing import Dict, List

class PhysicsInformedZeroCurtainDetector:
    """Minimal detector for testing."""
    
    def __init__(self, config):
        self.config = config
        self.thresholds = config.thresholds
        print(" Detector initialized (minimal version)")
    
    def get_site_permafrost_properties(self, lat: float, lon: float) -> Dict:
        """Check if site is in Arctic region (simplified)."""
        is_arctic = lat >= 50.0
        
        return {
            'permafrost_prob': 0.5 if is_arctic else 0.0,
            'permafrost_zone': 'arctic' if is_arctic else 'non_arctic',
            'is_permafrost_suitable': is_arctic,
            'suitability_reason': f'Latitude: {lat:.1f}°N'
        }
    
    def get_site_snow_properties(self, lat: float, lon: float, timestamps) -> Dict:
        """Placeholder snow properties."""
        return {
            'snow_depth': np.array([]),
            'has_snow_data': False
        }
    
    def detect_zero_curtain_with_physics(self, site_data: pd.DataFrame, lat: float, lon: float) -> List[Dict]:
        """Detect zero-curtain events using simplified criteria."""
        
        events = []
        
        if 'soil_temp_depth_zone' not in site_data.columns:
            site_data['soil_temp_depth_zone'] = 'intermediate'
        
        for depth_zone, group in site_data.groupby('soil_temp_depth_zone'):
            if len(group) < 10:
                continue
            
            group = group.sort_values('datetime')
            temps = group['soil_temp'].values
            times = group['datetime'].values
            
            temp_mask = np.abs(temps) <= 3.0
            temp_grad = np.abs(np.gradient(temps))
            grad_mask = temp_grad <= 1.0
            
            zc_mask = temp_mask & grad_mask
            periods = self._find_periods(zc_mask, min_length=5)
            
            for start_idx, end_idx in periods:
                duration_hours = (end_idx - start_idx + 1) * 24.0
                
                event = {
                    'start_time': times[start_idx],
                    'end_time': times[end_idx],
                    'duration_hours': duration_hours,
                    'intensity_percentile': 0.5,
                    'spatial_extent_meters': 0.5,
                    'depth_zone': depth_zone,
                    'mean_temperature': float(np.mean(temps[start_idx:end_idx+1])),
                    'temperature_variance': float(np.var(temps[start_idx:end_idx+1])),
                    'permafrost_probability': 0.5,
                    'permafrost_zone': 'arctic',
                    'detection_method': 'simplified'
                }
                events.append(event)
        
        return events
    
    def _find_periods(self, mask, min_length):
        """Find continuous True periods."""
        periods = []
        start = None
        
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                if i - start >= min_length:
                    periods.append((start, i-1))
                start = None
        
        if start is not None and len(mask) - start >= min_length:
            periods.append((start, len(mask)-1))
        
        return periods
