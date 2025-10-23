#!/usr/bin/env python3
"""
modules/validation.py
Pipeline Validation Module - COMPLETE IMPLEMENTATION


Comprehensive validation system ensuring data quality and completeness
at every stage of the Arctic remote sensing pipeline.
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json


class PipelineValidator:
    """
    Comprehensive validation system for Arctic remote sensing pipeline.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize validator with configuration
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Validation thresholds
        self.thresholds = {
            'displacement_min': config['displacement_min_m'],
            'displacement_max': config['displacement_max_m'],
            'temperature_min': config['temperature_min_c'],
            'temperature_max': config['temperature_max_c'],
            'moisture_min': config['moisture_min_frac'],
            'moisture_max': config['moisture_max_frac'],
            'coherence_min': config['coherence_threshold'],
            'valid_pixel_fraction': config['valid_pixel_fraction'],
            'lat_min': config['circumarctic_bounds']['min_lat'],
            'lat_max': config['circumarctic_bounds']['max_lat'],
            'lon_min': config['circumarctic_bounds']['min_lon'],
            'lon_max': config['circumarctic_bounds']['max_lon']
        }
        
        # Results storage
        self.validation_results = {}
    
    def validate_acquisition_phase(self) -> bool:
        """
        Validate Phase 1: Data Acquisition
        """
        self.logger.info("Validating acquisition phase...")
        
        acquisition_dir = Path(self.config['acquisition_dir'])
        
        if not acquisition_dir.exists():
            self.logger.error(f"Acquisition directory not found: {acquisition_dir}")
            return False
        
        # Find all HDF5 files
        h5_files = list(acquisition_dir.rglob("*.h5"))
        
        if not h5_files:
            self.logger.error("No HDF5 files found in acquisition directory")
            return False
        
        self.logger.info(f"Found {len(h5_files)} HDF5 files")
        
        # Validate sample of files
        import h5py
        valid_count = 0
        invalid_count = 0
        
        sample_size = min(100, len(h5_files))
        sample_files = np.random.choice(h5_files, sample_size, replace=False)
        
        for h5_file in sample_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'science' in f or 'LSAR' in f:
                        valid_count += 1
                    else:
                        invalid_count += 1
                        self.logger.warning(f"Unexpected structure: {h5_file}")
            except Exception as e:
                invalid_count += 1
                self.logger.warning(f"Cannot read {h5_file}: {e}")
        
        validation_rate = valid_count / sample_size
        self.logger.info(f"Validation rate: {validation_rate:.1%} ({valid_count}/{sample_size})")
        
        if validation_rate < 0.95:
            self.logger.error("Too many invalid files in acquisition")
            return False
        
        self.validation_results['acquisition'] = {
            'total_files': len(h5_files),
            'validated_sample': sample_size,
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'validation_rate': validation_rate
        }
        
        return True
    
    def validate_interferometry_phase(self) -> bool:
        """
        Validate Phase 2: Interferometric Processing
        """
        self.logger.info("Validating interferometry phase...")
        
        ifg_dir = Path(self.config['interferogram_dir'])
        
        if not ifg_dir.exists():
            self.logger.error(f"Interferogram directory not found: {ifg_dir}")
            return False
        
        # Find all interferogram products
        phase_files = list(ifg_dir.rglob("*_phase.tif"))
        coherence_files = list(ifg_dir.rglob("*_coherence.tif"))
        unwrapped_files = list(ifg_dir.rglob("*_unwrapped.tif"))
        
        self.logger.info(f"Phase files: {len(phase_files)}")
        self.logger.info(f"Coherence files: {len(coherence_files)}")
        self.logger.info(f"Unwrapped files: {len(unwrapped_files)}")
        
        if len(phase_files) == 0:
            self.logger.error("No phase files found")
            return False
        
        if len(coherence_files) == 0:
            self.logger.error("No coherence files found")
            return False
        
        # Check unwrapping success rate
        unwrap_rate = len(unwrapped_files) / len(phase_files) if phase_files else 0
        self.logger.info(f"Phase unwrapping success rate: {unwrap_rate:.1%}")
        
        if unwrap_rate < 0.5:
            self.logger.warning("Low phase unwrapping success rate")
        
        self.validation_results['interferometry'] = {
            'phase_files': len(phase_files),
            'coherence_files': len(coherence_files),
            'unwrapped_files': len(unwrapped_files),
            'unwrap_success_rate': unwrap_rate
        }
        
        return True
    
    def validate_displacement_phase(self) -> bool:
        """
        Validate Phase 3: Displacement Extraction
        """
        self.logger.info("Validating displacement phase...")
        
        disp_dir = Path(self.config['displacement_dir'])
        
        if not disp_dir.exists():
            self.logger.error(f"Displacement directory not found: {disp_dir}")
            return False
        
        # Find all displacement maps
        disp_files = list(disp_dir.rglob("*_displacement_30m.tif"))
        
        if not disp_files:
            self.logger.error("No displacement maps found")
            return False
        
        self.logger.info(f"Found {len(disp_files)} displacement maps")
        
        # Validate sample of displacement maps
        import rasterio
        
        sample_size = min(50, len(disp_files))
        sample_files = np.random.choice(disp_files, sample_size, replace=False)
        
        valid_count = 0
        resolution_correct = 0
        georef_correct = 0
        range_violations = 0
        
        for disp_file in sample_files:
            try:
                with rasterio.open(disp_file) as src:
                    # Check resolution
                    res_x = abs(src.transform[0])
                    res_y = abs(src.transform[4])
                    
                    # Allow tolerance for resolution
                    if 25 < res_x < 35 or 0.00025 < res_x < 0.00035:
                        resolution_correct += 1
                    
                    # Check georeferencing
                    if src.crs is not None:
                        georef_correct += 1
                    
                    # Check value range
                    data = src.read(1, masked=True)
                    
                    if data.min() < self.thresholds['displacement_min']:
                        range_violations += 1
                        self.logger.warning(f"Displacement below minimum: {disp_file}")
                    
                    if data.max() > self.thresholds['displacement_max']:
                        range_violations += 1
                        self.logger.warning(f"Displacement above maximum: {disp_file}")
                    
                    valid_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Cannot validate {disp_file}: {e}")
        
        self.logger.info(f"Valid displacement maps: {valid_count}/{sample_size}")
        self.logger.info(f"Correct resolution: {resolution_correct}/{sample_size}")
        self.logger.info(f"Georeferenced: {georef_correct}/{sample_size}")
        self.logger.info(f"Range violations: {range_violations}")
        
        self.validation_results['displacement'] = {
            'total_maps': len(disp_files),
            'validated_sample': sample_size,
            'valid_count': valid_count,
            'resolution_correct': resolution_correct,
            'georef_correct': georef_correct,
            'range_violations': range_violations
        }
        
        return (valid_count / sample_size) > 0.9 and (georef_correct / sample_size) > 0.9
    
    def validate_data_completeness(self) -> bool:
        """
        CRITICAL VALIDATION: Verify NO null values in final output
        """
        self.logger.info("="*80)
        self.logger.info("CRITICAL VALIDATION: DATA COMPLETENESS CHECK")
        self.logger.info("="*80)
        
        output_path = self.config['final_output_path']
        
        if not os.path.exists(output_path):
            self.logger.error(f"Output file not found: {output_path}")
            return False
        
        try:
            # Read parquet file
            table = pq.read_table(output_path)
            df = table.to_pandas()
            
            total_records = len(df)
            self.logger.info(f"Total records in output: {total_records:,}")
            
            # Define critical columns
            critical_columns = {
                'displacement_m': 'Displacement',
                'soil_temp_c': 'Soil Temperature',
                'soil_moist_frac': 'Soil Moisture'
            }
            
            # Check each critical column
            null_found = False
            
            for col, description in critical_columns.items():
                if col not in df.columns:
                    self.logger.error(f"Missing critical column: {col}")
                    return False
                
                null_count = df[col].isna().sum()
                
                if null_count > 0:
                    null_found = True
                    null_pct = (null_count / total_records) * 100
                    self.logger.error(f" {description}: {null_count:,} null values ({null_pct:.2f}%)")
                else:
                    self.logger.info(f" {description}: No null values")
            
            if null_found:
                self.logger.error("="*80)
                self.logger.error("DATA COMPLETENESS CHECK FAILED")
                self.logger.error("="*80)
                return False
            
            self.logger.info("="*80)
            self.logger.info("DATA COMPLETENESS CHECK PASSED")
            self.logger.info(f"All {total_records:,} records contain complete measurements")
            self.logger.info("="*80)
            
            self.validation_results['completeness'] = {
                'total_records': total_records,
                'null_displacement': 0,
                'null_temperature': 0,
                'null_moisture': 0,
                'completeness_verified': True
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during completeness check: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def validate_physical_constraints(self) -> bool:
        """
        Validate all measurements are within physical constraints
        """
        self.logger.info("Validating physical constraints...")
        
        output_path = self.config['final_output_path']
        
        try:
            table = pq.read_table(output_path)
            df = table.to_pandas()
            
            violations = []
            
            # Check displacement
            disp_below = (df['displacement_m'] < self.thresholds['displacement_min']).sum()
            disp_above = (df['displacement_m'] > self.thresholds['displacement_max']).sum()
            
            if disp_below > 0:
                violations.append(f"Displacement below minimum: {disp_below} records")
            if disp_above > 0:
                violations.append(f"Displacement above maximum: {disp_above} records")
            
            # Check temperature
            temp_below = (df['soil_temp_c'] < self.thresholds['temperature_min']).sum()
            temp_above = (df['soil_temp_c'] > self.thresholds['temperature_max']).sum()
            
            if temp_below > 0:
                violations.append(f"Temperature below minimum: {temp_below} records")
            if temp_above > 0:
                violations.append(f"Temperature above maximum: {temp_above} records")
            
            # Check moisture
            moist_below = (df['soil_moist_frac'] < self.thresholds['moisture_min']).sum()
            moist_above = (df['soil_moist_frac'] > self.thresholds['moisture_max']).sum()
            
            if moist_below > 0:
                violations.append(f"Moisture below minimum: {moist_below} records")
            if moist_above > 0:
                violations.append(f"Moisture above maximum: {moist_above} records")
            
            if violations:
                self.logger.error("Physical constraint violations detected:")
                for v in violations:
                    self.logger.error(f"  - {v}")
                return False
            
            self.logger.info("All physical constraints satisfied")
            
            self.validation_results['physical_constraints'] = {
                'displacement_range': [df['displacement_m'].min(), df['displacement_m'].max()],
                'temperature_range': [df['soil_temp_c'].min(), df['soil_temp_c'].max()],
                'moisture_range': [df['soil_moist_frac'].min(), df['soil_moist_frac'].max()],
                'violations': 0
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating physical constraints: {e}")
            return False
    
    def validate_geographic_bounds(self) -> bool:
        """
        Validate all coordinates within circumarctic bounds
        """
        self.logger.info("Validating geographic bounds...")
        
        output_path = self.config['final_output_path']
        
        try:
            table = pq.read_table(output_path)
            df = table.to_pandas()
            
            # Check latitude bounds
            lat_below = (df['latitude'] < self.thresholds['lat_min']).sum()
            lat_above = (df['latitude'] > self.thresholds['lat_max']).sum()
            
            # Check longitude bounds
            lon_out = (
                (df['longitude'] < self.thresholds['lon_min']) |
                (df['longitude'] > self.thresholds['lon_max'])
            ).sum()
            
            if lat_below > 0:
                self.logger.error(f"Latitude below minimum: {lat_below} records")
                return False
            
            if lat_above > 0:
                self.logger.error(f"Latitude above maximum: {lat_above} records")
                return False
            
            if lon_out > 0:
                self.logger.warning(f"Longitude outside bounds: {lon_out} records")
            
            self.logger.info(f"Geographic bounds satisfied")
            self.logger.info(f"  Latitude range: [{df['latitude'].min():.2f}, {df['latitude'].max():.2f}]")
            self.logger.info(f"  Longitude range: [{df['longitude'].min():.2f}, {df['longitude'].max():.2f}]")
            
            self.validation_results['geographic_bounds'] = {
                'lat_range': [df['latitude'].min(), df['latitude'].max()],
                'lon_range': [df['longitude'].min(), df['longitude'].max()],
                'records_within_bounds': len(df) - lat_below - lat_above
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating geographic bounds: {e}")
            return False
    
    def run_full_validation(self) -> Dict:
        """
        Run complete validation suite
        """
        self.logger.info("="*80)
        self.logger.info("RUNNING FULL VALIDATION SUITE")
        self.logger.info("="*80)
        
        validation_checks = {
            'completeness_check': self.validate_data_completeness(),
            'physical_constraints_check': self.validate_physical_constraints(),
            'geographic_bounds_check': self.validate_geographic_bounds(),
        }
        
        # Overall pass/fail
        all_passed = all(validation_checks.values())
        
        failed_checks = [k for k, v in validation_checks.items() if not v]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'passed': all_passed,
            'checks': validation_checks,
            'failed_checks': failed_checks,
            'detailed_results': self.validation_results
        }
        
        if all_passed:
            self.logger.info("="*80)
            self.logger.info(" ALL VALIDATION CHECKS PASSED")
            self.logger.info("="*80)
        else:
            self.logger.error("="*80)
            self.logger.error(" VALIDATION FAILED")
            self.logger.error(f"Failed checks: {', '.join(failed_checks)}")
            self.logger.error("="*80)
        
        # Save validation report
        report_path = Path(self.config['output_dir']) / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Validation report saved: {report_path}")
        
        return results
    
    def compute_quality_metrics(self) -> Dict:
        """
        Compute comprehensive quality metrics for final output
        """
        self.logger.info("Computing quality metrics...")
        
        output_path = self.config['final_output_path']
        
        try:
            table = pq.read_table(output_path)
            df = table.to_pandas()
            
            metrics = {
                'total_records': len(df),
                'temporal_coverage': {
                    'start_date': df['datetime'].min().isoformat() if 'datetime' in df else None,
                    'end_date': df['datetime'].max().isoformat() if 'datetime' in df else None,
                    'total_days': (df['datetime'].max() - df['datetime'].min()).days if 'datetime' in df else None
                },
                'spatial_coverage': {
                    'lat_min': float(df['latitude'].min()),
                    'lat_max': float(df['latitude'].max()),
                    'lon_min': float(df['longitude'].min()),
                    'lon_max': float(df['longitude'].max()),
                    'area_degrees_sq': (df['latitude'].max() - df['latitude'].min()) * 
                                      (df['longitude'].max() - df['longitude'].min())
                },
                'displacement_statistics': {
                    'mean': float(df['displacement_m'].mean()),
                    'std': float(df['displacement_m'].std()),
                    'median': float(df['displacement_m'].median()),
                    'min': float(df['displacement_m'].min()),
                    'max': float(df['displacement_m'].max()),
                    'subsidence_records': int((df['displacement_m'] < 0).sum()),
                    'uplift_records': int((df['displacement_m'] > 0).sum())
                },
                'temperature_statistics': {
                    'mean': float(df['soil_temp_c'].mean()),
                    'std': float(df['soil_temp_c'].std()),
                    'median': float(df['soil_temp_c'].median()),
                    'min': float(df['soil_temp_c'].min()),
                    'max': float(df['soil_temp_c'].max())
                },
                'moisture_statistics': {
                    'mean': float(df['soil_moist_frac'].mean()),
                    'std': float(df['soil_moist_frac'].std()),
                    'median': float(df['soil_moist_frac'].median()),
                    'min': float(df['soil_moist_frac'].min()),
                    'max': float(df['soil_moist_frac'].max())
                },
                'data_sources': {
                    'displacement_sources': df['source'].value_counts().to_dict() if 'source' in df else {},
                    'seasonal_distribution': df['season'].value_counts().to_dict() if 'season' in df else {}
                }
            }
            
            self.logger.info("Quality metrics computed successfully")
            
            # Save metrics
            metrics_path = Path(self.config['output_dir']) / 'quality_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            self.logger.info(f"Quality metrics saved: {metrics_path}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error computing quality metrics: {e}")
            return {}
    
    def compute_statistics(self) -> Dict:
        """
        Compute processing statistics across all phases
        """
        return {
            'acquisition': self.validation_results.get('acquisition', {}),
            'interferometry': self.validation_results.get('interferometry', {}),
            'displacement': self.validation_results.get('displacement', {}),
            'completeness': self.validation_results.get('completeness', {}),
            'physical_constraints': self.validation_results.get('physical_constraints', {}),
            'geographic_bounds': self.validation_results.get('geographic_bounds', {})
        }
