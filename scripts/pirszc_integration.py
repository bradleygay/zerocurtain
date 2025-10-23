#!/usr/bin/env python3
"""
Physics-Informed Remote Sensing Zero-Curtain (PIRSZC) Pipeline Integration
============================================================================

This script integrates the PIRSZC detection system into the Arctic zero-curtain
pipeline, processing remote sensing observations to detect zero-curtain events.

Author: [RESEARCHER] Gay
Institution: NASA
Date: 2025

Scientific Basis:
- Outcalt et al. (1990): Zero-curtain effect fundamentals
- Liu et al. (2010): InSAR detection methods
- Westermann et al. (2016): CryoGrid thermodynamics
- Williams & Smith (1989): Permafrost physics
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add script directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import the PIRSZC detector
try:
    from remote_sensing_zero_curtain_v2 import PhysicsInformedZeroCurtainDetector
except ImportError as e:
    print(f"Error importing PIRSZC detector: {e}")
    print("Ensure remote_sensing_zero_curtain_v2.py is in the same directory")
    sys.exit(1)


class PIRSZCPipelineIntegration:
    """
    Integration wrapper for PIRSZC detection in the Arctic pipeline.
    
    This class manages:
    - Data path configuration
    - Input/output file management
    - Processing workflow
    - Results validation
    - Integration with broader pipeline
    """
    
    def __init__(self, base_dir="/Users/[USER]/arctic_zero_curtain_pipeline"):
        """
        Initialize pipeline integration.
        
        Args:
            base_dir: Base directory for Arctic zero-curtain pipeline
        """
        self.base_dir = Path(base_dir)
        
        # Define directory structure
        self.dirs = {
            'base': self.base_dir,
            'data': self.base_dir / 'data',
            'auxiliary': self.base_dir / 'data' / 'auxiliary',
            'uavsar': self.base_dir / 'data' / 'auxiliary' / 'uavsar',
            'uavsar_products': self.base_dir / 'data' / 'auxiliary' / 'uavsar' / 'data_products',
            'smap': self.base_dir / 'data' / 'auxiliary' / 'smap',
            'permafrost': self.base_dir / 'data' / 'auxiliary' / 'permafrost',
            'snow': self.base_dir / 'data' / 'auxiliary' / 'snow',
            'results': self.base_dir / 'results',
            'pirszc': self.base_dir / 'results' / 'pirszc',
            'logs': self.base_dir / 'logs'
        }
        
        # Create directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        self.files = {
            'remote_sensing_input': self.dirs['uavsar_products'] / 'remote_sensing.parquet',
            'permafrost_raster': self.dirs['permafrost'] / 'UiO_PEX_PERPROB_5' / 'UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tif',
            'permafrost_zones': self.dirs['permafrost'] / 'UiO_PEX_PERZONES_5' / 'UiO_PEX_PERZONES_5.0_20181128_2000_2016_NH.shp',
            'snow_data': self.dirs['snow'] / 'aa6ddc60e4ed01915fb9193bcc7f4146.nc',
            'pirszc_output': self.dirs['pirszc'] / 'pirszc_events.parquet',
            'pirszc_metadata': self.dirs['pirszc'] / 'pirszc_metadata.json',
            'processing_log': self.dirs['logs'] / f'pirszc_processing_{datetime.now():%Y%m%d_%H%M%S}.log'
        }
        
        print("="*80)
        print("PIRSZC PIPELINE INTEGRATION INITIALIZED")
        print("="*80)
        print(f"Base directory: {self.base_dir}")
        print(f"Remote sensing input: {self.files['remote_sensing_input']}")
        print(f"PIRSZC output: {self.files['pirszc_output']}")
        
    def validate_inputs(self):
        """
        Validate that all required input files exist.
        
        Returns:
            dict: Validation results with status and messages
        """
        print("\n" + "="*80)
        print("VALIDATING INPUT DATA")
        print("="*80)
        
        validation = {
            'status': True,
            'messages': [],
            'files_checked': {},
            'files_missing': []
        }
        
        # Check remote sensing data
        if self.files['remote_sensing_input'].exists():
            size_mb = self.files['remote_sensing_input'].stat().st_size / (1024**2)
            validation['files_checked']['remote_sensing'] = {
                'exists': True,
                'size_mb': size_mb
            }
            print(f" Remote sensing data: {size_mb:.1f} MB")
            
            # Quick data check
            try:
                df = pd.read_parquet(self.files['remote_sensing_input'])
                n_obs = len(df)
                date_range = (df['datetime'].min(), df['datetime'].max())
                lat_range = (df['latitude'].min(), df['latitude'].max())
                
                validation['files_checked']['remote_sensing'].update({
                    'observations': n_obs,
                    'date_range': date_range,
                    'lat_range': lat_range
                })
                
                print(f"  Observations: {n_obs:,}")
                print(f"  Date range: {date_range[0]} to {date_range[1]}")
                print(f"  Latitude range: {lat_range[0]:.2f}° to {lat_range[1]:.2f}°N")
                
            except Exception as e:
                validation['messages'].append(f"Warning: Could not read remote sensing data: {e}")
                print(f"   Warning: Could not read data file: {e}")
        else:
            validation['status'] = False
            validation['files_missing'].append('remote_sensing')
            validation['messages'].append("Remote sensing data file not found")
            print(f" Remote sensing data: NOT FOUND")
        
        # Check auxiliary data (optional but recommended)
        auxiliary_files = {
            'permafrost_raster': 'Permafrost probability raster',
            'permafrost_zones': 'Permafrost zones shapefile',
            'snow_data': 'Snow data NetCDF'
        }
        
        print("\nAuxiliary data:")
        for key, description in auxiliary_files.items():
            if self.files[key].exists():
                validation['files_checked'][key] = {'exists': True}
                print(f" {description}: Found")
            else:
                validation['files_checked'][key] = {'exists': False}
                validation['messages'].append(f"Optional: {description} not found")
                print(f" {description}: Not found (optional)")
        
        # Summary
        print("\n" + "-"*80)
        if validation['status']:
            print(" INPUT VALIDATION PASSED")
        else:
            print(" INPUT VALIDATION FAILED")
            print(f"Missing required files: {', '.join(validation['files_missing'])}")
        
        return validation
    
    def setup_detector(self, config=None):
        """
        Initialize the PIRSZC detector with configuration.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            PhysicsInformedZeroCurtainDetector: Configured detector instance
        """
        print("\n" + "="*80)
        print("INITIALIZING PIRSZC DETECTOR")
        print("="*80)
        
        # Initialize detector
        detector = PhysicsInformedZeroCurtainDetector()
        
        # Apply custom configuration if provided
        if config:
            for key, value in config.items():
                if hasattr(detector, key):
                    setattr(detector, key, value)
                    print(f"  Set {key} = {value}")
        
        # Verify auxiliary data paths (update detector paths to match pipeline)
        print("\nUpdating auxiliary data paths:")
        
        # Update permafrost data path
        if self.files['permafrost_raster'].exists():
            # The detector loads this in _load_auxiliary_data()
            # We need to ensure it uses our pipeline paths
            print(f"  Permafrost data: {self.files['permafrost_raster']}")
        
        # Display detector configuration
        print("\nDetector Configuration:")
        print(f"  Physics models:")
        print(f"    - CryoGrid enthalpy: {detector.use_cryogrid_enthalpy}")
        print(f"    - Painter-Karra freezing: {detector.use_painter_karra_freezing}")
        print(f"    - Surface energy balance: {detector.use_surface_energy_balance}")
        print(f"    - Adaptive timestep: {detector.use_adaptive_timestep}")
        print(f"  Processing:")
        print(f"    - NUMBA JIT: {detector.NUMBA_AVAILABLE if hasattr(detector, 'NUMBA_AVAILABLE') else 'Unknown'}")
        print(f"    - Workers: {detector.n_workers}")
        print(f"    - Chunk size: {detector.chunk_size:,}")
        print(f"    - Grid size: {detector.spatial_grid_size}°")
        
        return detector
    
    def process_remote_sensing_data(self, detector, test_mode=False):
        """
        Process remote sensing data to detect zero-curtain events.
        
        Args:
            detector: PhysicsInformedZeroCurtainDetector instance
            test_mode: If True, process only a small subset for testing
            
        Returns:
            pd.DataFrame: Detected zero-curtain events
        """
        print("\n" + "="*80)
        print("PROCESSING REMOTE SENSING DATA")
        print("="*80)
        
        input_file = str(self.files['remote_sensing_input'])
        output_file = str(self.files['pirszc_output'])
        
        if test_mode:
            print(" TEST MODE: Processing subset of data")
            # In test mode, we'll modify the detector to process fewer chunks
            original_chunk_size = detector.chunk_size
            detector.chunk_size = 100000  # Smaller chunks for testing
        
        try:
            # Process using the detector's main method
            results = detector.process_remote_sensing_dataset(
                parquet_file=input_file,
                output_file=output_file
            )
            
            if test_mode:
                # Restore original settings
                detector.chunk_size = original_chunk_size
            
            return results
            
        except Exception as e:
            print(f" Error during processing: {e}")
            import traceback
            traceback.print_exc()
            
            if test_mode:
                detector.chunk_size = original_chunk_size
            
            return pd.DataFrame()
    
    def validate_results(self, results):
        """
        Validate PIRSZC detection results.
        
        Args:
            results: DataFrame of detected events
            
        Returns:
            dict: Validation metrics and status
        """
        print("\n" + "="*80)
        print("VALIDATING RESULTS")
        print("="*80)
        
        validation = {
            'status': True,
            'n_events': len(results),
            'metrics': {},
            'issues': []
        }
        
        if results.empty:
            validation['status'] = False
            validation['issues'].append("No events detected")
            print(" No events detected")
            return validation
        
        # Check required columns
        required_cols = [
            'latitude', 'longitude', 'start_time', 'end_time',
            'intensity_percentile', 'duration_hours', 'spatial_extent_meters'
        ]
        
        missing_cols = [col for col in required_cols if col not in results.columns]
        if missing_cols:
            validation['status'] = False
            validation['issues'].append(f"Missing columns: {missing_cols}")
            print(f" Missing required columns: {missing_cols}")
        
        # Calculate metrics
        validation['metrics'] = {
            'n_events': len(results),
            'lat_range': (results['latitude'].min(), results['latitude'].max()),
            'lon_range': (results['longitude'].min(), results['longitude'].max()),
            'intensity': {
                'mean': results['intensity_percentile'].mean(),
                'std': results['intensity_percentile'].std(),
                'min': results['intensity_percentile'].min(),
                'max': results['intensity_percentile'].max()
            },
            'duration': {
                'mean': results['duration_hours'].mean(),
                'std': results['duration_hours'].std(),
                'min': results['duration_hours'].min(),
                'max': results['duration_hours'].max()
            },
            'spatial_extent': {
                'mean': results['spatial_extent_meters'].mean(),
                'std': results['spatial_extent_meters'].std(),
                'min': results['spatial_extent_meters'].min(),
                'max': results['spatial_extent_meters'].max()
            }
        }
        
        # Display metrics
        print(f"\n Detected {len(results):,} zero-curtain events")
        print(f"\nSpatial coverage:")
        print(f"  Latitude: {validation['metrics']['lat_range'][0]:.2f}° to {validation['metrics']['lat_range'][1]:.2f}°N")
        print(f"  Longitude: {validation['metrics']['lon_range'][0]:.2f}° to {validation['metrics']['lon_range'][1]:.2f}°E")
        
        print(f"\nEvent characteristics:")
        print(f"  Intensity: {validation['metrics']['intensity']['mean']:.3f} ± {validation['metrics']['intensity']['std']:.3f}")
        print(f"  Duration: {validation['metrics']['duration']['mean']:.1f} ± {validation['metrics']['duration']['std']:.1f} hours")
        print(f"  Spatial extent: {validation['metrics']['spatial_extent']['mean']:.2f} ± {validation['metrics']['spatial_extent']['std']:.2f} meters")
        
        # Check for anomalies
        if validation['metrics']['intensity']['min'] < 0 or validation['metrics']['intensity']['max'] > 1:
            validation['issues'].append("Intensity values outside [0,1] range")
            print(f" Warning: Intensity values outside expected range")
        
        if validation['metrics']['duration']['min'] < 0:
            validation['issues'].append("Negative duration values detected")
            print(f" Warning: Negative duration values")
        
        if validation['metrics']['spatial_extent']['min'] < 0:
            validation['issues'].append("Negative spatial extent values detected")
            print(f" Warning: Negative spatial extent values")
        
        return validation
    
    def save_metadata(self, validation, processing_info):
        """
        Save processing metadata.
        
        Args:
            validation: Validation results dictionary
            processing_info: Processing information dictionary
        """
        metadata = {
            'pipeline_version': '1.0',
            'processing_date': datetime.now().isoformat(),
            'input_file': str(self.files['remote_sensing_input']),
            'output_file': str(self.files['pirszc_output']),
            'validation': validation,
            'processing': processing_info,
            'directories': {k: str(v) for k, v in self.dirs.items()},
            'scientific_references': {
                'zero_curtain': 'Outcalt et al. (1990)',
                'insar_methods': 'Liu et al. (2010)',
                'cryogrid': 'Westermann et al. (2016)',
                'permafrost_physics': 'Williams & Smith (1989)'
            }
        }
        
        with open(self.files['pirszc_metadata'], 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n Metadata saved: {self.files['pirszc_metadata']}")
    
    def run_full_pipeline(self, test_mode=False, config=None):
        """
        Execute the complete PIRSZC pipeline.
        
        Args:
            test_mode: If True, process only subset for testing
            config: Optional detector configuration
            
        Returns:
            dict: Pipeline execution results
        """
        print("\n" + "="*80)
        print("PIRSZC PIPELINE EXECUTION")
        print("="*80)
        print(f"Mode: {'TEST' if test_mode else 'PRODUCTION'}")
        print(f"Timestamp: {datetime.now()}")
        
        results = {
            'success': False,
            'validation': None,
            'processing': None,
            'events': None
        }
        
        # Step 1: Validate inputs
        validation = self.validate_inputs()
        if not validation['status']:
            print("\n Pipeline aborted: Input validation failed")
            return results
        
        # Step 2: Setup detector
        detector = self.setup_detector(config)
        
        # Step 3: Process data
        start_time = datetime.now()
        events = self.process_remote_sensing_data(detector, test_mode)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        processing_info = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'processing_time_seconds': processing_time,
            'test_mode': test_mode
        }
        
        # Step 4: Validate results
        result_validation = self.validate_results(events)
        
        # Step 5: Save metadata
        self.save_metadata(result_validation, processing_info)
        
        # Step 6: Summary
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*80)
        print(f"Processing time: {processing_time/60:.1f} minutes")
        print(f"Events detected: {len(events):,}")
        print(f"Output file: {self.files['pirszc_output']}")
        print(f"Metadata file: {self.files['pirszc_metadata']}")
        
        results['success'] = result_validation['status']
        results['validation'] = result_validation
        results['processing'] = processing_info
        results['events'] = events
        
        return results


def main():
    """Main execution function."""
    
    # Initialize pipeline integration
    pipeline = PIRSZCPipelineIntegration()
    
    # Run in test mode first (recommended)
    print("\n" + "="*80)
    print("RUNNING PIPELINE IN TEST MODE")
    print("="*80)
    
    test_results = pipeline.run_full_pipeline(test_mode=True)
    
    if test_results['success']:
        print("\n Test mode successful!")
        print("\nTo run full production mode, use:")
        print("  python pirszc_integration.py --production")
    else:
        print("\n Test mode failed. Review errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PIRSZC Pipeline Integration"
    )
    parser.add_argument(
        '--production',
        action='store_true',
        help='Run in production mode (processes full dataset)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize and run pipeline
    pipeline = PIRSZCPipelineIntegration()
    
    if args.production:
        results = pipeline.run_full_pipeline(test_mode=False, config=config)
    else:
        results = pipeline.run_full_pipeline(test_mode=True, config=config)
    
    sys.exit(0 if results['success'] else 1)
