"""
Orchestration layer for physics-informed zero-curtain detection.
Bridges the in situ measurement pipeline with the physics detection framework.
"""

import sys
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics_detection.zero_curtain_detector import PhysicsInformedZeroCurtainDetector
from physics_detection.physics_config import DetectionConfig, DataPaths, PhysicsParameters


class PhysicsDetectionOrchestrator:
    """
    Orchestrates the complete physics-informed detection workflow.
    Connects in situ measurements to physics-based zero-curtain identification.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize orchestrator with configuration.
        
        Args:
            config: Optional DetectionConfig. Creates default if None.
        """
        self.config = config if config is not None else DetectionConfig()
        self.detector = None
        self.results = None
        
        print("="*90)
        print("PHYSICS-INFORMED ZERO-CURTAIN DETECTION ORCHESTRATOR")
        print("="*90)
        print(f"Configuration loaded:")
        print(f"  Base directory: {self.config.paths.base_dir}")
        print(f"  Output directory: {self.config.paths.output_dir}")
        print(f"  Temperature threshold: ±{self.config.physics.temp_threshold}°C")
        print(f"  Minimum duration: {self.config.physics.min_duration_hours} hours")
        print(f"  CryoGrid integration: {self.config.physics.use_cryogrid_enthalpy}")
        print()
    
    def validate_configuration(self) -> bool:
        """
        Validate that all required data paths exist.
        
        Returns:
            bool: True if validation successful, False otherwise
        """
        print("Validating data paths...")
        paths_valid, missing = self.config.paths.validate_paths()
        
        if not paths_valid:
            print(" Path validation failed. Missing files:")
            for missing_path in missing:
                print(f"  - {missing_path}")
            return False
        
        print(" All required data paths validated successfully")
        return True
    
    def initialize_detector(self):
        """Initialize the physics-informed detector with configuration."""
        print("Initializing Physics-Informed Zero-Curtain Detector...")
        self.detector = PhysicsInformedZeroCurtainDetector(config=self.config)
        print(" Detector initialized successfully")
        print()
    
    def run_detection_pipeline(self) -> pd.DataFrame:
        """
        Execute the complete detection pipeline.
        
        Returns:
            pd.DataFrame: Zero-curtain events detection results
        """
        if self.detector is None:
            raise RuntimeError("Detector not initialized. Call initialize_detector() first.")
        
        print("="*90)
        print("EXECUTING PHYSICS-INFORMED DETECTION PIPELINE")
        print("="*90)
        
        # Load in situ measurements
        input_parquet = str(self.config.paths.insitu_measurements_parquet)
        output_parquet = str(self.config.paths.output_dir / f"{self.config.output_prefix}_results.parquet")
        
        print(f"Input: {input_parquet}")
        print(f"Output: {output_parquet}")
        print()
        
        # Execute detection
        start_time = datetime.now()
        
        self.results = self.detector.process_circumarctic_dataset(
            parquet_file=input_parquet,
            output_file=output_parquet
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print()
        print("="*90)
        print(f"DETECTION PIPELINE COMPLETED IN {duration:.1f} MINUTES")
        print("="*90)
        
        return self.results
    
    def generate_summary_report(self) -> dict:
        """
        Generate comprehensive summary of detection results.
        
        Returns:
            dict: Summary statistics and metadata
        """
        if self.results is None or self.results.empty:
            return {
                'status': 'no_events_detected',
                'total_events': 0
            }
        
        summary = {
            'status': 'success',
            'total_events': len(self.results),
            'spatial_coverage': {
                'latitude_range': (self.results['latitude'].min(), self.results['latitude'].max()),
                'longitude_range': (self.results['longitude'].min(), self.results['longitude'].max()),
                'unique_sites': self.results.groupby(['latitude', 'longitude']).ngroups
            },
            'temporal_coverage': {
                'earliest_event': self.results['start_time'].min(),
                'latest_event': self.results['end_time'].max()
            },
            'event_characteristics': {
                'mean_intensity': self.results['intensity_percentile'].mean(),
                'mean_duration_hours': self.results['duration_hours'].mean(),
                'mean_spatial_extent_m': self.results['spatial_extent_meters'].mean()
            },
            'permafrost_context': {
                'zones': self.results['permafrost_zone'].value_counts().to_dict(),
                'mean_probability': self.results['permafrost_probability'].mean()
            },
            'depth_distribution': self.results['depth_zone'].value_counts().to_dict()
        }
        
        # CryoGrid-specific statistics if available
        if 'cryogrid_thermal_conductivity' in self.results.columns:
            summary['cryogrid_statistics'] = {
                'mean_thermal_conductivity': self.results['cryogrid_thermal_conductivity'].mean(),
                'mean_heat_capacity': self.results.get('cryogrid_heat_capacity', pd.Series([0])).mean(),
                'mean_enthalpy_stability': self.results.get('cryogrid_enthalpy_stability', pd.Series([0])).mean()
            }
        
        return summary
    
    def save_summary_report(self, summary: dict):
        """Save summary report to JSON file."""
        import json
        
        output_path = self.config.paths.output_dir / f"{self.config.output_prefix}_summary.json"
        
        # Convert non-serializable types
        def serialize(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=serialize)
        
        print(f" Summary report saved: {output_path}")
    
    def execute_complete_workflow(self) -> tuple[pd.DataFrame, dict]:
        """
        Execute the complete end-to-end workflow.
        
        Returns:
            tuple: (results DataFrame, summary dict)
        """
        # Step 1: Validate configuration
        if not self.validate_configuration():
            raise RuntimeError("Configuration validation failed")
        
        # Step 2: Initialize detector
        self.initialize_detector()
        
        # Step 3: Run detection pipeline
        results = self.run_detection_pipeline()
        
        # Step 4: Generate summary
        summary = self.generate_summary_report()
        
        # Step 5: Save summary
        self.save_summary_report(summary)
        
        return results, summary


def main():
    """Main execution function for orchestrated detection."""
    # Create configuration with custom paths if needed
    config = DetectionConfig()
    
    # Initialize orchestrator
    orchestrator = PhysicsDetectionOrchestrator(config=config)
    
    # Execute complete workflow
    try:
        results, summary = orchestrator.execute_complete_workflow()
        
        print("\n" + "="*90)
        print("WORKFLOW EXECUTION SUMMARY")
        print("="*90)
        print(f"Status: {summary['status']}")
        print(f"Total events detected: {summary.get('total_events', 0):,}")
        if summary.get('total_events', 0) > 0:
            print(f"Unique sites: {summary['spatial_coverage']['unique_sites']}")
            print(f"Mean event intensity: {summary['event_characteristics']['mean_intensity']:.3f}")
            print(f"Mean event duration: {summary['event_characteristics']['mean_duration_hours']:.1f} hours")
        print("="*90)
        
        return results, summary
        
    except Exception as e:
        print(f"\n Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()