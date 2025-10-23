#!/usr/bin/env python3
"""
ArcticDEM Pipeline Orchestrator

Master script to coordinate the complete ArcticDEM auxiliary data pipeline
from acquisition through consolidation. Includes configuration management,
dry-run validation, and comprehensive error handling.

"""

import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import json

class PipelineOrchestrator:
    """Orchestrates complete ArcticDEM pipeline execution"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.start_time = datetime.now()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def _setup_logging(self) -> logging.Logger:
        """Configure pipeline logging"""
        log_dir = Path(self.config['directories']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"pipeline_orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logger = logging.getLogger('pipeline_orchestrator')
        logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(self.config['logging']['format'])
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _setup_directories(self):
        """Create required directory structure"""
        self.logger.info("Setting up directory structure")
        
        dirs = self.config['directories']
        for key, path in dirs.items():
            if key.endswith('_file'):
                # Create parent directory for files
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            else:
                # Create directory
                Path(path).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Directory structure ready")
    
    def _save_metadata(self, additional_data: Optional[Dict] = None):
        """Save pipeline metadata"""
        metadata = self.config['metadata'].copy()
        metadata['processing_date'] = datetime.now().isoformat()
        metadata['pipeline_config'] = str(self.config_path)
        
        if additional_data:
            metadata.update(additional_data)
        
        metadata_file = Path(self.config['directories']['metadata_file'])
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved: {metadata_file}")
    
    def run_dry_run(self) -> bool:
        """Execute dry-run testing"""
        self.logger.info("="*70)
        self.logger.info("STAGE 0: Dry-Run Testing")
        self.logger.info("="*70)
        
        if not self.config['testing']['dry_run_enabled']:
            self.logger.info("Dry-run testing disabled in configuration")
            return True
        
        try:
            # Import and run test suite
            from test_arcticdem_pipeline import run_all_tests
            
            self.logger.info("Executing comprehensive test suite")
            result = run_all_tests()
            
            if result == 0:
                self.logger.info(" Dry-run tests passed - Pipeline validated")
                return True
            else:
                self.logger.error(" Dry-run tests failed - Review errors before continuing")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during dry-run: {e}")
            return False
    
    def run_acquisition(self) -> bool:
        """Execute acquisition stage"""
        self.logger.info("="*70)
        self.logger.info("STAGE 1: Data Acquisition")
        self.logger.info("="*70)
        
        try:
            from arcticdem_acquisition import ArcticDEMAcquisition
            
            # Build configuration
            acquisition_config = {
                'stac_url': self.config['source']['stac_url'],
                'output_dir': self.config['directories']['output_dir'],
                'temp_dir': self.config['directories']['temp_dir'],
                'log_dir': self.config['directories']['log_dir'],
                'checkpoint_dir': self.config['directories']['checkpoint_dir']
            }
            
            # Run acquisition
            acquisition = ArcticDEMAcquisition(acquisition_config)
            success = acquisition.run()
            
            if success:
                self.logger.info(" Acquisition completed successfully")
            else:
                self.logger.error(" Acquisition failed or interrupted")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error during acquisition: {e}")
            return False
    
    def run_processing(self) -> bool:
        """Execute processing stage"""
        self.logger.info("="*70)
        self.logger.info("STAGE 2: Terrain-Aware Processing")
        self.logger.info("="*70)
        
        try:
            from arcticdem_processing import DEMProcessor
            
            # Build configuration
            processing_config = {
                'temp_dir': self.config['directories']['temp_dir'],
                'output_dir': self.config['directories']['output_dir'],
                'log_dir': self.config['directories']['log_dir'],
                'target_resolution': self.config['processing']['target_resolution']
            }
            
            # Run processing
            processor = DEMProcessor(processing_config)
            metrics = processor.process_directory(
                parallel=self.config['performance']['processing']['parallel_enabled']
            )
            
            # Check results
            if metrics.items_failed == 0:
                self.logger.info(" Processing completed successfully")
                self.logger.info(f"  Processed: {metrics.items_processed} files")
                self.logger.info(f"  Data volume: {metrics.total_bytes_processed / 1e9:.2f} GB")
                self.logger.info(f"  Elapsed: {metrics.elapsed_seconds():.1f} seconds")
                return True
            else:
                self.logger.warning(f"Processing completed with {metrics.items_failed} failures")
                
                # Check error threshold
                error_rate = metrics.items_failed / (metrics.items_processed + metrics.items_failed)
                max_error_rate = self.config['monitoring']['alerts']['max_error_rate_percent'] / 100
                
                if error_rate <= max_error_rate:
                    self.logger.info("Error rate within acceptable threshold, continuing")
                    return True
                else:
                    self.logger.error(f"Error rate {error_rate:.1%} exceeds threshold {max_error_rate:.1%}")
                    return not self.config['error_handling']['continue_on_error']
            
        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            return False
    
    def run_consolidation(self) -> bool:
        """Execute consolidation stage"""
        self.logger.info("="*70)
        self.logger.info("STAGE 3: Dataframe Consolidation")
        self.logger.info("="*70)
        
        try:
            from arcticdem_consolidation import ArcticDEMConsolidator
            
            # Build configuration
            consolidation_config = {
                'input_dir': self.config['directories']['output_dir'],
                'batch_dir': self.config['directories']['batch_dir'],
                'output_file': self.config['directories']['final_output'],
                'checkpoint_file': str(Path(self.config['directories']['checkpoint_dir']) / 'consolidation_progress.json'),
                'log_dir': self.config['directories']['log_dir']
            }
            
            # Run consolidation
            consolidator = ArcticDEMConsolidator(consolidation_config)
            consolidator.consolidate_tiles()
            
            # Verify output
            output_file = Path(consolidation_config['output_file'])
            if output_file.exists():
                self.logger.info(" Consolidation completed successfully")
                
                # Load and report statistics
                import pandas as pd
                df = pd.read_parquet(output_file)
                
                self.logger.info(f"  Total records: {len(df):,}")
                self.logger.info(f"  File size: {output_file.stat().st_size / 1e9:.2f} GB")
                self.logger.info(f"  Latitude range: {df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°")
                self.logger.info(f"  Longitude range: {df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°")
                self.logger.info(f"  Elevation range: {df['elevation'].min():.1f}m to {df['elevation'].max():.1f}m")
                
                # Save consolidation metadata
                consolidation_metadata = {
                    'total_records': len(df),
                    'file_size_gb': output_file.stat().st_size / 1e9,
                    'lat_range': [float(df['latitude'].min()), float(df['latitude'].max())],
                    'lon_range': [float(df['longitude'].min()), float(df['longitude'].max())],
                    'elev_range': [float(df['elevation'].min()), float(df['elevation'].max())],
                    'elev_mean': float(df['elevation'].mean()),
                    'elev_std': float(df['elevation'].std())
                }
                self._save_metadata({'consolidation_stats': consolidation_metadata})
                
                return True
            else:
                self.logger.error(" Consolidation output file not created")
                return False
            
        except Exception as e:
            self.logger.error(f"Error during consolidation: {e}")
            return False
    
    def run_pipeline(self, skip_dry_run: bool = False, stages: Optional[list] = None) -> bool:
        """
        Execute complete pipeline
        
        Parameters:
        -----------
        skip_dry_run : bool
            Skip dry-run testing (not recommended)
        stages : Optional[list]
            List of stages to run (default: all)
        
        Returns:
        --------
        bool
            Success status
        """
        self.logger.info("="*70)
        self.logger.info("ArcticDEM Auxiliary Data Pipeline - Orchestrator")
        self.logger.info("="*70)
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Started: {self.start_time.isoformat()}")
        
        # Setup
        self._setup_directories()
        
        # Define pipeline stages
        all_stages = {
            'dry_run': self.run_dry_run,
            'acquisition': self.run_acquisition,
            'processing': self.run_processing,
            'consolidation': self.run_consolidation
        }
        
        # Determine which stages to run
        if stages is None:
            stages_to_run = list(all_stages.keys())
        else:
            stages_to_run = stages
        
        if skip_dry_run and 'dry_run' in stages_to_run:
            stages_to_run.remove('dry_run')
            self.logger.warning("Dry-run testing skipped (not recommended)")
        
        # Execute stages
        for stage_name in stages_to_run:
            if stage_name not in all_stages:
                self.logger.error(f"Unknown stage: {stage_name}")
                continue
            
            stage_func = all_stages[stage_name]
            success = stage_func()
            
            if not success:
                if self.config['error_handling']['continue_on_error']:
                    self.logger.warning(f"Stage {stage_name} failed, continuing per configuration")
                else:
                    self.logger.error(f"Stage {stage_name} failed, stopping pipeline")
                    return False
        
        # Pipeline completed
        end_time = datetime.now()
        elapsed = (end_time - self.start_time).total_seconds()
        
        self.logger.info("="*70)
        self.logger.info("Pipeline Execution Complete")
        self.logger.info("="*70)
        self.logger.info(f"Completed: {end_time.isoformat()}")
        self.logger.info(f"Total elapsed: {elapsed / 3600:.2f} hours")
        self.logger.info(f"Final output: {self.config['directories']['final_output']}")
        
        # Save final metadata
        final_metadata = {
            'completion_time': end_time.isoformat(),
            'total_elapsed_hours': elapsed / 3600,
            'stages_completed': stages_to_run
        }
        self._save_metadata(final_metadata)
        
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ArcticDEM Auxiliary Data Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with dry-run
  python run_arcticdem_pipeline.py
  
  # Run complete pipeline without dry-run (not recommended)
  python run_arcticdem_pipeline.py --skip-dry-run
  
  # Run specific stages only
  python run_arcticdem_pipeline.py --stages acquisition processing
  
  # Use custom configuration
  python run_arcticdem_pipeline.py --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/arcticdem_config.yaml'),
        help='Path to configuration file (default: config/arcticdem_config.yaml)'
    )
    
    parser.add_argument(
        '--skip-dry-run',
        action='store_true',
        help='Skip dry-run testing (not recommended)'
    )
    
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['dry_run', 'acquisition', 'processing', 'consolidation'],
        help='Specific stages to run (default: all)'
    )
    
    args = parser.parse_args()
    
    # Verify configuration exists
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Run pipeline
    orchestrator = PipelineOrchestrator(args.config)
    success = orchestrator.run_pipeline(
        skip_dry_run=args.skip_dry_run,
        stages=args.stages
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
