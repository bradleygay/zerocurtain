#!/usr/bin/env python3
"""
run_pipeline.py
STREAMLINED Arctic Remote Sensing Pipeline


Pipeline stages:
1. H5 → Displacement GeoTIFFs (in-memory, no intermediate storage)
2. Multi-Source Consolidation (Displacement + SMAP → Unified Dataset)
3. Validation & Quality Metrics
"""

import sys
import os
import logging
import yaml
from pathlib import Path
from datetime import datetime

# Import pipeline modules
from modules.interferometry import InMemoryInterferometricProcessor
from modules.consolidation import MultiSourceConsolidator
from modules.spatial_join import SpatioTemporalJoiner
from modules.validation import PipelineValidator


def setup_logging(config: dict) -> logging.Logger:
    """Setup pipeline logging"""
    
    log_dir = Path(config['output_dir']) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, config.get('log_level', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('ArcticPipeline')
    logger.info(f"Logging to: {log_file}")
    
    return logger


def validate_prerequisites(config: dict, logger: logging.Logger) -> bool:
    """Validate all prerequisites before running pipeline"""
    
    logger.info("="*80)
    logger.info("VALIDATING PREREQUISITES")
    logger.info("="*80)
    
    all_valid = True
    
    # Check acquisition directory exists and has H5 files
    acq_dir = Path(config['acquisition_dir'])
    if not acq_dir.exists():
        logger.error(f" Data directory not found: {acq_dir}")
        logger.error("  Run: bash uavsar_download_consolidated.sh")
        all_valid = False
    else:
        h5_files = list(acq_dir.rglob("*.h5"))
        logger.info(f" Data directory exists: {acq_dir}")
        logger.info(f"  Found {len(h5_files)} H5 files")
        
        if len(h5_files) == 0:
            logger.warning(" No H5 files found - run download script first")
            all_valid = False
    
    # Check SMAP data exists
    smap_path = Path(config['smap_master_path'])
    if not smap_path.exists():
        logger.warning(f" SMAP data not found: {smap_path}")
        logger.warning("  Consolidation will be skipped")
    else:
        logger.info(f" SMAP data found: {smap_path}")
    
    logger.info("="*80)
    
    return all_valid


def run_pipeline(config_file: str):
    """Execute complete pipeline"""
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("="*80)
    logger.info("ARCTIC REMOTE SENSING PIPELINE")
    logger.info("Streamlined: H5 → Displacement GeoTIFFs → Consolidation")
    logger.info("="*80)
    logger.info(f"Configuration: {config_file}")
    logger.info(f"Start time: {datetime.now()}")
    logger.info("="*80)
    
    # Validate prerequisites
    if not validate_prerequisites(config, logger):
        logger.error("Prerequisites not met - cannot continue")
        return False
    
    pipeline_start = datetime.now()
    
    try:
        # ====================================================================
        # STAGE 1: H5 → DISPLACEMENT GeoTIFFs (In-Memory)
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: H5 → DISPLACEMENT GeoTIFFs")
        logger.info("Processing in-memory (no intermediate interferogram storage)")
        logger.info("="*80)
        
        processor = InMemoryInterferometricProcessor(
            base_dir=config['acquisition_dir'],
            output_dir=config['displacement_dir'],
            target_resolution=30,
            target_crs='EPSG:4326',
            coherence_threshold=config['coherence_threshold'],
            skip_existing=config['skip_existing'],
            logger=logger
        )
        
        proc_stats = processor.process_all_sites()
        
        logger.info(f"\nDisplacement extraction complete:")
        logger.info(f"  Sites processed: {proc_stats['sites_processed']}")
        logger.info(f"  Displacement maps generated: {proc_stats['displacement_maps_generated']}")
        logger.info(f"  Pairs failed: {proc_stats['pairs_failed']}")
        
        # ====================================================================
        # STAGE 2: MULTI-SOURCE CONSOLIDATION
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: MULTI-SOURCE CONSOLIDATION")
        logger.info("="*80)
        
        # Check if SMAP data exists
        if not os.path.exists(config['smap_master_path']):
            logger.warning("SMAP data not found - skipping consolidation")
            logger.warning(f"Expected at: {config['smap_master_path']}")
        else:
            # Initialize spatial-temporal joiner
            joiner = SpatioTemporalJoiner(
                temporal_window_days=config['temporal_window_days'],
                spatial_tolerance_m=config['spatial_tolerance_m'],
                logger=logger
            )
            
            # Initialize consolidator
            consolidator = MultiSourceConsolidator(
                displacement_dir=config['displacement_dir'],
                smap_path=config['smap_master_path'],
                output_path=config['final_output_path'],
                spatial_joiner=joiner,
                require_complete_records=config['require_complete_records'],
                logger=logger
            )
            
            consol_stats = consolidator.consolidate_all_sources()
            
            logger.info(f"\nConsolidation complete:")
            logger.info(f"  Displacement records: {consol_stats['displacement_records_loaded']:,}")
            logger.info(f"  SMAP records: {consol_stats['smap_records_loaded']:,}")
            logger.info(f"  Complete unified records: {consol_stats['total_complete_records']:,}")
            logger.info(f"  Match success rate: {consol_stats['match_success_rate']:.1%}")
        
        # ====================================================================
        # STAGE 3: VALIDATION
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("STAGE 3: VALIDATION & QUALITY METRICS")
        logger.info("="*80)
        
        validator = PipelineValidator(config=config, logger=logger)
        
        # Run validation if final output exists
        if os.path.exists(config['final_output_path']):
            validation_results = validator.run_full_validation()
            
            if validation_results['passed']:
                logger.info("\n ALL VALIDATION CHECKS PASSED")
            else:
                logger.error("\n SOME VALIDATION CHECKS FAILED")
        else:
            logger.warning("Final output not found - skipping validation")
        
        # ====================================================================
        # PIPELINE COMPLETE
        # ====================================================================
        pipeline_duration = datetime.now() - pipeline_start
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Total duration: {pipeline_duration}")
        logger.info(f"Output: {config['displacement_dir']}")
        if os.path.exists(config['final_output_path']):
            logger.info(f"Unified dataset: {config['final_output_path']}")
        logger.info("="*80)
        
        return True
    
    except KeyboardInterrupt:
        logger.info("\n\nPipeline interrupted by user")
        logger.info("Progress has been saved - restart to resume")
        return False
    
    except Exception as e:
        logger.error(f"\n\nPipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Arctic Remote Sensing Pipeline')
    parser.add_argument('--config', default='production/pipeline_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    success = run_pipeline(args.config)
    sys.exit(0 if success else 1)
