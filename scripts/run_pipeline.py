#!/usr/bin/env python3
"""
Main pipeline runner for Arctic zero-curtain in situ database construction.
Orchestrates synchronous execution of all pipeline stages.
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.paths import PATHS
from config.parameters import PARAMETERS
from src.preprocessing.data_inspection_module import inspect_parquet
from src.preprocessing.transformation_module import transform_uavsar_nisar, transform_smap
from src.preprocessing.merging_module import merge_arctic_datasets
from src.utils.visualization_module import plot_arctic_projection, analyze_data_coverage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ArcticPipelineRunner:
    """Orchestrate the complete Arctic in situ database pipeline."""
    
    def __init__(self, paths: dict, params: dict):
        self.paths = paths
        self.params = params
        self.results = {}
        
    def run_stage_1_inspection(self):
        """Stage 1: Data inspection and validation."""
        logger.info("="*60)
        logger.info("STAGE 1: DATA INSPECTION")
        logger.info("="*60)
        
        datasets = [
            ('in_situ', self.paths.get('in_situ')),
            ('uavsar_nisar', self.paths.get('uavsar_nisar_raw')),
            ('smap', self.paths.get('smap_raw'))
        ]
        
        for name, path in datasets:
            if path and Path(path).exists():
                logger.info(f"Inspecting {name}...")
                df = inspect_parquet(path, n_rows=self.params['n_preview_rows'])
                self.results[f'{name}_inspected'] = True
            else:
                logger.warning(f"Path not found: {path}")
        
        return True
    
    def run_stage_2_transformation(self):
        """Stage 2: Data transformation and harmonization."""
        logger.info("="*60)
        logger.info("STAGE 2: DATA TRANSFORMATION")
        logger.info("="*60)
        
        # Transform UAVSAR/NISAR
        uavsar_output = self.paths.get('uavsar_nisar_transformed')
        if not Path(uavsar_output).exists():
            logger.info("Transforming UAVSAR/NISAR data...")
            transform_uavsar_nisar(
                self.paths['uavsar_nisar_raw'],
                uavsar_output
            )
        
        # Transform SMAP
        smap_output = self.paths.get('smap_transformed')
        if not Path(smap_output).exists():
            logger.info("Transforming SMAP data...")
            transform_smap(
                self.paths['smap_raw'],
                smap_output
            )
        
        return True
    
    def run_stage_3_merging(self):
        """Stage 3: Dataset merging and consolidation."""
        logger.info("="*60)
        logger.info("STAGE 3: DATASET MERGING")
        logger.info("="*60)
        
        merge_datasets = {
            'in_situ': self.paths['in_situ'],
            'uavsar_nisar': self.paths['uavsar_nisar_transformed'],
            'smap': self.paths['smap_transformed']
        }
        
        output_path = self.paths.get('merged_final')
        if not Path(output_path).exists():
            logger.info("Merging Arctic datasets...")
            df_merged = merge_arctic_datasets(merge_datasets, output_path)
            self.results['merged_dataset'] = output_path
        else:
            logger.info(f"Merged dataset already exists: {output_path}")
        
        return True
    
    def run_stage_4_quality_control(self):
        """Stage 4: Quality control and coverage analysis."""
        logger.info("="*60)
        logger.info("STAGE 4: QUALITY CONTROL")
        logger.info("="*60)
        
        # Implement quality control checks here
        logger.info("Running quality control checks...")
        
        return True
    
    def run_stage_5_teacher_forcing_prep(self):
        """Stage 5: Prepare dataset for teacher forcing."""
        logger.info("="*60)
        logger.info("STAGE 5: TEACHER FORCING PREPARATION")
        logger.info("="*60)
        
        # Implement teacher forcing preparation here
        logger.info("Preparing dataset for teacher forcing...")
        
        return True
    
    def run_full_pipeline(self):
        """Execute complete pipeline synchronously."""
        start_time = datetime.now()
        logger.info("\n" + "="*60)
        logger.info("ARCTIC IN SITU DATABASE PIPELINE - START")
        logger.info("="*60 + "\n")
        
        try:
            # Execute stages in sequence
            stages = [
                ('Inspection', self.run_stage_1_inspection),
                ('Transformation', self.run_stage_2_transformation),
                ('Merging', self.run_stage_3_merging),
                ('Quality Control', self.run_stage_4_quality_control),
                ('Teacher Forcing Prep', self.run_stage_5_teacher_forcing_prep)
            ]
            
            for stage_name, stage_func in stages:
                logger.info(f"\nExecuting: {stage_name}")
                success = stage_func()
                if not success:
                    logger.error(f"Stage failed: {stage_name}")
                    return False
                logger.info(f"Completed: {stage_name}")
            
            elapsed = datetime.now() - start_time
            logger.info("\n" + "="*60)
            logger.info(f"PIPELINE COMPLETE - Elapsed time: {elapsed}")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            return False


def main():
    """Main entry point."""
    runner = ArcticPipelineRunner(PATHS, PARAMETERS)
    success = runner.run_full_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
