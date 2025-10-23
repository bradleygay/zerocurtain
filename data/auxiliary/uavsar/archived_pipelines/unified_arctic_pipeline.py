#!/usr/bin/env python3
"""
Unified Arctic Remote Sensing Pipeline - Master Coordinator

Complete end-to-end pipeline for Arctic displacement and soil property analysis.
Guarantees data completeness: every record contains displacement, soil temperature,
and soil moisture measurements.

Architecture:
    Phase 1: Geographic-constrained data acquisition (UAVSAR/NISAR)
    Phase 2: Interferometric processing with quality control
    Phase 3: Displacement extraction and 30m resampling
    Phase 4: SMAP integration with spatiotemporal joining
    Phase 5: Validation and completeness verification

Usage:
    python unified_arctic_pipeline.py --config arctic_config.yaml
    python unified_arctic_pipeline.py --config arctic_config.yaml --resume
    python unified_arctic_pipeline.py --config arctic_config.yaml --validate-only
"""

import sys
import os
import json
import yaml
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import argparse

# Import pipeline modules
from modules.acquisition import UAVSARNISARAcquisitionManager
from modules.interferometry import InterferometricProcessor
from modules.displacement import DisplacementExtractor
from modules.consolidation import MultiSourceConsolidator
from modules.validation import PipelineValidator
from modules.quality_control import DisplacementQualityControl
from modules.spatial_join import SpatioTemporalJoiner

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class PipelineLogger:
    """Centralized logging with file and console handlers"""
    
    @staticmethod
    def setup_logging(output_dir: Path, log_level: str = 'INFO') -> logging.Logger:
        """
        Configure hierarchical logging system
        
        Parameters:
        -----------
        output_dir : Path
            Base output directory
        log_level : str
            Logging level (DEBUG, INFO, WARNING, ERROR)
        
        Returns:
        --------
        logging.Logger : Configured logger instance
        """
        log_dir = output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'unified_pipeline_{timestamp}.log'
        
        # Create logger
        logger = logging.getLogger('ArcticPipeline')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        logger.handlers = []
        
        # File handler - detailed logging
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Console handler - summary logging
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ))
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        logger.info(f"Logging initialized. Log file: {log_file}")
        
        return logger


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class PipelineConfiguration:
    """Configuration loader and validator"""
    
    REQUIRED_KEYS = [
        'circumarctic_bounds',
        'acquisition_dir',
        'interferogram_dir',
        'displacement_dir',
        'output_dir',
        'smap_master_path',
        'final_output_path',
        'snaphu_path'
    ]
    
    DEFAULT_CONFIG = {
        'max_workers': 8,
        'target_resolution_m': 30,
        'coherence_threshold': 0.4,
        'displacement_min_m': -10.0,
        'displacement_max_m': 10.0,
        'temperature_min_c': -60.0,
        'temperature_max_c': 60.0,
        'moisture_min_frac': 0.0,
        'moisture_max_frac': 1.0,
        'valid_pixel_fraction': 0.2,
        'temporal_window_days': 3,
        'spatial_tolerance_m': 100,
        'log_level': 'INFO',
        'skip_existing': True,
        'require_complete_records': True
    }
    
    @staticmethod
    def load_configuration(config_path: str) -> Dict:
        """
        Load and validate configuration from YAML file
        
        Parameters:
        -----------
        config_path : str
            Path to configuration YAML file
        
        Returns:
        --------
        dict : Validated configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply defaults
        for key, value in PipelineConfiguration.DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value
        
        # Validate required keys
        missing_keys = [k for k in PipelineConfiguration.REQUIRED_KEYS if k not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        # Validate paths
        PipelineConfiguration._validate_paths(config)
        
        # Validate bounds
        PipelineConfiguration._validate_bounds(config)
        
        return config
    
    @staticmethod
    def _validate_paths(config: Dict):
        """Validate and expand file paths"""
        
        # Expand user paths
        path_keys = [
            'acquisition_dir', 'interferogram_dir', 'displacement_dir',
            'output_dir', 'smap_master_path', 'final_output_path', 'snaphu_path'
        ]
        
        for key in path_keys:
            if key in config:
                config[key] = os.path.expanduser(config[key])
        
        # Check SMAP file exists
        if not os.path.exists(config['smap_master_path']):
            raise FileNotFoundError(f"SMAP master file not found: {config['smap_master_path']}")
        
        # Check SNAPHU exists
        if not os.path.exists(config['snaphu_path']):
            raise FileNotFoundError(f"SNAPHU executable not found: {config['snaphu_path']}")
        
        # Create directories
        for key in ['acquisition_dir', 'interferogram_dir', 'displacement_dir', 'output_dir']:
            Path(config[key]).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _validate_bounds(config: Dict):
        """Validate geographic bounds"""
        
        bounds = config['circumarctic_bounds']
        
        required = ['min_lon', 'max_lon', 'min_lat', 'max_lat']
        if not all(k in bounds for k in required):
            raise ValueError(f"Circumarctic bounds must contain: {required}")
        
        # Check logical consistency
        if bounds['min_lon'] >= bounds['max_lon']:
            raise ValueError("min_lon must be less than max_lon")
        
        if bounds['min_lat'] >= bounds['max_lat']:
            raise ValueError("min_lat must be less than max_lat")
        
        # Check Arctic domain
        if bounds['min_lat'] < 49.0 or bounds['max_lat'] > 90.0:
            raise ValueError("Latitude bounds must be within Arctic domain (49°N - 90°N)")


# ============================================================================
# MAIN PIPELINE COORDINATOR
# ============================================================================

class ArcticRemoteSensingPipeline:
    """
    Master pipeline coordinator for unified Arctic remote sensing analysis.
    
    This class orchestrates the complete workflow from data acquisition through
    final consolidated output, with comprehensive validation at each stage.
    
    Key Features:
    - Geographic filtering at acquisition stage
    - Checkpoint-based resumption
    - Quality control at every processing step
    - Spatiotemporal data fusion with completeness guarantee
    - Comprehensive validation and reporting
    """
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration
        
        Parameters:
        -----------
        config_path : str
            Path to YAML configuration file
        """
        # Load configuration
        self.config = PipelineConfiguration.load_configuration(config_path)
        
        # Setup logging
        self.logger = PipelineLogger.setup_logging(
            Path(self.config['output_dir']),
            self.config['log_level']
        )
        
        # Initialize state tracking
        self.state_dir = Path(self.config['output_dir']) / '.pipeline_state'
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.state_dir / 'pipeline_state.json'
        self.state = self._load_state()
        
        # Initialize sub-modules
        self._initialize_modules()
        
        # Statistics tracking
        self.stats = {
            'pipeline_start': datetime.now().isoformat(),
            'acquisition': {},
            'interferometry': {},
            'displacement': {},
            'consolidation': {},
            'validation': {}
        }
        
        self.logger.info("="*80)
        self.logger.info("Arctic Remote Sensing Pipeline Initialized")
        self.logger.info(f"Configuration: {config_path}")
        self.logger.info(f"Output directory: {self.config['output_dir']}")
        self.logger.info("="*80)
    
    def _initialize_modules(self):
        """Initialize all pipeline processing modules"""
        
        self.logger.info("Initializing pipeline modules...")
        
        # Acquisition manager
        self.acquisition_mgr = UAVSARNISARAcquisitionManager(
            output_dir=self.config['acquisition_dir'],
            geographic_bounds=self.config['circumarctic_bounds'],
            max_workers=self.config['max_workers'],
            logger=self.logger
        )
        
        # Interferometric processor
        self.ifg_processor = InterferometricProcessor(
            base_dir=self.config['acquisition_dir'],
            output_dir=self.config['interferogram_dir'],
            snaphu_path=self.config['snaphu_path'],
            max_workers=self.config['max_workers'],
            skip_existing=self.config['skip_existing'],
            logger=self.logger
        )
        
        # Displacement extractor with quality control
        self.displacement_extractor = DisplacementExtractor(
            input_dir=self.config['interferogram_dir'],
            output_dir=self.config['displacement_dir'],
            target_resolution=self.config['target_resolution_m'],
            target_crs='EPSG:4326',
            quality_control=DisplacementQualityControl(
                displacement_min=self.config['displacement_min_m'],
                displacement_max=self.config['displacement_max_m'],
                coherence_threshold=self.config['coherence_threshold'],
                valid_pixel_fraction=self.config['valid_pixel_fraction']
            ),
            logger=self.logger
        )
        
        # Multi-source consolidator
        self.consolidator = MultiSourceConsolidator(
            displacement_dir=self.config['displacement_dir'],
            smap_path=self.config['smap_master_path'],
            output_path=self.config['final_output_path'],
            spatial_joiner=SpatioTemporalJoiner(
                temporal_window_days=self.config['temporal_window_days'],
                spatial_tolerance_m=self.config['spatial_tolerance_m']
            ),
            require_complete_records=self.config['require_complete_records'],
            logger=self.logger
        )
        
        # Pipeline validator
        self.validator = PipelineValidator(
            config=self.config,
            logger=self.logger
        )
        
        self.logger.info("All modules initialized successfully")
    
    def _load_state(self) -> Dict:
        """Load pipeline state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load state: {e}")
        
        return {
            'phases_completed': [],
            'last_update': None
        }
    
    def _save_state(self):
        """Save pipeline state to disk"""
        self.state['last_update'] = datetime.now().isoformat()
        
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _mark_phase_complete(self, phase_name: str):
        """Mark a processing phase as complete"""
        if phase_name not in self.state['phases_completed']:
            self.state['phases_completed'].append(phase_name)
            self._save_state()
    
    def _is_phase_complete(self, phase_name: str) -> bool:
        """Check if a phase is already complete"""
        return phase_name in self.state['phases_completed']
    
    # ========================================================================
    # PHASE 1: DATA ACQUISITION
    # ========================================================================
    
    def run_acquisition_phase(self, force_rerun: bool = False):
        """
        Phase 1: Acquire UAVSAR and NISAR data with geographic filtering
        
        Parameters:
        -----------
        force_rerun : bool
            Force re-acquisition even if phase is marked complete
        
        Returns:
        --------
        bool : Success status
        """
        phase_name = 'acquisition'
        
        if not force_rerun and self._is_phase_complete(phase_name):
            self.logger.info("Phase 1: Acquisition already complete (use --force-rerun to override)")
            return True
        
        self.logger.info("="*80)
        self.logger.info("PHASE 1: DATA ACQUISITION")
        self.logger.info("="*80)
        
        phase_start = datetime.now()
        
        try:
            # Acquire UAVSAR data
            self.logger.info("Acquiring UAVSAR data...")
            uavsar_stats = self.acquisition_mgr.acquire_uavsar(
                geojson_filter=self.config.get('geojson_filter'),
                flight_lines_filter=self.config.get('flight_lines_filter')
            )
            self.stats['acquisition']['uavsar'] = uavsar_stats
            
            # Acquire NISAR data
            self.logger.info("Acquiring NISAR data...")
            nisar_stats = self.acquisition_mgr.acquire_nisar(
                flight_info_path=self.config.get('flight_info_path')
            )
            self.stats['acquisition']['nisar'] = nisar_stats
            
            # Validate acquisition
            self.logger.info("Validating acquired data...")
            if not self.validator.validate_acquisition_phase():
                raise RuntimeError("Acquisition validation failed")
            
            self.logger.info(f"Phase 1 completed in {datetime.now() - phase_start}")
            self._mark_phase_complete(phase_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 1 failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    # ========================================================================
    # PHASE 2: INTERFEROMETRIC PROCESSING
    # ========================================================================
    
    def run_interferometry_phase(self, force_rerun: bool = False):
        """
        Phase 2: Generate interferograms and unwrap phase
        
        Parameters:
        -----------
        force_rerun : bool
            Force reprocessing even if phase is marked complete
        
        Returns:
        --------
        bool : Success status
        """
        phase_name = 'interferometry'
        
        if not force_rerun and self._is_phase_complete(phase_name):
            self.logger.info("Phase 2: Interferometry already complete (use --force-rerun to override)")
            return True
        
        self.logger.info("="*80)
        self.logger.info("PHASE 2: INTERFEROMETRIC PROCESSING")
        self.logger.info("="*80)
        
        phase_start = datetime.now()
        
        try:
            # Process all sites
            ifg_stats = self.ifg_processor.process_all_sites()
            self.stats['interferometry'] = ifg_stats
            
            # Validate interferograms
            self.logger.info("Validating interferometric products...")
            if not self.validator.validate_interferometry_phase():
                raise RuntimeError("Interferometry validation failed")
            
            self.logger.info(f"Phase 2 completed in {datetime.now() - phase_start}")
            self._mark_phase_complete(phase_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 2 failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    # ========================================================================
    # PHASE 3: DISPLACEMENT EXTRACTION
    # ========================================================================
    
    def run_displacement_phase(self, force_rerun: bool = False):
        """
        Phase 3: Extract displacement maps and resample to 30m
        
        Parameters:
        -----------
        force_rerun : bool
            Force reprocessing even if phase is marked complete
        
        Returns:
        --------
        bool : Success status
        """
        phase_name = 'displacement'
        
        if not force_rerun and self._is_phase_complete(phase_name):
            self.logger.info("Phase 3: Displacement extraction already complete (use --force-rerun to override)")
            return True
        
        self.logger.info("="*80)
        self.logger.info("PHASE 3: DISPLACEMENT EXTRACTION AND RESAMPLING")
        self.logger.info("="*80)
        
        phase_start = datetime.now()
        
        try:
            # Extract and resample all displacement maps
            disp_stats = self.displacement_extractor.process_all_unwrapped_phases()
            self.stats['displacement'] = disp_stats
            
            # Validate displacement products
            self.logger.info("Validating displacement products...")
            if not self.validator.validate_displacement_phase():
                raise RuntimeError("Displacement validation failed")
            
            self.logger.info(f"Phase 3 completed in {datetime.now() - phase_start}")
            self._mark_phase_complete(phase_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 3 failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    # ========================================================================
    # PHASE 4: MULTI-SOURCE CONSOLIDATION
    # ========================================================================
    
    def run_consolidation_phase(self, force_rerun: bool = False):
        """
        Phase 4: Consolidate displacement and SMAP data with completeness guarantee
        
        This phase ensures EVERY output record contains:
        - Displacement measurement
        - Soil temperature measurement  
        - Soil moisture measurement
        
        Parameters:
        -----------
        force_rerun : bool
            Force reprocessing even if phase is marked complete
        
        Returns:
        --------
        bool : Success status
        """
        phase_name = 'consolidation'
        
        if not force_rerun and self._is_phase_complete(phase_name):
            self.logger.info("Phase 4: Consolidation already complete (use --force-rerun to override)")
            return True
        
        self.logger.info("="*80)
        self.logger.info("PHASE 4: MULTI-SOURCE CONSOLIDATION")
        self.logger.info("Data Completeness Guarantee: Active")
        self.logger.info("="*80)
        
        phase_start = datetime.now()
        
        try:
            # Perform spatiotemporal join with completeness requirement
            consol_stats = self.consolidator.consolidate_all_sources()
            self.stats['consolidation'] = consol_stats
            
            # Critical validation: verify NO nulls in output
            self.logger.info("Enforcing data completeness guarantee...")
            if not self.validator.validate_data_completeness():
                raise RuntimeError("Data completeness validation failed - null values detected")
            
            self.logger.info(f"Phase 4 completed in {datetime.now() - phase_start}")
            self.logger.info(f"Generated {consol_stats.get('total_complete_records', 0)} complete records")
            self.logger.info(f"Rejected {consol_stats.get('rejected_incomplete', 0)} incomplete observations")
            
            self._mark_phase_complete(phase_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 4 failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    # ========================================================================
    # PHASE 5: FINAL VALIDATION
    # ========================================================================
    
    def run_validation_phase(self):
        """
        Phase 5: Comprehensive final validation and quality metrics
        
        Returns:
        --------
        bool : Success status
        """
        self.logger.info("="*80)
        self.logger.info("PHASE 5: FINAL VALIDATION AND QUALITY ASSESSMENT")
        self.logger.info("="*80)
        
        try:
            # Run comprehensive validation suite
            validation_results = self.validator.run_full_validation()
            self.stats['validation'] = validation_results
            
            if not validation_results['passed']:
                self.logger.error("Final validation failed")
                self.logger.error(f"Failed checks: {validation_results['failed_checks']}")
                return False
            
            self.logger.info("All validation checks passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation phase failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================
    
    def run_full_pipeline(self, resume: bool = False, force_rerun: bool = False):
        """
        Execute complete end-to-end pipeline
        
        Parameters:
        -----------
        resume : bool
            Resume from last completed phase
        force_rerun : bool
            Force re-execution of all phases
        
        Returns:
        --------
        bool : Overall success status
        """
        pipeline_start = datetime.now()
        
        self.logger.info("="*80)
        self.logger.info("UNIFIED ARCTIC REMOTE SENSING PIPELINE - FULL EXECUTION")
        self.logger.info(f"Started: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Resume mode: {resume}")
        self.logger.info(f"Force rerun: {force_rerun}")
        self.logger.info("="*80)
        
        try:
            # Phase 1: Acquisition
            if not self.run_acquisition_phase(force_rerun=force_rerun):
                raise RuntimeError("Acquisition phase failed")
            
            # Phase 2: Interferometry
            if not self.run_interferometry_phase(force_rerun=force_rerun):
                raise RuntimeError("Interferometry phase failed")
            
            # Phase 3: Displacement
            if not self.run_displacement_phase(force_rerun=force_rerun):
                raise RuntimeError("Displacement phase failed")
            
            # Phase 4: Consolidation
            if not self.run_consolidation_phase(force_rerun=force_rerun):
                raise RuntimeError("Consolidation phase failed")
            
            # Phase 5: Validation
            if not self.run_validation_phase():
                raise RuntimeError("Validation phase failed")
            
            # Generate final report
            self._generate_final_report(pipeline_start)
            
            pipeline_duration = datetime.now() - pipeline_start
            self.logger.info("="*80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total duration: {pipeline_duration}")
            self.logger.info(f"Output: {self.config['final_output_path']}")
            self.logger.info("="*80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.error(traceback.format_exc())
            
            # Save error report
            self._generate_error_report(pipeline_start, str(e))
            
            return False
    
    def _generate_final_report(self, start_time: datetime):
        """Generate comprehensive processing report"""
        
        report = {
            'pipeline_version': '2.0.0',
            'execution_summary': {
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': str(datetime.now() - start_time),
                'success': True
            },
            'configuration': self.config,
            'statistics': self.stats,
            'validation_results': self.stats.get('validation', {}),
            'data_completeness': {
                'guaranteed': self.config['require_complete_records'],
                'verification_passed': self.stats['validation'].get('completeness_check', False)
            }
        }
        
        report_path = Path(self.config['output_dir']) / 'pipeline_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Final report saved: {report_path}")
        
        # Also generate human-readable summary
        summary_path = Path(self.config['output_dir']) / 'pipeline_summary.txt'
        self._generate_text_summary(report, summary_path)
        
        self.logger.info(f"Summary report saved: {summary_path}")
    
    def _generate_text_summary(self, report: Dict, output_path: Path):
        """Generate human-readable text summary"""
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("UNIFIED ARCTIC REMOTE SENSING PIPELINE - EXECUTION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Execution Time: {report['execution_summary']['start_time']} to {report['execution_summary']['end_time']}\n")
            f.write(f"Total Duration: {report['execution_summary']['total_duration']}\n")
            f.write(f"Status: {'SUCCESS' if report['execution_summary']['success'] else 'FAILED'}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("PHASE STATISTICS\n")
            f.write("-"*80 + "\n\n")
            
            # Acquisition
            if 'acquisition' in report['statistics']:
                acq = report['statistics']['acquisition']
                f.write("Phase 1: Data Acquisition\n")
                f.write(f"  UAVSAR files acquired: {acq.get('uavsar', {}).get('files_downloaded', 0)}\n")
                f.write(f"  NISAR files acquired: {acq.get('nisar', {}).get('files_downloaded', 0)}\n")
                f.write(f"  Geographic filtering: Active (circumarctic bounds)\n\n")
            
            # Interferometry
            if 'interferometry' in report['statistics']:
                ifg = report['statistics']['interferometry']
                f.write("Phase 2: Interferometric Processing\n")
                f.write(f"  Sites processed: {ifg.get('sites_processed', 0)}\n")
                f.write(f"  Interferograms generated: {ifg.get('interferograms_generated', 0)}\n")
                f.write(f"  Phase unwrapping success rate: {ifg.get('unwrap_success_rate', 0):.1%}\n\n")
            
            # Displacement
            if 'displacement' in report['statistics']:
                disp = report['statistics']['displacement']
                f.write("Phase 3: Displacement Extraction\n")
                f.write(f"  Displacement maps generated: {disp.get('maps_generated', 0)}\n")
                f.write(f"  Quality control passed: {disp.get('qc_passed', 0)}\n")
                f.write(f"  Quality control failed: {disp.get('qc_failed', 0)}\n")
                f.write(f"  Target resolution: 30m\n\n")
            
            # Consolidation
            if 'consolidation' in report['statistics']:
                consol = report['statistics']['consolidation']
                f.write("Phase 4: Multi-Source Consolidation\n")
                f.write(f"  Complete records generated: {consol.get('total_complete_records', 0)}\n")
                f.write(f"  Incomplete observations rejected: {consol.get('rejected_incomplete', 0)}\n")
                f.write(f"  Spatiotemporal matching success rate: {consol.get('match_success_rate', 0):.1%}\n\n")
            
            # Validation
            if 'validation' in report['statistics']:
                val = report['statistics']['validation']
                f.write("Phase 5: Validation\n")
                f.write(f"  All checks passed: {val.get('passed', False)}\n")
                f.write(f"  Data completeness verified: {val.get('completeness_check', False)}\n")
                f.write(f"  Physical constraints validated: {val.get('physical_constraints_check', False)}\n")
                f.write(f"  Geographic bounds verified: {val.get('geographic_bounds_check', False)}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("DATA COMPLETENESS GUARANTEE\n")
            f.write("-"*80 + "\n\n")
            
            f.write("Every output record contains:\n")
            f.write("  âœ" Displacement measurement (UAVSAR/NISAR)\n")
            f.write("  âœ" Soil temperature measurement (SMAP)\n")
            f.write("  âœ" Soil moisture measurement (SMAP)\n\n")
            
            f.write(f"Output file: {report['configuration']['final_output_path']}\n")
            f.write(f"Format: Parquet (compressed)\n")
            f.write(f"Geographic extent: Circumarctic (49°N - 90°N)\n")
            
            f.write("\n" + "="*80 + "\n")
    
    def _generate_error_report(self, start_time: datetime, error_message: str):
        """Generate error report for failed pipeline"""
        
        error_report = {
            'pipeline_version': '2.0.0',
            'execution_summary': {
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': str(datetime.now() - start_time),
                'success': False,
                'error_message': error_message
            },
            'phases_completed': self.state.get('phases_completed', []),
            'statistics': self.stats,
            'traceback': traceback.format_exc()
        }
        
        error_path = Path(self.config['output_dir']) / 'pipeline_error_report.json'
        with open(error_path, 'w') as f:
            json.dump(error_report, f, indent=2, default=str)
        
        self.logger.error(f"Error report saved: {error_path}")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with comprehensive argument parsing"""
    
    parser = argparse.ArgumentParser(
        description='Unified Arctic Remote Sensing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python unified_arctic_pipeline.py --config arctic_config.yaml
  
  # Resume from last checkpoint
  python unified_arctic_pipeline.py --config arctic_config.yaml --resume
  
  # Force complete reprocessing
  python unified_arctic_pipeline.py --config arctic_config.yaml --force-rerun
  
  # Validation only (no processing)
  python unified_arctic_pipeline.py --config arctic_config.yaml --validate-only
  
  # Run specific phase
  python unified_arctic_pipeline.py --config arctic_config.yaml --phase acquisition
  python unified_arctic_pipeline.py --config arctic_config.yaml --phase consolidation
        """
    )
    
    parser.add_argument(
        '--config',
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last completed phase'
    )
    
    parser.add_argument(
        '--force-rerun',
        action='store_true',
        help='Force re-execution of all phases'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Run validation only (no processing)'
    )
    
    parser.add_argument(
        '--phase',
        choices=['acquisition', 'interferometry', 'displacement', 'consolidation', 'validation'],
        help='Run specific phase only'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = ArcticRemoteSensingPipeline(args.config)
        
        # Override log level if specified
        if args.log_level:
            pipeline.logger.setLevel(getattr(logging, args.log_level))
        
        # Validation-only mode
        if args.validate_only:
            success = pipeline.run_validation_phase()
            return 0 if success else 1
        
        # Single-phase mode
        if args.phase:
            phase_methods = {
                'acquisition': pipeline.run_acquisition_phase,
                'interferometry': pipeline.run_interferometry_phase,
                'displacement': pipeline.run_displacement_phase,
                'consolidation': pipeline.run_consolidation_phase,
                'validation': pipeline.run_validation_phase
            }
            
            success = phase_methods[args.phase](force_rerun=args.force_rerun)
            return 0 if success else 1
        
        # Full pipeline execution
        success = pipeline.run_full_pipeline(
            resume=args.resume,
            force_rerun=args.force_rerun
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        print("State has been saved. Use --resume to continue from last checkpoint.")
        return 130
    
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        print(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())