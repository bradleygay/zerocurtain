#!/usr/bin/env python3
"""
Unified UAVSAR/NISAR Processing Pipeline
Date: 2025-01-20

Consolidated pipeline orchestrator integrating:
- Checkpoint-based resumption for robustness
- Parallel interferogram generation
- Automatic displacement processing with coherence filtering
- 30m resampling with proper geographic transforms
- Enhanced polarization detection

Dependencies:
    - polarization_handler.py (checkpoint and polarization detection)
    - nisar_disp_res_v2.py (displacement processing)
    - generate_interferograms.py (interferogram generation)
"""

import os
import sys
import argparse
import logging
import json
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import glob
import h5py

# Import core processing modules
try:
    from polarization_handler import (
        H5CheckpointManager,
        UAVSARPolarizationHandler,
        compute_interferogram
    )
except ImportError:
    print("ERROR: polarization_handler.py not found. Please ensure it's in the same directory.")
    sys.exit(1)

try:
    from nisar_disp_res_v2 import process_single_file as process_displacement
except ImportError:
    print("ERROR: nisar_disp_res_v2.py not found. Please ensure it's in the same directory.")
    sys.exit(1)

try:
    from generate_interferograms import generate_ifg_for_polarization
except ImportError:
    print("ERROR: generate_interferograms.py not found. Please ensure it's in the same directory.")
    sys.exit(1)


class PipelineConfig:
    """Configuration management for the unified pipeline"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """Initialize configuration with defaults"""
        self.config = {
            # Directory paths
            'base_dir': None,
            'output_dir': None,
            'checkpoint_dir': None,
            'displacement_dir': None,
            
            # Processing options
            'unwrap_phase': True,
            'coherence_threshold': 0.4,
            'filter_strength': 0.5,
            'target_resolution': 30,  # meters
            
            # Parallelization
            'max_workers': max(1, mp.cpu_count() - 1),
            'site_parallel': False,
            'pair_parallel': True,
            
            # Pipeline control
            'skip_existing': True,
            'force_reprocess': False,
            'debug': False,
            'verbose': False,
            
            # SNAPHU configuration
            'snaphu_path': None,
        }
        
        # Update with provided configuration
        if config_dict:
            self.config.update(config_dict)
    
    def validate(self) -> bool:
        """Validate configuration and create necessary directories"""
        # Check required directories
        if not self.config['base_dir']:
            raise ValueError("base_dir is required")
        
        base_dir = Path(self.config['base_dir'])
        if not base_dir.exists():
            raise ValueError(f"base_dir does not exist: {base_dir}")
        
        # Set default output directory
        if not self.config['output_dir']:
            self.config['output_dir'] = base_dir.parent / f"{base_dir.name}_processed"
        
        # Set default checkpoint directory
        if not self.config['checkpoint_dir']:
            self.config['checkpoint_dir'] = Path(self.config['output_dir']) / "checkpoints"
        
        # Set default displacement directory
        if not self.config['displacement_dir']:
            self.config['displacement_dir'] = Path(self.config['output_dir']) / "displacement_30m"
        
        # Create directories
        for dir_key in ['output_dir', 'checkpoint_dir', 'displacement_dir']:
            Path(self.config[dir_key]).mkdir(parents=True, exist_ok=True)
        
        return True
    
    def save(self, filepath: Path):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: Path):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)


class UnifiedPipeline:
    """
    Unified pipeline orchestrator for UAVSAR/NISAR processing
    
    Processing stages:
    1. Site discovery and temporal pair identification
    2. Polarization detection with checkpoint validation
    3. Interferogram generation (parallel by polarization)
    4. Phase unwrapping (optional, via SNAPHU)
    5. Displacement calculation with coherence filtering
    6. Resampling to 30m with proper geographic transforms
    """
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration"""
        self.config = config.config
        self.checkpoint_mgr = H5CheckpointManager(self.config['checkpoint_dir'])
        self.pol_handler = UAVSARPolarizationHandler()
        self.logger = self._setup_logging()
        
        # Statistics tracking
        self.stats = {
            'sites_found': 0,
            'sites_processed': 0,
            'pairs_found': 0,
            'pairs_processed': 0,
            'pairs_skipped': 0,
            'pairs_failed': 0,
            'interferograms_generated': 0,
            'displacements_computed': 0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging with file and console handlers"""
        log_dir = Path(self.config['output_dir']) / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'pipeline_{timestamp}.log'
        
        logger = logging.getLogger('UnifiedPipeline')
        logger.setLevel(logging.DEBUG if self.config['debug'] else logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if self.config['verbose'] else logging.WARNING)
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        self.logger = logger
        self.logger.info(f"Logging initialized. Log file: {log_file}")
        
        return logger
    
    def run(self):
        """Execute the complete pipeline"""
        self.logger.info("="*80)
        self.logger.info("Starting Unified UAVSAR/NISAR Processing Pipeline")
        self.logger.info("="*80)
        self.logger.info(f"Base directory: {self.config['base_dir']}")
        self.logger.info(f"Output directory: {self.config['output_dir']}")
        self.logger.info(f"Target resolution: {self.config['target_resolution']}m")
        self.logger.info(f"Max workers: {self.config['max_workers']}")
        self.logger.info("="*80)
        
        start_time = datetime.now()
        
        # Discover sites
        sites = self._discover_sites()
        self.stats['sites_found'] = len(sites)
        self.logger.info(f"Discovered {len(sites)} site directories")
        
        # Process sites
        if self.config['site_parallel']:
            self._process_sites_parallel(sites)
        else:
            self._process_sites_sequential(sites)
        
        # Final reporting
        end_time = datetime.now()
        duration = end_time - start_time
        
        self._print_summary(duration)
        
        return self.stats
    
    def _discover_sites(self) -> List[Path]:
        """Discover all site directories containing H5 files"""
        base_dir = Path(self.config['base_dir'])
        sites = []
        
        # Look for directories containing H5 files
        for item in base_dir.iterdir():
            if item.is_dir():
                # Check if directory contains H5 files (directly or in subdirectories)
                h5_files = list(item.rglob("*.h5"))
                if h5_files:
                    sites.append(item)
                    self.logger.debug(f"Found site: {item.name} ({len(h5_files)} H5 files)")
        
        return sorted(sites)
    
    def _process_sites_sequential(self, sites: List[Path]):
        """Process sites sequentially with parallel pair processing"""
        for idx, site_dir in enumerate(sites, 1):
            self.logger.info(f"[SITE {idx}/{len(sites)}] Processing: {site_dir.name}")
            
            try:
                processed = self._process_site(site_dir)
                if processed:
                    self.stats['sites_processed'] += 1
            except Exception as e:
                self.logger.error(f"Failed to process site {site_dir.name}: {e}")
                if self.config['debug']:
                    import traceback
                    self.logger.debug(traceback.format_exc())
    
    def _process_sites_parallel(self, sites: List[Path]):
        """Process sites in parallel"""
        max_workers = min(self.config['max_workers'], len(sites))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_site, site): site 
                for site in sites
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Processing sites"):
                site = futures[future]
                try:
                    processed = future.result()
                    if processed:
                        self.stats['sites_processed'] += 1
                except Exception as e:
                    self.logger.error(f"Failed to process site {site.name}: {e}")
    
    def _process_site(self, site_dir: Path) -> bool:
        """Process all temporal pairs in a site directory"""
        # Find all H5 files
        h5_files = sorted(site_dir.rglob("*.h5"))
        
        if len(h5_files) < 2:
            self.logger.warning(f"Site {site_dir.name} has insufficient H5 files ({len(h5_files)})")
            return False
        
        # Find temporal pairs
        pairs = self._find_temporal_pairs(h5_files)
        self.stats['pairs_found'] += len(pairs)
        
        if not pairs:
            self.logger.warning(f"No valid temporal pairs found for {site_dir.name}")
            return False
        
        self.logger.info(f"Found {len(pairs)} temporal pairs for {site_dir.name}")
        
        # Process pairs
        if self.config['pair_parallel']:
            processed = self._process_pairs_parallel(pairs, site_dir)
        else:
            processed = self._process_pairs_sequential(pairs, site_dir)
        
        return processed > 0
    
    def _find_temporal_pairs(self, h5_files: List[Path]) -> List[Tuple[Path, Path]]:
        """Identify valid temporal pairs from H5 files"""
        # Extract dates from filenames
        dated_files = []
        for h5_file in h5_files:
            date = self._extract_date_from_filename(h5_file.name)
            if date:
                dated_files.append((h5_file, date))
        
        # Sort by date
        dated_files.sort(key=lambda x: x[1])
        
        # Generate all combinations
        import itertools
        pairs = list(itertools.combinations([f for f, d in dated_files], 2))
        
        return pairs
    
    def _extract_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract acquisition date from filename"""
        import re
        
        # Try various date patterns
        patterns = [
            r'(\d{6})',  # YYMMDD
            r'(\d{8})',  # YYYYMMDD
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, filename)
            for match in matches:
                try:
                    if len(match) == 6:
                        return datetime.strptime(match, "%y%m%d")
                    elif len(match) == 8:
                        return datetime.strptime(match, "%Y%m%d")
                except ValueError:
                    continue
        
        return None
    
    def _process_pairs_sequential(self, pairs: List[Tuple[Path, Path]], 
                                 site_dir: Path) -> int:
        """Process pairs sequentially"""
        processed = 0
        
        for idx, (ref, sec) in enumerate(pairs, 1):
            self.logger.info(f"  [PAIR {idx}/{len(pairs)}] {ref.name} -> {sec.name}")
            
            if self._process_pair(ref, sec, site_dir):
                processed += 1
                self.stats['pairs_processed'] += 1
            
        return processed
    
    def _process_pairs_parallel(self, pairs: List[Tuple[Path, Path]], 
                               site_dir: Path) -> int:
        """Process pairs in parallel"""
        processed = 0
        max_workers = self.config['max_workers']
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_pair, ref, sec, site_dir): (ref, sec)
                for ref, sec in pairs
            }
            
            for future in tqdm(as_completed(futures), total=len(futures),
                             desc=f"Processing pairs for {site_dir.name}"):
                ref, sec = futures[future]
                try:
                    if future.result():
                        processed += 1
                        self.stats['pairs_processed'] += 1
                except Exception as e:
                    self.logger.error(f"Failed to process pair {ref.name} -> {sec.name}: {e}")
                    self.stats['pairs_failed'] += 1
        
        return processed
    
    def _process_pair(self, ref: Path, sec: Path, site_dir: Path) -> bool:
        """
        Process a single interferogram pair
        
        Steps:
        1. Check checkpoint status
        2. Detect valid polarizations
        3. Generate interferogram
        4. Process displacement (if unwrapped phase exists)
        5. Update checkpoint
        """
        # Check if pair should be processed
        if not self.config['force_reprocess']:
            if not self.checkpoint_mgr.should_process_pair(str(ref), str(sec)):
                self.logger.info(f"    Skipping (already processed): {ref.name} -> {sec.name}")
                self.stats['pairs_skipped'] += 1
                return True
        
        try:
            # Determine output directory
            output_dir = self._get_pair_output_dir(ref, sec, site_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Detect polarizations
            ref_pols = self.pol_handler.find_valid_polarizations(str(ref))
            sec_pols = self.pol_handler.find_valid_polarizations(str(sec))
            common_pols = list(set(ref_pols) & set(sec_pols))
            
            if not common_pols:
                self.logger.warning(f"    No common polarizations found")
                self.checkpoint_mgr.save_pair_checkpoint(
                    str(ref), str(sec), 
                    status='no_polarizations',
                    output_dir=str(output_dir)
                )
                return False
            
            self.logger.info(f"    Common polarizations: {', '.join(common_pols)}")
            
            # Generate interferograms for each polarization
            ifg_success = False
            for pol in common_pols:
                try:
                    success = generate_ifg_for_polarization(
                        reference_path=str(ref),
                        secondary_path=str(sec),
                        output_dir=str(output_dir),
                        pol=pol,
                        unwrap=self.config['unwrap_phase'],
                        debug=self.config['debug']
                    )
                    
                    if success:
                        ifg_success = True
                        self.stats['interferograms_generated'] += 1
                        self.logger.info(f"    Generated interferogram for {pol}")
                    
                except Exception as e:
                    self.logger.error(f"    Failed to generate interferogram for {pol}: {e}")
            
            # Process displacement if unwrapped files exist
            if ifg_success:
                unw_files = list(output_dir.glob("*_unw.tif"))
                
                for unw_file in unw_files:
                    try:
                        self.logger.info(f"    Processing displacement: {unw_file.name}")
                        
                        success = process_displacement(
                            str(unw_file),
                            coherence_path=None,
                            output_dir=str(self.config['displacement_dir']),
                            coherence_threshold=self.config['coherence_threshold'],
                            filter_strength=self.config['filter_strength'],
                            debug=self.config['debug']
                        )
                        
                        if success:
                            self.stats['displacements_computed'] += 1
                            self.logger.info(f"    Displacement computed and resampled to 30m")
                        
                    except Exception as e:
                        self.logger.error(f"    Failed to process displacement: {e}")
            
            # Save checkpoint
            status = 'completed' if ifg_success else 'failed'
            self.checkpoint_mgr.save_pair_checkpoint(
                str(ref), str(sec),
                status=status,
                output_dir=str(output_dir),
                metadata={
                    'polarizations': common_pols,
                    'interferograms_generated': self.stats['interferograms_generated'],
                    'displacements_computed': self.stats['displacements_computed']
                }
            )
            
            return ifg_success
            
        except Exception as e:
            self.logger.error(f"    Error processing pair: {e}")
            if self.config['debug']:
                import traceback
                self.logger.debug(traceback.format_exc())
            
            self.stats['pairs_failed'] += 1
            return False
    
    def _get_pair_output_dir(self, ref: Path, sec: Path, site_dir: Path) -> Path:
        """Generate output directory path for a pair"""
        ref_name = ref.stem
        sec_name = sec.stem
        pair_name = f"{ref_name}__{sec_name}"
        
        return Path(self.config['output_dir']) / site_dir.name / pair_name
    
    def _print_summary(self, duration):
        """Print processing summary"""
        self.logger.info("="*80)
        self.logger.info("PIPELINE PROCESSING COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Total duration: {duration}")
        self.logger.info("")
        self.logger.info("Statistics:")
        self.logger.info(f"  Sites discovered: {self.stats['sites_found']}")
        self.logger.info(f"  Sites processed: {self.stats['sites_processed']}")
        self.logger.info(f"  Pairs discovered: {self.stats['pairs_found']}")
        self.logger.info(f"  Pairs processed: {self.stats['pairs_processed']}")
        self.logger.info(f"  Pairs skipped: {self.stats['pairs_skipped']}")
        self.logger.info(f"  Pairs failed: {self.stats['pairs_failed']}")
        self.logger.info(f"  Interferograms generated: {self.stats['interferograms_generated']}")
        self.logger.info(f"  Displacements computed: {self.stats['displacements_computed']}")
        self.logger.info("="*80)
        
        # Save statistics to JSON
        stats_file = Path(self.config['output_dir']) / 'pipeline_statistics.json'
        with open(stats_file, 'w') as f:
            stats_with_duration = self.stats.copy()
            stats_with_duration['duration_seconds'] = duration.total_seconds()
            stats_with_duration['completed_at'] = datetime.now().isoformat()
            json.dump(stats_with_duration, f, indent=2)
        
        self.logger.info(f"Statistics saved to: {stats_file}")


def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Unified UAVSAR/NISAR Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python unified_pipeline.py --base_dir /data/nisar --output_dir /data/processed
  
  # With custom settings
  python unified_pipeline.py --base_dir /data/nisar \\
      --output_dir /data/processed \\
      --max_workers 8 \\
      --coherence_threshold 0.5 \\
      --unwrap
  
  # Resume processing with force reprocess
  python unified_pipeline.py --base_dir /data/nisar \\
      --output_dir /data/processed \\
      --force_reprocess
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--base_dir', required=True,
        help='Base directory containing site subdirectories with H5 files'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output_dir', default=None,
        help='Output directory for processed data (default: base_dir_processed)'
    )
    parser.add_argument(
        '--checkpoint_dir', default=None,
        help='Checkpoint directory (default: output_dir/checkpoints)'
    )
    parser.add_argument(
        '--displacement_dir', default=None,
        help='Displacement output directory (default: output_dir/displacement_30m)'
    )
    
    # Processing options
    parser.add_argument(
        '--unwrap', action='store_true',
        help='Enable phase unwrapping'
    )
    parser.add_argument(
        '--no-unwrap', dest='unwrap', action='store_false',
        help='Disable phase unwrapping'
    )
    parser.set_defaults(unwrap=True)
    
    parser.add_argument(
        '--coherence_threshold', type=float, default=0.4,
        help='Coherence threshold for filtering (default: 0.4)'
    )
    parser.add_argument(
        '--filter_strength', type=float, default=0.5,
        help='Filter strength (default: 0.5)'
    )
    parser.add_argument(
        '--target_resolution', type=int, default=30,
        help='Target resolution in meters (default: 30)'
    )
    
    # Parallelization
    parser.add_argument(
        '--max_workers', type=int, default=None,
        help='Maximum number of parallel workers (default: CPU count - 1)'
    )
    parser.add_argument(
        '--site_parallel', action='store_true',
        help='Process sites in parallel (default: sequential sites, parallel pairs)'
    )
    
    # Pipeline control
    parser.add_argument(
        '--skip_existing', action='store_true', default=True,
        help='Skip already processed pairs (default: True)'
    )
    parser.add_argument(
        '--force_reprocess', action='store_true',
        help='Force reprocessing of all pairs'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose console output'
    )
    
    # SNAPHU
    parser.add_argument(
        '--snaphu_path', default=None,
        help='Path to SNAPHU executable'
    )
    
    # Configuration file
    parser.add_argument(
        '--config', default=None,
        help='Load configuration from JSON file'
    )
    parser.add_argument(
        '--save_config', default=None,
        help='Save configuration to JSON file and exit'
    )
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = PipelineConfig.load(Path(args.config))
        print(f"Loaded configuration from: {args.config}")
    else:
        # Build configuration from command-line arguments
        config_dict = {
            'base_dir': args.base_dir,
            'output_dir': args.output_dir,
            'checkpoint_dir': args.checkpoint_dir,
            'displacement_dir': args.displacement_dir,
            'unwrap_phase': args.unwrap,
            'coherence_threshold': args.coherence_threshold,
            'filter_strength': args.filter_strength,
            'target_resolution': args.target_resolution,
            'max_workers': args.max_workers if args.max_workers else max(1, mp.cpu_count() - 1),
            'site_parallel': args.site_parallel,
            'skip_existing': args.skip_existing,
            'force_reprocess': args.force_reprocess,
            'debug': args.debug,
            'verbose': args.verbose,
            'snaphu_path': args.snaphu_path,
        }
        config = PipelineConfig(config_dict)
    
    # Save configuration if requested
    if args.save_config:
        config.save(Path(args.save_config))
        print(f"Configuration saved to: {args.save_config}")
        return 0
    
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    
    # Create and run pipeline
    pipeline = UnifiedPipeline(config)
    
    try:
        stats = pipeline.run()
        
        # Return non-zero exit code if significant failures occurred
        if stats['pairs_failed'] > stats['pairs_processed']:
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


# # Basic usage
# python unified_pipeline.py \
#     --base_dir /Volumes/All/nisar/nisarsims_source \
#     --output_dir /Volumes/All/nisarsims_processed

# # With custom settings
# python unified_pipeline.py \
#     --base_dir /Volumes/All/nisar/nisarsims_source \
#     --output_dir /Volumes/All/nisarsims_processed \
#     --max_workers 8 \
#     --coherence_threshold 0.5 \
#     --unwrap \
#     --verbose

# # Force reprocess all pairs
# python unified_pipeline.py \
#     --base_dir /Volumes/All/nisar/nisarsims_source \
#     --output_dir /Volumes/All/nisarsims_processed \
#     --force_reprocess \
#     --debug