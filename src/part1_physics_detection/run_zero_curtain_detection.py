#!/usr/bin/env python3
"""
Zero-Curtain Detection Pipeline Integration

Integrates physics-informed zero-curtain detection with the existing Arctic 
pipeline to generate PINSZC (Physics-Informed In Situ Zero-Curtain) dataframe.

Usage:
    python src/part1_physics_detection/run_zero_curtain_detection.py \
        --input outputs/merged_compressed_corrected_final.parquet \
        --output outputs/pinszc_dataset.parquet \
        --config configs/detection_config.yaml

Institution: [REDACTED_AFFILIATION] Arctic Research
Date: October 2025
"""

import argparse
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.part1_physics_detection.detection_config import DetectionConfiguration, DEFAULT_CONFIG
from src.part1_physics_detection.zero_curtain_detector import PhysicsInformedZeroCurtainDetector
from src.utils.logging_config import setup_logger

warnings.filterwarnings('ignore')


class PINSZCPipeline:
    """
    Pipeline orchestrator for Physics-Informed In Situ Zero-Curtain detection.
    
    This class manages the end-to-end process of:
    1. Loading preprocessed in situ measurements
    2. Applying physics-informed detection algorithms
    3. Generating PINSZC dataset with comprehensive event characterization
    4. Validating detection quality and generating diagnostics
    """
    
    def __init__(
        self,
        config: DetectionConfiguration,
        logger: Optional[object] = None
    ):
        """
        Initialize PINSZC pipeline with configuration.
        
        Args:
            config: Detection configuration object
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or setup_logger('pinszc_pipeline')
        
        # Initialize detector with configuration
        self.detector = PhysicsInformedZeroCurtainDetector(config)
        
        # Pipeline state tracking
        self.total_sites = 0
        self.processed_sites = 0
        self.sites_with_events = 0
        self.total_events = 0
        self.processing_start_time = None
        
        self.logger.info("PINSZC Pipeline initialized")
        self.logger.info(f"\n{config.summary()}")
    
    def load_input_data(self, input_path: str) -> pd.DataFrame:
        """
        Load preprocessed in situ measurement data.
        
        Args:
            input_path: Path to input parquet file
            
        Returns:
            DataFrame with in situ measurements
        """
        self.logger.info(f"Loading input data from: {input_path}")
        
        try:
            df = pd.read_parquet(input_path)
            self.logger.info(f" Loaded {len(df):,} measurements")
            self.logger.info(f"   Columns: {list(df.columns)}")
            self.logger.info(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f" Failed to load input data: {e}")
            raise
    
    def filter_cold_season_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for cold season measurements (Sept-May) when zero-curtain occurs.
        
        Args:
            df: Input dataframe
            
        Returns:
            Filtered dataframe
        """
        self.logger.info("Filtering for cold season data...")
        
        cold_months = self.config.processing.cold_season_months
        df_cold = df[df['datetime'].dt.month.isin(cold_months)].copy()
        
        self.logger.info(f" Cold season filter applied:")
        self.logger.info(f"   Months: {cold_months}")
        self.logger.info(f"   Before: {len(df):,} measurements")
        self.logger.info(f"   After: {len(df_cold):,} measurements")
        self.logger.info(f"   Retention: {len(df_cold)/len(df)*100:.1f}%")
        
        return df_cold
    
    def identify_sites(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify unique measurement sites and assess data availability.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame of unique sites with metadata
        """
        self.logger.info("Identifying unique measurement sites...")
        
        # Group by site coordinates and source
        site_groups = df.groupby(['latitude', 'longitude', 'source']).agg({
            'datetime': ['min', 'max', 'count'],
            'soil_temp': ['mean', 'std', 'min', 'max'],
            'soil_temp_depth_zone': lambda x: x.mode()[0] if len(x) > 0 else None
        }).reset_index()
        
        site_groups.columns = [
            'latitude', 'longitude', 'source',
            'date_start', 'date_end', 'n_measurements',
            'temp_mean', 'temp_std', 'temp_min', 'temp_max',
            'primary_depth_zone'
        ]
        
        # Calculate temporal coverage
        site_groups['temporal_span_days'] = (
            site_groups['date_end'] - site_groups['date_start']
        ).dt.days
        
        # Filter by minimum measurements
        min_measurements = self.config.processing.min_measurements_per_site
        site_groups_filtered = site_groups[
            site_groups['n_measurements'] >= min_measurements
        ].copy()
        
        self.total_sites = len(site_groups_filtered)
        
        self.logger.info(f" Site identification complete:")
        self.logger.info(f"   Total unique sites: {len(site_groups):,}")
        self.logger.info(f"   Sites with ≥{min_measurements} measurements: {self.total_sites:,}")
        self.logger.info(f"   Latitude range: {site_groups_filtered['latitude'].min():.1f}° to "
                        f"{site_groups_filtered['latitude'].max():.1f}°N")
        self.logger.info(f"   Longitude range: {site_groups_filtered['longitude'].min():.1f}° to "
                        f"{site_groups_filtered['longitude'].max():.1f}°E")
        
        return site_groups_filtered
    
    def apply_permafrost_screening(
        self, 
        sites: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Screen sites for permafrost suitability before detection.
        
        Args:
            sites: DataFrame of unique sites
            
        Returns:
            Tuple of (suitable_sites, unsuitable_sites)
        """
        self.logger.info("Screening sites for permafrost suitability...")
        
        suitable_sites = []
        unsuitable_sites = []
        
        for idx, site in tqdm(sites.iterrows(), total=len(sites), 
                             desc="Permafrost screening"):
            
            lat, lon = site['latitude'], site['longitude']
            
            # Get permafrost properties
            pf_props = self.detector.get_site_permafrost_properties(lat, lon)
            
            site_info = site.to_dict()
            site_info.update({
                'permafrost_prob': pf_props['permafrost_prob'],
                'permafrost_zone': pf_props['permafrost_zone'],
                'is_permafrost_suitable': pf_props['is_permafrost_suitable'],
                'suitability_reason': pf_props.get('suitability_reason', '')
            })
            
            if pf_props['is_permafrost_suitable']:
                suitable_sites.append(site_info)
            else:
                unsuitable_sites.append(site_info)
        
        suitable_df = pd.DataFrame(suitable_sites)
        unsuitable_df = pd.DataFrame(unsuitable_sites)
        
        self.logger.info(f" Permafrost screening complete:")
        self.logger.info(f"   Suitable sites: {len(suitable_df):,} "
                        f"({len(suitable_df)/len(sites)*100:.1f}%)")
        self.logger.info(f"   Unsuitable sites: {len(unsuitable_df):,}")
        
        return suitable_df, unsuitable_df
    
    def detect_zero_curtain_events(
        self,
        df: pd.DataFrame,
        sites: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply physics-informed zero-curtain detection to all suitable sites.
        
        Args:
            df: Full measurement dataframe
            sites: Suitable sites to process
            
        Returns:
            DataFrame of detected zero-curtain events
        """
        self.logger.info(f"Applying zero-curtain detection to {len(sites):,} sites...")
        
        all_events = []
        self.processing_start_time = datetime.now()
        
        for idx, site in tqdm(sites.iterrows(), total=len(sites),
                             desc="Zero-curtain detection"):
            
            lat, lon, source = site['latitude'], site['longitude'], site['source']
            
            # Extract site data
            try:
                site_data = df[
                    (df['latitude'] == lat) &
                    (df['longitude'] == lon) &
                    (df['source'] == source)
                ].copy()
                
                if len(site_data) < self.config.processing.min_measurements_per_site:
                    continue
                
                # Apply detection
                events = self.detector.detect_zero_curtain_with_physics(
                    site_data, lat, lon
                )
                
                # Add site metadata to events
                for event in events:
                    event.update({
                        'latitude': lat,
                        'longitude': lon,
                        'source': source,
                        'site_index': idx
                    })
                
                all_events.extend(events)
                
                if len(events) > 0:
                    self.sites_with_events += 1
                    self.total_events += len(events)
                
                self.processed_sites += 1
                
                # Progress reporting
                if self.processed_sites % self.config.processing.progress_report_interval == 0:
                    self._report_progress()
                
                # Incremental checkpointing
                if (self.config.processing.save_incremental_checkpoints and
                    self.processed_sites % self.config.processing.incremental_save_interval == 0):
                    self._save_incremental_checkpoint(all_events)
                
            except Exception as e:
                self.logger.warning(f"Failed to process site {lat:.3f}, {lon:.3f}: {e}")
                continue
        
        # Create comprehensive event dataframe
        if all_events:
            events_df = pd.DataFrame(all_events)
            self.logger.info(f" Detection complete: {len(events_df):,} events detected")
            return events_df
        else:
            self.logger.warning("  No zero-curtain events detected")
            return pd.DataFrame()
    
    def _report_progress(self):
        """Log progress statistics during detection."""
        
        detection_rate = (self.sites_with_events / self.processed_sites * 100 
                         if self.processed_sites > 0 else 0)
        events_per_site = (self.total_events / self.processed_sites 
                          if self.processed_sites > 0 else 0)
        
        elapsed_time = (datetime.now() - self.processing_start_time).total_seconds()
        sites_per_minute = self.processed_sites / (elapsed_time / 60) if elapsed_time > 0 else 0
        
        self.logger.info(f"Progress: {self.processed_sites}/{self.total_sites} sites")
        self.logger.info(f"  Sites with events: {self.sites_with_events} ({detection_rate:.1f}%)")
        self.logger.info(f"  Total events: {self.total_events} ({events_per_site:.2f} per site)")
        self.logger.info(f"  Processing rate: {sites_per_minute:.1f} sites/min")
    
    def _save_incremental_checkpoint(self, events: list):
        """Save incremental checkpoint during processing."""
        
        checkpoint_path = (
            Path("outputs") / 
            f"pinszc_checkpoint_site_{self.processed_sites}.parquet"
        )
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        try:
            checkpoint_df = pd.DataFrame(events)
            checkpoint_df.to_parquet(
                checkpoint_path,
                compression=self.config.processing.compression,
                index=False
            )
            self.logger.info(f" Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self.logger.warning(f"Checkpoint save failed: {e}")
    
    def add_derived_features(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived classification features to event dataframe.
        
        Args:
            events_df: Raw event dataframe
            
        Returns:
            Enhanced dataframe with categorical features
        """
        self.logger.info("Adding derived classification features...")
        
        df = events_df.copy()
        
        # Intensity categories
        df['intensity_category'] = pd.cut(
            df['intensity_percentile'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['weak', 'moderate', 'strong', 'extreme']
        )
        
        # Duration categories
        df['duration_category'] = pd.cut(
            df['duration_hours'],
            bins=[0, 72, 168, 336, np.inf],
            labels=['short', 'medium', 'long', 'extended']
        )
        
        # Spatial extent categories
        df['extent_category'] = pd.cut(
            df['spatial_extent_meters'],
            bins=[0, 0.3, 0.8, 1.5, np.inf],
            labels=['shallow', 'moderate', 'deep', 'very_deep']
        )
        
        # CryoGrid-specific categories if available
        if 'cryogrid_thermal_conductivity' in df.columns:
            df['thermal_conductivity_category'] = pd.cut(
                df['cryogrid_thermal_conductivity'],
                bins=[0, 0.5, 1.0, 2.0, np.inf],
                labels=['low', 'medium', 'high', 'very_high']
            )
        
        if 'cryogrid_enthalpy_stability' in df.columns:
            df['enthalpy_stability_category'] = pd.cut(
                df['cryogrid_enthalpy_stability'],
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=['unstable', 'moderately_stable', 'stable', 'very_stable']
            )
        
        self.logger.info(f" Added {len([c for c in df.columns if 'category' in c])} "
                        "categorical features")
        
        return df
    
    def validate_output(self, events_df: pd.DataFrame) -> Dict:
        """
        Validate PINSZC dataset quality and generate diagnostics.
        
        Args:
            events_df: Final event dataframe
            
        Returns:
            Dictionary of validation metrics
        """
        self.logger.info("Validating PINSZC dataset...")
        
        validation = {}
        
        # Required features check
        required_features = [
            'intensity_percentile', 'duration_hours', 'spatial_extent_meters',
            'start_time', 'end_time', 'latitude', 'longitude'
        ]
        missing_features = [f for f in required_features if f not in events_df.columns]
        
        if missing_features:
            self.logger.error(f" Missing required features: {missing_features}")
            validation['missing_features'] = missing_features
            validation['is_valid'] = False
            return validation
        
        validation['missing_features'] = []
        
        # Data quality checks
        validation['n_events'] = len(events_df)
        validation['n_unique_sites'] = events_df[['latitude', 'longitude']].drop_duplicates().shape[0]
        validation['n_sources'] = events_df['source'].nunique()
        
        # Feature statistics
        validation['intensity_stats'] = {
            'mean': float(events_df['intensity_percentile'].mean()),
            'std': float(events_df['intensity_percentile'].std()),
            'min': float(events_df['intensity_percentile'].min()),
            'max': float(events_df['intensity_percentile'].max())
        }
        
        validation['duration_stats'] = {
            'mean_hours': float(events_df['duration_hours'].mean()),
            'median_hours': float(events_df['duration_hours'].median()),
            'min_hours': float(events_df['duration_hours'].min()),
            'max_hours': float(events_df['duration_hours'].max())
        }
        
        validation['spatial_extent_stats'] = {
            'mean_meters': float(events_df['spatial_extent_meters'].mean()),
            'median_meters': float(events_df['spatial_extent_meters'].median()),
            'min_meters': float(events_df['spatial_extent_meters'].min()),
            'max_meters': float(events_df['spatial_extent_meters'].max())
        }
        
        # Geographic coverage
        validation['latitude_range'] = {
            'min': float(events_df['latitude'].min()),
            'max': float(events_df['latitude'].max())
        }
        validation['longitude_range'] = {
            'min': float(events_df['longitude'].min()),
            'max': float(events_df['longitude'].max())
        }
        
        # Temporal coverage
        validation['temporal_range'] = {
            'start': str(events_df['start_time'].min()),
            'end': str(events_df['end_time'].max())
        }
        
        # Detection method distribution
        if 'detection_method' in events_df.columns:
            validation['detection_methods'] = (
                events_df['detection_method'].value_counts().to_dict()
            )
        
        # Quality flags
        validation['is_valid'] = True
        validation['warnings'] = []
        
        if validation['n_events'] < 100:
            validation['warnings'].append("Low event count (<100)")
        
        if validation['intensity_stats']['mean'] < 0.3:
            validation['warnings'].append("Low mean intensity (<0.3)")
        
        if validation['n_unique_sites'] < 10:
            validation['warnings'].append("Low spatial coverage (<10 sites)")
        
        self.logger.info(" Validation complete:")
        self.logger.info(f"   Events: {validation['n_events']:,}")
        self.logger.info(f"   Sites: {validation['n_unique_sites']:,}")
        self.logger.info(f"   Mean intensity: {validation['intensity_stats']['mean']:.3f}")
        self.logger.info(f"   Mean duration: {validation['duration_stats']['mean_hours']:.1f}h")
        self.logger.info(f"   Warnings: {len(validation['warnings'])}")
        
        return validation
    
    def save_outputs(
        self,
        events_df: pd.DataFrame,
        suitable_sites: pd.DataFrame,
        unsuitable_sites: pd.DataFrame,
        validation: Dict,
        output_path: str
    ):
        """
        Save all pipeline outputs with comprehensive metadata.
        
        Args:
            events_df: Final PINSZC dataset
            suitable_sites: Suitable sites dataframe
            unsuitable_sites: Unsuitable sites dataframe
            validation: Validation metrics
            output_path: Base output path
        """
        self.logger.info("Saving pipeline outputs...")
        
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(exist_ok=True)
        
        # Save main PINSZC dataset
        events_df.to_parquet(
            output_path,
            compression=self.config.processing.compression,
            index=False
        )
        self.logger.info(f" PINSZC dataset: {output_path}")
        
        # Save site suitability
        suitability_path = output_path.parent / f"{output_path.stem}_site_suitability.parquet"
        suitable_sites['is_suitable'] = True
        unsuitable_sites['is_suitable'] = False
        all_sites = pd.concat([suitable_sites, unsuitable_sites], ignore_index=True)
        all_sites.to_parquet(suitability_path, compression='snappy', index=False)
        self.logger.info(f" Site suitability: {suitability_path}")
        
        # Save validation metadata
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        metadata = {
            'pipeline_version': '1.0.0',
            'execution_timestamp': datetime.now().isoformat(),
            'configuration': self.config.to_dict(),
            'validation': validation,
            'processing_summary': {
                'total_sites_screened': self.total_sites,
                'sites_processed': self.processed_sites,
                'sites_with_events': self.sites_with_events,
                'total_events_detected': self.total_events,
                'detection_rate_percent': (self.sites_with_events / self.processed_sites * 100
                                          if self.processed_sites > 0 else 0)
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f" Metadata: {metadata_path}")
        
        # Save configuration
        config_path = output_path.parent / f"{output_path.stem}_config.txt"
        with open(config_path, 'w') as f:
            f.write(self.config.summary())
        self.logger.info(f" Configuration: {config_path}")
    
    def run(
        self,
        input_path: str,
        output_path: str
    ) -> pd.DataFrame:
        """
        Execute full PINSZC generation pipeline.
        
        Args:
            input_path: Path to input in situ measurements
            output_path: Path for output PINSZC dataset
            
        Returns:
            Final PINSZC dataframe
        """
        self.logger.info("="*80)
        self.logger.info("PHYSICS-INFORMED IN SITU ZERO-CURTAIN DETECTION PIPELINE")
        self.logger.info("="*80)
        
        # Step 1: Load data
        df = self.load_input_data(input_path)
        
        # Step 2: Filter cold season
        df_cold = self.filter_cold_season_data(df)
        
        # Step 3: Identify sites
        sites = self.identify_sites(df_cold)
        
        # Step 4: Permafrost screening
        suitable_sites, unsuitable_sites = self.apply_permafrost_screening(sites)
        
        # Step 5: Zero-curtain detection
        events_df = self.detect_zero_curtain_events(df_cold, suitable_sites)
        
        if events_df.empty:
            self.logger.error(" No events detected - pipeline terminating")
            return pd.DataFrame()
        
        # Step 6: Add derived features
        events_df = self.add_derived_features(events_df)
        
        # Step 7: Validate output
        validation = self.validate_output(events_df)
        
        # Step 8: Save outputs
        self.save_outputs(
            events_df,
            suitable_sites,
            unsuitable_sites,
            validation,
            output_path
        )
        
        self.logger.info("="*80)
        self.logger.info("PIPELINE EXECUTION COMPLETE")
        self.logger.info("="*80)
        
        return events_df


def parse_args():
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Physics-Informed Zero-Curtain Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Path to input in situ measurement parquet file"
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default="outputs/pinszc_dataset.parquet",
        help="Path for output PINSZC dataset"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="Path to custom configuration YAML (optional)"
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main execution entry point."""
    
    args = parse_args()
    
    # Initialize configuration
    config = DEFAULT_CONFIG
    config.processing.verbose = args.verbose
    
    # Initialize pipeline
    pipeline = PINSZCPipeline(config=config)
    
    # Execute pipeline
    try:
        pinszc_df = pipeline.run(
            input_path=args.input,
            output_path=args.output
        )
        
        if not pinszc_df.empty:
            print("\n PINSZC dataset successfully generated")
            print(f"   Output: {args.output}")
            print(f"   Events: {len(pinszc_df):,}")
            return 0
        else:
            print("\n Pipeline failed - no events detected")
            return 1
            
    except Exception as e:
        print(f"\n Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
