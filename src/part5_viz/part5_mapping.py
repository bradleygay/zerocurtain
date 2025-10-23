#!/usr/bin/env python3

"""
PART V: CIRCUMARCTIC MAPPING AND VISUALIZATION PIPELINE
=======================================================
Comprehensive geospatial analysis and explainability visualization suite

Generates:
1. High-resolution circumarctic maps (annual, seasonal, monthly)
2. Multi-decadal time series analysis (1891-2024)
3. Regional comparison visualizations
4. Model explainability figures
5. Comprehensive statistical summaries

[RESEARCHER] Gay
Arctic Zero-Curtain Detection Research
[RESEARCHER] Sciences Laboratory

Input: circumarctic_zero_curtain_predictions_complete.parquet (from Part IV)
Output: Maps, time series, statistics, explainability figures
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PART V: CIRCUMARCTIC MAPPING AND VISUALIZATION PIPELINE")
print("="*80)
print(f"Timestamp: {datetime.now()}")
print()

# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================

def validate_environment():
    """Validate required packages for mapping and visualization"""
    print("="*80)
    print("VALIDATING ENVIRONMENT")
    print("="*80)
    
    required_packages = {
        'numpy': 'np',
        'pandas': 'pd',
        'xarray': 'xr',
        'geopandas': 'gpd',
        'matplotlib': 'plt',
        'cartopy': 'ccrs',
        'scipy': 'scipy',
        'rasterio': 'rasterio',
        'shapely': 'shapely',
        'pyarrow': 'pyarrow'
    }
    
    missing = []
    for package, alias in required_packages.items():
        try:
            __import__(package)
            print(f" {package}: available")
        except ImportError:
            print(f" {package}: MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("\n All required packages available")
    return True

# ============================================================================
# FILE PATH VALIDATION
# ============================================================================

def validate_part4_outputs(base_dir=None):
    """Validate Part IV outputs exist before proceeding"""
    print("\n" + "="*80)
    print("VALIDATING PART IV OUTPUTS")
    print("="*80)
    
    if base_dir is None:
        base_dir = Path.home() / "arctic_zero_curtain_pipeline"
    else:
        base_dir = Path(base_dir)
    
    # Required Part IV output
    predictions_file = base_dir / 'outputs' / 'part4_transfer_learning' / 'predictions' / 'circumarctic_zero_curtain_predictions_complete.parquet'
    
    if not predictions_file.exists():
        print(f" Part IV predictions not found: {predictions_file}")
        print("\nPlease run Part IV first:")
        print("  python part4_transfer_learning.py --epochs 10 --batch-size 64")
        return False, None
    
    # Check file size
    file_size_gb = predictions_file.stat().st_size / 1e9
    print(f" Predictions file found: {predictions_file}")
    print(f"  Size: {file_size_gb:.2f} GB")
    
    # Validate parquet file
    try:
        import pandas as pd
        df = pd.read_parquet(predictions_file)
        print(f"  Records: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        
        # Check required columns
        required_cols = ['latitude', 'longitude', 'start_time',
                        'predicted_intensity_percentile', 'predicted_duration_hours',
                        'predicted_spatial_extent_meters']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f" Missing required columns: {missing_cols}")
            return False, None
        
        print(f"  Date range: {df['start_time'].min()} to {df['start_time'].max()}")
        print(f"  Spatial extent: {df['latitude'].min():.2f}°N to {df['latitude'].max():.2f}°N")
        print(f"                 {df['longitude'].min():.2f}°E to {df['longitude'].max():.2f}°E")
        
        return True, predictions_file
        
    except Exception as e:
        print(f" Error validating predictions file: {e}")
        return False, None

# ============================================================================
# NATURAL EARTH DATA SETUP
# ============================================================================

def setup_natural_earth_data(base_dir=None):
    """Setup Natural Earth data for land masking"""
    print("\n" + "="*80)
    print("SETTING UP NATURAL EARTH DATA")
    print("="*80)
    
    if base_dir is None:
        base_dir = Path.home()
    else:
        base_dir = Path(base_dir)
    
    ne_dir = base_dir / "natural_earth_data"
    
    # Check if Natural Earth data already exists
    if ne_dir.exists():
        land_shp = ne_dir / "ne_10m_land" / "ne_10m_land.shp"
        lakes_shp = ne_dir / "ne_10m_lakes" / "ne_10m_lakes.shp"
        
        if land_shp.exists() and lakes_shp.exists():
            print(f" Natural Earth data found: {ne_dir}")
            print(f"  Land polygons: {land_shp}")
            print(f"  Lake polygons: {lakes_shp}")
            return True, ne_dir
    
    print(f"Natural Earth data not found at: {ne_dir}")
    print("Data will be downloaded automatically during mapping")
    print("Location will be: {ne_dir}")
    
    # Create directory
    ne_dir.mkdir(parents=True, exist_ok=True)
    
    return True, ne_dir

# ============================================================================
# OUTPUT DIRECTORY SETUP
# ============================================================================

def setup_output_directories(base_dir=None):
    """Create Part V output directory structure"""
    print("\n" + "="*80)
    print("SETTING UP OUTPUT DIRECTORIES")
    print("="*80)
    
    if base_dir is None:
        base_dir = Path.home() / "arctic_zero_curtain_pipeline"
    else:
        base_dir = Path(base_dir)
    
    part5_base = base_dir / 'outputs' / 'part5_mapping_visualization'
    
    subdirs = {
        'Maps - Annual': part5_base / 'maps' / 'annual',
        'Maps - Seasonal': part5_base / 'maps' / 'seasonal',
        'Maps - Monthly': part5_base / 'maps' / 'monthly',
        'Time Series': part5_base / 'time_series',
        'Regional Analysis': part5_base / 'regional_analysis',
        'Statistics': part5_base / 'statistics',
        'Explainability': part5_base / 'explainability',
        'Logs': part5_base / 'logs'
    }
    
    for name, path in subdirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f" {name}: {path}")
    
    return part5_base

# ============================================================================
# MAPPING PIPELINE INTEGRATION
# ============================================================================

def integrate_mapping_pipeline(predictions_file, natural_earth_dir, output_dir):
    """Integrate mapping.py functionality for Part V"""
    print("\n" + "="*80)
    print("INTEGRATING MAPPING PIPELINE")
    print("="*80)
    
    # Import mapping functionality
    mapping_script = Path(__file__).parent / "mapping.py"
    
    if not mapping_script.exists():
        print(f" mapping.py not found at: {mapping_script}")
        print("Please ensure mapping.py is in the same directory")
        return False
    
    print(f" Found mapping script: {mapping_script}")
    
    # Import mapper class
    sys.path.insert(0, str(mapping_script.parent))
    
    try:
        from mapping import HybridArcticMapper
        print(" HybridArcticMapper imported successfully")
        
        # Load predictions
        import pandas as pd
        print(f"\nLoading predictions from: {predictions_file}")
        df = pd.read_parquet(predictions_file)
        print(f" Loaded {len(df):,} predictions")
        
        # Prepare data
        df['datetime'] = pd.to_datetime(df['start_time'])
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        
        # Rename prediction columns to match mapper expectations
        df['duration_hours'] = df['predicted_duration_hours']
        df['spatial_extent_meters'] = df['predicted_spatial_extent_meters']
        df['intensity_percentile'] = df['predicted_intensity_percentile']
        
        print("\n Data prepared for mapping")
        print(f"  Variables: duration_hours, spatial_extent_meters, intensity_percentile")
        print(f"  Temporal range: {df['year'].min()} - {df['year'].max()}")
        
        # Initialize mapper
        print("\nInitializing HybridArcticMapper...")
        mapper = HybridArcticMapper(
            bbox=(-180, 49, 180, 90),  # Full Arctic domain
            resolution_deg=0.05  # 5.5 km resolution
        )
        print(" Mapper initialized")
        
        # Set Natural Earth data directory
        mapper.natural_earth_data_dir = str(natural_earth_dir)
        
        # Process with enhanced pipeline
        print("\n" + "="*80)
        print("EXECUTING MAPPING PIPELINE")
        print("="*80)
        
        map_paths = mapper.process_enhanced_hybrid_pipeline(
            df,
            output_dir=str(output_dir / 'maps')
        )
        
        print(f"\n Mapping pipeline complete")
        print(f"  Generated {len(map_paths)} maps")
        
        return True
        
    except Exception as e:
        print(f" Error in mapping pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# TIME SERIES ANALYSIS
# ============================================================================

def generate_time_series_analysis(predictions_file, output_dir):
    """Generate comprehensive time series analysis"""
    print("\n" + "="*80)
    print("GENERATING TIME SERIES ANALYSIS")
    print("="*80)
    
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats
        
        # Load predictions
        df = pd.read_parquet(predictions_file)
        df['datetime'] = pd.to_datetime(df['start_time'])
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        
        variables = ['predicted_intensity_percentile', 'predicted_duration_hours', 
                    'predicted_spatial_extent_meters']
        
        # Annual trends
        print("\nGenerating annual trend analysis...")
        annual_data = df.groupby('year')[variables].agg(['mean', 'std', 'count']).reset_index()
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 18), facecolor='white')
        
        for i, var in enumerate(variables):
            ax = axes[i]
            years = annual_data['year']
            means = annual_data[(var, 'mean')]
            stds = annual_data[(var, 'std')]
            
            # Calculate trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, means)
            trend_line = slope * years + intercept
            
            # Plot
            ax.plot(years, means, 'o-', linewidth=2, markersize=4, label='Annual Mean')
            ax.fill_between(years, means - stds, means + stds, alpha=0.3, label='±1 STD')
            ax.plot(years, trend_line, '--', linewidth=2, color='red',
                   label=f'Trend (R²={r_value**2:.3f}, p={p_value:.3e})')
            
            ax.set_title(f'{var.replace("predicted_", "").replace("_", " ").title()} Trends',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel(var.replace('predicted_', '').replace('_', ' ').title(), fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Circumarctic Zero-Curtain Multi-Decadal Trends', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = output_dir / 'time_series' / 'multidecadal_trends.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f" Annual trends saved: {output_path}")
        
        # Seasonal patterns
        print("Generating seasonal pattern analysis...")
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}
        
        df['season'] = df['month'].map(season_map)
        df_seasonal = df[df['season'].notna()]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor='white')
        
        for i, var in enumerate(variables):
            ax = axes[i]
            seasonal_data = df_seasonal.groupby('season')[var].agg(['mean', 'std'])
            
            seasons = ['Winter', 'Spring', 'Fall']
            means = [seasonal_data.loc[s, 'mean'] if s in seasonal_data.index else 0 for s in seasons]
            stds = [seasonal_data.loc[s, 'std'] if s in seasonal_data.index else 0 for s in seasons]
            
            ax.bar(seasons, means, yerr=stds, alpha=0.7, capsize=5)
            ax.set_title(var.replace('predicted_', '').replace('_', ' ').title(),
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Seasonal Zero-Curtain Patterns', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = output_dir / 'time_series' / 'seasonal_patterns.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f" Seasonal patterns saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f" Error in time series analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# REGIONAL ANALYSIS
# ============================================================================

def generate_regional_analysis(predictions_file, output_dir):
    """Generate regional comparison analysis"""
    print("\n" + "="*80)
    print("GENERATING REGIONAL ANALYSIS")
    print("="*80)
    
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Load predictions
        df = pd.read_parquet(predictions_file)
        df['datetime'] = pd.to_datetime(df['start_time'])
        df['year'] = df['datetime'].dt.year
        
        # Define Arctic regions
        regions = {
            'Alaska': {'lat_min': 60, 'lat_max': 72, 'lon_min': -180, 'lon_max': -130},
            'Canadian Arctic': {'lat_min': 65, 'lat_max': 85, 'lon_min': -130, 'lon_max': -60},
            'Greenland': {'lat_min': 60, 'lat_max': 85, 'lon_min': -60, 'lon_max': -10},
            'Scandinavia': {'lat_min': 60, 'lat_max': 75, 'lon_min': 0, 'lon_max': 40},
            'Western Siberia': {'lat_min': 60, 'lat_max': 78, 'lon_min': 60, 'lon_max': 120},
            'Eastern Siberia': {'lat_min': 60, 'lat_max': 78, 'lon_min': 120, 'lon_max': 180}
        }
        
        variables = ['predicted_intensity_percentile', 'predicted_duration_hours',
                    'predicted_spatial_extent_meters']
        
        fig, axes = plt.subplots(len(variables), 1, figsize=(20, 6*len(variables)), 
                                facecolor='white')
        
        if len(variables) == 1:
            axes = [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))
        
        for var_idx, var in enumerate(variables):
            ax = axes[var_idx]
            
            for region_idx, (region_name, bounds) in enumerate(regions.items()):
                # Filter data for region
                region_data = df[
                    (df['latitude'] >= bounds['lat_min']) & 
                    (df['latitude'] <= bounds['lat_max']) &
                    (df['longitude'] >= bounds['lon_min']) & 
                    (df['longitude'] <= bounds['lon_max'])
                ]
                
                if len(region_data) > 0:
                    annual_means = region_data.groupby('year')[var].mean()
                    
                    # Smoothing
                    from scipy.ndimage import gaussian_filter1d
                    smoothed = gaussian_filter1d(annual_means.values, sigma=2)
                    
                    ax.plot(annual_means.index, smoothed, '-', linewidth=3,
                           color=colors[region_idx], label=region_name)
            
            ax.set_title(f'Regional {var.replace("predicted_", "").replace("_", " ").title()} Comparison',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel(var.replace('predicted_', '').replace('_', ' ').title(), fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Regional Zero-Curtain Dynamics Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = output_dir / 'regional_analysis' / 'regional_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f" Regional analysis saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f" Error in regional analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================

def generate_statistical_summary(predictions_file, output_dir):
    """Generate comprehensive statistical summary"""
    print("\n" + "="*80)
    print("GENERATING STATISTICAL SUMMARY")
    print("="*80)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Load predictions
        df = pd.read_parquet(predictions_file)
        df['datetime'] = pd.to_datetime(df['start_time'])
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        
        variables = ['predicted_intensity_percentile', 'predicted_duration_hours',
                    'predicted_spatial_extent_meters']
        
        # Overall statistics
        stats_summary = []
        
        for var in variables:
            var_data = df[var].dropna()
            
            stats = {
                'Variable': var.replace('predicted_', ''),
                'Count': len(var_data),
                'Mean': var_data.mean(),
                'Median': var_data.median(),
                'Std': var_data.std(),
                'Min': var_data.min(),
                'Max': var_data.max(),
                'Q25': var_data.quantile(0.25),
                'Q75': var_data.quantile(0.75)
            }
            
            stats_summary.append(stats)
        
        stats_df = pd.DataFrame(stats_summary)
        
        # Save to CSV
        output_path = output_dir / 'statistics' / 'overall_statistics.csv'
        stats_df.to_csv(output_path, index=False)
        print(f" Overall statistics saved: {output_path}")
        
        # Temporal statistics
        temporal_stats = df.groupby('year')[variables].agg(['mean', 'std', 'count'])
        
        output_path = output_dir / 'statistics' / 'temporal_statistics.csv'
        temporal_stats.to_csv(output_path)
        print(f" Temporal statistics saved: {output_path}")
        
        # Spatial statistics by latitude bands
        df['lat_band'] = pd.cut(df['latitude'], bins=np.arange(49, 91, 5))
        spatial_stats = df.groupby('lat_band')[variables].agg(['mean', 'std', 'count'])
        
        output_path = output_dir / 'statistics' / 'spatial_statistics.csv'
        spatial_stats.to_csv(output_path)
        print(f" Spatial statistics saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f" Error generating statistics: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute Part V mapping and visualization pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Part V: Mapping and Visualization')
    parser.add_argument('--base-dir', type=str, default=None,
                       help='Base pipeline directory')
    parser.add_argument('--skip-mapping', action='store_true',
                       help='Skip map generation (statistics only)')
    parser.add_argument('--skip-time-series', action='store_true',
                       help='Skip time series analysis')
    parser.add_argument('--skip-regional', action='store_true',
                       help='Skip regional analysis')
    
    args = parser.parse_args()
    
    # Set base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path.home() / "arctic_zero_curtain_pipeline"
    
    print(f"Base directory: {base_dir}")
    print()
    
    # Step 1: Validate environment
    if not validate_environment():
        print("\n Environment validation failed")
        return 1
    
    # Step 2: Validate Part IV outputs
    valid, predictions_file = validate_part4_outputs(base_dir)
    if not valid:
        print("\n Part IV output validation failed")
        return 1
    
    # Step 3: Setup Natural Earth data
    ne_valid, ne_dir = setup_natural_earth_data(base_dir)
    if not ne_valid:
        print("\n Natural Earth setup failed")
        return 1
    
    # Step 4: Setup output directories
    output_dir = setup_output_directories(base_dir)
    
    # Step 5: Execute mapping pipeline
    if not args.skip_mapping:
        if not integrate_mapping_pipeline(predictions_file, ne_dir, output_dir):
            print("\n Mapping pipeline failed")
            return 1
    else:
        print("\nSkipping mapping pipeline (--skip-mapping)")
    
    # Step 6: Generate time series analysis
    if not args.skip_time_series:
        if not generate_time_series_analysis(predictions_file, output_dir):
            print("\n Time series analysis failed")
            return 1
    else:
        print("\nSkipping time series analysis (--skip-time-series)")
    
    # Step 7: Generate regional analysis
    if not args.skip_regional:
        if not generate_regional_analysis(predictions_file, output_dir):
            print("\n Regional analysis failed")
            return 1
    else:
        print("\nSkipping regional analysis (--skip-regional)")
    
    # Step 8: Generate statistical summary
    if not generate_statistical_summary(predictions_file, output_dir):
        print("\n Statistical summary generation failed")
        return 1
    
    # Summary
    print("\n" + "="*80)
    print("PART V COMPLETE")
    print("="*80)
    print(f" All visualization and analysis components generated")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated outputs:")
    print(f"  - Circumarctic maps (annual, seasonal, monthly)")
    print(f"  - Multi-decadal time series analysis")
    print(f"  - Regional comparison visualizations")
    print(f"  - Comprehensive statistical summaries")
    print()
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
