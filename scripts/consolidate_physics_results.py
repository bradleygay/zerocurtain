#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def consolidate_emergency_saves(outputs_dir):
    """
    Consolidate all emergency save parquet files into single comprehensive dataset.
    Applies to ENTIRE dataset - NO SAMPLING.
    """
    print("="*90)
    print("CONSOLIDATING PHYSICS-INFORMED ZERO-CURTAIN DETECTION RESULTS")
    print("="*90)
    
    outputs_path = Path(outputs_dir)
    
    emergency_files = list(outputs_path.glob("zero_curtain_EMERGENCY_site_*.parquet"))
    
    if not emergency_files:
        print("No emergency save files found")
        return None
    
    print(f"\nFound {len(emergency_files)} emergency save files")
    
    all_events = []
    total_events = 0
    
    for i, file_path in enumerate(sorted(emergency_files), 1):
        try:
            df = pd.read_parquet(file_path)
            events_count = len(df)
            total_events += events_count
            all_events.append(df)
            
            print(f"  [{i}/{len(emergency_files)}] {file_path.name}: {events_count:,} events")
            
        except Exception as e:
            print(f"  Error reading {file_path.name}: {e}")
            continue
    
    if not all_events:
        print("No events loaded successfully")
        return None
    
    print(f"\nConcatenating {len(all_events)} dataframes with {total_events:,} total events...")
    consolidated_df = pd.concat(all_events, ignore_index=True)
    
    print(f"✅ Consolidated dataset: {len(consolidated_df):,} events")
    
    return consolidated_df


def verify_required_features(df):
    """
    Verify all three main features plus CryoGrid features are present.
    """
    print("\n" + "="*90)
    print("FEATURE VERIFICATION")
    print("="*90)
    
    main_features = ['intensity_percentile', 'duration_hours', 'spatial_extent_meters']
    
    print("\nMain Features (REQUIRED):")
    all_present = True
    for feature in main_features:
        if feature in df.columns:
            non_null = df[feature].notna().sum()
            pct = (non_null / len(df)) * 100
            print(f"  ✅ {feature}: {non_null:,}/{len(df):,} ({pct:.1f}% non-null)")
        else:
            print(f"  ❌ {feature}: MISSING")
            all_present = False
    
    if not all_present:
        raise ValueError("Missing required main features!")
    
    cryogrid_features = [
        'cryogrid_thermal_conductivity',
        'cryogrid_heat_capacity',
        'cryogrid_enthalpy_stability',
        'phase_change_energy',
        'freeze_penetration_depth',
        'thermal_diffusivity',
        'snow_insulation_factor'
    ]
    
    print("\nCryoGrid Physics Features:")
    for feature in cryogrid_features:
        if feature in df.columns:
            non_null = df[feature].notna().sum()
            pct = (non_null / len(df)) * 100
            print(f"  ✅ {feature}: {non_null:,}/{len(df):,} ({pct:.1f}% non-null)")
        else:
            print(f"  ⚠️  {feature}: not present")
    
    print("\nSpatiotemporal Features:")
    spatial_features = ['latitude', 'longitude', 'depth_zone', 'permafrost_zone', 'permafrost_probability']
    for feature in spatial_features:
        if feature in df.columns:
            non_null = df[feature].notna().sum()
            print(f"  ✅ {feature}: {non_null:,}/{len(df):,} non-null")
    
    temporal_features = ['start_time', 'end_time']
    for feature in temporal_features:
        if feature in df.columns:
            non_null = df[feature].notna().sum()
            print(f"  ✅ {feature}: {non_null:,}/{len(df):,} non-null")
    
    print("\n✅ Feature verification complete")
    return True


def add_derived_features(df):
    """
    Add derived features for enhanced analysis and ML preparation.
    """
    print("\n" + "="*90)
    print("ADDING DERIVED FEATURES")
    print("="*90)
    
    if 'start_time' in df.columns and 'end_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        df['year'] = df['start_time'].dt.year
        df['month'] = df['start_time'].dt.month
        df['day_of_year'] = df['start_time'].dt.dayofyear
        
        def get_season(month):
            if month in [12, 1, 2]: return 'winter'
            elif month in [3, 4, 5]: return 'spring'
            elif month in [6, 7, 8]: return 'summer'
            else: return 'fall'
        
        df['season'] = df['month'].apply(get_season)
        
        print("  ✅ Temporal features: year, month, day_of_year, season")
    
    if 'intensity_percentile' in df.columns:
        df['intensity_category'] = pd.cut(
            df['intensity_percentile'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['weak', 'moderate', 'strong', 'extreme']
        )
        print("  ✅ Intensity category")
    
    if 'duration_hours' in df.columns:
        df['duration_category'] = pd.cut(
            df['duration_hours'],
            bins=[0, 72, 168, 336, np.inf],
            labels=['short', 'medium', 'long', 'extended']
        )
        df['duration_days'] = df['duration_hours'] / 24.0
        print("  ✅ Duration category and duration_days")
    
    if 'spatial_extent_meters' in df.columns:
        df['extent_category'] = pd.cut(
            df['spatial_extent_meters'],
            bins=[0, 0.3, 0.8, 1.5, np.inf],
            labels=['shallow', 'moderate', 'deep', 'very_deep']
        )
        print("  ✅ Spatial extent category")
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['lat_zone'] = pd.cut(
            df['latitude'],
            bins=[50, 60, 70, 80, 90],
            labels=['subarctic', 'low_arctic', 'mid_arctic', 'high_arctic']
        )
        print("  ✅ Latitude zone")
    
    if 'intensity_percentile' in df.columns and 'duration_hours' in df.columns and 'spatial_extent_meters' in df.columns:
        df['composite_severity'] = (
            0.4 * df['intensity_percentile'] +
            0.3 * (df['duration_hours'] / df['duration_hours'].max()) +
            0.3 * (df['spatial_extent_meters'] / df['spatial_extent_meters'].max())
        )
        print("  ✅ Composite severity score")
    
    if 'phase_change_energy' in df.columns and df['phase_change_energy'].notna().any():
        df['energy_intensity'] = np.log1p(df['phase_change_energy'].abs())
        print("  ✅ Log-transformed energy intensity")
    
    print(f"\n✅ Derived features added. Total columns: {len(df.columns)}")
    
    return df


def generate_comprehensive_statistics(df, output_dir):
    """
    Generate comprehensive statistical summary of ENTIRE dataset.
    """
    print("\n" + "="*90)
    print("COMPREHENSIVE DATASET STATISTICS")
    print("="*90)
    
    stats = {
        'dataset_info': {
            'total_events': int(len(df)),
            'total_sites': int(df.groupby(['latitude', 'longitude']).ngroups) if 'latitude' in df.columns else 0,
            'total_sources': int(df['source'].nunique()) if 'source' in df.columns else 0,
            'generation_timestamp': datetime.now().isoformat()
        },
        'spatial_coverage': {},
        'temporal_coverage': {},
        'main_features': {},
        'physics_features': {},
        'categorical_distributions': {}
    }
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        stats['spatial_coverage'] = {
            'latitude_range': [float(df['latitude'].min()), float(df['latitude'].max())],
            'longitude_range': [float(df['longitude'].min()), float(df['longitude'].max())],
            'unique_locations': int(df.groupby(['latitude', 'longitude']).ngroups)
        }
        print(f"\nSpatial Coverage:")
        print(f"  Latitude range: {stats['spatial_coverage']['latitude_range'][0]:.2f}° to {stats['spatial_coverage']['latitude_range'][1]:.2f}°N")
        print(f"  Longitude range: {stats['spatial_coverage']['longitude_range'][0]:.2f}° to {stats['spatial_coverage']['longitude_range'][1]:.2f}°E")
        print(f"  Unique locations: {stats['spatial_coverage']['unique_locations']:,}")
    
    if 'start_time' in df.columns and 'end_time' in df.columns:
        stats['temporal_coverage'] = {
            'earliest_event': df['start_time'].min().isoformat(),
            'latest_event': df['end_time'].max().isoformat(),
            'time_span_years': float((df['end_time'].max() - df['start_time'].min()).days / 365.25)
        }
        print(f"\nTemporal Coverage:")
        print(f"  Earliest event: {df['start_time'].min()}")
        print(f"  Latest event: {df['end_time'].max()}")
        print(f"  Time span: {stats['temporal_coverage']['time_span_years']:.1f} years")
    
    main_features = ['intensity_percentile', 'duration_hours', 'spatial_extent_meters']
    print(f"\nMain Feature Statistics:")
    for feature in main_features:
        if feature in df.columns:
            stats['main_features'][feature] = {
                'mean': float(df[feature].mean()),
                'std': float(df[feature].std()),
                'min': float(df[feature].min()),
                'max': float(df[feature].max()),
                'median': float(df[feature].median()),
                'q25': float(df[feature].quantile(0.25)),
                'q75': float(df[feature].quantile(0.75))
            }
            print(f"  {feature}:")
            print(f"    Mean: {stats['main_features'][feature]['mean']:.3f}")
            print(f"    Std:  {stats['main_features'][feature]['std']:.3f}")
            print(f"    Range: [{stats['main_features'][feature]['min']:.3f}, {stats['main_features'][feature]['max']:.3f}]")
    
    physics_features = ['phase_change_energy', 'freeze_penetration_depth', 'thermal_diffusivity', 'snow_insulation_factor']
    print(f"\nPhysics Feature Statistics:")
    for feature in physics_features:
        if feature in df.columns and df[feature].notna().any():
            stats['physics_features'][feature] = {
                'mean': float(df[feature].mean()),
                'std': float(df[feature].std()),
                'non_null_count': int(df[feature].notna().sum())
            }
            print(f"  {feature}:")
            print(f"    Mean: {stats['physics_features'][feature]['mean']:.3e}")
            print(f"    Non-null: {stats['physics_features'][feature]['non_null_count']:,}")
    
    categorical_features = ['depth_zone', 'permafrost_zone', 'intensity_category', 'duration_category', 'extent_category', 'season']
    print(f"\nCategorical Distributions:")
    for feature in categorical_features:
        if feature in df.columns:
            distribution = df[feature].value_counts().to_dict()
            stats['categorical_distributions'][feature] = {k: int(v) for k, v in distribution.items()}
            print(f"  {feature}:")
            for cat, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(df)) * 100
                print(f"    {cat}: {count:,} ({pct:.1f}%)")
    
    stats_file = Path(output_dir) / 'consolidated_physics_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Statistics saved to: {stats_file}")
    
    return stats


def prepare_stratified_splits(df, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Prepare stratified train/validation/test splits for GeoCryoAI.
    This is the ONLY acceptable use of sampling per Dr. Gay's instructions.
    """
    print("\n" + "="*90)
    print("PREPARING STRATIFIED TRAIN/VALIDATION/TEST SPLITS FOR GEOCRYOAI")
    print("="*90)
    
    from sklearn.model_selection import train_test_split
    
    print(f"\nSplit ratios:")
    print(f"  Training:   {train_ratio*100:.0f}%")
    print(f"  Validation: {val_ratio*100:.0f}%")
    print(f"  Testing:    {test_ratio*100:.0f}%")
    
    stratify_column = None
    if 'intensity_category' in df.columns:
        stratify_column = 'intensity_category'
        print(f"\nStratifying by: {stratify_column}")
    elif 'depth_zone' in df.columns:
        stratify_column = 'depth_zone'
        print(f"\nStratifying by: {stratify_column}")
    
    if stratify_column:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            stratify=df[stratify_column],
            random_state=42
        )
        
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            stratify=train_val_df[stratify_column],
            random_state=42
        )
    else:
        print("\nNo stratification column available - using random split")
        train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=42)
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(train_val_df, test_size=val_ratio_adjusted, random_state=42)
    
    print(f"\nSplit sizes:")
    print(f"  Training:   {len(train_df):,} events ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df):,} events ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Testing:    {len(test_df):,} events ({len(test_df)/len(df)*100:.1f}%)")
    
    output_path = Path(output_dir)
    
    train_file = output_path / 'physics_informed_events_train.parquet'
    val_file = output_path / 'physics_informed_events_val.parquet'
    test_file = output_path / 'physics_informed_events_test.parquet'
    
    train_df.to_parquet(train_file, index=False, compression='snappy')
    val_df.to_parquet(val_file, index=False, compression='snappy')
    test_df.to_parquet(test_file, index=False, compression='snappy')
    
    print(f"\n✅ Train set saved: {train_file}")
    print(f"✅ Validation set saved: {val_file}")
    print(f"✅ Test set saved: {test_file}")
    
    return train_df, val_df, test_df


def main():
    """
    Main consolidation workflow - operates on ENTIRE dataset.
    """
    outputs_dir = "/Users/bagay/arctic_zero_curtain_pipeline/outputs"
    
    print("\nStep 1: Consolidating all emergency save files...")
    consolidated_df = consolidate_emergency_saves(outputs_dir)
    
    if consolidated_df is None:
        print("Consolidation failed")
        return 1
    
    print("\nStep 2: Verifying required features...")
    verify_required_features(consolidated_df)
    
    print("\nStep 3: Adding derived features...")
    enhanced_df = add_derived_features(consolidated_df)
    
    print("\nStep 4: Generating comprehensive statistics...")
    stats = generate_comprehensive_statistics(enhanced_df, outputs_dir)
    
    print("\nStep 5: Saving consolidated dataset...")
    consolidated_file = Path(outputs_dir) / 'physics_informed_zero_curtain_events_COMPLETE.parquet'
    enhanced_df.to_parquet(consolidated_file, index=False, compression='snappy')
    print(f"✅ Complete dataset saved: {consolidated_file}")
    print(f"   Size: {consolidated_file.stat().st_size / (1024**2):.1f} MB")
    
    print("\nStep 6: Preparing stratified train/val/test splits...")
    train_df, val_df, test_df = prepare_stratified_splits(enhanced_df, outputs_dir)
    
    print("\n" + "="*90)
    print("CONSOLIDATION COMPLETE")
    print("="*90)
    print(f"Total events processed: {len(enhanced_df):,}")
    print(f"Total features: {len(enhanced_df.columns)}")
    print(f"\nOutput files:")
    print(f"  Complete dataset: physics_informed_zero_curtain_events_COMPLETE.parquet")
    print(f"  Training set:     physics_informed_events_train.parquet")
    print(f"  Validation set:   physics_informed_events_val.parquet")
    print(f"  Testing set:      physics_informed_events_test.parquet")
    print(f"  Statistics:       consolidated_physics_statistics.json")
    print("="*90)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())