#!/usr/bin/env python3
"""Simple test for zero-curtain detection using available data."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.part1_physics_detection.detection_config import DetectionConfiguration
from src.part1_physics_detection.zero_curtain_detector import PhysicsInformedZeroCurtainDetector

def find_input_file():
    """Find any available input file."""
    
    # Check for teacher forcing files first
    tf_files = [
        "outputs/part2_geocryoai/teacher_forcing_in_situ_database_train.parquet",
        "outputs/part2_geocryoai/teacher_forcing_in_situ_database_val.parquet",
        "outputs/part2_geocryoai/teacher_forcing_in_situ_database_test.parquet"
    ]
    
    for tf_file in tf_files:
        if Path(tf_file).exists():
            return tf_file
    
    # Check for other parquet files in outputs
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        parquet_files = list(outputs_dir.glob("*.parquet"))
        if parquet_files:
            # Return the largest file
            largest = max(parquet_files, key=lambda x: x.stat().st_size)
            return str(largest)
    
    return None

def main():
    print("="*80)
    print("ZERO-CURTAIN DETECTION - SIMPLE TEST")
    print("="*80)
    
    # Find input file
    input_file = find_input_file()
    
    if input_file is None:
        print("\n ERROR: No input data files found!")
        print("\nSearching for ANY parquet files in project...")
        
        all_parquet = list(Path(".").rglob("*.parquet"))
        if all_parquet:
            print(f"\nFound {len(all_parquet)} parquet files:")
            for f in all_parquet[:10]:
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"  {f} ({size_mb:.1f} MB)")
            
            print("\nPlease specify which file to use.")
        else:
            print("No parquet files found in project directory!")
            print("\nDo you need to run the data preparation pipeline first?")
        
        return 1
    
    print(f"\n Found input file: {input_file}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(input_file)
    print(f" Loaded {len(df):,} measurements")
    print(f"  Columns: {list(df.columns)}")
    
    # Check required columns
    required_cols = ['datetime', 'latitude', 'longitude', 'soil_temp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return 1
    
    # Add depth zone if missing
    if 'soil_temp_depth_zone' not in df.columns:
        print("\n Adding default depth zone (intermediate)...")
        df['soil_temp_depth_zone'] = 'intermediate'
    
    # Add source if missing
    if 'source' not in df.columns:
        print(" Adding default source...")
        df['source'] = 'unknown'
    
    # Filter for cold season
    print("\nFiltering for cold season...")
    cold_months = [9, 10, 11, 12, 1, 2, 3, 4, 5]
    df_cold = df[df['datetime'].dt.month.isin(cold_months)].copy()
    print(f" Cold season data: {len(df_cold):,} measurements")
    
    # Sample for testing
    test_size = min(10000, len(df_cold))
    print(f"\nCreating test subset ({test_size:,} measurements)...")
    df_test = df_cold.sample(n=test_size, random_state=42)
    
    # Initialize detector
    print("\nInitializing detector...")
    config = DetectionConfiguration()
    detector = PhysicsInformedZeroCurtainDetector(config)
    print(" Detector initialized")
    
    # Get sites
    print("\nIdentifying sites...")
    sites = df_test.groupby(['latitude', 'longitude', 'source']).size().reset_index(name='n_measurements')
    sites = sites[sites['n_measurements'] >= 20]  # Lowered threshold for testing
    print(f" Found {len(sites)} sites with ≥20 measurements")
    
    if len(sites) == 0:
        print("\n No sites with sufficient data")
        print("Trying with ANY available data points...")
        sites = df_test.groupby(['latitude', 'longitude', 'source']).size().reset_index(name='n_measurements')
        sites = sites.head(5)
        print(f"Testing on {len(sites)} sites regardless of data count")
    
    # Test detection
    print(f"\nTesting detection on up to 5 sites...")
    all_events = []
    
    for idx, site in sites.head(5).iterrows():
        lat, lon, source = site['latitude'], site['longitude'], site['source']
        print(f"\n  Site {idx+1}: {lat:.3f}°N, {lon:.3f}°E ({source})")
        
        # Check permafrost
        pf_props = detector.get_site_permafrost_properties(lat, lon)
        print(f"    Permafrost: {pf_props['is_permafrost_suitable']} - {pf_props['suitability_reason']}")
        
        if not pf_props['is_permafrost_suitable']:
            print("    ⊘ Skipping - not suitable for permafrost analysis")
            continue
        
        # Get site data
        site_data = df_test[
            (df_test['latitude'] == lat) &
            (df_test['longitude'] == lon) &
            (df_test['source'] == source)
        ].copy()
        
        print(f"    Data points: {len(site_data)}")
        print(f"    Temp range: [{site_data['soil_temp'].min():.1f}, {site_data['soil_temp'].max():.1f}]°C")
        
        # Detect events
        try:
            events = detector.detect_zero_curtain_with_physics(site_data, lat, lon)
            print(f"     Detected {len(events)} zero-curtain events")
            
            for event in events:
                event['latitude'] = lat
                event['longitude'] = lon
                event['source'] = source
            
            all_events.extend(events)
            
        except Exception as e:
            print(f"     Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Sites tested: {min(5, len(sites))}")
    print(f"Total events detected: {len(all_events)}")
    
    if len(all_events) > 0:
        events_df = pd.DataFrame(all_events)
        
        print(f"\nEvent Statistics:")
        print(f"  Mean intensity: {events_df['intensity_percentile'].mean():.3f}")
        print(f"  Mean duration: {events_df['duration_hours'].mean():.1f} hours")
        print(f"  Mean spatial extent: {events_df['spatial_extent_meters'].mean():.2f} meters")
        
        print(f"\nDepth zones: {events_df['depth_zone'].value_counts().to_dict()}")
        
        # Save results
        output_path = Path("outputs/test_detection_results.parquet")
        output_path.parent.mkdir(exist_ok=True)
        events_df.to_parquet(output_path, compression='snappy', index=False)
        print(f"\n Results saved to: {output_path}")
        
        print("\n TEST SUCCESSFUL!")
        return 0
    else:
        print("\n No events detected in test")
        print("\nThis could mean:")
        print("  1. No permafrost-suitable sites in test subset")
        print("  2. No clear zero-curtain signatures in data")
        print("  3. Need to adjust detection thresholds")
        
        print("\nNext steps:")
        print("  - Try with full dataset instead of subset")
        print("  - Check data quality and temperature ranges")
        print("  - Review detection configuration")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())
