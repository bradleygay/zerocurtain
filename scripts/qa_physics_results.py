#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def quality_assurance_checks(parquet_file):
    """
    Comprehensive quality assurance on consolidated dataset.
    """
    print("="*90)
    print("QUALITY ASSURANCE - PHYSICS-INFORMED ZERO-CURTAIN EVENTS")
    print("="*90)
    
    df = pd.read_parquet(parquet_file)
    
    print(f"\nDataset loaded: {len(df):,} events, {len(df.columns)} features")
    
    issues = []
    
    print("\n1. Checking for duplicates...")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"  ⚠️  Found {duplicates} duplicate rows")
        issues.append(f"Duplicates: {duplicates}")
    else:
        print(f"  ✅ No duplicates found")
    
    print("\n2. Checking main features for invalid values...")
    main_features = ['intensity_percentile', 'duration_hours', 'spatial_extent_meters']
    
    for feature in main_features:
        if feature not in df.columns:
            print(f"  ❌ {feature}: MISSING")
            issues.append(f"{feature} missing")
            continue
        
        null_count = df[feature].isna().sum()
        if null_count > 0:
            print(f"  ⚠️  {feature}: {null_count} null values")
            issues.append(f"{feature}: {null_count} nulls")
        
        if feature == 'intensity_percentile':
            out_of_range = ((df[feature] < 0) | (df[feature] > 1)).sum()
            if out_of_range > 0:
                print(f"  ⚠️  {feature}: {out_of_range} values outside [0,1]")
                issues.append(f"{feature}: {out_of_range} out of range")
            else:
                print(f"  ✅ {feature}: all values in [0,1]")
        
        if feature == 'duration_hours':
            negative = (df[feature] < 0).sum()
            if negative > 0:
                print(f"  ⚠️  {feature}: {negative} negative values")
                issues.append(f"{feature}: {negative} negative")
            else:
                print(f"  ✅ {feature}: all positive values")
        
        if feature == 'spatial_extent_meters':
            negative = (df[feature] < 0).sum()
            if negative > 0:
                print(f"  ⚠️  {feature}: {negative} negative values")
                issues.append(f"{feature}: {negative} negative")
            else:
                print(f"  ✅ {feature}: all positive values")
    
    print("\n3. Checking spatial coordinates...")
    if 'latitude' in df.columns:
        invalid_lat = ((df['latitude'] < -90) | (df['latitude'] > 90)).sum()
        if invalid_lat > 0:
            print(f"  ⚠️  {invalid_lat} invalid latitude values")
            issues.append(f"Invalid latitude: {invalid_lat}")
        else:
            print(f"  ✅ All latitudes valid")
    
    if 'longitude' in df.columns:
        invalid_lon = ((df['longitude'] < -180) | (df['longitude'] > 180)).sum()
        if invalid_lon > 0:
            print(f"  ⚠️  {invalid_lon} invalid longitude values")
            issues.append(f"Invalid longitude: {invalid_lon}")
        else:
            print(f"  ✅ All longitudes valid")
    
    print("\n4. Checking temporal consistency...")
    if 'start_time' in df.columns and 'end_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        invalid_times = (df['end_time'] < df['start_time']).sum()
        if invalid_times > 0:
            print(f"  ⚠️  {invalid_times} events with end_time < start_time")
            issues.append(f"Invalid time order: {invalid_times}")
        else:
            print(f"  ✅ All event times consistent")
    
    print("\n5. Summary:")
    if not issues:
        print("  ✅ All quality checks passed!")
    else:
        print(f"  ⚠️  Found {len(issues)} issues:")
        for issue in issues:
            print(f"    - {issue}")
    
    print("\n" + "="*90)
    
    return len(issues) == 0


def main():
    outputs_dir = Path("/Users/bagay/arctic_zero_curtain_pipeline/outputs")
    consolidated_file = outputs_dir / 'physics_informed_zero_curtain_events_COMPLETE.parquet'
    
    if not consolidated_file.exists():
        print(f"Consolidated file not found: {consolidated_file}")
        print("Run consolidate_physics_results.py first")
        return 1
    
    passed = quality_assurance_checks(consolidated_file)
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())