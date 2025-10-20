#!/usr/bin/env python3
"""Test detection with your actual teacher forcing data."""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, '.')

from src.part1_physics_detection.detection_config import DetectionConfiguration
from src.part1_physics_detection.zero_curtain_detector import PhysicsInformedZeroCurtainDetector

print("="*70)
print("ZERO-CURTAIN DETECTION TEST - Real Data")
print("="*70)

# Load your training data
print("\n1. Loading data...")
df = pd.read_parquet('outputs/part2_geocryoai/teacher_forcing_in_situ_database_train.parquet')
print(f"    Loaded {len(df):,} measurements")
print(f"   Columns: {list(df.columns)[:10]}...")

# Sample for quick test
print("\n2. Sampling data for test...")
df_sample = df.sample(n=min(50000, len(df)), random_state=42)
print(f"    Sample: {len(df_sample):,} measurements")

# Filter cold season
print("\n3. Filtering cold season...")
cold_months = [9, 10, 11, 12, 1, 2, 3, 4, 5]
df_cold = df_sample[df_sample['datetime'].dt.month.isin(cold_months)]
print(f"    Cold season: {len(df_cold):,} measurements")

# Get sites
print("\n4. Identifying sites...")
sites = df_cold.groupby(['latitude', 'longitude', 'source']).size().reset_index(name='n')
sites = sites[sites['n'] >= 50].head(10)
print(f"    Testing {len(sites)} sites")

# Initialize detector
print("\n5. Initializing detector...")
config = DetectionConfiguration()
detector = PhysicsInformedZeroCurtainDetector(config)

# Run detection
print("\n6. Running detection...")
all_events = []

for idx, site in sites.iterrows():
    lat, lon, source = site['latitude'], site['longitude'], site['source']
    
    pf = detector.get_site_permafrost_properties(lat, lon)
    
    if not pf['is_permafrost_suitable']:
        continue
    
    site_data = df_cold[
        (df_cold['latitude'] == lat) &
        (df_cold['longitude'] == lon) &
        (df_cold['source'] == source)
    ]
    
    try:
        events = detector.detect_zero_curtain_with_physics(site_data, lat, lon)
        for e in events:
            e['latitude'] = lat
            e['longitude'] = lon
            e['source'] = source
        all_events.extend(events)
        
        if events:
            print(f"   Site {lat:.2f}, {lon:.2f}: {len(events)} events")
    except Exception as ex:
        print(f"   Error at {lat:.2f}, {lon:.2f}: {ex}")

print(f"\n7. Results:")
print(f"   Total events: {len(all_events)}")

if all_events:
    events_df = pd.DataFrame(all_events)
    
    print(f"   Mean duration: {events_df['duration_hours'].mean():.1f} hours")
    print(f"   Mean temp: {events_df['mean_temperature'].mean():.2f}°C")
    
    # Save
    output = Path('outputs/pinszc_test_real_data.parquet')
    events_df.to_parquet(output, index=False)
    print(f"\n    Saved to: {output}")
    print("\n TEST SUCCESSFUL!")
else:
    print("\n    No events detected")
    print("   This may be normal - try adjusting detection parameters")

print("\n" + "="*70)
