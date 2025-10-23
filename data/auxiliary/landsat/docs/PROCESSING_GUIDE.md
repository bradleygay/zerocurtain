# Landsat Processing Guide

Detailed step-by-step guide for processing Landsat 8/9 data.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Configuration](#configuration)
3. [Initial Download](#initial-download)
4. [Gap Filling](#gap-filling)
5. [Swath Gap Filling](#swath-gap-filling)
6. [Dataset Combination](#dataset-combination)
7. [Validation](#validation)
8. [Advanced Topics](#advanced-topics)

---

## Environment Setup

### System Requirements

- **OS**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.8+
- **RAM**: 32 GB minimum, 64 GB recommended
- **Storage**: 500 GB free space
- **CPU**: 8+ cores recommended

### Python Environment
```bash
# Create conda environment
conda create -n landsat_processing python=3.9
conda activate landsat_processing

# Install packages
pip install earthengine-api==0.1.374 \
            pandas==2.0.3 \
            numpy==1.24.3 \
            geopandas==0.14.0 \
            matplotlib==3.7.2 \
            scikit-learn==1.3.0 \
            scipy==1.11.2 \
            pyyaml==6.0.1 \
            rasterio==1.3.8 \
            pyarrow==12.0.1 \
            psutil==5.9.5
```

### Earth Engine Setup
```bash
# Initialize Earth Engine
earthengine authenticate

# Set project (replace with your project ID)
export EE_PROJECT=circumarcticzerocurtain

# Test connection
python -c "import ee; ee.Initialize(project='$EE_PROJECT'); print('EE initialized')"
```

---

## Configuration

### Review Configuration Files
```bash
cd ~/arctic_zero_curtain_pipeline/data/auxiliary/landsat

# Main configuration
cat config/landsat_config.yaml

# Processing parameters
cat config/processing_params.yaml
```

### Customize for Your Needs

#### Adjust Date Range
```yaml
# config/landsat_config.yaml
temporal:
  start_date: "2020-01-01"  # Your start date
  end_date: "2023-12-31"    # Your end date
```

#### Adjust Cloud Threshold
```yaml
# config/landsat_config.yaml
quality:
  cloud_cover:
    max_percentage: 15.0  # More lenient for Arctic
```

#### Adjust System Resources
```yaml
# config/landsat_config.yaml
processing:
  system:
    max_concurrent_processes: 4  # Reduce for limited resources
    max_threads_per_process: 4
```

---

## Initial Download

### Step 1: Dry Run Test
```bash
# Test with small region first
python scripts/landsat_test.py

# Expected output:
#  All imports successful
#  Directory structure validated
#  Sample data generated: 1,000 points
#  Coverage analysis: XX.XX%
#  Data combination: XXX points
#  ALL TESTS PASSED
```

### Step 2: Start Download
```bash
# Full download with logging
python scripts/landsat_downloader.py \
    --config config/landsat_config.yaml \
    --output-dir $(pwd) \
    --max-processes 8 \
    2>&1 | tee logs/download_$(date +%Y%m%d).log
```

### Step 3: Monitor Progress

In a separate terminal:
```bash
# Watch log file
tail -f logs/download_*.log

# Check processing statistics
python scripts/utils/monitor_progress.py --log logs/download_*.log
```

### Expected Output Structure
```
raw/
 landsat_arctic_data.parquet     # Main output
 processing_summary.json         # Statistics

checkpoints/
 region_-180_45_checkpoint.json
 region_-180_50_checkpoint.json
 ...

temp_csv/
 region_-180_45.csv
 region_-180_50.csv
 ...
```

### Checkpoint Management
```bash
# View completed regions
python -c "
import json
with open('checkpoints/completed_regions.json') as f:
    data = json.load(f)
    print(f'Completed: {len(data[\"completed_regions\"])} regions')
"

# Resume from checkpoint
python scripts/landsat_downloader.py --config config/landsat_config.yaml
```

---

## Gap Filling

### Step 1: Analyze Coverage
```bash
python scripts/landsat_gap_filler.py \
    --input-file raw/landsat_arctic_data.parquet \
    --output-dir gaps_filled \
    --analyze-only

# Check analysis output
cat gaps_filled/analysis/coverage_analysis.json
```

### Step 2: Review Gap Regions
```python
import json

with open('gaps_filled/analysis/coverage_analysis.json') as f:
    analysis = json.load(f)

print(f"Coverage: {analysis['coverage_percentage']:.2f}%")
print(f"Gap cells: {len(analysis['gap_cells'])}")
print(f"Boundary issues: {analysis['boundary_issues']}")
```

### Step 3: Fill Gaps
```bash
python scripts/landsat_gap_filler.py \
    --input-file raw/landsat_arctic_data.parquet \
    --output-dir gaps_filled \
    --start-date 2015-03-30 \
    --end-date 2024-12-31 \
    --cloud-threshold 10 \
    --max-processes 8 \
    2>&1 | tee logs/gap_filling_$(date +%Y%m%d).log
```

### Step 4: Verify Results
```bash
# Check output
ls -lh gaps_filled/raw/landsat_gaps_filled.parquet

# Validate
python -c "
import pandas as pd
df = pd.read_parquet('gaps_filled/raw/landsat_gaps_filled.parquet')
print(f'Gap-filled points: {len(df):,}')
print(f'Unique scenes: {df[\"scene_id\"].nunique():,}')
"
```

---

## Swath Gap Filling

### Step 1: Identify Swath Gaps
```bash
python scripts/landsat_swath_gap_filler.py \
    --input-file raw/landsat_arctic_data.parquet \
    --output-dir swath_gaps_filled \
    --min-gap-width 0.1 \
    --max-gap-width 2.0 \
    --identify-only
```

### Step 2: Review Swath Gaps
```bash
# Visualize identified gaps
python scripts/landsat_visualizer.py \
    --gap-file swath_gaps_filled/analysis/swath_gaps.json \
    --output swath_gaps_filled/analysis/swath_gaps_map.png
```

### Step 3: Fill Swath Gaps
```bash
python scripts/landsat_swath_gap_filler.py \
    --input-file raw/landsat_arctic_data.parquet \
    --output-dir swath_gaps_filled \
    --start-date 2015-03-30 \
    --end-date 2024-12-31 \
    --cloud-threshold 10 \
    --max-latitude 84.0 \
    --max-processes 8 \
    2>&1 | tee logs/swath_gap_filling_$(date +%Y%m%d).log
```

### Step 4: Quality Check
```bash
# Verify output
python -c "
import pandas as pd
df = pd.read_parquet('swath_gaps_filled/landsat_arctic_swath_gaps_data.parquet')
print(f'Swath gap-filled points: {len(df):,}')
print(f'Spatial extent: Lon [{df[\"longitude\"].min():.2f}, {df[\"longitude\"].max():.2f}]')
print(f'                Lat [{df[\"latitude\"].min():.2f}, {df[\"latitude\"].max():.2f}]')
"
```

---

## Dataset Combination

### Step 1: Prepare for Merging
```bash
# Verify all input files exist
for file in \
    raw/landsat_arctic_data.parquet \
    gaps_filled/raw/landsat_gaps_filled.parquet \
    swath_gaps_filled/landsat_arctic_swath_gaps_data.parquet
do
    if [ -f "$file" ]; then
        echo " Found: $file"
    else
        echo " Missing: $file"
    fi
done
```

### Step 2: Combine Datasets
```bash
python scripts/landsat_combiner.py \
    --original raw/landsat_arctic_data.parquet \
    --gaps gaps_filled/raw/landsat_gaps_filled.parquet \
    --swath-gaps swath_gaps_filled/landsat_arctic_swath_gaps_data.parquet \
    --output final/landsat_complete.parquet \
    2>&1 | tee logs/combining_$(date +%Y%m%d).log
```

### Step 3: Validate Final Dataset
```bash
# Comprehensive validation
python scripts/landsat_test.py --validate-final

# Manual checks
python -c "
import pandas as pd
import numpy as np

df = pd.read_parquet('final/landsat_complete.parquet')

print('=== Final Dataset Summary ===')
print(f'Total points: {len(df):,}')
print(f'Unique scenes: {df[\"scene_id\"].nunique():,}')
print(f'Date range: {df[\"acquisition_date\"].min()} to {df[\"acquisition_date\"].max()}')
print(f'Spatial extent:')
print(f'  Longitude: [{df[\"longitude\"].min():.2f}, {df[\"longitude\"].max():.2f}]')
print(f'  Latitude: [{df[\"latitude\"].min():.2f}, {df[\"latitude\"].max():.2f}]')
print(f'Cloud cover: {df[\"cloud_cover\"].mean():.2f}% (mean)')

# Check for duplicates
duplicates = df.duplicated(subset=['longitude', 'latitude', 'scene_id']).sum()
print(f'Duplicates: {duplicates}')

# Check for nulls
print(f'Null values: {df.isnull().sum().sum()}')
"
```

---

## Validation

### Comprehensive Validation Script
```bash
# Run full validation suite
python scripts/landsat_test.py --comprehensive

# Generate validation report
python scripts/landsat_test.py --generate-report \
    --output final/validation_report_$(date +%Y%m%d).html
```

### Manual Quality Checks

#### 1. Coverage Verification
```python
import pandas as pd
import numpy as np

df = pd.read_parquet('final/landsat_complete.parquet')

# Create coverage grid
lon_bins = np.arange(-180, 181, 5)
lat_bins = np.arange(45, 91, 5)

grid = np.zeros((len(lon_bins)-1, len(lat_bins)-1))

for _, row in df.iterrows():
    lon_idx = np.digitize(row['longitude'], lon_bins) - 1
    lat_idx = np.digitize(row['latitude'], lat_bins) - 1
    if 0 <= lon_idx < len(lon_bins)-1 and 0 <= lat_idx < len(lat_bins)-1:
        grid[lon_idx, lat_idx] = 1

coverage = (grid.sum() / grid.size) * 100
print(f"Coverage: {coverage:.2f}%")

if coverage >= 95:
    print(" Coverage requirement met")
else:
    print(f" Coverage below 95% (actual: {coverage:.2f}%)")
```

#### 2. Temporal Completeness
```python
import pandas as pd

df = pd.read_parquet('final/landsat_complete.parquet')
df['acquisition_date'] = pd.to_datetime(df['acquisition_date'])

# Check temporal distribution
yearly_counts = df.groupby(df['acquisition_date'].dt.year).size()
print("Yearly observation counts:")
print(yearly_counts)

# Check for temporal gaps
df_sorted = df.sort_values('acquisition_date')
df_sorted['date_diff'] = df_sorted['acquisition_date'].diff().dt.days

max_gap = df_sorted['date_diff'].max()
print(f"\nMaximum temporal gap: {max_gap} days")

if max_gap > 180:
    print(f" Warning: Large temporal gap detected ({max_gap} days)")
```

#### 3. Data Quality
```python
import pandas as pd
import numpy as np

df = pd.read_parquet('final/landsat_complete.parquet')

# Check band value ranges (scaled integers)
band_checks = {
    'B2': (0, 65535),
    'B3': (0, 65535),
    'B4': (0, 65535),
    'B10': (0, 65535)
}

print("Band value range checks:")
for band, (min_val, max_val) in band_checks.items():
    actual_min = df[band].min()
    actual_max = df[band].max()
    
    within_range = (min_val <= actual_min) and (actual_max <= max_val)
    status = "" if within_range else ""
    
    print(f"{status} {band}: [{actual_min}, {actual_max}] "
          f"(expected: [{min_val}, {max_val}])")

# Check cloud cover
print(f"\nCloud cover statistics:")
print(f"  Mean: {df['cloud_cover'].mean():.2f}%")
print(f"  Median: {df['cloud_cover'].median():.2f}%")
print(f"  Max: {df['cloud_cover'].max():.2f}%")
```

---

## Advanced Topics

### Parallel Processing Across Multiple Machines
```bash
# On machine 1: Process western hemisphere
python scripts/landsat_downloader.py \
    --config config/landsat_config.yaml \
    --lon-min -180 \
    --lon-max 0

# On machine 2: Process eastern hemisphere
python scripts/landsat_downloader.py \
    --config config/landsat_config.yaml \
    --lon-min 0 \
    --lon-max 180

# Combine results
python scripts/landsat_combiner.py \
    --input-dir machine1/final \
    --input-dir machine2/final \
    --output combined/landsat_complete.parquet
```

### Custom Region Processing
```python
# Create custom region file
import json

custom_regions = [
    {
        'name': 'alaska_north_slope',
        'coords': [-165, 68, -140, 71]
    },
    {
        'name': 'canadian_archipelago',
        'coords': [-110, 70, -60, 80]
    }
]

with open('config/custom_regions.json', 'w') as f:
    json.dump({'regions': custom_regions}, f, indent=2)

# Process custom regions
python scripts/landsat_downloader.py \
    --config config/landsat_config.yaml \
    --region-file config/custom_regions.json
```

### Performance Optimization
```yaml
# config/landsat_config.yaml

# For maximum speed (requires more memory)
processing:
  system:
    max_concurrent_processes: 16
    max_threads_per_process: 16
  
  batching:
    scene_batch_size: 128
    csv_batch_size: 10000
  
  optimization:
    use_adaptive_batching: true
    enable_caching: true

# For limited resources
processing:
  system:
    max_concurrent_processes: 2
    max_threads_per_process: 4
  
  batching:
    scene_batch_size: 16
    csv_batch_size: 1000
```

### Troubleshooting Failed Regions
```bash
# Find failed regions
grep "Error processing region" logs/download_*.log | \
    sed 's/.*region_/region_/' | \
    sed 's/:.*//' | \
    sort | uniq > failed_regions.txt

# Reprocess failed regions
while read region; do
    echo "Reprocessing $region"
    python scripts/landsat_downloader.py \
        --config config/landsat_config.yaml \
        --region-name "$region" \
        --force-restart
done < failed_regions.txt
```

---

## Next Steps

After completing Landsat processing:

1. **Proceed to SMAP Acquisition**: See `../smap/docs/README.md`
2. **Integrate with ArcticDEM**: Prepare for SMAP downscaling
3. **Quality Assessment**: Generate comprehensive data quality report
4. **Data Publication**: Prepare metadata for data repository

---

## References

- [USGS Landsat Collection 2](https://www.usgs.gov/landsat-missions/landsat-collection-2)
- [Google Earth Engine Documentation](https://developers.google.com/earth-engine)
- [Landsat Data Users Handbook](https://www.usgs.gov/landsat-missions/landsat-8-data-users-handbook)

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-01-20