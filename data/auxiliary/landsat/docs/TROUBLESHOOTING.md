# Landsat Processing Pipeline - Troubleshooting Guide

**Version**: 1.0.0  
**Last Updated**: 2025-01-20

Comprehensive guide for diagnosing and resolving common issues in the Landsat processing pipeline.

---

## Table of Contents

1. [Earth Engine Issues](#earth-engine-issues)
2. [Processing Errors](#processing-errors)
3. [Memory Issues](#memory-issues)
4. [Performance Problems](#performance-problems)
5. [Data Quality Issues](#data-quality-issues)
6. [File System Issues](#file-system-issues)
7. [Checkpoint and Resume Issues](#checkpoint-and-resume-issues)
8. [Coverage Gaps](#coverage-gaps)

---

## Earth Engine Issues

### Authentication Error

**Symptom**:
```
Error: Earth Engine authentication failed
EEException: Please authenticate using ee.Authenticate()
```

**Solution 1: Manual Authentication**
```bash
# Run authentication
earthengine authenticate

# Follow browser prompts to authorize
# Copy token back to terminal

# Test connection
python -c "import ee; ee.Initialize(project='circumarcticzerocurtain'); print('Success')"
```

**Solution 2: Check Project ID**
```bash
# Verify your project ID
gcloud config get-value project

# Set correct project
export EE_PROJECT=circumarcticzerocurtain

# Or edit config file
# config/landsat_config.yaml
earth_engine:
  project_id: "your-correct-project-id"
```

**Solution 3: Re-authenticate**
```bash
# Clear credentials
rm -rf ~/.config/earthengine/

# Re-authenticate
earthengine authenticate
```

---

### Rate Limit Errors

**Symptom**:
```
EEException: Too many concurrent requests
429 Error: Rate limit exceeded
```

**Solution 1: Reduce Parallelism**
```yaml
# config/landsat_config.yaml
processing:
  system:
    max_concurrent_processes: 2  # Reduce from 8
    max_threads_per_process: 2   # Reduce from 8
```

**Solution 2: Increase Delay Between Requests**
```python
# Modify RETRY_DELAY in Config class
Config.RETRY_DELAY = 15  # Increase from 8 seconds
```

**Solution 3: Use Smaller Batches**
```yaml
# config/landsat_config.yaml
processing:
  batching:
    scene_batch_size: 16  # Reduce from 64
```

**Verification**:
```bash
# Monitor rate limit recovery
python scripts/landsat_downloader.py --max-processes 1 --debug
```

---

### Project Access Error

**Symptom**:
```
EEException: User does not have access to project 'circumarcticzerocurtain'
```

**Solution**:
1. Verify project ownership or access permissions
2. Contact project administrator for access
3. Or use your own project:
```bash
# Create your own project
gcloud projects create your-project-id

# Enable Earth Engine API
gcloud services enable earthengine.googleapis.com --project=your-project-id

# Update configuration
# config/landsat_config.yaml
earth_engine:
  project_id: "your-project-id"
```

---

## Processing Errors

### Scene Processing Timeout

**Symptom**:
```
ERROR: Scene processing timed out after 300 seconds
Timeout in process_single_scene for LC08_001001_20200101
```

**Solution 1: Increase Tile Scale**
```python
# Reduce computational load by using coarser tiles
Config.TILE_SCALE = 32  # Increase from 16
```

**Solution 2: Reduce Sample Points**
```python
# Fewer points = faster processing
Config.POINTS_PER_SCENE = 64  # Reduce from 100
```

**Solution 3: Skip Problematic Scenes**
```bash
# Process with more lenient error handling
python scripts/landsat_downloader.py --skip-errors
```

---

### Missing Bands Error

**Symptom**:
```
KeyError: 'B10' not found in scene properties
Missing required band in scene LC08_001001_20200101
```

**Solution 1: Verify Collection**
```python
# Check if using correct Landsat collection
# Some older Landsat 8 scenes may have different band names

# In landsat_downloader.py, add band verification:
def verify_bands(scene):
    available_bands = scene.bandNames().getInfo()
    required_bands = Config.LANDSAT_BANDS
    
    missing = set(required_bands) - set(available_bands)
    if missing:
        logger.warning(f"Missing bands: {missing}")
        return False
    return True
```

**Solution 2: Use Alternative Bands**
```yaml
# config/landsat_config.yaml
# For Landsat 7 compatibility
bands:
  surface_reflectance:
    - name: "SR_B1"  # Instead of SR_B2 for L7
```

---

### Checkpoint Corruption

**Symptom**:
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
Error loading checkpoint file
```

**Solution**:
```bash
# Remove corrupted checkpoint
rm checkpoints/region_-180_45_checkpoint.json

# If multiple corrupted files
find checkpoints -name "*_checkpoint.json" -size 0 -delete

# Restart processing
python scripts/landsat_downloader.py
```

---

## Memory Issues

### Out of Memory Error

**Symptom**:
```
MemoryError: Unable to allocate array
Killed (process received SIGKILL)
```

**Solution 1: Reduce Batch Sizes**
```yaml
# config/landsat_config.yaml
processing:
  batching:
    scene_batch_size: 16     # Reduce from 64
    csv_batch_size: 1000     # Reduce from 5000
```

**Solution 2: Enable Aggressive Garbage Collection**
```python
# Add to processing loop
import gc

for batch in batches:
    process_batch(batch)
    gc.collect()  # Force garbage collection after each batch
```

**Solution 3: Use Chunked Processing**
```python
# Process data in smaller chunks
chunk_size = 1000
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    process_chunk(chunk)
    del chunk
    gc.collect()
```

**Solution 4: Monitor Memory Usage**
```python
import psutil

def check_memory():
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        logger.warning(f"High memory usage: {memory.percent}%")
        gc.collect()
        return False
    return True
```

---

### Parquet File Too Large

**Symptom**:
```
MemoryError when reading parquet file
OSError: [Errno 28] No space left on device
```

**Solution 1: Use Dask for Large Files**
```python
import dask.dataframe as dd

# Instead of pandas
# df = pd.read_parquet('large_file.parquet')

# Use dask
ddf = dd.read_parquet('large_file.parquet')
result = ddf.compute()  # Only when needed
```

**Solution 2: Process in Chunks**
```python
import pandas as pd

# Read parquet in chunks
parquet_file = 'final/landsat_complete.parquet'
chunk_size = 100000

for chunk in pd.read_parquet(parquet_file, chunksize=chunk_size):
    process_chunk(chunk)
```

**Solution 3: Use Filters**
```python
# Read only needed columns and rows
df = pd.read_parquet(
    'final/landsat_complete.parquet',
    columns=['longitude', 'latitude', 'B4'],
    filters=[('latitude', '>=', 70)]
)
```

---

## Performance Problems

### Slow Processing

**Symptom**:
- Processing takes longer than expected
- Low CPU utilization
- Frequent timeouts

**Diagnosis**:
```bash
# Monitor processing
python scripts/landsat_downloader.py --debug 2>&1 | tee process.log

# Check CPU usage
htop

# Check I/O wait
iostat -x 5
```

**Solution 1: Increase Parallelism**
```yaml
# config/landsat_config.yaml
processing:
  system:
    max_concurrent_processes: 16  # Increase if you have resources
    max_threads_per_process: 16
```

**Solution 2: Use SSD for Temp Files**
```bash
# Move temp_csv to faster storage
mkdir -p /fast/ssd/temp_csv
ln -s /fast/ssd/temp_csv temp_csv
```

**Solution 3: Optimize Earth Engine Requests**
```python
# Increase tile scale for faster (but lower resolution) requests
Config.TILE_SCALE = 32  # Higher = faster but more memory

# Reduce retry attempts for faster failures
Config.RETRY_LIMIT = 3  # Lower for faster failure detection
```

---

### Disk Space Issues

**Symptom**:
```
OSError: [Errno 28] No space left on device
```

**Solution 1: Check Space**
```bash
# Check available space
df -h ~/arctic_zero_curtain_pipeline/data/auxiliary/landsat

# Check directory sizes
du -sh */
```

**Solution 2: Clean Temporary Files**
```bash
# Remove temporary CSV files
rm -rf temp_csv/*

# Remove old logs
find logs -name "*.log" -mtime +7 -delete

# Remove checkpoints from completed processing
rm checkpoints/*_batch_*.json
```

**Solution 3: Compress Old Data**
```bash
# Compress old parquet files
gzip raw/landsat_arctic_data_backup.parquet

# Use external storage
rsync -avz final/ /external/storage/landsat/
```

---

## Data Quality Issues

### High Cloud Cover

**Symptom**:
- Many scenes with cloud cover >10%
- Poor data quality in output

**Solution 1: Adjust Threshold**
```yaml
# config/landsat_config.yaml
quality:
  cloud_cover:
    max_percentage: 15.0  # Increase for Arctic regions
```

**Solution 2: Use QA Band Filtering**
```python
# Add stricter QA filtering
def filter_clouds(scene):
    qa = scene.select('QA_PIXEL')
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)  # Clear
    return scene.updateMask(cloud_mask)
```

**Solution 3: Temporal Aggregation**
```python
# Use multiple scenes and take median
collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filterDate('2020-06-01', '2020-08-31') \
    .median()  # Composite to reduce clouds
```

---

### Missing Data Points

**Symptom**:
- Fewer points extracted than expected
- Empty or null band values

**Solution 1: Check Sample Points**
```python
# Verify sample points are within scene bounds
points = get_scene_sample_points(scene)
scene_bounds = scene.geometry()

# Check intersection
for point in points.getInfo()['features']:
    coords = point['geometry']['coordinates']
    if not scene_bounds.contains(ee.Geometry.Point(coords)).getInfo():
        logger.warning(f"Point outside scene: {coords}")
```

**Solution 2: Reduce Edge Buffer**
```python
# In get_scene_sample_points(), reduce inset
inset = 0.02  # Reduce from 0.05 to get points closer to edge
```

**Solution 3: Increase Sample Density**
```python
# Increase points per scene
Config.POINTS_PER_SCENE = 144  # 12x12 grid instead of 10x10
```

---

### Duplicate Data Points

**Symptom**:
- Same location appears multiple times
- Inflated dataset size

**Solution 1: Remove Duplicates in Combiner**
```python
# In landsat_combiner.py
combined_df = combined_df.drop_duplicates(
    subset=['longitude', 'latitude', 'scene_id'],
    keep='first'
)
```

**Solution 2: Check for Overlapping Regions**
```python
# Verify regions don't overlap
def check_region_overlap(regions):
    for i, r1 in enumerate(regions):
        for r2 in regions[i+1:]:
            if regions_overlap(r1['coords'], r2['coords']):
                logger.warning(f"Overlap: {r1['name']} and {r2['name']}")
```

---

## File System Issues

### Permission Denied

**Symptom**:
```
PermissionError: [Errno 13] Permission denied: 'checkpoints/region_-180_45_checkpoint.json'
```

**Solution**:
```bash
# Fix permissions
chmod -R u+rwx ~/arctic_zero_curtain_pipeline/data/auxiliary/landsat

# Check ownership
ls -la checkpoints/

# Change ownership if needed
chown -R $USER:$USER ~/arctic_zero_curtain_pipeline/data/auxiliary/landsat
```

---

### File Path Too Long

**Symptom**:
```
OSError: [Errno 36] File name too long
```

**Solution**:
```bash
# Use shorter region names in create_complete_coverage_regions()
def create_region_name(lon, lat):
    return f"r{lon}_{lat}"  # Shorter than "region_-180_45"
```

---

## Checkpoint and Resume Issues

### Cannot Resume After Crash

**Symptom**:
- Processing restarts from beginning
- Completed regions re-processed

**Solution 1: Verify Checkpoint File**
```bash
# Check if checkpoint file exists and is valid
cat checkpoints/completed_regions.json

# Should see:
# {
#   "completed_regions": ["region_-180_45", ...]
# }
```

**Solution 2: Manual Checkpoint Recovery**
```bash
# List all completed region checkpoints
ls checkpoints/*_checkpoint.json

# Create completed_regions.json manually
python -c "
import json
import glob

checkpoints = glob.glob('checkpoints/*_checkpoint.json')
completed = [c.split('/')[-1].replace('_checkpoint.json', '') 
             for c in checkpoints]

with open('checkpoints/completed_regions.json', 'w') as f:
    json.dump({'completed_regions': completed}, f, indent=2)
"
```

**Solution 3: Force Restart Specific Region**
```bash
# Remove specific checkpoint
rm checkpoints/region_-180_45_checkpoint.json

# Remove from completed list
python -c "
import json

with open('checkpoints/completed_regions.json') as f:
    data = json.load(f)

data['completed_regions'].remove('region_-180_45')

with open('checkpoints/completed_regions.json', 'w') as f:
    json.dump(data, f, indent=2)
"

# Reprocess
python scripts/landsat_downloader.py
```

---

## Coverage Gaps

### Persistent Gaps After Gap Filling

**Symptom**:
- Coverage still <95% after gap filling
- Certain regions have no data

**Diagnosis**:
```python
# Analyze which regions have gaps
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

# Find empty cells
empty_cells = []
for i in range(len(lon_bins)-1):
    for j in range(len(lat_bins)-1):
        if grid[i, j] == 0:
            empty_cells.append((lon_bins[i], lat_bins[j]))

print(f"Persistent gaps: {len(empty_cells)}")
for lon, lat in empty_cells[:10]:
    print(f"  Lon: {lon}, Lat: {lat}")
```

**Solution 1: Relax Cloud Threshold**
```bash
# Run gap filler with higher cloud threshold
python scripts/landsat_gap_filler.py --cloud-threshold 20
```

**Solution 2: Extend Temporal Window**
```bash
# Include more years
python scripts/landsat_gap_filler.py \
    --start-date 2013-04-11 \  # Landsat 8 launch date
    --end-date 2024-12-31
```

**Solution 3: Use Landsat 7 for Gaps**
```python
# Modify collection to include Landsat 7
collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
    .filterDate(start_date, end_date) \
    .filterBounds(region) \
    .filter(ee.Filter.lt('CLOUD_COVER', 30))
```

---

### High-Latitude Coverage Issues

**Symptom**:
- Sparse coverage above 80°N
- Many gaps in polar regions

**Explanation**:
Landsat orbits converge at poles, causing:
- Swath overlap increases
- Some areas visited more frequently
- Other areas have wider gaps

**Solution 1: Use Finer Grid**
```python
# In create_complete_coverage_regions(), for high latitudes:
if lat >= 80:
    lon_step = 5  # Finer resolution
    lat_step = 2
```

**Solution 2: Increase Buffer for Polar Regions**
```python
# In enhance_cross_track_coverage()
if region_info['coords'][1] >= 80:
    buffer_degrees = 2.0  # Larger buffer for polar regions
```

**Solution 3: Use Multiple Satellites**
```python
# Combine Landsat 8 and Landsat 9
collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
```

---

## Getting Help

### Enable Debug Logging
```bash
python scripts/landsat_downloader.py --debug 2>&1 | tee debug.log
```

### Generate Diagnostic Report
```python
# Create diagnostic script
import sys
import platform
import psutil
import ee

print("=== Diagnostic Information ===")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"CPU cores: {psutil.cpu_count()}")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"Disk space: {psutil.disk_usage('/').free / (1024**3):.1f} GB free")

try:
    ee.Initialize(project='circumarcticzerocurtain')
    print("Earth Engine: Connected")
except Exception as e:
    print(f"Earth Engine: Error - {e}")
```

### Check Configuration
```bash
# Validate YAML syntax
python -c "
import yaml
with open('config/landsat_config.yaml') as f:
    config = yaml.safe_load(f)
print('Configuration valid')
"
```

### Contact Support

If issues persist:

1. Collect logs: `tar -czf landsat_logs.tar.gz logs/`
2. Include configuration files
3. Provide error messages
4. Describe steps to reproduce

---

## Common Error Messages

### Quick Reference

| Error | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| `EEException: User not authenticated` | Not authenticated with EE | `earthengine authenticate` |
| `429 Rate limit` | Too many requests | Reduce `max_processes` |
| `MemoryError` | Insufficient RAM | Reduce batch sizes |
| `Timeout` | Slow EE response | Increase `TILE_SCALE` |
| `KeyError: 'B10'` | Missing band | Verify collection ID |
| `Permission denied` | File permissions | `chmod -R u+rwx .` |
| `No space left` | Disk full | Clean `temp_csv/` directory |
| `JSONDecodeError` | Corrupt checkpoint | Remove and restart |

---

## Preventive Measures

### Regular Maintenance
```bash
# Weekly cleanup
find logs -name "*.log" -mtime +7 -delete
find temp_csv -name "*.csv" -mtime +1 -delete

# Backup checkpoints
cp -r checkpoints checkpoints_backup_$(date +%Y%m%d)

# Verify data integrity
python scripts/landsat_test.py --validate-all
```

### Monitoring
```bash
# Monitor disk space
watch -n 300 "df -h | grep landsat"

# Monitor processing
tail -f logs/landsat_download_*.log | grep -E "ERROR|WARNING|Progress"
```

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-01-20

For additional support, see:
- [README.md](README.md) - General documentation
- [PROCESSING_GUIDE.md](PROCESSING_GUIDE.md) - Processing instructions
- [API_REFERENCE.md](API_REFERENCE.md) - Technical reference