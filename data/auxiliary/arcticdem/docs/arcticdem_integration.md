# ArcticDEM Integration Documentation

## Overview

This document describes the integration of ArcticDEM auxiliary data processing into the Arctic Zero-Curtain Pipeline. ArcticDEM provides high-resolution terrain data essential for terrain-aware Kriging regression interpolation to downscale SMAP data from 9 km to 30 m resolution.

## Purpose

ArcticDEM serves as an **auxiliary data source** rather than a primary component of the physics-informed remote sensing zero-curtain (PIRSZC) dataframe. The consolidated ArcticDEM dataframe enables:

1. **Terrain-aware spatial interpolation** for SMAP downscaling
2. **Elevation-based kriging weights** for improved spatial prediction
3. **Topographic correction** of remote sensing observations
4. **Geomorphological context** for zero-curtain detection

## Architecture

### Component Modules

```
arctic_zero_curtain_pipeline/
 auxiliary_data/
    arcticdem/
        arcticdem_acquisition.py     # STAC API data retrieval
        arcticdem_processing.py      # Terrain-aware resampling
        arcticdem_consolidation.py   # Parquet dataframe generation
        test_arcticdem_pipeline.py   # Comprehensive test suite
 data/
    auxiliary/
        arcticdem/
            processed/               # 30m resampled tiles
            temp/                    # Temporary downloads
            batches/                 # Intermediate batches
            arcticdem_consolidated.parquet  # Final output
 checkpoints/
    arcticdem/                       # Processing checkpoints
 logs/
     arcticdem/                       # Processing logs
```

### Data Flow

```
STAC API (PGC Minnesota)
    ↓
[Acquisition Module] → Download 2m native resolution tiles
    ↓
[Processing Module] → Terrain-aware resampling to 30m
    ↓
[Consolidation Module] → Generate unified parquet dataframe
    ↓
SMAP Downscaling Pipeline → Kriging interpolation
```

## Installation

### Dependencies

```bash
pip install rasterio numpy pandas pyproj pyarrow scipy requests tqdm
```

### Directory Structure Setup

```bash
# Create required directories
mkdir -p data/auxiliary/arcticdem/{processed,temp,batches}
mkdir -p checkpoints/arcticdem
mkdir -p logs/arcticdem
mkdir -p auxiliary_data/arcticdem
```

## Usage

### 1. Dry Run Testing (REQUIRED FIRST STEP)

Before processing production data, validate the pipeline with synthetic test data:

```bash
cd arctic_zero_curtain_pipeline/auxiliary_data/arcticdem
python test_arcticdem_pipeline.py
```

**Expected Output:**
```
======================================================================
ArcticDEM Pipeline Test Suite - Dry Run
======================================================================

TEST 1: Synthetic data generation
 Generated 3 test tiles

TEST 2: Checkpoint manager
 Checkpoint manager working correctly

TEST 3: Terrain-aware resampling
 Resampled 3 tiles successfully

TEST 4: Coordinate transformation
 Coordinate transformation working correctly

TEST 5: DEM tile processing
 Processed 3 tiles successfully

TEST 6: Batch writing
 Batch writing successful (XXX records)

TEST 7: Full consolidation pipeline
 Consolidation complete (XXX total records)

TEST 8: Performance metrics
 Metrics collection working (X.XXXs elapsed)

TEST 9: Error handling
 Error handling working correctly

TEST 10: Data integrity
 Data integrity verified:
  - Records: XXX
  - Unique locations: XXX
  - Latitude range: X.XX°
  - Longitude range: X.XX°
  - Mean elevation: XXX.Xm ± XX.Xm

======================================================================
Test Results: 10 passed, 0 failed
======================================================================

 ALL TESTS PASSED - Pipeline ready for deployment
```

### 2. Production Data Acquisition

Acquire ArcticDEM tiles from STAC API with checkpoint/resume support:

```bash
python arcticdem_acquisition.py
```

**Features:**
- Automatic retry logic for failed downloads
- Checkpoint-based resumption after interruptions
- Parallel downloads (configurable workers)
- Progress tracking and logging

**Configuration Options:**
```python
config = {
    'stac_url': "https://stac.pgc.umn.edu/api/v1/collections/arcticdem-mosaics-v4.1-2m/items",
    'output_dir': "data/auxiliary/arcticdem/processed",
    'temp_dir': "data/auxiliary/arcticdem/temp",
    'log_dir': "logs/arcticdem",
    'checkpoint_dir': "checkpoints/arcticdem"
}
```

### 3. Terrain-Aware Processing

Resample downloaded tiles to 30m resolution:

```bash
python arcticdem_processing.py
```

**Processing Features:**
- Terrain-adaptive Gaussian filtering
- Slope-weighted smoothing
- Block-averaging resampling
- NoData handling
- Parallel processing

**Algorithm:**
1. Compute terrain slope from elevation gradients
2. Apply Gaussian filter (σ=1.0 for efficiency)
3. Block-reduce to target 30m resolution
4. Update spatial metadata and CRS information

### 4. Dataframe Consolidation

Consolidate processed tiles into unified parquet dataframe:

```bash
python arcticdem_consolidation.py
```

**Consolidation Features:**
- Vectorized coordinate transformations (EPSG:3413 → EPSG:4326)
- Batch processing with checkpointing
- Efficient parquet compression (Snappy)
- Memory-optimized operations

**Output Schema:**
```
DataFrame Columns:
- datetime: datetime64[ns]     # File modification timestamp
- latitude: float64            # Geographic latitude (WGS84)
- longitude: float64           # Geographic longitude (WGS84)
- elevation: float32           # Elevation (meters above sea level)
```

## Performance Considerations

### Computational Resources

**Recommended Configuration:**
- CPU: 8+ cores for parallel processing
- RAM: 16+ GB for large tile operations
- Storage: 500+ GB for intermediate data
- Network: Stable connection for STAC API access

### Processing Time Estimates

Based on full Circumarctic coverage (~50,000 tiles):

| Stage | Estimated Time | Notes |
|-------|---------------|-------|
| Acquisition | 24-48 hours | Network-dependent |
| Processing | 12-24 hours | CPU-bound |
| Consolidation | 2-4 hours | I/O-bound |
| **Total** | **38-76 hours** | Can run unattended |

### Optimization Strategies

1. **Batch Processing**: Process tiles in configurable batches
2. **Checkpointing**: Resume from interruptions without data loss
3. **Parallel Execution**: Utilize multiple CPU cores
4. **Memory Management**: Explicit garbage collection between batches
5. **Compression**: LZW compression for intermediate files

## Integration with SMAP Downscaling

### Spatial Query Pattern

```python
import pandas as pd
import numpy as np

# Load consolidated ArcticDEM
dem_df = pd.read_parquet("data/auxiliary/arcticdem/arcticdem_consolidated.parquet")

# Spatial query for SMAP pixel region
def get_elevation_for_region(center_lat, center_lon, radius_km=4.5):
    """
    Extract elevation data for SMAP pixel footprint
    
    Parameters:
    -----------
    center_lat : float
        Center latitude of SMAP pixel
    center_lon : float
        Center longitude of SMAP pixel
    radius_km : float
        Radius for spatial query (SMAP = 9km → radius ~4.5km)
    
    Returns:
    --------
    pd.DataFrame
        Elevation data within region
    """
    # Convert radius to degrees (approximate)
    radius_deg = radius_km / 111.0
    
    # Spatial filter
    mask = (
        (dem_df['latitude'] >= center_lat - radius_deg) &
        (dem_df['latitude'] <= center_lat + radius_deg) &
        (dem_df['longitude'] >= center_lon - radius_deg) &
        (dem_df['longitude'] <= center_lon + radius_deg)
    )
    
    return dem_df[mask]
```

### Kriging Weight Calculation

```python
def calculate_kriging_weights(target_coords, dem_data):
    """
    Calculate terrain-aware kriging weights for downscaling
    
    Parameters:
    -----------
    target_coords : tuple
        (latitude, longitude) of target 30m pixel
    dem_data : pd.DataFrame
        ArcticDEM data for region
    
    Returns:
    --------
    np.ndarray
        Normalized kriging weights
    """
    # Compute distances
    lat_diff = dem_data['latitude'] - target_coords[0]
    lon_diff = dem_data['longitude'] - target_coords[1]
    distances = np.sqrt(lat_diff**2 + lon_diff**2)
    
    # Compute elevation similarity weights
    target_elev = dem_data['elevation'].median()  # Or from external source
    elev_diff = np.abs(dem_data['elevation'] - target_elev)
    elev_weights = np.exp(-elev_diff / 100)  # 100m characteristic scale
    
    # Combined weights
    distance_weights = np.exp(-distances / 0.1)  # 0.1 degree characteristic scale
    combined_weights = distance_weights * elev_weights
    
    # Normalize
    return combined_weights / combined_weights.sum()
```

## Quality Assurance

### Data Validation Checks

1. **Spatial Coverage**: Verify latitude range (60°N - 90°N)
2. **Elevation Range**: Check for realistic values (-500m to 5000m)
3. **NoData Handling**: Confirm proper masking of invalid regions
4. **Coordinate System**: Validate transformation accuracy
5. **Temporal Consistency**: Verify file timestamps

### Automated Testing

Run comprehensive test suite before production deployment:

```bash
python test_arcticdem_pipeline.py
```

### Manual Verification

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load consolidated data
df = pd.read_parquet("data/auxiliary/arcticdem/arcticdem_consolidated.parquet")

# Basic statistics
print(f"Total records: {len(df):,}")
print(f"Latitude range: {df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°")
print(f"Longitude range: {df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°")
print(f"Elevation range: {df['elevation'].min():.1f}m to {df['elevation'].max():.1f}m")
print(f"Mean elevation: {df['elevation'].mean():.1f}m ± {df['elevation'].std():.1f}m")

# Visualize spatial distribution
plt.figure(figsize=(12, 8))
plt.scatter(df['longitude'], df['latitude'], c=df['elevation'], 
            s=0.1, cmap='terrain', alpha=0.5)
plt.colorbar(label='Elevation (m)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('ArcticDEM Consolidated Data - Spatial Distribution')
plt.tight_layout()
plt.savefig('arcticdem_spatial_distribution.png', dpi=300)
```

## Troubleshooting

### Common Issues

**Issue 1: Download Failures**
```
Error: Failed to download DEM tiles
```
**Solution**: Check network connectivity and STAC API availability. The pipeline automatically retries with exponential backoff.

**Issue 2: Memory Errors During Processing**
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce `MAX_WORKERS` or `BATCH_SIZE` in configuration. Process tiles sequentially if needed.

**Issue 3: Checkpoint Corruption**
```
Error: Invalid checkpoint file format
```
**Solution**: Delete corrupted checkpoint and restart. Data integrity is maintained through atomic writes.

**Issue 4: Coordinate Transformation Errors**
```
Error: Invalid coordinate transformation
```
**Solution**: Verify CRS definitions and ensure pyproj is correctly installed with PROJ database.

### Log Analysis

Check processing logs for detailed diagnostics:

```bash
# View most recent acquisition log
tail -f logs/arcticdem/arcticdem_acquisition_*.log

# View most recent processing log
tail -f logs/arcticdem/arcticdem_processing_*.log

# View most recent consolidation log
tail -f logs/arcticdem/arcticdem_consolidation_*.log
```

## References

### Data Source

- **ArcticDEM**: Polar Geospatial Center, University of Minnesota
- **STAC API**: https://stac.pgc.umn.edu/
- **Documentation**: https://www.pgc.umn.edu/data/arcticdem/

### Technical Specifications

- **Native Resolution**: 2 meters
- **Target Resolution**: 30 meters
- **Coordinate System**: EPSG:3413 (NSIDC Polar Stereographic North) → EPSG:4326 (WGS84)
- **Vertical Datum**: WGS84 Ellipsoid
- **Coverage**: All land areas north of 60°N

### Related Publications

Porter, C., et al. (2018). ArcticDEM. Harvard Dataverse. https://doi.org/10.7910/DVN/OHHUKH

## Maintenance

### Checkpoint Management

Checkpoints should be periodically cleaned after successful completion:

```bash
# Remove old checkpoints (manual cleanup after verification)
rm -rf checkpoints/arcticdem/*.json

# Keep logs for audit trail (archive if needed)
tar -czf arcticdem_logs_$(date +%Y%m%d).tar.gz logs/arcticdem/
```

### Data Updates

ArcticDEM is periodically updated with new acquisitions. To incorporate updates:

1. Delete or archive existing processed data
2. Clear checkpoints
3. Re-run acquisition pipeline with updated STAC query
4. Validate new data against quality metrics

### Version Control

Track pipeline versions and data provenance:

```python
# Add metadata to consolidated dataframe
metadata = {
    'pipeline_version': '1.0.0',
    'creation_date': datetime.now().isoformat(),
    'source': 'ArcticDEM v4.1',
    'processing': 'terrain-aware resampling to 30m',
    'coordinate_system': 'EPSG:4326'
}

# Save metadata alongside dataframe
import json
metadata_file = "data/auxiliary/arcticdem/metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
```

## Contact

For technical issues or questions regarding this integration:

**[RESEARCHER]**  
Arctic Research and Data Scientist  
[RESEARCH_INSTITUTION]  
Email: [contact information]

---

*Document Version: 1.0*  
*Last Updated: October 20, 2025*  
*Next Review: December 2025*