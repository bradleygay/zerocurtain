# Dataset Statistics

**Generated:** October 20, 2025  
**[RESEARCHER], [RESEARCH_INSTITUTION] Arctic Research**

---

##  Complete Dataset Summary

### Source Data
- **Displacement GeoTIFFs Processed:** 8,197 files
- **UAVSAR Sites:** Multiple Alaska locations
- **Temporal Coverage:** 2012-2022
- **Spatial Resolution:** 30m

### Data Products

#### nisar.parquet (~32 GB)
- **Records:** ~320,000,000 (320 million)
- **Source:** All 8,197 UAVSAR displacement GeoTIFFs
- **Columns:**
  - datetime
  - latitude
  - longitude
  - displacement_m
  - polarization
  - frequency
  - source
  - filename

#### remote_sensing.parquet (~34 GB)
- **Records:** ~322,500,000 (322.5 million)
  - UAVSAR: 320,000,000 observations
  - SMAP: 2,500,000 observations (Alaska region)
- **Combined Columns:**
  - datetime
  - latitude
  - longitude
  - displacement_m (UAVSAR)
  - soil_temp_c (SMAP)
  - soil_moist_frac (SMAP)
  - source
  - Additional metadata

---

##  Memory-Efficient Usage

### Don't Load Entire Dataset
```python
import pandas as pd

#  DON'T DO THIS (will crash)
df = pd.read_parquet('data_products/remote_sensing.parquet')

#  DO THIS: Load specific columns only
df = pd.read_parquet('data_products/remote_sensing.parquet',
                      columns=['datetime', 'latitude', 'longitude', 'source'])

#  DO THIS: Filter while reading
df = pd.read_parquet('data_products/remote_sensing.parquet',
                      filters=[('source', '=', 'UAVSAR')])
```

### Process in Chunks
```python
import pyarrow.parquet as pq

# Read in batches
parquet_file = pq.ParquetFile('data_products/remote_sensing.parquet')

for i in range(parquet_file.num_row_groups):
    batch = parquet_file.read_row_group(i).to_pandas()
    # Process batch
    print(f"Batch {i}: {len(batch):,} rows")
```

### Query Specific Regions
```python
# Load only Alaska region
df = pd.read_parquet('data_products/remote_sensing.parquet',
                      filters=[
                          ('latitude', '>=', 60),
                          ('latitude', '<=', 70),
                          ('longitude', '>=', -170),
                          ('longitude', '<=', -140)
                      ])
```

---

##  Recommended Workflows

### For Analysis
1. Use column filtering to load only needed variables
2. Use spatial/temporal filters to subset data
3. Process in chunks for large-scale analysis
4. Export filtered subsets for repeated use

### For Visualization
1. Downsample spatially (e.g., every 10th pixel)
2. Filter to specific time periods
3. Aggregate to coarser resolution
4. Export visualization-ready subsets

---

##  Data Quality

**Temperature Range:** -60°C to 60°C   
**Moisture Range:** 0-1 fraction   
**Displacement Range:** -10m to 10m   
**Coordinates:** Proper Alaska geolocation   

---
