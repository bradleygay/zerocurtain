# UAVSAR Pipeline Usage Guide

## Quick Start

### Option 1: Full Pipeline (H5 → GeoTIFF → Parquet)
```bash
cd production
python3 run_pipeline.py --config pipeline_config.yaml
```

### Option 2: Consolidate Existing Data Only
```bash
cd production
python3 consolidate_all_data.py
```

## Directory Structure
```
uavsar/
 production/              ← ACTIVE SCRIPTS
    run_pipeline.py
    consolidate_all_data.py
    pipeline_config.yaml
 modules/                 ← Python modules (imported by production)
 utilities/               ← Helper scripts
 testing/                 ← Test/diagnostic scripts
 downloads/               ← Download scripts
 archived_pipelines/      ← Old pipeline implementations
 data_products/           ← Final outputs
    nisar.parquet
    remote_sensing.parquet
 nisar_downloads/         ← Raw H5 files
 displacement_30m/        ← Processed GeoTIFFs
 output/                  ← Pipeline working directory
```

## Data Products

**nisar.parquet**: UAVSAR displacement observations
- Columns: datetime, latitude, longitude, displacement_m, polarization, frequency, source

**remote_sensing.parquet**: Combined UAVSAR + SMAP
- UAVSAR columns + soil_temp_c, soil_moist_frac

## View Data
```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data_products/remote_sensing.parquet')
print(df.info())
print(df.head(50))
print(df.describe())
"
```
