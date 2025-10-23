# UAVSAR Directory Structure
**Last Updated:** October 20, 2025  
**[RESEARCHER], [RESEARCH_INSTITUTION] Arctic Research**

---

##  Organized Structure
```
uavsar/
  production/              ← **ACTIVE PIPELINE** (3 files)
    run_pipeline.py         Main pipeline entry point
    consolidate_all_data.py Standalone consolidator
    pipeline_config.yaml    Pipeline configuration

  modules/                 ← Python modules (12 files)
    interferometry.py       H5 → displacement processing
    consolidation.py        Multi-source consolidation
    spatial_join.py         Spatial-temporal joining
    validation.py           Quality control
    ...                     Other supporting modules

  utilities/               ← Helper scripts (4 files)
    generate_curl_script.py Generate download scripts
    analyze_dependencies.py Analyze file dependencies
    inventory_existing_data.py Inventory data files
    cross_ref_sh_txt.py     Cross-reference scripts

  testing/                 ← Test/diagnostic scripts (7 files)
    simple_consolidation.py Simple test
    first.py - fifth.py     Development tests
    test_config.yaml        Test configuration

  downloads/               ← Download scripts (3 files)
    uavsar_download_curl.sh
    uavsar_download_consolidated.sh
    nisar_download.sh

  archived_pipelines/      ← Old implementations (6 files)
    unified_arctic_pipeline.py Alternative pipeline
    multisource.py          Old consolidation
    generate_interferograms.py Old interferogram gen
    ...                     Other archived pipelines

  data_products/           ← **FINAL OUTPUTS** (2 files)
    nisar.parquet           UAVSAR displacement (317k obs)
    remote_sensing.parquet  Combined UAVSAR+SMAP (2.8M obs)

  nisar_downloads/         ← Raw H5 data (2 sites, 4 files)
    chevak*.h5              Chevak, Alaska H5 files
    Grnlnd*/                Greenland data

  displacement_30m/        ← Processed GeoTIFFs (300+ files)
    *_displacement_30m.tif  30m resolution displacement maps

  output/                  ← Pipeline working directory
    nisar.parquet
    remote_sensing.parquet
    logs/                   Pipeline logs

  documentation/           ← Documentation
    USAGE_GUIDE.md
    dependencies/MODULE_DEPENDENCIES.md
    workflows/*.log         Sample logs

  archive/                 ← Archived/deprecated files
    test_files/             Old test scripts
    deprecated_scripts/     Deprecated code
    experiments/            Experimental code

  logs/                    ← Log files (4 files)

  snaphu-v1.4.2/          ← Phase unwrapping software

  Root files
     data_inventory.json     Data inventory
     uavsar_urls.json        Download URLs
     MASTER_LAUNCHER.sh      **START HERE**
     DIRECTORY_STRUCTURE.md  This file
     README.md               Main documentation
```

---

##  Quick Start

### Run Pipeline
```bash
./MASTER_LAUNCHER.sh
# Select option [2] to consolidate all data
```

### Or run directly:
```bash
cd production
python3 consolidate_all_data.py
```

---

##  Data Products

### nisar.parquet (317,429 records)
UAVSAR displacement observations from all GeoTIFFs
- Columns: datetime, latitude, longitude, displacement_m, polarization, frequency, source, filename

### remote_sensing.parquet (2,818,250 records)
Combined UAVSAR + SMAP dataset
- UAVSAR: 317,429 records (2017)
- SMAP: 2,500,821 records (2015-2024, Alaska region)
- Combined columns: datetime, latitude, longitude, displacement_m, soil_temp_c, soil_moist_frac, source

---

##  View Data
```python
import pandas as pd

# Load data
df = pd.read_parquet('data_products/remote_sensing.parquet')

# Summary
print(df.info())
print(df.describe())
print(df.head(50))

# By source
print(df.groupby('source').size())
```

---

##  Data Quality

**Temperature:** -8.6°C to 9.9°C  (within -60 to 60°C range)  
**Moisture:** 0.0 to 1.0  (valid fraction)  
**Displacement:** Present in UAVSAR records   
**Coordinates:** Alaska region (60-63°N, -167 to -163°W)   

---

##  File Counts

| Directory | Files | Purpose |
|-----------|-------|---------|
| production | 3 | Active pipeline |
| modules | 12 | Python modules |
| utilities | 4 | Helper scripts |
| testing | 7 | Tests/diagnostics |
| downloads | 3 | Download scripts |
| archived_pipelines | 6 | Old implementations |
| data_products | 2 | **Final outputs** |
| displacement_30m | 300+ | GeoTIFFs |
| nisar_downloads | 4 | Raw H5 data |

**Total organized files:** 350+

---

##  Critical Dependencies

**run_pipeline.py depends on:**
- modules/interferometry.py
- modules/consolidation.py
- modules/spatial_join.py
- modules/validation.py
- pipeline_config.yaml

**consolidate_all_data.py depends on:**
- rasterio (read GeoTIFFs)
- pyarrow (read parquet)
- pandas (data processing)
- numpy (numerical operations)

---

##  Contact

**[RESEARCHER]**  
[RESEARCH_INSTITUTION] Arctic Research  
Arctic Remote Sensing Data Scientist  

---
