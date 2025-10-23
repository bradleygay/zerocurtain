# Landsat 8/9 ETM+ Processing Pipeline

**Arctic Zero-Curtain Detection Pipeline - Auxiliary Data Module**

## Overview

This module handles the acquisition, processing, and consolidation of Landsat 8/9 Collection 2 Level-2 data for the circumarctic domain (45-90°N). The processed Landsat data serves as auxiliary input for downscaling SMAP soil moisture data from 9 km to 30 m resolution, alongside ArcticDEM terrain data.

### Purpose

Landsat data provides:
- **Surface reflectance** (Bands 2, 3, 4): Vegetation indices, land cover characterization
- **Surface temperature** (Band 10): Thermal patterns, freeze-thaw indicators
- **Spatial context**: 30 m resolution bridge between SMAP (9 km) and field observations

### Key Features

- Complete circumarctic coverage (45-90°N, ±180°E)
- Multi-stage gap filling (spatial, boundary, swath)
- Automated quality control and cloud filtering
- Parallel processing with checkpointing
- Integration-ready for SMAP downscaling

---

## Quick Start

### Prerequisites
```bash
# Python environment
python >= 3.8

# Required packages
pip install earthengine-api pandas numpy geopandas matplotlib \
            scikit-learn scipy pyyaml rasterio
```

### Earth Engine Setup
```bash
# Authenticate (first time only)
earthengine authenticate

# Set project
export EE_PROJECT=circumarcticzerocurtain
```

### Basic Usage
```bash
cd ~/arctic_zero_curtain_pipeline/data/auxiliary/landsat

# 1. Download initial dataset
python scripts/landsat_downloader.py --config config/landsat_config.yaml

# 2. Fill gaps
python scripts/landsat_gap_filler.py --config config/landsat_config.yaml

# 3. Fill swath gaps
python scripts/landsat_swath_gap_filler.py --config config/landsat_config.yaml

# 4. Combine all datasets
python scripts/landsat_combiner.py --config config/landsat_config.yaml

# 5. Validate
python scripts/landsat_test.py
```

---

## Directory Structure
```
landsat/
 config/                          # Configuration files
    landsat_config.yaml         # Main configuration
    processing_params.yaml      # Processing parameters
    regions_config.yaml         # Region definitions

 scripts/                         # Processing scripts
    landsat_downloader.py       # Initial download
    landsat_gap_filler.py       # Spatial gap filling
    landsat_swath_gap_filler.py # Swath gap filling
    landsat_visualizer.py       # Visualization tools
    landsat_combiner.py         # Dataset merging
    landsat_test.py             # Testing suite
    utils/                       # Utility modules

 docs/                            # Documentation
    README.md                    # This file
    PROCESSING_GUIDE.md         # Detailed processing guide
    API_REFERENCE.md            # Code documentation
    TROUBLESHOOTING.md          # Common issues

 raw/                             # Raw downloaded data
 processed/                       # Processed intermediate data
 gaps_filled/                     # Gap-filled datasets
 swath_gaps_filled/              # Swath gap-filled datasets
 final/                           # Final consolidated dataset
 checkpoints/                     # Processing checkpoints
 temp_csv/                        # Temporary CSV files
 analysis/                        # Coverage analysis outputs
 logs/                            # Processing logs
```

---

## Processing Stages

### Stage 1: Initial Download

Downloads Landsat 8 Collection 2 Level-2 data for the entire circumarctic domain.

**Script**: `landsat_downloader.py`

**Key Parameters**:
- Date range: 2015-03-30 to 2024-12-31
- Cloud cover: ≤10%
- Spatial resolution: 30 m
- Bands: B2, B3, B4, B10

**Output**: `raw/landsat_arctic_data.parquet`

### Stage 2: Spatial Gap Filling

Identifies and fills empty grid cells in the spatial domain.

**Script**: `landsat_gap_filler.py`

**Strategy**:
- Grid-based coverage analysis (5° cells)
- Direct query for missing regions
- Relaxed cloud threshold for gap regions

**Output**: `gaps_filled/landsat_gaps_filled.parquet`

### Stage 3: Swath Gap Filling

Fills gaps between Landsat orbital swaths.

**Script**: `landsat_swath_gap_filler.py`

**Strategy**:
- Identifies swath gaps (0.1-2.0° width)
- Cross-track coverage enhancement
- Scene optimization for gap regions

**Output**: `swath_gaps_filled/landsat_arctic_swath_gaps_data.parquet`

### Stage 4: Dataset Combination

Merges all datasets and removes duplicates.

**Script**: `landsat_combiner.py`

**Process**:
1. Load all datasets
2. Concatenate
3. Remove spatial duplicates
4. Final validation

**Output**: `final/landsat_complete.parquet`

---

## Data Schema

### Output Parquet Schema

| Column             | Type    | Description                           | Units      |
|--------------------|---------|---------------------------------------|------------|
| `scene_id`         | string  | Landsat scene identifier              | -          |
| `acquisition_date` | string  | Scene acquisition date (YYYY-MM-DD)   | -          |
| `longitude`        | float64 | Sample point longitude                | degrees    |
| `latitude`         | float64 | Sample point latitude                 | degrees    |
| `B2`               | int32   | Blue band surface reflectance         | scaled int |
| `B3`               | int32   | Green band surface reflectance        | scaled int |
| `B4`               | int32   | Red band surface reflectance          | scaled int |
| `B10`              | int32   | Thermal band surface temperature      | scaled int |
| `cloud_cover`      | float32 | Scene cloud cover percentage          | %          |

### Band Scaling

Surface reflectance (B2, B3, B4):
```python
reflectance = band_value * 0.0000275 - 0.2
```

Surface temperature (B10):
```python
temperature_kelvin = band_value * 0.00341802 + 149.0
```

---

## Configuration

### Main Configuration File

`config/landsat_config.yaml` contains all primary settings:

- **Earth Engine**: Project ID, collection, authentication
- **Spatial Domain**: Bounds, grid resolution, special regions
- **Temporal Range**: Start/end dates, seasonal windows
- **Bands**: Specifications, scaling factors
- **Quality**: Cloud filtering, QA bands
- **Processing**: System resources, batching, optimization
- **Output**: Directories, formats, naming conventions

### Processing Parameters

`config/processing_params.yaml` contains detailed parameters for:

- Download strategies
- Quality control filters
- Gap identification methods
- Gap filling strategies
- Data merging rules
- Performance tuning

### Customization

Edit configuration files before processing:
```yaml
# Example: Adjust date range
temporal:
  start_date: "2020-01-01"
  end_date: "2023-12-31"

# Example: Relax cloud threshold
quality:
  cloud_cover:
    max_percentage: 15.0
```

---

## Quality Assurance

### Validation Checks

Automated validation includes:

1. **Spatial Coverage**: >95% of domain covered
2. **Temporal Coverage**: Minimum 1 scene per region
3. **Data Quality**: Band values within expected ranges
4. **Duplicate Detection**: No spatial/temporal duplicates
5. **Metadata Completeness**: All required fields present

### Quality Metrics

Tracked in `analysis/processing_summary.json`:
```json
{
  "total_points": 50000000,
  "unique_scenes": 125000,
  "coverage_percentage": 98.5,
  "date_range": ["2015-03-30", "2024-12-31"],
  "cloud_cover_range": [0.0, 10.0],
  "spatial_extent": {
    "lon_min": -180.0,
    "lon_max": 180.0,
    "lat_min": 45.0,
    "lat_max": 90.0
  }
}
```

---

## Performance

### Typical Processing Times

| Stage                | Regions | Scenes  | Time (8 cores) |
|----------------------|---------|---------|----------------|
| Initial Download     | ~900    | ~100K   | 72-96 hours    |
| Gap Filling          | ~150    | ~15K    | 12-18 hours    |
| Swath Gap Filling    | ~200    | ~20K    | 18-24 hours    |
| Dataset Combination  | N/A     | N/A     | 2-4 hours      |
| **Total**            | -       | -       | **~120 hours** |

### Optimization Tips

1. **Increase Parallelism**: `--max-processes 16`
2. **Use Larger Batches**: Modify `scene_batch_size` in config
3. **Enable Caching**: Set `enable_caching: true`
4. **Distributed Processing**: Split by regions across multiple machines

---

## Integration with Pipeline

### For SMAP Downscaling

The final Landsat dataset provides:
```python
import pandas as pd

# Load Landsat data
landsat = pd.read_parquet('final/landsat_complete.parquet')

# Extract features for SMAP downscaling
features = landsat[['longitude', 'latitude', 'B2', 'B3', 'B4', 'B10']]

# Calculate vegetation indices
landsat['NDVI'] = (landsat['B4'] - landsat['B3']) / (landsat['B4'] + landsat['B3'])

# Merge with ArcticDEM
# (See SMAP processing documentation)
```

### Data Products

| Product                     | File                          | Use Case              |
|-----------------------------|-------------------------------|-----------------------|
| Raw observations            | `raw/landsat_arctic_data.parquet` | Initial analysis      |
| Complete dataset            | `final/landsat_complete.parquet`  | SMAP downscaling      |
| Coverage analysis           | `analysis/*.json`             | Quality assessment    |
| Processing metadata         | `checkpoints/*.json`          | Reproducibility       |

---

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for detailed solutions.

### Common Issues

**Earth Engine Authentication Error**
```bash
earthengine authenticate --quiet
```

**Memory Error During Processing**
```yaml
# Reduce batch size in config
processing:
  batching:
    scene_batch_size: 32  # Reduced from 64
```

**Incomplete Coverage**
```bash
# Run gap filling multiple times
python scripts/landsat_gap_filler.py --force-restart
```

---

## Support

For issues, questions, or contributions:

- **Documentation**: `docs/PROCESSING_GUIDE.md`
- **API Reference**: `docs/API_REFERENCE.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`

---

## Version History

| Version | Date       | Changes                                      |
|---------|------------|----------------------------------------------|
| 1.0.0   | 2025-01-20 | Initial release                              |
| 1.0.1   | TBD        | Performance optimizations                    |
| 1.1.0   | TBD        | Landsat 9 integration                        |

---

Data products are subject to USGS Landsat data policy:
https://www.usgs.gov/landsat-missions/landsat-data-policy