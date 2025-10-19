# Auxiliary Data Sources
The physics-informed detection module requires several large auxiliary datasets that are not included in the repository.

## Required Datasets
### 1. Permafrost Probability Raster
**Dataset:** UiO PEX Permafrost Probability (UiO_PEX_PERPROB_5.0)
**Format:** GeoTIFF
**Size:** ~85 GB
**Source:** [University of Oslo Permafrost CCI](http://cci-permafrost.org/)
**Download:** `UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tif`
**Target Location:** `/Users/bagay/Downloads/UiO_PEX_PERPROB_5/`
### 2. Permafrost Zones Shapefile
**Dataset:** UiO PEX Permafrost Zones (UiO_PEX_PERZONES_5.0)
**Format:** Shapefile
**Size:** ~900 MB
**Source:** [University of Oslo Permafrost CCI](http://cci-permafrost.org/)
**Download:** `UiO_PEX_PERZONES_5.0_20181128_2000_2016_NH.shp` (+ associated files)
**Target Location:** `/Users/bagay/Downloads/UiO_PEX_PERZONES_5/`
### 3. Snow Depth and SWE Data
**Dataset:** ERA5-Land Snow Data
**Format:** NetCDF
**Size:** ~3.3 GB
**Source:** [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
**Download:** ERA5-Land monthly snow depth, snow water equivalent, and snowmelt
**Target Location:** `/Users/bagay/Downloads/aa6ddc60e4ed01915fb9193bcc7f4146.nc`

## Setup Instructions
1. Download datasets from sources above
2. Place in specified target locations
3. Verify paths in `src/physics_detection/physics_config.py`
4. Run validation: `python scripts/run_physics_detection.py --validate-only`

## Alternative: Automatic Path Detection
The `DataPaths` configuration class will automatically search `/Users/bagay/Downloads/` for matching files if specific paths are not provided.