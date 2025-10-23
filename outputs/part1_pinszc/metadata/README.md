# Metadata Files

This directory contains metadata about the dataset and detection configuration.

## File Types

### Site Suitability Files
`*_site_suitability.parquet`
- Records which sites were deemed suitable for physics analysis
- Includes permafrost probability and zone classifications
- Documents why sites were included/excluded

### Configuration Files
`*_cryogrid_config.txt`
- CryoGrid integration settings
- Physics model parameters
- Detection thresholds
- Numerical solver settings

## Purpose

Essential metadata for:
- Reproducibility of results
- Understanding detection methodology
- Validating physics model configuration
- Publication documentation
