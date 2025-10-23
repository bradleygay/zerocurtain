# Consolidated Datasets

This directory contains the final consolidated physics-informed zero-curtain event datasets.

## Files

- `physics_informed_zero_curtain_events_COMPLETE.parquet`: Complete dataset with all detected events
  - 54+ million events
  - All three main features (intensity, duration, spatial extent)
  - Full CryoGrid physics features
  - Derived categorical and temporal features

## Features

### Main Features (Target Variables)
- `intensity_percentile`: Zero-curtain intensity [0-1]
- `duration_hours`: Event duration in hours
- `spatial_extent_meters`: Vertical extent of zero-curtain zone

### Physics Features (CryoGrid)
- `cryogrid_thermal_conductivity`: Thermal conductivity [W/m/K]
- `cryogrid_heat_capacity`: Volumetric heat capacity [J/m³/K]
- `cryogrid_enthalpy_stability`: Enthalpy stability metric [0-1]
- `phase_change_energy`: Phase transition energy [J/m³]
- `freeze_penetration_depth`: Freeze depth [m]
- `thermal_diffusivity`: Thermal diffusivity [m²/s]
- `snow_insulation_factor`: Snow insulation effect [0-1]

### Spatiotemporal Features
- `latitude`, `longitude`: Site coordinates
- `depth_zone`: Measurement depth category
- `permafrost_zone`: Permafrost classification
- `permafrost_probability`: Permafrost probability [0-1]
- `start_time`, `end_time`: Event temporal bounds
- `year`, `month`, `season`: Temporal features

### Derived Features
- `intensity_category`: weak/moderate/strong/extreme
- `duration_category`: short/medium/long/extended
- `extent_category`: shallow/moderate/deep/very_deep
- `composite_severity`: Combined severity score [0-1]
- `energy_intensity`: Log-transformed phase change energy
