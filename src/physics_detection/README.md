# Physics-Informed Zero-Curtain Detection Module

## Overview
This module implements physics-informed detection of zero-curtain events across permafrost regions (i.e., continuous, discontinuous, sporadic, isolated) in the circumarctic domain, integrating advanced phase change and subsurface thermodynamic concepts and physical principles including:
- **LPJ-EOSIM permafrost physics**: Heat capacity, thermal conductivity, phase change dynamics
- **CryoGrid integration**: Enthalpy-based formulations, Painter-Karra freezing characteristics, surface energy balance
- **Stefan problem solver**: Crank-Nicholson numerical scheme for freeze-thaw boundary evolution
- **Spatiotemporal snow effects**: Snow insulation, thermal conductivity variations, melt energy dynamics

## Module Structure
```
physics_detection/
├── __init__.py
├── zero_curtain_detector.py    # Core PhysicsInformedZeroCurtainDetector class
├── physics_config.py            # Configuration management
└── README.md                    # This file
```

## Core Components
### PhysicsInformedZeroCurtainDetector Class
**Purpose:** Comprehensive zero-curtain detection using full thermodynamic physics.
**Key Methods:**
- `detect_zero_curtain_with_physics()`: Main detection pipeline
- `solve_stefan_problem_enhanced()`: Numerical solution of Stefan problem with CryoGrid
- `get_site_permafrost_properties()`: Extract permafrost probability and zone data
- `get_site_snow_properties()`: Extract spatiotemporal snow data
- `process_circumarctic_dataset()`: Process complete in situ measurement dataset
### Configuration System
**DetectionConfig:** Complete configuration including:
- Data paths (permafrost, snow, in situ measurements)
- Physics parameters (thresholds, constants)
- Processing parameters (intervals, output naming)

## Usage
### Standalone Execution
```python
from physics_detection.zero_curtain_detector import PhysicsInformedZeroCurtainDetector
from physics_detection.physics_config import DetectionConfig
# Create configuration
config = DetectionConfig()
# Initialize detector
detector = PhysicsInformedZeroCurtainDetector(config=config)
# Process dataset
results = detector.process_circumarctic_dataset()
```
### Orchestrated Execution
```python
from orchestration.physics_detection_orchestrator import PhysicsDetectionOrchestrator
# Initialize orchestrator
orchestrator = PhysicsDetectionOrchestrator()
# Execute complete workflow
results, summary = orchestrator.execute_complete_workflow()
```
### Command-Line Execution
```bash
python scripts/run_physics_detection.py
```

## Configuration Options
### Physics Parameters
```python
PhysicsParameters(
    temp_threshold=3.0,              # Temperature threshold (°C)
    gradient_threshold=1.0,          # Thermal gradient threshold (°C/day)
    min_duration_hours=12,           # Minimum event duration (hours)
    use_cryogrid_enthalpy=True,      # Enable CryoGrid enthalpy formulation
    use_painter_karra_freezing=True, # Enable Painter-Karra freezing characteristic
    use_surface_energy_balance=True, # Enable surface energy balance
    use_adaptive_timestep=True       # Enable adaptive time-stepping
)
```
### Data Paths
Paths are automatically detected from `/Users/bagay/Downloads/` directory. Override if needed:
```python
DataPaths(
    base_dir=Path("/custom/path"),
    permafrost_prob_raster=Path("/custom/permafrost.tif"),
    snow_data_netcdf=Path("/custom/snow.nc")
)
```

## Output Format
### Zero-Curtain Events DataFrame
Columns include:
- `start_time`, `end_time`: Event temporal bounds
- `duration_hours`: Event duration
- `intensity_percentile`: Physics-informed intensity (0-1)
- `spatial_extent_meters`: Vertical extent of zero-curtain
- `latitude`, `longitude`: Site coordinates
- `depth_zone`: Measurement depth category
- `permafrost_probability`: Site permafrost probability
- `permafrost_zone`: Site permafrost zone classification
- `phase_change_energy`: Energy associated with phase transition
- `freeze_penetration_depth`: Depth of freezing front
- `cryogrid_*`: CryoGrid-specific thermal characteristics
### Summary Report (JSON)
```json
{
  "status": "success",
  "total_events": 1234,
  "spatial_coverage": {...},
  "event_characteristics": {...},
  "permafrost_context": {...},
  "cryogrid_statistics": {...}
}
```

## Dependencies
Required packages:
- numpy, pandas, dask
- scipy (optimize, sparse, interpolate, ndimage)
- xarray
- rasterio
- geopandas
- pyproj

## Physics Implementation Details
### Stefan Problem Solver
- Crank-Nicholson implicit scheme
- Adaptive time-stepping for numerical stability
- Phase change energy tracking
- Freeze depth evolution
### CryoGrid Integration
- Enthalpy-based state formulation (Equation 1)
- Temperature-dependent heat capacity (Equation 2)
- Painter-Karra soil freezing characteristic (Equations 18-20)
- Surface energy balance (Equations 5-10)
- Lateral thermal interactions (Equations 32-33)
### Snow Physics
- Dynamic thermal conductivity (Sturm et al.)
- Snow density estimation from SWE
- Melt energy effects
- Insulation factor calculation

## Validation and Quality Control
- Permafrost suitability screening (excludes non-permafrost sites)
- Multi-pathway detection (standard, relaxed, isothermal, temperature-only)
- Fallback detection for weak signatures
- Comprehensive diagnostic logging
- Incremental saving (every 50 sites) with emergency save capability

## Future Enhancements
- GPU acceleration for Stefan problem solver
- Machine learning integration for parameter optimization
- Real-time detection capability
- Enhanced lateral thermal interaction modeling
- Integration with GeoCryoAI framework

## References
- LPJ-EOSIM permafrost module
- CryoGrid community model (Westermann et al.)
- Stefan problem numerical methods
- Painter & Karra (2014) freezing characteristic
- Sturm et al. (1997) snow thermal properties