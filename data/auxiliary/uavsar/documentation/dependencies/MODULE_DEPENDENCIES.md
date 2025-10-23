# Module Dependencies

## Production Pipeline
**run_pipeline.py** (Main Entry Point)
 modules/interferometry.py
 modules/consolidation.py
 modules/spatial_join.py
 modules/validation.py
 pipeline_config.yaml

**consolidate_all_data.py** (Standalone Consolidator)
 rasterio (reads displacement GeoTIFFs)
 pyarrow (reads SMAP parquet)
 pandas (data processing)

## Archived Pipelines (Not Currently Used)
- unified_arctic_pipeline.py - Alternative complete pipeline
- multisource.py - Multi-source consolidation experiments
- generate_interferograms.py - Older interferogram generator
- nisar_disp_res_v2.py - Version 2 displacement processor
- polarization_handler.py - Polarization-specific processing

## Testing Scripts (Experiments/Diagnostics)
- first.py through fifth.py - Sequential development tests
- simple_consolidation.py - Simplified consolidation test
- diagnose_spatial.py - Spatial matching diagnostics
