#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
from datetime import datetime

def organize_outputs_directory():
    """
    Organize outputs directory with proper subdirectory structure.
    """
    print("="*90)
    print("ORGANIZING OUTPUTS DIRECTORY STRUCTURE")
    print("="*90)
    
    base_outputs = Path("/path/to/user/arctic_zero_curtain_pipeline/outputs")
    
    subdirs = {
        'consolidated_datasets': 'Final consolidated physics-informed datasets',
        'emergency_saves': 'Emergency save files from detection runs',
        'incremental_saves': 'Incremental checkpoint files',
        'statistics': 'Statistical summaries and reports',
        'metadata': 'Dataset metadata and configuration files',
        'splits': 'Train/validation/test data splits',
        'archive': 'Archived intermediate results'
    }
    
    print("\nStep 1: Creating directory structure...")
    for subdir, description in subdirs.items():
        subdir_path = base_outputs / subdir
        subdir_path.mkdir(exist_ok=True)
        print(f"   {subdir}/ - {description}")
    
    print("\nStep 2: Moving files to appropriate directories...")
    
    moved_files = []
    
    emergency_files = list(base_outputs.glob("zero_curtain_EMERGENCY_*.parquet"))
    print(f"\n  Moving {len(emergency_files)} emergency save files...")
    for f in emergency_files:
        dest = base_outputs / 'emergency_saves' / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
            moved_files.append(('emergency_saves', f.name))
    
    incremental_files = list(base_outputs.glob("zero_curtain_INCREMENTAL_*.parquet"))
    print(f"  Moving {len(incremental_files)} incremental save files...")
    for f in incremental_files:
        dest = base_outputs / 'incremental_saves' / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
            moved_files.append(('incremental_saves', f.name))
    
    progress_files = list(base_outputs.glob("zero_curtain_PROGRESS_*.txt"))
    print(f"  Moving {len(progress_files)} progress files...")
    for f in progress_files:
        dest = base_outputs / 'incremental_saves' / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
            moved_files.append(('incremental_saves', f.name))
    
    suitability_files = list(base_outputs.glob("*_site_suitability.parquet"))
    print(f"  Moving {len(suitability_files)} site suitability files...")
    for f in suitability_files:
        dest = base_outputs / 'metadata' / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
            moved_files.append(('metadata', f.name))
    
    config_files = list(base_outputs.glob("*_cryogrid_config.txt"))
    print(f"  Moving {len(config_files)} configuration files...")
    for f in config_files:
        dest = base_outputs / 'metadata' / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
            moved_files.append(('metadata', f.name))
    
    stats_files = list(base_outputs.glob("*statistics*.json"))
    print(f"  Moving {len(stats_files)} statistics files...")
    for f in stats_files:
        dest = base_outputs / 'statistics' / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
            moved_files.append(('statistics', f.name))
    
    summary_files = list(base_outputs.glob("*summary*.json"))
    print(f"  Moving {len(summary_files)} summary files...")
    for f in summary_files:
        dest = base_outputs / 'statistics' / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
            moved_files.append(('statistics', f.name))
    
    complete_dataset = base_outputs / "physics_informed_zero_curtain_events_COMPLETE.parquet"
    if complete_dataset.exists():
        dest = base_outputs / 'consolidated_datasets' / complete_dataset.name
        if not dest.exists():
            shutil.move(str(complete_dataset), str(dest))
            moved_files.append(('consolidated_datasets', complete_dataset.name))
            print(f"   Moved complete dataset")
    
    split_files = [
        "physics_informed_events_train.parquet",
        "physics_informed_events_val.parquet",
        "physics_informed_events_test.parquet"
    ]
    for split_file in split_files:
        src = base_outputs / split_file
        if src.exists():
            dest = base_outputs / 'splits' / split_file
            if not dest.exists():
                shutil.move(str(src), str(dest))
                moved_files.append(('splits', split_file))
                print(f"   Moved {split_file}")
    
    print("\nStep 3: Creating README files for each directory...")
    
    readme_contents = {
        'consolidated_datasets': """# Consolidated Datasets

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
""",
        'emergency_saves': """# Emergency Save Files

This directory contains emergency save files generated during physics-informed detection runs.

These files are created automatically when:
- Processing is interrupted (Ctrl+C)
- High-yield sites are detected (>100 events)
- System shutdown is initiated

Each file represents a checkpoint of detected events up to that point in processing.

## File Naming Convention

`zero_curtain_EMERGENCY_site_{site_number}_{event_count}events.parquet`

Example: `zero_curtain_EMERGENCY_site_215_22480events.parquet`
- Processed up to site 215
- Contains 22,480 events total

## Usage

These files were consolidated into the complete dataset in `consolidated_datasets/`.
They serve as:
- Backup checkpoints during processing
- Debugging reference for specific processing stages
- Recovery points if final consolidation fails
""",
        'incremental_saves': """# Incremental Save Files

This directory contains incremental checkpoint files created during detection pipeline execution.

## File Types

### Incremental Datasets
`zero_curtain_INCREMENTAL_site_{site_number}.parquet`
- Saved every 50 sites processed
- Contains all events detected up to that site
- Includes ALL features (not truncated)

### Progress Files
`zero_curtain_PROGRESS_site_{site_number}.txt`
- Text files tracking processing progress
- Metadata about detection run
- Timestamp and processing statistics

## Purpose

- **Resume capability**: Pipeline can resume from last checkpoint
- **Progress monitoring**: Track detection progress in real-time
- **Data safety**: Prevent data loss from interruptions
- **Performance tracking**: Monitor events per site and detection rates
""",
        'statistics': """# Statistics and Summaries

This directory contains comprehensive statistical summaries and analysis reports.

## Files

### `consolidated_physics_statistics.json`
Complete statistical summary of the consolidated dataset including:
- Dataset info (total events, sites, sources)
- Spatial coverage (latitude/longitude ranges)
- Temporal coverage (time span, earliest/latest events)
- Main feature statistics (mean, std, min, max, quantiles)
- Physics feature statistics
- Categorical distributions

### Summary Reports
Additional summary files generated during processing:
- Detection performance summaries
- Site suitability summaries
- Processing time reports

## Usage

These statistics are essential for:
- Dataset documentation
- Publication materials
- Quality assurance
- GeoCryoAI training validation
""",
        'metadata': """# Metadata Files

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
""",
        'splits': """# Train/Validation/Test Splits

This directory contains stratified data splits for GeoCryoAI machine learning.

## Files

- `physics_informed_events_train.parquet`: Training set (70%)
- `physics_informed_events_val.parquet`: Validation set (15%)
- `physics_informed_events_test.parquet`: Testing set (15%)

## Stratification

Splits are stratified by `intensity_category` to ensure:
- Balanced representation of weak/moderate/strong/extreme events
- Consistent distribution across training, validation, and test sets
- Prevention of data leakage between sets

## Usage

**CRITICAL**: 
- Use ONLY training set for model training
- Use validation set for hyperparameter tuning
- Use test set ONLY ONCE for final model evaluation
- Never mix or combine these datasets during training

## Statistics

Training set: ~38 million events
Validation set: ~8 million events
Testing set: ~8 million events

All splits contain identical feature sets and distributions.
""",
        'archive': """# Archive

This directory is for archived intermediate results and deprecated files.

Move old or superseded files here to keep the main outputs directory clean.
"""
    }
    
    for subdir, content in readme_contents.items():
        readme_path = base_outputs / subdir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(content)
        print(f"   Created README for {subdir}/")
    
    print("\nStep 4: Creating master outputs README...")
    
    master_readme = base_outputs / 'README.md'
    master_content = f"""# Outputs Directory

Physics-informed zero-curtain detection pipeline outputs.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Directory Structure
```
outputs/
 consolidated_datasets/    # Final complete datasets (1.7 GB)
 splits/                   # Train/val/test splits (2.7 GB total)
 statistics/               # Statistical summaries (JSON)
 metadata/                 # Configuration and site metadata
 emergency_saves/          # Emergency checkpoint files ({len(emergency_files)} files)
 incremental_saves/        # Incremental checkpoints ({len(incremental_files)} files)
 archive/                  # Archived intermediate results
```

## Quick Access

### Primary Datasets
- **Complete Dataset**: `consolidated_datasets/physics_informed_zero_curtain_events_COMPLETE.parquet`
- **Training Set**: `splits/physics_informed_events_train.parquet`
- **Validation Set**: `splits/physics_informed_events_val.parquet`
- **Testing Set**: `splits/physics_informed_events_test.parquet`

### Statistics
- **Summary Statistics**: `statistics/consolidated_physics_statistics.json`

### Metadata
- **Site Suitability**: `metadata/*_site_suitability.parquet`
- **CryoGrid Configuration**: `metadata/*_cryogrid_config.txt`

## Dataset Overview

**Total Events**: 54,418,117
**Spatial Coverage**: 713 unique Arctic locations (49.4°N to 81.6°N)
**Temporal Coverage**: 132.4 years (1891-2023)
**Features**: 40 (main + physics + derived)

### Main Features
1. `intensity_percentile`: Zero-curtain intensity [0-1]
2. `duration_hours`: Event duration
3. `spatial_extent_meters`: Vertical extent

### Physics Features (CryoGrid)
- Thermal conductivity
- Heat capacity
- Enthalpy stability
- Phase change energy
- Freeze penetration depth
- Thermal diffusivity
- Snow insulation factor

## File Sizes

| Directory | Size | Files |
|-----------|------|-------|
| consolidated_datasets | ~1.7 GB | 1 |
| splits | ~2.7 GB | 3 |
| emergency_saves | ~{len(emergency_files) * 0.15:.1f} GB | {len(emergency_files)} |
| incremental_saves | ~{len(incremental_files) * 0.15:.1f} GB | {len(incremental_files)} |
| statistics | <1 MB | {len(stats_files) + len(summary_files)} |
| metadata | <100 MB | {len(suitability_files) + len(config_files)} |

## Usage Notes

1. **For GeoCryoAI Training**: Use files in `splits/`
2. **For Analysis**: Use `consolidated_datasets/physics_informed_zero_curtain_events_COMPLETE.parquet`
3. **For Statistics**: See `statistics/consolidated_physics_statistics.json`
4. **For Reproducibility**: Check `metadata/` for configuration details

## Next Steps

1. Run QA: `python scripts/qa_physics_results.py`
2. GeoCryoAI integration
3. Model training and evaluation

See individual directory READMEs for detailed information.
"""
    
    with open(master_readme, 'w') as f:
        f.write(master_content)
    print(f"   Created master README.md")
    
    print("\n" + "="*90)
    print("ORGANIZATION COMPLETE")
    print("="*90)
    print(f"\nDirectory structure created with {len(subdirs)} subdirectories")
    print(f"Moved {len(moved_files)} files")
    print(f"Created {len(readme_contents) + 1} README files")
    
    print("\nOrganized structure:")
    for subdir in subdirs.keys():
        subdir_path = base_outputs / subdir
        file_count = len(list(subdir_path.glob("*"))) - 1
        print(f"  {subdir}/: {file_count} files")
    
    print("\n Outputs directory is now properly organized")
    print("="*90)
    
    return base_outputs


if __name__ == "__main__":
    organize_outputs_directory()