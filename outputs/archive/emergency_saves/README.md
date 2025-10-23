# Emergency Save Files

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
