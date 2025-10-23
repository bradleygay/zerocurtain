# Incremental Save Files

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
