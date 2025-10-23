#!/usr/bin/env python3
"""
Landsat Gap Filler - Identifies and fills spatial gaps in Landsat coverage
Part of Arctic Zero-Curtain Pipeline
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ee

# Import from downloader
import sys
sys.path.append(str(Path(__file__).parent))
from landsat_downloader import (
    Config, initialize_earth_engine, setup_logging,
    process_region, merger_process, with_retry
)

# ============================================================================
# GAP ANALYSIS
# ============================================================================

def analyze_coverage(data_file: Path, grid_size: float = 5.0) -> Dict:
    """
    Analyze spatial coverage to identify gaps
    
    Args:
        data_file: Path to parquet file
        grid_size: Grid cell size in degrees
    
    Returns:
        Dictionary with coverage statistics and gap locations
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing coverage: {data_file}")
    
    df = pd.read_parquet(data_file)
    logger.info(f"Loaded {len(df):,} observations")
    
    # Create grid
    lon_edges = np.arange(Config.LON_MIN, Config.LON_MAX + grid_size, grid_size)
    lat_edges = np.arange(Config.LAT_MIN, Config.LAT_MAX + grid_size, grid_size)
    
    # Initialize grid
    grid = np.zeros((len(lon_edges)-1, len(lat_edges)-1))
    cell_counts = np.zeros((len(lon_edges)-1, len(lat_edges)-1))
    
    # Populate grid
    for _, row in df.iterrows():
        lon, lat = row['longitude'], row['latitude']
        
        if Config.LON_MIN <= lon <= Config.LON_MAX and Config.LAT_MIN <= lat <= Config.LAT_MAX:
            lon_idx = np.digitize(lon, lon_edges) - 1
            lat_idx = np.digitize(lat, lat_edges) - 1
            
            if 0 <= lon_idx < len(lon_edges)-1 and 0 <= lat_idx < len(lat_edges)-1:
                grid[lon_idx, lat_idx] = 1
                cell_counts[lon_idx, lat_idx] += 1
    
    # Calculate statistics
    total_cells = grid.size
    filled_cells = int(np.sum(grid))
    empty_cells = total_cells - filled_cells
    coverage_pct = (filled_cells / total_cells) * 100
    
    # Identify gaps
    gap_cells = []
    for i in range(len(lon_edges)-1):
        for j in range(len(lat_edges)-1):
            if grid[i, j] == 0:
                gap_cells.append({
                    'lon_min': float(lon_edges[i]),
                    'lon_max': float(lon_edges[i+1]),
                    'lat_min': float(lat_edges[j]),
                    'lat_max': float(lat_edges[j+1]),
                    'cell_id': f"gap_{float(lon_edges[i])}_{float(lat_edges[j])}"
                })
    
    # Boundary checks
    lon_range = (df['longitude'].min(), df['longitude'].max())
    lat_range = (df['latitude'].min(), df['latitude'].max())
    
    boundary_issues = {
        'western_missing': lon_range[0] > Config.LON_MIN + 1e-6,
        'eastern_missing': lon_range[1] < Config.LON_MAX - 1e-6,
        'southern_missing': lat_range[0] > Config.LAT_MIN + 1e-6,
        'northern_missing': lat_range[1] < Config.LAT_MAX - 1e-6
    }
    
    logger.info(f"Coverage: {coverage_pct:.2f}% ({filled_cells}/{total_cells} cells)")
    logger.info(f"Identified {len(gap_cells)} gap cells")
    
    return {
        'coverage_percentage': float(coverage_pct),
        'total_cells': int(total_cells),
        'filled_cells': int(filled_cells),
        'empty_cells': int(empty_cells),
        'gap_cells': gap_cells,
        'boundary_issues': boundary_issues,
        'lon_range': [float(lon_range[0]), float(lon_range[1])],
        'lat_range': [float(lat_range[0]), float(lat_range[1])]
    }

def create_gap_regions(coverage_analysis: Dict) -> List[Dict]:
    """
    Create region definitions for gap filling
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating gap-filling regions")
    
    gap_regions = []
    
    # Add empty cells
    for cell in coverage_analysis['gap_cells']:
        gap_regions.append({
            'name': cell['cell_id'],
            'coords': [cell['lon_min'], cell['lat_min'], 
                      cell['lon_max'], cell['lat_max']],
            'type': 'empty_cell'
        })
    
    # Add boundary regions
    boundary_issues = coverage_analysis['boundary_issues']
    
    if boundary_issues['western_missing']:
        for lat in range(Config.LAT_MIN, Config.LAT_MAX, 5):
            gap_regions.append({
                'name': f"boundary_west_{lat}",
                'coords': [-180, lat, -175, lat + 5],
                'type': 'western_boundary'
            })
    
    if boundary_issues['eastern_missing']:
        for lat in range(Config.LAT_MIN, Config.LAT_MAX, 5):
            gap_regions.append({
                'name': f"boundary_east_{lat}",
                'coords': [175, lat, 180, lat + 5],
                'type': 'eastern_boundary'
            })
    
    if boundary_issues['northern_missing']:
        for lon in range(Config.LON_MIN, Config.LON_MAX, 20):
            gap_regions.append({
                'name': f"boundary_north_{lon}",
                'coords': [lon, 80, lon + 20, 90],
                'type': 'northern_boundary'
            })
    
    logger.info(f"Created {len(gap_regions)} gap-filling regions")
    
    return gap_regions

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    import argparse
    import multiprocessing
    
    parser = argparse.ArgumentParser(description='Landsat Gap Filler')
    parser.add_argument('--input-file', type=str, required=True,
                       help='Input parquet file to analyze')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for gap-filled data')
    parser.add_argument('--start-date', type=str, default='2015-03-30')
    parser.add_argument('--end-date', type=str, default='2024-12-31')
    parser.add_argument('--cloud-threshold', type=int, default=10)
    parser.add_argument('--max-processes', type=int, default=Config.MAX_CONCURRENT_PROCESSES)
    parser.add_argument('--force-restart', action='store_true')
    
    args = parser.parse_args()
    
    # Setup
    input_file = Path(args.input_file).expanduser()
    output_base = Path(args.output_dir).expanduser()
    output_dirs = Config.get_output_dirs(output_base)
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dirs['logs'])
    
    logger.info("LANDSAT GAP FILLER")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_base}")
    
    # Analyze coverage
    coverage_analysis = analyze_coverage(input_file)
    
    # Save analysis
    analysis_file = output_dirs['analysis'] / "coverage_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(coverage_analysis, f, indent=2)
    
    # Create gap regions
    gap_regions = create_gap_regions(coverage_analysis)
    
    if len(gap_regions) == 0:
        logger.info("No gaps found - coverage is complete!")
        return 0
    
    # Initialize EE
    if not initialize_earth_engine(Config.EE_PROJECT):
        logger.error("EE initialization failed")
        return 1
    
    # Process gap regions (same as main downloader)
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    stats_dict = manager.dict({
        'completed_regions': manager.list(),
        'total_points': 0,
        'total_regions': len(gap_regions)
    })
    stats_lock = manager.Lock()
    
    output_file = output_dirs['raw'] / "landsat_gaps_filled.parquet"
    merger = multiprocessing.Process(
        target=merger_process,
        args=(result_queue, stats_dict, stats_lock, output_file)
    )
    merger.start()
    
    # Process regions
    for batch_idx in range(0, len(gap_regions), args.max_processes):
        batch_end = min(batch_idx + args.max_processes, len(gap_regions))
        current_batch = gap_regions[batch_idx:batch_end]
        
        processes = []
        for region in current_batch:
            p = multiprocessing.Process(
                target=process_region,
                args=(region, result_queue, stats_dict, stats_lock,
                     args.start_date, args.end_date, args.cloud_threshold,
                     output_dirs)
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    
    result_queue.put(("DONE", "", 0))
    merger.join()
    
    logger.info("Gap filling complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())