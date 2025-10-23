#!/usr/bin/env python3
"""
Unified Landsat 8/9 ETM+ Downloader for Arctic Zero-Curtain Pipeline
Consolidates acquisition, processing, and checkpointing functionality
"""

import sys
import os
import json
import time
import gc
import logging
import argparse
import math
import functools
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import ee
import pandas as pd
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration management"""
    
    # System resources
    MAX_CONCURRENT_PROCESSES = min(8, multiprocessing.cpu_count())
    MAX_THREADS_PER_PROCESS = 8
    
    # Processing parameters
    SCENE_BATCH_SIZE = 64
    POINTS_PER_SCENE = 100
    CSV_BATCH_SIZE = 5000
    TILE_SCALE = 16
    
    # Retry configuration
    RETRY_LIMIT = 10
    RETRY_DELAY = 8
    
    # Domain bounds (circumarctic: 45-90°N, -180 to 180°E)
    LON_MIN, LON_MAX = -180, 180
    LAT_MIN, LAT_MAX = 45, 90
    
    # Grid resolution for region creation
    LON_STEP = 10
    LAT_STEP = 5
    
    # Earth Engine project
    EE_PROJECT = 'circumarcticzerocurtain'
    
    # Landsat collection
    LANDSAT_COLLECTION = 'LANDSAT/LC08/C02/T1_L2'
    LANDSAT_BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'ST_B10']
    BAND_NAMES = ['B2', 'B3', 'B4', 'B10']
    
    @classmethod
    def get_output_dirs(cls, base_dir: str) -> Dict[str, Path]:
        """Generate output directory structure"""
        base = Path(base_dir)
        return {
            'raw': base / 'raw',
            'checkpoints': base / 'checkpoints',
            'temp_csv': base / 'temp_csv',
            'analysis': base / 'analysis',
            'logs': base / 'logs'
        }

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: Path, debug: bool = False) -> logging.Logger:
    """Configure logging with file and console handlers"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"landsat_download_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: {log_file}")
    return logger

# ============================================================================
# EARTH ENGINE INITIALIZATION
# ============================================================================

def initialize_earth_engine(project: str, max_retries: int = 3) -> bool:
    """Initialize Earth Engine with retry logic"""
    logger = logging.getLogger(__name__)
    
    for attempt in range(max_retries):
        try:
            ee.Initialize(project=project)
            logger.info("Earth Engine initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"EE initialization attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                try:
                    logger.info("Attempting EE authentication...")
                    ee.Authenticate()
                    ee.Initialize(project=project)
                    logger.info("Earth Engine authenticated and initialized")
                    return True
                except Exception as auth_e:
                    logger.error(f"EE authentication failed: {auth_e}")
                    return False
    
    return False

# ============================================================================
# RETRY DECORATOR
# ============================================================================

def with_retry(max_retries: int = Config.RETRY_LIMIT, 
               initial_delay: int = Config.RETRY_DELAY):
    """Exponential backoff retry decorator"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(__name__)
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt >= max_retries - 1:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                                 f"Retrying in {delay}s...")
                    time.sleep(delay)
            
            return None
        return wrapper
    return decorator

# ============================================================================
# REGION CREATION
# ============================================================================

def create_complete_coverage_regions() -> List[Dict]:
    """
    Create comprehensive region grid ensuring complete circumarctic coverage
    Returns list of region dictionaries with coordinates
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating complete coverage region grid")
    
    regions = []
    
    # Main grid covering entire domain
    for lon in range(Config.LON_MIN, Config.LON_MAX, Config.LON_STEP):
        for lat in range(Config.LAT_MIN, Config.LAT_MAX, Config.LAT_STEP):
            lon_end = min(lon + Config.LON_STEP, Config.LON_MAX)
            lat_end = min(lat + Config.LAT_STEP, Config.LAT_MAX)
            
            regions.append({
                'name': f"region_{lon}_{lat}",
                'coords': [lon, lat, lon_end, lat_end]
            })
    
    # Add dateline regions for complete coverage at ±180°
    for lat in range(Config.LAT_MIN, Config.LAT_MAX, Config.LAT_STEP):
        lat_end = min(lat + Config.LAT_STEP, Config.LAT_MAX)
        
        regions.append({
            'name': f"region_dateline_west_{lat}",
            'coords': [-180, lat, -179, lat_end]
        })
        
        regions.append({
            'name': f"region_dateline_east_{lat}",
            'coords': [179, lat, 180, lat_end]
        })
    
    # Add polar cap regions (85-90°N) with finer resolution
    for lon in range(Config.LON_MIN, Config.LON_MAX, 20):
        regions.append({
            'name': f"region_pole_{lon}",
            'coords': [lon, 85, min(lon + 20, Config.LON_MAX), 90]
        })
    
    logger.info(f"Created {len(regions)} regions for complete coverage")
    
    # Verify coverage
    min_lon = min(r['coords'][0] for r in regions)
    max_lon = max(r['coords'][2] for r in regions)
    min_lat = min(r['coords'][1] for r in regions)
    max_lat = max(r['coords'][3] for r in regions)
    
    coverage_complete = (min_lon <= Config.LON_MIN and max_lon >= Config.LON_MAX and
                        min_lat <= Config.LAT_MIN and max_lat >= Config.LAT_MAX)
    
    if coverage_complete:
        logger.info(f" Coverage verified: Lon [{min_lon}° to {max_lon}°], "
                   f"Lat [{min_lat}° to {max_lat}°]")
    else:
        logger.error(f" Coverage incomplete! Check region grid.")
    
    return regions

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def load_completed_regions(checkpoint_dir: Path) -> set:
    """Load set of completed region names from checkpoint file"""
    logger = logging.getLogger(__name__)
    completed_file = checkpoint_dir / "completed_regions.json"
    
    if not completed_file.exists():
        return set()
    
    try:
        with open(completed_file, 'r') as f:
            data = json.load(f)
            completed = set(data.get('completed_regions', []))
        logger.info(f"Loaded {len(completed)} completed regions")
        return completed
    except Exception as e:
        logger.error(f"Error loading completed regions: {e}")
        return set()

def save_completed_region(region_name: str, checkpoint_dir: Path, 
                         stats_dict: Optional[dict] = None,
                         stats_lock: Optional[multiprocessing.Lock] = None):
    """Mark region as completed in checkpoint file"""
    logger = logging.getLogger(__name__)
    completed_file = checkpoint_dir / "completed_regions.json"
    
    # Update shared dictionary if provided
    if stats_dict is not None and stats_lock is not None:
        with stats_lock:
            if region_name not in stats_dict['completed_regions']:
                stats_dict['completed_regions'].append(region_name)
    
    # Update checkpoint file
    try:
        completed = load_completed_regions(checkpoint_dir)
        completed.add(region_name)
        
        with open(completed_file, 'w') as f:
            json.dump({'completed_regions': list(completed)}, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving completed region {region_name}: {e}")

# ============================================================================
# SCENE SAMPLING
# ============================================================================

def get_scene_sample_points(scene: ee.Image, grid_size: int = 10) -> Optional[ee.FeatureCollection]:
    """
    Create systematic grid of sample points for a scene
    Args:
        scene: Earth Engine Image
        grid_size: Number of points per side of grid
    Returns:
        FeatureCollection of sample points or None if error
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Get scene footprint
        footprint = scene.geometry().bounds().getInfo()
        coords = footprint['coordinates'][0]
        
        # Calculate bounds
        x_coords = [p[0] for p in coords]
        y_coords = [p[1] for p in coords]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Apply 5% inset to avoid edge effects
        inset = 0.05
        width, height = max_x - min_x, max_y - min_y
        
        min_x += width * inset
        max_x -= width * inset
        min_y += height * inset
        max_y -= height * inset
        
        # Create grid
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                if grid_size > 1:
                    x = min_x + (max_x - min_x) * i / (grid_size - 1)
                    y = min_y + (max_y - min_y) * j / (grid_size - 1)
                else:
                    x = (min_x + max_x) / 2
                    y = (min_y + max_y) / 2
                
                points.append(ee.Feature(ee.Geometry.Point([x, y])))
        
        return ee.FeatureCollection(points)
    
    except Exception as e:
        logger.error(f"Error creating sample points: {e}")
        
        # Fallback to centroid
        try:
            center = scene.geometry().centroid(10).getInfo()
            return ee.FeatureCollection([ee.Feature(ee.Geometry.Point(center['coordinates']))])
        except Exception as fallback_e:
            logger.error(f"Fallback sampling failed: {fallback_e}")
            return None

# ============================================================================
# SCENE PROCESSING
# ============================================================================

@with_retry()
def process_single_scene(scene: ee.Image) -> List[Dict]:
    """
    Process single Landsat scene to extract band values at sample points
    Returns list of dictionaries containing extracted data
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Extract metadata
        scene_id = scene.get('LANDSAT_SCENE_ID').getInfo() or scene.get('system:index').getInfo()
        acq_time = scene.get('system:time_start').getInfo()
        acq_date = datetime.fromtimestamp(acq_time / 1000).strftime('%Y-%m-%d')
        cloud_cover = scene.get('CLOUD_COVER').getInfo()
        
        # Get sample points
        grid_size = int(math.sqrt(Config.POINTS_PER_SCENE))
        points = get_scene_sample_points(scene, grid_size)
        
        if points is None:
            return []
        
        # Select bands
        bands = scene.select(Config.LANDSAT_BANDS, Config.BAND_NAMES)
        
        # Sample at points
        samples = bands.reduceRegions(
            collection=points,
            reducer=ee.Reducer.first(),
            scale=30,
            tileScale=Config.TILE_SCALE
        ).getInfo()
        
        # Extract results
        results = []
        for feature in samples.get('features', []):
            props = feature['properties']
            coords = feature['geometry']['coordinates']
            
            # Verify all bands present
            if all(band in props and props[band] is not None for band in Config.BAND_NAMES):
                results.append({
                    'scene_id': scene_id,
                    'acquisition_date': acq_date,
                    'longitude': coords[0],
                    'latitude': coords[1],
                    **{band: props[band] for band in Config.BAND_NAMES},
                    'cloud_cover': cloud_cover
                })
        
        if results:
            logger.info(f"Extracted {len(results)} points from scene {scene_id}")
        else:
            logger.warning(f"No valid data from scene {scene_id}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing scene: {e}")
        return []

@with_retry()
def process_scenes_batch(scenes: List[ee.Image], region_name: str) -> List[Dict]:
    """Process batch of scenes in parallel"""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {len(scenes)} scenes for {region_name}")
    
    results = []
    
    with ThreadPoolExecutor(max_workers=Config.MAX_THREADS_PER_PROCESS) as executor:
        futures = {executor.submit(process_single_scene, scene): scene for scene in scenes}
        
        for future in as_completed(futures):
            try:
                scene_results = future.result()
                if scene_results:
                    results.extend(scene_results)
            except Exception as e:
                logger.error(f"Scene processing error in batch: {e}")
    
    logger.info(f"Batch complete: {len(results)} points extracted from {region_name}")
    return results

# ============================================================================
# REGION PROCESSING
# ============================================================================

def process_region(region_info: Dict, queue: multiprocessing.Queue,
                  stats_dict: dict, stats_lock: multiprocessing.Lock,
                  start_date: str, end_date: str, cloud_threshold: int,
                  output_dirs: Dict[str, Path]):
    """
    Process single region: query EE, process scenes, save results
    """
    logger = logging.getLogger(__name__)
    region_name = region_info['name']
    region_coords = region_info['coords']
    
    # Check if already completed
    checkpoint_file = output_dirs['checkpoints'] / f"{region_name}_checkpoint.json"
    if checkpoint_file.exists():
        logger.info(f"Region {region_name} already completed (checkpoint exists)")
        save_completed_region(region_name, output_dirs['checkpoints'], stats_dict, stats_lock)
        return
    
    # Initialize EE
    if not initialize_earth_engine(Config.EE_PROJECT):
        logger.error(f"EE initialization failed for {region_name}")
        return
    
    logger.info(f"Processing region {region_name}")
    
    try:
        # Define region geometry
        region = ee.Geometry.Rectangle(region_coords)
        
        # Query Landsat collection
        collection = ee.ImageCollection(Config.LANDSAT_COLLECTION) \
            .filterDate(start_date, end_date) \
            .filterBounds(region) \
            .filter(ee.Filter.lt('CLOUD_COVER', cloud_threshold))
        
        collection_size = collection.size().getInfo()
        logger.info(f"Found {collection_size} scenes for {region_name}")
        
        if collection_size == 0:
            # Mark as completed with no data
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "region": region_name,
                    "scenes": 0,
                    "points": 0,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            
            save_completed_region(region_name, output_dirs['checkpoints'], stats_dict, stats_lock)
            return
        
        # Process scenes in batches
        total_points = 0
        all_results = []
        temp_csv = output_dirs['temp_csv'] / f"{region_name}.csv"
        
        for batch_start in range(0, collection_size, Config.SCENE_BATCH_SIZE):
            batch_end = min(batch_start + Config.SCENE_BATCH_SIZE, collection_size)
            batch_size = batch_end - batch_start
            
            logger.info(f"{region_name}: Processing scenes {batch_start+1}-{batch_end}/{collection_size}")
            
            # Get batch
            scenes_list = collection.toList(batch_size, batch_start)
            scenes = [ee.Image(scenes_list.get(i)) for i in range(batch_size)]
            
            # Process batch
            batch_results = process_scenes_batch(scenes, region_name)
            all_results.extend(batch_results)
            total_points += len(batch_results)
            
            # Write intermediate results
            if len(all_results) >= Config.CSV_BATCH_SIZE:
                df = pd.DataFrame(all_results)
                mode = 'a' if temp_csv.exists() else 'w'
                header = not temp_csv.exists()
                df.to_csv(temp_csv, mode=mode, header=header, index=False)
                
                queue.put(("data", str(temp_csv), len(all_results)))
                all_results = []
                del df
                gc.collect()
        
        # Write remaining results
        if all_results:
            df = pd.DataFrame(all_results)
            mode = 'a' if temp_csv.exists() else 'w'
            header = not temp_csv.exists()
            df.to_csv(temp_csv, mode=mode, header=header, index=False)
            
            queue.put(("data", str(temp_csv), len(all_results)))
            del df
            gc.collect()
        
        # Create checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({
                "region": region_name,
                "scenes": collection_size,
                "points": total_points,
                "timestamp": datetime.now().isoformat(),
                "coords": region_coords
            }, f, indent=2)
        
        logger.info(f"Region {region_name} complete: {collection_size} scenes, {total_points} points")
        
        # Update stats
        with stats_lock:
            if region_name not in stats_dict['completed_regions']:
                stats_dict['completed_regions'].append(region_name)
            stats_dict['total_points'] += total_points
        
        queue.put(("complete", region_name, total_points))
    
    except Exception as e:
        logger.error(f"Error processing region {region_name}: {e}")
        
        # Save error checkpoint
        error_file = output_dirs['checkpoints'] / f"{region_name}_error.json"
        with open(error_file, 'w') as f:
            json.dump({
                "region": region_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

# ============================================================================
# MERGER PROCESS
# ============================================================================

def merger_process(queue: multiprocessing.Queue, stats_dict: dict, 
                   stats_lock: multiprocessing.Lock, output_file: Path):
    """
    Merge CSV files into consolidated parquet file
    """
    logger = logging.getLogger(__name__)
    logger.info("Merger process started")
    
    processed_files = set()
    total_points = 0
    
    try:
        while True:
            try:
                item = queue.get(timeout=300)
                
                if item[0] == "data":
                    _, file_path, points = item
                    file_path = Path(file_path)
                    
                    if file_path.exists() and file_path not in processed_files:
                        try:
                            df = pd.read_csv(file_path)
                            
                            if output_file.exists():
                                existing_df = pd.read_parquet(output_file)
                                combined_df = pd.concat([existing_df, df], ignore_index=True)
                                combined_df.to_parquet(output_file, index=False)
                                del existing_df, combined_df
                            else:
                                df.to_parquet(output_file, index=False)
                            
                            processed_files.add(file_path)
                            total_points += points
                            
                            with stats_lock:
                                stats_dict['total_points'] += points
                            
                            logger.info(f"Merged {points} points. Total: {total_points}")
                            
                            del df
                            gc.collect()
                        
                        except Exception as e:
                            logger.error(f"Error processing file {file_path}: {e}")
                
                elif item[0] == "complete":
                    _, region_name, points = item
                    logger.info(f"Region {region_name} merged: {points} points")
                
                elif item[0] == "DONE":
                    logger.info("Received completion signal")
                    break
            
            except Exception as e:
                if "Empty" in str(type(e).__name__):
                    # Check if processing complete
                    with stats_lock:
                        completed = len(stats_dict.get('completed_regions', []))
                        total = stats_dict.get('total_regions', 0)
                    
                    if total > 0 and completed >= total:
                        logger.info(f"All regions completed ({completed}/{total})")
                        break
                else:
                    logger.error(f"Merger error: {e}")
    
    finally:
        # Generate summary
        if output_file.exists():
            try:
                df = pd.read_parquet(output_file)
                
                summary = {
                    "total_points": len(df),
                    "unique_scenes": df['scene_id'].nunique(),
                    "date_range": [df['acquisition_date'].min(), df['acquisition_date'].max()],
                    "spatial_extent": {
                        "lon_min": float(df['longitude'].min()),
                        "lon_max": float(df['longitude'].max()),
                        "lat_min": float(df['latitude'].min()),
                        "lat_max": float(df['latitude'].max())
                    },
                    "regions_processed": len(stats_dict.get('completed_regions', [])),
                    "cloud_cover_range": [float(df['cloud_cover'].min()), 
                                         float(df['cloud_cover'].max())]
                }
                
                summary_file = output_file.parent / "processing_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                logger.info(f"Summary: {summary['total_points']} points from "
                          f"{summary['unique_scenes']} scenes")
                
                del df
                gc.collect()
            
            except Exception as e:
                logger.error(f"Error generating summary: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Unified Landsat 8/9 Downloader for Arctic Zero-Curtain Pipeline'
    )
    parser.add_argument('--start-date', type=str, default='2015-03-30',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--cloud-threshold', type=int, default=10,
                       help='Maximum cloud cover percentage')
    parser.add_argument('--output-dir', type=str, 
                       default='~/arctic_zero_curtain_pipeline/data/auxiliary/landsat',
                       help='Output directory for data')
    parser.add_argument('--max-processes', type=int, default=Config.MAX_CONCURRENT_PROCESSES,
                       help='Maximum concurrent processes')
    parser.add_argument('--force-restart', action='store_true',
                       help='Ignore checkpoints and restart')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup directories
    output_base = Path(args.output_dir).expanduser()
    output_dirs = Config.get_output_dirs(output_base)
    
    # Create directories (using Config.get_output_dirs already)
    # This replaces the need for setup_directories()
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging (must come after directory creation)
    logger = setup_logging(output_dirs['logs'], args.debug)
    
    logger.info("="*60)
    logger.info("UNIFIED LANDSAT DOWNLOADER")
    logger.info("="*60)
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Cloud threshold: {args.cloud_threshold}%")
    logger.info(f"Max processes: {args.max_processes}")
    logger.info(f"Output directory: {output_base}")
    
    # Handle force restart
    if args.force_restart:
        completed_file = output_dirs['checkpoints'] / "completed_regions.json"
        if completed_file.exists():
            completed_file.unlink()
            logger.info("Force restart: removed completed regions")
        
        for checkpoint in output_dirs['checkpoints'].glob("*_checkpoint.json"):
            checkpoint.unlink()
        logger.info("Force restart: removed all checkpoints")
    
    # Initialize Earth Engine
    if not initialize_earth_engine(Config.EE_PROJECT):
        logger.error("EE initialization failed")
        return 1
    
    # Create regions
    regions = create_complete_coverage_regions()
    
    # Load completed regions
    completed_regions = load_completed_regions(output_dirs['checkpoints'])
    regions_to_process = [r for r in regions if r['name'] not in completed_regions]
    
    logger.info(f"Processing {len(regions_to_process)} of {len(regions)} regions")
    
    # Setup multiprocessing
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    stats_dict = manager.dict({
        'completed_regions': manager.list(completed_regions),
        'total_points': 0,
        'total_regions': len(regions)
    })
    stats_lock = manager.Lock()
    
    # Start merger process
    output_file = output_dirs['raw'] / "landsat_arctic_data.parquet"
    merger = multiprocessing.Process(
        target=merger_process,
        args=(result_queue, stats_dict, stats_lock, output_file)
    )
    merger.start()
    
    # Process regions in batches
    start_time = time.time()
    
    for batch_idx in range(0, len(regions_to_process), args.max_processes):
        batch_end = min(batch_idx + args.max_processes, len(regions_to_process))
        current_batch = regions_to_process[batch_idx:batch_end]
        
        logger.info(f"Processing batch {batch_idx+1}-{batch_end} of {len(regions_to_process)}")
        
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
        
        with stats_lock:
            completed = len(stats_dict['completed_regions'])
            total = stats_dict['total_regions']
        
        logger.info(f"Progress: {completed}/{total} regions completed")
        gc.collect()
    
    # Signal merger to finish
    result_queue.put(("DONE", "", 0))
    merger.join()
    
    # Final summary
    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    if output_file.exists():
        df = pd.read_parquet(output_file)
        logger.info(f"Total points: {len(df):,}")
        logger.info(f"Unique scenes: {df['scene_id'].nunique():,}")
        logger.info(f"Date range: {df['acquisition_date'].min()} to {df['acquisition_date'].max()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())