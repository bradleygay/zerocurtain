"""
Orchestration module for Arctic zero-curtain pipeline.
Auto-generated from Jupyter notebook.
"""

from src.utils.imports import *
from src.utils.utilities import *

import os
import pandas as pd
import numpy as np
import pickle
import gc
import psutil
import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
import warnings
warnings.filterwarnings('ignore')

from physics_detection_orchestrator import PhysicsDetectionOrchestrator

# Define critical utility function
def memory_usage():
    """Monitor memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024**2  # Memory in MB

def log_progress(message, verbose=True):
    """Log progress with timestamp"""
    if verbose:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

# ------------------ HIGH-PERFORMANCE DATA LOADING ------------------

def load_site_data_optimized(feather_path, site_batch, include_moisture=True):
    """
    Load data for multiple sites in a single batch operation
    
    Parameters:
    -----------
    feather_path : str
        Path to the feather file
    site_batch : list of tuples
        List of (site, depth) tuples to process in this batch
    include_moisture : bool
        Whether to include soil moisture data
        
    Returns:
    --------
    dict
        Dictionary mapping (site, depth) tuples to their corresponding DataFrames
    """
    # Collect all sites in this batch
    sites = [site for site, _ in site_batch]
    
    # Create site filter condition
    site_filter = f"source in {sites}"
    
    # Load data for all sites in batch at once
    try:
        # First try reading with query
        columns = ['source', 'soil_temp_depth', 'soil_temp_standardized', 'datetime']
        if include_moisture:
            columns.extend(['soil_moist_depth', 'soil_moist_standardized'])
            
        # Read all data for these sites in one operation
        df_batch = pd.read_feather(feather_path, columns=columns)
        df_batch = df_batch[df_batch['source'].isin(sites)]
        
        # Split by site-depth combination
        result = {}
        for site, depth in site_batch:
            site_df = df_batch[(df_batch['source'] == site) & 
                              (df_batch['soil_temp_depth'] == depth)].copy()
            
            if len(site_df) > 0:
                # Process this site-depth combination
                # Ensure datetime column is datetime type
                if 'datetime' in site_df.columns and not pd.api.types.is_datetime64_dtype(site_df['datetime']):
                    site_df['datetime'] = pd.to_datetime(site_df['datetime'])
                
                # Add depth zone
                site_df['soil_temp_depth_zone'] = pd.cut(
                    site_df['soil_temp_depth'],
                    bins=[-float('inf'), 5, 20, 50, 100, float('inf')],
                    labels=['surface', 'shallow', 'mid', 'deep', 'very_deep']
                )
                
                # Add year
                site_df['year'] = site_df['datetime'].dt.year
                
                # Add season
                def get_season(month):
                    if month in [12, 1, 2]: return 'winter'
                    elif month in [3, 4, 5]: return 'spring'
                    elif month in [6, 7, 8]: return 'summer'
                    else: return 'fall'
                
                site_df['season'] = site_df['datetime'].dt.month.apply(get_season)
                
                # Store processed DataFrame
                result[(site, depth)] = site_df
            else:
                result[(site, depth)] = pd.DataFrame()  # Empty DataFrame if no data
        
        # Clean up
        del df_batch
        gc.collect()
        
        return result
    
    except Exception as e:
        print(f"Error in batch loading: {str(e)}")
        
        # Fallback: Process sites individually
        result = {}
        for site, depth in site_batch:
            try:
                # Read basic columns
                columns = ['source', 'soil_temp_depth', 'soil_temp_standardized', 'datetime']
                if include_moisture:
                    columns.extend(['soil_moist_depth', 'soil_moist_standardized'])
                
                # Filter directly with pandas
                df = pd.read_feather(feather_path, columns=columns)
                site_df = df[(df['source'] == site) & (df['soil_temp_depth'] == depth)].copy()
                
                # Process if we have data
                if len(site_df) > 0:
                    # Ensure datetime column is datetime type
                    if 'datetime' in site_df.columns and not pd.api.types.is_datetime64_dtype(site_df['datetime']):
                        site_df['datetime'] = pd.to_datetime(site_df['datetime'])
                    
                    # Add depth zone
                    site_df['soil_temp_depth_zone'] = pd.cut(
                        site_df['soil_temp_depth'],
                        bins=[-float('inf'), 5, 20, 50, 100, float('inf')],
                        labels=['surface', 'shallow', 'mid', 'deep', 'very_deep']
                    )
                    
                    # Add year and season
                    site_df['year'] = site_df['datetime'].dt.year
                    
                    def get_season(month):
                        if month in [12, 1, 2]: return 'winter'
                        elif month in [3, 4, 5]: return 'spring'
                        elif month in [6, 7, 8]: return 'summer'
                        else: return 'fall'
                    
                    site_df['season'] = site_df['datetime'].dt.month.apply(get_season)
                
                result[(site, depth)] = site_df
                
                # Clean up
                del df
                gc.collect()
                
            except Exception as site_error:
                print(f"Error loading site {site}, depth {depth}: {str(site_error)}")
                result[(site, depth)] = pd.DataFrame()  # Empty DataFrame on error
        
        return result

# ------------------ ZERO CURTAIN DETECTION ------------------

def detect_zero_curtains(site_df, site, temp_depth, max_gap_hours=8, 
                        interpolation_method='cubic'):
    """
    Detect zero curtain events in a site-depth DataFrame
    
    Parameters:
    -----------
    site_df : pandas.DataFrame
        DataFrame containing site-depth data
    site : str
        Site identifier
    temp_depth : float
        Temperature depth
    max_gap_hours : float
        Maximum gap hours for interpolation
    interpolation_method : str
        Interpolation method ('linear', 'cubic')
        
    Returns:
    --------
    list
        List of detected zero curtain events
    """
    # Skip if too few points
    if len(site_df) < 24:  # Require at least 24 measurements
        return []
    
    # Sort by time
    site_df = site_df.sort_values('datetime')
    
    # Calculate time differences
    site_df['time_diff'] = site_df['datetime'].diff().dt.total_seconds() / 3600
    
    # Check for moisture data availability
    has_moisture = ('soil_moist_standardized' in site_df.columns and 
                   not site_df['soil_moist_standardized'].isna().all())
    
    # Interpolate gaps if needed
    interpolation_needed = (site_df['time_diff'] > 1.0) & (site_df['time_diff'] <= max_gap_hours)
    
    if interpolation_needed.any():
        # Simple linear interpolation for gaps
        site_df = site_df.set_index('datetime')
        site_df = site_df.resample('1H').asfreq()
        site_df['soil_temp_standardized'] = site_df['soil_temp_standardized'].interpolate(
            method='linear', limit=max_gap_hours)
        
        if has_moisture:
            site_df['soil_moist_standardized'] = site_df['soil_moist_standardized'].interpolate(
                method='linear', limit=max_gap_hours)
        
        site_df = site_df.reset_index()
    
    # Calculate temperature gradient
    site_df['temp_gradient'] = site_df['soil_temp_standardized'].diff() / \
                             (site_df['datetime'].diff().dt.total_seconds() / 3600)
    
    # Calculate moisture gradient if moisture data exists
    if has_moisture:
        site_df['moist_gradient'] = site_df['soil_moist_standardized'].diff() / \
                                   (site_df['datetime'].diff().dt.total_seconds() / 3600)
    
    # Literature-based detection criteria
    mask_temp = (site_df['soil_temp_standardized'].abs() <= 0.5)  # Temperature near freezing
    mask_gradient = (site_df['temp_gradient'].abs() <= 0.02)  # Stable temperature
    
    # Integrate moisture in detection if available
    if has_moisture:
        # Use moisture gradient for phase change detection
        mask_moisture = (site_df['moist_gradient'].abs() >= 0.0005)  # Moisture changing during phase change
        # Combined detection criteria
        combined_mask = mask_temp & (mask_gradient | mask_moisture)
    else:
        # Use only temperature if no moisture data
        combined_mask = mask_temp & mask_gradient
    
    # Find continuous events
    site_df['zero_curtain_flag'] = combined_mask
    site_df['event_start'] = combined_mask & ~combined_mask.shift(1, fill_value=False)
    site_df['event_end'] = combined_mask & ~combined_mask.shift(-1, fill_value=False)
    
    # Get event starts and ends
    event_starts = site_df[site_df['event_start']]['datetime'].tolist()
    event_ends = site_df[site_df['event_end']]['datetime'].tolist()
    
    if len(event_starts) == 0 or len(event_ends) == 0:
        return []
    
    # Handle mismatched starts/ends
    if len(event_starts) > len(event_ends):
        event_starts = event_starts[:len(event_ends)]
    elif len(event_ends) > len(event_starts):
        event_ends = event_ends[:len(event_starts)]
    
    # Create events
    site_events = []
    
    for start, end in zip(event_starts, event_ends):
        event_duration = (end - start).total_seconds() / 3600
        
        # Literature-based minimum duration (24 hours)
        if event_duration < 24:
            continue
        
        # Get event data
        event_data = site_df[(site_df['datetime'] >= start) & (site_df['datetime'] <= end)]
        
        if len(event_data) < 5:
            continue
        
        # Create event info
        event_info = {
            'source': site,
            'soil_temp_depth': temp_depth,
            'soil_temp_depth_zone': event_data['soil_temp_depth_zone'].iloc[0],
            'datetime_min': start,
            'datetime_max': end,
            'duration_hours': event_duration,
            'observation_count': len(event_data),
            'observations_per_day': len(event_data) / (event_duration / 24) if event_duration > 0 else 0,
            'soil_temp_mean': event_data['soil_temp_standardized'].mean(),
            'soil_temp_min': event_data['soil_temp_standardized'].min(),
            'soil_temp_max': event_data['soil_temp_standardized'].max(),
            'soil_temp_std': event_data['soil_temp_standardized'].std(),
            'season': event_data['season'].iloc[0],
            'year': event_data['year'].iloc[0],
            'month': start.month,
            'temp_gradient_mean': event_data['temp_gradient'].mean() if 'temp_gradient' in event_data else np.nan,
            'temp_stability': event_data['temp_gradient'].abs().mean() if 'temp_gradient' in event_data else np.nan
        }
        
        # Add moisture metrics when available
        if has_moisture and not event_data['soil_moist_standardized'].isna().all():
            event_info['soil_moist_mean'] = event_data['soil_moist_standardized'].mean()
            event_info['soil_moist_std'] = event_data['soil_moist_standardized'].std()
            event_info['soil_moist_min'] = event_data['soil_moist_standardized'].min()
            event_info['soil_moist_max'] = event_data['soil_moist_standardized'].max()
            event_info['soil_moist_change'] = event_data['soil_moist_standardized'].max() - event_data['soil_moist_standardized'].min()
            
            if 'soil_moist_depth' in event_data.columns and not event_data['soil_moist_depth'].isna().all():
                event_info['soil_moist_depth'] = event_data['soil_moist_depth'].iloc[0]
            else:
                event_info['soil_moist_depth'] = np.nan
                
            # Add moisture gradient metrics
            if 'moist_gradient' in event_data.columns:
                event_info['soil_moist_gradient_mean'] = event_data['moist_gradient'].mean()
                event_info['soil_moist_gradient_max'] = event_data['moist_gradient'].abs().max()
        else:
            # Set empty moisture values but preserve columns
            event_info['soil_moist_mean'] = np.nan
            event_info['soil_moist_std'] = np.nan
            event_info['soil_moist_min'] = np.nan
            event_info['soil_moist_max'] = np.nan
            event_info['soil_moist_change'] = np.nan
            event_info['soil_moist_depth'] = np.nan
            event_info['soil_moist_gradient_mean'] = np.nan
            event_info['soil_moist_gradient_max'] = np.nan
        
        # Add year-month
        event_info['year_month'] = f"{event_info['year']}-{event_info['month']:02d}"
        
        site_events.append(event_info)
    
    # Clean up
    del site_df
    gc.collect()
    
    return site_events

# ------------------ WORKER FUNCTIONS FOR PARALLEL PROCESSING ------------------

def process_batch_sites(feather_path, site_batch, include_moisture=True, max_gap_hours=8):
    """
    Process a batch of sites in parallel
    
    Parameters:
    -----------
    feather_path : str
        Path to the feather file
    site_batch : list of tuples
        List of (site, depth) tuples to process
    include_moisture : bool
        Whether to include moisture data
    max_gap_hours : float
        Maximum gap hours for interpolation
        
    Returns:
    --------
    list
        List of detected zero curtain events for this batch
    """
    # Load data for all sites in batch
    batch_data = load_site_data_optimized(feather_path, site_batch, include_moisture)
    
    # Process each site-depth combination
    batch_events = []
    
    for (site, depth), site_df in batch_data.items():
        if len(site_df) < 24:  # Skip if insufficient data
            continue
        
        # Detect zero curtain events
        site_events = detect_zero_curtains(
            site_df=site_df,
            site=site,
            temp_depth=depth,
            max_gap_hours=max_gap_hours
        )
        
        batch_events.extend(site_events)
        
        # Clean up
        del site_df
    
    # Clean up
    del batch_data
    gc.collect()
    
    return batch_events

def worker_init():
    """Initialize worker process"""
    # Reduce memory footprint of worker processes
    import gc
    gc.collect()

# ------------------ HIGH-PERFORMANCE PIPELINE ------------------

def run_high_performance_pipeline(feather_path, output_dir=None, 
                                super_batch_size=200, max_workers=None,
                                max_gap_hours=8, include_moisture=True):
    """
    High-performance zero curtain detection pipeline with parallel processing
    
    Parameters:
    -----------
    feather_path : str
        Path to the feather file
    output_dir : str
        Directory to save results
    super_batch_size : int
        Number of sites to process in each super-batch (will be divided among workers)
    max_workers : int
        Maximum number of worker processes (None=auto)
    max_gap_hours : float
        Maximum gap hours for interpolation
    include_moisture : bool
        Whether to include soil moisture data
    """
    start_time = time.time()
    
    print("=" * 80)
    print("HIGH-PERFORMANCE ZERO CURTAIN DETECTION WITH MOISTURE INTEGRATION")
    print("=" * 80)
    print(f"Initial memory usage: {memory_usage():.1f} MB")
    
    # Set up directories
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        checkpoint_dir = None
    
    # 1. Detect moisture-capable sites
    log_progress("Step 1: Detecting moisture-capable sites...")
    
    # Load site-depth combinations from existing checkpoint if available
    site_depths = None
    moisture_capable_sites = None
    
    if checkpoint_dir:
        try:
            with open(os.path.join(checkpoint_dir, 'site_depths.pkl'), 'rb') as f:
                site_depths = pickle.load(f)
            print(f"Loaded {len(site_depths)} site-depth combinations from checkpoint")
            
            with open(os.path.join(checkpoint_dir, 'moisture_capable_sites.pkl'), 'rb') as f:
                moisture_capable_sites = pickle.load(f)
            print(f"Loaded {len(moisture_capable_sites)} moisture-capable sites from checkpoint")
        except:
            print("Could not load checkpoints, will regenerate")
    
    if site_depths is None:
        # Detect moisture-capable sites first
        try:
            log_progress("Reading moisture data columns...")
            df_sample = pd.read_feather(feather_path, columns=['source', 'soil_moist_standardized'])
            moisture_sites = df_sample.dropna(subset=['soil_moist_standardized'])['source'].unique()
            moisture_capable_sites = set(moisture_sites)
            log_progress(f"Found {len(moisture_capable_sites)} sites with moisture capability")
            
            # Save moisture sites checkpoint
            if checkpoint_dir:
                with open(os.path.join(checkpoint_dir, 'moisture_capable_sites.pkl'), 'wb') as f:
                    pickle.dump(moisture_capable_sites, f)
                log_progress(f"Saved moisture sites checkpoint")
                
            # Get all site-depth combinations
            log_progress("Finding unique site-depth combinations...")
            site_df = pd.read_feather(feather_path, columns=['source', 'soil_temp_depth'])
            valid_df = site_df.dropna(subset=['soil_temp_depth'])
            site_temps = valid_df.drop_duplicates(['source', 'soil_temp_depth'])
            
            # Mark moisture capability
            site_temps['has_moisture_data'] = site_temps['source'].isin(moisture_capable_sites)
            
            # Create list of site-depth tuples
            site_depths = [(row['source'], row['soil_temp_depth'], row['has_moisture_data']) 
                          for _, row in site_temps.iterrows()]
            
            # Save site-depths checkpoint
            if checkpoint_dir:
                with open(os.path.join(checkpoint_dir, 'site_depths.pkl'), 'wb') as f:
                    pickle.dump(site_depths, f)
                log_progress(f"Saved site-depths checkpoint with {len(site_depths)} combinations")
                
            # Clean up
            del df_sample, site_df, valid_df, site_temps
            gc.collect()
            
        except Exception as e:
            log_progress(f"Error in site-depth extraction: {str(e)}")
            return None
    
    total_combinations = len(site_depths)
    moisture_count = sum(1 for _, _, has_moisture in site_depths if has_moisture)
    log_progress(f"Working with {total_combinations} unique site-depth combinations")
    log_progress(f"  {moisture_count} combinations ({moisture_count/total_combinations*100:.1f}%) have moisture data")
    
    # 2. Process site-depths in parallel super-batches
    log_progress("\nStep 2: Processing site-depths in parallel...")
    
    # Load processed indices from checkpoint if available
    processed_indices = set()
    all_events = []
    
    if checkpoint_dir:
        try:
            with open(os.path.join(checkpoint_dir, 'processed_indices.pkl'), 'rb') as f:
                processed_indices = set(pickle.load(f))
            log_progress(f"Loaded {len(processed_indices)} processed indices from checkpoint")
            
            with open(os.path.join(checkpoint_dir, 'all_events.pkl'), 'rb') as f:
                all_events = pickle.load(f)
            log_progress(f"Loaded {len(all_events)} events from checkpoint")
        except:
            log_progress("Could not load events checkpoint, starting fresh")
    
    # Create super-batches for parallel processing
    remaining_indices = [i for i in range(total_combinations) if i not in processed_indices]
    log_progress(f"Processing {len(remaining_indices)} remaining site-depth combinations")
    
    # Determine optimal batch size and number of workers
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    # Adjust super_batch_size to be divisible by max_workers
    worker_batch_size = super_batch_size // max_workers
    if worker_batch_size < 1:
        worker_batch_size = 1
    super_batch_size = worker_batch_size * max_workers
    
    log_progress(f"Using {max_workers} workers with batch size {super_batch_size}")
    
    # Create super-batches
    super_batches = []
    for i in range(0, len(remaining_indices), super_batch_size):
        batch_indices = remaining_indices[i:i+super_batch_size]
        
        # Convert indices to site-depth tuples
        batch_tuples = []
        for idx in batch_indices:
            site, depth, has_moisture = site_depths[idx]
            batch_tuples.append((site, depth))
        
        super_batches.append((batch_indices, batch_tuples))
    
    log_progress(f"Created {len(super_batches)} super-batches")
    
    # Process super-batches one at a time (to control memory usage)
    new_events_count = 0
    
    for batch_idx, (batch_indices, batch_tuples) in enumerate(super_batches):
        log_progress(f"\nProcessing super-batch {batch_idx+1}/{len(super_batches)} " +
                    f"({len(batch_tuples)} site-depths)")
        log_progress(f"Memory before batch: {memory_usage():.1f} MB")
        
        # Split super-batch into worker batches
        worker_batches = []
        for i in range(0, len(batch_tuples), worker_batch_size):
            worker_batches.append(batch_tuples[i:i+worker_batch_size])
        
        batch_start_time = time.time()
        
        # Process worker batches in parallel
        batch_events = []
        
        with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init) as executor:
            # Submit all worker batches
            future_to_batch = {
                executor.submit(
                    process_batch_sites, feather_path, worker_batch, 
                    include_moisture, max_gap_hours
                ): i for i, worker_batch in enumerate(worker_batches)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    events = future.result()
                    if events:
                        batch_events.extend(events)
                        log_progress(f"  Worker {batch_num+1}/{len(worker_batches)} found {len(events)} events")
                except Exception as e:
                    log_progress(f"  Error in worker {batch_num+1}: {str(e)}")
        
        # Add events from this super-batch
        all_events.extend(batch_events)
        new_events_count += len(batch_events)
        
        # Mark indices as processed
        processed_indices.update(batch_indices)
        
        # Save checkpoints
        if checkpoint_dir:
            with open(os.path.join(checkpoint_dir, 'all_events.pkl'), 'wb') as f:
                pickle.dump(all_events, f)
            
            with open(os.path.join(checkpoint_dir, 'processed_indices.pkl'), 'wb') as f:
                pickle.dump(list(processed_indices), f)
            
            log_progress(f"Saved checkpoints with {len(all_events)} total events")
        
        # Also save intermediate results as CSV
        if all_events:
            events_df = pd.DataFrame(all_events)
            if output_dir is not None:
                interim_path = os.path.join(output_dir, 'zero_curtain_events_interim.csv')
                events_df.to_csv(interim_path, index=False)
                log_progress(f"Saved interim results to {interim_path}")
            
            # Report on moisture data
            if 'soil_moist_mean' in events_df.columns:
                moisture_present = events_df['soil_moist_mean'].notna().sum()
                log_progress(f"  {moisture_present} events ({moisture_present/len(events_df)*100:.1f}%) have moisture data")
            
            # Clean up
            del events_df
            gc.collect()
        
        batch_time = time.time() - batch_start_time
        log_progress(f"Super-batch completed in {batch_time:.1f} seconds " +
                    f"({len(batch_tuples)/batch_time:.1f} sites/second)")
        log_progress(f"Memory after batch: {memory_usage():.1f} MB")
        
        # Check if user requested to stop after this batch
        stop_file = 'STOP.txt'
        if os.path.exists(stop_file):
            log_progress(f"Found {stop_file}, stopping after this batch as requested")
            os.remove(stop_file)
            break
    
    # 3. Create final output
    log_progress("\nStep 3: Creating final output...")
    
    if all_events:
        events_df = pd.DataFrame(all_events)
        log_progress(f"Created events dataframe with {len(events_df)} total events " +
                    f"({new_events_count} new in this run)")
        
        # Report on moisture data
        if 'soil_moist_mean' in events_df.columns:
            moisture_present = events_df['soil_moist_mean'].notna().sum()
            log_progress(f"  {moisture_present} events ({moisture_present/len(events_df)*100:.1f}%) have moisture data")
        
        # Save final results
        if output_dir is not None:
            final_path = os.path.join(output_dir, 'zero_curtain_events.csv')
            events_df.to_csv(final_path, index=False)
            log_progress(f"Saved final results to {final_path}")
        
        # Clean up
        del events_df
        gc.collect()
    else:
        log_progress("No events found")
    
    # Report timing
    total_time = time.time() - start_time
    log_progress(f"\nPipeline completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    log_progress(f"Final memory usage: {memory_usage():.1f} MB")
    
    return all_events


# Fix for the structure mismatch error

def run_high_performance_pipeline_fixed(feather_path, output_dir=None, 
                                super_batch_size=200, max_workers=None,
                                max_gap_hours=8, include_moisture=True):
    """
    High-performance zero curtain detection pipeline with parallel processing
    Fixed to handle different site_depths data structures
    """
    start_time = time.time()
    
    print("=" * 80)
    print("HIGH-PERFORMANCE ZERO CURTAIN DETECTION WITH MOISTURE INTEGRATION")
    print("=" * 80)
    print(f"Initial memory usage: {memory_usage():.1f} MB")
    
    # Set up directories
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        checkpoint_dir = None
    
    # 1. Detect moisture-capable sites
    log_progress("Step 1: Detecting moisture-capable sites...")
    
    # Load site-depth combinations from existing checkpoint if available
    site_depths = None
    moisture_capable_sites = None
    
    if checkpoint_dir:
        try:
            with open(os.path.join(checkpoint_dir, 'site_depths.pkl'), 'rb') as f:
                site_depths = pickle.load(f)
            print(f"Loaded {len(site_depths) if isinstance(site_depths, list) else len(site_depths.index) if hasattr(site_depths, 'index') else 0} site-depth combinations from checkpoint")
            
            # Try to load moisture-capable sites
            try:
                with open(os.path.join(checkpoint_dir, 'moisture_capable_sites.pkl'), 'rb') as f:
                    moisture_capable_sites = pickle.load(f)
                print(f"Loaded {len(moisture_capable_sites)} moisture-capable sites from checkpoint")
            except:
                print("Could not load moisture sites checkpoint")
        except:
            print("Could not load site-depths checkpoint, will regenerate")
    
    # Convert site_depths to the expected format if needed
    # We need a list of (site, depth, has_moisture) tuples
    site_depth_tuples = []
    
    if site_depths is not None:
        # Check data type and convert accordingly
        if isinstance(site_depths, pd.DataFrame):
            # Convert DataFrame to list of tuples
            print("Converting DataFrame site_depths to list of tuples")
            for _, row in site_depths.iterrows():
                site = row['source']
                depth = row['soil_temp_depth']
                has_moisture = row.get('has_moisture_data', False)  # Default to False if missing
                site_depth_tuples.append((site, depth, has_moisture))
                
        elif isinstance(site_depths, list):
            # Check format of the list items
            if len(site_depths) > 0:
                sample_item = site_depths[0]
                if isinstance(sample_item, tuple):
                    if len(sample_item) == 3:
                        # Already in the expected format
                        site_depth_tuples = site_depths
                    elif len(sample_item) == 2:
                        # Missing has_moisture flag, add it
                        print("Adding moisture flags to site_depths tuples")
                        if moisture_capable_sites:
                            site_depth_tuples = [(site, depth, site in moisture_capable_sites) 
                                                for site, depth in site_depths]
                        else:
                            site_depth_tuples = [(site, depth, False) for site, depth in site_depths]
                else:
                    # Not a tuple, regenerate
                    print("site_depths list has unexpected format, will regenerate")
                    site_depths = None
        else:
            # Unknown format, regenerate
            print(f"Unknown site_depths format: {type(site_depths)}, will regenerate")
            site_depths = None
    
    # If site_depths is still None or conversion failed, regenerate it
    if site_depths is None or len(site_depth_tuples) == 0:
        # Detect moisture-capable sites first
        try:
            log_progress("Reading moisture data columns...")
            df_sample = pd.read_feather(feather_path, columns=['source', 'soil_moist_standardized'])
            moisture_sites = df_sample.dropna(subset=['soil_moist_standardized'])['source'].unique()
            moisture_capable_sites = set(moisture_sites)
            log_progress(f"Found {len(moisture_capable_sites)} sites with moisture capability")
            
            # Save moisture sites checkpoint
            if checkpoint_dir:
                with open(os.path.join(checkpoint_dir, 'moisture_capable_sites.pkl'), 'wb') as f:
                    pickle.dump(moisture_capable_sites, f)
                log_progress(f"Saved moisture sites checkpoint")
                
            # Get all site-depth combinations
            log_progress("Finding unique site-depth combinations...")
            site_df = pd.read_feather(feather_path, columns=['source', 'soil_temp_depth'])
            valid_df = site_df.dropna(subset=['soil_temp_depth'])
            site_temps = valid_df.drop_duplicates(['source', 'soil_temp_depth'])
            
            # Create list of site-depth tuples
            for _, row in site_temps.iterrows():
                site = row['source']
                depth = row['soil_temp_depth']
                has_moisture = site in moisture_capable_sites
                site_depth_tuples.append((site, depth, has_moisture))
            
            # Save site-depths in both formats
            if checkpoint_dir:
                # Save the original DataFrame
                site_temps['has_moisture_data'] = site_temps['source'].isin(moisture_capable_sites)
                with open(os.path.join(checkpoint_dir, 'site_depths_df.pkl'), 'wb') as f:
                    pickle.dump(site_temps, f)
                
                # Save the tuple list format
                with open(os.path.join(checkpoint_dir, 'site_depths.pkl'), 'wb') as f:
                    pickle.dump(site_depth_tuples, f)
                
                log_progress(f"Saved site-depths checkpoint with {len(site_depth_tuples)} combinations")
                
            # Clean up
            del df_sample, site_df, valid_df, site_temps
            gc.collect()
            
        except Exception as e:
            log_progress(f"Error in site-depth extraction: {str(e)}")
            return None
    
    # Now site_depth_tuples should be a list of (site, depth, has_moisture) tuples
    total_combinations = len(site_depth_tuples)
    moisture_count = sum(1 for _, _, has_moisture in site_depth_tuples if has_moisture)
    log_progress(f"Working with {total_combinations} unique site-depth combinations")
    log_progress(f"  {moisture_count} combinations ({moisture_count/total_combinations*100:.1f}%) have moisture data")
    
    # 2. Process site-depths in parallel super-batches
    log_progress("\nStep 2: Processing site-depths in parallel...")
    
    # Load processed indices from checkpoint if available
    processed_indices = set()
    all_events = []
    
    if checkpoint_dir:
        try:
            with open(os.path.join(checkpoint_dir, 'processed_indices.pkl'), 'rb') as f:
                processed_indices = set(pickle.load(f))
            log_progress(f"Loaded {len(processed_indices)} processed indices from checkpoint")
            
            with open(os.path.join(checkpoint_dir, 'all_events.pkl'), 'rb') as f:
                all_events = pickle.load(f)
            log_progress(f"Loaded {len(all_events)} events from checkpoint")
        except:
            log_progress("Could not load events checkpoint, starting fresh")
    
    # Create super-batches for parallel processing
    remaining_indices = [i for i in range(total_combinations) if i not in processed_indices]
    log_progress(f"Processing {len(remaining_indices)} remaining site-depth combinations")
    
    # Determine optimal batch size and number of workers
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    # Adjust super_batch_size to be divisible by max_workers
    worker_batch_size = super_batch_size // max_workers
    if worker_batch_size < 1:
        worker_batch_size = 1
    super_batch_size = worker_batch_size * max_workers
    
    log_progress(f"Using {max_workers} workers with batch size {super_batch_size}")
    
    # Create super-batches
    super_batches = []
    for i in range(0, len(remaining_indices), super_batch_size):
        batch_indices = remaining_indices[i:i+super_batch_size]
        
        # Convert indices to site-depth tuples
        batch_tuples = []
        for idx in batch_indices:
            site, depth, _ = site_depth_tuples[idx]
            batch_tuples.append((site, depth))
        
        super_batches.append((batch_indices, batch_tuples))
    
    log_progress(f"Created {len(super_batches)} super-batches")
    
    # Process super-batches one at a time (to control memory usage)
    new_events_count = 0
    
    for batch_idx, (batch_indices, batch_tuples) in enumerate(super_batches):
        log_progress(f"\nProcessing super-batch {batch_idx+1}/{len(super_batches)} " +
                    f"({len(batch_tuples)} site-depths)")
        log_progress(f"Memory before batch: {memory_usage():.1f} MB")
        
        # Split super-batch into worker batches
        worker_batches = []
        for i in range(0, len(batch_tuples), worker_batch_size):
            worker_batches.append(batch_tuples[i:i+worker_batch_size])
        
        batch_start_time = time.time()
        
        # Process worker batches in parallel
        batch_events = []
        
        with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init) as executor:
            # Submit all worker batches
            future_to_batch = {
                executor.submit(
                    process_batch_sites, feather_path, worker_batch, 
                    include_moisture, max_gap_hours
                ): i for i, worker_batch in enumerate(worker_batches)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    events = future.result()
                    if events:
                        batch_events.extend(events)
                        log_progress(f"  Worker {batch_num+1}/{len(worker_batches)} found {len(events)} events")
                except Exception as e:
                    log_progress(f"  Error in worker {batch_num+1}: {str(e)}")
        
        # Add events from this super-batch
        all_events.extend(batch_events)
        new_events_count += len(batch_events)
        
        # Mark indices as processed
        processed_indices.update(batch_indices)
        
        # Save checkpoints
        if checkpoint_dir:
            with open(os.path.join(checkpoint_dir, 'all_events.pkl'), 'wb') as f:
                pickle.dump(all_events, f)
            
            with open(os.path.join(checkpoint_dir, 'processed_indices.pkl'), 'wb') as f:
                pickle.dump(list(processed_indices), f)
            
            log_progress(f"Saved checkpoints with {len(all_events)} total events")
        
        # Also save intermediate results as CSV
        if all_events:
            events_df = pd.DataFrame(all_events)
            if output_dir is not None:
                interim_path = os.path.join(output_dir, 'zero_curtain_events_interim.csv')
                events_df.to_csv(interim_path, index=False)
                log_progress(f"Saved interim results to {interim_path}")
            
            # Report on moisture data
            if 'soil_moist_mean' in events_df.columns:
                moisture_present = events_df['soil_moist_mean'].notna().sum()
                log_progress(f"  {moisture_present} events ({moisture_present/len(events_df)*100:.1f}%) have moisture data")
            
            # Clean up
            del events_df
            gc.collect()
        
        batch_time = time.time() - batch_start_time
        log_progress(f"Super-batch completed in {batch_time:.1f} seconds " +
                    f"({len(batch_tuples)/batch_time:.1f} sites/second)")
        log_progress(f"Memory after batch: {memory_usage():.1f} MB")
        
        # Check if user requested to stop after this batch
        stop_file = 'STOP.txt'
        if os.path.exists(stop_file):
            log_progress(f"Found {stop_file}, stopping after this batch as requested")
            os.remove(stop_file)
            break
    
    # 3. Create final output
    log_progress("\nStep 3: Creating final output...")
    
    if all_events:
        events_df = pd.DataFrame(all_events)
        log_progress(f"Created events dataframe with {len(events_df)} total events " +
                    f"({new_events_count} new in this run)")
        
        # Report on moisture data
        if 'soil_moist_mean' in events_df.columns:
            moisture_present = events_df['soil_moist_mean'].notna().sum()
            log_progress(f"  {moisture_present} events ({moisture_present/len(events_df)*100:.1f}%) have moisture data")
        
        # Save final results
        if output_dir is not None:
            final_path = os.path.join(output_dir, 'zero_curtain_events.csv')
            events_df.to_csv(final_path, index=False)
            log_progress(f"Saved final results to {final_path}")
        
        # Clean up
        del events_df
        gc.collect()
    else:
        log_progress("No events found")
    
    # Report timing
    total_time = time.time() - start_time
    log_progress(f"\nPipeline completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    log_progress(f"Final memory usage: {memory_usage():.1f} MB")
    
    return all_events


def estimate_completion_time(site_count, sites_per_second, max_workers):
    """
    Estimate completion time for the pipeline
    
    Parameters:
    -----------
    site_count : int
        Number of sites to process
    sites_per_second : float
        Processing rate (sites/second)
    max_workers : int
        Number of worker processes
    
    Returns:
    --------
    str
        Estimated completion time
    """
    # Estimate total seconds
    total_seconds = site_count / sites_per_second
    
    # Convert to hours, minutes, seconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    return f"{hours}h {minutes}m {seconds}s"

# ------------------ USAGE EXAMPLES ------------------

# Example 1: Quick test run with a small batch
"""
run_high_performance_pipeline(
    feather_path='/Users/bgay/Desktop/Research/Code/merged_compressed.feather',
    output_dir='zero_curtain_moisture_run/events',
    super_batch_size=100,  # Process 100 sites at a time
    max_workers=4  # Use 4 parallel workers
)
"""

# Example 2: Full production run with all CPU cores
"""
# Determine optimal settings
cpu_count = multiprocessing.cpu_count()
max_workers = max(1, cpu_count - 1)  # Leave one CPU free
optimal_batch_size = max_workers * 50  # 50 sites per worker

# Run the pipeline
run_high_performance_pipeline(
    feather_path='/Users/bgay/Desktop/Research/Code/merged_compressed.feather',
    output_dir='zero_curtain_moisture_run/events',
    super_batch_size=optimal_batch_size,
    max_workers=max_workers
)
"""

# Example 3: Run with progress monitoring and graceful shutdown
"""
# Create a file to monitor progress
with open('pipeline_progress.txt', 'w') as f:
    f.write('Pipeline starting...\n')

# Note: To stop the pipeline after the current batch completes,
# create a file named "STOP.txt" in the working directory

# Run the pipeline
run_high_performance_pipeline(
    feather_path='/Users/bgay/Desktop/Research/Code/merged_compressed.feather',
    output_dir='zero_curtain_moisture_run/events',
    super_batch_size=500,
    max_workers=None  # Automatically use optimal number of workers
)
"""

# ------------------ PERFORMANCE OPTIMIZER ------------------

def optimize_settings():
    """Determine optimal settings based on system resources"""
    # Get system info
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # CPU optimization
    if cpu_count >= 16:
        max_workers = cpu_count - 2  # Leave 2 CPUs free on high-core systems
    elif cpu_count >= 8:
        max_workers = cpu_count - 1  # Leave 1 CPU free on multi-core systems
    else:
        max_workers = max(1, cpu_count - 1)  # At least 1 worker
    
    # Memory-based batch sizing
    # Balance: each worker needs ~2GB RAM for efficient processing
    memory_constraint = int(memory_gb / 2)
    cpu_constraint = max_workers * 50  # Each worker handles 50 sites efficiently
    
    # Choose the limiting factor
    super_batch_size = min(memory_constraint, cpu_constraint)
    super_batch_size = max(50, super_batch_size)  # Minimum batch size
    super_batch_size = min(2000, super_batch_size)  # Maximum batch size
    
    # Round to nearest 50
    super_batch_size = (super_batch_size // 50) * 50
    if super_batch_size == 0:
        super_batch_size = 50
    
    return {
        'max_workers': max_workers,
        'super_batch_size': super_batch_size,
        'estimated_sites_per_second': max_workers * 0.8  # Empirical estimation
    }

# """
# Simple fixed runner for the high-performance zero curtain pipeline

# This version requires no command-line arguments and handles different site_depths formats correctl...
# """
# import os
# import pandas as pd
# import numpy as np
# import pickle
# import gc
# import psutil
# import time
# import multiprocessing
# from datetime import datetime
# from concurrent.futures import ProcessPoolExecutor, as_completed

# # === COPY THE ENTIRE optimized-pipeline.py CODE HERE ===
# # (Including all the functions like memory_usage, detect_zero_curtains, etc.)

# # Then add this fixed run_high_performance_pipeline_fixed function:
# # === COPY THE FIXED FUNCTION FROM fixed-pipeline HERE ===

# # === DIRECT RUNNER CODE ===
# if __name__ == "__main__":
#     # Configuration settings - EDIT THESE
#     FEATHER_PATH = '/Users/bgay/Desktop/Research/Code/merged_compressed.feather'
#     OUTPUT_DIR = 'zero_curtain_moisture_run/events'
#     MAX_WORKERS = None  # None means auto-determine based on CPU count
#     SUPER_BATCH_SIZE = None  # None means auto-determine based on system resources
#     MAX_GAP_HOURS = 8
#     INCLUDE_MOISTURE = True
#     ESTIMATE_ONLY = False  # Set to True to only estimate without running

#     # Get optimal settings
#     settings = optimize_settings()

#     # Use provided settings if specified
#     if MAX_WORKERS is not None:
#         settings['max_workers'] = MAX_WORKERS
        
#     if SUPER_BATCH_SIZE is not None:
#         settings['super_batch_size'] = SUPER_BATCH_SIZE

#     print(f"Using {settings['max_workers']} worker processes")
#     print(f"Using super-batch size of {settings['super_batch_size']}")

#     # Estimate time to completion
#     import os
#     import pickle
#     import pandas as pd

#     # Get site count
#     site_depths = None
#     checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
#     os.makedirs(checkpoint_dir, exist_ok=True)
    
#     if os.path.exists(os.path.join(checkpoint_dir, 'site_depths.pkl')):
#         with open(os.path.join(checkpoint_dir, 'site_depths.pkl'), 'rb') as f:
#             site_depths = pickle.load(f)

#     if site_depths is None:
#         print("Reading site data to estimate completion time...")
#         site_df = pd.read_feather(FEATHER_PATH, columns=['source', 'soil_temp_depth'])
#         valid_df = site_df.dropna(subset=['soil_temp_depth'])
#         site_count = len(valid_df.drop_duplicates(['source', 'soil_temp_depth']))
#     else:
#         if isinstance(site_depths, list):
#             site_count = len(site_depths)
#         elif isinstance(site_depths, pd.DataFrame):
#             site_count = len(site_depths)
#         else:
#             print(f"Unknown site_depths format: {type(site_depths)}")
#             site_count = 0

#     # Check for already processed sites
#     processed_count = 0
#     if os.path.exists(os.path.join(checkpoint_dir, 'processed_indices.pkl')):
#         with open(os.path.join(checkpoint_dir, 'processed_indices.pkl'), 'rb') as f:
#             processed_indices = pickle.load(f)
#             processed_count = len(processed_indices)

#     remaining_count = site_count - processed_count

#     # Estimate completion time
#     est_time = estimate_completion_time(
#         remaining_count, 
#         settings['estimated_sites_per_second'],
#         settings['max_workers']
#     )

#     print(f"\nPipeline Settings:")
#     print(f"  Worker processes:  {settings['max_workers']}")
#     print(f"  Super batch size:  {settings['super_batch_size']}")
#     print(f"  Total sites:       {site_count}")
#     print(f"  Already processed: {processed_count}")
#     print(f"  Remaining sites:   {remaining_count}")
#     print(f"  Estimated time:    {est_time}\n")

#     if ESTIMATE_ONLY:
#         print("Estimation complete. Exiting without running pipeline.")
#         import sys
#         sys.exit(0)

#     # Run with simple confirmation
#     if remaining_count > 1000:
#         confirm = input(f"This will process {remaining_count} sites. Continue? (y/n): ")
#         if confirm.lower() != 'y':
#             print("Aborting pipeline run.")
#             import sys
#             sys.exit(0)

#     # Run the FIXED pipeline
#     print("\nStarting pipeline run...\n")
#     run_high_performance_pipeline_fixed(
#         feather_path=FEATHER_PATH,
#         output_dir=OUTPUT_DIR,
#         super_batch_size=settings['super_batch_size'],
#         max_workers=settings['max_workers'],
#         max_gap_hours=MAX_GAP_HOURS,
#         include_moisture=INCLUDE_MOISTURE
#     )

# # ------------------ COMMAND-LINE INTERFACE ------------------

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="High-Performance Zero Curtain Detection Pipeline...
#     parser.add_argument("--feather_path", required=True, help="Path to feather file")
#     parser.add_argument("--output_dir", required=True, help="Output directory")
#     parser.add_argument("--max_workers", type=int, help="Maximum number of worker processes")
#     parser.add_argument("--super_batch_size", type=int, help="Number of sites per super-batch")
#     parser.add_argument("--estimate_only", action="store_true", help="Only estimate completion tim...
#     args = parser.parse_args()
    
#     # Optimize settings if not specified
#     settings = optimize_settings()
    
#     if args.max_workers:
#         settings['max_workers'] = args.max_workers
    
#     if args.super_batch_size:
#         settings['super_batch_size'] = args.super_batch_size
    
#     # Get site count
#     try:
#         site_depths = None
#         checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        
#         if os.path.exists(os.path.join(checkpoint_dir, 'site_depths.pkl')):
#             with open(os.path.join(checkpoint_dir, 'site_depths.pkl'), 'rb') as f:
#                 site_depths = pickle.load(f)
        
#         if site_depths is None:
#             print("Reading site data to estimate completion time...")
#             site_df = pd.read_feather(args.feather_path, columns=['source', 'soil_temp_depth'])
#             valid_df = site_df.dropna(subset=['soil_temp_depth'])
#             site_count = len(valid_df.drop_duplicates(['source', 'soil_temp_depth']))
#         else:
#             site_count = len(site_depths)
        
#         # Check for already processed sites
#         processed_count = 0
#         if os.path.exists(os.path.join(checkpoint_dir, 'processed_indices.pkl')):
#             with open(os.path.join(checkpoint_dir, 'processed_indices.pkl'), 'rb') as f:
#                 processed_indices = pickle.load(f)
#                 processed_count = len(processed_indices)
        
#         remaining_count = site_count - processed_count
        
#         # Estimate completion time
#         est_time = estimate_completion_time(
#             remaining_count, 
#             settings['estimated_sites_per_second'],
#             settings['max_workers']
#         )
        
#         print(f"\nPipeline Settings:")
#         print(f"  Worker processes:  {settings['max_workers']}")
#         print(f"  Super batch size:  {settings['super_batch_size']}")
#         print(f"  Total sites:       {site_count}")
#         print(f"  Already processed: {processed_count}")
#         print(f"  Remaining sites:   {remaining_count}")
#         print(f"  Estimated time:    {est_time}\n")
        
#         if args.estimate_only:
#             print("Estimation complete. Exiting without running pipeline.")
#             sys.exit(0)
        
#         # Confirm before running
#         if remaining_count > 1000:
#             confirm = input("This will process a large number of sites. Continue? (y/n): ")
#             if confirm.lower() != 'y':
#                 print("Aborting pipeline run.")
#                 sys.exit(0)
        
#         # Run the pipeline
#         print("\nStarting pipeline run...\n")
#         run_high_performance_pipeline(
#             feather_path=args.feather_path,
#             output_dir=args.output_dir,
#             super_batch_size=settings['super_batch_size'],
#             max_workers=settings['max_workers']
#         )
        
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         sys.exit(1)

# # Directly run the high-performance pipeline without command-line arguments

# # Copy and paste the entire optimized-pipeline code here, then add this execution code at the end:

# # ---- Direct execution without command-line arguments ----
# if __name__ == "__main__":
#     # Configuration settings - EDIT THESE
#     FEATHER_PATH = '/Users/bgay/Desktop/Research/Code/merged_compressed.feather'
#     OUTPUT_DIR = 'zero_curtain_moisture_run/events'
#     MAX_WORKERS = None  # None means auto-determine based on CPU count
#     SUPER_BATCH_SIZE = None  # None means auto-determine based on system resources
#     MAX_GAP_HOURS = 8
#     INCLUDE_MOISTURE = True
#     ESTIMATE_ONLY = False  # Set to True to only estimate without running

#     # Get optimal settings
#     settings = optimize_settings()

#     # Use provided settings if specified
#     if MAX_WORKERS is not None:
#         settings['max_workers'] = MAX_WORKERS
        
#     if SUPER_BATCH_SIZE is not None:
#         settings['super_batch_size'] = SUPER_BATCH_SIZE

#     print(f"Using {settings['max_workers']} worker processes")
#     print(f"Using super-batch size of {settings['super_batch_size']}")

#     # Estimate time to completion
#     import os
#     import pickle
#     import pandas as pd

#     # Get site count
#     site_depths = None
#     checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
#     os.makedirs(checkpoint_dir, exist_ok=True)
    
#     if os.path.exists(os.path.join(checkpoint_dir, 'site_depths.pkl')):
#         with open(os.path.join(checkpoint_dir, 'site_depths.pkl'), 'rb') as f:
#             site_depths = pickle.load(f)

#     if site_depths is None:
#         print("Reading site data to estimate completion time...")
#         site_df = pd.read_feather(FEATHER_PATH, columns=['source', 'soil_temp_depth'])
#         valid_df = site_df.dropna(subset=['soil_temp_depth'])
#         site_count = len(valid_df.drop_duplicates(['source', 'soil_temp_depth']))
#     else:
#         site_count = len(site_depths)

#     # Check for already processed sites
#     processed_count = 0
#     if os.path.exists(os.path.join(checkpoint_dir, 'processed_indices.pkl')):
#         with open(os.path.join(checkpoint_dir, 'processed_indices.pkl'), 'rb') as f:
#             processed_indices = pickle.load(f)
#             processed_count = len(processed_indices)

#     remaining_count = site_count - processed_count

#     # Estimate completion time
#     est_time = estimate_completion_time(
#         remaining_count, 
#         settings['estimated_sites_per_second'],
#         settings['max_workers']
#     )

#     print(f"\nPipeline Settings:")
#     print(f"  Worker processes:  {settings['max_workers']}")
#     print(f"  Super batch size:  {settings['super_batch_size']}")
#     print(f"  Total sites:       {site_count}")
#     print(f"  Already processed: {processed_count}")
#     print(f"  Remaining sites:   {remaining_count}")
#     print(f"  Estimated time:    {est_time}\n")

#     if ESTIMATE_ONLY:
#         print("Estimation complete. Exiting without running pipeline.")
#         import sys
#         sys.exit(0)

#     # Run with simple confirmation
#     if remaining_count > 1000:
#         confirm = input(f"This will process {remaining_count} sites. Continue? (y/n): ")
#         if confirm.lower() != 'y':
#             print("Aborting pipeline run.")
#             import sys
#             sys.exit(0)

#     # Run the pipeline
#     print("\nStarting pipeline run...\n")
#     run_high_performance_pipeline(
#         feather_path=FEATHER_PATH,
#         output_dir=OUTPUT_DIR,
#         super_batch_size=settings['super_batch_size'],
#         max_workers=settings['max_workers'],
#         max_gap_hours=MAX_GAP_HOURS,
#         include_moisture=INCLUDE_MOISTURE
#     )

# # Directly run the high-performance pipeline without command-line arguments

# # Copy and paste the entire optimized-pipeline code here, then add this execution code at the end:

# # ---- Direct execution without command-line arguments ----
# if __name__ == "__main__":
#     # Configuration settings - EDIT THESE
#     FEATHER_PATH = '/Users/bgay/Desktop/Research/Code/merged_compressed.feather'
#     OUTPUT_DIR = 'zero_curtain_moisture_run/events'
#     MAX_WORKERS = None  # None means auto-determine based on CPU count
#     SUPER_BATCH_SIZE = None  # None means auto-determine based on system resources
#     MAX_GAP_HOURS = 8
#     INCLUDE_MOISTURE = True
#     ESTIMATE_ONLY = False  # Set to True to only estimate without running

#     # Get optimal settings
#     settings = optimize_settings()

#     # Use provided settings if specified
#     if MAX_WORKERS is not None:
#         settings['max_workers'] = MAX_WORKERS
        
#     if SUPER_BATCH_SIZE is not None:
#         settings['super_batch_size'] = SUPER_BATCH_SIZE

#     print(f"Using {settings['max_workers']} worker processes")
#     print(f"Using super-batch size of {settings['super_batch_size']}")

#     # Estimate time to completion
#     import os
#     import pickle
#     import pandas as pd

#     # Get site count
#     site_depths = None
#     checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
#     os.makedirs(checkpoint_dir, exist_ok=True)
    
#     if os.path.exists(os.path.join(checkpoint_dir, 'site_depths.pkl')):
#         with open(os.path.join(checkpoint_dir, 'site_depths.pkl'), 'rb') as f:
#             site_depths = pickle.load(f)

#     if site_depths is None:
#         print("Reading site data to estimate completion time...")
#         site_df = pd.read_feather(FEATHER_PATH, columns=['source', 'soil_temp_depth'])
#         valid_df = site_df.dropna(subset=['soil_temp_depth'])
#         site_count = len(valid_df.drop_duplicates(['source', 'soil_temp_depth']))
#     else:
#         site_count = len(site_depths)

#     # Check for already processed sites
#     processed_count = 0
#     if os.path.exists(os.path.join(checkpoint_dir, 'processed_indices.pkl')):
#         with open(os.path.join(checkpoint_dir, 'processed_indices.pkl'), 'rb') as f:
#             processed_indices = pickle.load(f)
#             processed_count = len(processed_indices)

#     remaining_count = site_count - processed_count

#     # Estimate completion time
#     est_time = estimate_completion_time(
#         remaining_count, 
#         settings['estimated_sites_per_second'],
#         settings['max_workers']
#     )

#     print(f"\nPipeline Settings:")
#     print(f"  Worker processes:  {settings['max_workers']}")
#     print(f"  Super batch size:  {settings['super_batch_size']}")
#     print(f"  Total sites:       {site_count}")
#     print(f"  Already processed: {processed_count}")
#     print(f"  Remaining sites:   {remaining_count}")
#     print(f"  Estimated time:    {est_time}\n")

#     if ESTIMATE_ONLY:
#         print("Estimation complete. Exiting without running pipeline.")
#         import sys
#         sys.exit(0)

#     # Run with simple confirmation
#     if remaining_count > 1000:
#         confirm = input(f"This will process {remaining_count} sites. Continue? (y/n): ")
#         if confirm.lower() != 'y':
#             print("Aborting pipeline run.")
#             import sys
#             sys.exit(0)

#     # Run the pipeline
#     print("\nStarting pipeline run...\n")
#     run_high_performance_pipeline_fixed(
#         feather_path=FEATHER_PATH,
#         output_dir=OUTPUT_DIR,
#         super_batch_size=settings['super_batch_size'],
#         max_workers=settings['max_workers'],
#         max_gap_hours=MAX_GAP_HOURS,
#         include_moisture=INCLUDE_MOISTURE
#     )

# # if __name__ == "__main__":
# #     import warnings
# #     warnings.filterwarnings('ignore')
    
# #     print("Starting preprocessing...")
# #     print("\nPreprocessing parameters:")
# #     print("- Chunk size: 10 sites")
# #     print("- Window size: 30 days")
# #     print("- Minimum depths: 4")
# #     print("- Depth range: -2m to 20m")
# #     print("- Missing value threshold: 30%")
    
# #     processed_chunks = preprocess_in_chunks(temp_df, zc_df, chunk_size=10)
# #     print(f"\nTotal processed windows: {len(processed_chunks)}")

def run_integrated_pipeline_with_physics(feather_path, output_dir, settings):
    """
    Integrated pipeline: zero-curtain detection followed by physics-informed analysis
    
    Parameters:
    -----------
    feather_path : str
        Path to the feather file
    output_dir : str
        Output directory for results
    settings : dict
        Optimized settings dictionary with max_workers, super_batch_size, etc.
    """
    print("="*90)
    print("INTEGRATED ZERO-CURTAIN PIPELINE WITH PHYSICS DETECTION")
    print("="*90)
    
    print(f"\nPhase 1: Traditional Zero-Curtain Detection")
    print(f"Using {settings['max_workers']} worker processes")
    print(f"Using super-batch size of {settings['super_batch_size']}")
    
    phase1_start = time.time()
    
    events = run_high_performance_pipeline_fixed(
        feather_path=feather_path,
        output_dir=output_dir,
        super_batch_size=settings['super_batch_size'],
        max_workers=settings['max_workers'],
        max_gap_hours=8,
        include_moisture=True
    )
    
    phase1_time = time.time() - phase1_start
    
    if events and len(events) > 0:
        print(f"\n{'='*90}")
        print(f"PHASE 1 COMPLETE: {len(events)} events detected")
        print(f"Phase 1 Duration: {phase1_time:.1f} seconds ({phase1_time/60:.1f} minutes)")
        print(f"{'='*90}")
        
        print(f"\nPhase 2: Physics-Informed Analysis")
        print(f"Initializing physics detection framework...")
        
        phase2_start = time.time()
        
        try:
            physics_orchestrator = PhysicsDetectionOrchestrator()
            
            print("Validating physics detection configuration...")
            if physics_orchestrator.validate_configuration():
                print("Configuration validated successfully")
                
                print("Initializing physics-informed detector...")
                physics_orchestrator.initialize_detector()
                
                print("Running physics-informed detection pipeline...")
                physics_results = physics_orchestrator.run_detection_pipeline()
                
                print("Generating summary report...")
                summary = physics_orchestrator.generate_summary_report()
                physics_orchestrator.save_summary_report(summary)
                
                phase2_time = time.time() - phase2_start
                total_time = phase1_time + phase2_time
                
                print(f"\n{'='*90}")
                print(f"PHASE 2 COMPLETE: {len(physics_results)} physics-informed events")
                print(f"Phase 2 Duration: {phase2_time:.1f} seconds ({phase2_time/60:.1f} minutes)")
                print(f"{'='*90}")
                
                print(f"\n{'='*90}")
                print(f"INTEGRATED PIPELINE COMPLETE")
                print(f"{'='*90}")
                print(f"Phase 1 Events (Traditional): {len(events)}")
                print(f"Phase 2 Events (Physics-Informed): {len(physics_results)}")
                print(f"Total Pipeline Duration: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
                print(f"{'='*90}")
                
                return events, physics_results
                
            else:
                print("\n" + "="*90)
                print("PHYSICS DETECTION CONFIGURATION VALIDATION FAILED")
                print("="*90)
                print("Required auxiliary data files not found in /path/to/user/Downloads/:")
                print("  1. UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tif (Permafrost probability)")
                print("  2. UiO_PEX_PERZONES_5.0_20181128_2000_2016_NH.shp (Permafrost zones)")
                print("  3. aa6ddc60e4ed01915fb9193bcc7f4146.nc (ERA5 snow data)")
                print("\nPhase 1 results are still available in:", output_dir)
                print("="*90)
                
                return events, None
                
        except Exception as e:
            phase2_time = time.time() - phase2_start
            
            print(f"\n{'='*90}")
            print(f"PHASE 2 ERROR")
            print(f"{'='*90}")
            print(f"Physics detection encountered an error after {phase2_time:.1f} seconds:")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nPhase 1 results are still available in:", output_dir)
            print(f"{'='*90}")
            
            return events, None
    else:
        print(f"\n{'='*90}")
        print("PHASE 1 COMPLETE: No events detected")
        print("Cannot proceed to Phase 2 without events from Phase 1")
        print(f"{'='*90}")
        return None, None


if __name__ == "__main__":
    import sys
    
    MODE_TRADITIONAL = "traditional"
    MODE_INTEGRATED = "integrated"
    
    mode = MODE_TRADITIONAL
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--integrated":
            mode = MODE_INTEGRATED
        elif sys.argv[1] == "--traditional":
            mode = MODE_TRADITIONAL
        elif sys.argv[1] == "--help":
            print("="*90)
            print("ARCTIC ZERO-CURTAIN PIPELINE")
            print("="*90)
            print("\nUsage:")
            print("  python orchestration/orchestration_module.py [--traditional|--integrated]")
            print("\nModes:")
            print("  --traditional  : Run only traditional zero-curtain detection (default)")
            print("  --integrated   : Run traditional detection + physics-informed analysis")
            print("  --help         : Show this help message")
            print("\nConfiguration:")
            print("  Edit the FEATHER_PATH and OUTPUT_DIR variables in this file")
            print("="*90)
            sys.exit(0)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
            sys.exit(1)
    
    FEATHER_PATH = '/path/to/user/Desktop/Research/Code/merged_compressed.feather'
    OUTPUT_DIR = 'zero_curtain_moisture_run/events'
    MAX_WORKERS = None
    SUPER_BATCH_SIZE = None
    MAX_GAP_HOURS = 8
    INCLUDE_MOISTURE = True
    ESTIMATE_ONLY = False
    
    settings = optimize_settings()
    
    if MAX_WORKERS is not None:
        settings['max_workers'] = MAX_WORKERS
        
    if SUPER_BATCH_SIZE is not None:
        settings['super_batch_size'] = SUPER_BATCH_SIZE
    
    print("="*90)
    print("ARCTIC ZERO-CURTAIN DETECTION PIPELINE")
    print("="*90)
    print(f"Mode: {mode.upper()}")
    print(f"Feather path: {FEATHER_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Worker processes: {settings['max_workers']}")
    print(f"Super batch size: {settings['super_batch_size']}")
    print("="*90)
    
    site_depths = None
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(checkpoint_dir, 'site_depths.pkl')):
        try:
            with open(os.path.join(checkpoint_dir, 'site_depths.pkl'), 'rb') as f:
                site_depths = pickle.load(f)
            print(f"\nLoaded checkpoint data")
        except:
            pass
    
    if site_depths is None:
        print("\nReading site data to estimate completion time...")
        site_df = pd.read_feather(FEATHER_PATH, columns=['source', 'soil_temp_depth'])
        valid_df = site_df.dropna(subset=['soil_temp_depth'])
        site_count = len(valid_df.drop_duplicates(['source', 'soil_temp_depth']))
        del site_df, valid_df
        gc.collect()
    else:
        if isinstance(site_depths, list):
            site_count = len(site_depths)
        elif isinstance(site_depths, pd.DataFrame):
            site_count = len(site_depths)
        else:
            site_count = 0
    
    processed_count = 0
    if os.path.exists(os.path.join(checkpoint_dir, 'processed_indices.pkl')):
        try:
            with open(os.path.join(checkpoint_dir, 'processed_indices.pkl'), 'rb') as f:
                processed_indices = pickle.load(f)
                processed_count = len(processed_indices)
        except:
            pass
    
    remaining_count = site_count - processed_count
    
    est_time = estimate_completion_time(
        remaining_count, 
        settings['estimated_sites_per_second'],
        settings['max_workers']
    )
    
    print(f"\nPipeline Estimates:")
    print(f"  Total sites:       {site_count:,}")
    print(f"  Already processed: {processed_count:,}")
    print(f"  Remaining sites:   {remaining_count:,}")
    print(f"  Estimated time:    {est_time}")
    print("="*90)
    
    if ESTIMATE_ONLY:
        print("\nEstimation complete. Exiting without running pipeline.")
        sys.exit(0)
    
    if remaining_count > 1000:
        confirm = input(f"\nThis will process {remaining_count:,} sites. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Pipeline execution aborted by user.")
            sys.exit(0)
    
    print(f"\nStarting pipeline in {mode.upper()} mode...\n")
    
    pipeline_start_time = time.time()
    
    if mode == MODE_INTEGRATED:
        trad_events, phys_events = run_integrated_pipeline_with_physics(
            feather_path=FEATHER_PATH,
            output_dir=OUTPUT_DIR,
            settings=settings
        )
    else:
        print("="*90)
        print("TRADITIONAL ZERO-CURTAIN DETECTION")
        print("="*90)
        
        trad_events = run_high_performance_pipeline_fixed(
            feather_path=FEATHER_PATH,
            output_dir=OUTPUT_DIR,
            super_batch_size=settings['super_batch_size'],
            max_workers=settings['max_workers'],
            max_gap_hours=MAX_GAP_HOURS,
            include_moisture=INCLUDE_MOISTURE
        )
        
        pipeline_time = time.time() - pipeline_start_time
        
        print(f"\n{'='*90}")
        print("PIPELINE COMPLETE")
        print(f"{'='*90}")
        if trad_events:
            print(f"Total events detected: {len(trad_events):,}")
        else:
            print("No events detected")
        print(f"Total duration: {pipeline_time:.1f} seconds ({pipeline_time/60:.1f} minutes)")
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"{'='*90}")
    
    print("\nPipeline execution finished.")