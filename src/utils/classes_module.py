"""
Classes module for Arctic zero-curtain pipeline.
Auto-generated from Jupyter notebook.
"""

from src.utils.imports import *
from src.utils.utilities import *

# @contextmanager
# def measure_time(description="Operation"):
#     """Context manager to measure and print execution time of code blocks."""
#     start_time = time.time()
#     try:
#         yield
#     finally:
#         elapsed = time.time() - start_time
#         print(f"{description} completed in {elapsed:.2f} seconds")

# class MemoryMonitor:
#     """Class for monitoring memory usage in a separate thread."""
#     def __init__(self, interval=10, threshold=75):
#         self.interval = interval
#         self.threshold = threshold
#         self.stop_flag = threading.Event()
#         self.max_usage = 0
#         self.monitor_thread = None
        
#     def memory_usage(self):
#         """Get current memory usage percentage."""
#         return psutil.virtual_memory().percent
        
#     def monitor(self):
#         """Monitor memory usage periodically."""
#         while not self.stop_flag.is_set():
#             usage = self.memory_usage()
#             self.max_usage = max(self.max_usage, usage)
#             if usage > self.threshold:
#                 print(f"WARNING: High memory usage detected: {usage:.1f}%")
#             time.sleep(self.interval)
                
#     def start(self):
#         """Start the memory monitoring thread."""
#         self.monitor_thread = threading.Thread(target=self.monitor, daemon=True)
#         self.monitor_thread.start()
        
#     def stop(self):
#         """Stop the memory monitoring thread."""
#         self.stop_flag.set()
#         if self.monitor_thread:
#             self.monitor_thread.join(timeout=1.0)
#         print(f"Maximum memory usage: {self.max_usage:.1f}%")

# class CheckpointManager:
#     """Class for managing checkpoints and metadata."""
#     def __init__(self, base_path, metadata_path, save_frequency=10):
#         self.base_path = base_path
#         self.metadata_path = metadata_path
#         self.save_frequency = save_frequency
#         self.completed_batches = self.load_metadata()
        
#     def save_checkpoint(self, batch_data, batch_index):
#         """Save a batch checkpoint to disk."""
#         filepath = f"{self.base_path}_batch_{batch_index}.pkl"
#         with open(filepath, "wb") as f:
#             pickle.dump(batch_data, f)
        
#     def load_checkpoint(self, batch_index):
#         """Load a batch checkpoint from disk."""
#         filepath = f"{self.base_path}_batch_{batch_index}.pkl"
#         if os.path.exists(filepath):
#             with open(filepath, "rb") as f:
#                 return pickle.load(f)
#         return None
        
#     def save_metadata(self, force_save=False):
#         """Save checkpoint metadata to disk."""
#         if force_save or len(self.completed_batches) % self.save_frequency == 0:
#             with open(self.metadata_path, "w") as f:
#                 json.dump({"completed_batches": sorted(list(self.completed_batches))}, f)
                
#     def load_metadata(self):
#         """Load checkpoint metadata from disk."""
#         if os.path.exists(self.metadata_path):
#             with open(self.metadata_path, "r") as f:
#                 return set(json.load(f).get("completed_batches", []))
#         return set()
        
#     def is_batch_completed(self, batch_index):
#         """Check if a batch has been completed."""
#         return batch_index in self.completed_batches
        
#     def mark_batch_completed(self, batch_index):
#         """Mark a batch as completed."""
#         self.completed_batches.add(batch_index)
#         self.save_metadata()

# def haversine_to_cartesian(lat, lon):
#     """Convert latitude and longitude to 3D cartesian coordinates (x,y,z)."""
#     # Convert to radians
#     lat_rad = np.radians(lat)
#     lon_rad = np.radians(lon)
    
#     # Convert to cartesian coordinates on a unit sphere
#     x = np.cos(lat_rad) * np.cos(lon_rad)
#     y = np.cos(lat_rad) * np.sin(lon_rad)
#     z = np.sin(lat_rad)
    
#     return np.column_stack((x, y, z))

# def cartesian_to_haversine_distance(distance_cartesian):
#     """Convert cartesian distance to angular distance in radians."""
#     # Ensure distance is at most 2.0 (diameter of unit sphere)
#     distance_cartesian = np.minimum(distance_cartesian, 2.0)
    
#     # Convert chord length to angular distance using inverse haversine formula
#     return 2 * np.arcsin(distance_cartesian / 2)

# def process_batch(tree, batch_points, k):
#     """Process a single batch of points using KDTree."""
#     distances, _ = tree.query(batch_points, k=k)
#     # Convert cartesian distances to angular distances (radians)
#     return cartesian_to_haversine_distance(distances)

# def calculate_optimal_batch_size(total_points, available_memory_mb=None):
#     """Calculate optimal batch size based on available memory and dataset size."""
#     if available_memory_mb is None:
#         # Use 20% of available memory if not specified (to leave room for parallelism)
#         available_memory_mb = psutil.virtual_memory().available / (1024 * 1024) * 0.2
    
#     # Estimate memory per point (assuming float64 coordinates and distances)
#     memory_per_point_kb = 0.5  # Approximate memory per point in KB
    
#     # Calculate batch size that would use available memory
#     batch_size = int(available_memory_mb * 1024 / memory_per_point_kb)
    
#     # Cap batch size to reasonable limits
#     min_batch_size = 1000
#     max_batch_size = 50000
#     batch_size = max(min(batch_size, max_batch_size), min_batch_size)
    
#     return min(batch_size, total_points)

# def aggregate_checkpoints(checkpoint_mgr, batch_indices):
#     """Load and aggregate checkpoints for the specified batch indices."""
#     all_distances = []
#     for batch_idx in batch_indices:
#         distances = checkpoint_mgr.load_checkpoint(batch_idx)
#         if distances is not None:
#             all_distances.append(distances)
#     return all_distances

# def process_spatial_data(latitudes, longitudes, k=5, leaf_size=40, batch_size=None, n_jobs=-1):
#     """
#     Process spatial data using KDTree for nearest neighbor calculations.
    
#     Parameters:
#     -----------
#     latitudes : array-like
#         Latitude values in degrees
#     longitudes : array-like
#         Longitude values in degrees
#     k : int
#         Number of nearest neighbors to find
#     leaf_size : int
#         Leaf size for the KDTree (affects performance)
#     batch_size : int or None
#         Batch size for processing. If None, will be calculated automatically.
#     n_jobs : int
#         Number of parallel jobs to run. -1 means using all processors.
    
#     Returns:
#     --------
#     distances : list
#         List of distances to k nearest neighbors for each point (in radians)
#     """
#     # Convert input arrays to numpy if they aren't already
#     lat_array = np.asarray(latitudes, dtype=np.float64)
#     lon_array = np.asarray(longitudes, dtype=np.float64)
    
#     # Convert to 3D cartesian coordinates for better KDTree performance
#     with measure_time("Coordinate conversion to cartesian"):
#         cartesian_points = haversine_to_cartesian(lat_array, lon_array)
    
#     # Calculate optimal batch size if not provided
#     n_points = len(cartesian_points)
#     if batch_size is None:
#         batch_size = calculate_optimal_batch_size(n_points)
    
#     # Adjust batch size to create a reasonable number of batches (target ~200 batches)
#     if n_points / batch_size > 200:
#         batch_size = max(batch_size, n_points // 200)
    
#     print(f"Using batch size: {batch_size} for {n_points} points")
    
#     # Initialize checkpoint manager
#     checkpoint_mgr = CheckpointManager(
#         CHECKPOINT_BASE, 
#         METADATA_PATH,
#         METADATA_SAVE_FREQUENCY
#     )
    
#     # Start memory monitoring
#     memory_monitor = MemoryMonitor(interval=MONITOR_INTERVAL, threshold=MEMORY_THRESHOLD)
#     memory_monitor.start()
    
#     try:
#         # Build cKDTree for fast spatial queries (much faster than BallTree)
#         with measure_time("KDTree construction"):
#             tree = cKDTree(cartesian_points, leafsize=leaf_size)
#             gc.collect()  # Force garbage collection
#             print(f"Available memory after tree creation: {psutil.virtual_memory().available / (10...
        
#         # Calculate number of batches
#         num_batches = int(np.ceil(n_points / batch_size))
#         print(f"Total number of batches: {num_batches}")
        
#         # Check which batches are already completed
#         completed_batches = set()
#         for batch_index in range(num_batches):
#             if checkpoint_mgr.is_batch_completed(batch_index):
#                 completed_batches.add(batch_index)
                
#         print(f"Found {len(completed_batches)} completed batches")
        
#         # Process remaining batches
#         remaining_batches = set(range(num_batches)) - completed_batches
#         print(f"Processing {len(remaining_batches)} remaining batches")
        
#         if remaining_batches:
#             # Convert to list and sort for deterministic processing
#             remaining_batch_indices = sorted(list(remaining_batches))
            
#             # Test with a small batch first
#             print("Testing KDTree query with a small sample...")
#             sample_idx = remaining_batch_indices[0]
#             start_idx = sample_idx * batch_size
#             end_idx = min((sample_idx + 1) * batch_size, n_points)
            
#             # Take just 10 points for testing
#             test_points = cartesian_points[start_idx:start_idx+10]
            
#             with measure_time("Test KDTree query"):
#                 test_distances, _ = tree.query(test_points, k=k)
#                 test_distances = cartesian_to_haversine_distance(test_distances)
            
#             print(f"Test successful! Sample distances: {test_distances[0]}")
#             print(f"Average test distance: {np.mean(test_distances):.6f} radians")
            
#             # Process batches in parallel or sequentially
#             if n_jobs != 1:
#                 # Process in parallel using joblib
#                 print(f"Processing batches in parallel with {n_jobs} jobs")
                
#                 # Process in smaller chunks to avoid memory issues
#                 chunk_size = min(20, len(remaining_batch_indices))  # Smaller chunks for better mo...
#                 num_chunks = int(np.ceil(len(remaining_batch_indices) / chunk_size))
                
#                 for chunk_idx in range(num_chunks):
#                     chunk_start = chunk_idx * chunk_size
#                     chunk_end = min((chunk_idx + 1) * chunk_size, len(remaining_batch_indices))
#                     current_chunk = remaining_batch_indices[chunk_start:chunk_end]
                    
#                     print(f"Processing chunk {chunk_idx + 1}/{num_chunks} with {len(current_chunk)...
                    
#                     # Prepare batch data
#                     batch_data = []
#                     for batch_idx in current_chunk:
#                         start_idx = batch_idx * batch_size
#                         end_idx = min((batch_idx + 1) * batch_size, n_points)
#                         batch_points = cartesian_points[start_idx:end_idx]
#                         batch_data.append((batch_idx, batch_points))
                    
#                     # Process in parallel with timeout monitoring
#                     start_time = time.time()
#                     results = Parallel(n_jobs=n_jobs, verbose=10, timeout=3600)(
#                         delayed(process_batch)(tree, points, k) for batch_idx, points in batch_dat...
#                     )
#                     end_time = time.time()
                    
#                     # Calculate performance statistics
#                     elapsed_time = end_time - start_time
#                     points_processed = sum(len(points) for _, points in batch_data)
#                     points_per_second = points_processed / elapsed_time
                    
#                     print(f"Chunk processed {points_processed} points in {elapsed_time:.2f} second...
#                     print(f"Performance: {points_per_second:.2f} points/second")
                    
#                     # Save results
#                     for (batch_idx, _), distances in zip(batch_data, results):
#                         checkpoint_mgr.save_checkpoint(distances, batch_idx)
#                         checkpoint_mgr.mark_batch_completed(batch_idx)
                    
#                     # Estimate remaining time
#                     remaining_chunks = num_chunks - (chunk_idx + 1)
#                     if remaining_chunks > 0:
#                         remaining_time = remaining_chunks * elapsed_time
#                         hours = remaining_time // 3600
#                         minutes = (remaining_time % 3600) // 60
#                         print(f"Estimated remaining time: {int(hours)}h {int(minutes)}m")
                    
#                     # Free memory
#                     del batch_data
#                     del results
#                     gc.collect()
#             else:
#                 # Process sequentially
#                 print("Processing batches sequentially")
#                 with tqdm(total=len(remaining_batch_indices), desc="Processing Batches") as pbar:
#                     for batch_idx in remaining_batch_indices:
#                         start_idx = batch_idx * batch_size
#                         end_idx = min((batch_idx + 1) * batch_size, n_points)
#                         batch_points = cartesian_points[start_idx:end_idx]
                        
#                         # Process full batch
#                         distances = process_batch(tree, batch_points, k)
                        
#                         # Save checkpoint
#                         checkpoint_mgr.save_checkpoint(distances, batch_idx)
#                         checkpoint_mgr.mark_batch_completed(batch_idx)
                        
#                         # Free memory
#                         del distances
#                         gc.collect()
                        
#                         pbar.update(1)
        
#         # Combine results from all batches
#         with measure_time("Loading checkpoints"):
#             print("Loading and combining checkpoints...")
            
#             # Load checkpoints in chunks to avoid memory issues
#             all_batches = sorted(list(range(num_batches)))
#             chunk_size = min(1000, num_batches)  # Adjust based on available memory
#             num_chunks = int(np.ceil(len(all_batches) / chunk_size))
            
#             all_distances = []
            
#             for chunk_idx in range(num_chunks):
#                 chunk_start = chunk_idx * chunk_size
#                 chunk_end = min((chunk_idx + 1) * chunk_size, len(all_batches))
#                 current_chunk = all_batches[chunk_start:chunk_end]
                
#                 print(f"Loading checkpoint chunk {chunk_idx + 1}/{num_chunks} ({len(current_chunk)...
#                 chunk_distances = aggregate_checkpoints(checkpoint_mgr, current_chunk)
                
#                 if chunk_distances:
#                     all_distances.extend(chunk_distances)
            
#         with measure_time("Combining results"):
#             print(f"Combining {len(all_distances)} checkpoint results...")
#             combined_distances = np.vstack(all_distances) if all_distances else np.array([])
            
#         # Final metadata save
#         checkpoint_mgr.save_metadata(force_save=True)
        
#         return combined_distances
        
#     finally:
#         # Stop memory monitoring
#         memory_monitor.stop()

import os
import gc
import json
import pickle
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import psutil
import threading
import time
from contextlib import contextmanager
from joblib import Parallel, delayed

# Configuration parameters
CHECKPOINT_DIR = "zero_curtain_pipeline/modeling/checkpoints"
MEMORY_THRESHOLD = 75  # Memory usage percentage threshold
METADATA_SAVE_FREQUENCY = 5  # Save metadata every N batches
MONITOR_INTERVAL = 10  # Memory monitoring interval in seconds

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_BASE = os.path.join(CHECKPOINT_DIR, "geo_density_checkpoint")
METADATA_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint_metadata.json")

@contextmanager
def measure_time(description="Operation"):
    """Context manager to measure and print execution time of code blocks."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        print(f"{description} completed in {elapsed:.2f} seconds")

class MemoryMonitor:
    """Class for monitoring memory usage in a separate thread."""
    def __init__(self, interval=10, threshold=75):
        self.interval = interval
        self.threshold = threshold
        self.stop_flag = threading.Event()
        self.max_usage = 0
        self.monitor_thread = None
        
    def memory_usage(self):
        """Get current memory usage percentage."""
        return psutil.virtual_memory().percent
        
    def monitor(self):
        """Monitor memory usage periodically."""
        while not self.stop_flag.is_set():
            usage = self.memory_usage()
            self.max_usage = max(self.max_usage, usage)
            if usage > self.threshold:
                print(f"WARNING: High memory usage detected: {usage:.1f}%")
            time.sleep(self.interval)
                
    def start(self):
        """Start the memory monitoring thread."""
        self.monitor_thread = threading.Thread(target=self.monitor, daemon=True)
        self.monitor_thread.start()
        
    def stop(self):
        """Stop the memory monitoring thread."""
        self.stop_flag.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print(f"Maximum memory usage: {self.max_usage:.1f}%")

class CheckpointManager:
    """Class for managing checkpoints and metadata."""
    def __init__(self, base_path, metadata_path, save_frequency=5):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.save_frequency = save_frequency
        self.completed_batches = self.load_metadata()
        
    def save_checkpoint(self, batch_data, batch_index):
        """Save a batch checkpoint to disk."""
        filepath = f"{self.base_path}_batch_{batch_index}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(batch_data, f)
        
    def load_checkpoint(self, batch_index):
        """Load a batch checkpoint from disk."""
        filepath = f"{self.base_path}_batch_{batch_index}.pkl"
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)
        return None
        
    def save_metadata(self, force_save=False):
        """Save checkpoint metadata to disk."""
        if force_save or len(self.completed_batches) % self.save_frequency == 0:
            with open(self.metadata_path, "w") as f:
                json.dump({"completed_batches": sorted(list(self.completed_batches))}, f)
                
    def load_metadata(self):
        """Load checkpoint metadata from disk."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                return set(json.load(f).get("completed_batches", []))
        return set()
        
    def is_batch_completed(self, batch_index):
        """Check if a batch has been completed."""
        return batch_index in self.completed_batches
        
    def mark_batch_completed(self, batch_index):
        """Mark a batch as completed."""
        self.completed_batches.add(batch_index)
        self.save_metadata()

def haversine_to_cartesian(lat, lon):
    """Convert latitude and longitude to 3D cartesian coordinates (x,y,z)."""
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Convert to cartesian coordinates on a unit sphere
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    return np.column_stack((x, y, z))

def cartesian_to_haversine_distance(distance_cartesian):
    """Convert cartesian distance to angular distance in radians."""
    # Ensure distance is at most 2.0 (diameter of unit sphere)
    distance_cartesian = np.minimum(distance_cartesian, 2.0)
    
    # Convert chord length to angular distance using inverse haversine formula
    return 2 * np.arcsin(distance_cartesian / 2)

def process_batch(tree, batch_points, k):
    """Process a single batch of points using KDTree, excluding self-matches."""
    # Query k+1 neighbors to include self (which will be removed)
    distances, _ = tree.query(batch_points, k=k+1)
    
    # Remove the first neighbor (self with distance 0)
    nn_distances = distances[:, 1:]
    
    # Convert cartesian distances to angular distances (radians)
    return cartesian_to_haversine_distance(nn_distances)

def calculate_optimal_batch_size(total_points, available_memory_mb=None):
    """Calculate optimal batch size based on available memory and dataset size."""
    if available_memory_mb is None:
        # Use 20% of available memory if not specified (to leave room for parallelism)
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024) * 0.2
    
    # Estimate memory per point (assuming float64 coordinates and distances)
    memory_per_point_kb = 0.5  # Approximate memory per point in KB
    
    # Calculate batch size that would use available memory
    batch_size = int(available_memory_mb * 1024 / memory_per_point_kb)
    
    # Cap batch size to reasonable limits
    min_batch_size = 1000
    max_batch_size = 50000
    batch_size = max(min(batch_size, max_batch_size), min_batch_size)
    
    return min(batch_size, total_points)

def aggregate_checkpoints(checkpoint_mgr, batch_indices):
    """Load and aggregate checkpoints for the specified batch indices."""
    all_distances = []
    for batch_idx in batch_indices:
        distances = checkpoint_mgr.load_checkpoint(batch_idx)
        if distances is not None:
            all_distances.append(distances)
    return all_distances

def calculate_spatial_density(latitudes, longitudes, k=5, leaf_size=40, batch_size=None, n_jobs=4, 
                             checkpoint_dir=CHECKPOINT_DIR, overwrite=False):
    """
    Calculate spatial density based on K-nearest neighbors with checkpointing.
    
    Parameters:
    -----------
    latitudes : array-like
        Latitude values in degrees
    longitudes : array-like
        Longitude values in degrees
    k : int
        Number of nearest neighbors to find (excluding self)
    leaf_size : int
        Leaf size for the KDTree (affects performance)
    batch_size : int or None
        Batch size for processing. If None, will be calculated automatically.
    n_jobs : int
        Number of parallel jobs to run. -1 means using all processors.
    checkpoint_dir : str
        Directory for saving checkpoints
    overwrite : bool
        Whether to overwrite existing checkpoints
    
    Returns:
    --------
    density : array
        Density values for each point
    distances : array
        Distances to k nearest neighbors, shape (n_points, k)
    """
    global CHECKPOINT_DIR
    CHECKPOINT_DIR = checkpoint_dir
    global CHECKPOINT_BASE
    CHECKPOINT_BASE = os.path.join(CHECKPOINT_DIR, "geo_density_checkpoint")
    global METADATA_PATH
    METADATA_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint_metadata.json")
    
    # Never overwrite existing checkpoints - we need them to resume processing
    if overwrite:
        print("WARNING: Overwrite flag ignored - checkpoints are preserved for resuming processing")
        # Do NOT remove checkpoints regardless of the overwrite flag
    
    # Convert input arrays to numpy if they aren't already
    lat_array = np.asarray(latitudes, dtype=np.float64)
    lon_array = np.asarray(longitudes, dtype=np.float64)
    
    # Convert to 3D cartesian coordinates for better KDTree performance
    with measure_time("Coordinate conversion to cartesian"):
        cartesian_points = haversine_to_cartesian(lat_array, lon_array)
    
    # Calculate optimal batch size if not provided
    n_points = len(cartesian_points)
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(n_points)
    
    # Adjust batch size to create a reasonable number of batches (target ~200 batches)
    if n_points / batch_size > 200:
        batch_size = max(batch_size, n_points // 200)
    
    print(f"Using batch size: {batch_size} for {n_points} points")
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(
        CHECKPOINT_BASE, 
        METADATA_PATH,
        METADATA_SAVE_FREQUENCY
    )
    
    # Start memory monitoring
    memory_monitor = MemoryMonitor(interval=MONITOR_INTERVAL, threshold=MEMORY_THRESHOLD)
    memory_monitor.start()
    
    try:
        # Build cKDTree for fast spatial queries (much faster than BallTree)
        with measure_time("KDTree construction"):
            tree = cKDTree(cartesian_points, leafsize=leaf_size)
            gc.collect()  # Force garbage collection
            print(f"Available memory after tree creation: {psutil.virtual_memory().available / (1024*1024*1024):.2f} GB")
        
        # Calculate number of batches
        num_batches = int(np.ceil(n_points / batch_size))
        print(f"Total number of batches: {num_batches}")
        
        # Check which batches are already completed
        completed_batches = set()
        for batch_index in range(num_batches):
            if checkpoint_mgr.is_batch_completed(batch_index):
                completed_batches.add(batch_index)
                
        print(f"Found {len(completed_batches)} completed batches")
        
        # Process remaining batches
        remaining_batches = set(range(num_batches)) - completed_batches
        print(f"Processing {len(remaining_batches)} remaining batches")
        
        if remaining_batches:
            # Convert to list and sort for deterministic processing
            remaining_batch_indices = sorted(list(remaining_batches))
            
            # Test with a small batch first
            print("Testing KDTree query with a small sample...")
            sample_idx = remaining_batch_indices[0]
            start_idx = sample_idx * batch_size
            end_idx = min((sample_idx + 1) * batch_size, n_points)
            
            # Take just 10 points for testing
            test_points = cartesian_points[start_idx:start_idx+10]
            
            with measure_time("Test KDTree query"):
                # Query k+1 neighbors (including self) for test
                test_distances, _ = tree.query(test_points, k=k+1)
                # Remove first column (self-matches)
                test_distances = test_distances[:, 1:]
                # Convert to angular distances
                test_distances = cartesian_to_haversine_distance(test_distances)
            
            print(f"Test successful! Sample distances: {test_distances[0]}")
            print(f"Average test distance: {np.mean(test_distances):.6f} radians")
            
            # Process batches in parallel or sequentially
            if n_jobs != 1:
                # Process in parallel using joblib
                print(f"Processing batches in parallel with {n_jobs} jobs")
                
                # Process in smaller chunks to avoid memory issues
                chunk_size = min(20, len(remaining_batch_indices))  # Smaller chunks for better monitoring
                num_chunks = int(np.ceil(len(remaining_batch_indices) / chunk_size))
                
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min((chunk_idx + 1) * chunk_size, len(remaining_batch_indices))
                    current_chunk = remaining_batch_indices[chunk_start:chunk_end]
                    
                    print(f"Processing chunk {chunk_idx + 1}/{num_chunks} with {len(current_chunk)} batches")
                    
                    # Prepare batch data
                    batch_data = []
                    for batch_idx in current_chunk:
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, n_points)
                        batch_points = cartesian_points[start_idx:end_idx]
                        batch_data.append((batch_idx, batch_points))
                    
                    # Process in parallel with timeout monitoring
                    start_time = time.time()
                    results = Parallel(n_jobs=n_jobs, verbose=10, timeout=3600)(
                        delayed(process_batch)(tree, points, k) for batch_idx, points in batch_data
                    )
                    end_time = time.time()
                    
                    # Calculate performance statistics
                    elapsed_time = end_time - start_time
                    points_processed = sum(len(points) for _, points in batch_data)
                    points_per_second = points_processed / elapsed_time
                    
                    print(f"Chunk processed {points_processed} points in {elapsed_time:.2f} seconds")
                    print(f"Performance: {points_per_second:.2f} points/second")
                    
                    # Save results
                    for (batch_idx, _), distances in zip(batch_data, results):
                        checkpoint_mgr.save_checkpoint(distances, batch_idx)
                        checkpoint_mgr.mark_batch_completed(batch_idx)
                    
                    # Estimate remaining time
                    remaining_chunks = num_chunks - (chunk_idx + 1)
                    if remaining_chunks > 0:
                        remaining_time = remaining_chunks * elapsed_time
                        hours = remaining_time // 3600
                        minutes = (remaining_time % 3600) // 60
                        print(f"Estimated remaining time: {int(hours)}h {int(minutes)}m")
                    
                    # Free memory
                    del batch_data
                    del results
                    gc.collect()
            else:
                # Process sequentially
                print("Processing batches sequentially")
                with tqdm(total=len(remaining_batch_indices), desc="Processing Batches") as pbar:
                    for batch_idx in remaining_batch_indices:
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, n_points)
                        batch_points = cartesian_points[start_idx:end_idx]
                        
                        # Process full batch
                        distances = process_batch(tree, batch_points, k)
                        
                        # Save checkpoint
                        checkpoint_mgr.save_checkpoint(distances, batch_idx)
                        checkpoint_mgr.mark_batch_completed(batch_idx)
                        
                        # Free memory
                        del distances
                        gc.collect()
                        
                        pbar.update(1)
        
        # At this point, all batches should be processed and saved to disk
        # We have two options:
        # 1. Load all checkpoints into memory (might be too large)
        # 2. Calculate density from checkpoints directly (more memory efficient)
        
        # Option 2: Calculate density directly from checkpoints
        print("Calculating density from checkpoints...")
        n_processed = 0
        density = np.zeros(n_points)
        
        # Process in chunks to save memory
        chunk_size = min(1000, num_batches)
        num_chunks = int(np.ceil(num_batches / chunk_size))
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, num_batches)
            current_chunk = list(range(chunk_start, chunk_end))
            
            print(f"Processing density chunk {chunk_idx+1}/{num_chunks}")
            
            # Load each batch and compute density
            for batch_idx in tqdm(current_chunk):
                batch_distances = checkpoint_mgr.load_checkpoint(batch_idx)
                
                if batch_distances is not None:
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, n_points)
                    
                    # Mean distance to k nearest neighbors
                    mean_distances = np.mean(batch_distances, axis=1)
                    
                    # Density is inversely proportional to mean distance
                    # Add small epsilon to avoid division by zero
                    epsilon = 1e-10
                    density[start_idx:end_idx] = 1.0 / (mean_distances + epsilon)
                    
                    n_processed += end_idx - start_idx
                    
                    # Clean up
                    del batch_distances, mean_distances
                    gc.collect()
        
        print(f"Processed density for {n_processed} points")
        
        # Normalize density for easier interpretation
        if n_processed > 0:
            density = density / np.mean(density[0:n_processed])
        
        # Create weights (inverse of density)
        weights = 1.0 / (density + 1e-8)
        
        # Normalize weights
        weights = weights / np.mean(weights) * n_points
        
        # Clip extreme weights (beyond 3 std from mean)
        weights_mean = np.mean(weights)
        weights_std = np.std(weights)
        weights = np.clip(weights, 0, weights_mean + 3*weights_std)
        
        # Final normalization
        weights = weights / np.mean(weights) * n_points
        
        # Save the final density and weights
        with open(os.path.join(CHECKPOINT_DIR, "spatial_density.pkl"), "wb") as f:
            pickle.dump({"density": density, "weights": weights}, f)
        
        print(f"Saved final density and weights to {os.path.join(CHECKPOINT_DIR, 'spatial_density.pkl')}")
        
        # Final metadata save
        checkpoint_mgr.save_metadata(force_save=True)
        
        return density, weights
        
    finally:
        # Stop memory monitoring
        memory_monitor.stop()

def stratified_spatiotemporal_split(X, y, metadata, test_size=0.2, val_size=0.15, 
                                   random_state=42, checkpoint_dir=CHECKPOINT_DIR):
    """
    Split data with balanced spatiotemporal representation in train/val/test sets.
    
    Parameters:
    -----------
    X : array
        Features array
    y : array
        Labels array
    metadata : list
        List of metadata dicts containing spatial and temporal information
    test_size : float
        Fraction of data for testing
    val_size : float
        Fraction of data for validation
    random_state : int
        Random seed
    checkpoint_dir : str
        Directory for checkpoints
        
    Returns:
    --------
    train_indices, val_indices, test_indices : arrays
        Indices for each split
    """
    print("Performing spatiotemporal split with checkpointing...")
    global CHECKPOINT_DIR
    CHECKPOINT_DIR = checkpoint_dir
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # ALWAYS check for existing splits to resume from previous run
    split_file = os.path.join(CHECKPOINT_DIR, "spatiotemporal_split.pkl")
    if os.path.exists(split_file):
        print(f"Loading existing split from {split_file}")
        with open(split_file, "rb") as f:
            split_data = pickle.load(f)
        print(f"Successfully loaded existing train/val/test split with {len(split_data['train_indices'])} training, " +
              f"{len(split_data['val_indices'])} validation, and {len(split_data['test_indices'])} test samples")
        return split_data["train_indices"], split_data["val_indices"], split_data["test_indices"]
    
    # If we get here, no existing split was found, so we'll create a new one
    print("No existing split found, creating new train/val/test split")
    
    n_samples = len(X)
    np.random.seed(random_state)
    
    # Extract relevant metadata
    timestamps = np.array([meta['start_time'] for meta in metadata])
    
    # Geography (handle cases where lat/lon may be missing)
    latitudes = np.array([
        meta.get('latitude', 0) if meta.get('latitude') is not None else 0 
        for meta in metadata
    ])
    longitudes = np.array([
        meta.get('longitude', 0) if meta.get('longitude') is not None else 0 
        for meta in metadata
    ])
    depths = np.array([
        meta.get('soil_temp_depth', 0) if meta.get('soil_temp_depth') is not None else 0 
        for meta in metadata
    ])
    
    has_geo_info = (np.count_nonzero(latitudes) > 0 and np.count_nonzero(longitudes) > 0)
    
    # Time-based sorting
    time_indices = np.argsort(timestamps)
    
    # Keep the most recent data as a true test set (chronological split)
    test_count = int(n_samples * test_size)
    test_indices = time_indices[-test_count:]
    
    # Remaining data for train/val
    remaining_indices = time_indices[:-test_count]
    
    # Check for existing density weights
    density_file = os.path.join(CHECKPOINT_DIR, "spatial_density.pkl")
    if os.path.exists(density_file) and has_geo_info:
        print(f"Loading existing density weights from {density_file}")
        with open(density_file, "rb") as f:
            density_data = pickle.load(f)
        weights = density_data["weights"][remaining_indices]
    elif has_geo_info:
        # Calculate density-based weights for remaining data
        print("Calculating spatial density for weighting...")
        _, weights = calculate_spatial_density(
            latitudes[remaining_indices],
            longitudes[remaining_indices],
            k=5,
            checkpoint_dir=CHECKPOINT_DIR
        )
    else:
        # If no geography, use uniform weights
        weights = np.ones(len(remaining_indices))
    
    # Calculate validation size
    val_count = int(n_samples * val_size)
    
    # Stratification criteria
    print("Creating stratification features...")
    
    # 1. Time-based features
    # Extract year, month, day features
    years = np.array([ts.year for ts in timestamps[remaining_indices]])
    months = np.array([ts.month for ts in timestamps[remaining_indices]])
    # Group months into seasons
    seasons = np.floor((months - 1) / 3).astype(int)
    
    # 2. Spatial features
    if has_geo_info:
        # Latitude bands
        lat_bands = np.zeros_like(remaining_indices, dtype=int)
        lat_bands[(latitudes[remaining_indices] >= 50) & (latitudes[remaining_indices] < 60)] = 1
        lat_bands[(latitudes[remaining_indices] >= 60) & (latitudes[remaining_indices] < 66.5)] = 2
        lat_bands[(latitudes[remaining_indices] >= 66.5) & (latitudes[remaining_indices] < 75)] = 3
        lat_bands[(latitudes[remaining_indices] >= 75)] = 4
        
        # Longitude sectors (8 sectors of 45 degrees each)
        lon_sectors = ((longitudes[remaining_indices] + 180) / 45).astype(int) % 8
    else:
        # Dummy spatial features if no geographic data
        lat_bands = np.zeros_like(remaining_indices)
        lon_sectors = np.zeros_like(remaining_indices)
    
    # 3. Depth strata
    depth_strata = np.zeros_like(remaining_indices)
    depth_strata[(depths[remaining_indices] > 0) & (depths[remaining_indices] <= 0.2)] = 1
    depth_strata[(depths[remaining_indices] > 0.2) & (depths[remaining_indices] <= 0.5)] = 2
    depth_strata[(depths[remaining_indices] > 0.5) & (depths[remaining_indices] <= 1.0)] = 3
    depth_strata[(depths[remaining_indices] > 1.0)] = 4
    
    # 4. Class labels
    labels = y[remaining_indices]
    
    # Create strata by combining features
    strata = (
        years * 10000 + 
        seasons * 1000 + 
        lat_bands * 100 + 
        lon_sectors * 10 + 
        depth_strata
    )
    
    # If labels are binary, include them in strata
    if len(np.unique(labels)) <= 5:  # Few enough classes to use for stratification
        strata = strata * 10 + labels
    
    unique_strata = np.unique(strata)
    
    print(f"Found {len(unique_strata)} unique strata")
    
    # Initialize val indices with checkpoint
    val_indices_file = os.path.join(CHECKPOINT_DIR, "val_indices_temp.pkl")
    if os.path.exists(val_indices_file):
        print(f"Loading partial validation indices from {val_indices_file}")
        with open(val_indices_file, "rb") as f:
            val_indices = pickle.load(f)
        
        # Remove already sampled from potential pool
        sampled_mask = np.ones(len(remaining_indices), dtype=bool)
        sampled_indices = np.where(np.isin(remaining_indices, val_indices))[0]
        sampled_mask[sampled_indices] = False
        
        # Update sampling pool
        remaining_pool = np.arange(len(remaining_indices))[sampled_mask]
        
        print(f"Loaded {len(val_indices)} validation indices, {len(remaining_pool)} remaining to sample")
    else:
        val_indices = []
        remaining_pool = np.arange(len(remaining_indices))
    
    # Sample from each stratum proportionally to create validation set
    checkpoint_frequency = 10  # Save every 10 strata
    for i, stratum in enumerate(tqdm(unique_strata, desc="Sampling validation set")):
        # Find indices for this stratum
        stratum_positions = np.where((strata == stratum) & np.isin(np.arange(len(remaining_indices)), remaining_pool))[0]
        
        if len(stratum_positions) == 0:
            continue
            
        stratum_indices = remaining_indices[stratum_positions]
        stratum_weights = weights[stratum_positions]
        
        # Calculate target sample size
        stratum_weight_sum = np.sum(stratum_weights)
        total_weight_sum = np.sum(weights)
        target_val_size = int(val_count * stratum_weight_sum / total_weight_sum)
        target_val_size = max(1, min(target_val_size, len(stratum_indices) - 1))
        
        # Sample without replacement, weighted by inverse density
        if len(stratum_indices) > target_val_size:
            # Weighted sampling
            sampled_positions = np.random.choice(
                len(stratum_positions),
                size=target_val_size,
                replace=False,
                p=stratum_weights/np.sum(stratum_weights)
            )
            sampled_indices = stratum_positions[sampled_positions]
            val_indices.extend(remaining_indices[sampled_indices])
        else:
            # Take all if we need more than available
            val_indices.extend(stratum_indices)
        
        # Save checkpoint periodically
        if (i + 1) % checkpoint_frequency == 0 or i == len(unique_strata) - 1:
            with open(val_indices_file, "wb") as f:
                pickle.dump(val_indices, f)
            print(f"Saved checkpoint with {len(val_indices)} validation indices")
    
    # Remaining indices go to train set
    train_indices = np.setdiff1d(remaining_indices, val_indices)
    
    # Print statistics about the split
    print(f"Split sizes: Train={len(train_indices)}, Validation={len(val_indices)}, Test={len(test_indices)}")
    
    train_pos = np.sum(y[train_indices])
    val_pos = np.sum(y[val_indices])
    test_pos = np.sum(y[test_indices])
    
    print(f"Positive examples: Train={train_pos} ({train_pos/len(train_indices)*100:.1f}%), " +
          f"Val={val_pos} ({val_pos/len(val_indices)*100:.1f}%), " +
          f"Test={test_pos} ({test_pos/len(test_indices)*100:.1f}%)")
    
    # Save the final split
    with open(split_file, "wb") as f:
        pickle.dump({
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices
        }, f)
    
    print(f"Saved final split to {split_file}")
    
    # Keep temporary files for potential debugging/recovery
    # Do NOT remove val_indices_file to preserve all checkpoints
    
    return train_indices, val_indices, test_indices

# class DataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, X_file, y_file, indices, batch_size=32, shuffle=True, weights=None):
#         """
#         Data generator for efficient loading from memory-mapped arrays
        
#         Parameters:
#         -----------
#         X_file : str
#             Path to features file
#         y_file : str
#             Path to labels file
#         indices : array
#             Indices to sample from
#         batch_size : int
#             Batch size
#         shuffle : bool
#             Whether to shuffle indices
#         weights : array, optional
#             Sample weights (must be same length as indices)
#         """
#         self.X_file = X_file
#         self.y_file = y_file
#         self.indices = np.asarray(indices)  # Ensure array type
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.weights = weights
        
#         # Verify weights array
#         if self.weights is not None:
#             assert len(self.weights) == len(self.indices), "Weights array must match indices lengt...
        
#         # Load as memory-mapped arrays
#         self.X = np.load(self.X_file, mmap_mode='r')
#         self.y = np.load(self.y_file, mmap_mode='r')
        
#         # Get input shape from first sample
#         self.input_shape = self.X[self.indices[0]].shape
        
#         self.on_epoch_end()
    
#     def __len__(self):
#         """Number of batches per epoch"""
#         return int(np.ceil(len(self.indices) / self.batch_size))
    
#     def __getitem__(self, idx):
#         """Get batch at position idx"""
#         start_idx = idx * self.batch_size
#         end_idx = min((idx + 1) * self.batch_size, len(self.indices))
#         batch_indices = self.indices_array[start_idx:end_idx]
        
#         # Load data
#         X_batch = self.X[batch_indices]
#         y_batch = self.y[batch_indices]
        
#         if self.weights is not None:
#             # Get weights for these specific indices
#             batch_positions = np.where(np.isin(self.indices, batch_indices))[0]
#             w_batch = self.weights[batch_positions]
#             return X_batch, y_batch, w_batch
#         else:
#             return X_batch, y_batch
    
#     def on_epoch_end(self):
#         """Called at the end of each epoch"""
#         self.indices_array = np.copy(self.indices)
#         if self.shuffle:
#             np.random.shuffle(self.indices_array)
            
#     def get_input_shape(self):
#         """Get input shape of samples"""
#         return self.input_shape

# y_sample = np.load(y_file, mmap_mode='r')
# train_y = y_sample[train_indices]
# pos_count = np.sum(train_y)
# neg_count = len(train_y) - pos_count
# pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
# class_weight = {0: 1.0, 1: pos_weight}
# print(f"Positive class weight: {pos_weight:.2f}")
# del y_sample, train_y
# gc.collect()

# Combine sample weights with class weights for imbalanced data
pos_weight = (len(train_y) - np.sum(train_y)) / max(1, np.sum(train_y))
class_weight = {0: 1.0, 1: pos_weight}
print(f"Using class weight {pos_weight:.2f} for positive examples")
# Free memory
del train_y, val_y, test_y
gc.collect()

# # Custom data generator
# from tensorflow.keras.utils import Sequence

# class DataGenerator(Sequence):
#     def __init__(self, X, y, indices, batch_size=32, shuffle=True, weights=None):
#         self.X = X
#         self.y = y
#         self.indices = indices
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.weights = weights
#         self.on_epoch_end()
        
#     def __len__(self):
#         return int(np.ceil(len(self.indices) / self.batch_size))
        
#     def __getitem__(self, idx):
#         start_idx = idx * self.batch_size
#         end_idx = min((idx + 1) * self.batch_size, len(self.indices))
#         batch_indices = self.indices_array[start_idx:end_idx]
        
#         X_batch = self.X[batch_indices]
#         y_batch = self.y[batch_indices]
        
#         if self.weights is not None:
#             weights_batch = np.array([self.weights[i] for i in range(len(self.indices)) 
#                                      if self.indices[i] in batch_indices])
#             return X_batch, y_batch, weights_batch
#         else:
#             return X_batch, y_batch
        
#     def on_epoch_end(self):
#         self.indices_array = np.array(self.indices)
#         if self.shuffle:
#             np.random.shuffle(self.indices_array)

# # Custom data generator
# from tensorflow.keras.utils import Sequence

# class DataGenerator(Sequence):
#     def __init__(self, X, y, indices, batch_size=32, shuffle=True, weights=None):
#         self.X = X
#         self.y = y
#         self.indices = indices
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.weights = weights
#         self.on_epoch_end()
        
#     def __len__(self):
#         return int(np.ceil(len(self.indices) / self.batch_size))
        
#     def __getitem__(self, idx):
#         start_idx = idx * self.batch_size
#         end_idx = min((idx + 1) * self.batch_size, len(self.indices))
#         batch_indices = self.indices_array[start_idx:end_idx]
        
#         X_batch = self.X[batch_indices]
#         y_batch = self.y[batch_indices]
        
#         if self.weights is not None:
#             weights_batch = np.array([self.weights[i] for i in range(len(self.indices)) 
#                                      if self.indices[i] in batch_indices])
#             return X_batch, y_batch, weights_batch
#         else:
#             return X_batch, y_batch
        
#     def on_epoch_end(self):
#         self.indices_array = np.array(self.indices)
#         if self.shuffle:
#             np.random.shuffle(self.indices_array)

# class OptimizedDataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, X_file, y_file, indices, batch_size=32, shuffle=True, weights=None):
#         """
#         Optimized data generator for memory-mapped arrays
#         """
#         self.X = np.load(X_file, mmap_mode='r')
#         self.y = np.load(y_file, mmap_mode='r')
#         self.indices = np.array(indices)
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.weights = weights
#         self.on_epoch_end()
        
#     def __len__(self):
#         return int(np.ceil(len(self.indices) / self.batch_size))
        
#     def __getitem__(self, idx):
#         # Get batch indices
#         batch_indices = self.indices_array[idx * self.batch_size:
#                                           min((idx + 1) * self.batch_size, len(self.indices))]
        
#         # Load only the required data
#         X_batch = np.array([self.X[i] for i in batch_indices])
#         y_batch = np.array([self.y[i] for i in batch_indices])
        
#         if self.weights is not None:
#             # Find positions of batch_indices in original indices array
#             positions = np.array([np.where(self.indices == i)[0][0] for i in batch_indices])
#             weights_batch = self.weights[positions]
#             return X_batch, y_batch, weights_batch
#         else:
#             return X_batch, y_batch
        
#     def on_epoch_end(self):
#         self.indices_array = np.copy(self.indices)
#         if self.shuffle:
#             np.random.shuffle(self.indices_array)

# Data paths
data_dir = os.path.join('/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling', 'ml_data')
X_file = os.path.join(data_dir, 'X_features.npy')
y_file = os.path.join(data_dir, 'y_labels.npy')

# Load memory-mapped data for shape information
X = np.load(X_file, mmap_mode='r')
y = np.load(y_file, mmap_mode='r')

# Verify indices
print("Verifying indices and class weights...")
train_y = y[train_indices]
val_y = y[val_indices]
test_y = y[test_indices]

print(f"Train/val/test sizes: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")
print(f"Positive examples: Train={np.sum(train_y)} ({np.sum(train_y)/len(train_y)*100:.1f}%), " +
      f"Val={np.sum(val_y)} ({np.sum(val_y)/len(val_y)*100:.1f}%), " +
      f"Test={np.sum(test_y)} ({np.sum(test_y)/len(test_y)*100:.1f}%)")

# Combine sample weights with class weights for imbalanced data
pos_weight = (len(train_y) - np.sum(train_y)) / max(1, np.sum(train_y))
class_weight = {0: 1.0, 1: pos_weight}
print(f"Using class weight {pos_weight:.2f} for positive examples")

# Free memory
del train_y, val_y, test_y
gc.collect()

# model, final_model_path = efficient_balanced_training(
#     model, X_file, y_file,
#     train_indices, val_indices, test_indices,
#     output_dir=output_dir,
#     batch_size=256,          # Standard batch size
#     chunk_size=25000,        # Process 25k samples per chunk
#     epochs_per_chunk=2,      # Train each chunk for 2 epochs
#     save_frequency=5,        # Save every 5 chunks
#     class_weight=class_weight # Use class weights
# )

# print(f"Training complete. Final model saved to: {final_model_path}")

# model, final_model_path = efficient_balanced_training(
#     model, X_file, y_file,
#     train_indices, val_indices, test_indices,
#     output_dir=output_dir,
#     batch_size=256,          # Standard batch size
#     chunk_size=25000,        # Process 25k samples per chunk
#     epochs_per_chunk=2,      # Train each chunk for 2 epochs
#     save_frequency=5,        # Save every 5 chunks
#     class_weight=class_weight, # Use class weights
#     #start_chunk=45
#     #start_chunk=90
#     #start_chunk=135
#     #start_chunk=180
#     #start_chunk=225
#     #start_chunk=270
#     start_chunk=315
# )

# print(f"Training complete. Final model saved to: {final_model_path}")

# All 3,703,206 test sequences were predicted as negative (not zero-curtain events) by the AI model,...
# The model probabilities are all very low (the highest value shown is 0.154322), well below the 0.5...
# There's 100% agreement between the physical detection method and the AI model because all sequence...

# This situation raises important considerations:

# Class Imbalance: The test set appears to contain only negative examples (non-zero-curtain events) ...
# Model Calibration: The AI model is correctly predicting all negatives, but with very low confidenc...
# Evaluation Limitation: Without any positive examples in our test set, metrics like precision, reca...

# To get a more comprehensive evaluation:

# We should verify if the original test set contains any positive examples according to the physical...
# If there are truly no positives in the test set, we should consider restructuring the test set to ...
# We should extend our analysis to include training and validation sets, which may contain positive ...

# This result suggests a potential issue with either the test set sampling or with the original phys...

# Custom data generator
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, X, y, indices, batch_size=32, shuffle=True, weights=None):
        self.X = X
        self.y = y
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weights = weights
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
        
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices_array[start_idx:end_idx]
        
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        if self.weights is not None:
            weights_batch = np.array([self.weights[i] for i in range(len(self.indices)) 
                                     if self.indices[i] in batch_indices])
            return X_batch, y_batch, weights_batch
        else:
            return X_batch, y_batch
        
    def on_epoch_end(self):
        self.indices_array = np.array(self.indices)
        if self.shuffle:
            np.random.shuffle(self.indices_array)

# And you can use the weights in your model training
class_weight = {0: 1, 1: positive_ratio}  # Basic class weights
sample_weight = weights[train_indices]    # Spatial balancing weights
model.fit(X_train, y_train, sample_weight=sample_weight * class_weight[y_train])"

import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization

class BatchNorm5D(Layer):
    """
    Fixed implementation of BatchNorm5D that properly handles 5D inputs from ConvLSTM2D.
    This implementation correctly manages shapes and serialization.
    """
    def __init__(self, **kwargs):
        super(BatchNorm5D, self).__init__(**kwargs)
        self.bn = BatchNormalization()
        
    def call(self, inputs, training=None):
        # Get the dimensions
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        time_steps = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        channels = input_shape[4]
        
        # Reshape to 4D for BatchNorm by combining batch and time dimensions
        x_reshaped = tf.reshape(inputs, [-1, height, width, channels])
        
        # Apply BatchNorm
        x_bn = self.bn(x_reshaped, training=training)
        
        # Reshape back to 5D
        x_back = tf.reshape(x_bn, [batch_size, time_steps, height, width, channels])
        return x_back
    
    def get_config(self):
        config = super(BatchNorm5D, self).get_config()
        return config

# Register the custom layer globally
tf.keras.utils.get_custom_objects().update({'BatchNorm5D': BatchNorm5D})

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
from scipy.spatial import QhullError

class InSituDataProcessor:
    def __init__(self):
        self.temp_scaler = StandardScaler()
        self.alt_scaler = StandardScaler()
        
    def load_and_clean_temperature_data(self, df):
        """Process soil temperature data with enhanced cleaning"""
        required_cols = ['datetime', 'latitude', 'longitude', 'temperature']
        
        # Convert datetime if it's string or in 'Date' column
        if 'Date' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'])
        else:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Convert columns to numeric, handling potential string values
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        
        # Remove invalid coordinates and temperatures
        df = df[
            (df['latitude'].between(-90, 90)) &
            (df['longitude'].between(-180, 180)) &
            (~df['temperature'].isna())
        ]
        
        # Sort and remove duplicates
        df = df.sort_values(['datetime', 'latitude', 'longitude'])
        df = df.drop_duplicates(subset=['datetime', 'latitude', 'longitude', 'temperature'])
        
        print(f"Processed temperature data: {len(df)} valid points")
        return df[required_cols]
        
    def load_and_clean_alt_data(self, df):
        """Process active layer thickness data"""
        required_cols = ['datetime', 'latitude', 'longitude', 'thickness']
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Convert coordinates and thickness to numeric
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
        
        # Remove invalid values
        df = df[
            (df['latitude'].between(-90, 90)) &
            (df['longitude'].between(-180, 180)) &
            (~df['thickness'].isna())
        ]
        
        # Sort and remove duplicates
        df = df.sort_values(['datetime', 'latitude', 'longitude'])
        df = df.drop_duplicates(subset=['datetime', 'latitude', 'longitude', 'thickness'])
        
        print(f"Processed ALT data: {len(df)} valid points")
        return df[required_cols]
    
    def detect_zero_curtain(self, temp_series, threshold=0.1, duration=24):
        """Detect zero curtain periods in temperature time series"""
        # Handle NaN values
        temp_series = temp_series.dropna()
        if len(temp_series) < duration:
            return pd.Series(False, index=temp_series.index)
            
        # Find periods where temperature is near 0°C
        zero_curtain_mask = np.abs(temp_series) <= threshold
        
        # Find continuous periods
        zero_curtain_groups = (zero_curtain_mask != zero_curtain_mask.shift()).cumsum()
        
        # Filter for minimum duration
        valid_periods = (
            zero_curtain_mask.groupby(zero_curtain_groups)
            .filter(lambda x: len(x) >= duration and x.all())
        )
        
        return valid_periods
    
    def prepare_for_model(self, temp_df, alt_df):
        """Prepare data for the quantum-enhanced zero curtain model"""
        # Process individual datasets
        temp_df = self.load_and_clean_temperature_data(temp_df)
        alt_df = self.load_and_clean_alt_data(alt_df)
        
        print("\nCleaned data ranges:")
        print("Temperature data range:", temp_df['datetime'].min(), "to", temp_df['datetime'].max())
        print("ALT data range:", alt_df['datetime'].min(), "to", alt_df['datetime'].max())
        
        # Find common temporal range
        start_date = max(temp_df['datetime'].min(), alt_df['datetime'].min())
        end_date = min(temp_df['datetime'].max(), alt_df['datetime'].max())
        
        # Create regular monthly time grid
        time_grid = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Filter to common temporal range
        temp_df = temp_df[
            (temp_df['datetime'] >= start_date) & 
            (temp_df['datetime'] <= end_date)
        ].copy()
        
        alt_df = alt_df[
            (alt_df['datetime'] >= start_date) & 
            (alt_df['datetime'] <= end_date)
        ].copy()
        
        print("\nResampling data to monthly frequency...")
        
        # Convert datetime to period for consistent monthly grouping
        temp_df['month_period'] = temp_df['datetime'].dt.to_period('M')
        alt_df['month_period'] = alt_df['datetime'].dt.to_period('M')
        
        # Group by period and location
        temp_monthly = (temp_df.groupby(['month_period', 'latitude', 'longitude'])
                       ['temperature'].mean().reset_index())
        alt_monthly = (alt_df.groupby(['month_period', 'latitude', 'longitude'])
                      ['thickness'].mean().reset_index())
        
        # Convert period back to timestamp for interpolation
        temp_monthly['datetime'] = temp_monthly['month_period'].dt.to_timestamp()
        alt_monthly['datetime'] = alt_monthly['month_period'].dt.to_timestamp()
        
        print(f"Monthly temperature points: {len(temp_monthly)}")
        print(f"Monthly ALT points: {len(alt_monthly)}")
        
        # Create spatial grid
        lat_min = max(temp_monthly['latitude'].min(), alt_monthly['latitude'].min())
        lat_max = min(temp_monthly['latitude'].max(), alt_monthly['latitude'].max())
        lon_min = max(temp_monthly['longitude'].min(), alt_monthly['longitude'].min())
        lon_max = min(temp_monthly['longitude'].max(), alt_monthly['longitude'].max())
        
        # Create spatial grid
        spatial_resolution = 0.5  # degrees
        lat_grid = np.arange(lat_min, lat_max + spatial_resolution, spatial_resolution)
        lon_grid = np.arange(lon_min, lon_max + spatial_resolution, spatial_resolution)
        
        print("\nGrid dimensions:")
        print(f"Time steps: {len(time_grid)}")
        print(f"Latitude points: {len(lat_grid)}")
        print(f"Longitude points: {len(lon_grid)}")
        
        # Create meshgrid for interpolation
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Initialize output grids
        temp_grid = np.full((len(time_grid), len(lat_grid), len(lon_grid)), np.nan)
        alt_grid = np.full((len(time_grid), len(lat_grid), len(lon_grid)), np.nan)
        
        print("\nPerforming interpolation...")
        # Process each time step
        for t_idx, t in enumerate(time_grid):
            # Get data for current month
            temp_month = temp_monthly[temp_monthly['datetime'].dt.to_period('M') == 
                                   t.to_period('M')]
            alt_month = alt_monthly[alt_monthly['datetime'].dt.to_period('M') == 
                                  t.to_period('M')]
            
            # Debug information
            if t_idx % 100 == 0:
                print(f"Processing {t}: Temp points = {len(temp_month)}, "
                      f"ALT points = {len(alt_month)}")
            
            # Temperature interpolation
            if len(temp_month) > 3:
                try:
                    grid_slice = griddata(
                        temp_month[['latitude', 'longitude']].values,
                        temp_month['temperature'].values,
                        (lat_mesh, lon_mesh),
                        method='linear',
                        fill_value=np.nan
                    )
                    temp_grid[t_idx] = grid_slice
                except Exception as e:
                    print(f"Temperature interpolation failed for {t}: {str(e)}")
            
            # ALT interpolation
            if len(alt_month) > 3:
                try:
                    grid_slice = griddata(
                        alt_month[['latitude', 'longitude']].values,
                        alt_month['thickness'].values,
                        (lat_mesh, lon_mesh),
                        method='linear',
                        fill_value=np.nan
                    )
                    alt_grid[t_idx] = grid_slice
                except Exception as e:
                    print(f"ALT interpolation failed for {t}: {str(e)}")
        
        # Debug information for interpolation results
        temp_valid = np.sum(~np.isnan(temp_grid))
        alt_valid = np.sum(~np.isnan(alt_grid))
        
        print("\nInterpolation stats:")
        print(f"Temperature grid shape: {temp_grid.shape}")
        print(f"Temperature valid points: {temp_valid}")
        print(f"Temperature coverage: {temp_valid / temp_grid.size * 100:.2f}%")
        print(f"ALT grid shape: {alt_grid.shape}")
        print(f"ALT valid points: {alt_valid}")
        print(f"ALT coverage: {alt_valid / alt_grid.size * 100:.2f}%")
        
        # Detect zero curtain periods
        print("\nDetecting zero curtain periods...")
        n_times, n_lat, n_lon = temp_grid.shape
        zero_curtain_mask = np.zeros((n_times, n_lat, n_lon), dtype=bool)
        zero_curtain_count = 0
        
        for i in range(n_lat):
            for j in range(n_lon):
                # Get temperature time series for this point
                temps = temp_grid[:, i, j]
                
                # Skip if all NaN
                if np.all(np.isnan(temps)):
                    continue
                    
                # Find zero curtain periods (temps near 0°C)
                near_zero = np.abs(temps) <= 0.1
                near_zero[np.isnan(temps)] = False
                
                # Find continuous periods (3 months or more)
                count = 0
                start_idx = 0
                
                for t in range(n_times):
                    if near_zero[t]:
                        if count == 0:
                            start_idx = t
                        count += 1
                    else:
                        if count >= 3:  # 3 months minimum duration
                            zero_curtain_mask[start_idx:t, i, j] = True
                            zero_curtain_count += t - start_idx
                        count = 0
                
                # Handle case where series ends in zero curtain
                if count >= 3:
                    zero_curtain_mask[start_idx:, i, j] = True
                    zero_curtain_count += n_times - start_idx
        
        print(f"Zero curtain periods detected: {zero_curtain_count}")
        
        # Calculate quantum parameters
        print("\nCalculating quantum parameters...")
        quantum_params = self.get_quantum_parameters(temp_grid)
        
        return {
            'temperature': temp_grid,
            'alt': alt_grid,
            'zero_curtain_mask': zero_curtain_mask,
            'time_grid': time_grid,
            'lat_grid': lat_grid,
            'lon_grid': lon_grid,
            'thermal_wavelength': quantum_params['thermal_wavelength'],
            'tunneling_probability': quantum_params['tunneling_probability']
        }

    def get_quantum_parameters(self, temp_grid):
        """Calculate quantum mechanical parameters for phase transition"""
        # Constants
        hbar = 1.0545718e-34  # Reduced Planck constant
        k_B = 1.380649e-23    # Boltzmann constant
        m_water = 2.99e-26    # Mass of water molecule
        
        # Handle zero temperatures to avoid division by zero
        temp_grid_abs = np.abs(temp_grid)
        temp_grid_abs[temp_grid_abs == 0] = 1e-10
        
        # Calculate thermal wavelength
        thermal_wavelength = np.sqrt((2 * np.pi * hbar**2) / (m_water * k_B * temp_grid_abs))
        
        # Calculate quantum tunneling probability near 0°C
        V0 = 2.4e-21  # Potential barrier height
        tunneling_prob = np.exp(-2 * np.sqrt(2 * m_water * V0) * thermal_wavelength / hbar)
        
        return {
            'thermal_wavelength': thermal_wavelength,
            'tunneling_probability': tunneling_prob
        }

class PhysicsLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Physical constants
        self.L_fusion = 334000.0  # J/kg, latent heat of fusion for water
        self.k_ice = 2.22  # W/m/K, thermal conductivity of ice
        self.k_water = 0.58  # W/m/K, thermal conductivity of water
        self.c_ice = 2090.0  # J/kg/K, specific heat capacity of ice
        self.c_water = 4186.0  # J/kg/K, specific heat capacity of water
        
    def build(self, input_shape):
        self.porosity = self.add_weight(
            name="porosity",
            shape=[1],
            initializer=tf.keras.initializers.Constant(0.4),
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.9)
        )
        self.output_shape = input_shape[0]
        super().build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return self.output_shape
        
    def call(self, inputs):
        temperatures, time_steps = inputs
        temp_padded = tf.pad(temperatures, [[0, 0], [1, 1], [0, 0], [0, 0]])
        temp_gradients = (temp_padded[:, 2:] - temp_padded[:, :-2]) / 2.0
        phase_fraction = tf.sigmoid((temperatures + 0.1) * 100)
        k_eff = (phase_fraction * self.k_water + (1 - phase_fraction) * self.k_ice) * self.porosity
        return temperatures + k_eff * temp_gradients

# Model improvement via:
# Architecture Improvements:
# Add more Conv2D layers to capture spatial features better
# Experiment with different ConvLSTM2D configurations
# Add spatial attention mechanisms to focus on important regions
# Try residual connections to help with gradient flow

# Training Strategy:
# Implement learning rate scheduling
# Use different loss functions (e.g., focal loss for imbalanced data)
# Increase training epochs with better early stopping
# Add class weights to handle imbalance (high recall but low precision suggests imbalance)

# Data Handling:
# Add more data augmentation specific to time series
# Better handling of the temporal dependencies (maybe longer sequences)
# Add more relevant physical parameters to the PhysicsLayer
# Consider seasonal patterns more explicitly

