#!/usr/bin/env python3
"""
ArcticDEM Acquisition Module

Handles STAC API-based acquisition of ArcticDEM tiles with checkpoint/resume
functionality for robust long-running downloads. This module supports the
downscaling pipeline by providing high-resolution terrain data.

"""

import os
import sys
import time
import json
import logging
import requests
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import concurrent.futures
from dataclasses import dataclass, asdict
from enum import Enum

# Configuration constants
MAX_WORKERS = 32
BATCH_SIZE = 32
MAX_BATCHES = 6
MAX_RUNTIME = 10800  # 3 hours

class ProcessingStatus(Enum):
    """Enumeration of processing states"""
    INITIALIZING = "initializing"
    ITEMS_RETRIEVED = "items_retrieved"
    FILTERING_COMPLETE = "filtering_complete"
    ITEMS_SHUFFLED = "items_shuffled"
    BATCHES_CREATED = "batches_created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    ERROR = "error"

@dataclass
class CheckpointData:
    """Data structure for checkpoint information"""
    status: str
    last_updated: str
    total_items: Optional[int] = None
    items_to_process: Optional[int] = None
    total_completed: Optional[int] = None
    total_failed: Optional[int] = None
    total_remaining: Optional[int] = None
    completion_percentage: Optional[float] = None
    batches: Optional[Dict] = None
    error: Optional[str] = None
    reason: Optional[str] = None

class CheckpointManager:
    """Manages checkpoint operations for resumable processing"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.main_checkpoint = self.checkpoint_dir / "processing_checkpoint.json"
    
    def load_checkpoint(self, checkpoint_file: Optional[Path] = None) -> Dict:
        """Load checkpoint data from file"""
        file_path = checkpoint_file or self.main_checkpoint
        
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading checkpoint {file_path}: {e}")
                return {}
        return {}
    
    def save_checkpoint(self, data: Dict, checkpoint_file: Optional[Path] = None) -> bool:
        """Save checkpoint data with atomic write"""
        file_path = checkpoint_file or self.main_checkpoint
        temp_file = file_path.with_suffix('.tmp')
        
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(file_path)
            return True
        except Exception as e:
            logging.error(f"Error saving checkpoint {file_path}: {e}")
            return False
    
    def update_main_checkpoint(self, status: str, additional_data: Optional[Dict] = None) -> bool:
        """Update main checkpoint with current status"""
        checkpoint = self.load_checkpoint()
        checkpoint['last_updated'] = datetime.now().isoformat()
        checkpoint['status'] = status
        
        if additional_data:
            checkpoint.update(additional_data)
        
        return self.save_checkpoint(checkpoint)
    
    def get_batch_checkpoint_file(self, batch_idx: int) -> Path:
        """Get batch-specific checkpoint file path"""
        return self.checkpoint_dir / f"batch_{batch_idx}_checkpoint.json"
    
    def update_batch_checkpoint(self, batch_idx: int, status: str, 
                               items_data: Optional[Dict] = None) -> bool:
        """Update batch checkpoint file"""
        checkpoint_file = self.get_batch_checkpoint_file(batch_idx)
        checkpoint = self.load_checkpoint(checkpoint_file)
        
        checkpoint['last_updated'] = datetime.now().isoformat()
        checkpoint['batch_idx'] = batch_idx
        checkpoint['status'] = status
        
        if items_data:
            if 'items' not in checkpoint:
                checkpoint['items'] = {}
            checkpoint['items'].update(items_data)
        
        return self.save_checkpoint(checkpoint, checkpoint_file)
    
    def update_item_checkpoint(self, batch_idx: int, item_id: str, 
                              status: str, details: Optional[str] = None) -> bool:
        """Update individual item status in batch checkpoint"""
        checkpoint_file = self.get_batch_checkpoint_file(batch_idx)
        checkpoint = self.load_checkpoint(checkpoint_file)
        
        if 'items' not in checkpoint:
            checkpoint['items'] = {}
        
        checkpoint['items'][item_id] = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        
        return self.save_checkpoint(checkpoint, checkpoint_file)
    
    def get_item_status(self, batch_idx: int, item_id: str) -> Optional[Dict]:
        """Get item status from batch checkpoint"""
        checkpoint_file = self.get_batch_checkpoint_file(batch_idx)
        checkpoint = self.load_checkpoint(checkpoint_file)
        return checkpoint.get('items', {}).get(item_id)

class STACClient:
    """Client for interacting with STAC API"""
    
    def __init__(self, base_url: str, max_items: int = 1000):
        self.base_url = base_url
        self.max_items = max_items
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create session with retry logic"""
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
        session.mount('https://', adapter)
        return session
    
    def get_all_items(self, logger: logging.Logger) -> List[Dict]:
        """Retrieve all items from STAC API with pagination"""
        items = []
        next_url = f"{self.base_url}?limit={self.max_items}"
        
        logger.info(f"Retrieving items from STAC API: {self.base_url}")
        
        while next_url:
            try:
                logger.info(f"Querying: {next_url}")
                response = self.session.get(next_url)
                response.raise_for_status()
                data = response.json()
                
                batch_items = data.get('features', [])
                items.extend(batch_items)
                logger.info(f"Retrieved {len(batch_items)} items, total: {len(items)}")
                
                # Find next page
                next_url = None
                for link in data.get('links', []):
                    if link.get('rel') == 'next':
                        next_url = link.get('href')
                        break
                        
            except Exception as e:
                logger.error(f"Error retrieving items: {e}")
                time.sleep(10)
                continue
        
        logger.info(f"Retrieved total of {len(items)} items")
        return items

class DEMDownloader:
    """Handles DEM file downloads with retry logic"""
    
    def __init__(self):
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create session with retry logic"""
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
        session.mount('https://', adapter)
        return session
    
    def download_file(self, url: str, destination: Path, 
                     logger: logging.Logger) -> bool:
        """Download file with progress logging"""
        if destination.exists():
            logger.info(f"File exists: {destination}")
            return True
        
        logger.info(f"Downloading: {url}")
        
        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            file_size = int(response.headers.get('content-length', 0))
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            with open(destination, 'wb') as f:
                if file_size > 0:
                    downloaded = 0
                    last_progress = -1
                    for chunk in response.iter_content(chunk_size=8192):
                        downloaded += len(chunk)
                        f.write(chunk)
                        
                        progress = int((downloaded / file_size) * 100)
                        if progress in {25, 50, 75, 100} and progress != last_progress:
                            logger.info(f"Progress {destination.name}: {progress}%")
                            last_progress = progress
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logger.info(f"Download complete: {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Download error {url}: {e}")
            if destination.exists():
                destination.unlink()
            time.sleep(random.uniform(1, 3))
            return False

class ArcticDEMAcquisition:
    """Main acquisition orchestrator"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.checkpoint_mgr = CheckpointManager(Path(config['checkpoint_dir']))
        self.stac_client = STACClient(config['stac_url'])
        self.downloader = DEMDownloader()
        self.logger = self._setup_logging()
        self.start_time = time.time()
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        log_dir = Path(self.config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"arcticdem_acquisition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logger = logging.getLogger('arcticdem_acquisition')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def check_runtime(self) -> bool:
        """Check if maximum runtime exceeded"""
        elapsed = time.time() - self.start_time
        if elapsed > MAX_RUNTIME:
            self.logger.warning(f"Max runtime {MAX_RUNTIME}s exceeded")
            return False
        return True
    
    def retrieve_items(self) -> List[Dict]:
        """Retrieve or load cached STAC items"""
        items_file = Path(self.config['checkpoint_dir']) / "items.json"
        
        if items_file.exists():
            self.logger.info(f"Loading cached items: {items_file}")
            with open(items_file, 'r') as f:
                items = json.load(f)
            self.logger.info(f"Loaded {len(items)} cached items")
        else:
            self.logger.info("Retrieving items from STAC API")
            items = self.stac_client.get_all_items(self.logger)
            
            # Cache items
            with open(items_file, 'w') as f:
                json.dump(items, f)
            self.logger.info(f"Cached items to: {items_file}")
            
            self.checkpoint_mgr.update_main_checkpoint(
                ProcessingStatus.ITEMS_RETRIEVED.value
            )
        
        return items
    
    def filter_items(self, items: List[Dict]) -> Tuple[List[Dict], int]:
        """Filter items based on existing outputs and checkpoints"""
        output_dir = Path(self.config['output_dir'])
        items_to_process = []
        completed_count = 0
        
        for item in items:
            item_id = item['id']
            output_path = output_dir / f"{item_id}_30m_terrain.tif"
            
            # Check file existence
            if output_path.exists():
                completed_count += 1
                continue
            
            # Check checkpoint status
            is_completed = False
            for batch_idx in range(MAX_BATCHES):
                item_status = self.checkpoint_mgr.get_item_status(batch_idx, item_id)
                if item_status and item_status.get('status') == 'completed':
                    is_completed = True
                    completed_count += 1
                    break
            
            if not is_completed:
                items_to_process.append(item)
        
        self.logger.info(
            f"Processing {len(items_to_process)}/{len(items)} items "
            f"({completed_count} completed)"
        )
        
        return items_to_process, completed_count
    
    def download_item(self, item: Dict, batch_idx: int) -> bool:
        """Download and track single item"""
        item_id = item['id']
        dem_asset = item['assets'].get('dem')
        
        if not dem_asset:
            self.logger.warning(f"No DEM asset: {item_id}")
            self.checkpoint_mgr.update_item_checkpoint(
                batch_idx, item_id, 'failed', 'No DEM asset'
            )
            return False
        
        dem_url = dem_asset['href']
        temp_dir = Path(self.config['temp_dir'])
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        destination = temp_dir / f"{item_id}_original.tif"
        
        # Check checkpoint
        item_status = self.checkpoint_mgr.get_item_status(batch_idx, item_id)
        if item_status and item_status.get('status') == 'download_completed':
            if destination.exists():
                self.logger.info(f"Download previously completed: {item_id}")
                return True
        
        # Download
        success = self.downloader.download_file(dem_url, destination, self.logger)
        
        if success:
            self.checkpoint_mgr.update_item_checkpoint(
                batch_idx, item_id, 'download_completed', str(destination)
            )
        else:
            self.checkpoint_mgr.update_item_checkpoint(
                batch_idx, item_id, 'failed', 'Download failed'
            )
        
        return success
    
    def process_batch(self, batch: List[Dict], batch_idx: int, 
                     total_batches: int) -> Tuple[int, int]:
        """Process batch of items in parallel"""
        self.logger.info(
            f"Processing batch {batch_idx+1}/{total_batches} ({len(batch)} items)"
        )
        
        # Update checkpoints
        self.checkpoint_mgr.update_main_checkpoint(
            ProcessingStatus.PROCESSING.value,
            {str(batch_idx): {'status': 'started', 'timestamp': datetime.now().isoformat()}}
        )
        self.checkpoint_mgr.update_batch_checkpoint(batch_idx, 'started')
        
        batch_completed = 0
        batch_failed = 0
        
        # Filter batch based on checkpoint
        checkpoint_file = self.checkpoint_mgr.get_batch_checkpoint_file(batch_idx)
        checkpoint_data = self.checkpoint_mgr.load_checkpoint(checkpoint_file)
        
        filtered_batch = []
        for item in batch:
            item_id = item['id']
            item_status = checkpoint_data.get('items', {}).get(item_id, {}).get('status')
            
            if item_status == 'completed':
                self.logger.info(f"Item completed per checkpoint: {item_id}")
                batch_completed += 1
            else:
                filtered_batch.append(item)
        
        if len(filtered_batch) < len(batch):
            self.logger.info(
                f"Reduced batch from {len(batch)} to {len(filtered_batch)} items"
            )
        
        # Process with thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_item = {
                executor.submit(self.download_item, item, batch_idx): item['id']
                for item in filtered_batch
            }
            
            for future in concurrent.futures.as_completed(future_to_item):
                item_id = future_to_item[future]
                try:
                    success = future.result()
                    if success:
                        batch_completed += 1
                    else:
                        batch_failed += 1
                    
                    # Periodic checkpoint update
                    if (batch_completed + batch_failed) % 5 == 0:
                        self.checkpoint_mgr.update_batch_checkpoint(
                            batch_idx, 'processing',
                            {'completed': batch_completed, 'failed': batch_failed}
                        )
                        
                except Exception as e:
                    self.logger.error(f"Error processing {item_id}: {e}")
                    batch_failed += 1
                    self.checkpoint_mgr.update_item_checkpoint(
                        batch_idx, item_id, 'failed', str(e)
                    )
        
        # Final batch status
        status = 'completed' if batch_failed == 0 else 'completed_with_errors'
        self.logger.info(
            f"Batch {batch_idx+1}/{total_batches} {status}: "
            f"{batch_completed} completed, {batch_failed} failed"
        )
        
        self.checkpoint_mgr.update_batch_checkpoint(
            batch_idx, status,
            {'completed': batch_completed, 'failed': batch_failed, 'total': len(batch)}
        )
        
        return batch_completed, batch_failed
    
    def run(self) -> bool:
        """Execute full acquisition pipeline"""
        self.logger.info("=== ArcticDEM Acquisition Started ===")
        self.logger.info(f"Output: {self.config['output_dir']}")
        self.logger.info(f"Temp: {self.config['temp_dir']}")
        self.logger.info(f"Checkpoint: {self.config['checkpoint_dir']}")
        
        try:
            # Check for resume
            main_checkpoint = self.checkpoint_mgr.load_checkpoint()
            is_resuming = main_checkpoint.get('status') in ['processing', 'interrupted']
            
            if is_resuming:
                self.logger.info(f"Resuming from checkpoint")
            else:
                self.logger.info("Starting new acquisition")
                self.checkpoint_mgr.update_main_checkpoint(
                    ProcessingStatus.INITIALIZING.value
                )
            
            # Retrieve items
            items = self.retrieve_items()
            
            # Filter items
            items_to_process, completed_count = self.filter_items(items)
            
            if not items_to_process:
                self.logger.info("No items to process")
                self.checkpoint_mgr.update_main_checkpoint(
                    ProcessingStatus.COMPLETED.value
                )
                return True
            
            # Shuffle for geographic distribution
            random.shuffle(items_to_process)
            self.checkpoint_mgr.update_main_checkpoint(
                ProcessingStatus.ITEMS_SHUFFLED.value
            )
            
            # Create batch groups
            items_per_batch = max(BATCH_SIZE, len(items_to_process) // MAX_BATCHES)
            batch_groups = []
            
            for i in range(0, len(items_to_process), items_per_batch):
                batch = items_to_process[i:i+items_per_batch]
                if batch:
                    batch_groups.append(batch)
                    if len(batch_groups) >= MAX_BATCHES:
                        break
            
            self.logger.info(f"Created {len(batch_groups)} batch groups")
            self.checkpoint_mgr.update_main_checkpoint(
                ProcessingStatus.BATCHES_CREATED.value,
                {'num_batches': len(batch_groups)}
            )
            
            # Process batches
            total_completed = 0
            total_failed = 0
            
            for batch_idx, batch in enumerate(batch_groups):
                if not self.check_runtime():
                    self.checkpoint_mgr.update_main_checkpoint(
                        ProcessingStatus.INTERRUPTED.value,
                        {
                            'total_completed': total_completed,
                            'total_failed': total_failed,
                            'reason': 'max_runtime'
                        }
                    )
                    return False
                
                batch_completed, batch_failed = self.process_batch(
                    batch, batch_idx, len(batch_groups)
                )
                
                total_completed += batch_completed
                total_failed += batch_failed
            
            # Final status
            self.logger.info("=== Acquisition Completed ===")
            self.logger.info(f"Completed: {total_completed}, Failed: {total_failed}")
            
            self.checkpoint_mgr.update_main_checkpoint(
                ProcessingStatus.COMPLETED.value,
                {
                    'total_completed': total_completed,
                    'total_failed': total_failed,
                    'completion_percentage': round((completed_count + total_completed) / len(items) * 100, 2)
                }
            )
            
            return True
            
        except KeyboardInterrupt:
            self.logger.warning("User interrupted")
            self.checkpoint_mgr.update_main_checkpoint(
                ProcessingStatus.INTERRUPTED.value,
                {'reason': 'user_interrupt'}
            )
            return False
            
        except Exception as e:
            self.logger.error(f"Unhandled exception: {e}")
            self.checkpoint_mgr.update_main_checkpoint(
                ProcessingStatus.ERROR.value,
                {'error': str(e)}
            )
            raise

def main():
    """Main entry point"""
    config = {
        'stac_url': "https://stac.pgc.umn.edu/api/v1/collections/arcticdem-mosaics-v4.1-2m/items",
        'output_dir': "data/auxiliary/arcticdem/processed",
        'temp_dir': "data/auxiliary/arcticdem/temp",
        'log_dir': "logs/arcticdem",
        'checkpoint_dir': "checkpoints/arcticdem"
    }
    
    acquisition = ArcticDEMAcquisition(config)
    success = acquisition.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
