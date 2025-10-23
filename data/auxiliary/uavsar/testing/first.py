#!/usr/bin/env python3
"""
Unified UAVSAR/NISAR Data Acquisition System
Combines robust async downloading, state management, file discovery, and filtering.
"""

import asyncio
import aiohttp
import aiofiles
from aiohttp import ClientSession, BasicAuth
from urllib.parse import urlparse, urljoin
from pathlib import Path
from tqdm.asyncio import tqdm
import time
import json
import subprocess
import os
import sys
import argparse
import logging
import netrc
from bs4 import BeautifulSoup
from datetime import datetime
import re

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

class Config:
    """Centralized configuration management"""
    # Base URLs
    UAVSAR_BASE_URL = "https://uavsar.asf.alaska.edu"
    EARTHDATA_HOST = 'urs.earthdata.[RESEARCH_INSTITUTION_DOMAIN]'
    
    # Download settings
    MAX_CONCURRENT = 5
    BATCH_SIZE = 20
    TIMEOUT_SECONDS = 3600
    WGET_TIMEOUT = 300
    WGET_RETRIES = 5
    CHUNK_SIZE = 1024 * 1024  # 1MB
    
    # File patterns
    ALLOWED_EXTENSIONS = ['.grd', '.grd.zip', '.grd.tiff', '.grd.tiff.zip', '.ann', '.h5']
    PRIORITY_SUFFIXES = [
        [".unw.grd", ".int.grd"],
        [".cor.grd"],
        [".hgt.grd"],
        [".ann"]
    ]
    
    # Paths (override via command-line arguments)
    DEFAULT_DOWNLOAD_DIR = "/Volumes/JPL/uavsar"
    STATE_FILE = "uavsar_download_state.json"
    LOG_DIR = "logs"

# ============================================================================
# AUTHENTICATION AND STATE MANAGEMENT
# ============================================================================

class AuthenticationManager:
    """Handles [RESEARCH_INSTITUTION] Earthdata authentication"""
    
    @staticmethod
    def get_credentials():
        """Retrieve credentials from .netrc or prompt user"""
        try:
            nrc = netrc.netrc()
            auth_info = nrc.authenticators(Config.EARTHDATA_HOST)
            if auth_info:
                return auth_info[0], auth_info[2]
            raise ValueError("No credentials found")
        except Exception as e:
            logging.warning(f"Credentials not in .netrc: {e}")
            username = input("Enter Earthdata username: ")
            password = input("Enter Earthdata password: ")
            return username, password
    
    @staticmethod
    def setup_netrc(username, password):
        """Create/update .netrc file"""
        netrc_path = os.path.expanduser("~/.netrc")
        
        if os.path.exists(netrc_path):
            with open(netrc_path, 'r') as f:
                if Config.EARTHDATA_HOST in f.read():
                    return
        
        with open(netrc_path, 'a') as f:
            f.write(f"\nmachine {Config.EARTHDATA_HOST}\n")
            f.write(f"login {username}\n")
            f.write(f"password {password}\n")
        
        os.chmod(netrc_path, 0o600)
        logging.info(f"Credentials saved to {netrc_path}")


class StateManager:
    """Persistent state management for resumable downloads"""
    
    def __init__(self, state_file=Config.STATE_FILE):
        self.state_file = state_file
        self.downloaded_files = set()
        self.processed_base_names = set()
        self.completed_dirs = set()
        self.last_completed_batch = 0
        self.timestamp = None
        self.load_state()
    
    def load_state(self):
        """Load previous download state"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.downloaded_files = set(state.get("downloaded_files", []))
                    self.processed_base_names = set(state.get("processed_base_names", []))
                    self.completed_dirs = set(state.get("completed_dirs", []))
                    self.last_completed_batch = state.get("last_completed_batch", 0)
                    self.timestamp = state.get("timestamp")
                    
                logging.info(f"Loaded state: {len(self.downloaded_files)} files, "
                           f"{len(self.processed_base_names)} base names processed")
                if self.timestamp:
                    logging.info(f"Last saved: {time.ctime(self.timestamp)}")
            except Exception as e:
                logging.error(f"Failed to load state: {e}")
    
    async def save_state(self):
        """Asynchronously save current state"""
        state = {
            "downloaded_files": list(self.downloaded_files),
            "processed_base_names": list(self.processed_base_names),
            "completed_dirs": list(self.completed_dirs),
            "last_completed_batch": self.last_completed_batch,
            "timestamp": time.time()
        }
        
        try:
            async with aiofiles.open(self.state_file, 'w') as f:
                await f.write(json.dumps(state, indent=2))
            logging.debug(f"State saved: {len(self.downloaded_files)} files")
        except Exception as e:
            logging.error(f"Failed to save state: {e}")
    
    def mark_file_complete(self, url):
        """Mark a file as successfully downloaded"""
        self.downloaded_files.add(url)
    
    def mark_directory_complete(self, dir_url):
        """Mark a directory as fully processed"""
        self.completed_dirs.add(dir_url)
    
    def is_file_complete(self, url):
        """Check if file was previously downloaded"""
        return url in self.downloaded_files
    
    def is_directory_complete(self, dir_url):
        """Check if directory was previously processed"""
        return dir_url in self.completed_dirs


class StateReconstructor:
    """Rebuild state from existing filesystem"""
    
    @staticmethod
    def scan_directories(directories):
        """Scan multiple directories for existing files"""
        completed_files = set()
        completed_dirs = set()
        
        for base_dir in directories:
            if not os.path.exists(base_dir):
                continue
                
            for data_dir in Path(base_dir).glob("uavsar_data_*"):
                line_id = data_dir.name.replace("uavsar_data_", "")
                
                # Scan for all relevant files
                extensions = Config.ALLOWED_EXTENSIONS + ['.tiff']
                for ext in extensions:
                    for file_path in data_dir.glob(f"*{ext}"):
                        # Reconstruct probable source URL
                        url = f"{Config.UAVSAR_BASE_URL}/{line_id}/{file_path.name}"
                        completed_files.add(url)
                        completed_dirs.add(f"{Config.UAVSAR_BASE_URL}/{line_id}/")
        
        return completed_files, completed_dirs
    
    @staticmethod
    def rebuild_state(directories, output_file=Config.STATE_FILE):
        """Rebuild and save state from directories"""
        files, dirs = StateReconstructor.scan_directories(directories)
        
        state = {
            "downloaded_files": list(files),
            "completed_dirs": list(dirs),
            "processed_base_names": [],
            "last_completed_batch": 0,
            "timestamp": time.time()
        }
        
        with open(output_file, 'w') as f:
            json.dump(state, indent=2)
        
        logging.info(f"Rebuilt state: {len(files)} files, {len(dirs)} directories")
        return state

# ============================================================================
# FILE DISCOVERY AND FILTERING
# ============================================================================

class FileDiscovery:
    """Discover and filter UAVSAR files from web directories"""
    
    def __init__(self, session, base_url=Config.UAVSAR_BASE_URL):
        self.session = session
        self.base_url = base_url
    
    async def get_directory_listing(self, url):
        """Retrieve file listing from directory URL"""
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    logging.error(f"Failed to fetch {url}: HTTP {resp.status}")
                    return []
                text = await resp.text()
                soup = BeautifulSoup(text, 'html.parser')
                return [link.get('href') for link in soup.find_all('a') 
                       if link.get('href')]
        except Exception as e:
            logging.error(f"Error fetching directory {url}: {e}")
            return []
    
    async def find_related_files(self, base_name, site_prefix):
        """Find all related files for a given base name"""
        found_urls = []
        
        # Get root directory listing
        root_listing = await self.get_directory_listing(self.base_url + "/")
        related_dirs = [d for d in root_listing 
                       if site_prefix in d and d.endswith("/")]
        
        logging.debug(f"Found {len(related_dirs)} directories for prefix {site_prefix}")
        
        # Search each related directory
        matched_types = set()
        for dir_suffix in related_dirs:
            full_dir_url = f"{self.base_url}/{dir_suffix}"
            filenames = await self.get_directory_listing(full_dir_url)
            
            # Match files by priority suffix groups
            for group in Config.PRIORITY_SUFFIXES:
                for suffix in group:
                    file_type = suffix.split('.')[-2] if '.' in suffix else suffix
                    if file_type in matched_types:
                        continue
                    
                    for fname in filenames:
                        if fname.endswith(suffix) and base_name in fname:
                            found_urls.append(f"{full_dir_url}{fname}")
                            matched_types.add(file_type)
                            break
            
            await asyncio.sleep(0.5)  # Rate limiting
        
        return found_urls
    
    @staticmethod
    def filter_by_extension(urls, allowed_extensions=Config.ALLOWED_EXTENSIONS):
        """Filter URLs by file extension"""
        return [url for url in urls 
                if any(url.endswith(ext) for ext in allowed_extensions)]
    
    @staticmethod
    def filter_by_pattern(urls, exclude_pattern=None):
        """Filter URLs by regex pattern (exclusion)"""
        if not exclude_pattern:
            return urls
        pattern = re.compile(exclude_pattern)
        return [url for url in urls if not pattern.search(url)]

# ============================================================================
# DOWNLOAD MANAGER
# ============================================================================

class DownloadManager:
    """Manages file downloads with progress tracking and error handling"""
    
    def __init__(self, session, download_dir, state_manager):
        self.session = session
        self.download_dir = Path(download_dir)
        self.state = state_manager
        self.download_dir.mkdir(parents=True, exist_ok=True)
    
    async def download_with_wget(self, url, output_path, username, password):
        """Download using wget subprocess for reliability"""
        if self.state.is_file_complete(url):
            logging.info(f"Skipping (already downloaded): {output_path.name}")
            return True
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if partial download exists
        if output_path.exists():
            logging.info(f"Resuming partial download: {output_path.name}")
        
        cmd = [
            "wget",
            f"--timeout={Config.WGET_TIMEOUT}",
            f"--tries={Config.WGET_RETRIES}",
            "--waitretry=5",
            "--continue",  # Resume capability
            f"--user={username}",
            f"--password={password}",
            "--no-check-certificate",
            "-O", str(output_path),
            url
        ]
        
        try:
            logging.info(f"Downloading: {output_path.name}")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=Config.TIMEOUT_SECONDS
            )
            
            if process.returncode == 0:
                self.state.mark_file_complete(url)
                logging.info(f" Downloaded: {output_path.name}")
                return True
            else:
                error = stderr.decode().strip()
                logging.error(f" Failed: {output_path.name} - {error}")
                return False
                
        except asyncio.TimeoutError:
            try:
                process.kill()
            except:
                pass
            logging.error(f"⏱ Timeout: {output_path.name}")
            return False
        except Exception as e:
            logging.error(f"Error downloading {url}: {e}")
            return False
    
    async def download_with_aiohttp(self, url, output_path, pbar=None):
        """Download using aiohttp with progress tracking"""
        if self.state.is_file_complete(url):
            if pbar:
                pbar.update(1)
            return True
        
        resume_size = 0
        headers = {}
        
        # Check for partial download
        if output_path.exists():
            resume_size = output_path.stat().st_size
            if resume_size > 0:
                headers['Range'] = f'bytes={resume_size}-'
                logging.info(f"Resuming from byte {resume_size}: {output_path.name}")
        
        try:
            async with self.session.get(url, headers=headers) as resp:
                if resp.status == 206:  # Partial content
                    mode = 'ab'
                elif resp.status == 200:
                    mode = 'wb'
                    resume_size = 0
                else:
                    logging.error(f"HTTP {resp.status}: {url}")
                    return False
                
                file_size = int(resp.headers.get('Content-Length', 0))
                
                async with aiofiles.open(output_path, mode) as f:
                    async for chunk in resp.content.iter_chunked(Config.CHUNK_SIZE):
                        await f.write(chunk)
                        if pbar:
                            pbar.update(len(chunk))
                
                self.state.mark_file_complete(url)
                logging.info(f" Downloaded: {output_path.name}")
                return True
                
        except Exception as e:
            logging.error(f"Error downloading {url}: {e}")
            return False
    
    async def batch_download(self, urls, method='wget', username=None, password=None):
        """Download multiple files with concurrency control"""
        semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT)
        
        async def download_with_limit(url):
            async with semaphore:
                filename = Path(urlparse(url).path).name
                output_path = self.download_dir / filename
                
                if method == 'wget' and username and password:
                    return await self.download_with_wget(url, output_path, username, password)
                else:
                    return await self.download_with_aiohttp(url, output_path)
        
        tasks = [download_with_limit(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        logging.info(f"Batch complete: {success_count}/{len(urls)} successful")
        
        # Save state after batch
        await self.state.save_state()
        
        return success_count

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class UAVSARPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, download_dir, geojson_file=None, use_wget=True):
        self.download_dir = download_dir
        self.geojson_file = geojson_file
        self.use_wget = use_wget
        
        # Initialize components
        self.state = StateManager()
        self.username, self.password = AuthenticationManager.get_credentials()
        
        # Setup logging
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        log_file = f"{Config.LOG_DIR}/uavsar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    async def run(self):
        """Execute the complete acquisition pipeline"""
        # Setup HTTP session
        conn = aiohttp.TCPConnector(limit=Config.MAX_CONCURRENT, ssl=False)
        auth = BasicAuth(self.username, self.password)
        timeout = aiohttp.ClientTimeout(total=Config.TIMEOUT_SECONDS)
        
        async with aiohttp.ClientSession(
            connector=conn, 
            auth=auth, 
            timeout=timeout,
            cookie_jar=aiohttp.CookieJar()
        ) as session:
            # Authenticate with Earthdata
            logging.info("Authenticating with [RESEARCH_INSTITUTION] Earthdata...")
            try:
                async with session.get(f"https://{Config.EARTHDATA_HOST}") as resp:
                    if resp.status == 200:
                        logging.info(" Authentication successful")
            except Exception as e:
                logging.error(f"Authentication failed: {e}")
                return
            
            # Initialize managers
            discovery = FileDiscovery(session)
            downloader = DownloadManager(session, self.download_dir, self.state)
            
            # Load target list
            if self.geojson_file:
                urls = await self.load_from_geojson(discovery)
            else:
                urls = await self.discover_all_files(discovery)
            
            # Filter and download
            filtered_urls = FileDiscovery.filter_by_extension(urls)
            logging.info(f"Filtered to {len(filtered_urls)} files")
            
            # Download in batches
            for i in range(0, len(filtered_urls), Config.BATCH_SIZE):
                batch = filtered_urls[i:i+Config.BATCH_SIZE]
                batch_num = i // Config.BATCH_SIZE + 1
                total_batches = (len(filtered_urls) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
                
                logging.info(f"\n{'='*80}")
                logging.info(f"BATCH {batch_num}/{total_batches} - {len(batch)} files")
                logging.info(f"{'='*80}")
                
                method = 'wget' if self.use_wget else 'aiohttp'
                await downloader.batch_download(
                    batch, 
                    method=method,
                    username=self.username,
                    password=self.password
                )
                
                self.state.last_completed_batch = batch_num
                await self.state.save_state()
    
    async def load_from_geojson(self, discovery):
        """Load Line IDs from GeoJSON and discover files"""
        with open(self.geojson_file, 'r') as f:
            data = json.load(f)
        
        all_urls = []
        for feature in data['features']:
            line_id = feature['properties'].get('Line ID')
            if not line_id:
                continue
            
            # Discover directories for this Line ID
            root_listing = await discovery.get_directory_listing(Config.UAVSAR_BASE_URL + "/")
            matching_dirs = [d for d in root_listing if line_id in d]
            
            for dir_name in matching_dirs:
                dir_url = f"{Config.UAVSAR_BASE_URL}/{dir_name}"
                files = await discovery.get_directory_listing(dir_url)
                all_urls.extend([f"{dir_url}{f}" for f in files 
                               if not f.endswith('/')])
        
        return all_urls
    
    async def discover_all_files(self, discovery):
        """Discover all available files (fallback mode)"""
        root_listing = await discovery.get_directory_listing(Config.UAVSAR_BASE_URL + "/")
        all_urls = []
        
        for dir_name in root_listing:
            if not dir_name.endswith('/'):
                continue
            
            dir_url = f"{Config.UAVSAR_BASE_URL}/{dir_name}"
            files = await discovery.get_directory_listing(dir_url)
            all_urls.extend([f"{dir_url}{f}" for f in files 
                           if not f.endswith('/')])
        
        return all_urls

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Unified UAVSAR/NISAR Data Acquisition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download using GeoJSON target list
  %(prog)s --geojson filtered_uavsar_swaths.geojson --download-dir /data/uavsar
  
  # Resume interrupted download
  %(prog)s --resume
  
  # Rebuild state from existing files
  %(prog)s --rebuild-state /Volumes/JPL/uavsar /Volumes/All/zerocurtain/data/uavsar
  
  # Download with aiohttp instead of wget
  %(prog)s --method aiohttp
        """
    )
    
    parser.add_argument(
        '--download-dir',
        default=Config.DEFAULT_DOWNLOAD_DIR,
        help='Directory for downloaded files'
    )
    
    parser.add_argument(
        '--geojson',
        help='GeoJSON file with Line IDs to download'
    )
    
    parser.add_argument(
        '--method',
        choices=['wget', 'aiohttp'],
        default='wget',
        help='Download method (default: wget)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous state'
    )
    
    parser.add_argument(
        '--rebuild-state',
        nargs='+',
        metavar='DIR',
        help='Rebuild state from existing directories'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=Config.MAX_CONCURRENT,
        help=f'Maximum concurrent downloads (default: {Config.MAX_CONCURRENT})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=Config.BATCH_SIZE,
        help=f'Files per batch (default: {Config.BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--filter-extension',
        nargs='+',
        help='File extensions to download (overrides defaults)'
    )
    
    parser.add_argument(
        '--exclude-pattern',
        help='Regex pattern for files to exclude'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


async def main():
    args = parse_arguments()
    
    # Update configuration from arguments
    Config.MAX_CONCURRENT = args.max_concurrent
    Config.BATCH_SIZE = args.batch_size
    if args.filter_extension:
        Config.ALLOWED_EXTENSIONS = args.filter_extension
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle state rebuilding
    if args.rebuild_state:
        StateReconstructor.rebuild_state(args.rebuild_state)
        print("State rebuilt successfully")
        return
    
    # Initialize and run pipeline
    pipeline = UAVSARPipeline(
        download_dir=args.download_dir,
        geojson_file=args.geojson,
        use_wget=(args.method == 'wget')
    )
    
    await pipeline.run()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("UAVSAR/NISAR UNIFIED DATA ACQUISITION SYSTEM")
    print("="*80 + "\n")
    
    try:
        asyncio.run(main())
        print("\n Pipeline completed successfully")
    except KeyboardInterrupt:
        print("\n Process interrupted by user")
    except Exception as e:
        print(f"\n Critical error: {e}")
        import traceback
        traceback.print_exc()


# PHASE 1
# # Download UAVSAR GRD files
# python uavsar_unified_acquisition.py \
#     --geojson filtered_uavsar_swaths.geojson \
#     --download-dir /Volumes/JPL/uavsar \
#     --filter-extension .grd .ann
# # Download NISAR .h5 files (keep separate script)
# python uavsar_h5_downloader.py \
#     --flight-info-file flight_info.txt \
#     --output-dir /Volumes/JPL/nisar


# PHASE 2
# Convert GRD to GeoTIFF
# python uavsar_convert.py /Volumes/JPL/uavsar/uavsar_data_12345


# PHASE 3
# # Generate interferograms and displacement maps
# python uavsar_zerocurtain.py \
#     --ref_rslc ref.h5 \
#     --sec_rslc sec.h5 \
#     --work_dir ./processing_output
