#!/usr/bin/env python3
"""
modules/acquisition.py
UAVSAR/NISAR Data Acquisition Module - COMPLETE IMPLEMENTATION
Handles authentication redirects properly

"""

import os
import json
import logging
import netrc
import re
import urllib.request
import urllib.error
import http.cookiejar
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time
import ssl


class GeographicFilter:
    """Filter data by geographic bounds"""
    
    def __init__(self, bounds: Dict):
        self.min_lon = bounds['min_lon']
        self.max_lon = bounds['max_lon']
        self.min_lat = bounds['min_lat']
        self.max_lat = bounds['max_lat']
    
    def is_within_bounds(self, metadata: Dict) -> bool:
        """Check if data falls within geographic bounds"""
        lat_keys = ['center_lat', 'min_lat', 'latitude', 'lat']
        lon_keys = ['center_lon', 'min_lon', 'longitude', 'lon']
        
        lat = None
        lon = None
        
        for key in lat_keys:
            if key in metadata and metadata[key] is not None:
                lat = float(metadata[key])
                break
        
        for key in lon_keys:
            if key in metadata and metadata[key] is not None:
                lon = float(metadata[key])
                break
        
        if lat is None or lon is None:
            return True
        
        if lat < self.min_lat or lat > self.max_lat:
            return False
        
        if lon < self.min_lon or lon > self.max_lon:
            return False
        
        return True


class DownloadStateManager:
    """Manages download state for resumable operations"""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.downloaded_files = set()
        self.failed_files = set()
        self.load_state()
    
    def load_state(self):
        """Load state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.downloaded_files = set(state.get('downloaded', []))
                    self.failed_files = set(state.get('failed', []))
            except Exception as e:
                logging.warning(f"Failed to load state: {e}")
    
    def save_state(self):
        """Save state to disk"""
        try:
            state = {
                'downloaded': list(self.downloaded_files),
                'failed': list(self.failed_files),
                'timestamp': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save state: {e}")
    
    def mark_downloaded(self, filename: str):
        self.downloaded_files.add(filename)
        if filename in self.failed_files:
            self.failed_files.remove(filename)
        self.save_state()
    
    def mark_failed(self, filename: str):
        self.failed_files.add(filename)
        self.save_state()
    
    def is_downloaded(self, filename: str) -> bool:
        return filename in self.downloaded_files


class UAVSARNISARAcquisitionManager:
    """Complete acquisition manager for UAVSAR and NISAR data"""
    
    def __init__(self, 
                 output_dir: str,
                 geographic_bounds: Dict,
                 max_workers: int = 8,
                 logger: Optional[logging.Logger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.geographic_filter = GeographicFilter(geographic_bounds)
        self.max_workers = max_workers
        self.logger = logger or logging.getLogger(__name__)
        
        state_file = self.output_dir / '.download_state.json'
        self.state_manager = DownloadStateManager(state_file)
        
        self.stats = {
            'urls_generated': 0,
            'urls_filtered': 0,
            'files_downloaded': 0,
            'files_skipped': 0,
            'files_failed': 0
        }
        
        self._setup_authentication()
        
        self.logger.info("UAVSARNISARAcquisitionManager initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Geographic bounds: {geographic_bounds}")
    
    def _setup_authentication(self):
        """Setup [RESEARCH_INSTITUTION] Earthdata authentication with cookie handling"""
        try:
            nrc = netrc.netrc()
            auth_info = nrc.authenticators('urs.earthdata.[RESEARCH_INSTITUTION_DOMAIN]')
            if auth_info:
                self.username = auth_info[0]
                self.password = auth_info[2]
                
                # Setup cookie jar for session persistence
                cookie_jar = http.cookiejar.CookieJar()
                cookie_handler = urllib.request.HTTPCookieProcessor(cookie_jar)
                
                # Setup password manager
                password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
                password_mgr.add_password(None, 'https://urs.earthdata.[RESEARCH_INSTITUTION_DOMAIN]', 
                                         self.username, self.password)
                password_mgr.add_password(None, 'https://uavsar.asf.alaska.edu',
                                         self.username, self.password)
                
                auth_handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
                
                # Create SSL context that doesn't verify certificates
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                
                https_handler = urllib.request.HTTPSHandler(context=ctx)
                
                # Build opener with all handlers (order matters!)
                opener = urllib.request.build_opener(
                    cookie_handler,
                    auth_handler,
                    https_handler,
                    urllib.request.HTTPRedirectHandler()
                )
                
                # Install opener globally
                urllib.request.install_opener(opener)
                
                self.logger.info(" Authentication configured with cookie/redirect handling")
                self.logger.info(f"  Username: {self.username}")
                return
        except Exception as e:
            self.logger.warning(f"Could not load credentials from .netrc: {e}")
        
        self.username = os.environ.get('EARTHDATA_USERNAME')
        self.password = os.environ.get('EARTHDATA_PASSWORD')
        
        if self.username and self.password:
            self.logger.info(" Loaded credentials from environment variables")
            return
        
        self.logger.warning(" No credentials found")
        self.logger.warning("  Run: python3 setup_earthdata_auth.py")
        self.username = None
        self.password = None
    
    def acquire_uavsar_from_wget_file(self, wget_file: str, file_types: List[str] = None) -> Dict:
        """
        Acquire UAVSAR data from wget file
        
        Parameters:
        -----------
        wget_file : str
            Path to wget file
        file_types : list, optional
            List of file extensions to download (e.g., ['h5', 'ann', 'grd'])
            If None, downloads only H5 files
        """
        
        self.logger.info("="*80)
        self.logger.info("UAVSAR DATA ACQUISITION FROM WGET FILE")
        self.logger.info("="*80)
        
        if not os.path.exists(wget_file):
            self.logger.error(f"Wget file not found: {wget_file}")
            return {'files_downloaded': 0, 'error': 'file_not_found'}
        
        # Parse wget file
        urls = self._parse_wget_file(wget_file)
        
        self.logger.info(f"Parsed {len(urls)} URLs from {wget_file}")
        
        # Filter by file type
        if file_types is None:
            file_types = ['h5']  # Default to H5 only
        
        filtered_urls = []
        for url_info in urls:
            url = url_info['url']
            for ftype in file_types:
                if url.endswith(f'.{ftype}') or f'.{ftype}' in url:
                    filtered_urls.append(url_info)
                    break
        
        self.logger.info(f"Selected {len(filtered_urls)} files matching types: {file_types}")
        
        if not filtered_urls:
            self.logger.warning("No matching files found")
            return {'files_downloaded': 0}
        
        # Download files
        downloaded = self._download_files(filtered_urls)
        
        self.stats['files_downloaded'] += downloaded
        
        return {
            'files_downloaded': downloaded,
            'urls_total': len(urls),
            'urls_filtered': len(filtered_urls)
        }
    
    def _parse_wget_file(self, wget_file: str) -> List[Dict]:
        """Parse wget file and extract URLs"""
        
        with open(wget_file, 'r') as f:
            content = f.read()
        
        # Extract all wget URLs
        url_pattern = r'wget\s+(https://[^\s]+)'
        urls = re.findall(url_pattern, content)
        
        url_list = []
        for url in urls:
            filename = os.path.basename(url)
            url_list.append({
                'url': url,
                'filename': filename,
                'metadata': {}
            })
        
        self.stats['urls_generated'] = len(url_list)
        
        return url_list
    
    def acquire_uavsar(self, 
                       geojson_filter: Optional[str] = None,
                       flight_lines_filter: Optional[str] = None) -> Dict:
        """Acquire UAVSAR data"""
        
        # Check for wget file
        wget_files = ['uavsar_wget_parallel.txt', 'uavsar_wget_parallet.txt']
        
        for wget_file in wget_files:
            if os.path.exists(wget_file):
                self.logger.info(f"Found wget file: {wget_file}")
                return self.acquire_uavsar_from_wget_file(wget_file)
        
        self.logger.warning("No wget file found")
        return {'files_downloaded': 0}
    
    def _download_files(self, urls: List[Dict]) -> int:
        """Download files using Python urllib with progress tracking"""
        
        if not urls:
            self.logger.info("No URLs to download")
            return 0
        
        self.logger.info(f"Starting download of {len(urls)} files")
        self.logger.info("Press Ctrl+C to pause (progress is saved)")
        
        downloaded_count = 0
        
        for idx, url_info in enumerate(urls, 1):
            self.logger.info(f"\n[{idx}/{len(urls)}] {url_info['filename']}")
            success = self._download_single_file(url_info)
            if success:
                downloaded_count += 1
                self.logger.info(f"  Progress: {downloaded_count}/{idx} successful ({100*downloaded_count/idx:.1f}%)")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"DOWNLOAD COMPLETE: {downloaded_count}/{len(urls)} files")
        self.logger.info(f"{'='*80}")
        
        return downloaded_count
    
    def _download_single_file(self, url_info: Dict) -> bool:
        """Download a single file with authentication"""
        
        url = url_info['url']
        filename = url_info['filename']
        output_path = self.output_dir / filename
        
        # Skip if already downloaded
        if self.state_manager.is_downloaded(filename):
            self.logger.info(f"   Already downloaded")
            self.stats['files_skipped'] += 1
            return True
        
        # Skip if file exists and is non-empty
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            if size_mb > 0.1:  # At least 100 KB
                self.logger.info(f"   Already exists ({size_mb:.1f} MB)")
                self.state_manager.mark_downloaded(filename)
                self.stats['files_skipped'] += 1
                return True
        
        # Download
        try:
            self.logger.info(f"   Downloading...")
            start_time = time.time()
            
            # Download with retry logic
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    # Open URL with authentication
                    response = urllib.request.urlopen(url, timeout=600)
                    
                    # Get content length if available
                    content_length = response.headers.get('Content-Length')
                    total_size = int(content_length) if content_length else None
                    
                    # Read and write in chunks with progress
                    chunk_size = 1024 * 1024  # 1 MB chunks
                    downloaded_size = 0
                    last_progress_mb = 0
                    
                    with open(output_path, 'wb') as f:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # Log progress every 10 MB
                            current_mb = downloaded_size / (1024 * 1024)
                            if current_mb - last_progress_mb >= 10:
                                if total_size:
                                    pct = 100 * downloaded_size / total_size
                                    self.logger.info(f"    {current_mb:.0f} MB / {total_size/(1024*1024):.0f} MB ({pct:.1f}%)")
                                else:
                                    self.logger.info(f"    {current_mb:.0f} MB downloaded")
                                last_progress_mb = current_mb
                    
                    # Success!
                    elapsed = time.time() - start_time
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    speed_mbps = size_mb / elapsed if elapsed > 0 else 0
                    
                    self.logger.info(f"   Complete: {size_mb:.1f} MB in {elapsed:.1f}s ({speed_mbps:.2f} MB/s)")
                    self.state_manager.mark_downloaded(filename)
                    return True
                
                except urllib.error.HTTPError as e:
                    if e.code in [401, 403]:
                        self.logger.error(f"   Authentication failed (HTTP {e.code})")
                        self.logger.error("    Check credentials: python3 setup_earthdata_auth.py")
                        self.state_manager.mark_failed(filename)
                        self.stats['files_failed'] += 1
                        return False
                    
                    if e.code == 404:
                        self.logger.error(f"   File not found (HTTP 404)")
                        self.state_manager.mark_failed(filename)
                        self.stats['files_failed'] += 1
                        return False
                    
                    if attempt < max_retries - 1:
                        self.logger.warning(f"   HTTP {e.code}, retrying in {retry_delay}s... (attempt {attempt+1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        raise
                
                except (urllib.error.URLError, TimeoutError, ConnectionResetError) as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"   Network error, retrying in {retry_delay}s... (attempt {attempt+1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        raise
            
        except KeyboardInterrupt:
            self.logger.info(f"  ⏸ Download paused by user")
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            raise
        
        except Exception as e:
            self.logger.error(f"   Error: {e}")
            self.state_manager.mark_failed(filename)
            self.stats['files_failed'] += 1
            
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            
            return False
