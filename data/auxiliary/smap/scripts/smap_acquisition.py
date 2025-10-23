#!/usr/bin/env python3
"""
SMAP SPL4SMGP Data Acquisition Pipeline
Consolidates download logic with robust error handling and resume capability
"""

import os
import sys
import time
import json
import hashlib
import sqlite3
import requests
import netrc
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse
from urllib.request import urlopen, Request, build_opener, HTTPCookieProcessor
from urllib.error import HTTPError, URLError
import base64
from getpass import getpass
import socket
import random
import math

# Configuration
BASE_DIR = Path.home() / "arctic_zero_curtain_pipeline"
DATA_DIR = BASE_DIR / "data" / "auxiliary" / "smap"
RAW_DIR = DATA_DIR / "raw"
LOG_DIR = BASE_DIR / "logs" / "smap"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# Create directories
for d in [RAW_DIR, LOG_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Download parameters
DOWNLOAD_TIMEOUT = 120
STALL_TIMEOUT = 60
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.5
CHUNK_SIZE = 1024 * 1024  # 1MB
SESSION_REFRESH_INTERVAL = 3600

# SMAP collection parameters
SHORT_NAME = 'SPL4SMGP'
VERSION = '007'
TIME_START = '2015-03-31T00:00:00Z'
TIME_END = '2024-12-31T23:59:59Z'
BOUNDING_BOX = '-180,49,180,85.044'  # Circumarctic

# CMR endpoints
CMR_URL = 'https://cmr.earthdata.[RESEARCH_INSTITUTION_DOMAIN]'
URS_URL = 'https://urs.earthdata.[RESEARCH_INSTITUTION_DOMAIN]'
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = f'{CMR_URL}/search/granules.json?&sort_key[]=start_date&sort_key[]=producer_granule_id&page_size={CMR_PAGE_SIZE}'

# SQLite tracking database
DB_PATH = CHECKPOINT_DIR / "smap_downloads.db"

class SMAPDownloader:
    """Robust SMAP data downloader with SQLite tracking"""
    
    def __init__(self):
        self.credentials = None
        self.token = None
        self.last_auth_time = 0
        self.db_path = DB_PATH  # Instance attribute for backward compatibility
        self.stats = {
            'success': 0,
            'skipped': 0,
            'failed': 0,
            'resumed': 0,
            'total_bytes': 0,
            'start_time': time.time()
        }
        
        # Initialize database
        self.init_database()
        
        # Get credentials
        self.get_credentials()
    
    def init_database(self):
        """Initialize SQLite tracking database"""
        conn = sqlite3.connect(DB_PATH)  # Use module-level constant
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS downloads (
                filename TEXT PRIMARY KEY,
                url TEXT,
                status TEXT,
                size_bytes INTEGER,
                download_date TEXT,
                checksum TEXT
            )
        """)
        conn.commit()
        conn.close()
        
    def get_credentials(self):
        """Get Earthdata credentials from .netrc or prompt"""
        try:
            info = netrc.netrc()
            username, account, password = info.authenticators(urlparse(URS_URL).hostname)
            if username == 'token':
                self.token = password
            else:
                credentials = f'{username}:{password}'
                self.credentials = base64.b64encode(credentials.encode('ascii')).decode('ascii')
        except Exception as e:
            print(f"Could not read .netrc: {e}")
            username = input('Earthdata username (or press Return for token): ')
            if len(username):
                password = getpass('password: ')
                credentials = f'{username}:{password}'
                self.credentials = base64.b64encode(credentials.encode('ascii')).decode('ascii')
            else:
                self.token = getpass('bearer token: ')
    
    def is_downloaded(self, filename):
        """Check if file is already downloaded and verified"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT status, checksum FROM downloads WHERE filename=?", (filename,))
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0] == "completed":
            # Verify file still exists and matches checksum
            filepath = RAW_DIR / filename
            if filepath.exists():
                if row[1]:
                    current_hash = self.calculate_checksum(filepath)
                    return current_hash == row[1]
                return True
        return False
    
    def calculate_checksum(self, filepath):
        """Calculate SHA-256 checksum"""
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                h.update(chunk)
        return h.hexdigest()
    
    def mark_download(self, filename, url, status, size=0, checksum=None):
        """Record download status in database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO downloads (filename, url, status, size_bytes, download_date, checksum)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (filename, url, status, size, datetime.now().isoformat(), checksum))
        conn.commit()
        conn.close()
    
    def build_cmr_query(self):
        """Build CMR query URL"""
        params = f'&short_name={SHORT_NAME}'
        params += self.build_version_params(VERSION)
        if TIME_START or TIME_END:
            params += f'&temporal[]={TIME_START},{TIME_END}'
        if BOUNDING_BOX:
            params += f'&bounding_box={BOUNDING_BOX}'
        params += '&provider=NSIDC_ECS'
        return CMR_FILE_URL + params
    
    def build_version_params(self, version):
        """Build version query parameters"""
        desired_pad_length = 3
        version = str(int(version))
        query_params = ''
        while len(version) <= desired_pad_length:
            padded_version = version.zfill(desired_pad_length)
            query_params += f'&version={padded_version}'
            desired_pad_length -= 1
        return query_params
    
    def query_cmr(self):
        """Query CMR for SMAP granules"""
        print(f"\n[CMR] Querying SMAP {SHORT_NAME}.{VERSION}")
        print(f"[CMR] Time range: {TIME_START} to {TIME_END}")
        print(f"[CMR] Bounding box: {BOUNDING_BOX}")
        
        cmr_query_url = self.build_cmr_query()
        print(f"[CMR] Query URL: {cmr_query_url}\n")
        
        urls = []
        page_id = None
        page_count = 0
        
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        while True:
            req = Request(cmr_query_url)
            if page_id:
                req.add_header('cmr-search-after', page_id)
            
            try:
                response = urlopen(req, context=ctx)
            except Exception as e:
                print(f"[ERROR] CMR query failed: {e}")
                sys.exit(1)
            
            headers = {k.lower(): v for k, v in dict(response.info()).items()}
            
            if not page_id:
                hits = int(headers['cmr-hits'])
                print(f"[CMR] Found {hits:,} granules")
            
            page_id = headers.get('cmr-search-after')
            page_count += 1
            
            search_page = json.loads(response.read().decode('utf-8'))
            page_urls = self.extract_urls(search_page)
            
            if not page_urls:
                break
            
            if hits > CMR_PAGE_SIZE:
                print(f"[CMR] Processing page {page_count}...")
            
            urls += page_urls
        
        print(f"[CMR] Retrieved {len(urls):,} download URLs\n")
        return urls
    
    def extract_urls(self, search_results):
        """Extract download URLs from CMR response"""
        if 'feed' not in search_results or 'entry' not in search_results['feed']:
            return []
        
        urls = []
        unique_filenames = set()
        
        for entry in search_results['feed']['entry']:
            if 'links' not in entry:
                continue
            
            for link in entry['links']:
                if 'href' not in link:
                    continue
                if 'inherited' in link and link['inherited']:
                    continue
                if 'rel' in link and 'data#' not in link['rel']:
                    continue
                if 'title' in link and 'opendap' in link['title'].lower():
                    continue
                
                href = link['href']
                filename = href.split('/')[-1]
                
                if not filename.endswith(('.h5', '.he5')):
                    continue
                
                if filename in unique_filenames:
                    continue
                
                unique_filenames.add(filename)
                urls.append(href)
        
        return urls
    
    def download_file(self, url):
        """Download single file with resume capability"""
        filename = url.split('/')[-1]
        filepath = RAW_DIR / filename
        temp_filepath = RAW_DIR / f"{filename}.part"
        
        # Check if already downloaded
        if self.is_downloaded(filename):
            print(f"[SKIP] {filename} - already downloaded")
            self.stats['skipped'] += 1
            return True
        
        # Check for partial download
        resume_position = 0
        if temp_filepath.exists():
            resume_position = temp_filepath.stat().st_size
            if resume_position > 0:
                print(f"[RESUME] {filename} from {resume_position:,} bytes")
                self.stats['resumed'] += 1
        
        # Download with retries
        for attempt in range(MAX_RETRIES):
            try:
                success = self._download_attempt(url, filepath, temp_filepath, resume_position)
                if success:
                    # Calculate checksum
                    checksum = self.calculate_checksum(filepath)
                    file_size = filepath.stat().st_size
                    
                    # Record in database
                    self.mark_download(filename, url, "completed", file_size, checksum)
                    self.stats['success'] += 1
                    self.stats['total_bytes'] += file_size
                    
                    print(f"[OK] {filename} ({file_size:,} bytes)")
                    return True
                
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = BACKOFF_FACTOR ** attempt * (1 + random.random() * 0.1)
                    print(f"[RETRY] Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                    print(f"[RETRY] Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                else:
                    print(f"[FAIL] {filename} - {e}")
                    self.mark_download(filename, url, "failed")
                    self.stats['failed'] += 1
                    return False
        
        return False
    
    def _download_attempt(self, url, filepath, temp_filepath, resume_position):
        """Single download attempt"""
        # Refresh authentication if needed
        if time.time() - self.last_auth_time > SESSION_REFRESH_INTERVAL:
            self.get_credentials()
            self.last_auth_time = time.time()
        
        # Build request
        req = Request(url)
        if self.token:
            req.add_header('Authorization', f'Bearer {self.token}')
        elif self.credentials:
            req.add_header('Authorization', f'Basic {self.credentials}')
        
        if resume_position > 0:
            req.add_header('Range', f'bytes={resume_position}-')
        
        opener = build_opener(HTTPCookieProcessor())
        socket.setdefaulttimeout(DOWNLOAD_TIMEOUT)
        
        response = opener.open(req)
        total_size = int(response.headers.get('content-length', 0))
        
        # Open file for writing/appending
        mode = 'ab' if resume_position > 0 else 'wb'
        downloaded = resume_position
        
        with open(temp_filepath, mode) as f:
            last_update = time.time()
            last_downloaded = downloaded
            stall_counter = 0
            
            for data in iter(lambda: response.read(CHUNK_SIZE), b''):
                f.write(data)
                downloaded += len(data)
                
                # Check for stall
                current_time = time.time()
                if current_time - last_update > 5:
                    if downloaded == last_downloaded:
                        stall_counter += 1
                        if stall_counter >= (STALL_TIMEOUT // 5):
                            raise TimeoutError(f"Download stalled for {STALL_TIMEOUT}s")
                    else:
                        stall_counter = 0
                    
                    last_update = current_time
                    last_downloaded = downloaded
        
        # Move to final location
        temp_filepath.rename(filepath)
        return True
    
    def run(self):
        """Execute full download pipeline"""
        print("=" * 80)
        print("SMAP SPL4SMGP Data Acquisition Pipeline")
        print("=" * 80)
        
        # Query CMR
        urls = self.query_cmr()
        
        if not urls:
            print("[WARNING] No URLs found to download")
            return
        
        # Download files
        print(f"\n[DOWNLOAD] Starting download of {len(urls):,} files\n")
        
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] {url.split('/')[-1]}")
            self.download_file(url)
        
        # Print statistics
        self.print_stats()
    
    def print_stats(self):
        """Print download statistics"""
        total_time = time.time() - self.stats['start_time']
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n" + "=" * 80)
        print("Download Summary")
        print("=" * 80)
        print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Successfully downloaded: {self.stats['success']:,}")
        print(f"Resumed: {self.stats['resumed']:,}")
        print(f"Skipped (already downloaded): {self.stats['skipped']:,}")
        print(f"Failed: {self.stats['failed']:,}")
        print(f"Total data: {self.stats['total_bytes'] / (1024**3):.2f} GB")
        
        if total_time > 0 and self.stats['total_bytes'] > 0:
            speed = self.stats['total_bytes'] / total_time / (1024**2)
            print(f"Average speed: {speed:.2f} MB/s")
        
        print("=" * 80)


def main():
    """Main entry point"""
    downloader = SMAPDownloader()
    
    try:
        downloader.run()
        return 0
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Download stopped by user")
        downloader.print_stats()
        return 130
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())