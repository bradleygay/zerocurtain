#!/usr/bin/env python3
"""
UAVSAR/NISAR Retrieval Manager - Consolidated Script
Integrates flight metadata parsing, Release pattern discovery, URL generation, and parallel downloads
"""

import argparse
import os
import re
import json
import time
import requests
import logging
import datetime
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
import urllib.request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("uavsar_retrieval_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_MAX_WORKERS = 8
WGET_TIMEOUT = 300
WGET_RETRIES = 5
CACHE_FILE = "release_patterns_cache.json"


class UAVSARRetrievalManager:
    """Consolidated manager for UAVSAR/NISAR data retrieval"""
    
    def __init__(self, output_dir="./downloads", timeout=5, max_workers=8):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_download_url = "https://downloaduav.jpl.[RESEARCH_INSTITUTION_DOMAIN]"
        self.dithering_types = ["CX", "CG", "CD"]
        self.frequency_modes = ["129A", "129B", "138A", "138B", "143A", "143B"]
        self.timeout = timeout
        self.max_workers = max_workers
        
        # Generate all possible release patterns
        self.release_patterns = self._generate_release_patterns()
        
        # Cache for validated release patterns
        self.validated_release_patterns = {}
        self._load_release_patterns_cache()
        
        # Flight data storage
        self.flight_data = []
        self.flight_lines_filter = None
        
    def _generate_release_patterns(self):
        """Generate all possible Release pattern combinations"""
        patterns = ["Release"]
        for num in range(10):
            for char in "abcdefghijklmnopqrstuvwxyz":
                patterns.append(f"Release{num}{char}")
        return patterns
    
    def _load_release_patterns_cache(self):
        """Load cached release patterns from previous runs"""
        cache_path = self.output_dir / CACHE_FILE
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    self.validated_release_patterns = json.load(f)
                logger.info(f"Loaded {len(self.validated_release_patterns)} cached release patterns")
            except Exception as e:
                logger.warning(f"Error loading release patterns cache: {str(e)}")
    
    def _save_release_patterns_cache(self):
        """Save validated release patterns to cache"""
        cache_path = self.output_dir / CACHE_FILE
        with open(cache_path, 'w') as f:
            json.dump(self.validated_release_patterns, f, indent=2)
        logger.info(f"Saved release patterns cache to {cache_path}")
    
    def url_exists(self, url):
        """Test if a URL exists by trying to open it"""
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return response.status == 200
        except (HTTPError, URLError):
            return False
        except Exception as e:
            logger.debug(f"Error checking URL {url}: {str(e)}")
            return False
    
    def find_valid_release_pattern(self, job_name):
        """Brute force approach to find a valid Release pattern"""
        # Check cache first
        if job_name in self.validated_release_patterns:
            pattern = self.validated_release_patterns[job_name]
            logger.info(f"Using cached release pattern '{pattern}' for {job_name}")
            
            # Verify cached pattern still works
            base_name = self._extract_base_name_from_job(job_name)
            version = job_name.split('_')[-1]
            test_file = f"{base_name}_CX_129A_{version}.ann"
            test_url = f"{self.base_download_url}/{pattern}/{job_name}/{test_file}"
            
            if self.url_exists(test_url):
                return pattern
            else:
                logger.warning(f"Cached pattern '{pattern}' no longer valid. Re-checking.")
        
        logger.info(f"Searching for valid release pattern for {job_name}...")
        
        # Extract components for URL construction
        base_name = self._extract_base_name_from_job(job_name)
        if not base_name:
            logger.error(f"Could not extract base name from job: {job_name}")
            return None
        
        version = job_name.split('_')[-1]
        test_file = f"{base_name}_CX_129A_{version}.ann"
        
        # Search through all release patterns
        for pattern in self.release_patterns:
            test_url = f"{self.base_download_url}/{pattern}/{job_name}/{test_file}"
            logger.debug(f"Testing pattern: {pattern}")
            
            if self.url_exists(test_url):
                # Verify with H5 file as well
                h5_test_file = f"{base_name}_CX_129_{version}.h5"
                h5_test_url = f"{self.base_download_url}/{pattern}/{job_name}/{h5_test_file}"
                
                if self.url_exists(h5_test_url):
                    logger.info(f"Found valid pattern: {pattern} for {job_name}")
                    self.validated_release_patterns[job_name] = pattern
                    self._save_release_patterns_cache()
                    return pattern
        
        logger.warning(f"Could not find valid release pattern for {job_name}")
        return None
    
    def _extract_base_name_from_job(self, job_name):
        """Extract the base filename from a job name"""
        if not job_name:
            return None
        parts = job_name.split('_')
        if len(parts) >= 7:
            return "_".join(parts[:-2])
        return None
    
    def _calculate_day_of_year(self, year, month, day):
        """Calculate day of year (1-366) from year, month, and day"""
        days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        # Adjust for leap year
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            days_in_month[2] = 29
        
        return sum(days_in_month[:month]) + day
    
    def parse_flight_info(self, flight_info_file):
        """Parse flight information from text file"""
        with open(flight_info_file, 'r') as f:
            text = f.read()
        
        # Split by flight line entries
        flight_blocks = re.split(r'(\w+_\d+)\s+\(\d+\)\s+-\s+', text)[1:]
        
        # Process in pairs (identifier, content)
        for i in range(0, len(flight_blocks), 2):
            if i+1 < len(flight_blocks):
                flight_id = flight_blocks[i]
                content = flight_blocks[i+1]
                
                # Apply filter if specified
                if self.flight_lines_filter and flight_id not in self.flight_lines_filter:
                    continue
                
                # Extract flight information
                for match in re.finditer(
                    r'PolSAR:\s+Flight\s+(\d+)\s+\((\d{4})-(\d{2})-(\d{2})\),\s+DT\s+(\d+),\s+v(\d+)',
                    content
                ):
                    flight_num, year, month_str, day_str = match.group(1), match.group(2), match.group(3), match.group(4)
                    dt, version = match.group(5), match.group(6)
                    
                    month, day = int(month_str), int(day_str)
                    day_of_year = self._calculate_day_of_year(int(year), month, day)
                    
                    self.flight_data.append({
                        'flight_line': flight_id,
                        'flight_num': flight_num,
                        'date': f"{year}-{month_str}-{day_str}",
                        'year': year[-2:],
                        'day': f"{day_of_year:03d}",
                        'dt': dt.zfill(3),
                        'version': version.zfill(2),
                        'mmdd': f"{month:02d}{day:02d}"
                    })
        
        logger.info(f"Parsed {len(self.flight_data)} flight entries")
        return self.flight_data
    
    def load_flight_lines_filter(self, flight_lines_file):
        """Load specific flight lines to process"""
        with open(flight_lines_file, 'r') as f:
            self.flight_lines_filter = set(line.strip() for line in f if line.strip())
        logger.info(f"Loaded {len(self.flight_lines_filter)} flight lines to process")
    
    def filter_newest_only(self):
        """Keep only the newest version for each flight line"""
        flight_line_groups = {}
        for data in self.flight_data:
            flight_line = data['flight_line']
            if flight_line not in flight_line_groups:
                flight_line_groups[flight_line] = []
            flight_line_groups[flight_line].append(data)
        
        newest_data = []
        for flight_line, group in flight_line_groups.items():
            group.sort(key=lambda x: x['date'], reverse=True)
            newest_data.append(group[0])
        
        self.flight_data = newest_data
        logger.info(f"Filtered to {len(self.flight_data)} newest entries")
    
    def generate_urls(self, modes=None, dithering=None, h5_only=True, frequency_subset=None):
        """Generate download URLs from flight data"""
        if modes is None:
            modes = ["129", "138", "143"]
        if dithering is None:
            dithering = self.dithering_types
        
        # Apply frequency subset filter
        if frequency_subset:
            modes = [m for m in modes if m.startswith(frequency_subset)]
        
        urls = []
        
        for data in self.flight_data:
            flight_line = data['flight_line']
            
            # Construct job name
            job_name = f"{flight_line}_{data['year']}{data['day']}_{data['dt']}_{data['year']}{data['mmdd']}_L090_CX_{data['version']}"
            
            # Find release pattern
            release_pattern = self.find_valid_release_pattern(job_name)
            if not release_pattern:
                logger.warning(f"Skipping {job_name} - no valid release pattern found")
                continue
            
            base_name = self._extract_base_name_from_job(job_name)
            version = data['version']
            
            # Generate URLs for all combinations
            for dither in dithering:
                for mode in modes:
                    freq_base = mode[:3]
                    
                    # H5 file
                    h5_filename = f"{base_name}_{dither}_{freq_base}_{version}.h5"
                    h5_url = f"{self.base_download_url}/{release_pattern}/{job_name}/{h5_filename}"
                    urls.append({
                        'url': h5_url,
                        'filename': h5_filename,
                        'job_name': job_name,
                        'flight_line': flight_line,
                        'type': 'h5'
                    })
                    
                    # Annotation files (if requested)
                    if not h5_only:
                        for suffix in ['A', 'B']:
                            freq_mode = f"{freq_base}{suffix}"
                            ann_filename = f"{base_name}_{dither}_{freq_mode}_{version}.ann"
                            ann_url = f"{self.base_download_url}/{release_pattern}/{job_name}/{ann_filename}"
                            urls.append({
                                'url': ann_url,
                                'filename': ann_filename,
                                'job_name': job_name,
                                'flight_line': flight_line,
                                'type': 'ann'
                            })
        
        logger.info(f"Generated {len(urls)} URLs for download")
        return urls
    
    def check_file_exists(self, url_info, reference_dir=None):
        """Check if file already exists in download or reference directory"""
        filename = url_info['filename']
        
        # Check download directory
        if (self.output_dir / filename).exists():
            return True
        
        # Check reference directory if provided
        if reference_dir:
            ref_path = Path(reference_dir) / filename
            if ref_path.exists():
                return True
        
        return False
    
    def download_file(self, url_info, reference_dir=None):
        """Download a single file using requests"""
        url = url_info['url']
        filename = url_info['filename']
        
        # Skip if file already exists
        if self.check_file_exists(url_info, reference_dir):
            logger.info(f"Skipping (exists): {filename}")
            return {'status': 'skipped', 'filename': filename}
        
        output_path = self.output_dir / filename
        
        try:
            logger.info(f"Downloading: {filename}")
            session = requests.Session()
            session.trust_env = True  # Use .netrc
            
            with session.get(url, stream=True, timeout=60) as r:
                if r.status_code == 404:
                    logger.warning(f"Not found: {filename}")
                    return {'status': 'not_found', 'filename': filename}
                
                r.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            logger.info(f"Success: {filename}")
            return {'status': 'success', 'filename': filename}
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {str(e)}")
            return {'status': 'error', 'filename': filename, 'error': str(e)}
    
    def download_all(self, urls, reference_dir=None):
        """Download all files using thread pool"""
        logger.info(f"Starting download of {len(urls)} files using {self.max_workers} workers")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self.download_file, url_info, reference_dir): url_info 
                for url_info in urls
            }
            
            for future in as_completed(future_to_url):
                result = future.result()
                results.append(result)
        
        # Summary
        success = sum(1 for r in results if r['status'] == 'success')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        not_found = sum(1 for r in results if r['status'] == 'not_found')
        errors = sum(1 for r in results if r['status'] == 'error')
        
        logger.info(f"Download summary:")
        logger.info(f"  Success: {success}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"  Not found: {not_found}")
        logger.info(f"  Errors: {errors}")
        
        return results
    
    def generate_wget_script(self, urls, script_name="download_uavsar.sh"):
        """Generate bash script with wget commands"""
        script_path = self.output_dir / script_name
        
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# UAVSAR/NISAR Download Script\n")
            f.write(f"# Generated: {datetime.datetime.now()}\n\n")
            f.write("LOG_FILE=\"uavsar_download.log\"\n")
            f.write("echo \"Download started at $(date)\" > $LOG_FILE\n\n")
            
            # Group by job name
            job_groups = {}
            for url_info in urls:
                job_name = url_info['job_name']
                if job_name not in job_groups:
                    job_groups[job_name] = []
                job_groups[job_name].append(url_info)
            
            # Write commands for each job
            for job_name, url_list in job_groups.items():
                f.write(f"\n# Job: {job_name}\n")
                f.write(f"mkdir -p downloads/{job_name}\n")
                
                for url_info in url_list:
                    f.write(f"wget -P downloads/{job_name} -nc \"{url_info['url']}\" || ")
                    f.write(f"echo \"Failed: {url_info['filename']}\" | tee -a $LOG_FILE\n")
            
            f.write("\necho \"Download completed at $(date)\" | tee -a $LOG_FILE\n")
        
        # Make executable
        os.chmod(script_path, 0o755)
        logger.info(f"Generated wget script: {script_path}")
        return script_path
    
    def setup_netrc(self):
        """Create .netrc file for Earthdata Login if needed"""
        netrc_path = Path.home() / ".netrc"
        
        if not netrc_path.exists():
            print("[RESEARCH_INSTITUTION] Earthdata Login credentials required.")
            username = input("Enter your Earthdata Login username: ")
            password = input("Enter your Earthdata Login password: ")
            
            with open(netrc_path, 'w') as netrc:
                netrc.write(f"machine urs.earthdata.[RESEARCH_INSTITUTION_DOMAIN]\n")
                netrc.write(f"  login {username}\n")
                netrc.write(f"  password {password}\n")
            
            os.chmod(netrc_path, 0o600)
            logger.info(f"Created .netrc file at {netrc_path}")


def main():
    parser = argparse.ArgumentParser(
        description='UAVSAR/NISAR Consolidated Retrieval Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input files
    parser.add_argument('--flight-info', required=True,
                        help='Path to flight_info_cleaned.txt')
    parser.add_argument('--flight-lines', 
                        help='Path to flight_lines_cleaned.txt (optional filter)')
    
    # Output configuration
    parser.add_argument('--output-dir', default='./downloads',
                        help='Output directory for downloads (default: ./downloads)')
    parser.add_argument('--reference-dir',
                        help='Reference directory to check for existing files')
    
    # Download options
    parser.add_argument('--modes', default='129,138,143',
                        help='Comma-separated frequency modes (default: 129,138,143)')
    parser.add_argument('--dithering', default='CX,CG,CD',
                        help='Comma-separated dithering types (default: CX,CG,CD)')
    parser.add_argument('--frequency-subset',
                        help='Download only specific frequency (e.g., 129)')
    parser.add_argument('--h5-only', action='store_true',
                        help='Download only .h5 files (not .ann files)')
    parser.add_argument('--newest-only', action='store_true',
                        help='Process only newest version for each flight line')
    
    # Processing options
    parser.add_argument('--max-workers', type=int, default=DEFAULT_MAX_WORKERS,
                        help=f'Number of parallel workers (default: {DEFAULT_MAX_WORKERS})')
    parser.add_argument('--timeout', type=int, default=5,
                        help='Timeout for URL checks in seconds (default: 5)')
    parser.add_argument('--use-cached-patterns', action='store_true',
                        help='Use cached release patterns without re-validation')
    
    # Script generation
    parser.add_argument('--generate-script-only', action='store_true',
                        help='Generate wget script without downloading')
    parser.add_argument('--script-name', default='download_uavsar.sh',
                        help='Name for generated wget script')
    
    # Other
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--setup-auth', action='store_true',
                        help='Setup .netrc authentication and exit')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup authentication if requested
    if args.setup_auth:
        manager = UAVSARRetrievalManager()
        manager.setup_netrc()
        return 0
    
    # Initialize manager
    manager = UAVSARRetrievalManager(
        output_dir=args.output_dir,
        timeout=args.timeout,
        max_workers=args.max_workers
    )
    
    # Load flight lines filter if provided
    if args.flight_lines:
        manager.load_flight_lines_filter(args.flight_lines)
    
    # Parse flight information
    manager.parse_flight_info(args.flight_info)
    
    # Filter to newest only if requested
    if args.newest_only:
        manager.filter_newest_only()
    
    # Parse modes and dithering
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    dithering = [d.strip() for d in args.dithering.split(',') if d.strip()]
    
    # Generate URLs
    urls = manager.generate_urls(
        modes=modes,
        dithering=dithering,
        h5_only=args.h5_only,
        frequency_subset=args.frequency_subset
    )
    
    if not urls:
        logger.error("No URLs generated. Check flight info and patterns.")
        return 1
    
    # Generate wget script
    script_path = manager.generate_wget_script(urls, args.script_name)
    
    if args.generate_script_only:
        logger.info(f"Script generated: {script_path}")
        logger.info(f"To download, run: bash {script_path}")
        return 0
    
    # Download files
    results = manager.download_all(urls, args.reference_dir)
    
    logger.info(f"Download complete. Files saved to: {manager.output_dir}")
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit(1)


# # Basic usage - download all data
# python uavsar_nisar_retrieval_manager.py \
#   --flight-info flight_info_cleaned.txt \
#   --output-dir ./downloads

# # Filter specific flight lines, newest versions only
# python uavsar_nisar_retrieval_manager.py \
#   --flight-info flight_info_cleaned.txt \
#   --flight-lines flight_lines_cleaned.txt \
#   --newest-only \
#   --output-dir ./downloads

# # Download only 129 frequency .h5 files
# python uavsar_nisar_retrieval_manager.py \
#   --flight-info flight_info_cleaned.txt \
#   --frequency-subset 129 \
#   --h5-only \
#   --max-workers 16

# # Generate wget script without downloading
# python uavsar_nisar_retrieval_manager.py \
#   --flight-info flight_info_cleaned.txt \
#   --generate-script-only \
#   --script-name download_129_only.sh

# # Setup authentication
# python uavsar_nisar_retrieval_manager.py --setup-auth