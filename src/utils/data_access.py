"""
Data Access Layer for Arctic Zero-Curtain Pipeline
Handles transparent retrieval of large files from Google Drive cache.
"""

import json
import os
import zipfile
from pathlib import Path
from typing import Optional, Dict
import requests
from tqdm import tqdm


class DataAccessError(Exception):
    """Raised when data access operations fail."""
    pass


class RemoteDataHandler:
    """
    Manages transparent access to large datasets stored on Google Drive.
    
    Files are downloaded on-demand and cached locally. Subsequent access
    uses the local cache automatically.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize data handler.
        
        Args:
            cache_dir: Local cache directory. Defaults to project root.
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent
        
        self.cache_dir = Path(cache_dir)
        self.session = requests.Session()
        self.extracted_archives = set()
        
    def get_file(self, filepath: str) -> Path:
        """
        Get file path, downloading from Google Drive if necessary.
        
        Args:
            filepath: Relative path to file (e.g., 'data/arcticdem/arcticdem.parquet')
        
        Returns:
            Path: Absolute path to file (either cached or downloaded)
        
        Raises:
            DataAccessError: If file cannot be accessed
        
        Example:
            >>> handler = RemoteDataHandler()
            >>> data_path = handler.get_file('data/arcticdem/arcticdem.parquet')
            >>> df = pd.read_parquet(data_path)
        """
        full_path = self.cache_dir / filepath
        
        # File already cached locally
        if full_path.exists() and full_path.stat().st_size > 1024:
            return full_path
        
        # Check for placeholder file
        placeholder_path = full_path.with_suffix(full_path.suffix + '.gdrive')
        
        if not placeholder_path.exists():
            raise DataAccessError(
                f"File not found and no placeholder exists: {filepath}\n"
                f"Expected placeholder at: {placeholder_path}"
            )
        
        # Load placeholder metadata
        with open(placeholder_path, 'r') as f:
            metadata = json.load(f)
        
        file_id = metadata.get('gdrive_file_id')
        
        if not file_id or file_id == "REPLACE_WITH_ACTUAL_FILE_ID":
            raise DataAccessError(
                f"Google Drive file ID not configured for: {filepath}\n"
                f"Please update the placeholder file: {placeholder_path}"
            )
        
        # Download file
        print(f"\nDownloading {filepath} from Google Drive...")
        success = self._download_from_gdrive(file_id, full_path)
        
        if not success:
            raise DataAccessError(f"Failed to download file: {filepath}")
        
        # Extract if archive
        if metadata.get('type') == 'zip_archive':
            extract_to = self.cache_dir / metadata.get('extract_to', '')
            self._extract_zip(full_path, extract_to)
        
        return full_path
    
    def _download_from_gdrive(self, file_id: str, destination: Path) -> bool:
        """Download file from Google Drive with progress bar."""
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        try:
            response = self.session.get(url, stream=True)
            
            # Handle large file confirmation
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            
            if token:
                params = {'id': file_id, 'confirm': token}
                response = self.session.get(url, params=params, stream=True)
            
            if response.status_code != 200:
                return False
            
            # Create parent directories
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress bar
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB
            
            with open(destination, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=destination.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f" Downloaded: {destination}")
            return True
            
        except Exception as e:
            print(f" Download error: {e}")
            return False
    
    def _extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """Extract zip archive."""
        print(f"\nExtracting {zip_path.name}...")
        
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            print(f" Extracted to: {extract_to}")
            self.extracted_archives.add(zip_path)
            return True
            
        except Exception as e:
            print(f" Extraction error: {e}")
            return False
    
    def check_cache_status(self) -> Dict[str, bool]:
        """
        Check which large files are cached locally.
        
        Returns:
            Dictionary mapping file paths to cache status (True = cached)
        """
        status = {}
        
        # Scan for .gdrive placeholder files
        for placeholder in self.cache_dir.rglob('*.gdrive'):
            # Get actual file path (remove .gdrive extension)
            actual_file = placeholder.with_suffix('')
            relative_path = actual_file.relative_to(self.cache_dir)
            
            status[str(relative_path)] = (
                actual_file.exists() and 
                actual_file.stat().st_size > 1024
            )
        
        return status
    
    def clear_cache(self, file_pattern: Optional[str] = None):
        """
        Remove cached files to free disk space.
        
        Args:
            file_pattern: Optional glob pattern to match specific files.
                         If None, removes all cached files.
        
        Example:
            >>> handler.clear_cache('*.parquet')  # Remove all parquet files
            >>> handler.clear_cache()  # Remove all cached files
        """
        if file_pattern is None:
            file_pattern = '*'
        
        removed_count = 0
        freed_space = 0
        
        for placeholder in self.cache_dir.rglob('*.gdrive'):
            actual_file = placeholder.with_suffix('')
            
            if actual_file.match(file_pattern) and actual_file.exists():
                file_size = actual_file.stat().st_size
                actual_file.unlink()
                removed_count += 1
                freed_space += file_size
        
        print(f"Removed {removed_count} files, freed {freed_space / (1024**3):.2f} GB")


# Global singleton instance
_handler = None


def get_data_file(filepath: str) -> Path:
    """
    Convenience function to get data file path with automatic download.
    
    This is the primary interface for accessing large datasets throughout
    the pipeline. Files are downloaded from Google Drive on first access
    and cached locally for subsequent use.
    
    Args:
        filepath: Relative path to file from project root
                 (e.g., 'data/arcticdem/arcticdem.parquet')
    
    Returns:
        Path: Absolute path to file (local cache or downloaded)
    
    Example:
        >>> from src.utils.data_access import get_data_file
        >>> import pandas as pd
        >>> 
        >>> # Automatically downloads if not cached
        >>> data_path = get_data_file('data/arcticdem/arcticdem.parquet')
        >>> df = pd.read_parquet(data_path)
    """
    global _handler
    
    if _handler is None:
        _handler = RemoteDataHandler()
    
    return _handler.get_file(filepath)


def check_data_availability() -> None:
    """
    Print report of data availability status.
    
    Shows which large files are cached locally vs. requiring download.
    """
    global _handler
    
    if _handler is None:
        _handler = RemoteDataHandler()
    
    status = _handler.check_cache_status()
    
    cached_files = [f for f, cached in status.items() if cached]
    missing_files = [f for f, cached in status.items() if not cached]
    
    total_files = len(status)
    cached_count = len(cached_files)
    
    print(f"\n{'='*70}")
    print(f"DATA AVAILABILITY STATUS")
    print(f"{'='*70}")
    print(f"Cached locally: {cached_count}/{total_files} files\n")
    
    if cached_files:
        print(" Available locally:")
        for f in sorted(cached_files)[:10]:  # Show first 10
            print(f"  - {f}")
        if len(cached_files) > 10:
            print(f"  ... and {len(cached_files) - 10} more")
    
    if missing_files:
        print(f"\n Require download ({len(missing_files)} files):")
        for f in sorted(missing_files)[:10]:
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        
        print("\nThese files will be downloaded automatically on first use.")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # CLI for checking data status
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        check_data_availability()
    else:
        print("Usage: python data_access.py check")
