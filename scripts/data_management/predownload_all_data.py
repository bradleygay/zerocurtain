"""
Pre-download all large files from Google Drive.
Run this once to populate local cache before running pipeline offline.
"""

from setup_gdrive_placeholders import get_data_file, LARGE_FILES_MAPPING
import os

def predownload_all():
    """Download all files defined in LARGE_FILES_MAPPING."""
    print("Pre-downloading all data files from Google Drive...")
    print(f"Total files to download: {len(LARGE_FILES_MAPPING)}\n")
    
    success_count = 0
    failed_files = []
    
    for i, filepath in enumerate(LARGE_FILES_MAPPING.keys(), 1):
        print(f"\n[{i}/{len(LARGE_FILES_MAPPING)}] Processing: {filepath}")
        
        try:
            result_path = get_data_file(filepath)
            success_count += 1
            print(f"   Available at: {result_path}")
            
        except Exception as e:
            print(f"   Failed: {e}")
            failed_files.append(filepath)
    
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"Successful: {success_count}/{len(LARGE_FILES_MAPPING)}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")

if __name__ == "__main__":
    predownload_all()