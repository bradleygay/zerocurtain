"""
Google Drive setup for Circumarctic Zero Curtain Pipeline.
########################################
How Downloads Are Triggered -> Explicitly import
In any script that needs to access large data files, add this at the top:
    from setup_gdrive_placeholders import get_data_file
    import pandas as pd
    import torch
# For parquet files:
df = pd.read_parquet(get_data_file('data/arcticdem/arcticdem.parquet'))
# For PyTorch models:
model_path = get_data_file('part2_geocryoai/models/best_model_tf.pth')
model.load_state_dict(torch.load(model_path))
# For NetCDF files
import xarray as xr
snow_data = xr.open_dataset(get_data_file('data/auxiliary/snow/aa6ddc60e4ed01915fb9193bcc7f4146.nc'))
########################################
"""

import json
import os
import requests
from pathlib import Path

# ============================================================================
# COMPLETE FILE MAPPING - ALL LARGE FILES FROM YOUR SCREENSHOTS
# ============================================================================

LARGE_FILES_MAPPING = {

    # data/auxiliary/snow/ - Single NetCDF file
    "data/auxiliary/snow/aa6ddc60e4ed01915fb9193bcc7f4146.nc": {
        "gdrive_id": "19lPnWsTlF8Ha1C-LrpKmU_5_LzqMHMfV",
        "size": 2816835788  # 2.62 GB
    },

    # data/auxiliary/permafrost/UiO_PEX_PERPROB_5/ - Probability raster
    "data/auxiliary/permafrost/UiO_PEX_PERPROB_5/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tif": {
        "gdrive_id": "1tw6isYpV9zFbZBhz9Bae2XMMFd89CrHo",
        "size": 1780000000
    },
    "data/auxiliary/permafrost/UiO_PEX_PERPROB_5/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tif.aux.xml": {
        "gdrive_id": "1NhPaF1LIy3PvKdc9wYRLbntOrc5NAMB7",
        "size": 100000  # Small metadata file
    },
    "data/auxiliary/permafrost/UiO_PEX_PERPROB_5/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tif.ovr": {
        "gdrive_id": "1g3ZCu32CXcPgb5HgnTBVvZ1aUIAQO8Tm",
        "size": 200000000  # Pyramid overviews
    },
    "data/auxiliary/permafrost/UiO_PEX_PERPROB_5/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.lyr": {
        "gdrive_id": "1UW0_Zjfs1iKSjnTNDZ7Opf4uylSTa4Eb",
        "size": 1400  # Pyramid overviews
    },
    "data/auxiliary/permafrost/UiO_PEX_PERPROB_5/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tfw": {
        "gdrive_id": "1V_cEFSnxkgw01VL0raS9QCxIU8J341Kv",
        "size": 97
    },
    # data/auxiliary/permafrost/UiO_PEX_PERZONES_5/ - Shapefile components
    "data/auxiliary/permafrost/UiO_PEX_PERZONES_5/UiO_PEX_PERZONES_5.0_20181128_2000_2016_NH.shp": {
        "gdrive_id": "1QSsQ7LxeTLxPf7r3cqM_tahlRqTuPf6w",
        "size": 2200000  # Estimate
    },
    "data/auxiliary/permafrost/UiO_PEX_PERZONES_5/UiO_PEX_PERZONES_5.0_20181128_2000_2016_NH.dbf": {
        "gdrive_id": "1pREbwFZMtdeCXCIR4s4QbI9_lHGHexMr",
        "size": 2080000
    },
    "data/auxiliary/permafrost/UiO_PEX_PERZONES_5/UiO_PEX_PERZONESLegend_5.0_20181128.lyr": {
        "gdrive_id": "11-lqlz6gBwcPgXLZZRtWTz-z_Cw3C3YX",
        "size": 800
    },
    "data/auxiliary/permafrost/UiO_PEX_PERZONES_5/UiO_PEX_PERZONES_5.0_20181128_2000_2016_NH.shx": {
        "gdrive_id": "1T3qg9eKV9mPuBjV-WACh1dVVnboIkqhX",
        "size": 61900
    },
    "data/auxiliary/permafrost/UiO_PEX_PERZONES_5/UiO_PEX_PERZONES_5.0_20181128_2000_2016_NH.sbn": {
        "gdrive_id": "1hUin8fAbR_ED7CyEYM5YeYrxSrlH6YoI",
        "size": 76900
    },
    "data/auxiliary/permafrost/UiO_PEX_PERZONES_5/UiO_PEX_PERZONES_5.0_20181128_2000_2016_NH.prj": {
        "gdrive_id": "1IAtbaM8pdM4pDcxm-xSVC3syrbc16Us9",
        "size": 381
    },
    
    # data/auxiliary/natural_earth_data/ne_10m_land/ - Shapefile
    "data/auxiliary/natural_earth_data/ne_10m_land/ne_10m_land.dbf": {
        "gdrive_id": "1Rm0Z1YiUVYD_5p_KgqySh7rUYqBTWp57",
        "size": 350  # Tiny
    },
    "data/auxiliary/natural_earth_data/ne_10m_land/ne_10m_land.shx": {
        "gdrive_id": "1FuJLJMke2Vb3arH9s2R_JiZxW6P9pSqM",
        "size": 188 # Tiny
    },
    "data/auxiliary/natural_earth_data/ne_10m_land/ne_10m_land.prj": {
        "gdrive_id": "1CUADRkob2ePjTYv25ElIZZvvz9P__ltx",
        "size": 145  # Tiny
    },
    "data/auxiliary/natural_earth_data/ne_10m_land/ne_10m_land.cpg": {
        "gdrive_id": "1hYEj0BFdLAbqwm_sVTb6YVKdJ51PADxB",
        "size": 5 # Tiny
    },
    
    # data/auxiliary/natural_earth_data/ne_10m_lakes/ - Shapefile
    "data/auxiliary/natural_earth_data/ne_10m_lakes/ne_10m_lakes.shp": {
        "gdrive_id": "1sUOqOmEimlBIkVZ5yDnph_4YaMm2mLf8",
        "size": 26000
    },
    "data/auxiliary/natural_earth_data/ne_10m_lakes/ne_10m_lakes.dbf": {
        "gdrive_id": "1n4ghA4onMraIHmZsyq1Uv9NxNmJGHTwD",
        "size": 95000
    },
    "data/auxiliary/natural_earth_data/ne_10m_lakes/ne_10m_lakes.shx": {
        "gdrive_id": "1et2yVCUSfvcvR1C99kLOuzPAcPiL86Ag",
        "size": 1100
    },
    "data/auxiliary/natural_earth_data/ne_10m_lakes/ne_10m_lakes.prj": {
        "gdrive_id": "1BCRepo3rVkY1m6qn-kE8ougfSrdpRy8D",
        "size": 1000  # Tiny
    },
    
    # data/auxiliary/uavsar/displacement_30m
    "data/auxiliary/uavsar/displacement_30m.zip": {
        "gdrive_id": "1nYIFnUHFH7Y61MWHy_JxtsswFyC956NH",
        "size": 2571000,
        "extract_to": "data/auxiliary/uavsar/displacement_30m",
        "type": "zip_archive"
    },

    # outputs/part1_pinszc/consolidated_datasets/displacement_30m
    "outputs/part1_pinszc/consolidated_datasets/zero_curtain_enhanced_cryogrid_physics_dataset.parquet": {
        "gdrive_id": "1d8-nOxTdOV8ZdJSjjrG6TrVSQKnspfjh",
        "size": 2571000,
        "extract_to": "data/auxiliary/uavsar/displacement_30m",
        "type": "zip_archive"
    },

    # outputs/part1_pinszc/splits/
    "outputs/part1_pinszc/splits/physics_informed_events_test.parquet": {
        "gdrive_id": "1l4XbZehu03BMVD-kIVG8zT8VeG7cVrUA",
        "size": 464691200  # 443.4 MB
    },
    "outputs/part1_pinszc/splits/physics_informed_events_val.parquet": {
        "gdrive_id": "1pyDdV9r74A-4aKXgVAFPFH8YaMBfPgs0",
        "size": 465356800  # 443.7 MB
    },
    "outputs/part1_pinszc/splits/physics_informed_events_train.parquet": {
        "gdrive_id": "1B0KM5eJBw0A1qNQiLnldx-HM0sYdIndd",
        "size": 2222981120  # 2.07 GB
    },
    
    # outputs/part1_pinszc/consolidated_datasets/
    "outputs/part1_pinszc/consolidated_datasets/physics_informed_zero_curtain_events_COMPLETE.parquet": {
        "gdrive_id": "1btr-j14bjvysSQYffIkZ8s5Em-NvRzRT",
        "size": 1943928217  # 1.81 GB
    },
    
    # part3_pirszc/
    "outputs/part3_pirszc/remote_sensing_physics_informed_comprehensive.parquet": {
        "gdrive_id": "1nLg029zIxUrNdK7AZtfIe2hVmL5aApQX",
        "size": 7828000
    },
    
    # part2_geocryoai/models/
    "outputs/part2_geocryoai/models/checkpoint_epoch_10.pth": {
        "gdrive_id": "1Mx4aflx8WafmFAIPRDsWm39dAXWkkx_l",
        "size": 61655040  # 58.8 MB
    },
    "outputs/part2_geocryoai/models/best_model_tf.pth": {
        "gdrive_id": "1J2EBa2j2_cJlWaFYyTfneQSIkNddvHJm",
        "size": 61550592  # 58.7 MB
    },
    "outputs/part2_geocryoai/models/checkpoint_epoch_15.pth": {
        "gdrive_id": "1iXzBW0hGaioObPbi45jYNkwyGocvQlt1",
        "size": 61655040  # 58.8 MB
    },
    "outputs/part2_geocryoai/models/checkpoint_epoch_5.pth": {
        "gdrive_id": "1pj6AsqjxZlPN2bQZHiN2DD0cx2wNRF-s",
        "size": 61655040  # 58.8 MB
    },
    "outputs/part2_geocryoai/models/checkpoint_epoch_25.pth": {
        "gdrive_id": "1ABPHqBKO7LRkJTOz7JKKq3fQr9iqZWFq",
        "size": 61655040  # 58.8 MB
    },
    "outputs/part2_geocryoai/models/checkpoint_epoch_20.pth": {
        "gdrive_id": "1SxeNWfapgmQY7S12y5k13DytVXAniwfv",
        "size": 61655040  # 58.8 MB
    },
    
    # part2_geocryoai/
    "outputs/part2_geocryoai/teacher_forcing_in_situ_database_val.parquet": {
        "gdrive_id": "1qA7Old3JwrvXSmFGcCxN-pirQDged5nj",
        "size": 137756262  # 131.3 MB
    },
    "outputs/part2_geocryoai/teacher_forcing_in_situ_database_train.parquet": {
        "gdrive_id": "14YsaneK_YqZ6FjlCySEPot6n4N0qDsWZ",
        "size": 481951129  # 459.6 MB
    },
    "outputs/part2_geocryoai/teacher_forcing_in_situ_database_test.parquet": {
        "gdrive_id": "1M4vFu2aAlHkjhBgdqG8VbhKo6Jx3mMpP",
        "size": 68812390  # 65.6 MB
    },
    
    # data/auxiliary/data_products/
    "data/auxiliary/uavsar/data_products/remote_sensing.parquet": {
        "gdrive_id": "1D1ZNtCRfsO6ELoMsVnaZLhqhzPnbEakS",
        "size": 38264987238  # 35.64 GB
    },
    "data/auxiliary/uavsar/data_products/nisar.parquet": {
        "gdrive_id": "1V415cLEJan1D8BrNtp6JuQAa_wGzEWy0",
        "size": 36491116339  # 33.99 GB
    },
    
    # data/landsat/
    "data/auxiliary/landsat/landsat.parquet": {
        "gdrive_id": "1IjxeYrraelcMqngu-B2IR4PfyED35raw",
        "size": 344883404  # 328.8 MB
    },
    
    # data/smap/
    "data/auxiliary/smap/smap.parquet": {
        "gdrive_id": "11Iqp_LxAWu7VANC5jFo9SYtQeKl8vv7d",
        "size": 5390983987  # 5.02 GB
    },
    
    # data/arcticdem/
    "data/auxiliary/arcticdem/arcticdem.parquet": {
        "gdrive_id": "1eO86EgHcywk3niXI3OTXljwvNKCQuU63",
        "size": 25052364390  # 23.32 GB
    }
}

# ============================================================================
# ENHANCED DOWNLOAD HANDLER WITH ZIP EXTRACTION
# ============================================================================

import zipfile

class GDriveDataHandler:
    """Handles downloading and extracting files from Google Drive."""
    
    def __init__(self):
        self.session = requests.Session()
        self.extracted_archives = set()
    
    def download_from_gdrive(self, file_id: str, destination: str) -> bool:
        """Download file from Google Drive."""
        print(f"\nDownloading {os.path.basename(destination)} from Google Drive...")
        
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
            
            if response.status_code == 200:
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024  # 1MB
                downloaded = 0
                
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                size_gb = downloaded / (1024**3)
                                print(f"\rProgress: {progress:.1f}% ({size_gb:.2f} GB)", end='')
                
                print(f"\n Downloaded: {destination}")
                return True
            else:
                print(f" Failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f" Error: {str(e)}")
            return False
    
    def extract_zip(self, zip_path: str, extract_to: str) -> bool:
        """Extract zip archive."""
        print(f"\nExtracting {os.path.basename(zip_path)} to {extract_to}...")
        
        try:
            os.makedirs(extract_to, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            print(f" Extracted to: {extract_to}")
            self.extracted_archives.add(zip_path)
            return True
            
        except Exception as e:
            print(f" Extraction error: {str(e)}")
            return False
    
    def get_file(self, filepath: str) -> str:
        """Get file, downloading and extracting if necessary."""
        
        # Check if file already exists
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
            return filepath
        
        # Check if file comes from a zip archive
        for archive_path, metadata in LARGE_FILES_MAPPING.items():
            if metadata.get('type') == 'zip_archive':
                extract_to = metadata.get('extract_to', '')
                if filepath.startswith(extract_to):
                    # Need to download and extract archive first
                    if archive_path not in self.extracted_archives:
                        placeholder_path = f"{archive_path}.gdrive"
                        
                        if not os.path.exists(placeholder_path):
                            raise FileNotFoundError(f"Placeholder not found: {placeholder_path}")
                        
                        with open(placeholder_path, 'r') as f:
                            placeholder_data = json.load(f)
                        
                        file_id = placeholder_data['gdrive_file_id']
                        
                        if file_id == "REPLACE_WITH_ACTUAL_FILE_ID":
                            raise ValueError(f"File ID not configured for {archive_path}")
                        
                        # Download zip
                        if not os.path.exists(archive_path):
                            success = self.download_from_gdrive(file_id, archive_path)
                            if not success:
                                raise RuntimeError(f"Failed to download {archive_path}")
                        
                        # Extract zip
                        success = self.extract_zip(archive_path, extract_to)
                        if not success:
                            raise RuntimeError(f"Failed to extract {archive_path}")
                    
                    # Check if file now exists
                    if os.path.exists(filepath):
                        return filepath
                    else:
                        raise FileNotFoundError(f"File not found after extraction: {filepath}")
        
        # Standard placeholder approach for individual files
        placeholder_path = f"{filepath}.gdrive"
        if not os.path.exists(placeholder_path):
            raise FileNotFoundError(f"Neither {filepath} nor {placeholder_path} found")
        
        with open(placeholder_path, 'r') as f:
            metadata = json.load(f)
        
        file_id = metadata['gdrive_file_id']
        
        if file_id == "REPLACE_WITH_ACTUAL_FILE_ID":
            raise ValueError(f"File ID not configured for {filepath}")
        
        success = self.download_from_gdrive(file_id, filepath)
        
        if not success:
            raise RuntimeError(f"Failed to download {filepath}")
        
        return filepath

# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

def create_all_placeholders():
    """Create placeholder files."""
    for filepath, metadata in LARGE_FILES_MAPPING.items():
        placeholder_path = f"{filepath}.gdrive"
        
        placeholder_data = {
            "original_filename": os.path.basename(filepath),
            "original_path": filepath,
            "gdrive_file_id": metadata["gdrive_id"],
            "gdrive_folder": "1hI2D7wMeLw_DxcD_huOoBWCxmsCnQF0g",
            "file_size_bytes": metadata["size"]
        }
        
        # Add type info if it's a zip
        if metadata.get('type') == 'zip_archive':
            placeholder_data['type'] = 'zip_archive'
            placeholder_data['extract_to'] = metadata['extract_to']
        
        os.makedirs(os.path.dirname(placeholder_path), exist_ok=True)
        
        with open(placeholder_path, 'w') as f:
            json.dump(placeholder_data, f, indent=2)
        
        print(f" Created: {placeholder_path}")

def create_gitignore():
    """Create .gitignore."""
    content = """# Large data files
*.parquet
*.pth
*.nc
*.tif
*.h5
*.zip
*.grd

# Keep placeholders
!*.gdrive

# Geospatial files
*.shp
*.shx
*.dbf
*.prj
*.sbn
*.sbx

# Python
__pycache__/
*.py[cod]

# Jupyter
.ipynb_checkpoints/

# Environment
.env
venv/
"""
    with open('.gitignore', 'w') as f:
        f.write(content)
    print(" Created .gitignore")

def setup_repository():
    """Run setup."""
    print("Setting up Google Drive integration...\n")
    create_all_placeholders()
    create_gitignore()
    print("\n Setup complete!")
    print("\nIMPORTANT: For displacement_30m directory:")
    print("  1. Create zip: cd data/auxiliary && zip -r displacement_30m.zip displacement_30m/")
    print("  2. Upload displacement_30m.zip to Google Drive")
    print("  3. Get file ID and update LARGE_FILES_MAPPING")

# Global handler
_handler = None

def get_data_file(filepath: str) -> str:
    """
    Use in scripts:
    
    from setup_gdrive_placeholders import get_data_file
    import pandas as pd
    
    df = pd.read_parquet(get_data_file('data/arcticdem/arcticdem.parquet'))
    """
    global _handler
    if _handler is None:
        _handler = GDriveDataHandler()
    return _handler.get_file(filepath)

if __name__ == "__main__":
    setup_repository()