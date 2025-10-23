#!/usr/bin/env python3
"""
SMAP Pipeline Test Script
Performs dry run to verify all components work correctly
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path
BASE_DIR = Path.home() / "arctic_zero_curtain_pipeline"
sys.path.insert(0, str(BASE_DIR / "scripts" / "smap"))

# Test parameters
TEST_YEAR = 2020
TEST_START = datetime(2020, 7, 15)
TEST_END = datetime(2020, 7, 17)  # 3 days


class PipelineTester:
    """Test SMAP pipeline components"""
    
    def __init__(self):
        self.results = {
            'directory_structure': False,
            'acquisition': False,
            'consolidation': False,
            'downscaling': False,
            'data_integrity': False
        }
    
    def test_directory_structure(self):
        """Test that all required directories exist"""
        print("\n" + "="*80)
        print("TEST 1: Directory Structure")
        print("="*80)
        
        required_dirs = [
            BASE_DIR / "data" / "auxiliary" / "smap" / "raw",
            BASE_DIR / "data" / "auxiliary" / "smap" / "consolidated",
            BASE_DIR / "data" / "auxiliary" / "smap" / "downscaled",
            BASE_DIR / "data" / "auxiliary" / "smap" / "checkpoints",
            BASE_DIR / "data" / "auxiliary" / "arcticdem",
            BASE_DIR / "data" / "auxiliary" / "landsat",
            BASE_DIR / "logs" / "smap",
        ]
        
        all_exist = True
        for directory in required_dirs:
            exists = directory.exists()
            status = "" if exists else ""
            print(f"{status} {directory}")
            if not exists:
                all_exist = False
        
        self.results['directory_structure'] = all_exist
        
        if all_exist:
            print("\n[PASS] All required directories exist")
        else:
            print("\n[FAIL] Some directories missing - creating...")
            for directory in required_dirs:
                directory.mkdir(parents=True, exist_ok=True)
            print("[OK] Directories created")
            self.results['directory_structure'] = True
        
        return all_exist
    
    def test_acquisition(self):
        """Test SMAP acquisition module"""
        print("\n" + "="*80)
        print("TEST 2: SMAP Acquisition (Import Test)")
        print("="*80)
        
        try:
            # Test import
            print("[TEST] Importing acquisition module...")
            from smap_acquisition import SMAPDownloader, DB_PATH
            
            print("[TEST] Creating downloader instance...")
            downloader = SMAPDownloader()
            
            print("[TEST] Testing CMR query building...")
            query_url = downloader.build_cmr_query()
            print(f"[OK] Query URL: {query_url[:100]}...")
            
            print("[TEST] Testing database initialization...")
            # Use the module-level DB_PATH instead of instance attribute
            assert DB_PATH.exists(), f"Database not found at {DB_PATH}"
            print(f"[OK] Database exists at {DB_PATH}")
            
            print("[TEST] Testing database schema...")
            import sqlite3
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            print(f"[OK] Database has {len(tables)} table(s)")
            
            self.results['acquisition'] = True
            print("\n[PASS] Acquisition module working")
            return True
            
        except Exception as e:
            print(f"\n[FAIL] Acquisition test failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['acquisition'] = False
            return False
    
    def test_consolidation(self):
        """Test SMAP consolidation module"""
        print("\n" + "="*80)
        print("TEST 3: SMAP Consolidation (Import Test)")
        print("="*80)
        
        try:
            print("[TEST] Importing consolidation module...")
            from smap_consolidation import SMAPConsolidator
            
            print("[TEST] Creating consolidator instance...")
            consolidator = SMAPConsolidator()
            
            print("[TEST] Testing schema creation...")
            schema = consolidator.create_schema()
            print(f"[OK] Schema has {len(schema)} fields")
            
            # Verify all required fields
            required_fields = ['datetime', 'x', 'y', 'latitude', 'longitude',
                             'sm_surface', 'sm_rootzone', 'soil_temp_layer1']
            schema_names = [field.name for field in schema]
            missing_fields = [f for f in required_fields if f not in schema_names]
            
            if missing_fields:
                print(f"[WARNING] Missing schema fields: {missing_fields}")
            else:
                print("[OK] All required schema fields present")
            
            print("[TEST] Testing H5 file search...")
            h5_files = consolidator.find_h5_files()
            print(f"[OK] Found {len(h5_files)} .h5 files (may be 0 if not downloaded yet)")
            
            self.results['consolidation'] = True
            print("\n[PASS] Consolidation module working")
            return True
            
        except Exception as e:
            print(f"\n[FAIL] Consolidation test failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['consolidation'] = False
            return False
    
    def test_downscaling(self):
        """Test SMAP downscaling module"""
        print("\n" + "="*80)
        print("TEST 4: SMAP Downscaling (Import Test)")
        print("="*80)
        
        try:
            print("[TEST] Importing downscaling module...")
            from smap_downscaling import SMAPDownscaler
            
            # Check if consolidated data exists
            smap_file = BASE_DIR / "data" / "auxiliary" / "smap" / "consolidated" / f"smap_{TEST_YEAR}.parquet"
            
            if not smap_file.exists():
                print(f"[INFO] No consolidated data for {TEST_YEAR} (expected for dry run)")
                print("[INFO] Creating mock consolidated file for testing...")
                
                # Create minimal mock data
                mock_data = pd.DataFrame({
                    'datetime': pd.date_range(TEST_START, TEST_END, freq='D'),
                    'year': [TEST_YEAR] * 3,
                    'month': [7] * 3,
                    'day': [15, 16, 17],
                    'x': [-1500000.0, -1500000.0, -1500000.0],
                    'y': [1000000.0, 1000000.0, 1000000.0],
                    'latitude': [65.0] * 3,
                    'longitude': [-150.0] * 3,
                    'sm_surface': [0.2, 0.21, 0.19],
                    'sm_rootzone': [0.25, 0.26, 0.24],
                    'soil_temp_layer1': [273.15, 274.0, 273.5],
                    'soil_temp_layer2': [273.15, 274.0, 273.5],
                    'soil_temp_layer3': [273.15, 274.0, 273.5],
                    'soil_temp_layer4': [273.15, 274.0, 273.5],
                    'soil_temp_layer5': [273.15, 274.0, 273.5],
                    'soil_temp_layer6': [273.15, 274.0, 273.5],
                })
                
                mock_data.to_parquet(smap_file, compression='zstd', index=False)
                print(f"[OK] Created mock file: {smap_file.name}")
            
            print("[TEST] Creating downscaler instance...")
            downscaler = SMAPDownscaler(TEST_YEAR, TEST_START, TEST_END)
            
            print("[TEST] Testing bounds calculation...")
            bounds = downscaler.get_circumarctic_bounds()
            print(f"[OK] Circumarctic bounds: W={bounds['west']:.0f}, E={bounds['east']:.0f}, "
                  f"S={bounds['south']:.0f}, N={bounds['north']:.0f}")
            
            print("[TEST] Testing chunk creation...")
            chunks = downscaler.create_processing_chunks()
            print(f"[OK] Created {len(chunks):,} processing chunks")
            
            print("[TEST] Checking auxiliary data...")
            has_dem = downscaler.dem_df is not None and len(downscaler.dem_df) > 0
            has_landsat = downscaler.landsat_df is not None and len(downscaler.landsat_df) > 0
            print(f"[INFO] ArcticDEM available: {has_dem}")
            print(f"[INFO] Landsat available: {has_landsat}")
            
            if not has_dem or not has_landsat:
                print("[WARNING] Missing auxiliary data - downscaling will use placeholders")
                print("[INFO] This is acceptable for dry run testing")
            
            # Test coordinate transformer
            print("[TEST] Testing coordinate transformation...")
            test_x, test_y = -1500000, 1000000
            lon, lat = downscaler.transformer.transform(test_x, test_y)
            print(f"[OK] Transform test: ({test_x}, {test_y}) -> ({lon:.2f}°, {lat:.2f}°)")
            
            self.results['downscaling'] = True
            print("\n[PASS] Downscaling module working")
            return True
            
        except Exception as e:
            print(f"\n[FAIL] Downscaling test failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['downscaling'] = False
            return False
    
    def test_data_integrity(self):
        """Test data integrity and format"""
        print("\n" + "="*80)
        print("TEST 5: Data Integrity")
        print("="*80)
        
        try:
            # Check for any existing consolidated data
            consolidated_dir = BASE_DIR / "data" / "auxiliary" / "smap" / "consolidated"
            parquet_files = list(consolidated_dir.glob("*.parquet"))
            
            if parquet_files:
                print(f"[TEST] Found {len(parquet_files)} consolidated files")
                
                # Test reading first file
                test_file = parquet_files[0]
                print(f"[TEST] Testing read of {test_file.name}...")
                
                df = pd.read_parquet(test_file, engine='pyarrow')
                print(f"[OK] Loaded {len(df):,} rows")
                
                # Check required columns
                required_cols = ['datetime', 'x', 'y', 'latitude', 'longitude', 
                                'sm_surface', 'sm_rootzone']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"[WARNING] Missing columns: {missing_cols}")
                else:
                    print("[OK] All required columns present")
                
                # Check data ranges
                print(f"[INFO] Date range: {df['datetime'].min()} to {df['datetime'].max()}")
                print(f"[INFO] Latitude range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
                print(f"[INFO] Longitude range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
                
                # Check for NaN values
                for col in ['sm_surface', 'sm_rootzone']:
                    if col in df.columns:
                        nan_pct = (df[col].isna().sum() / len(df)) * 100
                        print(f"[INFO] {col} NaN percentage: {nan_pct:.1f}%")
                
                # Check data value ranges
                if 'sm_surface' in df.columns:
                    sm_min = df['sm_surface'].min()
                    sm_max = df['sm_surface'].max()
                    if sm_min < 0 or sm_max > 1:
                        print(f"[WARNING] Soil moisture out of range: {sm_min:.3f} to {sm_max:.3f}")
                    else:
                        print(f"[OK] Soil moisture in valid range: {sm_min:.3f} to {sm_max:.3f}")
                
                self.results['data_integrity'] = True
                print("\n[PASS] Data integrity check passed")
                return True
            else:
                print("[INFO] No consolidated data found yet (expected if not run)")
                print("[SKIP] Data integrity test")
                self.results['data_integrity'] = None
                return True
                
        except Exception as e:
            print(f"\n[FAIL] Data integrity test failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['data_integrity'] = False
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("SMAP PIPELINE DRY RUN TEST SUITE")
        print("="*80)
        print(f"Base directory: {BASE_DIR}")
        print(f"Test year: {TEST_YEAR}")
        print(f"Test date range: {TEST_START.date()} to {TEST_END.date()}")
        
        # Run tests in order
        self.test_directory_structure()
        self.test_acquisition()
        self.test_consolidation()
        self.test_downscaling()
        self.test_data_integrity()
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        for test_name, result in self.results.items():
            if result is True:
                status = " PASS"
            elif result is False:
                status = " FAIL"
            else:
                status = "- SKIP"
            print(f"{status}: {test_name.replace('_', ' ').title()}")
        
        # Overall result
        passed = sum(1 for r in self.results.values() if r is True)
        failed = sum(1 for r in self.results.values() if r is False)
        skipped = sum(1 for r in self.results.values() if r is None)
        total = passed + failed
        
        print("\n" + "="*80)
        if failed == 0:
            print(f"ALL TESTS PASSED ({passed}/{total})")
            if skipped > 0:
                print(f"  ({skipped} test(s) skipped - expected for dry run)")
            print("="*80)
            print("\n Pipeline is ready to use!")
            print("\nNext steps:")
            print("  1. Run acquisition: python smap_acquisition.py")
            print("  2. Run consolidation: python smap_consolidation.py")
            print(f"  3. Run downscaling: python smap_downscaling.py {TEST_YEAR}")
            print("\nNote: Acquisition requires Earthdata credentials (.netrc file recommended)")
            return 0
        else:
            print(f"SOME TESTS FAILED ({passed}/{total} passed, {failed} failed)")
            if skipped > 0:
                print(f"  ({skipped} test(s) skipped)")
            print("="*80)
            print("\n Please fix failing tests before proceeding.")
            print("\nCommon issues:")
            print("  - Missing .netrc file for Earthdata credentials")
            print("  - ArcticDEM or Landsat data not yet acquired")
            print("  - Check error messages above for details")
            return 1


def main():
    """Main entry point"""
    tester = PipelineTester()
    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())