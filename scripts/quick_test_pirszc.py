#!/usr/bin/env python3
"""
Quick Test for PIRSZC Pipeline - Memory Safe Version
====================================================

Lightweight test that checks critical components without loading large datasets.

Usage:
    python quick_test_pirszc.py
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test critical imports."""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    required = {
        'numpy': None,
        'pandas': None,
        'dask': None,
        'numba': None,
        'rasterio': None,
        'geopandas': None
    }
    
    all_ok = True
    for package in required:
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f" {package}: {version}")
        except ImportError as e:
            print(f" {package}: NOT INSTALLED")
            all_ok = False
    
    return all_ok

def test_numba():
    """Test NUMBA compilation."""
    print("\n" + "="*60)
    print("TESTING NUMBA JIT")
    print("="*60)
    
    try:
        from numba import jit
        
        @jit(nopython=True)
        def test_func(x):
            return x * 2
        
        result = test_func(5)
        if result == 10:
            print(" NUMBA JIT: Working")
            return True
        else:
            print(" NUMBA JIT: Failed")
            return False
    except Exception as e:
        print(f" NUMBA JIT: {e}")
        return False

def test_directories():
    """Test directory structure."""
    print("\n" + "="*60)
    print("TESTING DIRECTORIES")
    print("="*60)
    
    base = Path.home() / 'arctic_zero_curtain_pipeline'
    
    dirs = {
        'base': base,
        'data': base / 'data' / 'auxiliary' / 'uavsar' / 'data_products',
        'results': base / 'results' / 'pirszc',
        'logs': base / 'logs'
    }
    
    all_ok = True
    for name, path in dirs.items():
        if path.exists():
            print(f" {name}: {path}")
        else:
            print(f" {name}: NOT FOUND (will create on run)")
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"  Created: {path}")
            except:
                all_ok = False
    
    return all_ok

def test_input_file():
    """Test input file existence and basic metadata."""
    print("\n" + "="*60)
    print("TESTING INPUT FILE")
    print("="*60)
    
    rs_file = Path.home() / 'arctic_zero_curtain_pipeline' / 'data' / 'auxiliary' / 'uavsar' / 'data_products' / 'remote_sensing.parquet'
    
    if not rs_file.exists():
        print(f" Remote sensing file not found: {rs_file}")
        return False
    
    print(f" File exists: {rs_file}")
    
    # Get file size without loading
    size_mb = rs_file.stat().st_size / (1024**2)
    print(f"  Size: {size_mb:.1f} MB")
    
    # Try to get row count using parquet metadata (no data loading)
    try:
        import pyarrow.parquet as pq
        
        parquet_file = pq.ParquetFile(rs_file)
        n_rows = parquet_file.metadata.num_rows
        n_cols = parquet_file.metadata.num_columns
        
        print(f"  Rows: {n_rows:,}")
        print(f"  Columns: {n_cols}")
        
        # Get schema
        schema = parquet_file.schema_arrow
        print(f"  Schema preview: {schema.names[:5]}...")
        
        return True
        
    except Exception as e:
        print(f" Could not read metadata: {e}")
        print(f"  File exists but metadata check failed")
        return True  # File exists, that's what matters

def test_detector_import():
    """Test detector import."""
    print("\n" + "="*60)
    print("TESTING DETECTOR IMPORT")
    print("="*60)
    
    try:
        # Check if file exists
        script_dir = Path(__file__).parent
        detector_file = script_dir / 'remote_sensing_zero_curtain_v2.py'
        
        if not detector_file.exists():
            print(f" Detector script not found: {detector_file}")
            return False
        
        print(f" Detector script found: {detector_file}")
        
        # Try import without initialization (faster)
        from remote_sensing_zero_curtain_v2 import PhysicsInformedZeroCurtainDetector
        
        print(" Detector class imported successfully")
        print("  Note: Not initializing to save time/memory")
        print("  Full initialization will happen during actual run")
        
        return True
        
    except ImportError as e:
        print(f" Import failed: {e}")
        return False
    except Exception as e:
        print(f" Error: {e}")
        return False

def main():
    """Run quick tests."""
    print("="*60)
    print("PIRSZC QUICK TEST - Memory Safe")
    print("="*60)
    print("This test checks critical components WITHOUT loading data")
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['numba'] = test_numba()
    results['directories'] = test_directories()
    results['input_file'] = test_input_file()
    results['detector'] = test_detector_import()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = " PASSED" if passed else " FAILED"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print(" ALL QUICK TESTS PASSED")
        print("\nReady to run full test:")
        print("  python test_pirszc.py")
        print("\nOr proceed directly to processing:")
        print("  python pirszc_integration.py")
    else:
        print(" SOME TESTS FAILED")
        print("\nReview errors above and fix before proceeding")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
