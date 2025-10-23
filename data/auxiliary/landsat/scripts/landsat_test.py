#!/usr/bin/env python3
"""
Landsat Processing Pipeline Test Script
Validates entire workflow with dry run
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_directory_structure(base_dir: Path):
    """Test directory creation"""
    logger.info("Testing directory structure...")
    
    expected_dirs = ['raw', 'processed', 'gaps_filled', 'swath_gaps_filled',
                    'final', 'checkpoints', 'temp_csv', 'analysis', 'logs']
    
    for dir_name in expected_dirs:
        dir_path = base_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        assert dir_path.exists(), f"Failed to create {dir_path}"
    
    logger.info(" Directory structure validated")

def test_sample_data_generation(output_dir: Path):
    """Generate and validate sample data"""
    logger.info("Testing sample data generation...")
    
    # Create sample data
    n_points = 1000
    sample_data = pd.DataFrame({
        'scene_id': [f'LC08_TEST_{i//10:04d}' for i in range(n_points)],
        'acquisition_date': pd.date_range('2020-01-01', periods=n_points, freq='D').strftime('%Y-%m-%d'),
        'longitude': np.random.uniform(-180, 180, n_points),
        'latitude': np.random.uniform(45, 90, n_points),
        'B2': np.random.randint(0, 10000, n_points),
        'B3': np.random.randint(0, 10000, n_points),
        'B4': np.random.randint(0, 10000, n_points),
        'B10': np.random.randint(20000, 35000, n_points),
        'cloud_cover': np.random.uniform(0, 10, n_points)
    })
    
    # Save
    output_file = output_dir / 'raw' / 'test_landsat_data.parquet'
    sample_data.to_parquet(output_file, index=False)
    
    # Validate
    loaded_data = pd.read_parquet(output_file)
    assert len(loaded_data) == n_points, "Data length mismatch"
    assert set(loaded_data.columns) == set(sample_data.columns), "Column mismatch"
    
    logger.info(f" Sample data generated: {len(sample_data):,} points")
    return output_file

def test_coverage_analysis(data_file: Path, analysis_dir: Path):
    """Test coverage analysis"""
    logger.info("Testing coverage analysis...")
    
    from landsat_gap_filler import analyze_coverage
    
    coverage = analyze_coverage(data_file, grid_size=10.0)
    
    assert 'coverage_percentage' in coverage
    assert 'gap_cells' in coverage
    assert 0 <= coverage['coverage_percentage'] <= 100
    
    logger.info(f" Coverage analysis: {coverage['coverage_percentage']:.2f}%")
    return coverage

def test_data_combination(output_dir: Path):
    """Test dataset combination"""
    logger.info("Testing data combination...")
    
    # Create three sample datasets
    datasets = []
    for i, name in enumerate(['original', 'gaps', 'swath_gaps']):
        n_points = 100 * (i + 1)
        df = pd.DataFrame({
            'scene_id': [f'LC08_{name.upper()}_{j:04d}' for j in range(n_points)],
            'acquisition_date': pd.date_range('2020-01-01', periods=n_points, freq='D').strftime('%Y-%m-%d'),
            'longitude': np.random.uniform(-180, 180, n_points),
            'latitude': np.random.uniform(45, 90, n_points),
            'B2': np.random.randint(0, 10000, n_points),
            'B3': np.random.randint(0, 10000, n_points),
            'B4': np.random.randint(0, 10000, n_points),
            'B10': np.random.randint(20000, 35000, n_points),
            'cloud_cover': np.random.uniform(0, 10, n_points)
        })
        
        file_path = output_dir / 'raw' / f'test_{name}.parquet'
        df.to_parquet(file_path, index=False)
        datasets.append(file_path)
    
    # Combine
    from landsat_combiner import combine_datasets
    
    output_file = output_dir / 'final' / 'test_combined.parquet'
    combined_df = combine_datasets(*datasets, output_file)
    
    assert len(combined_df) <= 600, "Duplicates not removed properly"
    logger.info(f" Data combination: {len(combined_df):,} points")

def test_imports():
    """Test all module imports"""
    logger.info("Testing module imports...")
    
    try:
        from landsat_downloader import Config, initialize_earth_engine
        from landsat_gap_filler import analyze_coverage, create_gap_regions
        from landsat_combiner import combine_datasets
        logger.info(" All imports successful")
        return True
    except ImportError as e:
        logger.error(f" Import failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("LANDSAT PROCESSING PIPELINE TEST SUITE")
    logger.info("="*60)
    
    # Test directory
    test_dir = Path('~/arctic_zero_curtain_pipeline/data/auxiliary/landsat/test').expanduser()
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run tests
        test_imports()
        test_directory_structure(test_dir)
        data_file = test_sample_data_generation(test_dir)
        test_coverage_analysis(data_file, test_dir / 'analysis')
        test_data_combination(test_dir)
        
        logger.info("="*60)
        logger.info(" ALL TESTS PASSED")
        logger.info("="*60)
        return 0
    
    except Exception as e:
        logger.error(f" TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())