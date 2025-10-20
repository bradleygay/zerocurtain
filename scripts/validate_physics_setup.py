#!/usr/bin/env python3
"""
Validation script to verify physics detection setup.
Checks configuration, data availability, and module imports.
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def validate_imports():
    """Validate that all required modules can be imported."""
    print("Validating module imports...")
    
    try:
        from physics_detection.zero_curtain_detector import PhysicsInformedZeroCurtainDetector
        print("   PhysicsInformedZeroCurtainDetector")
    except ImportError as e:
        print(f"   PhysicsInformedZeroCurtainDetector: {e}")
        return False
    
    try:
        from physics_detection.physics_config import DetectionConfig
        print("   DetectionConfig")
    except ImportError as e:
        print(f"   DetectionConfig: {e}")
        return False
    
    try:
        from orchestration.physics_detection_orchestrator import PhysicsDetectionOrchestrator
        print("   PhysicsDetectionOrchestrator")
    except ImportError as e:
        print(f"   PhysicsDetectionOrchestrator: {e}")
        return False
    
    return True


def validate_dependencies():
    """Validate that all required dependencies are installed."""
    print("\nValidating dependencies...")
    
    required = ['numpy', 'pandas', 'dask', 'scipy', 'xarray', 
                'rasterio', 'geopandas', 'pyproj']
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"   {package}")
        except ImportError:
            print(f"   {package}")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def validate_configuration():
    """Validate configuration and data paths."""
    print("\nValidating configuration...")
    
    from physics_detection.physics_config import DetectionConfig
    
    config = DetectionConfig()
    is_valid, missing = config.paths.validate_paths()
    
    if is_valid:
        print("   All data paths validated")
        print(f"     Permafrost raster: {config.paths.permafrost_prob_raster}")
        print(f"     Permafrost zones: {config.paths.permafrost_zones_shapefile}")
        print(f"     Snow data: {config.paths.snow_data_netcdf}")
        print(f"     In situ data: {config.paths.insitu_measurements_parquet}")
    else:
        print("    Some data paths missing:")
        for path in missing:
            print(f"     - {path}")
        print("\n  See data/auxiliary/DATA_SOURCES.md for download instructions")
    
    return is_valid


def validate_detector_initialization():
    """Validate that detector can be initialized."""
    print("\nValidating detector initialization...")
    
    try:
        from physics_detection.zero_curtain_detector import PhysicsInformedZeroCurtainDetector
        from physics_detection.physics_config import DetectionConfig
        
        config = DetectionConfig()
        detector = PhysicsInformedZeroCurtainDetector(config=config)
        
        print("   Detector initialized successfully")
        print(f"     Temperature threshold: ±{detector.TEMP_THRESHOLD}°C")
        print(f"     Minimum duration: {detector.MIN_DURATION_HOURS} hours")
        print(f"     CryoGrid enthalpy: {detector.use_cryogrid_enthalpy}")
        
        return True
        
    except Exception as e:
        print(f"   Detector initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks."""
    print("="*90)
    print("PHYSICS DETECTION SETUP VALIDATION")
    print("="*90)
    
    results = {
        'imports': validate_imports(),
        'dependencies': validate_dependencies(),
        'configuration': validate_configuration(),
        'detector': validate_detector_initialization()
    }
    
    print("\n" + "="*90)
    print("VALIDATION SUMMARY")
    print("="*90)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = " PASS" if passed else " FAIL"
        print(f"{check.upper():20s}: {status}")
    
    if all_passed:
        print("\n All validation checks passed!")
        print("   Ready to run physics-informed detection.")
        return 0
    else:
        print("\n  Some validation checks failed.")
        print("   Address issues above before running detection.")
        return 1


if __name__ == "__main__":
    sys.exit(main())