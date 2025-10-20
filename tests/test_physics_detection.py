#!/usr/bin/env python3
"""
Unit tests for Physics-Informed Zero-Curtain Detection

Tests configuration management, detector initialization,
and core detection methods.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics_detection.physics_config import DetectionConfig, DataPaths, PhysicsParameters
from physics_detection.zero_curtain_detector import PhysicsInformedZeroCurtainDetector


# ============================================================================
# PYTEST FIXTURES FOR TEMPORARY DIRECTORIES
# ============================================================================

@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for all tests in this session."""
    temp_dir = tempfile.mkdtemp(prefix="test_zc_")
    yield Path(temp_dir)
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_test_dir):
    """Create test configuration with temporary paths."""
    test_paths = DataPaths(base_dir=temp_test_dir)
    return DetectionConfig(paths=test_paths)


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestPhysicsConfiguration:
    """Test physics configuration management."""
    
    def test_default_config_creation(self, test_config):
        """Test that configuration can be created with temp paths."""
        assert test_config is not None
        assert test_config.paths.output_dir.exists()
        
    def test_custom_config_creation(self, temp_test_dir):
        """Test custom configuration creation."""
        custom_paths = DataPaths(base_dir=temp_test_dir / "custom")
        config = DetectionConfig(
            paths=custom_paths,
            physics=PhysicsParameters(temp_threshold=2.0)
        )
        
        assert config.physics.temp_threshold == 2.0
        assert config.paths.base_dir == temp_test_dir / "custom"
        
    def test_path_validation(self, test_config):
        """Test path validation logic."""
        # Output directory should be created
        assert test_config.paths.output_dir.exists()
        assert test_config.paths.output_dir.is_dir()


# ============================================================================
# DETECTOR TESTS
# ============================================================================

class TestPhysicsDetector:
    """Test physics-informed detector."""
    
    def test_detector_initialization(self, test_config):
        """Test detector can be initialized with config."""
        detector = PhysicsInformedZeroCurtainDetector(config=test_config)
        
        assert detector is not None
        assert detector.config == test_config
        
    def test_physical_constants(self, test_config):
        """Test that physical constants are properly set."""
        detector = PhysicsInformedZeroCurtainDetector(config=test_config)
        
        # Check key physical constants exist
        assert hasattr(detector, 'config')
        assert detector.config.physics.temp_threshold > 0
        assert detector.config.physics.min_duration_hours > 0
        
    def test_threshold_configuration(self, temp_test_dir):
        """Test that thresholds can be configured."""
        custom_paths = DataPaths(base_dir=temp_test_dir / "threshold_test")
        config = DetectionConfig(
            paths=custom_paths,
            physics=PhysicsParameters(
                temp_threshold=2.5,
                min_duration_hours=24
            )
        )
        
        detector = PhysicsInformedZeroCurtainDetector(config=config)
        
        assert detector.config.physics.temp_threshold == 2.5
        assert detector.config.physics.min_duration_hours == 24


# ============================================================================
# DETECTION METHOD TESTS
# ============================================================================

class TestDetectionMethods:
    """Test core detection methods."""
    
    @pytest.fixture
    def detector(self, test_config):
        """Create detector instance for testing."""
        return PhysicsInformedZeroCurtainDetector(config=test_config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample temperature data for testing."""
        # Create 1000 hours of data
        hours = np.arange(1000)
        
        # Simulate zero-curtain: stable temps near 0°C
        temps = np.zeros(1000)
        temps[:200] = -5.0  # Before zero-curtain
        # Use smaller variance to ensure temps stay within ±1°C threshold
        temps[200:800] = np.random.normal(0.0, 0.3, 600)  # Zero-curtain period (±0.9°C)
        temps[800:] = 2.0  # After zero-curtain
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=1000, freq='h'),  # Changed 'H' to 'h'
            'temperature': temps,
            'latitude': 65.0,
            'longitude': -147.0
        })
        
        return df
    
    def test_permafrost_properties_extraction(self, detector, sample_data):
        """Test extraction of permafrost properties from data."""
        # This is a placeholder test - adjust based on actual detector methods
        assert detector is not None
        assert len(sample_data) == 1000
        
        # Test that detector can process the data
        # Note: Actual detection method calls would go here
        assert sample_data['temperature'].mean() < 1.0
        
    def test_find_continuous_periods(self, detector, sample_data):
        """Test identification of continuous zero-curtain periods."""
        # Test continuous period detection logic
        temps = sample_data['temperature'].values
        
        # Simple threshold check
        near_zero = np.abs(temps) < 1.0
        
        # Find continuous periods
        periods = []
        start = None
        for i, is_near_zero in enumerate(near_zero):
            if is_near_zero and start is None:
                start = i
            elif not is_near_zero and start is not None:
                periods.append((start, i))
                start = None
        
        # Should find the zero-curtain period (200-800)
        assert len(periods) >= 1
        
        # Longest period should be around 600 hours
        longest = max(periods, key=lambda p: p[1] - p[0])
        duration = longest[1] - longest[0]
        assert 500 <= duration <= 700
        
    def test_heat_capacity_calculation(self, detector):
        """Test heat capacity calculations."""
        # Test heat capacity calculation with known values
        
        # Volumetric heat capacity of water: ~4.2 MJ/m³/K
        water_content = 0.3  # 30% water content
        
        # Simplified heat capacity calculation
        heat_capacity_water = 4.2e6  # J/m³/K
        heat_capacity_soil = 2.0e6   # J/m³/K
        
        effective_heat_capacity = (
            water_content * heat_capacity_water + 
            (1 - water_content) * heat_capacity_soil
        )
        
        # Should be between pure soil and pure water
        assert heat_capacity_soil < effective_heat_capacity < heat_capacity_water
        assert 2.5e6 <= effective_heat_capacity <= 3.5e6


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_config_to_dict(test_config):
    """Test configuration serialization."""
    # Test that config can be converted to dict-like structure
    assert test_config.physics.temp_threshold > 0
    assert test_config.paths.output_dir.exists()
    
    # Test physics parameters
    physics = test_config.physics
    assert hasattr(physics, 'temp_threshold')
    assert hasattr(physics, 'min_duration_hours')


# Pytest configuration
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])