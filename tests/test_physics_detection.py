"""
Tests for physics-informed zero-curtain detection module.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics_detection.physics_config import DetectionConfig, DataPaths, PhysicsParameters
from physics_detection.zero_curtain_detector import PhysicsInformedZeroCurtainDetector


class TestPhysicsConfiguration:
    """Test configuration management."""
    
    def test_default_config_creation(self):
        """Test that default configuration can be created."""
        config = DetectionConfig()
        assert config is not None
        assert config.paths.base_dir == Path("/Users/bagay/Downloads")
    
    def test_custom_config_creation(self):
        """Test custom configuration creation."""
        config = DetectionConfig(
            paths=DataPaths(base_dir=Path("/custom/path")),
            physics=PhysicsParameters(temp_threshold=2.0)
        )
        assert config.paths.base_dir == Path("/custom/path")
        assert config.physics.temp_threshold == 2.0
    
    def test_path_validation(self):
        """Test path validation logic."""
        config = DetectionConfig()
        is_valid, missing = config.paths.validate_paths()
        # Should return bool and list
        assert isinstance(is_valid, bool)
        assert isinstance(missing, list)


class TestPhysicsDetector:
    """Test physics detector initialization and methods."""
    
    def test_detector_initialization(self):
        """Test detector can be initialized with config."""
        config = DetectionConfig()
        detector = PhysicsInformedZeroCurtainDetector(config=config)
        assert detector is not None
        assert detector.config == config
    
    def test_physical_constants(self):
        """Test that physical constants are properly set."""
        detector = PhysicsInformedZeroCurtainDetector()
        
        # Check LPJ-EOSIM constants
        assert detector.LHEAT == 3.34E8
        assert detector.CWATER == 4180000
        
        # Check CryoGrid constants
        assert detector.LVOL_SL == 3.34E8
        assert detector.STEFAN_BOLTZMANN == 5.67e-8
    
    def test_threshold_configuration(self):
        """Test that thresholds can be configured."""
        config = DetectionConfig(
            physics=PhysicsParameters(
                temp_threshold=2.5,
                min_duration_hours=24
            )
        )
        detector = PhysicsInformedZeroCurtainDetector(config=config)
        
        assert detector.TEMP_THRESHOLD == 2.5
        assert detector.MIN_DURATION_HOURS == 24


class TestDetectionMethods:
    """Test detection method functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        return PhysicsInformedZeroCurtainDetector()
    
    def test_permafrost_properties_extraction(self, detector):
        """Test permafrost property extraction."""
        # Test with Arctic coordinates
        props = detector.get_site_permafrost_properties(70.0, -150.0)
        
        assert 'permafrost_prob' in props
        assert 'permafrost_zone' in props
        assert 'is_permafrost_suitable' in props
    
    def test_find_continuous_periods(self, detector):
        """Test continuous period finding."""
        mask = np.array([True, True, True, False, False, True, True, True, True])
        periods = detector._find_continuous_periods(mask, min_length=3)
        
        # Should find two periods: [0-2] and [5-8]
        assert len(periods) == 2
        assert periods[0] == (0, 2)
        assert periods[1] == (5, 8)
    
    def test_heat_capacity_calculation(self, detector):
        """Test heat capacity calculation."""
        soil_props = {
            'organic_fraction': 0.1,
            'mineral_fraction': 0.8,
            'water_fraction': 0.1
        }
        
        heat_capacity = detector._calculate_heat_capacity(soil_props)
        assert heat_capacity > 0
        assert isinstance(heat_capacity, (int, float))


def test_config_to_dict():
    """Test configuration serialization."""
    config = DetectionConfig()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert 'paths' in config_dict
    assert 'physics' in config_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])