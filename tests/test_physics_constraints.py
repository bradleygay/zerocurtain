#!/usr/bin/env python3
"""
Unit tests for Physics-Informed Constraints

Tests CryoGrid thermodynamics, Stefan problem, pattern-aware constraints,
and energy conservation in physics-informed loss.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zero_curtain_ml_model import (
    PhysicsInformedLoss,
    PhysicsInformedTemporalConstraints
)


class TestStefanProblem:
    """Test Stefan problem energy conservation constraints."""
    
    def test_stefan_problem_scaling(self):
        """Test expected extent based on Stefan problem."""
        # Stefan: extent ∝ sqrt(thermal_diffusivity * time) * intensity
        
        thermal_diffusivity = 1e-6  # m²/s
        duration_hours = 100.0
        intensity = 0.5
        
        # Convert to seconds
        duration_seconds = duration_hours * 3600
        
        # Calculate expected extent
        expected_extent = np.sqrt(thermal_diffusivity * duration_seconds) * intensity
        
        # Verify reasonable range (should be ~0.3-0.4m for these values)
        assert 0.2 < expected_extent < 0.5, f"Expected extent {expected_extent} out of realistic range"
    
    def test_stefan_thermal_diffusivity_relationship(self):
        """Test that extent increases with thermal diffusivity."""
        duration = 100.0
        intensity = 0.5
        
        # Test with different diffusivities
        diff1 = 1e-7
        diff2 = 1e-6
        
        extent1 = np.sqrt(diff1 * duration * 3600) * intensity
        extent2 = np.sqrt(diff2 * duration * 3600) * intensity
        
        # Higher diffusivity should give larger extent
        assert extent2 > extent1, "Extent should increase with thermal diffusivity"
        assert abs(extent2 / extent1 - np.sqrt(10)) < 0.01, "Ratio should be sqrt(10)"


class TestPhysicsInformedLoss:
    """Test physics-informed loss function."""
    
    def setup_method(self):
        """Setup loss function for testing."""
        self.criterion = PhysicsInformedLoss(
            alpha_mse=1.0,
            alpha_physics=0.1,
            alpha_temporal=0.05,
            alpha_pattern=0.1
        )
        self.device = torch.device('cpu')
    
    def test_mse_loss_component(self):
        """Test MSE loss calculation."""
        predictions = torch.tensor([[0.5, 100.0, 1.5]], dtype=torch.float32)
        targets = torch.tensor([[0.6, 110.0, 1.6]], dtype=torch.float32)
        
        loss, components = self.criterion(predictions, targets)
        
        # Verify MSE component exists
        assert 'mse_loss' in components
        assert components['mse_loss'] > 0, "MSE loss should be positive"
        
        # Verify MSE calculation
        expected_mse = ((0.5-0.6)**2 + (100-110)**2 + (1.5-1.6)**2) / 3
        assert abs(components['mse_loss'] - expected_mse) < 0.01
    
    def test_pattern_aware_duration_constraints(self):
        """Test pattern-specific duration constraints."""
        # Test rapid event (pattern 0) with excessive duration
        predictions = torch.tensor([[0.5, 100.0, 1.5]], dtype=torch.float32)  # 100h duration
        targets = torch.tensor([[0.5, 100.0, 1.5]], dtype=torch.float32)
        pattern_predictions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)  # Rapid pattern
        
        loss, components = self.criterion(predictions, targets, pattern_predictions=pattern_predictions)
        
        # Physics loss should penalize long duration for rapid events
        assert 'physics_loss' in components
        assert components['physics_loss'] > 0, "Should penalize duration > 72h for rapid events"
    
    def test_extent_bounds_constraint(self):
        """Test spatial extent physics bounds."""
        # Test extent outside realistic bounds
        predictions = torch.tensor([[0.5, 100.0, 6.0]], dtype=torch.float32)  # 6m extent (too large)
        targets = torch.tensor([[0.5, 100.0, 1.5]], dtype=torch.float32)
        
        loss, components = self.criterion(predictions, targets)
        
        # Should penalize extent > 5.0m
        assert components['physics_loss'] > 0, "Should penalize extent > 5m"
    
    def test_energy_conservation_with_physics_features(self):
        """Test energy conservation constraint."""
        predictions = torch.tensor([[0.5, 100.0, 1.5]], dtype=torch.float32)
        targets = torch.tensor([[0.5, 100.0, 1.5]], dtype=torch.float32)
        
        # Physics features: [thermal_diffusivity, permafrost_prob, ...]
        physics_features = torch.tensor([[1e-6, 0.8, 0.0]], dtype=torch.float32)
        
        loss, components = self.criterion(predictions, targets, physics_features=physics_features)
        
        # Should apply Stefan problem constraint
        assert 'physics_loss' in components
        # Loss should be non-zero due to Stefan problem mismatch
        assert components['total_loss'] > components['mse_loss']
    
    def test_loss_component_weights(self):
        """Test that loss weights are applied correctly."""
        predictions = torch.tensor([[0.5, 100.0, 1.5]], dtype=torch.float32)
        targets = torch.tensor([[0.6, 110.0, 1.6]], dtype=torch.float32)
        
        loss, components = self.criterion(predictions, targets)
        
        # Verify all components exist
        assert 'mse_loss' in components
        assert 'physics_loss' in components
        assert 'temporal_loss' in components
        assert 'total_loss' in components
        
        # Verify all components are reasonable (non-negative, finite)
        assert components['mse_loss'] >= 0, "MSE loss should be non-negative"
        assert components['physics_loss'] >= 0, "Physics loss should be non-negative"
        assert components['temporal_loss'] >= 0, "Temporal loss should be non-negative"
        assert components['total_loss'] >= 0, "Total loss should be non-negative"
        
        assert torch.isfinite(torch.tensor(components['total_loss'])), "Total loss should be finite"
        
        # Total should be at least as large as MSE component (since all weights are positive)
        assert components['total_loss'] >= components['mse_loss'] * self.criterion.alpha_mse, \
            "Total loss should include MSE component"
    
    def test_nan_protection(self):
        """Test NaN/Inf protection in loss calculation."""
        # Test with extreme values that might cause NaN
        predictions = torch.tensor([[float('inf'), 100.0, 1.5]], dtype=torch.float32)
        targets = torch.tensor([[0.5, 100.0, 1.5]], dtype=torch.float32)
        
        loss, components = self.criterion(predictions, targets)
        
        # Loss should not be NaN
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"


class TestPhysicsInformedTemporalConstraints:
    """Test temporal pattern physics constraints."""
    
    def setup_method(self):
        """Setup temporal constraints module."""
        self.constraints = PhysicsInformedTemporalConstraints()
    
    def test_rapid_duration_constraint(self):
        """Test rapid event duration constraints."""
        batch_size = 2
        seq_len = 5
        num_patterns = 4
        
        # Create pattern probabilities (pattern 0 = rapid)
        pattern_probs = torch.zeros(batch_size, seq_len, num_patterns)
        pattern_probs[:, :, 0] = 1.0  # All rapid events
        
        # Event features: [duration, intensity, extent]
        event_features = torch.tensor([
            [[80.0, 0.5, 1.0], [60.0, 0.4, 0.8], [50.0, 0.6, 1.2], [70.0, 0.5, 1.0], [65.0, 0.4, 0.9]],
            [[90.0, 0.5, 1.0], [75.0, 0.4, 0.8], [55.0, 0.6, 1.2], [85.0, 0.5, 1.0], [70.0, 0.4, 0.9]]
        ])
        
        timestamps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).float()
        continuity_scores = torch.ones(batch_size, seq_len, 1) * 0.5
        
        # Apply constraints
        constrained_probs = self.constraints(pattern_probs, event_features, timestamps, continuity_scores)
        
        # Verify output shape
        assert constrained_probs.shape == pattern_probs.shape
        
        # Events > 72h should have reduced rapid probability
        for b in range(batch_size):
            for t in range(seq_len):
                if event_features[b, t, 0] > 72.0:
                    assert constrained_probs[b, t, 0] < pattern_probs[b, t, 0], \
                        f"Rapid probability should decrease for duration {event_features[b, t, 0]}"
    
    def test_extended_duration_constraint(self):
        """Test extended event duration constraints."""
        batch_size = 2
        seq_len = 3
        num_patterns = 4
        
        # Pattern 2 = extended persistent
        pattern_probs = torch.zeros(batch_size, seq_len, num_patterns)
        pattern_probs[:, :, 2] = 1.0
        
        # Short durations (< 168h minimum for extended)
        event_features = torch.tensor([
            [[100.0, 0.5, 1.5], [120.0, 0.6, 1.8], [90.0, 0.4, 1.2]],
            [[110.0, 0.5, 1.5], [130.0, 0.6, 1.8], [95.0, 0.4, 1.2]]
        ])
        
        timestamps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).float()
        continuity_scores = torch.ones(batch_size, seq_len, 1) * 0.8
        
        constrained_probs = self.constraints(pattern_probs, event_features, timestamps, continuity_scores)
        
        # All durations < 168h should have reduced extended probability
        for b in range(batch_size):
            for t in range(seq_len):
                assert constrained_probs[b, t, 2] < pattern_probs[b, t, 2], \
                    "Extended probability should decrease for short durations"


class TestPermafrostEnhancement:
    """Test permafrost probability enhancement of intensity."""
    
    def test_permafrost_intensity_enhancement(self):
        """Test that permafrost enhances intensity predictions."""
        criterion = PhysicsInformedLoss(alpha_physics=0.1)
        
        # Test with high permafrost probability
        predictions = torch.tensor([[0.5, 100.0, 1.5]], dtype=torch.float32)
        targets = torch.tensor([[0.7, 100.0, 1.5]], dtype=torch.float32)
        
        # High permafrost probability should enhance intensity
        physics_features = torch.tensor([[1e-6, 0.9]], dtype=torch.float32)  # 90% permafrost
        
        loss_high_pf, components_high = criterion(predictions, targets, physics_features=physics_features)
        
        # Test with low permafrost
        physics_features_low = torch.tensor([[1e-6, 0.1]], dtype=torch.float32)  # 10% permafrost
        loss_low_pf, components_low = criterion(predictions, targets, physics_features=physics_features_low)
        
        # Physics loss should be different
        assert components_high['physics_loss'] != components_low['physics_loss'], \
            "Permafrost should affect physics loss"


# Pytest configuration
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])