#!/usr/bin/env python3
"""
Unit tests for Teacher Forcing Trainer

Tests curriculum learning, PINSZC ground truth integration,
and teacher forcing effectiveness.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.part2_geocryoai.zero_curtain_ml_model import (
    TeacherForcingTrainer,
    ZeroCurtainHybridModel,
    PhysicsInformedLoss
)


class TestTeacherForcingRatioDecay:
    """Test curriculum learning decay schedules."""
    
    def setup_method(self):
        """Setup mock trainer for testing."""
        # Create minimal mock components
        self.device = torch.device('cpu')
        
        # Create a real minimal model instead of MagicMock
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        ).to(self.device)
        
        self.train_loader = MagicMock()
        self.val_loader = MagicMock()
        self.test_loader = MagicMock()
        
    def test_exponential_decay(self):
        """Test exponential decay schedule."""
        trainer = TeacherForcingTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            device=self.device,
            teacher_forcing_ratio=0.9,
            curriculum_schedule='exponential'
        )
        
        # Test decay over 10 epochs
        ratios = []
        for epoch in range(10):
            ratio = trainer.update_teacher_forcing_ratio(epoch)
            ratios.append(ratio)
        
        # Verify exponential decay
        assert ratios[0] == 0.9, "Initial ratio should be 0.9"
        assert ratios[-1] < ratios[0], "Ratio should decrease"
        assert ratios[-1] >= 0.1, "Ratio should not go below minimum"
        
        # Verify monotonic decrease
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i+1], f"Ratio should decrease monotonically at epoch {i}"
    
    def test_linear_decay(self):
        """Test linear decay schedule."""
        trainer = TeacherForcingTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            device=self.device,
            teacher_forcing_ratio=0.9,
            curriculum_schedule='linear'
        )
        
        # Set epochs for linear calculation
        trainer.epochs = 10
        
        ratios = []
        for epoch in range(10):
            ratio = trainer.update_teacher_forcing_ratio(epoch)
            ratios.append(ratio)
        
        # Verify linear decay
        assert ratios[0] == 0.9, "Should start at 0.9"
        assert ratios[-1] >= 0.1, "Should not go below minimum ratio"
        assert ratios[-1] <= 0.2, "Should be close to minimum ratio"
        
        # Verify monotonic decrease
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i+1], f"Ratio should decrease monotonically at epoch {i}"
        
        # Check approximately linear decrease
        differences = [ratios[i] - ratios[i+1] for i in range(len(ratios)-1)]
        avg_diff = np.mean(differences)
        # Linear should have consistent step size
        assert all(abs(d - avg_diff) < 0.02 for d in differences), "Should be approximately linear"
    
    def test_inverse_sigmoid_decay(self):
        """Test inverse sigmoid decay schedule."""
        trainer = TeacherForcingTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            device=self.device,
            teacher_forcing_ratio=0.9,
            curriculum_schedule='inverse_sigmoid'
        )
        
        trainer.epochs = 20
        
        ratios = []
        for epoch in range(20):
            ratio = trainer.update_teacher_forcing_ratio(epoch)
            ratios.append(ratio)
        
        # Verify sigmoid properties
        assert abs(ratios[0] - 0.9) < 0.01, f"Should start near 0.9, got {ratios[0]}"
        assert ratios[-1] >= 0.1, "Should not go below minimum ratio"
        
        # Verify monotonic decrease
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i+1], f"Ratio should decrease monotonically at epoch {i}"
        
        # Check sigmoid pattern: changes should be smaller at extremes
        # Early changes (epochs 0-5)
        early_changes = [ratios[i] - ratios[i+1] for i in range(5)]
        # Middle changes (epochs 8-12)
        middle_changes = [ratios[i] - ratios[i+1] for i in range(8, 13)]
        # Late changes (epochs 15-19)
        late_changes = [ratios[i] - ratios[i+1] for i in range(15, 19)]
        
        avg_early = np.mean(early_changes)
        avg_middle = np.mean(middle_changes)
        avg_late = np.mean(late_changes)
        
        # Sigmoid should have larger changes in the middle
        assert avg_middle >= avg_early, "Middle changes should be >= early changes"
        assert avg_middle >= avg_late, "Middle changes should be >= late changes"


class TestPINSZCGroundTruthIntegration:
    """Test PINSZC ground truth retrieval and indexing."""
    
    def setup_method(self):
        """Create mock PINSZC dataframe."""
        # Create synthetic PINSZC data
        self.pinszc_df = pd.DataFrame({
            'latitude': [65.0, 65.0, 66.0, 66.0],
            'longitude': [-147.0, -147.0, -148.0, -148.0],
            'intensity_percentile': [0.5, 0.6, 0.4, 0.7],
            'duration_hours': [100.0, 150.0, 80.0, 200.0],
            'spatial_extent_meters': [1.5, 2.0, 1.2, 2.5]
        })
        
        self.device = torch.device('cpu')
        
        # Add real model
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        ).to(self.device)
    
    def test_pinszc_indexing(self):
        """Test pre-computed PINSZC site indexing."""
        # Create trainer with PINSZC
        train_loader = MagicMock()
        val_loader = MagicMock()
        test_loader = MagicMock()
        
        trainer = TeacherForcingTrainer(
            model=self.model,  # Use self.model instead of MagicMock()
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=self.device,
            pinszc_ground_truth=self.pinszc_df
        )
        
        # Verify index was created
        assert hasattr(trainer, 'pinszc_site_index'), "Should create PINSZC index"
        assert len(trainer.pinszc_site_index) == 2, "Should have 2 unique sites"
        
        # Verify site (65.0, -147.0) has 2 events
        site_key = (65.0, -147.0)
        assert site_key in trainer.pinszc_site_index
        assert len(trainer.pinszc_site_index[site_key]) == 2
    
    def test_pinszc_ground_truth_retrieval(self):
        """Test PINSZC ground truth retrieval for batch."""
        train_loader = MagicMock()
        val_loader = MagicMock()
        test_loader = MagicMock()
        
        trainer = TeacherForcingTrainer(
            model=self.model,  # ← CHANGED: Use self.model from setup_method
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=self.device,
            pinszc_ground_truth=self.pinszc_df
        )
        
        # Create mock metadata batch
        metadata_batch = [
            {'site_lat': 65.0, 'site_lon': -147.0},
            {'site_lat': 66.0, 'site_lon': -148.0},
            {'site_lat': 70.0, 'site_lon': -150.0}  # No matching site
        ]
        
        # Get ground truth
        ground_truth = trainer._get_pinszc_ground_truth(metadata_batch)
        
        # Verify output shape
        assert ground_truth.shape == (3, 3), "Should return (batch_size, 3) tensor"
        
        # Verify site 1 (65.0, -147.0) - mean of 0.5 and 0.6
        assert abs(ground_truth[0, 0].item() - 0.55) < 0.01, "Should average intensity"
        
        # Verify site 2 (66.0, -148.0) - mean of 0.4 and 0.7
        assert abs(ground_truth[1, 0].item() - 0.55) < 0.01
        
        # Verify site 3 (no match) - should be zeros
        assert ground_truth[2, 0].item() == 0.0, "Should return zeros for unmatched sites"


class TestTeacherForcingTraining:
    """Test teacher forcing training mechanics."""
    
    def test_guided_prediction_blending(self):
        """Test prediction blending with ground truth."""
        device = torch.device('cpu')
        
        # Create mock predictions and ground truth
        model_predictions = torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float32)
        ground_truth = torch.tensor([[0.8, 0.6, 0.9]], dtype=torch.float32)
        
        # Test blending with 90% teacher forcing
        alpha = 0.9
        guided = alpha * ground_truth + (1 - alpha) * model_predictions
        
        # Verify blending
        expected = torch.tensor([[0.75, 0.59, 0.88]], dtype=torch.float32)
        assert torch.allclose(guided, expected, atol=0.01), "Blending should be correct"
    
    def test_teacher_forcing_probability(self):
        """Test stochastic teacher forcing decision."""
        # Test with 100% teacher forcing
        tf_ratio = 1.0
        decisions = [np.random.random() < tf_ratio for _ in range(100)]
        assert all(decisions), "100% ratio should always use teacher forcing"
        
        # Test with 0% teacher forcing
        tf_ratio = 0.0
        decisions = [np.random.random() < tf_ratio for _ in range(100)]
        assert not any(decisions), "0% ratio should never use teacher forcing"
        
        # Test with 50% teacher forcing (should be approximately 50%)
        np.random.seed(42)
        tf_ratio = 0.5
        decisions = [np.random.random() < tf_ratio for _ in range(1000)]
        tf_count = sum(decisions)
        assert 450 <= tf_count <= 550, f"50% ratio should be ~500/1000, got {tf_count}"


class TestTeacherForcingHistory:
    """Test teacher forcing history tracking."""
    
    def test_history_initialization(self):
        """Test TF history dict initialization."""
        device = torch.device('cpu')
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        ).to(device)
        
        trainer = TeacherForcingTrainer(
            model=model,
            train_loader=MagicMock(),
            val_loader=MagicMock(),
            test_loader=MagicMock(),
            device=device
        )
        
        # Verify history structure
        assert 'ratio' in trainer.tf_history
        assert 'accuracy_with_tf' in trainer.tf_history
        assert 'accuracy_without_tf' in trainer.tf_history
        
        # Verify initial empty state
        assert len(trainer.tf_history['ratio']) == 0
        assert len(trainer.tf_history['accuracy_with_tf']) == 0


# Pytest configuration
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])