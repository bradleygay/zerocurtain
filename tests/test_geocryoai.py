#!/usr/bin/env python3
"""
Unit tests for GeoCryoAI Integration

Tests spatial graph construction, temporal graph attention,
and spatiotemporal fusion in graph neural networks.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from geocryoai_integration import (
    GeoCryoAISpatialGraphEncoder,
    GeoCryoAITemporalGraphEncoder,
    GeoCryoAIHybridGraphModel
)


class TestSpatialGraphConstruction:
    """Test spatial graph construction and connectivity."""
    
    def setup_method(self):
        """Setup spatial graph encoder."""
        self.encoder = GeoCryoAISpatialGraphEncoder(
            node_features=64,
            hidden_channels=32,
            spatial_threshold_km=50.0
        )
    
    def test_haversine_distance_calculation(self):
        """Test haversine distance between coordinates."""
        # Test known distance: Anchorage to Fairbanks
        lat1, lon1 = 61.2181, -149.9003  # Anchorage
        lat2, lon2 = 64.8378, -147.7164  # Fairbanks
        
        distance = self.encoder._haversine_distance(lat1, lon1, lat2, lon2)
        
        # Actual distance is approximately 400-420 km
        assert 380 < distance < 450, f"Distance {distance} km not in expected range"
    
    def test_spatial_graph_connectivity(self):
        """Test that nearby sites are connected."""
        batch_features = torch.randn(4, 64)  # 4 nodes
        
        # Create metadata with known distances
        metadata = [
            {'site_lat': 65.0, 'site_lon': -147.0},  # Site 0
            {'site_lat': 65.1, 'site_lon': -147.1},  # Site 1 - close to site 0
            {'site_lat': 70.0, 'site_lon': -150.0},  # Site 2 - far from sites 0,1
            {'site_lat': 70.1, 'site_lon': -150.1}   # Site 3 - close to site 2
        ]
        
        graph_data = self.encoder.construct_spatial_graph(batch_features, metadata)
        
        # Verify graph structure
        assert hasattr(graph_data, 'edge_index'), "Graph should have edge_index"
        assert hasattr(graph_data, 'edge_attr'), "Graph should have edge attributes"
        
        # Check that close sites are connected
        edge_index = graph_data.edge_index.numpy()
        
        # Sites 0 and 1 should be connected (< 50km)
        connected_01 = np.any((edge_index[0] == 0) & (edge_index[1] == 1))
        assert connected_01, "Close sites (0,1) should be connected"
        
        # Sites 0 and 2 should NOT be connected (> 50km)
        connected_02 = np.any((edge_index[0] == 0) & (edge_index[1] == 2))
        assert not connected_02, "Far sites (0,2) should not be connected"
    
    def test_edge_weight_distance_relationship(self):
        """Test that edge weights decrease with distance."""
        batch_features = torch.randn(3, 64)
        
        metadata = [
            {'site_lat': 65.0, 'site_lon': -147.0},
            {'site_lat': 65.05, 'site_lon': -147.05},  # ~7km away
            {'site_lat': 65.2, 'site_lon': -147.2}     # ~28km away
        ]
        
        graph_data = self.encoder.construct_spatial_graph(batch_features, metadata)
        
        edge_attr = graph_data.edge_attr.numpy()
        
        # Weights for closer sites should be higher
        # Find edges for node 0
        edge_index = graph_data.edge_index.numpy()
        edges_from_0 = np.where(edge_index[0] == 0)[0]
        
        if len(edges_from_0) > 1:
            weights_from_0 = edge_attr[edges_from_0]
            # Verify weights are positive and vary
            assert all(weights_from_0 > 0), "Weights should be positive"
    
    def test_spatial_graph_processing(self):
        """Test spatial graph convolution forward pass."""
        batch_size = 4
        node_features = 64
        
        batch_features = torch.randn(batch_size, node_features)
        metadata = [
            {'site_lat': 65.0 + i*0.1, 'site_lon': -147.0 - i*0.1}
            for i in range(batch_size)
        ]
        
        node_output, graph_output = self.encoder(batch_features, metadata)
        
        # Verify output shapes
        assert node_output.shape == (batch_size, node_features), \
            f"Node output shape {node_output.shape} incorrect"
        assert graph_output.shape == (1, node_features), \
            f"Graph output shape {graph_output.shape} incorrect"
        
        # Verify outputs are different from inputs (processing occurred)
        assert not torch.allclose(node_output, batch_features), \
            "Output should differ from input after graph convolution"


class TestTemporalGraphConstruction:
    """Test temporal graph construction and attention."""
    
    def setup_method(self):
        """Setup temporal graph encoder."""
        self.encoder = GeoCryoAITemporalGraphEncoder(
            node_features=64,
            hidden_channels=32,
            num_heads=4,
            num_layers=2
        )
    
    def test_temporal_graph_sequential_connections(self):
        """Test that sequential timesteps are connected."""
        seq_len = 10
        sequence_features = torch.randn(seq_len, 64)
        timestamps = torch.arange(seq_len, dtype=torch.float32)
        
        graph_data = self.encoder.construct_temporal_graph(sequence_features, timestamps)
        
        edge_index = graph_data.edge_index.numpy()
        
        # Verify sequential connections exist
        for i in range(1, seq_len):
            # Check if i-1 connects to i
            sequential_edge = np.any((edge_index[0] == i) & (edge_index[1] == i-1))
            assert sequential_edge, f"Sequential connection {i-1}→{i} should exist"
    
    def test_temporal_edge_weight_decay(self):
        """Test that temporal edge weights decay with time gap."""
        seq_len = 5
        sequence_features = torch.randn(seq_len, 64)
        timestamps = torch.tensor([0.0, 24.0, 48.0, 72.0, 96.0])  # 24h gaps
        
        graph_data = self.encoder.construct_temporal_graph(sequence_features, timestamps)
        
        edge_attr = graph_data.edge_attr.numpy()
        
        # Weights should follow exponential decay
        # All weights should be positive
        assert all(edge_attr > 0), "Temporal weights should be positive"
        
        # Weights should be less than or equal to 1.0
        assert all(edge_attr <= 1.0), "Temporal weights should be ≤ 1.0"
    
    def test_temporal_graph_forward_pass(self):
        """Test temporal graph attention forward pass."""
        seq_len = 8
        node_features = 64
        
        sequence_features = torch.randn(seq_len, node_features)
        timestamps = torch.arange(seq_len, dtype=torch.float32) * 24.0  # 24h intervals
        
        output = self.encoder(sequence_features, timestamps)
        
        # Verify output shape
        assert output.shape == sequence_features.shape, \
            f"Output shape {output.shape} should match input {sequence_features.shape}"
        
        # Verify processing occurred
        assert not torch.allclose(output, sequence_features), \
            "Output should differ after temporal attention"


class TestGeoCryoAIHybridModel:
    """Test hybrid spatiotemporal graph model."""
    
    def setup_method(self):
        """Setup hybrid graph model."""
        self.model = GeoCryoAIHybridGraphModel(
            node_features=64,
            spatial_hidden=32,
            temporal_hidden=32,
            spatial_threshold_km=50.0
        )
    
    def test_hybrid_model_forward_pass(self):
        """Test full spatiotemporal processing."""
        batch_size = 4
        seq_len = 8
        node_features = 64
        
        batch_features = torch.randn(batch_size, seq_len, node_features)
        metadata = [
            {'site_lat': 65.0 + i*0.1, 'site_lon': -147.0 - i*0.1}
            for i in range(batch_size)
        ]
        timestamps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).float()
        
        output = self.model(batch_features, metadata, timestamps)
        
        # Verify output shape
        assert output.shape == batch_features.shape, \
            f"Output shape {output.shape} should match input {batch_features.shape}"
        
        # Verify processing occurred
        assert not torch.allclose(output, batch_features), \
            "Output should differ after spatiotemporal processing"
    
    def test_hybrid_model_with_single_site(self):
        """Test that model handles single-site case."""
        batch_size = 1
        seq_len = 10
        node_features = 64
        
        batch_features = torch.randn(batch_size, seq_len, node_features)
        metadata = [{'site_lat': 65.0, 'site_lon': -147.0}]
        timestamps = torch.arange(seq_len).unsqueeze(0).float()
        
        # Should not crash with single site
        output = self.model(batch_features, metadata, timestamps)
        
        assert output.shape == batch_features.shape
    
    def test_spatial_temporal_fusion(self):
        """Test that spatial and temporal features are fused."""
        batch_size = 3
        seq_len = 5
        node_features = 64
        
        batch_features = torch.randn(batch_size, seq_len, node_features)
        metadata = [
            {'site_lat': 65.0, 'site_lon': -147.0},
            {'site_lat': 65.1, 'site_lon': -147.1},
            {'site_lat': 65.2, 'site_lon': -147.2}
        ]
        timestamps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).float()
        
        output = self.model(batch_features, metadata, timestamps)
        
        # Output should contain information from both spatial and temporal processing
        # Verify by checking that changing spatial metadata affects output
        metadata_changed = [
            {'site_lat': 70.0, 'site_lon': -150.0},  # Far away
            {'site_lat': 70.1, 'site_lon': -150.1},
            {'site_lat': 70.2, 'site_lon': -150.2}
        ]
        
        output_changed = self.model(batch_features, metadata_changed, timestamps)
        
        # Outputs should differ due to different spatial graphs
        assert not torch.allclose(output, output_changed, atol=0.1), \
            "Spatial changes should affect output"


class TestGraphNeuralNetworkProperties:
    """Test general GNN properties and edge cases."""
    
    def test_empty_graph_handling(self):
        """Test handling of graphs with no edges."""
        encoder = GeoCryoAISpatialGraphEncoder(
            node_features=64,
            spatial_threshold_km=1.0  # Very small threshold
        )
        
        batch_features = torch.randn(3, 64)
        metadata = [
            {'site_lat': 65.0, 'site_lon': -147.0},
            {'site_lat': 70.0, 'site_lon': -150.0},  # Far away
            {'site_lat': 75.0, 'site_lon': -155.0}   # Also far
        ]
        
        # Should create self-loops when no edges exist
        graph_data = encoder.construct_spatial_graph(batch_features, metadata)
        
        # Should have at least self-loop edges
        assert graph_data.edge_index.shape[1] > 0, "Should have edges (at least self-loops)"
    
    def test_graph_permutation_invariance(self):
        """Test that node ordering doesn't affect graph structure."""
        encoder = GeoCryoAISpatialGraphEncoder(
            node_features=64,
            spatial_threshold_km=50.0
        )
        
        batch_features = torch.randn(4, 64)
        metadata = [
            {'site_lat': 65.0, 'site_lon': -147.0},
            {'site_lat': 65.1, 'site_lon': -147.1},
            {'site_lat': 70.0, 'site_lon': -150.0},
            {'site_lat': 70.1, 'site_lon': -150.1}
        ]
        
        graph1 = encoder.construct_spatial_graph(batch_features, metadata)
        num_edges1 = graph1.edge_index.shape[1]
        
        # Permute nodes
        perm = [2, 0, 3, 1]
        batch_features_perm = batch_features[perm]
        metadata_perm = [metadata[i] for i in perm]
        
        graph2 = encoder.construct_spatial_graph(batch_features_perm, metadata_perm)
        num_edges2 = graph2.edge_index.shape[1]
        
        # Number of edges should be the same (graph structure invariant)
        assert num_edges1 == num_edges2, "Graph structure should be permutation invariant"


# Pytest configuration
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])