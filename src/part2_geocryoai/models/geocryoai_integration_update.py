#!/usr/bin/env python3

"""
GeoCryoAI Framework Integration for Zero-Curtain Detection

Integrates graph neural network components for spatiotemporal 
permafrost modeling with physics-informed constraints.

References:
- GeoCryoAI: https://github.com/geocryoai/geocryoai
- Arctic permafrost graph modeling methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.data import Data, Batch
import numpy as np


class GeoCryoAISpatialGraphEncoder(nn.Module):
    """
    Spatial graph encoder using GeoCryoAI methodology.
    
    Constructs spatial graphs connecting nearby zero-curtain events
    based on geographic proximity and permafrost characteristics.
    """
    
    def __init__(self, 
                 node_features=256,
                 hidden_channels=128,
                 num_layers=3,
                 spatial_threshold_km=50.0):
        
        super().__init__()
        
        self.node_features = node_features
        self.spatial_threshold = spatial_threshold_km
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(node_features, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, node_features))
        self.norms.append(nn.LayerNorm(node_features))
        
        # Attention-based pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(node_features, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )
    
    def construct_spatial_graph(self, batch_features, batch_metadata):
        """
        Construct spatial graph from batch of events.
        
        Args:
            batch_features: (batch_size, node_features)
            batch_metadata: List of metadata dicts with lat/lon
        
        Returns:
            torch_geometric.data.Data: Spatial graph
        """
        
        batch_size = len(batch_metadata)
        
        # Extract coordinates
        coords = np.array([[m['site_lat'], m['site_lon']] 
                          for m in batch_metadata])
        
        # Calculate pairwise distances (haversine)
        edge_index = []
        edge_attr = []
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                dist = self._haversine_distance(
                    coords[i, 0], coords[i, 1],
                    coords[j, 0], coords[j, 1]
                )
                
                if dist < self.spatial_threshold:
                    # Add bidirectional edges
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    
                    # Edge weight based on inverse distance
                    weight = 1.0 / (1.0 + dist)
                    edge_attr.append(weight)
                    edge_attr.append(weight)
        
        if len(edge_index) == 0:
            # No edges - create self-loops
            edge_index = [[i, i] for i in range(batch_size)]
            edge_attr = [1.0] * batch_size
        
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        edge_attr = torch.FloatTensor(edge_attr)
        
        # Create graph data
        graph_data = Data(
            x=batch_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        return graph_data
    
    def forward(self, batch_features, batch_metadata):
        """
        Process spatial graph.
        
        Args:
            batch_features: (batch_size, node_features)
            batch_metadata: List of metadata dicts
        
        Returns:
            Enhanced node features with spatial context
        """
        
        # Construct spatial graph
        graph_data = self.construct_spatial_graph(batch_features, batch_metadata)
        x = graph_data.x
        edge_index = graph_data.edge_index
        
        # Apply graph convolutions
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            
            if i < len(self.convs) - 1:  # No activation on last layer
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        
        # Attention-based pooling for graph-level representation
        attention_weights = self.attention_pooling(x)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        graph_representation = torch.sum(attention_weights * x, dim=0, keepdim=True)
        
        return x, graph_representation
    
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate haversine distance between two points (in km).
        """
        
        R = 6371.0  # Earth radius in kilometers
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon = np.radians(lon2 - lon1)
        dlat = np.radians(lat2 - lat1)
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance = R * c
        
        return distance


class GeoCryoAITemporalGraphEncoder(nn.Module):
    """
    Temporal graph encoder for zero-curtain event sequences.
    
    Models temporal dependencies using graph attention with
    physics-informed edge weights.
    """
    
    def __init__(self,
                 node_features=256,
                 hidden_channels=128,
                 num_heads=4,
                 num_layers=2):
        
        super().__init__()
        
        # Temporal graph attention layers
        self.temporal_attentions = nn.ModuleList()
        
        # First layer
        self.temporal_attentions.append(
            GATConv(node_features, hidden_channels, heads=num_heads, concat=True)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.temporal_attentions.append(
                GATConv(hidden_channels * num_heads, hidden_channels, 
                       heads=num_heads, concat=True)
            )
        
        # Output layer
        self.temporal_attentions.append(
            GATConv(hidden_channels * num_heads, node_features, 
                   heads=1, concat=False)
        )
    
    def construct_temporal_graph(self, sequence_features, timestamps):
        """
        Construct temporal graph from event sequence.
        
        Args:
            sequence_features: (seq_len, node_features)
            timestamps: (seq_len,) - Event timestamps
        
        Returns:
            torch_geometric.data.Data: Temporal graph
        """
        
        seq_len = len(timestamps)
        
        # Construct temporal edges (sequential + skip connections)
        edge_index = []
        edge_attr = []
        
        for i in range(seq_len):
            # Connect to previous events
            for j in range(max(0, i - 5), i):  # Look back up to 5 timesteps
                edge_index.append([i, j])
                edge_index.append([j, i])
                
                # Edge weight based on temporal proximity
                time_diff = abs(timestamps[i] - timestamps[j])
                weight = np.exp(-time_diff / 24.0)  # Decay over 24 hours
                
                edge_attr.append(weight)
                edge_attr.append(weight)
        
        if len(edge_index) == 0:
            # No edges - create self-loops
            edge_index = [[i, i] for i in range(seq_len)]
            edge_attr = [1.0] * seq_len
        
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        edge_attr = torch.FloatTensor(edge_attr)
        
        graph_data = Data(
            x=sequence_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        return graph_data
    
    def forward(self, sequence_features, timestamps):
        """
        Process temporal graph.
        
        Args:
            sequence_features: (seq_len, node_features)
            timestamps: (seq_len,) - Event timestamps
        
        Returns:
            Enhanced sequence features with temporal context
        """
        
        # Construct temporal graph
        graph_data = self.construct_temporal_graph(sequence_features, timestamps)
        x = graph_data.x
        edge_index = graph_data.edge_index
        
        # Apply graph attention layers
        for i, gat in enumerate(self.temporal_attentions):
            x = gat(x, edge_index)
            
            if i < len(self.temporal_attentions) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        
        return x


class GeoCryoAIHybridGraphModel(nn.Module):
    """
    Hybrid spatiotemporal graph model combining GeoCryoAI methodology
    with physics-informed zero-curtain constraints.
    """
    
    def __init__(self,
                 node_features=256,
                 spatial_hidden=128,
                 temporal_hidden=128,
                 spatial_threshold_km=50.0):
        
        super().__init__()
        
        # Spatial graph encoder
        self.spatial_encoder = GeoCryoAISpatialGraphEncoder(
            node_features=node_features,
            hidden_channels=spatial_hidden,
            spatial_threshold_km=spatial_threshold_km
        )
        
        # Temporal graph encoder
        self.temporal_encoder = GeoCryoAITemporalGraphEncoder(
            node_features=node_features,
            hidden_channels=temporal_hidden
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(node_features * 2, node_features),
            nn.LayerNorm(node_features),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, batch_features, batch_metadata, timestamps):
        """
        Process spatiotemporal graph.
        
        Args:
            batch_features: (batch_size, seq_len, node_features)
            batch_metadata: List of metadata dicts
            timestamps: (batch_size, seq_len) - Event timestamps
        
        Returns:
            Enhanced features with spatiotemporal context
        """
        
        batch_size, seq_len, node_features = batch_features.shape
        
        # Process spatial graph for each timestep
        spatial_outputs = []
        
        for t in range(seq_len):
            timestep_features = batch_features[:, t, :]
            spatial_features, _ = self.spatial_encoder(timestep_features, batch_metadata)
            spatial_outputs.append(spatial_features)
        
        spatial_sequence = torch.stack(spatial_outputs, dim=1)
        
        # Process temporal graph for each sample
        temporal_outputs = []
        
        for b in range(batch_size):
            sample_sequence = batch_features[b]  # (seq_len, node_features)
            sample_timestamps = timestamps[b]
            
            temporal_features = self.temporal_encoder(sample_sequence, sample_timestamps)
            temporal_outputs.append(temporal_features)
        
        temporal_sequence = torch.stack(temporal_outputs, dim=0)
        
        # Fuse spatial and temporal features
        combined = torch.cat([spatial_sequence, temporal_sequence], dim=-1)
        fused_features = self.fusion(combined)
        
        return fused_features