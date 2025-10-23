#!/usr/bin/env python3

"""
Temporal Pattern Analysis for Zero-Curtain Event Discrimination

Implements advanced pattern recognition to distinguish between:
1. Rapid/Abrupt Events (< 72 hours)
2. Consecutive Abrupt Events (multiple rapid events, gaps < 48 hours)
3. Extended Persistent Events (≥ 168 hours, stable)
4. Composite Extended Events (≥ 168 hours, variable dynamics)

Based on Part I PINSZC detection framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.cluster import DBSCAN


class TemporalPatternDiscriminator:
    """
    Physics-informed temporal pattern discrimination for zero-curtain events.
    """
    
    def __init__(self, 
                 rapid_max_duration=72.0,
                 extended_min_duration=168.0,
                 consecutive_gap_threshold=48.0):
        
        # Duration thresholds (hours)
        self.RAPID_MAX = rapid_max_duration
        self.EXTENDED_MIN = extended_min_duration
        self.CONSECUTIVE_GAP = consecutive_gap_threshold
        
        # Pattern labels
        self.PATTERN_LABELS = {
            0: 'Rapid/Abrupt',
            1: 'Consecutive Abrupt',
            2: 'Extended Persistent',
            3: 'Composite Extended'
        }
    
    def classify_event_sequence(self, events_df):
        """
        Classify temporal patterns in zero-curtain event sequence.
        
        Args:
            events_df: DataFrame with columns:
                - duration_hours
                - start_time
                - intensity_percentile
                - spatial_extent_meters
        
        Returns:
            pattern_labels: Array of pattern classifications (0-3)
            pattern_features: Dictionary of extracted pattern features
        """
        
        n_events = len(events_df)
        pattern_labels = np.zeros(n_events, dtype=int)
        
        # Convert timestamps if not already datetime
        if 'start_time' in events_df.columns:
            events_df = events_df.copy()
            events_df['start_time'] = pd.to_datetime(events_df['start_time'])
        else:
            raise ValueError("Events dataframe must contain 'start_time' column")
        
        # Sort by time
        events_df = events_df.sort_values('start_time').reset_index(drop=True)
        
        # Extract features
        durations = events_df['duration_hours'].values
        intensities = events_df['intensity_percentile'].values if 'intensity_percentile' in events_df.columns else np.zeros(n_events)
        
        # Calculate temporal gaps
        time_gaps = np.diff(events_df['start_time']).astype('timedelta64[h]').astype(float)
        time_gaps = np.concatenate([[np.inf], time_gaps])  # First event has no previous gap
        
        # Classify each event
        for i in range(n_events):
            duration = durations[i]
            gap_to_prev = time_gaps[i]
            
            # Pattern 0: Rapid/Abrupt Events
            if duration <= self.RAPID_MAX:
                if gap_to_prev > self.CONSECUTIVE_GAP:
                    pattern_labels[i] = 0  # Isolated rapid event
                else:
                    # Check if part of consecutive sequence
                    if i > 0 and durations[i-1] <= self.RAPID_MAX:
                        pattern_labels[i] = 1  # Consecutive abrupt
                    else:
                        pattern_labels[i] = 0
            
            # Pattern 2/3: Extended Events
            elif duration >= self.EXTENDED_MIN:
                # Analyze internal variability
                if i > 0 and i < n_events - 1:
                    # Look at surrounding events for context
                    surrounding_durations = [durations[i-1], duration, durations[i+1]]
                    duration_variance = np.var(surrounding_durations)
                    
                    # High variance suggests composite event
                    if duration_variance > 5000:  # Threshold based on Part I analysis
                        pattern_labels[i] = 3  # Composite extended
                    else:
                        pattern_labels[i] = 2  # Persistent extended
                else:
                    # Edge cases default to persistent
                    pattern_labels[i] = 2
            
            # Intermediate durations (72-168 hours)
            else:
                # Classify based on context
                if gap_to_prev <= self.CONSECUTIVE_GAP and i > 0:
                    pattern_labels[i] = 1  # Part of consecutive sequence
                else:
                    pattern_labels[i] = 0  # Treat as rapid
        
        # Extract pattern features
        pattern_features = self._extract_pattern_features(
            events_df, pattern_labels, durations, time_gaps
        )
        
        return pattern_labels, pattern_features
    
    def _extract_pattern_features(self, events_df, pattern_labels, durations, time_gaps):
        """
        Extract statistical features for each pattern type.
        """
        
        features = {}
        
        for pattern_id in range(4):
            mask = pattern_labels == pattern_id
            
            if mask.sum() > 0:
                features[self.PATTERN_LABELS[pattern_id]] = {
                    'count': mask.sum(),
                    'mean_duration': durations[mask].mean(),
                    'std_duration': durations[mask].std(),
                    'mean_gap': time_gaps[mask][time_gaps[mask] != np.inf].mean() if (mask & (time_gaps != np.inf)).sum() > 0 else np.nan,
                    'total_duration': durations[mask].sum()
                }
            else:
                features[self.PATTERN_LABELS[pattern_id]] = {
                    'count': 0,
                    'mean_duration': np.nan,
                    'std_duration': np.nan,
                    'mean_gap': np.nan,
                    'total_duration': 0.0
                }
        
        return features
    
    def detect_composite_events(self, events_df, spatial_threshold_m=10.0):
        """
        Detect composite extended events using spatiotemporal clustering.
        
        Composite events are extended zero-curtain periods that appear
        as multiple short events but represent a single persistent phenomenon.
        
        Args:
            events_df: DataFrame with zero-curtain events
            spatial_threshold_m: Maximum spatial distance for clustering (meters)
        
        Returns:
            composite_groups: List of event indices for each composite group
        """
        
        if len(events_df) < 2:
            return []
        
        # Prepare features for clustering
        features = []
        
        for idx, row in events_df.iterrows():
            # Normalize features
            lat = row['latitude'] if 'latitude' in events_df.columns else 0.0
            lon = row['longitude'] if 'longitude' in events_df.columns else 0.0
            timestamp = pd.to_datetime(row['start_time']).timestamp() / (3600 * 24)  # Days since epoch
            
            features.append([lat, lon, timestamp])
        
        features = np.array(features)
        
        # Normalize features for DBSCAN
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Apply DBSCAN clustering
        # eps and min_samples tuned for zero-curtain events
        clustering = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
        cluster_labels = clustering.fit_predict(features_normalized)
        
        # Group events by cluster
        composite_groups = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise points
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = events_df.index[cluster_mask].tolist()
            
            # Verify this is a composite event (not just consecutive)
            cluster_events = events_df.iloc[cluster_indices]
            total_duration = cluster_events['duration_hours'].sum()
            
            if total_duration >= self.EXTENDED_MIN:
                composite_groups.append(cluster_indices)
        
        return composite_groups


class TemporalPatternFeatureExtractor(nn.Module):
    """
    Neural network module for extracting temporal pattern features.
    
    Integrates with ZeroCurtainHybridModel to provide pattern-aware
    feature representations.
    """
    
    def __init__(self, input_dim=256, pattern_dim=64):
        super().__init__()
        
        # Pattern-specific feature extractors
        self.rapid_extractor = nn.Sequential(
            nn.Linear(input_dim, pattern_dim),
            nn.LayerNorm(pattern_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.consecutive_extractor = nn.Sequential(
            nn.Linear(input_dim, pattern_dim),
            nn.LayerNorm(pattern_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.extended_extractor = nn.Sequential(
            nn.Linear(input_dim, pattern_dim),
            nn.LayerNorm(pattern_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.composite_extractor = nn.Sequential(
            nn.Linear(input_dim, pattern_dim),
            nn.LayerNorm(pattern_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Pattern fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(pattern_dim * 4, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU()
        )
    
    def forward(self, features, pattern_probs):
        """
        Extract pattern-specific features and fuse them.
        
        Args:
            features: (batch_size, input_dim) - Input features
            pattern_probs: (batch_size, 4) - Pattern probabilities from classifier
        
        Returns:
            pattern_features: (batch_size, input_dim) - Pattern-enhanced features
        """
        
        # Extract pattern-specific features
        rapid_features = self.rapid_extractor(features)
        consecutive_features = self.consecutive_extractor(features)
        extended_features = self.extended_extractor(features)
        composite_features = self.composite_extractor(features)
        
        # Stack pattern features
        pattern_stack = torch.cat([
            rapid_features,
            consecutive_features,
            extended_features,
            composite_features
        ], dim=-1)
        
        # Fuse with pattern probabilities as attention weights
        # Expand pattern_probs to match feature dimensions
        pattern_attention = pattern_probs.unsqueeze(-1).repeat(1, 1, rapid_features.shape[-1])
        pattern_attention = pattern_attention.view(pattern_probs.shape[0], -1)
        
        # Weighted fusion
        pattern_features = self.fusion(pattern_stack * pattern_attention)
        
        return pattern_features