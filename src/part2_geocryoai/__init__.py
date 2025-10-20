"""
Part II: GeoCryoAI with Teacher Forcing

Spatiotemporal graph neural networks with teacher forcing curriculum learning
for physics-informed zero-curtain event prediction.

Components:
- zero_curtain_ml_model: Main training pipeline and hybrid model architecture
- geocryoai_integration: Spatial and temporal graph neural network encoders
- temporal_pattern_analyzer: Event pattern classification (4 pattern types)
- teacher_forcing_prep: Ground truth preparation for curriculum learning
- config_loader: YAML configuration management
- modeling_module: General modeling utilities
"""

# Don't import at module level to avoid circular dependencies
# Users should import directly from submodules

__all__ = [
    'zero_curtain_ml_model',
    'geocryoai_integration',
    'temporal_pattern_analyzer',
    'teacher_forcing_prep',
    'config_loader',
    'modeling_module',
]