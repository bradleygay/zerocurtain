"""
Part I: Physics-Informed Zero-Curtain Detection (PINSZC)

Physics-informed detection of zero-curtain events from in situ temperature
measurements, incorporating CryoGrid thermodynamics and Stefan problem constraints.

Components:
- physics_config: Configuration management for paths and parameters
- zero_curtain_detector: Core detection algorithms with CryoGrid integration
- detection_config: Detection configuration and thresholds
- physics_models: Physical model implementations (Stefan problem, heat capacity)
- run_zero_curtain_detection: Main execution script
"""

# Don't import at module level to avoid issues
# Users should import directly from submodules

__all__ = [
    'physics_config',
    'zero_curtain_detector',
    'detection_config',
    'physics_models',
    'run_zero_curtain_detection',
]