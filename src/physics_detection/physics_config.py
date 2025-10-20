"""
Configuration management for physics-informed zero-curtain detection.
Centralizes all file paths, physical constants, and detection parameters.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import warnings


@dataclass
class DataPaths:
    """Centralized data path configuration."""
    
    # Base directory (default to current working directory for portability)
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    
    # Permafrost probability raster
    permafrost_prob_raster: Optional[Path] = None
    
    # Permafrost zones shapefile
    permafrost_zones_shapefile: Optional[Path] = None
    
    # Snow data NetCDF
    snow_data_netcdf: Optional[Path] = None
    
    # Input in situ measurements parquet
    insitu_measurements_parquet: Optional[Path] = None
    
    # Output directory (relative to base_dir by default)
    output_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Validate and set default paths."""
        # Set output_dir relative to base_dir if not provided
        if self.output_dir is None:
            self.output_dir = self.base_dir / "outputs"
        
        # Set default paths if not provided
        if self.permafrost_prob_raster is None:
            # Attempt to locate in base_dir
            try:
                candidates = list(self.base_dir.glob("**/UiO_PEX_PERPROB*.tif"))
                if candidates:
                    self.permafrost_prob_raster = candidates[0]
            except (PermissionError, OSError):
                pass
        
        if self.permafrost_zones_shapefile is None:
            try:
                candidates = list(self.base_dir.glob("**/UiO_PEX_PERZONES*.shp"))
                if candidates:
                    self.permafrost_zones_shapefile = candidates[0]
            except (PermissionError, OSError):
                pass
        
        if self.snow_data_netcdf is None:
            # Look for snow data NetCDF files
            try:
                specific_file = self.base_dir / "aa6ddc60e4ed01915fb9193bcc7f4146.nc"
                if specific_file.exists():
                    self.snow_data_netcdf = specific_file
                else:
                    candidates = list(self.base_dir.glob("**/*.nc"))
                    # Filter for likely snow data files
                    for candidate in candidates:
                        if any(keyword in candidate.name.lower() for keyword in ['snow', 'swe', 'era5']):
                            self.snow_data_netcdf = candidate
                            break
                    # If still not found, take the first .nc file
                    if self.snow_data_netcdf is None and candidates:
                        self.snow_data_netcdf = candidates[0]
            except (PermissionError, OSError):
                pass
        
        if self.insitu_measurements_parquet is None:
            # Default to merged compressed corrected final parquet
            default_parquet = self.base_dir / "merged_compressed_corrected_final.parquet"
            if default_parquet.exists():
                self.insitu_measurements_parquet = default_parquet
        
        # Ensure output directory exists (with error handling)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError, FileNotFoundError) as e:
            # If we can't create the directory (e.g., in tests or permission issues),
            # just warn - the directory will be created when actually needed
            warnings.warn(
                f"Could not create output directory {self.output_dir}: {e}. "
                f"Directory will be created when needed.",
                UserWarning
            )
    
    def validate_paths(self) -> tuple[bool, list[str]]:
        """Validate that all required paths exist."""
        missing = []
        
        if self.permafrost_prob_raster is None or not self.permafrost_prob_raster.exists():
            missing.append(f"Permafrost probability raster: {self.permafrost_prob_raster}")
        
        if self.permafrost_zones_shapefile is None or not self.permafrost_zones_shapefile.exists():
            missing.append(f"Permafrost zones shapefile: {self.permafrost_zones_shapefile}")
        
        if self.snow_data_netcdf is None or not self.snow_data_netcdf.exists():
            missing.append(f"Snow data NetCDF: {self.snow_data_netcdf}")
        
        if self.insitu_measurements_parquet is None or not self.insitu_measurements_parquet.exists():
            missing.append(f"In situ measurements parquet: {self.insitu_measurements_parquet}")
        
        return len(missing) == 0, missing


@dataclass
class PhysicsParameters:
    """Physical constants and detection thresholds."""
    
    # Temperature thresholds (°C)
    temp_threshold: float = 3.0
    relaxed_temp_threshold: float = 5.0
    
    # Gradient thresholds (°C/day)
    gradient_threshold: float = 1.0
    relaxed_gradient_threshold: float = 2.0
    
    # Duration thresholds (hours)
    min_duration_hours: int = 12
    relaxed_min_duration: int = 6
    
    # Energy thresholds
    phase_change_energy: float = 0.05
    
    # CryoGrid integration flags
    use_cryogrid_enthalpy: bool = True
    use_painter_karra_freezing: bool = True
    use_surface_energy_balance: bool = True
    use_adaptive_timestep: bool = True


@dataclass
class DetectionConfig:
    """Complete detection configuration."""
    
    paths: DataPaths = field(default_factory=DataPaths)
    physics: PhysicsParameters = field(default_factory=PhysicsParameters)
    
    # Processing parameters
    progress_interval: int = 10
    incremental_save_interval: int = 50
    emergency_save_threshold: int = 100
    
    # Output naming
    output_prefix: str = "zero_curtain_physics_informed"
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'paths': {
                'base_dir': str(self.paths.base_dir),
                'permafrost_prob_raster': str(self.paths.permafrost_prob_raster) if self.paths.permafrost_prob_raster else None,
                'permafrost_zones_shapefile': str(self.paths.permafrost_zones_shapefile) if self.paths.permafrost_zones_shapefile else None,
                'snow_data_netcdf': str(self.paths.snow_data_netcdf) if self.paths.snow_data_netcdf else None,
                'insitu_measurements_parquet': str(self.paths.insitu_measurements_parquet) if self.paths.insitu_measurements_parquet else None,
                'output_dir': str(self.paths.output_dir) if self.paths.output_dir else None
            },
            'physics': {
                'temp_threshold': self.physics.temp_threshold,
                'gradient_threshold': self.physics.gradient_threshold,
                'min_duration_hours': self.physics.min_duration_hours,
                'use_cryogrid_enthalpy': self.physics.use_cryogrid_enthalpy
            }
        }