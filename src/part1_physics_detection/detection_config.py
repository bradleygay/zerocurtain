"""
Physics-Informed Zero-Curtain Detection Configuration

This module centralizes all configurable parameters for zero-curtain detection,
enabling systematic parameter tuning and ablation studies.

Institution: [REDACTED_AFFILIATION] Arctic Research
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class ThermodynamicConstants:
    """Physical constants from LPJ-EOSIM and CryoGrid frameworks."""
    
    # Latent heat and heat capacities [J m-3 K-1]
    LHEAT: float = 3.34E8  # Latent heat of fusion
    CWATER: float = 4180000  # Heat capacity water
    CICE: float = 1700000  # Heat capacity ice
    CORG: float = 3117800  # Heat capacity organic matter
    CMIN: float = 2380000  # Heat capacity mineral soil
    
    # Thermal conductivities [W m-1 K-1]
    KWATER: float = 0.57
    KICE: float = 2.2
    KORG: float = 0.25
    KMIN: float = 2.0
    
    # Soil physics
    RHO_WATER: float = 1000  # Water density [kg m-3]
    G: float = 9.81  # Gravitational acceleration [m s-2]
    MU_WATER: float = 1.0e-3  # Dynamic viscosity [Pa s]
    
    # CryoGrid constants
    LVOL_SL: float = 3.34E8  # Volumetric latent heat [J m-3]
    STEFAN_BOLTZMANN: float = 5.67e-8  # [W m-2 K-4]
    TMFW: float = 273.15  # Freezing temperature [K]


@dataclass
class NumericalParameters:
    """Numerical solver configuration for Stefan problem and time integration."""
    
    DT: float = 86400  # Time step [s] - daily resolution
    DZ_MIN: float = 0.01  # Minimum layer thickness [m]
    MAX_LAYERS: int = 50  # Maximum soil layers
    CONVERGENCE_TOL: float = 1e-6  # Solver convergence tolerance
    MAX_ENTHALPY_CHANGE: float = 50e3  # Maximum enthalpy change per timestep [J m-3]


@dataclass
class DetectionThresholds:
    """Multi-pathway detection criteria with standard and relaxed thresholds."""
    
    # Standard pathway (restrictive)
    TEMP_THRESHOLD: float = 3.0  # Temperature threshold [°C]
    GRADIENT_THRESHOLD: float = 1.0  # Thermal gradient threshold [°C/day]
    MIN_DURATION_HOURS: int = 12  # Minimum event duration [hours]
    PHASE_CHANGE_ENERGY: float = 0.05  # Energy threshold for phase change
    
    # Relaxed pathway (permissive)
    RELAXED_TEMP_THRESHOLD: float = 5.0  # Very permissive temperature range
    RELAXED_GRADIENT_THRESHOLD: float = 2.0  # Very permissive gradient
    RELAXED_MIN_DURATION: int = 6  # Very short duration acceptable
    
    # Isothermal detection
    ISOTHERMAL_WINDOW: int = 24  # Rolling window size [points]
    ISOTHERMAL_STD_THRESHOLD: float = 0.5  # Standard deviation threshold [°C]
    ISOTHERMAL_MEAN_THRESHOLD: float = 2.0  # Mean temperature threshold [°C]


@dataclass
class CryoGridIntegration:
    """CryoGrid-specific model integration flags and parameters."""
    
    use_cryogrid_enthalpy: bool = True
    use_painter_karra_freezing: bool = True
    use_surface_energy_balance: bool = True
    use_adaptive_timestep: bool = True
    
    # Van Genuchten parameters (soil-specific)
    van_genuchten_alpha: float = 0.5  # [m^-1]
    van_genuchten_n: float = 2.0
    soil_porosity: float = 0.4


@dataclass
class AuxiliaryDataPaths:
    """File paths for auxiliary geospatial datasets."""
    
    permafrost_probability_raster: str = (
        '/Users/[USER]/new2/UiO_PEX_PERPROB_5/'
        'UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tif'
    )
    
    permafrost_zones_shapefile: str = (
        '/Users/[USER]/new2/UiO_PEX_PERZONES_5/'
        'UiO_PEX_PERZONES_5.0_20181128_2000_2016_NH.shp'
    )
    
    snow_reanalysis_netcdf: str = (
        '/Users/[USER]/aa6ddc60e4ed01915fb9193bcc7f4146.nc'
    )


@dataclass
class ProcessingConfiguration:
    """High-level processing and output configuration."""
    
    # Data filtering
    cold_season_months: List[int] = field(
        default_factory=lambda: [9, 10, 11, 12, 1, 2, 3, 4, 5]
    )
    min_measurements_per_site: int = 50  # Reduced from 200 for sparse datasets
    
    # Permafrost suitability
    permafrost_prob_threshold: float = 0.0  # Permissive: any probability > 0
    arctic_latitude_threshold: float = 50.0  # Minimum latitude for Arctic region
    
    # Output configuration
    save_incremental_checkpoints: bool = True
    incremental_save_interval: int = 50  # Save every N sites
    compression: str = 'snappy'
    
    # Progress reporting
    progress_report_interval: int = 10
    enable_diagnostic_logging: bool = True
    verbose: bool = True


@dataclass
class DetectionConfiguration:
    """Master configuration aggregating all detection parameters."""
    
    thermodynamics: ThermodynamicConstants = field(default_factory=ThermodynamicConstants)
    numerics: NumericalParameters = field(default_factory=NumericalParameters)
    thresholds: DetectionThresholds = field(default_factory=DetectionThresholds)
    cryogrid: CryoGridIntegration = field(default_factory=CryoGridIntegration)
    auxiliary_paths: AuxiliaryDataPaths = field(default_factory=AuxiliaryDataPaths)
    processing: ProcessingConfiguration = field(default_factory=ProcessingConfiguration)
    
    def to_dict(self) -> Dict:
        """Export configuration as nested dictionary for serialization."""
        return {
            'thermodynamics': self.thermodynamics.__dict__,
            'numerics': self.numerics.__dict__,
            'thresholds': self.thresholds.__dict__,
            'cryogrid': self.cryogrid.__dict__,
            'auxiliary_paths': self.auxiliary_paths.__dict__,
            'processing': self.processing.__dict__
        }
    
    def validate(self) -> bool:
        """Validate configuration parameters for physical consistency."""
        
        # Thermodynamic validation
        assert self.thermodynamics.LHEAT > 0, "Latent heat must be positive"
        assert self.thermodynamics.TMFW > 0, "Freezing temperature must be positive"
        
        # Numerical validation
        assert self.numerics.DT > 0, "Time step must be positive"
        assert self.numerics.MAX_LAYERS > 0, "Must have at least one soil layer"
        
        # Detection threshold validation
        assert self.thresholds.TEMP_THRESHOLD > 0, "Temperature threshold must be positive"
        assert self.thresholds.MIN_DURATION_HOURS > 0, "Minimum duration must be positive"
        
        # Relaxed thresholds must be more permissive than standard
        assert (self.thresholds.RELAXED_TEMP_THRESHOLD >= 
                self.thresholds.TEMP_THRESHOLD), "Relaxed temp threshold must be >= standard"
        assert (self.thresholds.RELAXED_MIN_DURATION <= 
                self.thresholds.MIN_DURATION_HOURS), "Relaxed duration must be <= standard"
        
        return True
    
    def summary(self) -> str:
        """Generate human-readable configuration summary."""
        
        lines = [
            "=== Zero-Curtain Detection Configuration ===",
            "",
            "Detection Thresholds:",
            f"  Standard: ±{self.thresholds.TEMP_THRESHOLD}°C, "
            f"min {self.thresholds.MIN_DURATION_HOURS}h",
            f"  Relaxed:  ±{self.thresholds.RELAXED_TEMP_THRESHOLD}°C, "
            f"min {self.thresholds.RELAXED_MIN_DURATION}h",
            "",
            "CryoGrid Integration:",
            f"  Enthalpy formulation: {self.cryogrid.use_cryogrid_enthalpy}",
            f"  Painter-Karra freezing: {self.cryogrid.use_painter_karra_freezing}",
            f"  Surface energy balance: {self.cryogrid.use_surface_energy_balance}",
            f"  Adaptive timestep: {self.cryogrid.use_adaptive_timestep}",
            "",
            "Processing:",
            f"  Cold season months: {self.processing.cold_season_months}",
            f"  Minimum measurements/site: {self.processing.min_measurements_per_site}",
            f"  Incremental saves: {self.processing.save_incremental_checkpoints} "
            f"(every {self.processing.incremental_save_interval} sites)",
            "",
            "Numerical Parameters:",
            f"  Time step: {self.numerics.DT}s ({self.numerics.DT/86400:.1f} days)",
            f"  Soil layers: max {self.numerics.MAX_LAYERS}, "
            f"min spacing {self.numerics.DZ_MIN}m",
            "============================================"
        ]
        
        return "\n".join(lines)


# Default configuration instance
DEFAULT_CONFIG = DetectionConfiguration()


def get_ablation_configs() -> Dict[str, DetectionConfiguration]:
    """
    Generate ablation study configurations for systematic physics evaluation.
    
    Returns ablation configurations testing:
    1. Standard physics (LPJ-EOSIM only)
    2. CryoGrid enthalpy only
    3. Full CryoGrid integration
    4. Relaxed detection thresholds
    5. Conservative detection thresholds
    """
    
    configs = {}
    
    # Baseline: Standard physics without CryoGrid
    baseline = DetectionConfiguration()
    baseline.cryogrid.use_cryogrid_enthalpy = False
    baseline.cryogrid.use_painter_karra_freezing = False
    baseline.cryogrid.use_surface_energy_balance = False
    baseline.cryogrid.use_adaptive_timestep = False
    configs['baseline_lpj'] = baseline
    
    # CryoGrid enthalpy only
    enthalpy_only = DetectionConfiguration()
    enthalpy_only.cryogrid.use_painter_karra_freezing = False
    enthalpy_only.cryogrid.use_surface_energy_balance = False
    configs['cryogrid_enthalpy'] = enthalpy_only
    
    # Full CryoGrid (default)
    configs['cryogrid_full'] = DetectionConfiguration()
    
    # Relaxed detection (for sparse/noisy data)
    relaxed = DetectionConfiguration()
    relaxed.thresholds.TEMP_THRESHOLD = 5.0
    relaxed.thresholds.GRADIENT_THRESHOLD = 2.0
    relaxed.thresholds.MIN_DURATION_HOURS = 6
    configs['relaxed_thresholds'] = relaxed
    
    # Conservative detection (high confidence)
    conservative = DetectionConfiguration()
    conservative.thresholds.TEMP_THRESHOLD = 1.0
    conservative.thresholds.GRADIENT_THRESHOLD = 0.3
    conservative.thresholds.MIN_DURATION_HOURS = 48
    configs['conservative_thresholds'] = conservative
    
    return configs


if __name__ == "__main__":
    """Configuration validation and summary display."""
    
    config = DEFAULT_CONFIG
    
    print("Validating configuration...")
    try:
        config.validate()
        print(" Configuration validation passed\n")
    except AssertionError as e:
        print(f" Configuration validation failed: {e}\n")
        exit(1)
    
    print(config.summary())
    
    print("\n--- Ablation Study Configurations ---")
    ablation_configs = get_ablation_configs()
    for name, cfg in ablation_configs.items():
        print(f"\n{name}:")
        print(f"  CryoGrid enthalpy: {cfg.cryogrid.use_cryogrid_enthalpy}")
        print(f"  Painter-Karra: {cfg.cryogrid.use_painter_karra_freezing}")
        print(f"  Temp threshold: ±{cfg.thresholds.TEMP_THRESHOLD}°C")
        print(f"  Min duration: {cfg.thresholds.MIN_DURATION_HOURS}h")
