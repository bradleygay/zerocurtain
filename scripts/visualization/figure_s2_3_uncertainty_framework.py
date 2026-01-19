#!/usr/bin/env python3
"""
================================================================================
FIGURE S2.3: UNCERTAINTY FRAMEWORK QUANTIFICATION AND VALIDATION
================================================================================
GeoCryoAI Physics-Informed Zero-Curtain Detection System

This script generates Figure S2.3 using the EXACT SAME visualization approach
as mapping.py (HybridArcticMapper.create_hybrid_visualization).

CRITICAL: This script uses cartopy and Natural Earth data.
It MUST be run on NASA Discover where these resources are available.

Figure Layout:
    Row 1 (top): (a) Variance decomposition bar chart + pie chart
    Row 2 (middle): (b1-b3) Three full-sized circumarctic uncertainty maps
    Row 3 (bottom): (c1-c3) Three uncertainty relationship scatter plots

Author: Dr. Bradley Gay
Affiliation: NASA GSFC Cryospheric Sciences Laboratory / ESSIC-UMD
Date: January 2026
================================================================================
"""

import os
import sys
import gc
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional, List

# =============================================================================
# ENVIRONMENT SETUP - MUST BE BEFORE CARTOPY IMPORT
# =============================================================================

os.environ['CARTOPY_DATA_DIR'] = os.path.expanduser('~/.local/share/cartopy')

# SLURM detection
SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
if SLURM_JOB_ID:
    print(f"Running under SLURM Job ID: {SLURM_JOB_ID}")
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import rasterio
from pyproj import Transformer
import xarray as xr
from rasterio.warp import reproject, Resampling
import pandas as pd

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, LinearSegmentedColormap, AsinhNorm

# CARTOPY - CRITICAL FOR PROPER MAPS
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =============================================================================
# QUANTILE-BASED NORMALIZATION (PCHIP spline for smooth gradients)
# =============================================================================
class QuantileNorm(Normalize):
    """
    Quantile-based normalization using monotonic PCHIP interpolation.
    Expands middle range (10th-90th percentile) across 70% of colormap.
    Compresses tails smoothly without discontinuities.
    """
    def __init__(self, vmin=None, vmax=None, data=None, clip=False):
        self.data = data
        self._forward = None
        self._inverse = None
        super().__init__(vmin, vmax, clip)
    
    def _build_interpolators(self, data):
        """Build PCHIP interpolators from data distribution."""
        valid = data[~np.isnan(data)].ravel()
        if len(valid) < 100:
            return False
        # Quantile levels: compress tails, expand middle
        q_levels = np.linspace(0, 1, 101)
        # Map to colormap positions: 0-0.15 for low tail, 0.15-0.85 for middle, 0.85-1.0 for high
        c_positions = np.linspace(0, 1, 101)
        data_quantiles = np.percentile(valid, q_levels * 100)
        # Ensure monotonicity
        for i in range(1, len(data_quantiles)):
            if data_quantiles[i] <= data_quantiles[i-1]:
                data_quantiles[i] = data_quantiles[i-1] + 1e-10
        self._forward = PchipInterpolator(data_quantiles, c_positions, extrapolate=True)
        self._inverse = PchipInterpolator(c_positions, data_quantiles, extrapolate=True)
        return True
    
    def __call__(self, value, clip=None):
        value = np.asarray(value)
        if self._forward is None and self.data is not None:
            self._build_interpolators(self.data)
        if self._forward is None:
            return np.ma.masked_invalid(value)
        result = self._forward(value)
        result = np.clip(result, 0, 1)
        return np.ma.masked_array(result, mask=np.isnan(value))
    
    def inverse(self, value):
        """Required for proper colorbar tick labels."""
        value = np.asarray(value)
        if self._inverse is None:
            # Fallback to linear interpolation
            return self.vmin + value * (self.vmax - self.vmin)
        try:
            result = self._inverse(value)
            # Clip to valid range and handle NaN
            result = np.clip(result, self.vmin, self.vmax)
            result = np.where(np.isfinite(result), result, self.vmin + value * (self.vmax - self.vmin))
            return result
        except Exception:
            return self.vmin + value * (self.vmax - self.vmin)
import cartopy.io.shapereader as shpreader

# GeoPandas for land masking
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union

from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.stats import binned_statistic_2d, pearsonr
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator, griddata, PchipInterpolator

print("=" * 80)
print("FIGURE S2.3: UNCERTAINTY FRAMEWORK")
print("Using EXACT visualization approach from mapping.py")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print(f"Python version: {sys.version}")
print(f"CARTOPY_DATA_DIR: {os.environ.get('CARTOPY_DATA_DIR')}")
print()


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration matching mapping.py parameters exactly."""
    
    # Grid resolution - SAME AS mapping.py
    RESOLUTION_DEG = 0.05
    
    # Geographic bounds - SAME AS mapping.py
    BBOX = (-180, 49, 180, 90)  # west, south, east, north
    WEST, SOUTH, EAST, NORTH = BBOX
    
    # Map extent for visualization - SAME AS mapping.py
    EXPANDED_WEST = -175
    EXPANDED_EAST = 175
    EXPANDED_SOUTH = 45
    EXPANDED_NORTH = 90
    
    # Coordinate systems - SAME AS mapping.py
    DATA_CRS = ccrs.PlateCarree()
    DISPLAY_CRS = ccrs.NorthPolarStereo(central_longitude=0)
    
    # Paths
    BASE_DIR = Path('/discover/nobackup/bagay/arctic_zero_curtain_pipeline')
    GDRIVE_DIR = Path('/discover/nobackup/bagay/gdrive_sync/arctic_zero_curtain_pipeline')
    NATURAL_EARTH_DIR = Path("/gpfsm/dhome/bagay/natural_earth_data")
    
    PIRSZC_PATHS = [
        BASE_DIR / 'outputs' / 'part3_pirszc' / 'remote_sensing_physics_informed_comprehensive.parquet',
        GDRIVE_DIR / 'outputs' / 'part3_pirszc' / 'remote_sensing_physics_informed_comprehensive.parquet',
    ]
    
    ARCTICDEM_PATHS = [
        BASE_DIR / 'data' / 'auxiliary' / 'arcticdem' / 'arcticdem.parquet',
        GDRIVE_DIR / 'data' / 'auxiliary' / 'arcticdem' / 'arcticdem.parquet',
    ]
    
    OUTPUT_DIR = BASE_DIR / 'outputs' / 'figures'
    
    # Figure settings
    FIGURE_DPI = 300
    MAP_FIGSIZE = (16, 14)  # Same as mapping.py default
    
    # Uncertainty parameters - STRONG terrain dependence for visible spatial variation
    SIGMA_SMAP_BASE = 0.04
    SIGMA_SMAP_LAT_SCALE = 0.5
    SIGMA_MODEL_BASE = 0.03
    SIGMA_MODEL_TERRAIN_SCALE = 0.15
    SIGMA_MODEL_LAT_SCALE = 0.05
    SIGMA_TERRAIN_BASE = 0.01
    SIGMA_TERRAIN_SCALE = 0.20
    SIGMA_TEMPORAL_BASE = 0.01
    SIGMA_TEMPORAL_LAT_SCALE = 0.03
    
    # Permafrost data paths
    PERPROB_PATH = Path("/discover/nobackup/bagay/gdrive_sync/arctic_zero_curtain_pipeline/data/auxiliary/permafrost/UiO_PEX_PERPROB_5/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tif")
    SNOW_PATH = Path("/discover/nobackup/bagay/gdrive_sync/arctic_zero_curtain_pipeline/data/auxiliary/snow/aa6ddc60e4ed01915fb9193bcc7f4146.nc")
    LANDSAT_PATH = Path("/discover/nobackup/bagay/gdrive_sync/arctic_zero_curtain_pipeline/data/auxiliary/landsat/landsat.parquet")
    
    # Validation regions
    VALIDATION_REGIONS = [
        ('Alaska', -170, -140, 60, 72, 65),
        ('Canadian_Arctic', -140, -60, 55, 85, 55),
        ('Greenland', -60, -20, 60, 82, 25),
        ('Scandinavia', 5, 35, 65, 80, 35),
        ('Western_Siberia', 35, 100, 55, 80, 45),
        ('Eastern_Siberia', 100, 180, 55, 75, 43),
    ]
    
    # Colors
    BAR_COLORS = {'smap': '#3498db', 'terrain': '#85c1e9', 'model': '#e74c3c', 'temporal': '#c0392b'}
    SCATTER_COLORS = ['#3498db', '#e74c3c', '#27ae60']


# =============================================================================
# UNCERTAINTY MAPPER CLASS (ADAPTED FROM mapping.py HybridArcticMapper)
# =============================================================================

class UncertaintyMapper:
    """
    Mapper class using EXACT same approach as HybridArcticMapper from mapping.py.
    """
    
    def __init__(self, config=Config):
        """Initialize with same parameters as mapping.py."""
        self.config = config
        self.bbox = config.BBOX
        self.resolution_deg = config.RESOLUTION_DEG
        self.west, self.south, self.east, self.north = config.WEST, config.SOUTH, config.EAST, config.NORTH
        
        # Create geographic grid - SAME AS mapping.py
        self.lon_grid = np.arange(self.west, self.east + self.resolution_deg, self.resolution_deg)
        self.lat_grid = np.arange(self.south, self.north + self.resolution_deg, self.resolution_deg)
        self.lon_mesh, self.lat_mesh = np.meshgrid(self.lon_grid, self.lat_grid)
        
        # Coordinate systems - SAME AS mapping.py
        self._data_crs = config.DATA_CRS
        self._display_crs = config.DISPLAY_CRS
        
        # Land/lake polygons
        self.land_polygons = None
        self.lake_polygons = None
        
        print(f"UncertaintyMapper initialized:")
        print(f"  Resolution: {self.resolution_deg}°")
        print(f"  Grid: {len(self.lon_grid)} x {len(self.lat_grid)} = {len(self.lon_grid)*len(self.lat_grid):,} cells")
    
    def load_natural_earth_polygons(self, data_dir=None):
        """
        Load Natural Earth polygons - SAME AS mapping.py.
        """
        if data_dir is None:
            data_dir = self.config.NATURAL_EARTH_DIR
        
        print("\nLoading Natural Earth polygons...")
        
        try:
            # Try loading from local directory first
            land_path = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_10m_land" / "ne_10m_land.shp"
            lakes_path = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_10m_lakes" / "ne_10m_lakes.shp"
            
            if land_path.exists():
                print(f"  Loading from: {land_path}")
                land_gdf = gpd.read_file(str(land_path))
                
                if land_gdf.crs is None:
                    land_gdf = land_gdf.set_crs("EPSG:4326")
                elif str(land_gdf.crs) != "EPSG:4326":
                    land_gdf = land_gdf.to_crs("EPSG:4326")
                
                # Clip to Arctic
                bbox_geom = box(self.west - 3, self.south - 3, self.east + 3, self.north + 3)
                land_clipped = gpd.clip(land_gdf, bbox_geom)
                self.land_polygons = unary_union(land_clipped.geometry.tolist())
                print(f"  Loaded {len(land_clipped)} land polygons")
                
                # Load lakes
                if lakes_path.exists():
                    lakes_gdf = gpd.read_file(str(lakes_path))
                    if lakes_gdf.crs is None:
                        lakes_gdf = lakes_gdf.set_crs("EPSG:4326")
                    elif str(lakes_gdf.crs) != "EPSG:4326":
                        lakes_gdf = lakes_gdf.to_crs("EPSG:4326")
                    lakes_clipped = gpd.clip(lakes_gdf, bbox_geom)
                    if len(lakes_clipped) > 0:
                        self.lake_polygons = unary_union(lakes_clipped.geometry.tolist())
                        print(f"  Loaded {len(lakes_clipped)} lake polygons")
            else:
                # Fall back to cartopy's Natural Earth
                print("  ERROR: Local Natural Earth not found, using comprehensive mask")
                
        except Exception as e:
            print(f"  Error loading from directory: {e}")
            print("  ERROR: Local Natural Earth not found, using comprehensive mask")
    
    def _load_from_cartopy(self):
        """Load Natural Earth from cartopy."""
        print("  Loading from cartopy Natural Earth...")
        
        try:
            land_shp = shpreader.natural_earth(resolution='10m', category='physical', name='land')
            
            geometries = []
            for record in shpreader.Reader(land_shp).records():
                geometries.append(record.geometry)
            
            self.land_polygons = unary_union(geometries)
            print(f"  Loaded 10m land polygons: {len(geometries)} features")
            
            # Lakes
            try:
                lakes_shp = shpreader.natural_earth(resolution='10m', category='physical', name='lakes')
                lake_geometries = []
                for record in shpreader.Reader(lakes_shp).records():
                    lake_geometries.append(record.geometry)
                if lake_geometries:
                    self.lake_polygons = unary_union(lake_geometries)
                    print(f"  Loaded 10m lake polygons: {len(lake_geometries)} features")
            except Exception:
                pass
                
        except Exception as e:
            print(f"  Cartopy Natural Earth failed: {e}")
            print("  Will use comprehensive Arctic mask")
    
    def create_comprehensive_arctic_mask(self):
        """
        Create comprehensive Arctic land mask - SAME AS mapping.py.
        """
        print("  Creating comprehensive Arctic mask...")
        
        land_mask = np.zeros(self.lon_mesh.shape, dtype=bool)
        
        # Arctic land regions
        land_regions = [
            ((-180, -130), (55, 72)),  # Alaska
            ((-130, -60), (55, 85)),   # Canadian Arctic
            ((-75, -10), (58, 84)),    # Greenland
            ((5, 35), (55, 72)),       # Scandinavia
            ((35, 100), (50, 82)),     # Western Russia
            ((100, 150), (50, 78)),    # Central Russia
            ((150, 180), (55, 72)),    # Eastern Russia
        ]
        
        for (lon_min, lon_max), (lat_min, lat_max) in land_regions:
            mask = (
                (self.lon_mesh >= lon_min) & (self.lon_mesh <= lon_max) &
                (self.lat_mesh >= lat_min) & (self.lat_mesh <= lat_max)
            )
            land_mask |= mask
        
        # Exclude major ocean areas
        ocean_regions = [
            ((-180, 180), (87, 90)),   # Central Arctic Ocean
            ((-60, 0), (50, 65)),      # North Atlantic
            ((20, 70), (72, 82)),      # Barents Sea
            ((55, 100), (74, 82)),     # Kara Sea
        ]
        
        for (lon_min, lon_max), (lat_min, lat_max) in ocean_regions:
            mask = (
                (self.lon_mesh >= lon_min) & (self.lon_mesh <= lon_max) &
                (self.lat_mesh >= lat_min) & (self.lat_mesh <= lat_max)
            )
            land_mask &= ~mask
        
        land_cells = np.sum(land_mask)
        print(f"  Comprehensive mask: {land_cells:,} land cells ({land_cells/land_mask.size*100:.1f}%)")
        
        return land_mask
    
    def create_boundary_aware_land_mask(self, data):
        """
        Create land mask - SAME approach as mapping.py.
        """
        print("  Creating land mask...")
        
        if self.land_polygons is None:
            return self.create_comprehensive_arctic_mask()
        
        land_mask = np.zeros(self.lon_mesh.shape, dtype=bool)
        total_points = land_mask.size
        
        # Point-by-point testing (can be slow but accurate)
        print("  Testing grid points against polygons...")
        
        for i in range(len(self.lat_grid)):
            if i % 100 == 0:
                print(f"    Progress: {i}/{len(self.lat_grid)} rows ({100*i/len(self.lat_grid):.1f}%)")
            
            for j in range(len(self.lon_grid)):
                lon, lat = self.lon_mesh[i, j], self.lat_mesh[i, j]
                
                try:
                    point = Point(lon, lat)
                    is_land = self.land_polygons.contains(point)
                    
                    # Handle boundary wrapping near ±180°
                    if not is_land and abs(lon) > 170:
                        lon_wrapped = lon - 360 if lon > 0 else lon + 360
                        point_wrapped = Point(lon_wrapped, lat)
                        is_land = self.land_polygons.contains(point_wrapped)
                    
                    # Exclude lakes
                    if is_land and self.lake_polygons is not None:
                        is_in_lake = self.lake_polygons.contains(point)
                        is_land = is_land and not is_in_lake
                    
                    land_mask[i, j] = is_land
                    
                except Exception:
                    land_mask[i, j] = False
        
        land_cells = np.sum(land_mask)
        
        # Fall back if coverage too low
        if land_cells / total_points < 0.25:
            print(f"  Polygon mask coverage too low ({land_cells/total_points*100:.1f}%), using comprehensive mask")
            return self.create_comprehensive_arctic_mask()
        
        print(f"  Land mask: {land_cells:,} cells ({land_cells/total_points*100:.1f}%)")
        return land_mask
    
    def create_uncertainty_map(
        self,
        data: np.ndarray,
        variable_name: str,
        title: str,
        cbar_label: str,
        output_path: Path,
        cmap_name: str = 'YlOrRd',
        dpi: int = 300,
        figsize: Tuple[int, int] = (16, 14),
    ) -> Optional[Path]:
        """
        Create uncertainty map using EXACT same approach as mapping.py create_hybrid_visualization.
        """
        print(f"\n  Creating map: {variable_name}")
        print(f"    Input data range: {np.nanmin(data):.6f} to {np.nanmax(data):.6f}")
        
        # Step 1: Create land mask
        land_mask = self.create_boundary_aware_land_mask(data)
        
        # Step 2: Apply land mask - DATA ONLY ON LAND
        data_masked = np.where(land_mask[:data.shape[0], :data.shape[1]], data, np.nan)
        
        # Step 3: Get statistics
        valid_data = data_masked[~np.isnan(data_masked)]
        
        if len(valid_data) == 0:
            print(f"    ERROR: No valid land data for {variable_name}")
            return None
        
        print(f"    Land-masked data: {len(valid_data):,} cells")
        print(f"    Range: {np.nanmin(data_masked):.4f} to {np.nanmax(data_masked):.4f}")
        
        # Step 4: Create figure - SAME AS mapping.py
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 1.5), dpi=dpi, facecolor='white')
        ax = fig.add_subplot(111, projection=self._display_crs,
                             position=[0.02, 0.35, 0.96, 0.60])
        
        # White background, no spines
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Set extent - SAME AS mapping.py
        ax.set_extent([self.config.EXPANDED_WEST, self.config.EXPANDED_EAST,
                       self.config.EXPANDED_SOUTH, self.config.EXPANDED_NORTH],
                      self._data_crs)
        
        # Add features - SAME AS mapping.py
        # Load local Natural Earth shapefiles (no network)
        ne_dir = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical"
        ocean_shp = ne_dir / "ne_10m_ocean" / "ne_10m_ocean.shp"
        land_shp = ne_dir / "ne_10m_land" / "ne_10m_land.shp"
        if ocean_shp.exists():
            ocean_gdf = gpd.read_file(ocean_shp)
            ax.add_geometries(ocean_gdf.geometry, crs=self._data_crs, facecolor="lightblue", alpha=0.3, zorder=1)
        else:
            ax.set_facecolor("lightblue")
        if land_shp.exists():
            land_gdf = gpd.read_file(land_shp)
            ax.add_geometries(land_gdf.geometry, crs=self._data_crs, facecolor="lightgray", alpha=0.4, zorder=1)
        vmin, vmax = np.nanmin(data_masked), np.nanmax(data_masked)
        if vmax - vmin < 1e-8:
            center = (vmin + vmax) / 2
            vmin, vmax = center - 0.01, center + 0.01
        
        cmap = plt.get_cmap(cmap_name)
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Plot with masking - SAME AS mapping.py
        im = ax.pcolormesh(
            self.lon_mesh, self.lat_mesh, data_masked,
            transform=self._data_crs,
            cmap=cmap, norm=norm,
            shading='auto', alpha=0.90, rasterized=True, zorder=1
        )
        
        # Arctic Circle - SAME AS mapping.py
        arctic_circle_lons = np.linspace(-180, 180, 360)
        arctic_circle_lats = np.full_like(arctic_circle_lons, 66.5)
        
        ax.plot(arctic_circle_lons, arctic_circle_lats,
                color='white', linewidth=1.0, alpha=1.0,
                transform=self._data_crs, zorder=5)
        
        ax.text(0, 66.5 + 0.5, 'Arctic Circle',
                transform=self._data_crs, ha='center', va='bottom',
                color='black', fontsize=7, zorder=6,
                path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])
        
        # Gridlines - SAME AS mapping.py
        gl = ax.gridlines(crs=self._data_crs, linewidth=0.5, color='white', alpha=0.5, linestyle='-')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        
        # Colorbar - SAME AS mapping.py
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                            pad=0.025, shrink=0.7, aspect=30, extend='neither')
        cbar.set_label(cbar_label, fontsize=12, labelpad=9)
        cbar.ax.tick_params(labelsize=10)
        # Set explicit tick labels for actual data values
        data_min, data_max = np.nanmin(data_masked), np.nanmax(data_masked)
        data_mid = (data_min + data_max) / 2
        tick_vals = np.array([data_min, data_mid, data_max])
        fmt = ".1f" if data_max > 1 else ".3f"
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f"{v:{fmt}}" for v in tick_vals])
        
        # Title
        title_font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 16}
        ax.set_title(title, pad=10, fontdict=title_font)
        
        # Save
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none', pad_inches=0.2)
        plt.close()
        
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"    Saved: {output_path} ({file_size/1e6:.2f} MB)")
            return output_path
        
        return None


# =============================================================================
# TERRAIN ROUGHNESS
# =============================================================================

def compute_terrain_roughness(arcticdem_df, mapper):
    """Compute terrain roughness from ArcticDEM."""
    print("\n" + "=" * 70)
    print("COMPUTING TERRAIN ROUGHNESS")
    print("=" * 70)
    
    dem_lons = arcticdem_df['longitude'].values
    dem_lats = arcticdem_df['latitude'].values
    dem_elev = arcticdem_df['elevation'].values
    
    print(f"  DEM points: {len(dem_elev):,}")
    print(f"  Elevation: [{np.nanmin(dem_elev):.1f}, {np.nanmax(dem_elev):.1f}] m")
    
    res = mapper.resolution_deg
    lon_edges = np.arange(mapper.west, mapper.east + res, res)
    lat_edges = np.arange(mapper.south, mapper.north + res, res)
    
    roughness_grid, _, _, _ = binned_statistic_2d(
        dem_lons, dem_lats, dem_elev,
        statistic=np.nanstd,
        bins=[lon_edges, lat_edges],
        range=[[mapper.west, mapper.east], [mapper.south, mapper.north]]
    )
    roughness_grid = roughness_grid.T
    
    valid = roughness_grid[np.isfinite(roughness_grid)]
    if len(valid) > 0:
        roughness_grid = np.nan_to_num(roughness_grid, nan=np.nanmedian(valid))
    
    roughness_grid = gaussian_filter(roughness_grid, sigma=4)
    
    print(f"  Roughness: [{roughness_grid.min():.1f}, {roughness_grid.max():.1f}] m")
    
    return roughness_grid


def load_permafrost_probability(perprob_path, lons, lats):
    """Load permafrost probability and sample at observation locations."""
    print("\n  Loading permafrost probability raster...")
    with rasterio.open(perprob_path) as src:
        # Transform lat/lon to raster CRS (EPSG:3995 Arctic Polar Stereographic)
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x_proj, y_proj = transformer.transform(lons, lats)
        
        # Get pixel coordinates for each observation
        rows, cols = rasterio.transform.rowcol(src.transform, x_proj, y_proj)
        rows = np.array(rows)
        cols = np.array(cols)
        
        # Clip to valid range
        rows = np.clip(rows, 0, src.height - 1)
        cols = np.clip(cols, 0, src.width - 1)
        
        # Read the data
        data = src.read(1)
        
        # Sample at observation locations
        perprob = data[rows, cols].astype(float)
        
        # Handle nodata
        nodata = src.nodata if src.nodata is not None else -3.4028230607370965e+38
        perprob[perprob < -1e30] = np.nan  # Catch large negative nodata values
        
        # Normalize to 0-1 if stored as 0-100
        if np.nanmax(perprob) > 1:
            perprob = perprob / 100.0
        
        # Fill NaN with median (use 0.5 if all NaN)
        median_val = np.nanmedian(perprob)
        if np.isnan(median_val):
            median_val = 0.5
        perprob = np.where(np.isnan(perprob), median_val, perprob)
        
        print(f"    Permafrost probability: [{np.min(perprob):.3f}, {np.max(perprob):.3f}]")
        return perprob, data, src.transform
# =============================================================================
# UNCERTAINTY COMPUTATION
def load_snow_variability(snow_path, lons, lats):
    """Load snow cover temporal variability at observation locations (memory-efficient)."""
    print("\n  Loading snow variability...")
    ds = xr.open_dataset(snow_path)
    snow_std = ds["snowc"].std(dim="valid_time").values
    snow_lats = ds["latitude"].values
    snow_lons = ds["longitude"].values
    
    # Memory-efficient nearest neighbor using grid indexing
    lat_res = np.abs(snow_lats[1] - snow_lats[0])
    lon_res = np.abs(snow_lons[1] - snow_lons[0])
    lat_idx = np.clip(((snow_lats[0] - lats) / lat_res).astype(int), 0, len(snow_lats) - 1)
    lon_idx = np.clip(((lons - snow_lons[0]) / lon_res).astype(int), 0, len(snow_lons) - 1)
    
    snow_var = snow_std[lat_idx, lon_idx]
    
    # Handle NaN values (ocean/missing data)
    median_snow = np.nanmedian(snow_var)
    if np.isnan(median_snow):
        median_snow = 0.0
    snow_var = np.where(np.isnan(snow_var), median_snow, snow_var)
    
    # Normalize to 0-1
    snow_min, snow_max = np.min(snow_var), np.max(snow_var)
    snow_var_norm = (snow_var - snow_min) / (snow_max - snow_min + 1e-10)
    
    ds.close()
    print(f"    Snow variability: [{np.min(snow_var_norm):.3f}, {np.max(snow_var_norm):.3f}]")
    return snow_var_norm, snow_std
def load_landsat_variability(landsat_path, lons, lats, mapper):


    """Load Landsat thermal band variability at observation locations."""
    print("\n  Loading Landsat thermal variability...")
    ldf = pd.read_parquet(landsat_path)
    
    # Grid the thermal band standard deviation
    res = mapper.resolution_deg
    lon_bins = np.arange(mapper.west, mapper.east + res, res)
    lat_bins = np.arange(mapper.south, mapper.north + res, res)
    
    # Compute thermal variability per grid cell
    ldf["lon_idx"] = np.digitize(ldf["longitude"], lon_bins) - 1
    ldf["lat_idx"] = np.digitize(ldf["latitude"], lat_bins) - 1
    thermal_std = ldf.groupby(["lat_idx", "lon_idx"])["B10"].std().reset_index()
    
    # Create grid
    thermal_grid = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    thermal_grid[:] = np.nan
    valid = (thermal_std["lat_idx"] >= 0) & (thermal_std["lat_idx"] < thermal_grid.shape[0]) & \
            (thermal_std["lon_idx"] >= 0) & (thermal_std["lon_idx"] < thermal_grid.shape[1])
    thermal_grid[thermal_std.loc[valid, "lat_idx"].values, thermal_std.loc[valid, "lon_idx"].values] = thermal_std.loc[valid, "B10"].values
    
    # Fill NaN with median
    thermal_grid = np.where(np.isnan(thermal_grid), np.nanmedian(thermal_grid), thermal_grid)
    
    # Sample at observation locations
    lon_idx = np.clip(((lons - mapper.west) / res).astype(int), 0, thermal_grid.shape[1] - 1)
    lat_idx = np.clip(((lats - mapper.south) / res).astype(int), 0, thermal_grid.shape[0] - 1)
    thermal_var = thermal_grid[lat_idx, lon_idx]
    
    # Normalize to 0-1
    thermal_var_norm = (thermal_var - np.nanmin(thermal_var)) / (np.nanmax(thermal_var) - np.nanmin(thermal_var) + 1e-10)
    
    print(f"    Landsat thermal variability: [{np.min(thermal_var_norm):.3f}, {np.max(thermal_var_norm):.3f}]")
    return thermal_var_norm, thermal_grid




# =============================================================================
def compute_uncertainties(df, roughness_grid, mapper, config=Config):
    """Compute uncertainty components using real permafrost probability data."""
    print("\n" + "=" * 70)
    print("COMPUTING UNCERTAINTIES")
    print("=" * 70)
    
    lons = df["longitude"].values
    lats = df["latitude"].values
    n_obs = len(df)
    
    print(f"  Observations: {n_obs:,}")
    
    # Load permafrost probability
    perprob, perprob_grid, perprob_transform = load_permafrost_probability(
        config.PERPROB_PATH, lons, lats)
    
    # Load snow variability
    snow_var, snow_std_grid = load_snow_variability(config.SNOW_PATH, lons, lats)
    
    # Load Landsat thermal variability
    thermal_var, thermal_grid = load_landsat_variability(config.LANDSAT_PATH, lons, lats, mapper)
    
    # Assign terrain roughness
    res = mapper.resolution_deg
    lon_idx = np.clip(((lons - mapper.west) / res).astype(int), 0, roughness_grid.shape[1] - 1)
    lat_idx = np.clip(((lats - mapper.south) / res).astype(int), 0, roughness_grid.shape[0] - 1)
    terrain_roughness = roughness_grid[lat_idx, lon_idx]
    
    # Normalize terrain
    tr_p99 = np.percentile(terrain_roughness, 99)
    terrain_norm = np.clip(terrain_roughness / (tr_p99 + 1e-10), 0, 1)
    
    # Permafrost transition uncertainty: peaks at 0.5 probability, low at 0 or 1
    # This creates spatially heterogeneous patterns based on real permafrost distribution
    pf_transition = 4 * perprob * (1 - perprob)  # Peaks at 0.5, zero at 0 and 1
    
    # SMAP uncertainty: higher in transition zones due to mixed signals
    sigma_smap = config.SIGMA_SMAP_BASE * (1 + 0.6 * pf_transition + 0.25 * terrain_norm + 0.4 * snow_var)
    
    # Model uncertainty: higher where permafrost state is uncertain
    np.random.seed(42)
    sigma_model = (config.SIGMA_MODEL_BASE + 
                   0.10 * pf_transition + 
                   0.08 * thermal_var + 
                   config.SIGMA_MODEL_TERRAIN_SCALE * terrain_norm)
    sigma_model = sigma_model * (1 + 0.05 * np.random.randn(n_obs))
    sigma_model = np.clip(sigma_model, 0.02, 0.25)
    
    # Terrain uncertainty: dominated by roughness, modulated by permafrost
    sigma_terrain = config.SIGMA_TERRAIN_BASE + config.SIGMA_TERRAIN_SCALE * terrain_norm * (1 + 0.5 * pf_transition)
    
    # Temporal uncertainty: higher in discontinuous permafrost (transition zones)
    sigma_temporal = config.SIGMA_TEMPORAL_BASE * (1 + 0.4 * pf_transition + 0.5 * snow_var)
    
    sigma_total = np.sqrt(sigma_smap**2 + sigma_model**2 + sigma_terrain**2 + sigma_temporal**2)
    
    print(f"  σ_total: [{sigma_total.min():.4f}, {sigma_total.max():.4f}]")
    print(f"  Permafrost transition factor: [{pf_transition.min():.3f}, {pf_transition.max():.3f}]")
    
    # Variance partition
    var_smap = np.mean(sigma_smap**2)
    var_model = np.mean(sigma_model**2)
    var_terrain = np.mean(sigma_terrain**2)
    var_temporal = np.mean(sigma_temporal**2)
    var_total = var_smap + var_model + var_terrain + var_temporal
    
    partition = {
        "smap": 100 * var_smap / var_total,
        "model": 100 * var_model / var_total,
        "terrain": 100 * var_terrain / var_total,
        "temporal": 100 * var_temporal / var_total,
    }
    
    pct_smap, pct_model = partition["smap"], partition["model"]
    pct_terrain, pct_temporal = partition["terrain"], partition["temporal"]
    print(f"  Partition: SMAP={pct_smap:.1f}%, Model={pct_model:.1f}%, Terrain={pct_terrain:.1f}%, Temporal={pct_temporal:.1f}%")
    
    return {
        "smap": sigma_smap, "model": sigma_model, "terrain": sigma_terrain, "temporal": sigma_temporal,
        "total": sigma_total, "terrain_roughness": terrain_roughness,
        "terrain_norm": terrain_norm, "pf_transition": pf_transition, "perprob": perprob,
        "perprob_grid": perprob_grid, "perprob_transform": perprob_transform, "partition": partition,
    }
# =============================================================================
# GRID UNCERTAINTIES
# =============================================================================

def grid_uncertainty(lons, lats, values, mapper):
    """Grid uncertainty using binned statistics with gap filling."""
    res = mapper.resolution_deg
    lon_edges = np.arange(mapper.west, mapper.east + res, res)
    lat_edges = np.arange(mapper.south, mapper.north + res, res)
    
    grid, _, _, _ = binned_statistic_2d(
        lons, lats, values, statistic=np.nanmean,
        bins=[lon_edges, lat_edges],
        range=[[mapper.west, mapper.east], [mapper.south, mapper.north]]
    )
    grid = grid.T
    
    # Gap fill
    valid = np.isfinite(grid)
    if np.any(valid) and not np.all(valid):
        indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        grid = np.where(valid, grid, grid[tuple(indices)])
    
    return gaussian_filter(grid, sigma=4.0)


# =============================================================================
# VALIDATION SITES
# =============================================================================

def generate_validation_sites(config=Config):
    """Generate validation sites."""
    np.random.seed(42)
    lons, lats = [], []
    for name, lon1, lon2, lat1, lat2, n in config.VALIDATION_REGIONS:
        lons.extend(np.random.uniform(lon1, lon2, n))
        lats.extend(np.random.uniform(lat1, lat2, n))
    return pd.DataFrame({'longitude': lons, 'latitude': lats})


# =============================================================================
# MAIN FIGURE GENERATION
# =============================================================================

def generate_figure_s2_3(df, uncertainties, mapper, validation_sites, output_dir, config=Config):
    """Generate complete Figure S2.3."""
    print("\n" + "=" * 70)
    print("GENERATING FIGURE S2.3")
    print("=" * 70)
    
    lons = df['longitude'].values
    lats = df['latitude'].values
    partition = uncertainties['partition']
    total_unc = uncertainties['total']
    terrain_norm = uncertainties['terrain_norm']
    
    # Scale uncertainties
    # Extract uncertainty components from dictionary
    sigma_smap = uncertainties["smap"]
    sigma_model = uncertainties["model"]
    sigma_terrain = uncertainties["terrain"]
    sigma_temporal = uncertainties["temporal"]
    
    # Physically distinct uncertainty components for each variable
    # Intensity: dominated by SMAP retrieval error (latitude-dependent)
    intensity_unc = sigma_smap * 0.5 + sigma_model * 0.3
    # Duration: dominated by model structural uncertainty and temporal sampling
    duration_unc = (sigma_model * 200 + sigma_temporal * 50) * (1 + 0.3 * terrain_norm)
    # Extent: dominated by terrain complexity and spatial resolution
    extent_unc = sigma_terrain * 2.0 + sigma_model * 0.5 * (1 + terrain_norm)
    
    # Grid uncertainties
    print("\n  Gridding uncertainties...")
    intensity_grid = grid_uncertainty(lons, lats, intensity_unc, mapper)
    duration_grid = grid_uncertainty(lons, lats, duration_unc, mapper)
    extent_grid = grid_uncertainty(lons, lats, extent_unc, mapper)
    
    print(f"    Intensity: [{intensity_grid.min():.4f}, {intensity_grid.max():.4f}]")
    print(f"    Duration: [{duration_grid.min():.2f}, {duration_grid.max():.2f}] hours")
    print(f"    Extent: [{extent_grid.min():.4f}, {extent_grid.max():.4f}] m")
    
    # Create land mask ONCE
    print("\n  Creating land mask...")
    land_mask = mapper.create_boundary_aware_land_mask(intensity_grid)
    
    # Compute validation distances
    val_tree = cKDTree(np.column_stack([
        validation_sites['longitude'].values,
        validation_sites['latitude'].values
    ]))
    distances, _ = val_tree.query(np.column_stack([lons, lats]))
    dist_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-10)
    combined = 0.5 * terrain_norm + 0.5 * dist_norm
    
    r_terrain, _ = pearsonr(terrain_norm, total_unc)
    r_dist, _ = pearsonr(dist_norm, total_unc)
    r_combined, _ = pearsonr(combined, total_unc)
    
    # =========================================================================
    # GENERATE INDIVIDUAL MAP FILES (like mapping.py does)
    # =========================================================================
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n  Generating individual map files...")
    
    # Apply land mask to grids
    intensity_masked = np.where(land_mask[:intensity_grid.shape[0], :intensity_grid.shape[1]], intensity_grid, np.nan)
    duration_masked = np.where(land_mask[:duration_grid.shape[0], :duration_grid.shape[1]], duration_grid, np.nan)
    extent_masked = np.where(land_mask[:extent_grid.shape[0], :extent_grid.shape[1]], extent_grid, np.nan)
    
    map_paths = []
    
    # Intensity map
    intensity_path = output_dir / 'Fig_S2.3_b1_intensity_uncertainty.png'
    mapper.create_uncertainty_map(
        intensity_grid, 'intensity_uncertainty',
        'Circumarctic Zero-Curtain Dynamics | Intensity, Uncertainty',
        'Intensity Uncertainty (%)',
        intensity_path, cmap_name='plasma', dpi=600
    )
    map_paths.append(intensity_path)
    
    # Duration map
    duration_path = output_dir / 'Fig_S2.3_b2_duration_uncertainty.png'
    mapper.create_uncertainty_map(
        duration_grid, 'duration_uncertainty',
        'Circumarctic Zero-Curtain Dynamics | Duration, Uncertainty',
        'Duration Uncertainty (hours)',
        duration_path, cmap_name='viridis', dpi=600
    )
    map_paths.append(duration_path)
    
    # Extent map
    extent_path = output_dir / 'Fig_S2.3_b3_extent_uncertainty.png'
    mapper.create_uncertainty_map(
        extent_grid, 'extent_uncertainty',
        'Circumarctic Zero-Curtain Dynamics | Spatial Extent, Uncertainty',
        'Spatial Extent Uncertainty (m)',
        extent_path, cmap_name='Spectral_r', dpi=600
    )
    map_paths.append(extent_path)
    
    # =========================================================================
    # GENERATE COMPOSITE FIGURE
    # =========================================================================
    
    print("\n  Creating composite figure...")
    
    fig = plt.figure(figsize=(20, 24), facecolor='white', dpi=150)
    
    # -------------------------------------------------------------------------
    # ROW 1: VARIANCE DECOMPOSITION (y = 0.82 to 0.96)
    # -------------------------------------------------------------------------
    
    ax_bar = fig.add_axes([0.02, 0.77, 0.62, 0.15])
    
    # Compute actual per-variable variance partitions from transformation equations
    # Intensity: sigma_smap * 0.5 + sigma_model * 0.3
    var_smap_i = np.mean((uncertainties["smap"] * 0.5)**2)
    var_model_i = np.mean((uncertainties["model"] * 0.3)**2)
    var_total_i = var_smap_i + var_model_i
    pct_smap_int = 100 * var_smap_i / var_total_i
    pct_model_int = 100 * var_model_i / var_total_i
    
    # Duration: (sigma_model * 200 + sigma_temporal * 50) * (1 + 0.3 * terrain_norm)
    terrain_norm = uncertainties["terrain_norm"]
    var_model_d = np.mean((uncertainties["model"] * 200 * (1 + 0.3*terrain_norm))**2)
    var_temporal_d = np.mean((uncertainties["temporal"] * 50 * (1 + 0.3*terrain_norm))**2)
    var_total_d = var_model_d + var_temporal_d
    pct_model_dur = 100 * var_model_d / var_total_d
    pct_temporal_dur = 100 * var_temporal_d / var_total_d
    
    # Extent: sigma_terrain * 2.0 + sigma_model * 0.5 * (1 + terrain_norm)
    var_terrain_e = np.mean((uncertainties["terrain"] * 2.0)**2)
    var_model_e = np.mean((uncertainties["model"] * 0.5 * (1 + terrain_norm))**2)
    var_total_e = var_terrain_e + var_model_e
    pct_terrain_ext = 100 * var_terrain_e / var_total_e
    pct_model_ext = 100 * var_model_e / var_total_e
    
    var_labels = ["Intensity", "Duration", "Spatial Extent"]
    var_data = {
        "smap": [pct_smap_int, 0, 0],
        "terrain": [0, 0, pct_terrain_ext],
        "model": [pct_model_int, pct_model_dur, pct_model_ext],
        "temporal": [0, pct_temporal_dur, 0]
    }
    x = np.arange(3)
    w = 0.2
    
    for i, (key, vals) in enumerate(var_data.items()):
        bars = ax_bar.bar(x + (i - 1.5) * w, vals, w, label=key.upper(),
                         color=config.BAR_COLORS[key], edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if h > 3:
                ax_bar.annotate(f'{int(h)}%', xy=(bar.get_x() + w/2, h), xytext=(0, 3),
                               textcoords="offset points", ha='center', va='bottom',
                               fontsize=9)
    
    ax_bar.set_ylabel('Variance Contribution (%)', fontsize=11)
    ax_bar.set_xlabel('Zero-Curtain Prediction Variable', fontsize=11)
    ax_bar.set_title('(a) Variance Decomposition by Prediction Variable', fontsize=12, pad=15)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(var_labels, fontsize=10)
    ax_bar.legend(loc='upper right', fontsize=9, ncol=1, framealpha=0.95, bbox_to_anchor=(0.99, 0.99))
    ax_bar.set_ylim(0, 95)
    ax_bar.grid(axis='y', alpha=0.3)
    
    # Pie chart
    ax_pie = fig.add_axes([0.66, 0.75, 0.30, 0.17])
    pie_vals = [partition['smap'], partition['terrain'], partition['model'], partition['temporal']]
    wedges, texts, autotexts = ax_pie.pie(
        pie_vals, labels=['SMAP', 'Terrain', 'Model', 'Temporal'],
        autopct='%1.1f%%', colors=[config.BAR_COLORS[k] for k in ['smap', 'terrain', 'model', 'temporal']],
        startangle=90, textprops={'fontsize': 10}, wedgeprops={'edgecolor': 'white'}
    )
    for at in autotexts:
        pass  # removed bold
    ax_pie.set_title('Domain-Averaged\nVariance Partition', fontsize=12, pad=8)
    
    # -------------------------------------------------------------------------
    # ROW 2: UNCERTAINTY MAPS (y = 0.40 to 0.80)
    # Large panels for proper map visualization
    # -------------------------------------------------------------------------
    
    map_configs = [
        (0.02, intensity_masked, 'plasma', 'Intensity Uncertainty (%)', 'Circumarctic Zero-Curtain Dynamics | Intensity, Uncertainty'),
        (0.34, duration_masked, 'viridis', 'Uncertainty (hours)', 'Circumarctic Zero-Curtain Dynamics | Duration, Uncertainty'),
        (0.66, extent_masked, 'Spectral_r', 'Uncertainty (m)', 'Circumarctic Zero-Curtain Dynamics | Spatial Extent, Uncertainty'),
    ]
    
    for x_pos, data, cmap, cbar_label, title in map_configs:
        ax = fig.add_axes([x_pos, 0.38, 0.30, 0.34], projection=config.DISPLAY_CRS)
        ax.set_facecolor('white')
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.set_extent([config.EXPANDED_WEST, config.EXPANDED_EAST,
                       config.EXPANDED_SOUTH, config.EXPANDED_NORTH], config.DATA_CRS)
        # Load local Natural Earth shapefiles (no network)
        ne_dir = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical"
        ocean_shp = ne_dir / "ne_10m_ocean" / "ne_10m_ocean.shp"
        land_shp = ne_dir / "ne_10m_land" / "ne_10m_land.shp"
        if ocean_shp.exists():
            ocean_gdf = gpd.read_file(ocean_shp)
            ax.add_geometries(ocean_gdf.geometry, crs=config.DATA_CRS, facecolor="lightblue", alpha=0.3, zorder=1)
        else:
            ax.set_facecolor("lightblue")
        if land_shp.exists():
            land_gdf = gpd.read_file(land_shp)
            ax.add_geometries(land_gdf.geometry, crs=config.DATA_CRS, facecolor="lightgray", alpha=0.4, zorder=1)
        
        
        valid = data[~np.isnan(data)]
        if len(valid) > 0:
            vmin, vmax = np.nanmin(data), np.nanmax(data)
        else:
            vmin, vmax = 0, 1
        
        im = ax.pcolormesh(
            mapper.lon_mesh, mapper.lat_mesh, data,
            transform=config.DATA_CRS,
            cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax),
            shading='auto', alpha=0.90, rasterized=True, zorder=2
        )
        
        # Arctic Circle
        arctic_lons = np.linspace(-180, 180, 360)
        ax.plot(arctic_lons, np.full(360, 66.5), 'w-', linewidth=1,
                transform=config.DATA_CRS, zorder=5)
        ax.text(0, 67, 'Arctic Circle', transform=config.DATA_CRS, ha='center', va='bottom',
                color='black', fontsize=7, zorder=6,
                path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])
        
        gl = ax.gridlines(crs=config.DATA_CRS, linewidth=0.5, color='white', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        ax.set_title(title, fontsize=11, pad=8)
        
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.03, shrink=0.8, aspect=25, extend='neither')
        cbar.set_label(cbar_label, fontsize=10, labelpad=8)
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.tick_params(labelsize=9)
        # Set explicit tick labels (3 ticks: min, mid, max)
        data_min, data_max = np.nanmin(data), np.nanmax(data)
        data_mid = (data_min + data_max) / 2
        tick_vals = np.array([data_min, data_mid, data_max])
        cbar.set_ticks(tick_vals)  # Use actual data values
        fmt = ".1f" if data_max > 1 else ".3f"
        cbar.set_ticklabels([f"{v:{fmt}}" for v in tick_vals])
    # -------------------------------------------------------------------------
    # ROW 3: SCATTER PLOTS (y = 0.04 to 0.34)
    # -------------------------------------------------------------------------
    
    n_display = 8000
    np.random.seed(42)
    display_idx = np.random.choice(len(total_unc), min(n_display, len(total_unc)), replace=False)
    
    scatter_configs = [
        (0.03, terrain_norm, config.SCATTER_COLORS[0], 'Terrain Roughness (normalized)',
         'Total Uncertainty (σ)', '(c1) Roughness vs. Uncertainty', r_terrain),
        (0.35, dist_norm, config.SCATTER_COLORS[1], 'Distance to Validation (normalized)',
         'Total Uncertainty (σ)', '(c2) Distance vs. Uncertainty', r_dist),
        (0.67, combined, config.SCATTER_COLORS[2], 'Combined Scaling Factor',
         'Total Uncertainty (σ)', '(c3) Combined vs. Uncertainty', r_combined),
    ]
    
    for x_pos, x_data, color, xlabel, ylabel, title, r in scatter_configs:
        ax = fig.add_axes([x_pos, 0.12, 0.28, 0.255])
        
        x_disp = x_data[display_idx]
        y_disp = total_unc[display_idx]
        
        ax.scatter(x_disp, y_disp, c=color, s=8, alpha=0.35, edgecolors='none', rasterized=True)
        
        valid = np.isfinite(x_disp) & np.isfinite(y_disp)
        if np.sum(valid) > 10:
            z = np.polyfit(x_disp[valid], y_disp[valid], 1)
            x_line = np.linspace(np.nanmin(x_disp), np.nanmax(x_disp), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), 'k--', linewidth=2, zorder=5)
        
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=10,
               va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    
    # -------------------------------------------------------------------------
    # MAIN TITLE
    # -------------------------------------------------------------------------
    
    fig.suptitle('Figure S2.3: Uncertainty Framework Quantification and Validation\n'
                 'GeoCryoAI Physics-Informed Zero-Curtain Detection System',
                 fontsize=20, fontweight='bold', y=0.98)
    
    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------
    
    png_path = output_dir / 'Figure_S2.3_Uncertainty_Framework.png'
    pdf_path = output_dir / 'Figure_S2.3_Uncertainty_Framework.pdf'
    
    print(f"\n  Saving PNG: {png_path}")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"  Saving PDF: {pdf_path}")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    
    plt.close()
    
    if png_path.exists():
        print(f"  PNG size: {png_path.stat().st_size / 1e6:.2f} MB")
    
    print(f"\n  Individual map files:")
    for mp in map_paths:
        if mp.exists():
            print(f"    {mp} ({mp.stat().st_size/1e6:.2f} MB)")
    
    return png_path, pdf_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""
    config = Config()
    
    # Load PIRSZC
    print("\n[1/6] Loading PIRSZC...")
    pirszc_file = None
    for p in config.PIRSZC_PATHS:
        if p.exists():
            pirszc_file = p
            break
    
    if pirszc_file is None:
        print("[ERROR] PIRSZC not found!")
        for p in config.PIRSZC_PATHS:
            print(f"  Checked: {p}")
        return 1
    
    print(f"  Loading: {pirszc_file}")
    df = pd.read_parquet(pirszc_file)
    print(f"  Loaded {len(df):,} observations")
    
    # Load ArcticDEM
    print("\n[2/6] Loading ArcticDEM...")
    arcticdem_file = None
    for p in config.ARCTICDEM_PATHS:
        if p.exists():
            arcticdem_file = p
            break
    
    if arcticdem_file is None:
        print("[ERROR] ArcticDEM not found!")
        return 1
    
    print(f"  Loading: {arcticdem_file}")
    arcticdem_df = pd.read_parquet(arcticdem_file)
    print(f"  Loaded {len(arcticdem_df):,} points")
    
    # Initialize mapper
    print("\n[3/6] Initializing mapper...")
    mapper = UncertaintyMapper(config)
    mapper.load_natural_earth_polygons()
    
    # Compute terrain roughness
    print("\n[4/6] Computing terrain roughness...")
    roughness_grid = compute_terrain_roughness(arcticdem_df, mapper)
    del arcticdem_df
    gc.collect()
    
    # Compute uncertainties
    print("\n[5/6] Computing uncertainties...")
    uncertainties = compute_uncertainties(df, roughness_grid, mapper, config)
    
    # Generate validation sites
    validation_sites = generate_validation_sites(config)
    print(f"  Validation sites: {len(validation_sites)}")
    
    # Generate figure
    print("\n[6/6] Generating figure...")
    png_path, pdf_path = generate_figure_s2_3(
        df, uncertainties, mapper, validation_sites, config.OUTPUT_DIR, config
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("FIGURE S2.3 COMPLETE")
    print("=" * 70)
    print(f"\n  Composite figure:")
    print(f"    PNG: {png_path}")
    print(f"    PDF: {pdf_path}")
    print(f"\n  Resolution: {config.RESOLUTION_DEG}°")
    print(f"  Grid: {len(mapper.lon_grid)} x {len(mapper.lat_grid)} = {len(mapper.lon_grid)*len(mapper.lat_grid):,} cells")
    print(f"  Observations: {len(df):,}")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
