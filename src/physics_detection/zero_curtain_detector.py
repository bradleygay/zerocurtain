#!/usr/bin/env python3

import os, sys
import numpy as np
import pandas as pd
import dask.dataframe as dd
from scipy import optimize, sparse, interpolate
from scipy.sparse.linalg import spsolve
import xarray as xr
import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
from scipy.ndimage import map_coordinates
import warnings

import signal
import gc

from typing import Optional

# Global variables for graceful shutdown and memory optimization
all_events_global = []
processed_sites_global = 0

def signal_handler(sig, frame):
    """Handle Ctrl+C or system shutdown gracefully with ALL features"""
    print(f'\n💾 EMERGENCY SHUTDOWN DETECTED - Saving {len(all_events_global)} events with ALL features...')
    
    try:
        if all_events_global:
            emergency_df = pd.DataFrame(all_events_global)
            
            # Add derived classifications
            emergency_df['intensity_category'] = pd.cut(
                emergency_df['intensity_percentile'],
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['weak', 'moderate', 'strong', 'extreme']
            )
            
            emergency_df['duration_category'] = pd.cut(
                emergency_df['duration_hours'],
                bins=[0, 72, 168, 336, np.inf],
                labels=['short', 'medium', 'long', 'extended']
            )
            
            emergency_df['extent_category'] = pd.cut(
                emergency_df['spatial_extent_meters'],
                bins=[0, 0.3, 0.8, 1.5, np.inf],
                labels=['shallow', 'moderate', 'deep', 'very_deep']
            )
            
            emergency_file = f"/Users/bagay/Downloads/zero_curtain_EMERGENCY_SHUTDOWN_{processed_sites_global}sites.parquet"
            emergency_df.to_parquet(emergency_file, index=False, compression='snappy')
            
            # Verify three main features
            main_features = ['intensity_percentile', 'duration_hours', 'spatial_extent_meters']
            verified = all(f in emergency_df.columns for f in main_features)
            
            print(f"✅ Emergency shutdown save complete: {emergency_file}")
            print(f"✅ Three main features saved: {verified}")
            print(f"✅ Total features saved: {len(emergency_df.columns)}")
        else:
            print("No events to save")
    except Exception as e:
        print(f"⚠️  Emergency save failed: {e}")
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class PhysicsInformedZeroCurtainDetector:
    """
    Comprehensive zero-curtain detection using full thermodynamic physics
    including Crank-Nicholson solver, Stefan problem, and permafrost dynamics.
    """
    
    def __init__(self, config: Optional['DetectionConfig'] = None):
        """
        Initialize physics-informed zero-curtain detector with configuration.
        
        Args:
            config: DetectionConfig object with paths and parameters
        """
        # Import configuration
        if config is None:
            from .physics_config import DetectionConfig
            config = DetectionConfig()
        
        self.config = config
        
        # Physical constants from LPJ-EOSIM permafrost.c
        self.LHEAT = 3.34E8
        self.CWATER = 4180000
        self.CICE = 1700000
        self.CORG = 3117800
        self.CMIN = 2380000
        
        self.KWATER = 0.57
        self.KICE = 2.2
        self.KORG = 0.25
        self.KMIN = 2.0
        
        # Soil physics parameters
        self.RHO_WATER = 1000
        self.G = 9.81
        self.MU_WATER = 1.0e-3
        
        # CryoGrid constants
        self.LVOL_SL = 3.34E8
        self.STEFAN_BOLTZMANN = 5.67e-8
        self.TMFW = 273.15
        
        # Numerical solver parameters
        self.DT = 86400
        self.DZ_MIN = 0.01
        self.MAX_LAYERS = 50
        self.CONVERGENCE_TOL = 1e-6
        self.MAX_ENTHALPY_CHANGE = 50e3
        
        # Detection thresholds from configuration
        self.TEMP_THRESHOLD = config.physics.temp_threshold
        self.GRADIENT_THRESHOLD = config.physics.gradient_threshold
        self.MIN_DURATION_HOURS = config.physics.min_duration_hours
        self.PHASE_CHANGE_ENERGY = config.physics.phase_change_energy
        
        self.RELAXED_TEMP_THRESHOLD = config.physics.relaxed_temp_threshold
        self.RELAXED_GRADIENT_THRESHOLD = config.physics.relaxed_gradient_threshold
        self.RELAXED_MIN_DURATION = config.physics.relaxed_min_duration
        
        # CryoGrid integration flags from configuration
        self.use_cryogrid_enthalpy = config.physics.use_cryogrid_enthalpy
        self.use_painter_karra_freezing = config.physics.use_painter_karra_freezing
        self.use_surface_energy_balance = config.physics.use_surface_energy_balance
        self.use_adaptive_timestep = config.physics.use_adaptive_timestep
        
        # Load auxiliary datasets with configured paths
        self.permafrost_prob = None
        self.permafrost_zones = None
        self.snow_data = None
        
        # Initialize detector attributes
        self.snow_coord_system = None
        self.snow_lat_coord = None
        self.snow_lon_coord = None
        self.snow_alignment_score = 0.0
        
        self._load_auxiliary_data()
    
    def _load_auxiliary_data(self):
        """Load permafrost probability, zones, and snow data from configured paths."""
        try:
            # Validate paths first
            paths_valid, missing = self.config.paths.validate_paths()
            if not paths_valid:
                print(f"⚠️ Warning: Some auxiliary data files missing:")
                for missing_path in missing:
                    print(f"  - {missing_path}")
                print("Proceeding with available data...")
            
            # Load permafrost probability raster
            if self.config.paths.permafrost_prob_raster and self.config.paths.permafrost_prob_raster.exists():
                with rasterio.open(str(self.config.paths.permafrost_prob_raster)) as src:
                    self.permafrost_prob = {
                        'data': src.read(1),
                        'transform': src.transform,
                        'crs': src.crs,
                        'bounds': src.bounds
                    }
                    print(f"✓ Permafrost probability loaded: shape={self.permafrost_prob['data'].shape}, "
                        f"CRS={self.permafrost_prob['crs']}, bounds={self.permafrost_prob['bounds']}")
                    print(f"  Data range: {np.nanmin(self.permafrost_prob['data'])} to {np.nanmax(self.permafrost_prob['data'])}")
                    print(f"  NoData values: {np.sum(self.permafrost_prob['data'] < 0)} pixels")
            
            # Load permafrost zones shapefile
            if self.config.paths.permafrost_zones_shapefile and self.config.paths.permafrost_zones_shapefile.exists():
                self.permafrost_zones = gpd.read_file(str(self.config.paths.permafrost_zones_shapefile))
                print(f"✓ Permafrost zones loaded: {len(self.permafrost_zones)} features")
                print(f"  Zone types: {self.permafrost_zones['EXTENT'].value_counts().to_dict()}")
                print(f"  CRS: {self.permafrost_zones.crs}")
            
            # Load and validate snow data coordinate system
            if self.config.paths.snow_data_netcdf and self.config.paths.snow_data_netcdf.exists():
                self.snow_data = xr.open_dataset(str(self.config.paths.snow_data_netcdf))
                print(f"✓ Snow data loaded: variables={list(self.snow_data.variables.keys())}")
                
                # Comprehensive snow coordinate system validation
                self._validate_snow_coordinates()
            
            print("✓ Auxiliary datasets loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load auxiliary data: {e}")
            import traceback
            traceback.print_exc()
    
    def _validate_snow_coordinates(self):
        """Validate and analyze snow dataset coordinate system alignment."""
        
        print("\n" + "="*70)
        print("SNOW DATASET COORDINATE SYSTEM VALIDATION")
        print("="*70)
        
        # Identify coordinate variables
        coord_info = {}
        for coord_name in self.snow_data.coords:
            coord_data = self.snow_data.coords[coord_name]
            coord_info[coord_name] = {
                'shape': coord_data.shape,
                'min': float(coord_data.min()) if coord_data.size > 0 else None,
                'max': float(coord_data.max()) if coord_data.size > 0 else None,
                'dtype': coord_data.dtype
            }
        
        print("Coordinate variables found:")
        for name, info in coord_info.items():
            print(f"  {name}: shape={info['shape']}, range=[{info['min']:.3f}, {info['max']:.3f}], dtype={info['dtype']}")
        
        # Identify spatial coordinates
        lat_coords = [c for c in coord_info.keys() if 'lat' in c.lower()]
        lon_coords = [c for c in coord_info.keys() if 'lon' in c.lower()]
        
        if not lat_coords or not lon_coords:
            print("❌ ERROR: Could not identify latitude/longitude coordinates in snow data")
            return False
        
        lat_coord = lat_coords[0]
        lon_coord = lon_coords[0]
        
        lat_data = self.snow_data.coords[lat_coord].values
        lon_data = self.snow_data.coords[lon_coord].values
        
        print(f"\nPrimary spatial coordinates: {lat_coord}, {lon_coord}")
        print(f"  Latitude range: {lat_data.min():.3f} to {lat_data.max():.3f}")
        print(f"  Longitude range: {lon_data.min():.3f} to {lon_data.max():.3f}")
        
        # Determine coordinate system type
        is_geographic = self._is_geographic_coordinates(lat_data, lon_data)
        
        if is_geographic:
            print("✅ Snow data appears to use GEOGRAPHIC coordinates (WGS84/EPSG:4326)")
            self.snow_coord_system = 'geographic'
            self.snow_lat_coord = lat_coord
            self.snow_lon_coord = lon_coord
            
            # Validate geographic extent for Arctic region
            if lat_data.max() < 50.0:
                print("⚠️  WARNING: Maximum latitude < 50°N - may not cover Arctic region")
            elif lat_data.max() >= 50.0:
                print(f"✅ Arctic coverage confirmed: max latitude = {lat_data.max():.1f}°N")
            
            # Check for global vs regional coverage
            lat_span = lat_data.max() - lat_data.min()
            lon_span = lon_data.max() - lon_data.min()
            
            if lat_span > 90 and lon_span > 180:
                print("✅ Global coverage detected")
            else:
                print(f"ℹ️  Regional coverage: {lat_span:.1f}° lat × {lon_span:.1f}° lon")
        
        else:
            print("⚠️  Snow data appears to use PROJECTED coordinates")
            self.snow_coord_system = 'projected'
            self.snow_lat_coord = lat_coord
            self.snow_lon_coord = lon_coord
            
            # Try to determine projection
            if hasattr(self.snow_data, 'crs'):
                print(f"  CRS detected: {self.snow_data.crs}")
            elif hasattr(self.snow_data, 'spatial_ref'):
                print(f"  Spatial reference: {self.snow_data.spatial_ref}")
            else:
                print("  CRS information not found - will assume Arctic Polar Stereographic")
        
        # Test coordinate alignment with sample in situ coordinates
        self._test_snow_coordinate_alignment()
        
        return True
    
    def _is_geographic_coordinates(self, lat_data, lon_data):
        """Determine if coordinates are geographic (lat/lon) or projected."""
        
        # Geographic coordinates should be within reasonable lat/lon bounds
        lat_in_range = np.all(lat_data >= -90) and np.all(lat_data <= 90)
        lon_in_range = np.all(lon_data >= -180) and np.all(lon_data <= 360)
        
        # Check for typical projected coordinate magnitudes (usually much larger)
        lat_not_projected = np.all(np.abs(lat_data) < 1000)
        lon_not_projected = np.all(np.abs(lon_data) < 1000)
        
        return lat_in_range and lon_in_range and lat_not_projected and lon_not_projected
    
    def _test_snow_coordinate_alignment(self):
        """Test snow coordinate system alignment with sample in situ coordinates."""
        
        print("\nTesting coordinate alignment with in situ data...")
        
        # Sample Arctic coordinates from typical permafrost regions
        test_sites = [
            (70.0, -150.0, "Alaska North Slope"),
            (68.7, -108.0, "Canadian Arctic"),
            (71.0, 25.0, "Svalbard"),
            (64.0, -51.0, "Greenland"),
            (69.0, 88.0, "Siberia")
        ]
        
        successful_extractions = 0
        
        for lat, lon, location in test_sites:
            try:
                if self.snow_coord_system == 'geographic':
                    # Direct coordinate matching
                    lat_data = self.snow_data.coords[self.snow_lat_coord].values
                    lon_data = self.snow_data.coords[self.snow_lon_coord].values
                    
                    lat_idx = np.argmin(np.abs(lat_data - lat))
                    lon_idx = np.argmin(np.abs(lon_data - lon))
                    
                    nearest_lat = lat_data[lat_idx]
                    nearest_lon = lon_data[lon_idx]
                    
                    distance = np.sqrt((lat - nearest_lat)**2 + (lon - nearest_lon)**2)
                    
                    if distance < 5.0:  # Within 5 degrees
                        print(f"  ✅ {location}: {lat:.1f}, {lon:.1f} → {nearest_lat:.1f}, {nearest_lon:.1f} (Δ={distance:.2f}°)")
                        successful_extractions += 1
                    else:
                        print(f"  ❌ {location}: {lat:.1f}, {lon:.1f} → {nearest_lat:.1f}, {nearest_lon:.1f} (Δ={distance:.2f}° - too far)")
                
                else:
                    # Projected coordinates - attempt transformation
                    from pyproj import Transformer
                    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3995", always_xy=True)
                    x, y = transformer.transform(lon, lat)
                    
                    lat_data = self.snow_data.coords[self.snow_lat_coord].values
                    lon_data = self.snow_data.coords[self.snow_lon_coord].values
                    
                    lat_idx = np.argmin(np.abs(lat_data - y))
                    lon_idx = np.argmin(np.abs(lon_data - x))
                    
                    nearest_x = lon_data[lon_idx]
                    nearest_y = lat_data[lat_idx]
                    
                    distance = np.sqrt((x - nearest_x)**2 + (y - nearest_y)**2)
                    
                    if distance < 100000:  # Within 100km
                        print(f"  ✅ {location}: ({x:.0f}, {y:.0f}) → ({nearest_x:.0f}, {nearest_y:.0f}) (Δ={distance/1000:.1f}km)")
                        successful_extractions += 1
                    else:
                        print(f"  ❌ {location}: ({x:.0f}, {y:.0f}) → ({nearest_x:.0f}, {nearest_y:.0f}) (Δ={distance/1000:.1f}km - too far)")
                        
            except Exception as e:
                print(f"  ❌ {location}: Coordinate test failed - {e}")
        
        alignment_score = successful_extractions / len(test_sites)
        
        print(f"\nCoordinate alignment results:")
        print(f"  Successful extractions: {successful_extractions}/{len(test_sites)} ({alignment_score*100:.0f}%)")
        
        if alignment_score >= 0.8:
            print("✅ EXCELLENT coordinate alignment - snow data ready for physics integration")
        elif alignment_score >= 0.6:
            print("⚠️  MODERATE coordinate alignment - may have some spatial mismatches")
        else:
            print("❌ POOR coordinate alignment - significant coordinate system issues detected")
            print("   Consider verifying snow dataset CRS or using different reprojection strategy")
        
        self.snow_alignment_score = alignment_score
        
        print("="*70)
    
    def get_site_permafrost_properties(self, lat, lon):
        """Extract permafrost probability and zone for site coordinates."""
        properties = {'permafrost_prob': None, 'permafrost_zone': None, 'is_permafrost_suitable': False}
        
        try:
            # Handle permafrost probability raster (EPSG:3995)
            if self.permafrost_prob is not None:
                from pyproj import Transformer
                
                # Transform from WGS84 (EPSG:4326) to Arctic Polar Stereographic (EPSG:3995)
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:3995", always_xy=True)
                x, y = transformer.transform(lon, lat)
                
                # Convert projected coordinates to raster pixel coordinates
                transform = self.permafrost_prob['transform']
                col, row = ~transform * (x, y)
                col, row = int(col), int(row)
                
                # Debug: Print coordinate conversion for first few sites
                if hasattr(self, '_debug_count') and self._debug_count < 5:
                    print(f"Debug site {lat:.3f}, {lon:.3f}: projected=({x:.0f}, {y:.0f}), "
                          f"pixel=({row}, {col}), raster_shape={self.permafrost_prob['data'].shape}")
                    if hasattr(self, '_debug_count'):
                        self._debug_count += 1
                    else:
                        self._debug_count = 1
                
                # Extract probability if within bounds
                if (0 <= row < self.permafrost_prob['data'].shape[0] and
                    0 <= col < self.permafrost_prob['data'].shape[1]):
                    raw_value = self.permafrost_prob['data'][row, col]
                    
                    # Debug: Print raw values for first few sites
                    if hasattr(self, '_debug_count') and self._debug_count <= 5:
                        print(f"Debug raw permafrost value: {raw_value}")
                    
                    # Handle NoData values - the large negative number indicates NoData
                    if not np.isnan(raw_value) and raw_value > -1e30 and raw_value >= 0 and raw_value <= 1.0:
                        properties['permafrost_prob'] = float(raw_value)  # Already in 0-1 range
                        if hasattr(self, '_debug_count') and self._debug_count <= 5:
                            print(f"Debug converted probability: {properties['permafrost_prob']}")
                    else:
                        properties['permafrost_prob'] = 0.0  # No permafrost
                        if hasattr(self, '_debug_count') and self._debug_count <= 5:
                            print(f"Debug: NoData value detected, setting prob=0")
                else:
                    if hasattr(self, '_debug_count') and self._debug_count <= 5:
                        print(f"Debug: Coordinates outside raster bounds")
            
            # Handle permafrost zones shapefile
            if self.permafrost_zones is not None:
                from shapely.geometry import Point
                import geopandas as gpd
                
                # Create point in WGS84
                point_wgs84 = Point(lon, lat)
                
                # Convert to GeoDataFrame for reprojection
                point_gdf = gpd.GeoDataFrame([1], geometry=[point_wgs84], crs="EPSG:4326")
                
                # Reproject to match permafrost zones CRS
                point_reprojected = point_gdf.to_crs(self.permafrost_zones.crs)
                point_proj = point_reprojected.geometry.iloc[0]
                
                # Find intersecting zone
                intersecting = self.permafrost_zones[self.permafrost_zones.geometry.contains(point_proj)]
                if not intersecting.empty:
                    zone_extent = intersecting.iloc[0].get('EXTENT', 'unknown')
                    properties['permafrost_zone'] = zone_extent
                    
                    if hasattr(self, '_debug_count') and self._debug_count <= 5:
                        print(f"Debug zone found: {zone_extent}")
                    
                    # Map zone abbreviations to full names
                    zone_mapping = {
                        'Cont': 'continuous',
                        'Discon': 'discontinuous',
                        'Spora': 'sporadic',
                        'Isol': 'isolated'
                    }
                    
                    full_zone_name = zone_mapping.get(zone_extent, zone_extent.lower())
                    properties['permafrost_zone'] = full_zone_name
                    
                    # Valid permafrost zones for zero-curtain analysis
                    valid_zones = ['continuous', 'discontinuous', 'sporadic', 'isolated']
                    if full_zone_name in valid_zones:
                        properties['is_permafrost_suitable'] = True
                else:
                    if hasattr(self, '_debug_count') and self._debug_count <= 5:
                        print(f"Debug: No intersecting permafrost zone found")
            
            # Enhanced suitability determination based on literature review
            # Kane et al. (1991): Local conditions override regional patterns

            # Method 1: Any permafrost probability > 0 indicates potential
            prob_indicates_potential = (properties['permafrost_prob'] is not None and
                                       properties['permafrost_prob'] > 0.0)

            # Method 2: Any valid permafrost zone classification
            zone_indicates_potential = (properties['permafrost_zone'] is not None and
                                       properties['permafrost_zone'] in ['continuous', 'discontinuous', 'sporadic', 'isolated'])

            # Method 3: Arctic/Subarctic latitude bands (≥50°N) - permafrost possible
            latitude_indicates_potential = lat >= 50.0

            # Method 4: Fallback for areas with missing data but Arctic location
            arctic_fallback = lat >= 60.0  # High Arctic - assume permafrost potential

            # MUCH MORE PERMISSIVE: Site is suitable if ANY condition suggests permafrost potential
            properties['is_permafrost_suitable'] = (prob_indicates_potential or
                                                   zone_indicates_potential or
                                                   latitude_indicates_potential or
                                                   arctic_fallback)

            # Add reasoning for transparency
            if prob_indicates_potential:
                properties['suitability_reason'] = f'Permafrost probability: {properties["permafrost_prob"]:.3f}'
            elif zone_indicates_potential:
                properties['suitability_reason'] = f'Permafrost zone: {properties["permafrost_zone"]}'
            elif latitude_indicates_potential:
                properties['suitability_reason'] = f'Arctic latitude: {lat:.1f}°N'
            elif arctic_fallback:
                properties['suitability_reason'] = f'High Arctic location: {lat:.1f}°N'
            else:
                properties['suitability_reason'] = 'No permafrost indicators'
            
            if hasattr(self, '_debug_count') and self._debug_count <= 5:
                print(f"Debug final suitability: prob_suitable={prob_suitable}, "
                      f"zone_suitable={zone_suitable}, final={properties['is_permafrost_suitable']}")
                    
        except Exception as e:
            print(f"Warning: Error extracting permafrost properties for {lat:.3f}, {lon:.3f}: {e}")
            import traceback
            traceback.print_exc()
        
        return properties
    
    def get_site_snow_properties(self, lat, lon, timestamps):
        """
        Extract spatiotemporal snow depth, SWE, and melt data for site and time period.
        This data is used everywhere to inform physics at specific times/locations.
        """
        snow_props = {
            'snow_depth': np.array([]),
            'snow_water_equiv': np.array([]),
            'snow_melt': np.array([]),
            'timestamps': np.array([]),
            'has_snow_data': False
        }
        
        try:
            if self.snow_data is not None:
                # Convert timestamps to match snow data temporal resolution
                timestamps_pd = pd.to_datetime(timestamps)
                
                # Handle coordinate system for snow data
                if 'lat' in self.snow_data.coords:
                    snow_lat = self.snow_data.coords['lat'].values
                    snow_lon = self.snow_data.coords['lon'].values
                elif 'latitude' in self.snow_data.coords:
                    snow_lat = self.snow_data.coords['latitude'].values
                    snow_lon = self.snow_data.coords['longitude'].values
                else:
                    # Try to infer coordinate names
                    coord_names = list(self.snow_data.coords.keys())
                    lat_coords = [c for c in coord_names if 'lat' in c.lower()]
                    lon_coords = [c for c in coord_names if 'lon' in c.lower()]
                    
                    if lat_coords and lon_coords:
                        lat_coord = lat_coords[0]
                        lon_coord = lon_coords[0]
                        snow_lat = self.snow_data.coords[lat_coord].values
                        snow_lon = self.snow_data.coords[lon_coord].values
                    else:
                        print(f"Warning: Cannot identify lat/lon coordinates in snow data")
                        return snow_props
                
                # Check if snow coordinates need reprojection
                # If coordinates are in projected system (large values), need to handle differently
                if np.abs(snow_lat).max() > 180 or np.abs(snow_lon).max() > 360:
                    # Snow data appears to be in projected coordinates
                    print(f"Debug: Snow data appears to be in projected coordinates")
                    print(f"  Lat range: {snow_lat.min():.0f} to {snow_lat.max():.0f}")
                    print(f"  Lon range: {snow_lon.min():.0f} to {snow_lon.max():.0f}")
                    
                    # Need to determine CRS and reproject site coordinates
                    # Check if snow data has CRS information
                    if hasattr(self.snow_data, 'crs'):
                        snow_crs = self.snow_data.crs
                        print(f"  Snow CRS: {snow_crs}")
                        
                        # Transform site coordinates to match snow CRS
                        try:
                            from pyproj import Transformer
                            transformer = Transformer.from_crs("EPSG:4326", snow_crs, always_xy=True)
                            snow_x, snow_y = transformer.transform(lon, lat)
                            
                            # Find nearest grid point in projected space
                            lat_idx = np.argmin(np.abs(snow_lat - snow_y))
                            lon_idx = np.argmin(np.abs(snow_lon - snow_x))
                            
                        except Exception as e:
                            print(f"Warning: Could not reproject to snow CRS: {e}")
                            # Fall back to nearest neighbor without reprojection
                            lat_idx = np.argmin(np.abs(snow_lat - lat))
                            lon_idx = np.argmin(np.abs(snow_lon - lon))
                    else:
                        # No CRS info, assume it might be a common projection like polar stereographic
                        # Try Arctic Polar Stereographic (EPSG:3995) as it's common for Arctic data
                        try:
                            from pyproj import Transformer
                            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3995", always_xy=True)
                            snow_x, snow_y = transformer.transform(lon, lat)
                            
                            # Find nearest grid point
                            lat_idx = np.argmin(np.abs(snow_lat - snow_y))
                            lon_idx = np.argmin(np.abs(snow_lon - snow_x))
                            
                            print(f"Debug: Assuming EPSG:3995 for snow data")
                            print(f"  Site projected: ({snow_x:.0f}, {snow_y:.0f})")
                            print(f"  Nearest grid: lat_idx={lat_idx}, lon_idx={lon_idx}")
                            
                        except Exception as e:
                            print(f"Warning: Projection assumption failed: {e}")
                            # Last resort: use indices based on position in array
                            lat_idx = len(snow_lat) // 2
                            lon_idx = len(snow_lon) // 2
                else:
                    # Snow data appears to be in geographic coordinates (lat/lon)
                    lat_idx = np.argmin(np.abs(snow_lat - lat))
                    lon_idx = np.argmin(np.abs(snow_lon - lon))
                    
                    print(f"Debug: Snow data in geographic coordinates")
                    print(f"  Site: {lat:.3f}, {lon:.3f}")
                    print(f"  Nearest grid: {snow_lat[lat_idx]:.3f}, {snow_lon[lon_idx]:.3f}")
                
                # Extract time series for each timestamp
                snow_time_axis = None
                if 'time' in self.snow_data.coords:
                    snow_time_axis = self.snow_data.coords['time']
                elif 'valid_time' in self.snow_data.coords:
                    snow_time_axis = self.snow_data.coords['valid_time']
                
                if snow_time_axis is not None:
                    # Map variable names to snow properties
                    var_mappings = {
                        'sd': 'snow_depth',      # Snow depth
                        'sde': 'snow_depth',     # Snow depth (alternative)
                        'depth': 'snow_depth',   # Snow depth
                        'swe': 'snow_water_equiv', # Snow water equivalent
                        'smlt': 'snow_melt',     # Snow melt
                        'snowmelt': 'snow_melt', # Snow melt (alternative)
                        'melt': 'snow_melt'      # Snow melt
                    }
                    
                    # Interpolate snow data to measurement timestamps
                    for var_name, prop_name in var_mappings.items():
                        if var_name in self.snow_data.variables:
                            try:
                                # Extract spatial slice
                                var_data = self.snow_data[var_name][:, lat_idx, lon_idx]
                                
                                # Handle ensemble data (if 'number' dimension exists)
                                if 'number' in var_data.dims:
                                    var_data = var_data.mean(dim='number')  # Average across ensemble members
                                
                                # Get valid (non-NaN) data
                                valid_mask = ~np.isnan(var_data.values)
                                
                                if np.any(valid_mask):
                                    # Create interpolation function
                                    from scipy.interpolate import interp1d
                                    
                                    snow_times = pd.to_datetime(var_data.coords[snow_time_axis.name].values)
                                    valid_times = snow_times[valid_mask]
                                    valid_values = var_data.values[valid_mask]
                                    
                                    if len(valid_times) > 1:
                                        f_interp = interp1d(
                                            valid_times.astype(np.int64),
                                            valid_values,
                                            kind='linear',
                                            bounds_error=False,
                                            fill_value=0.0
                                        )
                                        
                                        # Interpolate to measurement times
                                        interp_values = f_interp(timestamps_pd.astype(np.int64))
                                        
                                        # Store results
                                        if prop_name not in snow_props or len(snow_props[prop_name]) == 0:
                                            snow_props[prop_name] = interp_values
                                        
                                        snow_props['has_snow_data'] = True
                                        
                                        print(f"Debug: Successfully extracted {var_name} -> {prop_name}")
                                        print(f"  Value range: {np.nanmin(interp_values):.2f} to {np.nanmax(interp_values):.2f}")
                                        
                            except Exception as e:
                                print(f"Warning: Error extracting {var_name}: {e}")
                                continue
                
                snow_props['timestamps'] = timestamps_pd.values
                        
        except Exception as e:
            print(f"Warning: Error extracting snow properties for {lat:.3f}, {lon:.3f}: {e}")
            import traceback
            traceback.print_exc()
        
        return snow_props
            
# ===== CRYOGRID ENHANCED METHODS =====
    
    def _calculate_cryogrid_enthalpy(self, temperature, water_content, ice_content, soil_props):
        """
        Calculate enthalpy using CryoGrid formulation (Equation 1).
        e(T, θ_w) = c*T - L_vol_sl*(θ_wi - θ_w)
        """
        c_eff = self._calculate_effective_heat_capacity_cryogrid(soil_props, temperature)
        theta_wi = water_content + ice_content
        
        # CryoGrid Equation 1: enthalpy with sensible and latent components
        enthalpy = c_eff * temperature - self.LVOL_SL * (theta_wi - water_content)
        
        return enthalpy
    
    def _calculate_effective_heat_capacity_cryogrid(self, soil_props, temperature):
        """
        Calculate effective heat capacity using CryoGrid formulation (Equation 2).
        Temperature-dependent heat capacity accounting for phase state.
        """
        theta_m = soil_props.get('mineral_fraction', 0.7)
        theta_o = soil_props.get('organic_fraction', 0.1)
        theta_wi = soil_props.get('water_fraction', 0.2)
        
        # CryoGrid specific heat capacities
        c_m = self.CMIN   # Mineral heat capacity
        c_o = self.CORG   # Organic heat capacity
        c_w = self.CWATER # Water heat capacity
        c_i = self.CICE   # Ice heat capacity
        
        # Temperature-dependent formulation (CryoGrid Equation 2)
        if temperature < 0:
            # Below freezing: use ice heat capacity for water phase
            c_eff = theta_m * c_m + theta_o * c_o + theta_wi * c_i
        else:
            # Above freezing: use water heat capacity
            c_eff = theta_m * c_m + theta_o * c_o + theta_wi * c_w
            
        return c_eff
    
    def _invert_enthalpy_to_temperature_cryogrid(self, enthalpy, theta_wi, soil_props):
        """
        Invert enthalpy to derive temperature and liquid water fraction.
        Based on CryoGrid diagnostic step (Section 2.2.3).
        """
        # Free water freezing characteristic implementation (CryoGrid Equations 16-17)
        if self.use_painter_karra_freezing:
            return self._painter_karra_inversion(enthalpy, theta_wi, soil_props)
        else:
            return self._free_water_inversion(enthalpy, theta_wi, soil_props)
    
    def _free_water_inversion(self, enthalpy, theta_wi, soil_props):
        """Free water freezing characteristic (CryoGrid Equations 16-17)."""
        c_eff_frozen = self._calculate_effective_heat_capacity_cryogrid(soil_props, -1.0)
        c_eff_thawed = self._calculate_effective_heat_capacity_cryogrid(soil_props, 1.0)
        
        # Phase change boundaries
        e_thaw_start = 0  # Enthalpy at 0°C, all unfrozen
        e_freeze_complete = -self.LVOL_SL * theta_wi  # All frozen
        
        if enthalpy >= e_thaw_start:
            # Completely thawed (CryoGrid Eq. 16, case 1)
            temperature = enthalpy / c_eff_thawed
            liquid_fraction = theta_wi
            
        elif e_freeze_complete <= enthalpy < e_thaw_start:
            # Phase change zone (CryoGrid Eq. 16, case 2)
            temperature = 0.0
            liquid_fraction = theta_wi * (1 + enthalpy / (self.LVOL_SL * theta_wi))
            
        else:
            # Completely frozen (CryoGrid Eq. 16, case 3)
            temperature = (enthalpy + self.LVOL_SL * theta_wi) / c_eff_frozen
            liquid_fraction = 0.0
            
        return temperature, max(0, min(liquid_fraction, theta_wi))
    
    def _painter_karra_inversion(self, enthalpy, theta_wi, soil_props):
        """
        Painter-Karra soil freezing characteristic inversion.
        Based on CryoGrid implementation (Equations 18-20).
        """
        # For enthalpy >= 0, use free water formulation
        if enthalpy >= 0:
            return self._free_water_inversion(enthalpy, theta_wi, soil_props)
        
        # Parameter validation and safeguards
        alpha = max(soil_props.get('van_genuchten_alpha', 0.5), 0.01)  # Minimum alpha
        n = max(soil_props.get('van_genuchten_n', 2.0), 1.1)          # Minimum n > 1
        m = 1 - 1/n
        porosity = max(soil_props.get('porosity', 0.4), 0.1)          # Minimum porosity
        theta_wi = min(theta_wi, porosity * 0.99)
        
        # For enthalpy < 0, use Painter-Karra characteristic
        alpha = soil_props.get('van_genuchten_alpha', 0.5)  # m^-1
        n = soil_props.get('van_genuchten_n', 2.0)
        m = 1 - 1/n
        porosity = soil_props.get('porosity', 0.4)
        
        # Use lookup table approach for efficiency (as mentioned in CryoGrid paper)
        # For simplification, use iterative approach here
        def enthalpy_residual(T_kelvin):
            T_celsius = T_kelvin - self.TMFW
            
            if T_celsius >= 0:
                return enthalpy  # Should not reach here
            
            # Calculate matric potential (CryoGrid Equations 18-19) with numerical safeguards
            saturation = theta_wi / porosity
            saturation = np.clip(saturation, 1e-6, 0.999)  # Prevent numerical issues

            # Safeguard the power calculations
            base_term = saturation**(1/m) - 1
            if base_term <= 0:
                base_term = 1e-6  # Small positive value to prevent invalid powers

            psi_0 = (1/alpha) * (base_term)**(1/n)
            
            # Ice-liquid surface tension ratio
            eta = 2.2  # As suggested in Painter and Karra (2014)
            
            psi = psi_0 + eta * (self.LVOL_SL / (self.G * self.RHO_WATER)) * T_celsius / self.TMFW
            
            # Calculate liquid water content (CryoGrid Equation 20)
            theta_w = porosity * (1 + (alpha * abs(psi))**n)**(-m)
            theta_w = max(0, min(theta_w, theta_wi))
            
            # Calculate enthalpy for this state
            c_eff = self._calculate_effective_heat_capacity_cryogrid(soil_props, T_celsius)
            calc_enthalpy = c_eff * T_celsius - self.LVOL_SL * (theta_wi - theta_w)
            
            return calc_enthalpy - enthalpy
        
        # Solve for temperature iteratively
        try:
            from scipy.optimize import brentq
            T_kelvin = brentq(enthalpy_residual, 200, self.TMFW)
            temperature = T_kelvin - self.TMFW
            
            # Recalculate liquid fraction at solution with numerical safeguards
            saturation = theta_wi / porosity
            saturation = np.clip(saturation, 1e-6, 0.999)

            base_term = saturation**(1/m) - 1
            if base_term <= 0:
                base_term = 1e-6

            psi_0 = (1/alpha) * (base_term)**(1/n)
            eta = 2.2
            psi = psi_0 + eta * (self.LVOL_SL / (self.G * self.RHO_WATER)) * temperature / self.TMFW
            liquid_fraction = porosity * (1 + (alpha * abs(psi))**n)**(-m)
            liquid_fraction = max(0, min(liquid_fraction, theta_wi))
            
        except:
            # Fallback to free water characteristic
            temperature, liquid_fraction = self._free_water_inversion(enthalpy, theta_wi, soil_props)
        
        return temperature, liquid_fraction
    
    def _calculate_surface_energy_balance_cryogrid(self, forcing_data, surface_properties):
        """
        Calculate surface energy balance following CryoGrid Equation 5.
        F_ub = S_in - S_out + L_in - L_out - Q_h - Q_e
        """
        if not self.use_surface_energy_balance:
            return forcing_data.get('surface_temperature', 0)
        
        # Shortwave radiation balance (CryoGrid Equation 6)
        S_in = forcing_data.get('shortwave_in', 200)
        albedo = self._calculate_dynamic_albedo_cryogrid(surface_properties)
        S_out = albedo * S_in
        
        # Longwave radiation balance (CryoGrid Equation 7)
        L_in = forcing_data.get('longwave_in', 300)
        emissivity = surface_properties.get('emissivity', 0.95)
        surface_temp_K = surface_properties.get('temperature', 273.15) + self.TMFW
        L_out = (emissivity * self.STEFAN_BOLTZMANN * surface_temp_K**4 +
                (1 - emissivity) * L_in)
        
        # Sensible heat flux (CryoGrid Equations 8-9)
        Q_h = self._calculate_sensible_heat_flux_cryogrid(forcing_data, surface_properties)
        
        # Latent heat flux (CryoGrid Equation 10)
        Q_e = self._calculate_latent_heat_flux_cryogrid(forcing_data, surface_properties)
        
        # Surface energy balance
        F_ub = S_in - S_out + L_in - L_out - Q_h - Q_e
        
        return F_ub
    
    def _calculate_dynamic_albedo_cryogrid(self, surface_properties):
        """Calculate dynamic albedo based on surface conditions."""
        base_albedo = 0.2  # Soil albedo
        snow_depth = surface_properties.get('snow_depth', 0)
        
        if snow_depth > 0.01:  # Snow present
            # Fresh vs aged snow albedo
            snow_age_days = surface_properties.get('snow_age_days', 0)
            fresh_albedo = 0.8
            aged_albedo = 0.5
            
            # Exponential decay with age
            albedo = aged_albedo + (fresh_albedo - aged_albedo) * np.exp(-snow_age_days / 5.0)
            
            # Depth-dependent weighting
            depth_factor = min(1.0, snow_depth / 0.1)  # Full coverage at 10cm
            albedo = base_albedo + (albedo - base_albedo) * depth_factor
        else:
            albedo = base_albedo
            
        return np.clip(albedo, 0.1, 0.9)
    
    def _calculate_sensible_heat_flux_cryogrid(self, forcing_data, surface_properties):
        """Calculate sensible heat flux using CryoGrid formulation."""
        air_temp = forcing_data.get('air_temperature', 273.15)
        surface_temp = surface_properties.get('temperature', 273.15)
        wind_speed = forcing_data.get('wind_speed', 3.0)
        
        # Air properties
        rho_air = 1.225  # kg/m3
        cp_air = 1005    # J/kg/K
        
        # Aerodynamic resistance (simplified)
        z0 = 0.01  # Roughness length [m]
        height = 2.0  # Measurement height [m]
        kappa = 0.4  # von Karman constant
        
        r_a = (1 / (kappa**2 * wind_speed)) * (np.log(height / z0))**2
        
        # Sensible heat flux (CryoGrid Equation 8)
        Q_h = rho_air * cp_air * (air_temp - surface_temp) / r_a
        
        return Q_h
    
    def _calculate_latent_heat_flux_cryogrid(self, forcing_data, surface_properties):
        """Calculate latent heat flux using CryoGrid formulation."""
        # Simplified implementation for demonstration
        air_humidity = forcing_data.get('specific_humidity', 0.005)
        surface_temp = surface_properties.get('temperature', 273.15)
        
        # Simplified latent heat calculation
        if surface_temp < self.TMFW:
            L_lg_sg = 2.834e6  # Sublimation [J/kg]
        else:
            L_lg_sg = 2.501e6  # Evaporation [J/kg]
        
        # Simplified flux calculation
        Q_e = L_lg_sg * 0.001 * max(0, 0.01 - air_humidity)  # Simplified
        
        return Q_e
    
    def _adaptive_timestep_control_cryogrid(self, current_state):
        """
        Implement CryoGrid's adaptive time-stepping for stability.
        Based on maximum enthalpy change per time step (Section 2.2.9).
        """
        if not self.use_adaptive_timestep:
            return self.DT
        
        # Calculate maximum heat flux in domain
        max_flux = 0
        min_spacing = np.inf
        
        for i, layer in enumerate(current_state.get('layers', [])):
            # Heat flux calculation
            if i < len(current_state['layers']) - 1:
                next_layer = current_state['layers'][i + 1]
                dT_dz = (next_layer['temperature'] - layer['temperature']) / layer['thickness']
                flux = layer['thermal_conductivity'] * abs(dT_dz)
                max_flux = max(max_flux, flux)
            
            min_spacing = min(min_spacing, layer['thickness'])
        
        # CFL-based time step for stability
        max_diffusivity = self._calculate_max_thermal_diffusivity(current_state)
        dt_cfl = 0.4 * min_spacing**2 / (2 * max_diffusivity) if max_diffusivity > 0 else self.DT
        
        # Energy-based time step (CryoGrid approach)
        dt_energy = self.MAX_ENTHALPY_CHANGE / (max_flux + 1e-12) if max_flux > 0 else self.DT
        
        # Return minimum of constraints
        return max(min(dt_cfl, dt_energy, self.DT), 60)  # At least 1 minute
    
    def _calculate_max_thermal_diffusivity(self, current_state):
        """Calculate maximum thermal diffusivity in the domain."""
        max_diffusivity = 0
        
        for layer in current_state.get('layers', []):
            thermal_conductivity = layer.get('thermal_conductivity', self.KMIN)
            heat_capacity = layer.get('heat_capacity', self.CMIN)
            diffusivity = thermal_conductivity / heat_capacity
            max_diffusivity = max(max_diffusivity, diffusivity)
        
        return max_diffusivity
    
    def _apply_lateral_thermal_effects_cryogrid(self, site_data, permafrost_props, spatial_context):
        """
        Apply lateral thermal interactions following CryoGrid Section 2.3.1.
        Accounts for spatial heterogeneity in permafrost thermal regime.
        """
        if not spatial_context or not spatial_context.get('thermal_reservoir_distance'):
            return site_data
        
        # Lateral heat reservoir parameters (CryoGrid Equation 32)
        lateral_distance = spatial_context['thermal_reservoir_distance']  # [m]
        reservoir_temp = spatial_context.get('reservoir_temperature', 0)  # [°C]
        contact_length = spatial_context.get('contact_length', 1.0)  # [m]
        lateral_timestep = spatial_context.get('lateral_timestep', 3600)  # [s]
        
        # Apply to layers within reservoir bounds
        reservoir_lower = spatial_context.get('reservoir_lower', 0)
        reservoir_upper = spatial_context.get('reservoir_upper', 2.0)
        
        for i, layer in enumerate(site_data.get('layers', [])):
            layer_depth = layer.get('depth', i * 0.1)
            
            if reservoir_lower <= layer_depth <= reservoir_upper:
                # Calculate lateral heat flux (CryoGrid Equation 32)
                thermal_conductivity = layer.get('thermal_conductivity', self.KMIN)
                layer_temp = layer.get('temperature', 0)
                
                j_lat_hc = thermal_conductivity * (layer_temp - reservoir_temp) / lateral_distance
                
                # Calculate enthalpy change (CryoGrid Equation 33)
                layer_thickness = layer.get('thickness', 0.1)
                delta_E = lateral_timestep * layer_thickness * contact_length * j_lat_hc
                
                # Apply thermal modification
                heat_capacity = layer.get('heat_capacity', self.CMIN)
                delta_T = delta_E / (heat_capacity * layer_thickness * contact_length)
                
                layer['temperature'] -= delta_T  # Heat loss to reservoir
                
                # Store lateral effects for analysis
                if 'lateral_effects' not in layer:
                    layer['lateral_effects'] = {}
                layer['lateral_effects']['heat_flux'] = j_lat_hc
                layer['lateral_effects']['temperature_change'] = -delta_T
        
        return site_data

# ===== ENHANCED STEFAN PROBLEM WITH CRYOGRID =====
    
    def solve_stefan_problem_enhanced(self, initial_temp, boundary_temp, soil_properties,
                                     duration_days, forcing_data=None):
        """
        Enhanced Stefan problem solver integrating CryoGrid formulations.
        Combines original LPJ-EOSIM approach with CryoGrid thermodynamics.
        """
        
        # Discretization
        nz = min(self.MAX_LAYERS, max(10, int(soil_properties['depth_range'] / self.DZ_MIN)))
        nt = duration_days
        
        dz = soil_properties['depth_range'] / nz
        
        # Initialize with CryoGrid enthalpy-based state
        if self.use_cryogrid_enthalpy:
            # Initialize using enthalpy formulation
            T = np.linspace(boundary_temp, initial_temp, nz)
            theta_wi = np.full(nz, soil_properties.get('water_fraction', 0.2))
            
            # Calculate initial enthalpies
            enthalpies = np.zeros(nz)
            liquid_fractions = np.zeros(nz)
            
            for i in range(nz):
                initial_liquid = theta_wi[i] if T[i] >= 0 else 0
                enthalpies[i] = self._calculate_cryogrid_enthalpy(
                    T[i], initial_liquid, theta_wi[i] - initial_liquid, soil_properties
                )
                liquid_fractions[i] = initial_liquid
        else:
            # Use original formulation
            T = np.linspace(boundary_temp, initial_temp, nz)
            liquid_fractions = np.ones(nz)
            ice_fractions = np.zeros(nz)
        
        # Storage for solution
        temp_history = np.zeros((nt, nz))
        freeze_depths = np.zeros(nt)
        phase_change_energy = np.zeros(nt)
        enthalpy_history = np.zeros((nt, nz)) if self.use_cryogrid_enthalpy else None
        
        # Time integration with adaptive stepping
        for timestep in range(nt):
            # Determine time step
            if self.use_adaptive_timestep:
                current_state = {
                    'layers': [
                        {
                            'temperature': T[i],
                            'thickness': dz,
                            'thermal_conductivity': self._calculate_thermal_conductivity(soil_properties),
                            'heat_capacity': self._calculate_heat_capacity(soil_properties)
                        }
                        for i in range(nz)
                    ]
                }
                dt = self._adaptive_timestep_control_cryogrid(current_state)
            else:
                dt = self.DT
            
            # Update boundary conditions with surface energy balance
            if forcing_data and self.use_surface_energy_balance:
                surface_props = {
                    'temperature': T[0],
                    'snow_depth': forcing_data.get('snow_depth', 0),
                    'emissivity': 0.95
                }
                boundary_flux = self._calculate_surface_energy_balance_cryogrid(
                    forcing_data, surface_props
                )
                # Convert flux to temperature (simplified)
                k_thermal = self._calculate_thermal_conductivity(soil_properties)
                T[0] = T[1] + boundary_flux * dz / k_thermal
            else:
                # Seasonal variation
                T[0] = boundary_temp + np.sin(2*np.pi*timestep/365) * 5
            
            if self.use_cryogrid_enthalpy:
                # Enhanced enthalpy-based solution
                T_new, liquid_fractions_new = self._solve_enthalpy_timestep_cryogrid(
                    T, enthalpies, theta_wi, soil_properties, dt, dz, nz
                )
                
                # Update enthalpies
                for i in range(nz):
                    ice_fraction = theta_wi[i] - liquid_fractions_new[i]
                    enthalpies[i] = self._calculate_cryogrid_enthalpy(
                        T_new[i], liquid_fractions_new[i], ice_fraction, soil_properties
                    )
                
                T = T_new
                liquid_fractions = liquid_fractions_new
                
                if enthalpy_history is not None:
                    enthalpy_history[timestep, :] = enthalpies
                    
            else:
                # Original Crank-Nicholson approach
                T_old = T.copy()
                rho_c = self._calculate_heat_capacity(soil_properties)
                k_thermal = self._calculate_thermal_conductivity(soil_properties)
                alpha = k_thermal / rho_c
                r = alpha * dt / (dz**2)
                
                A, b = self._setup_crank_nicholson_matrix(T, T_old, r, nz, dz)
                T_new = spsolve(A, b)
                
                # Handle phase change
                for i in range(1, nz-1):
                    if abs(T_new[i]) <= self.TEMP_THRESHOLD:
                        delta_H = self._calculate_phase_change_energy(T_new[i], T[i], dz)
                        
                        if delta_H > self.PHASE_CHANGE_ENERGY:
                            water_frozen = min(liquid_fractions[i], delta_H / self.LHEAT)
                            liquid_fractions[i] -= water_frozen
                            ice_fractions[i] += water_frozen
                            T_new[i] = 0.0
                            
                        elif delta_H < -self.PHASE_CHANGE_ENERGY:
                            ice_melted = min(ice_fractions[i], abs(delta_H) / self.LHEAT)
                            ice_fractions[i] -= ice_melted
                            liquid_fractions[i] += ice_melted
                            T_new[i] = 0.0
                
                T = T_new
            
            # Store results
            temp_history[timestep, :] = T
            freeze_depths[timestep] = self._calculate_freeze_depth(T, dz)
            phase_change_energy[timestep] = np.sum(np.abs(theta_wi - liquid_fractions)) * self.LHEAT
        
        result = {
            'temperature_profile': temp_history,
            'freeze_depths': freeze_depths,
            'phase_change_energy': phase_change_energy,
            'liquid_fraction': liquid_fractions,
            'depths': np.arange(nz) * dz
        }
        
        if enthalpy_history is not None:
            result['enthalpy_profile'] = enthalpy_history
            
        return result
    
    def _solve_enthalpy_timestep_cryogrid(self, T_old, enthalpies, theta_wi, soil_props, dt, dz, nz):
        """Solve single timestep using CryoGrid enthalpy formulation."""
        
        # Replace only mathematically invalid values (NaN/inf) - not physical values
        enthalpies = np.nan_to_num(enthalpies, nan=0.0, posinf=1e15, neginf=-1e15)
        T_old = np.nan_to_num(T_old, nan=0.0, posinf=1e15, neginf=-1e15)
        
        # Calculate thermal properties
        thermal_conductivities = np.zeros(nz)
        heat_capacities = np.zeros(nz)
        
        for i in range(nz):
            thermal_conductivities[i] = self._calculate_thermal_conductivity(soil_props)
            heat_capacities[i] = self._calculate_effective_heat_capacity_cryogrid(soil_props, T_old[i])
        
        # Heat conduction fluxes (CryoGrid Equation 14)
        heat_fluxes = np.zeros(nz + 1)
        
        for i in range(1, nz):
            # Interface thermal conductivity (series resistance)
            k_interface = 2 * thermal_conductivities[i-1] * thermal_conductivities[i] / \
                         (thermal_conductivities[i-1] + thermal_conductivities[i])
            
            # Heat flux with overflow protection
            # Heat flux with mathematical overflow protection only
            temp_diff = T_old[i] - T_old[i-1]
            flux_value = -k_interface * temp_diff / dz

            # Only protect against mathematical invalidity
            if np.isfinite(flux_value):
                heat_fluxes[i] = flux_value
            else:
                heat_fluxes[i] = 0.0  # Mathematical fallback
                print(f"Warning: Non-finite heat flux at interface {i}")
        
        # Update enthalpies
        enthalpies_new = enthalpies.copy()
        
        for i in range(nz):
            # Flux divergence
            flux_divergence = (heat_fluxes[i+1] - heat_fluxes[i]) / dz if i < nz-1 else -heat_fluxes[i] / dz
            
            # Enthalpy update (CryoGrid Equation 13)
            # Prevent overflow in enthalpy updates
            flux_term = dt * flux_divergence

            # Check for potential overflow before adding
            if np.isfinite(flux_term) and abs(flux_term) < 1e30:
                enthalpies_new[i] += flux_term
            else:
                # Handle overflow/invalid values
                if flux_divergence > 0:
                    enthalpies_new[i] += min(dt * 1e6, 1e15)  # Cap positive additions
                else:
                    enthalpies_new[i] += max(dt * -1e6, -1e15)  # Cap negative additions
                
                print(f"Warning: Capped enthalpy flux at layer {i}: flux_divergence={flux_divergence}")
        
        # Invert enthalpies to get temperature and liquid fractions
        T_new = np.zeros(nz)
        liquid_fractions_new = np.zeros(nz)
        
        for i in range(nz):
            T_new[i], liquid_fractions_new[i] = self._invert_enthalpy_to_temperature_cryogrid(
                enthalpies_new[i], theta_wi[i], soil_props
            )
        
        return T_new, liquid_fractions_new
        
# ===== ORIGINAL METHODS (PRESERVED) =====
    
    def _setup_crank_nicholson_matrix(self, T, T_old, r, nz, dz):
        """Setup Crank-Nicholson matrix system following LPJ-EOSIM cnstep."""
        
        # Tridiagonal matrix for Crank-Nicholson
        main_diag = np.ones(nz) * (1 + r)
        upper_diag = np.ones(nz-1) * (-r/2)
        lower_diag = np.ones(nz-1) * (-r/2)
        
        # Handle boundary conditions
        main_diag[0] = 1.0
        main_diag[-1] = 1 + r/2
        upper_diag[0] = 0.0
        
        # Create sparse matrix
        A = sparse.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
        
        # Right-hand side
        b = np.zeros(nz)
        b[0] = T[0]  # Boundary condition
        
        for i in range(1, nz-1):
            b[i] = T_old[i] + r/2 * (T_old[i+1] - 2*T_old[i] + T_old[i-1])
        
        b[-1] = T_old[-1] + r/2 * (-T_old[-1] + T_old[-2])
        
        return A, b
    
    def _calculate_heat_capacity(self, soil_props):
        """Calculate effective heat capacity from soil composition."""
        
        # From LPJ-EOSIM permafrost.c heat capacity calculation
        f_organic = soil_props.get('organic_fraction', 0.1)
        f_mineral = soil_props.get('mineral_fraction', 0.8)
        f_water = soil_props.get('water_fraction', 0.1)
        
        return (f_organic * self.CORG +
                f_mineral * self.CMIN +
                f_water * self.CWATER)
    
    def _calculate_thermal_conductivity(self, soil_props):
        """Calculate effective thermal conductivity using geometric mean."""
        
        # From LPJ-EOSIM thermal conductivity calculation
        f_organic = soil_props.get('organic_fraction', 0.1)
        f_mineral = soil_props.get('mineral_fraction', 0.8)
        f_water = soil_props.get('water_fraction', 0.1)
        f_ice = soil_props.get('ice_fraction', 0.0)
        
        # Geometric mean following LPJ-EOSIM approach
        k_effective = (self.KORG**f_organic *
                      self.KMIN**f_mineral *
                      self.KWATER**f_water *
                      self.KICE**f_ice)
        
        return k_effective
    
    def _calculate_phase_change_energy(self, T_new, T_old, dz):
        """Calculate energy available for phase change."""
        return self.CWATER * (T_new - T_old) * dz
    
    def _calculate_freeze_depth(self, temperature_profile, dz):
        """Calculate depth of freezing front."""
        freezing_indices = np.where(temperature_profile <= 0)[0]
        if len(freezing_indices) > 0:
            return freezing_indices[-1] * dz
        return 0.0
    
    def apply_darcy_moisture_transport(self, soil_props, temperature_gradient, freeze_depth):
        """
        Apply Darcy's Law for moisture transport during freeze-thaw.
        
        q = -k(∇P) where k is hydraulic conductivity, P is pressure potential
        """
        
        # Hydraulic conductivity (temperature dependent)
        k_sat = soil_props.get('hydraulic_conductivity', 1e-6)  # m/s
        
        # Cryosuction effect - enhanced flow toward freezing front
        if freeze_depth > 0:
            # Pressure gradient due to ice lens formation
            pressure_gradient = self._calculate_cryosuction_pressure(temperature_gradient, freeze_depth)
            
            # Darcy flux
            moisture_flux = k_sat * pressure_gradient
            
            # Modify effective moisture content
            enhanced_moisture = soil_props.get('water_fraction', 0.1) * (1 + moisture_flux * 0.1)
            
            return np.clip(enhanced_moisture, 0.0, 0.5)
        
        return soil_props.get('water_fraction', 0.1)
    
    def _calculate_cryosuction_pressure(self, temp_gradient, freeze_depth):
        """Calculate pressure gradient due to cryosuction."""
        
        # Simplified cryosuction model
        # P = ρ_w * g * h + ρ_i * L_f * ΔT/T_f
        
        pressure_grad = (self.RHO_WATER * self.G * freeze_depth +
                        self.RHO_WATER * self.LHEAT * abs(temp_gradient) / 273.15)
        
        return pressure_grad
    
    def detect_zero_curtain_with_physics(self, site_data, lat, lon):
        """
        Comprehensive zero-curtain detection using full physics integration
        with CryoGrid enhancements.
        """
        
        # Get auxiliary data for site
        permafrost_props = self.get_site_permafrost_properties(lat, lon)
        snow_props = self.get_site_snow_properties(lat, lon, site_data['datetime'].values)
        
        # Prepare soil properties based on permafrost characteristics
        soil_properties = self._infer_soil_properties_enhanced(permafrost_props, site_data)
        
        # Spatial context for lateral effects
        spatial_context = self._determine_spatial_context(lat, lon, permafrost_props)
        
        # Group by depth for vertical analysis
        depth_groups = site_data.groupby('soil_temp_depth_zone')
        
        zero_curtain_events = []
        
        for depth_zone, group_data in depth_groups:
            if len(group_data) < self.MIN_DURATION_HOURS:
                continue
            
            # Sort by time
            group_sorted = group_data.sort_values('datetime')
            temperatures = group_sorted['soil_temp'].values
            timestamps = group_sorted['datetime'].values
            
            # Apply snow insulation effects with CryoGrid enhancements
            if len(snow_props['snow_depth']) > 0:
                temperatures = self._apply_snow_thermal_effects_enhanced(
                    temperatures, snow_props, timestamps, soil_properties
                )
            
            # Prepare forcing data for enhanced Stefan solver
            forcing_data = self._prepare_forcing_data(group_sorted, snow_props)
            
            # Solve enhanced Stefan problem with CryoGrid integration
            stefan_solution = self.solve_stefan_problem_enhanced(
                initial_temp=temperatures[0],
                boundary_temp=temperatures[-1],
                soil_properties=soil_properties,
                duration_days=len(temperatures),
                forcing_data=forcing_data
            )
            
            # Apply lateral thermal effects
            enhanced_site_data = self._apply_lateral_thermal_effects_cryogrid(
                {'layers': [{'temperature': t, 'depth': i*0.1} for i, t in enumerate(temperatures)]},
                permafrost_props,
                spatial_context
            )
            
            # Detect zero-curtain periods using enhanced physics-based criteria
            events = self._identify_zero_curtain_physics_enhanced(
                temperatures, timestamps, stefan_solution,
                permafrost_props, snow_props, depth_zone,
                soil_properties
            )
            
            zero_curtain_events.extend(events)
        
        return zero_curtain_events
    
    def _infer_soil_properties_enhanced(self, permafrost_props, site_data):
        """Enhanced soil property inference with CryoGrid parameters."""
        
        # Base properties
        properties = {
            'depth_range': 2.0,  # meters
            'organic_fraction': 0.1,
            'mineral_fraction': 0.8,
            'water_fraction': 0.1,
            'ice_fraction': 0.0,
            'hydraulic_conductivity': 1e-6,
            'porosity': 0.4,
            # CryoGrid-specific parameters
            'van_genuchten_alpha': 0.5,  # m^-1
            'van_genuchten_n': 2.0,
            'emissivity': 0.95
        }
        
        # Adjust based on permafrost probability
        pf_prob = permafrost_props['permafrost_prob']
        
        if pf_prob and pf_prob > 0.7:  # High permafrost probability
            properties['organic_fraction'] = 0.3
            properties['mineral_fraction'] = 0.6
            properties['water_fraction'] = 0.15
            properties['ice_fraction'] = 0.05
            properties['van_genuchten_n'] = 1.8  # Finer texture
            
        elif pf_prob and pf_prob > 0.3:  # Moderate permafrost probability
            properties['organic_fraction'] = 0.2
            properties['mineral_fraction'] = 0.7
            properties['water_fraction'] = 0.12
            properties['van_genuchten_n'] = 1.9
        
        # Adjust based on permafrost zone
        zone = permafrost_props.get('permafrost_zone', 'none')
        if zone == 'continuous':
            properties['ice_fraction'] += 0.1
            properties['van_genuchten_alpha'] = 0.3  # Lower permeability
        elif zone == 'discontinuous':
            properties['ice_fraction'] += 0.05
            properties['van_genuchten_alpha'] = 0.4
        
        return properties
    
    def _determine_spatial_context(self, lat, lon, permafrost_props):
        """Determine spatial context for lateral thermal effects."""
        
        # Simplified spatial context based on permafrost characteristics
        context = {}
        
        pf_prob = permafrost_props.get('permafrost_prob', 0)
        zone = permafrost_props.get('permafrost_zone', 'none')
        
        # Determine if lateral thermal reservoir effects should be applied
        if pf_prob and pf_prob > 0.5:
            # Strong permafrost areas may have lateral thermal reservoirs
            context['thermal_reservoir_distance'] = 50.0  # meters
            context['reservoir_temperature'] = -2.0  # °C
            context['reservoir_lower'] = 0.5  # m
            context['reservoir_upper'] = 1.5  # m
            context['contact_length'] = 1.0  # m
            context['lateral_timestep'] = 3600  # s
        
        return context
    
    def _prepare_forcing_data(self, group_data, snow_props):
        """Prepare forcing data for enhanced Stefan solver."""
        
        # Extract or estimate forcing variables
        forcing = {
            'air_temperature': group_data['soil_temp'].mean() + 5,  # Estimate from soil temp
            'shortwave_in': 200,  # W/m2 - simplified
            'longwave_in': 300,   # W/m2 - simplified
            'wind_speed': 3.0,    # m/s - simplified
            'specific_humidity': 0.005  # kg/kg - simplified
        }
        
        # Add snow data if available
        if snow_props['has_snow_data'] and len(snow_props['snow_depth']) > 0:
            forcing['snow_depth'] = np.mean(snow_props['snow_depth']) / 100.0  # cm to m
        
        return forcing
    
    def _apply_snow_thermal_effects_enhanced(self, temperatures, snow_props, timestamps, soil_props):
        """
        Enhanced snow thermal effects using CryoGrid principles.
        Integrates thermal conductivity calculations and energy balance.
        """
        
        if not snow_props['has_snow_data'] or len(snow_props['snow_depth']) == 0:
            return temperatures
        
        # Enhanced snow thermal properties based on CryoGrid
        modified_temps = temperatures.copy()
        
        for i, (temp, timestamp) in enumerate(zip(temperatures, timestamps)):
            
            if i < len(snow_props['snow_depth']):
                snow_depth = snow_props['snow_depth'][i] / 100.0  # cm to m
                snow_swe = snow_props['snow_water_equiv'][i] if len(snow_props['snow_water_equiv']) > i else 0
                snow_melt = snow_props['snow_melt'][i] if len(snow_props['snow_melt']) > i else 0
                
                if snow_depth > 0.01:  # Significant snow cover
                    
                    # CryoGrid-based snow thermal conductivity
                    snow_density = self._estimate_snow_density(snow_depth, snow_swe)
                    snow_k = self._calculate_snow_thermal_conductivity_cryogrid(snow_density)
                    
                    # Enhanced insulation calculation
                    soil_k = self._calculate_thermal_conductivity(soil_props)
                    thermal_resistance_ratio = snow_k / soil_k
                    depth_factor = np.tanh(snow_depth / 0.3)  # Saturation at 30cm
                    
                    insulation_factor = np.exp(-depth_factor / thermal_resistance_ratio)
                    
                    # Apply CryoGrid-style energy balance for snow effects
                    if snow_melt > 0:
                        # Melt energy following CryoGrid latent heat treatment
                        melt_energy = snow_melt * self.LVOL_SL / 1000  # J/m2
                        soil_heat_capacity = self._calculate_heat_capacity(soil_props)
                        melt_temp_effect = melt_energy / (soil_heat_capacity * 0.1)
                        modified_temps[i] += min(melt_temp_effect / 10, 2.0)
                    
                    # Enhanced zero-curtain promotion under thick snow
                    if snow_depth > 0.3 and abs(modified_temps[i]) < 2.0:
                        # CryoGrid-style thermal buffering
                        buffering_strength = 0.5 * (1 - insulation_factor)
                        modified_temps[i] *= (1 - buffering_strength)
                    
                    # Apply thermal damping for temperature variations
                    if i > 0:
                        temp_change = temp - temperatures[i-1]
                        dampened_change = temp_change * insulation_factor
                        modified_temps[i] = temperatures[i-1] + dampened_change
        
        return modified_temps
    
    def _estimate_snow_density(self, snow_depth, snow_swe):
        """Estimate snow density from depth and SWE."""
        if snow_depth > 0 and snow_swe > 0:
            return (snow_swe * 1000) / (snow_depth * 1000)  # kg/m3
        else:
            return 300  # Default fresh snow density
    
    def _calculate_snow_thermal_conductivity_cryogrid(self, snow_density):
        """
        Calculate snow thermal conductivity using CryoGrid parameterizations.
        Implements both Yen (1981) and Sturm et al. (1997) formulations.
        """
        
        # Yen (1981) exponential relationship
        k_yen = 0.138 - 1.01e-3 * snow_density + 3.233e-6 * snow_density**2
        
        # Sturm et al. (1997) quadratic relationship
        k_sturm = 0.138 - 1.01e-3 * snow_density + 3.233e-6 * snow_density**2
        
        # Use Sturm formulation as default (more suitable for Arctic conditions)
        k_snow = max(0.05, min(k_sturm, 0.8))  # Reasonable bounds
        
        return k_snow
    
    def _identify_zero_curtain_physics_enhanced(self, temperatures, timestamps, stefan_solution,
                                           permafrost_props, snow_props, depth_zone, soil_props):
        """
        Enhanced zero-curtain identification using CryoGrid physics integration.
        With comprehensive diagnostics to identify detection issues.
        """
        
        events = []
        n = len(temperatures)
        
        # COMPREHENSIVE DIAGNOSTIC OUTPUT
        print(f"\n--- ZERO-CURTAIN DETECTION DIAGNOSTIC ---")
        print(f"Site depth zone: {depth_zone}")
        print(f"Temperature data: n={n} points")
        print(f"  Range: [{np.min(temperatures):.3f}, {np.max(temperatures):.3f}]°C")
        print(f"  Mean: {np.mean(temperatures):.3f}°C, Std: {np.std(temperatures):.3f}°C")
        print(f"  Median: {np.median(temperatures):.3f}°C")
        print(f"Permafrost probability: {permafrost_props.get('permafrost_prob', 'None')}")
        print(f"Permafrost zone: {permafrost_props.get('permafrost_zone', 'None')}")
        print(f"Snow data available: {snow_props.get('has_snow_data', False)}")
        if snow_props.get('has_snow_data', False):
            print(f"  Snow depth range: {np.min(snow_props.get('snow_depth', [0])):.2f} - {np.max(snow_props.get('snow_depth', [0])):.2f} cm")
        
        # Check minimum duration requirement
        if n < self.MIN_DURATION_HOURS:
            print(f"REJECTED: Insufficient data ({n} < {self.MIN_DURATION_HOURS} required)")
            print("--- END DIAGNOSTIC ---\n")
            return events
        
        # Enhanced physics-based criteria with detailed analysis
        phase_energy = stefan_solution['phase_change_energy']
        freeze_depths = stefan_solution['freeze_depths']
        
        print(f"Stefan solution:")
        print(f"  Phase change energy range: [{np.min(phase_energy):.3e}, {np.max(phase_energy):.3e}] J/m³")
        print(f"  Freeze depths range: [{np.min(freeze_depths):.3f}, {np.max(freeze_depths):.3f}] m")
        
        # Temperature criteria analysis
        temp_criteria = np.abs(temperatures) <= self.TEMP_THRESHOLD
        temp_matches = np.sum(temp_criteria)
        temp_percentage = (temp_matches / n) * 100
        
        print(f"Temperature criteria (±{self.TEMP_THRESHOLD}°C):")
        print(f"  Matches: {temp_matches}/{n} ({temp_percentage:.1f}%)")
        
        # Additional temperature analysis
        near_zero_1 = np.sum(np.abs(temperatures) <= 1.0)
        near_zero_2 = np.sum(np.abs(temperatures) <= 2.0)
        near_zero_05 = np.sum(np.abs(temperatures) <= 0.5)
        
        print(f"  Within ±0.5°C: {near_zero_05}/{n} ({(near_zero_05/n)*100:.1f}%)")
        print(f"  Within ±1.0°C: {near_zero_1}/{n} ({(near_zero_1/n)*100:.1f}%)")
        print(f"  Within ±2.0°C: {near_zero_2}/{n} ({(near_zero_2/n)*100:.1f}%)")
        
        # Enhanced energy criteria using CryoGrid formulations
        if self.use_cryogrid_enthalpy and 'enthalpy_profile' in stefan_solution:
            enthalpy_profile = stefan_solution['enthalpy_profile']
            print(f"Using CryoGrid enthalpy analysis:")
            print(f"  Enthalpy profile shape: {enthalpy_profile.shape}")
            
            # Detect enthalpy plateaus (zero-curtain signature) with overflow protection
            enthalpy_mean = enthalpy_profile.mean(axis=1)
            # Clean mathematical invalidity before gradient calculation
            enthalpy_mean_clean = np.nan_to_num(enthalpy_mean, nan=0.0, posinf=1e15, neginf=-1e15)
            enthalpy_gradient = np.gradient(enthalpy_mean_clean)
            
            # Additional check for valid gradient values
            if np.any(~np.isfinite(enthalpy_gradient)):
                print(f"  Warning: Non-finite enthalpy gradients detected, cleaning...")
                enthalpy_gradient = np.nan_to_num(enthalpy_gradient, nan=0.0, posinf=1e15, neginf=-1e15)
            
            energy_criteria = np.abs(enthalpy_gradient) <= 0.1 * self.MAX_ENTHALPY_CHANGE
            energy_matches = np.sum(energy_criteria)
            
            # Safe min/max calculation
            if len(enthalpy_gradient) > 0 and np.any(np.isfinite(enthalpy_gradient)):
                grad_min = np.min(enthalpy_gradient[np.isfinite(enthalpy_gradient)])
                grad_max = np.max(enthalpy_gradient[np.isfinite(enthalpy_gradient)])
                print(f"  Enthalpy gradient range: [{grad_min:.2e}, {grad_max:.2e}] J/m³")
            else:
                print(f"  Enthalpy gradient range: [invalid data] J/m³")
                
            print(f"  Enthalpy threshold: {0.1 * self.MAX_ENTHALPY_CHANGE:.2e} J/m³")
            print(f"  Energy criteria matches: {energy_matches}/{len(energy_criteria)} ({(energy_matches/len(energy_criteria))*100:.1f}%)")
        else:
            print(f"Using traditional phase change energy analysis:")
            energy_criteria = phase_energy > self.PHASE_CHANGE_ENERGY
            energy_matches = np.sum(energy_criteria)
            print(f"  Energy threshold: {self.PHASE_CHANGE_ENERGY}")
            print(f"  Energy criteria matches: {energy_matches}/{len(energy_criteria)} ({(energy_matches/len(energy_criteria))*100:.1f}%)")
        
        # Enhanced thermal gradient analysis
        temp_gradient = np.gradient(temperatures)
        gradient_criteria = np.abs(temp_gradient) <= self.GRADIENT_THRESHOLD
        gradient_matches = np.sum(gradient_criteria)
        
        print(f"Thermal gradient analysis:")
        print(f"  Gradient range: [{np.min(temp_gradient):.4f}, {np.max(temp_gradient):.4f}] °C/day")
        print(f"  Gradient threshold: ±{self.GRADIENT_THRESHOLD} °C/day")
        print(f"  Gradient criteria matches: {gradient_matches}/{n} ({(gradient_matches/n)*100:.1f}%)")
        
        # Alternative gradient thresholds
        grad_005 = np.sum(np.abs(temp_gradient) <= 0.05)
        grad_01 = np.sum(np.abs(temp_gradient) <= 0.1)
        print(f"  Within ±0.05°C/day: {grad_005}/{n} ({(grad_005/n)*100:.1f}%)")
        print(f"  Within ±0.10°C/day: {grad_01}/{n} ({(grad_01/n)*100:.1f}%)")
        
        # Permafrost-informed criteria
        pf_prob = permafrost_props.get('permafrost_prob', 0)
        pf_enhancement = 1.0 + 0.5 * pf_prob if pf_prob else 1.0
        
        print(f"Permafrost enhancement factor: {pf_enhancement:.2f}")
        
        # Snow-informed criteria
        snow_enhancement = self._calculate_snow_insulation(snow_props)
        print(f"Snow insulation factor: {snow_enhancement:.3f}")
        
        # Multiple detection pathways instead of restrictive AND logic

        # Pathway 1: Standard criteria (original approach)
        standard_zc_mask = temp_criteria & gradient_criteria
        if len(energy_criteria) == len(temp_criteria):
            standard_zc_mask = standard_zc_mask & energy_criteria

        # Pathway 2: Relaxed temperature criteria
        relaxed_temp_criteria = np.abs(temperatures) <= self.RELAXED_TEMP_THRESHOLD
        relaxed_gradient_criteria = np.abs(temp_gradient) <= self.RELAXED_GRADIENT_THRESHOLD
        relaxed_zc_mask = relaxed_temp_criteria & relaxed_gradient_criteria

        # Pathway 3: Temperature-only pathway (for very stable thermal regimes)
        temp_only_criteria = np.abs(temperatures) <= 1.0  # Within ±1°C
        temp_stability = np.abs(temp_gradient) <= 0.1     # Very low gradients
        temp_only_mask = temp_only_criteria & temp_stability

        # Pathway 4: Isothermal plateau detection (zero-curtain signature)
        temp_variance_window = 24  # 24-point rolling window
        isothermal_mask = np.zeros_like(temperatures, dtype=bool)
        for i in range(len(temperatures) - temp_variance_window):
            window_temps = temperatures[i:i+temp_variance_window]
            if np.std(window_temps) < 0.5 and np.abs(np.mean(window_temps)) < 2.0:
                isothermal_mask[i:i+temp_variance_window] = True

        # COMBINE ALL PATHWAYS - if ANY pathway detects zero-curtain, include it
        enhanced_zc_mask = standard_zc_mask | relaxed_zc_mask | temp_only_mask | isothermal_mask

        print(f"Multi-pathway detection results:")
        print(f"  Standard pathway: {np.sum(standard_zc_mask)}/{len(standard_zc_mask)} ({(np.sum(standard_zc_mask)/len(standard_zc_mask))*100:.1f}%)")
        print(f"  Relaxed pathway: {np.sum(relaxed_zc_mask)}/{len(relaxed_zc_mask)} ({(np.sum(relaxed_zc_mask)/len(relaxed_zc_mask))*100:.1f}%)")
        print(f"  Temperature-only pathway: {np.sum(temp_only_mask)}/{len(temp_only_mask)} ({(np.sum(temp_only_mask)/len(temp_only_mask))*100:.1f}%)")
        print(f"  Isothermal plateau pathway: {np.sum(isothermal_mask)}/{len(isothermal_mask)} ({(np.sum(isothermal_mask)/len(isothermal_mask))*100:.1f}%)")
        print(f"  Combined enhanced detection: {np.sum(enhanced_zc_mask)}/{len(enhanced_zc_mask)} ({(np.sum(enhanced_zc_mask)/len(enhanced_zc_mask))*100:.1f}%)")
        
        enhanced_matches = np.sum(enhanced_zc_mask)
        print(f"Enhanced criteria: {enhanced_matches}/{n} ({(enhanced_matches/n)*100:.1f}%)")
        
        # Apply additional enhancements based on permafrost and snow context
        enhancement_applied = 0
        original_enhanced_mask = enhanced_zc_mask.copy()
        
        for i in range(len(enhanced_zc_mask)):
            if not enhanced_zc_mask[i]:  # Only enhance points not already detected
                # Check if conditions warrant enhancement
                enhancement_factor = pf_enhancement * (1 + snow_enhancement)
                if enhancement_factor > 1.2:  # Significant enhancement threshold
                    # Apply lenient criteria for enhancement
                    if abs(temperatures[i]) <= self.RELAXED_TEMP_THRESHOLD:
                        enhanced_zc_mask[i] = True
                        enhancement_applied += 1
        
        final_enhanced_matches = np.sum(enhanced_zc_mask)
        print(f"Final enhanced criteria: {final_enhanced_matches}/{n} ({(final_enhanced_matches/n)*100:.1f}%)")
        print(f"Enhancement applied to: {enhancement_applied} additional points")
        
        # Continuity analysis for debugging
        if enhanced_matches > 0:
            print(f"Analyzing continuity patterns:")
            # Find where enhanced criteria are True
            true_indices = np.where(enhanced_zc_mask)[0]
            if len(true_indices) > 1:
                gaps = np.diff(true_indices)
                max_continuous = 1
                current_continuous = 1
                for gap in gaps:
                    if gap == 1:  # Consecutive
                        current_continuous += 1
                        max_continuous = max(max_continuous, current_continuous)
                    else:
                        current_continuous = 1
                print(f"  Maximum consecutive points meeting criteria: {max_continuous}")
                print(f"  Required consecutive points: {self.MIN_DURATION_HOURS}")
                print(f"  Average gap between criteria matches: {np.mean(gaps):.1f} points")
        
        # Alternative detection with relaxed thresholds for diagnostic
        relaxed_temp = np.abs(temperatures) <= 1.0  # More lenient temperature
        relaxed_grad = np.abs(temp_gradient) <= 0.05  # More lenient gradient
        relaxed_combined = relaxed_temp & relaxed_grad
        relaxed_matches = np.sum(relaxed_combined)
        print(f"Relaxed criteria (±1°C, ±0.05°C/day): {relaxed_matches}/{n} ({(relaxed_matches/n)*100:.1f}%)")
        
        # Find continuous periods with adaptive duration
        if enhanced_matches > 0:
            # Adaptive minimum duration based on data characteristics
            data_length = len(enhanced_zc_mask)
            data_density = data_length / 365 if data_length > 365 else data_length / 30  # Daily or sub-daily

            if data_density >= 1.0:  # Daily or better resolution
                adaptive_min_duration = max(6, int(self.RELAXED_MIN_DURATION))
            elif data_density >= 0.5:  # Every other day
                adaptive_min_duration = max(3, int(self.RELAXED_MIN_DURATION * 0.5))
            else:  # Weekly or coarser
                adaptive_min_duration = max(1, int(self.RELAXED_MIN_DURATION * 0.25))

            print(f"Using adaptive minimum duration: {adaptive_min_duration} points (data density: {data_density:.2f})")

            # Multiple duration thresholds for different pathway strengths
            # Primary detection with adaptive duration
            zc_periods_primary = self._find_continuous_periods(enhanced_zc_mask, adaptive_min_duration)
            
            # Secondary detection with even more relaxed duration
            zc_periods_secondary = self._find_continuous_periods(enhanced_zc_mask, max(1, adaptive_min_duration // 2))
            
            # Combine periods, prioritizing longer ones
            all_periods = list(set(zc_periods_primary + zc_periods_secondary))
            zc_periods = sorted(all_periods, key=lambda x: x[1] - x[0], reverse=True)
            
            print(f"Period detection results:")
            print(f"  Primary periods (min duration {adaptive_min_duration}): {len(zc_periods_primary)}")
            print(f"  Secondary periods (min duration {max(1, adaptive_min_duration // 2)}): {len(zc_periods_secondary)}")
            print(f"  Total unique periods: {len(zc_periods)}")
            
            for i, (start_idx, end_idx) in enumerate(zc_periods):
                period_length = end_idx - start_idx + 1
                print(f"  Period {i+1}: {period_length} points (indices {start_idx}-{end_idx})")
        else:
            zc_periods = []
            print("No continuous periods found with current criteria")
        
        # Alternative analysis with relaxed criteria
        if len(zc_periods) == 0 and relaxed_matches > 0:
            print("Trying relaxed criteria for period detection...")
            relaxed_periods = self._find_continuous_periods(relaxed_zc_mask, max(12, self.MIN_DURATION_HOURS // 2))
            print(f"Relaxed continuous periods found: {len(relaxed_periods)}")
            
            for i, (start_idx, end_idx) in enumerate(relaxed_periods):
                period_length = end_idx - start_idx + 1
                print(f"  Relaxed period {i+1}: {period_length} points (indices {start_idx}-{end_idx})")
                temp_subset = temperatures[start_idx:end_idx+1]
                print(f"    Temperature range: [{np.min(temp_subset):.3f}, {np.max(temp_subset):.3f}]°C")
                print(f"    Mean temperature: {np.mean(temp_subset):.3f}°C")
        
        # Characterize detected events
        for start_idx, end_idx in zc_periods:
            event = self._characterize_physics_informed_event_enhanced(
                temperatures[start_idx:end_idx+1],
                timestamps[start_idx:end_idx+1],
                stefan_solution,
                permafrost_props,
                snow_props,
                depth_zone,
                soil_props,
                start_idx,
                end_idx
            )
            events.append(event)
            print(f"Event characterized: duration={event['duration_hours']:.1f}h, intensity={event['intensity_percentile']:.3f}")
        
        # If no events detected with standard approaches, try fallback detection
        if len(events) == 0:
            print("No events detected with standard criteria, applying fallback detection...")
            fallback_events = self._fallback_zero_curtain_detection(temperatures, timestamps, depth_zone)
            events.extend(fallback_events)
            
            if len(fallback_events) > 0:
                print(f"Fallback detection successful: {len(fallback_events)} events found")
            else:
                print("Even fallback detection found no events - site may lack zero-curtain signature")

        # Summary with enhanced diagnostic information
        total_events = len(events)
        if total_events > 0:
            print(f"FINAL DETECTION SUMMARY: {total_events} zero-curtain events")
            for i, event in enumerate(events):
                method = event.get('detection_method', 'standard')
                print(f"  Event {i+1}: {event['duration_hours']:.1f}h duration, method={method}")
        else:
            print("FINAL RESULT: No zero-curtain events detected")
            
            # Enhanced diagnostic for failed detection
            temp_range = np.max(temperatures) - np.min(temperatures)
            temp_near_zero = np.sum(np.abs(temperatures) <= 1.0) / len(temperatures)
            print(f"Site thermal characteristics:")
            print(f"  Temperature range: {temp_range:.2f}°C")
            print(f"  Time near zero (±1°C): {temp_near_zero*100:.1f}%")
            print(f"  Mean temperature: {np.mean(temperatures):.2f}°C")
            print(f"  Temperature std dev: {np.std(temperatures):.2f}°C")

        print("--- END DIAGNOSTIC ---\n")

        return events
    
    def _characterize_physics_informed_event_enhanced(self, temps, times, stefan_solution,
                                                     permafrost_props, snow_props, depth_zone,
                                                     soil_props, start_idx, end_idx):
        """
        Enhanced zero-curtain event characterization with CryoGrid physics.
        """
        
        duration_hours = len(times) * 24.0  # Assuming daily data
        
        # 1. Enhanced intensity calculation with CryoGrid formulations
        intensity = self._calculate_physics_intensity_enhanced(
            temps, stefan_solution, permafrost_props, snow_props, depth_zone, soil_props
        )
        
        # 2. Enhanced spatial extent with CryoGrid thermal diffusion
        spatial_extent = self._calculate_physics_spatial_extent_enhanced(
            stefan_solution, duration_hours, intensity, permafrost_props, soil_props
        )
        
        # 3. CryoGrid-specific thermal characteristics
        thermal_characteristics = self._calculate_cryogrid_thermal_characteristics(
            temps, stefan_solution, soil_props
        )
        
        event = {
            'start_time': times[0],
            'end_time': times[-1],
            'duration_hours': duration_hours,
            'intensity_percentile': intensity,
            'spatial_extent_meters': spatial_extent,
            'depth_zone': depth_zone,
            'mean_temperature': np.mean(temps),
            'temperature_variance': np.var(temps),
            'permafrost_probability': permafrost_props['permafrost_prob'],
            'permafrost_zone': permafrost_props['permafrost_zone'],
            'phase_change_energy': np.mean(stefan_solution['phase_change_energy'][start_idx:end_idx+1]),
            'freeze_penetration_depth': np.mean(stefan_solution['freeze_depths'][start_idx:end_idx+1]),
            'thermal_diffusivity': self._calculate_effective_diffusivity(permafrost_props),
            'snow_insulation_factor': self._calculate_snow_insulation(snow_props),
            
            # CryoGrid-enhanced characteristics
            'cryogrid_thermal_conductivity': thermal_characteristics['thermal_conductivity'],
            'cryogrid_heat_capacity': thermal_characteristics['heat_capacity'],
            'cryogrid_enthalpy_stability': thermal_characteristics.get('enthalpy_stability', 0),
            'surface_energy_balance': thermal_characteristics.get('surface_energy_balance', 0),
            'lateral_thermal_effects': thermal_characteristics.get('lateral_effects', 0),
            'soil_freezing_characteristic': 'painter_karra' if self.use_painter_karra_freezing else 'free_water',
            'adaptive_timestep_used': self.use_adaptive_timestep,
            'van_genuchten_alpha': soil_props.get('van_genuchten_alpha', 0.5),
            'van_genuchten_n': soil_props.get('van_genuchten_n', 2.0)
        }
        
        return event
    
    def _calculate_physics_intensity_enhanced(self, temps, stefan_solution, permafrost_props,
                                        snow_props, depth_zone, soil_props):
        """
        Enhanced intensity calculation with CryoGrid physics and comprehensive safety checks.
        
        Calculates zero-curtain intensity based on multiple physics-informed criteria:
        1. Thermal stability (isothermal behavior)
        2. Phase change energy signature
        3. CryoGrid enthalpy stability
        4. Permafrost context influence
        5. Snow insulation effects
        6. Stefan problem consistency
        7. Soil-specific enhancement factors
        8. Depth-dependent weighting
        
        Returns intensity score [0.0, 1.0] where 1.0 = maximum zero-curtain signature
        """
        
        # Safety check for input data
        if len(temps) == 0:
            print(f"Warning: Empty temperature array in intensity calculation")
            return 0.1  # Minimal intensity for empty data
        
        # 1. THERMAL STABILITY - isothermal behavior around 0°C
        try:
            temp_variance = np.var(temps)
            if temp_variance == 0:
                temp_stability = 1.0  # Perfect stability
            else:
                temp_stability = np.exp(-temp_variance * 20)
                temp_stability = np.clip(temp_stability, 0.0, 1.0)
        except Exception as e:
            print(f"Warning: Error calculating thermal stability: {e}")
            temp_stability = 0.5
        
        # 2. PHASE CHANGE ENERGY INTENSITY
        try:
            if 'phase_change_energy' in stefan_solution and len(stefan_solution['phase_change_energy']) > 0:
                phase_energies = stefan_solution['phase_change_energy']
                # Filter out extreme values that might be numerical artifacts
                valid_energies = phase_energies[np.isfinite(phase_energies)]
                
                if len(valid_energies) > 0:
                    mean_phase_energy = np.mean(valid_energies)
                    # Normalize by latent heat with safety bounds
                    if self.LVOL_SL > 0:
                        energy_intensity = np.tanh(mean_phase_energy / self.LVOL_SL)
                        energy_intensity = np.clip(energy_intensity, 0.0, 1.0)
                    else:
                        energy_intensity = 0.5
                else:
                    energy_intensity = 0.3  # Some energy signature assumed
            else:
                energy_intensity = 0.3  # Default for missing energy data
        except Exception as e:
            print(f"Warning: Error calculating energy intensity: {e}")
            energy_intensity = 0.3
        
        # 3. CRYOGRID ENTHALPY STABILITY
        try:
            enthalpy_stability = 1.0  # Default high stability
            if 'enthalpy_profile' in stefan_solution:
                enthalpy_profile = stefan_solution['enthalpy_profile']
                
                if enthalpy_profile is not None and enthalpy_profile.size > 0:
                    # Calculate variance across the enthalpy profile
                    enthalpy_flat = enthalpy_profile.flatten()
                    valid_enthalpy = enthalpy_flat[np.isfinite(enthalpy_flat)]
                    
                    if len(valid_enthalpy) > 1:
                        enthalpy_variance = np.var(valid_enthalpy)
                        if self.MAX_ENTHALPY_CHANGE > 0 and enthalpy_variance > 0:
                            enthalpy_stability = np.exp(-enthalpy_variance / self.MAX_ENTHALPY_CHANGE)
                            enthalpy_stability = np.clip(enthalpy_stability, 0.0, 1.0)
        except Exception as e:
            print(f"Warning: Error calculating enthalpy stability: {e}")
            enthalpy_stability = 1.0
        
        # 4. PERMAFROST INFLUENCE
        try:
            pf_prob = permafrost_props.get('permafrost_prob', 0)
            if pf_prob is not None and pf_prob >= 0:
                pf_intensity = min(pf_prob, 1.0)  # Ensure within [0,1]
            else:
                pf_intensity = 0.0
        except Exception as e:
            print(f"Warning: Error calculating permafrost intensity: {e}")
            pf_intensity = 0.0
        
        # 5. ENHANCED SNOW INSULATION
        try:
            snow_intensity = self._calculate_snow_insulation_enhanced(snow_props, soil_props)
            snow_intensity = np.clip(snow_intensity, 0.0, 1.0)
        except Exception as e:
            print(f"Warning: Error calculating snow intensity: {e}")
            # Fallback to basic snow calculation
            try:
                snow_intensity = self._calculate_snow_insulation(snow_props)
                snow_intensity = np.clip(snow_intensity, 0.0, 1.0)
            except:
                snow_intensity = 0.1  # Minimal snow effect
        
        # 6. SOIL-SPECIFIC ENHANCEMENT
        try:
            soil_enhancement = self._calculate_soil_enhancement_factor(soil_props)
            soil_enhancement = np.clip(soil_enhancement, 0.0, 1.0)
        except Exception as e:
            print(f"Warning: Error calculating soil enhancement: {e}")
            soil_enhancement = 0.5  # Neutral soil effect
        
        # 7. DEPTH-DEPENDENT WEIGHTING
        try:
            depth_weights = {
                'surface': 0.7, 'shallow': 0.8, 'intermediate': 1.0,
                'deep': 1.2, 'very_deep': 1.4
            }
            depth_factor = depth_weights.get(depth_zone, 1.0)
            # Normalize depth factor to [0,1] range
            normalized_depth_factor = depth_factor / 1.4
        except Exception as e:
            print(f"Warning: Error calculating depth factor: {e}")
            normalized_depth_factor = 0.7  # Default intermediate depth
        
        # 8. STEFAN PROBLEM FREEZE CONSISTENCY
        try:
            freeze_consistency = 1.0  # Default high consistency
            
            if ('freeze_depths' in stefan_solution and
                stefan_solution['freeze_depths'] is not None and
                len(stefan_solution['freeze_depths']) > 1):
                
                freeze_depths = stefan_solution['freeze_depths']
                valid_depths = freeze_depths[np.isfinite(freeze_depths)]
                
                if len(valid_depths) > 1:
                    freeze_std = np.std(valid_depths)
                    freeze_mean = np.mean(valid_depths)
                elif len(valid_depths) == 1:
                    # Single depth value - perfect consistency
                    freeze_consistency = 1.0
                    freeze_std = 0.0
                    freeze_mean = valid_depths[0]
                else:
                    # No valid depths
                    freeze_consistency = 0.5
                    freeze_std = 0.0
                    freeze_mean = 1.0  # Default depth
                
                if len(valid_depths) > 1:
                    freeze_std = np.std(valid_depths)
                    freeze_mean = np.mean(valid_depths)
                    
                    if freeze_mean > 1e-6:  # Avoid division by zero
                        freeze_consistency = 1.0 - min(freeze_std / freeze_mean, 1.0)
                        freeze_consistency = max(freeze_consistency, 0.0)
                else:
                    freeze_consistency = 0.8  # Good consistency for single/uniform depth
                    
        except Exception as e:
            print(f"Warning: Error calculating freeze consistency: {e}")
            freeze_consistency = 0.8
        
        # 9. ENHANCED WEIGHTED COMBINATION with error handling
        try:
            # Ensure all components are finite and within bounds
            components = {
                'temp_stability': temp_stability,
                'energy_intensity': energy_intensity,
                'enthalpy_stability': enthalpy_stability,
                'pf_intensity': pf_intensity,
                'snow_intensity': snow_intensity,
                'freeze_consistency': freeze_consistency,
                'soil_enhancement': soil_enhancement,
                'depth_factor': normalized_depth_factor
            }
            
            # Validate all components
            for name, value in components.items():
                if not np.isfinite(value):
                    print(f"Warning: Non-finite {name}: {value}, setting to 0.5")
                    components[name] = 0.5
                elif value < 0 or value > 1:
                    print(f"Warning: {name} out of bounds: {value}, clipping to [0,1]")
                    components[name] = np.clip(value, 0.0, 1.0)
            
            # Calculate weighted intensity
            intensity = (
                0.20 * components['temp_stability'] +      # Isothermal behavior
                0.15 * components['energy_intensity'] +    # Phase change energy
                0.15 * components['enthalpy_stability'] +  # CryoGrid enthalpy stability
                0.12 * components['pf_intensity'] +        # Permafrost context
                0.12 * components['snow_intensity'] +      # Enhanced snow insulation
                0.10 * components['freeze_consistency'] +  # Stefan solution consistency
                0.08 * components['soil_enhancement'] +    # Soil-specific factors
                0.08 * components['depth_factor']          # Depth significance
            )
            
            # Final safety checks and bounds
            if not np.isfinite(intensity):
                print(f"Warning: Non-finite intensity calculated, using default 0.5")
                intensity = 0.5
            
            intensity = np.clip(intensity, 0.0, 1.0)
            
            # Debug output for problematic cases
            if intensity < 0.1:
                print(f"Low intensity warning ({intensity:.3f}): Components = {components}")
            
        except Exception as e:
            print(f"Error in intensity calculation: {e}")
            print(f"Using fallback intensity calculation")
            
            # Fallback calculation using only basic metrics
            try:
                basic_temp_score = 1.0 - min(np.std(temps) / 10.0, 1.0)  # Temperature stability
                basic_zero_proximity = 1.0 - min(abs(np.mean(temps)) / 5.0, 1.0)  # Proximity to 0°C
                intensity = 0.5 * basic_temp_score + 0.5 * basic_zero_proximity
                intensity = np.clip(intensity, 0.1, 1.0)  # Ensure reasonable bounds
            except:
                intensity = 0.3  # Last resort default
        
        return float(intensity)
    
    def _calculate_snow_insulation_enhanced(self, snow_props, soil_props):
        """Enhanced snow insulation calculation with CryoGrid thermal properties."""
        
        base_insulation = self._calculate_snow_insulation(snow_props)
        
        if not snow_props['has_snow_data']:
            return base_insulation
        
        # CryoGrid-enhanced calculation
        if len(snow_props['snow_depth']) > 0:
            mean_depth = np.mean(snow_props['snow_depth'][snow_props['snow_depth'] > 0]) / 100.0
            
            # Calculate thermal resistance enhancement
            snow_density = self._estimate_snow_density(mean_depth,
                np.mean(snow_props.get('snow_water_equiv', [30])))
            snow_k = self._calculate_snow_thermal_conductivity_cryogrid(snow_density)
            soil_k = self._calculate_thermal_conductivity(soil_props)
            
            # Thermal resistance ratio
            resistance_ratio = soil_k / snow_k if snow_k > 0 else 1
            thermal_enhancement = np.tanh(resistance_ratio / 10.0)
            
            enhanced_insulation = base_insulation * (1 + 0.5 * thermal_enhancement)
            return min(enhanced_insulation, 1.0)
        
        return base_insulation
    
    def _calculate_soil_enhancement_factor(self, soil_props):
        """Calculate soil-specific enhancement factor based on CryoGrid parameters."""
        
        # Van Genuchten parameters influence
        alpha = soil_props.get('van_genuchten_alpha', 0.5)
        n = soil_props.get('van_genuchten_n', 2.0)
        
        # Lower alpha (finer soil) enhances zero-curtain formation
        alpha_factor = np.exp(-alpha * 2)  # Range 0-1
        
        # Higher n (more uniform pore size) enhances zero-curtain
        n_factor = np.tanh((n - 1.5) / 2.0)  # Range 0-1
        
        # Organic content enhancement
        organic_fraction = soil_props.get('organic_fraction', 0.1)
        organic_factor = np.tanh(organic_fraction * 5)  # Range 0-1
        
        # Combined soil enhancement
        soil_enhancement = 0.4 * alpha_factor + 0.3 * n_factor + 0.3 * organic_factor
        
        return np.clip(soil_enhancement, 0.0, 1.0)
    
    def _calculate_physics_spatial_extent_enhanced(self, stefan_solution, duration_hours,
                                                  intensity, permafrost_props, soil_props):
        """Enhanced spatial extent calculation with CryoGrid thermal diffusion."""
        
        # Original Stefan solution component
        mean_freeze_depth = np.mean(stefan_solution['freeze_depths'])
        
        # Enhanced thermal diffusion calculation using CryoGrid formulations
        thermal_conductivity = self._calculate_thermal_conductivity(soil_props)
        heat_capacity = self._calculate_heat_capacity(soil_props)
        effective_diffusivity = thermal_conductivity / heat_capacity
        
        duration_seconds = duration_hours * 3600
        diffusion_depth = np.sqrt(4 * effective_diffusivity * duration_seconds)
        
        # CryoGrid-enhanced combining weights
        stefan_weight = 0.6
        diffusion_weight = 0.4
        
        # Account for soil-specific thermal properties
        soil_enhancement = self._calculate_soil_enhancement_factor(soil_props)
        thermal_enhancement = 1.0 + 0.3 * soil_enhancement
        
        # Enhanced spatial extent calculation
        spatial_extent = (stefan_weight * mean_freeze_depth +
                         diffusion_weight * diffusion_depth) * thermal_enhancement
        
        # Intensity and permafrost modulation
        spatial_extent *= (0.5 + 0.5 * intensity)
        
        pf_prob = permafrost_props.get('permafrost_prob', 0)
        if pf_prob:
            pf_factor = 1.0 + 0.5 * pf_prob
            spatial_extent *= pf_factor
        
        # CryoGrid-informed physical bounds
        min_extent = 0.05  # 5 cm minimum
        max_extent = 5.0   # 5 m maximum (increased for enhanced model)
        
        return np.clip(spatial_extent, min_extent, max_extent)
    
    def _calculate_cryogrid_thermal_characteristics(self, temps, stefan_solution, soil_props):
        """Calculate CryoGrid-specific thermal characteristics for event characterization."""
        
        characteristics = {}
        
        # Thermal conductivity using CryoGrid formulations
        characteristics['thermal_conductivity'] = self._calculate_thermal_conductivity(soil_props)
        
        # Heat capacity using CryoGrid temperature-dependent formulation
        mean_temp = np.mean(temps)
        characteristics['heat_capacity'] = self._calculate_effective_heat_capacity_cryogrid(
            soil_props, mean_temp
        )
        
        # Enthalpy stability (if available)
        if 'enthalpy_profile' in stefan_solution:
            enthalpy_variance = np.var(stefan_solution['enthalpy_profile'])
            characteristics['enthalpy_stability'] = np.exp(-enthalpy_variance / self.MAX_ENTHALPY_CHANGE)
        
        # Surface energy balance significance (simplified)
        if self.use_surface_energy_balance:
            characteristics['surface_energy_balance'] = 1.0
        else:
            characteristics['surface_energy_balance'] = 0.0
        
        # Lateral thermal effects significance
        characteristics['lateral_effects'] = 0.5  # Placeholder - would be calculated from actual lateral interactions
        
        return characteristics
    
    def _calculate_effective_diffusivity(self, permafrost_props):
        """Calculate effective thermal diffusivity based on permafrost properties."""
        
        # Base diffusivity
        base_alpha = 5e-7  # m2/s
        
        # Permafrost enhancement
        pf_prob = permafrost_props.get('permafrost_prob', 0)
        pf_enhancement = 1.0 + pf_prob * 0.5 if pf_prob else 1.0
        
        return base_alpha * pf_enhancement
    
    def _calculate_snow_insulation(self, snow_props):
        """
        Calculate spatiotemporal snow insulation factor.
        Uses time-specific snow depth, SWE, and melt conditions.
        """
        
        if not snow_props['has_snow_data'] or len(snow_props['snow_depth']) == 0:
            return 0.0
        
        # Time-averaged snow characteristics
        valid_depths = snow_props['snow_depth'][snow_props['snow_depth'] > 0]
        if len(valid_depths) == 0:
            return 0.0
            
        mean_depth = np.mean(valid_depths) / 100.0  # cm to m
        max_depth = np.max(snow_props['snow_depth']) / 100.0
        
        # Snow persistence factor
        snow_days = np.sum(snow_props['snow_depth'] > 1.0)  # Days with >1cm snow
        total_days = len(snow_props['snow_depth'])
        persistence = snow_days / total_days if total_days > 0 else 0
        
        # SWE-based insulation quality
        if len(snow_props['snow_water_equiv']) > 0:
            valid_swe = snow_props['snow_water_equiv'][snow_props['snow_water_equiv'] > 0]
            if len(valid_swe) > 0:
                mean_swe = np.mean(valid_swe)
                swe_factor = np.tanh(mean_swe / 100.0)  # Normalize to 0-1
            else:
                swe_factor = 0.5
        else:
            swe_factor = 0.5
        
        # Combined insulation factor
        depth_factor = np.tanh(mean_depth / 0.5)  # Saturation at 50cm
        max_depth_factor = np.tanh(max_depth / 1.0)  # Maximum depth contribution
        
        insulation = (0.4 * depth_factor +
                     0.3 * persistence +
                     0.2 * swe_factor +
                     0.1 * max_depth_factor)
        
        return np.clip(insulation, 0.0, 1.0)
    
    def _find_continuous_periods(self, mask, min_length):
        """Find continuous True periods in boolean mask."""
        periods = []
        start = None
        
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                if i - start >= min_length:
                    periods.append((start, i-1))
                start = None
        
        if start is not None and len(mask) - start >= min_length:
            periods.append((start, len(mask)-1))
        
        return periods
        
    def _fallback_zero_curtain_detection(self, temperatures, timestamps, depth_zone):
        """
        Fallback detection for sites that show minimal zero-curtain signatures.
        Uses very permissive criteria to capture weak or brief events.
        """
        
        fallback_events = []
        
        # Criterion 1: Any sustained near-zero temperatures
        near_zero_mask = np.abs(temperatures) <= 2.5  # ±2.5°C
        near_zero_periods = self._find_continuous_periods(near_zero_mask, 3)  # Just 3 points minimum
        
        # Criterion 2: Low thermal variance periods (isothermal-like)
        variance_threshold = np.percentile(np.abs(np.gradient(temperatures)), 25)  # Bottom quartile
        low_variance_mask = np.abs(np.gradient(temperatures)) <= variance_threshold
        if len(low_variance_mask) == len(temperatures) - 1:
            low_variance_mask = np.append(low_variance_mask, False)  # Match length
        low_variance_periods = self._find_continuous_periods(low_variance_mask, 3)
        
        # Criterion 3: Transition periods (freeze-thaw signatures) - MORE SELECTIVE
        zero_crossings = np.where(np.diff(np.sign(temperatures)))[0]
        transition_periods = []
        for crossing in zero_crossings:
            # Check if this is a sustained transition (not just noise)
            window_start = max(0, crossing - 10)
            window_end = min(len(temperatures) - 1, crossing + 10)
            window_temps = temperatures[window_start:window_end+1]
            
            # Only include if temperature change is significant and sustained
            temp_range = np.max(window_temps) - np.min(window_temps)
            if temp_range >= 2.0:  # At least 2°C temperature change
                start_idx = max(0, crossing - 3)  # Smaller window
                end_idx = min(len(temperatures) - 1, crossing + 3)
                if end_idx - start_idx >= 3:
                    transition_periods.append((start_idx, end_idx))
        
        # Combine all fallback periods
        all_fallback_periods = near_zero_periods + low_variance_periods + transition_periods
        
        # Merge overlapping periods and sort by length
        merged_periods = []
        sorted_periods = sorted(all_fallback_periods, key=lambda x: x[0])

        for start, end in sorted_periods:
            if not merged_periods:
                merged_periods.append((start, end))
            else:
                last_start, last_end = merged_periods[-1]
                if start <= last_end + 5:  # Allow small gaps (5 points)
                    # Merge overlapping/adjacent periods
                    merged_periods[-1] = (last_start, max(end, last_end))
                else:
                    merged_periods.append((start, end))

        # Filter out very short periods and limit total number
        unique_periods = [period for period in merged_periods if period[1] - period[0] >= 5]
        unique_periods = sorted(unique_periods, key=lambda x: x[1] - x[0], reverse=True)[:10]  # Max 10 events
        
        print(f"Fallback detection for {depth_zone}:")
        print(f"  Near-zero periods: {len(near_zero_periods)}")
        print(f"  Low variance periods: {len(low_variance_periods)}")
        print(f"  Transition periods: {len(transition_periods)}")
        print(f"  Unique fallback periods: {len(unique_periods)}")
        
        # Characterize fallback events with reduced intensity scoring
        for start_idx, end_idx in unique_periods:
            if end_idx > start_idx:
                duration_hours = (end_idx - start_idx + 1) * 24.0  # Assuming daily data
                event = {
                    'start_time': timestamps[start_idx],
                    'end_time': timestamps[end_idx],
                    'duration_hours': duration_hours,
                    'intensity_percentile': 0.3,  # Reduced intensity for fallback events
                    'spatial_extent_meters': 0.2,  # Reduced spatial extent
                    'depth_zone': depth_zone,
                    'mean_temperature': np.mean(temperatures[start_idx:end_idx+1]),
                    'temperature_variance': np.var(temperatures[start_idx:end_idx+1]),
                    'detection_method': 'fallback',
                    'fallback_criterion': 'permissive_thermal_criteria'
                }
                fallback_events.append(event)
        
        return fallback_events
        
    def _log_detection_diagnostics(self, site_idx, lat, lon, events, site_data_length):
        """Log detailed diagnostics for each site processing attempt."""
        
        detection_status = "SUCCESS" if len(events) > 0 else "NO_EVENTS"
        
        diagnostic_info = {
            'site_index': site_idx,
            'latitude': lat,
            'longitude': lon,
            'data_points': site_data_length,
            'events_detected': len(events),
            'detection_status': detection_status
        }
        
        # Add to class-level diagnostic log if it doesn't exist
        if not hasattr(self, 'site_diagnostics'):
            self.site_diagnostics = []
        
        self.site_diagnostics.append(diagnostic_info)
        
        # Periodic diagnostic summary
        if site_idx % 25 == 0:
            recent_diagnostics = self.site_diagnostics[-25:]
            success_rate = sum(1 for d in recent_diagnostics if d['events_detected'] > 0) / len(recent_diagnostics)
            avg_data_points = np.mean([d['data_points'] for d in recent_diagnostics])
            
            print(f"📊 Last 25 sites diagnostic summary:")
            print(f"   Success rate: {success_rate*100:.1f}%")
            print(f"   Average data points: {avg_data_points:.0f}")
            print(f"   Sites processed: {len(self.site_diagnostics)}")
    
    def process_circumarctic_dataset(self, parquet_file: Optional[str] = None, output_file: Optional[str] = None):
        """
        Process full dataset with enhanced CryoGrid integration and permafrost suitability filtering.
        
        Args:
            parquet_file: Optional path to input parquet. Uses config if None.
            output_file: Optional path to output parquet. Uses config if None.
        """
        # Use configured paths if not provided
        if parquet_file is None:
            parquet_file = str(self.config.paths.insitu_measurements_parquet)
        
        if output_file is None:
            output_file = str(self.config.paths.output_dir / f"{self.config.output_prefix}_dataset.parquet")
        
        print("Loading circumarctic dataset with enhanced CryoGrid physics and permafrost suitability filtering...")
        df = dd.read_parquet(parquet_file)
        
        # Filter for cold season soil temperature measurements
        cold_months = [9, 10, 11, 12, 1, 2, 3, 4, 5]
        df_filtered = df[
            (df['data_type'] == 'soil_temperature') &
            (~df['soil_temp'].isna()) &
            (df['datetime'].dt.month.isin(cold_months))
        ]
        
        print(f"Evaluating permafrost suitability for {len(df_filtered)} measurements...")
        print(f"Enhanced with CryoGrid physics: Enthalpy={self.use_cryogrid_enthalpy}, " f"Painter-Karra={self.use_painter_karra_freezing}, SEB={self.use_surface_energy_balance}")

        # Initialize comprehensive tracking variables
        suitable_sites = []
        unsuitable_sites = []
        all_events = []
        processed_sites = 0
        sites_with_events = 0
        total_events_detected = 0

        # Progress tracking for iterative output
        progress_interval = 10  # Report every 10 sites
        detection_history = []  # Track detection rate over time
        
        print("Phase 1: Permafrost suitability screening...")
        
        # Get unique sites for screening - convert to pandas for groupby operations
        unique_sites = df_filtered[['latitude', 'longitude', 'source']].drop_duplicates().compute()
        
        suitable_sites = []
        unsuitable_sites = []
        
        print(f"Screening {len(unique_sites)} unique sites for permafrost suitability...")
        
        # Sample a few sites to show coordinate ranges
        sample_sites = unique_sites.head(10)
        print(f"Sample site coordinates:")
        for idx, row in sample_sites.iterrows():
            print(f"  {row['latitude']:.3f}, {row['longitude']:.3f}")
        
        # Pre-compute site counts to avoid Dask boolean indexing issues
        print("Computing site data counts...")
        try:
            site_counts = df_filtered.groupby(['latitude', 'longitude', 'source']).size().compute()
            print(f"Site counts computed successfully for {len(site_counts)} site combinations")
        except Exception as e:
            print(f"Warning: Could not compute site counts directly: {e}")
            # Fallback: set all counts to estimated value
            site_counts = pd.Series(1000, index=pd.MultiIndex.from_frame(unique_sites))

        for idx, row in unique_sites.iterrows():
            lat, lon, source = row['latitude'], row['longitude'], row['source']
            
            # CRITICAL: Check permafrost suitability FIRST before any processing
            permafrost_props = self.get_site_permafrost_properties(lat, lon)
            
            # Get data count safely
            try:
                data_count = site_counts.loc[(lat, lon, source)]
            except (KeyError, IndexError):
                # Estimate data count if lookup fails
                data_count = 500  # Conservative estimate
            
            if not permafrost_props['is_permafrost_suitable']:
                unsuitable_sites.append({
                    'latitude': lat,
                    'longitude': lon,
                    'source': source,
                    'reason': 'No permafrost present',
                    'permafrost_prob': permafrost_props['permafrost_prob'],
                    'permafrost_zone': permafrost_props['permafrost_zone'],
                    'data_count': data_count
                })
                continue
            
            # Site is suitable - add to processing list
            suitable_sites.append({
                'latitude': lat,
                'longitude': lon,
                'source': source,
                'permafrost_prob': permafrost_props['permafrost_prob'],
                'permafrost_zone': permafrost_props['permafrost_zone'],
                'data_count': data_count
            })
            
            # Iterative progress for suitability screening
            total_screened = len(suitable_sites) + len(unsuitable_sites)
            if total_screened % 50 == 0:  # Every 50 sites screened
                suitable_pct = (len(suitable_sites) / total_screened) * 100
                print(f"  Screening progress: {total_screened}/{len(unique_sites)} sites "
                      f"({suitable_pct:.1f}% suitable for permafrost)")
        
        print(f"✓ Permafrost suitability screening complete:")
        print(f"  - Suitable sites: {len(suitable_sites)}")
        print(f"  - Unsuitable sites: {len(unsuitable_sites)} (no permafrost)")
        
        if len(suitable_sites) == 0:
            print("No suitable permafrost sites found. Exiting.")
            return pd.DataFrame()
        
        print(f"\nPhase 2: Enhanced zero-curtain detection with real-time progress tracking...")
        print(f"Target sites: {len(suitable_sites)}")
        print("=" * 80)
        
        # Process only suitable sites with iterative progress tracking
        for site_idx, site_info in enumerate(suitable_sites):
            lat, lon, source = site_info['latitude'], site_info['longitude'], site_info['source']
            
            # Extract site data with existing error handling
            try:
                # Method 1: Try query-based filtering (most reliable for Dask)
                query_str = f"latitude == {lat} and longitude == {lon} and source == '{source}'"
                site_data = df_filtered.query(query_str).compute()
                
                if len(site_data) < 50:  # Reduced requirement from 200 to 50
                    continue
                    
            except Exception as e:
                try:
                    # Method 2: Fallback to manual filtering with persistence
                    site_data_list = []
                    
                    # Process in chunks to avoid memory issues
                    for partition in df_filtered.to_delayed():
                        partition_df = partition.compute()
                        site_subset = partition_df[
                            (partition_df['latitude'] == lat) &
                            (partition_df['longitude'] == lon) &
                            (partition_df['source'] == source)
                        ]
                        if len(site_subset) > 0:
                            site_data_list.append(site_subset)
                    
                    if site_data_list:
                        site_data = pd.concat(site_data_list, ignore_index=True)
                    else:
                        continue
                        
                    if len(site_data) < 50:
                        continue
                        
                except Exception as e2:
                    print(f"Data extraction failed for site {lat:.3f}, {lon:.3f}: {e2}")
                    continue
            
            # Track processing attempt
            processed_sites += 1
            site_events_before = len(all_events)
            
            try:
                # Apply comprehensive physics detection
                events = self.detect_zero_curtain_with_physics(site_data, lat, lon)
                
                # Add site metadata to events
                for event in events:
                    event.update({
                        'latitude': lat,
                        'longitude': lon,
                        'source': source,
                        'site_index': processed_sites
                    })
                
                all_events.extend(events)
                site_events_after = len(all_events)
                site_event_count = site_events_after - site_events_before
                
                # Track sites with events
                if site_event_count > 0:
                    sites_with_events += 1
                    
                # Update globals for emergency save and memory optimization
                all_events_global = all_events.copy()
                processed_sites_global = processed_sites
                
                # Memory optimization every 25 sites
                if processed_sites % 25 == 0:
                    gc.collect()
                
                # Log diagnostics
                self._log_detection_diagnostics(processed_sites, lat, lon, events, len(site_data))
                
                # ITERATIVE PROGRESS OUTPUT
                if processed_sites % progress_interval == 0 or site_event_count > 0:
                    detection_rate = (sites_with_events / processed_sites) * 100
                    events_per_site = len(all_events) / processed_sites
                    
                    print(f"PROGRESS UPDATE - Site {processed_sites}/{len(suitable_sites)}:")
                    print(f"  Current site: {lat:.3f}°N, {lon:.3f}°E ({source})")
                    print(f"  Events at this site: {site_event_count}")
                    print(f"  Running totals:")
                    print(f"    ├─ Sites with events: {sites_with_events}/{processed_sites} ({detection_rate:.1f}%)")
                    print(f"    ├─ Total events: {len(all_events)}")
                    print(f"    ├─ Events per site: {events_per_site:.2f}")
                    print(f"    └─ Remaining sites: {len(suitable_sites) - processed_sites}")
                    
                    # Store detection history for trend analysis
                    detection_history.append({
                        'sites_processed': processed_sites,
                        'sites_with_events': sites_with_events,
                        'total_events': len(all_events),
                        'detection_rate_pct': detection_rate,
                        'events_per_site': events_per_site
                    })
                    
                    # Trend analysis every 50 sites
                    if processed_sites % 50 == 0 and len(detection_history) >= 2:
                        recent_rate = detection_history[-1]['detection_rate_pct']
                        prev_rate = detection_history[-2]['detection_rate_pct']
                        trend = "↗" if recent_rate > prev_rate else "↘" if recent_rate < prev_rate else "→"
                        print(f"  Detection rate trend: {trend} ({prev_rate:.1f}% → {recent_rate:.1f}%)")
                    
                    print("-" * 60)
                
                # INCREMENTAL SAVING EVERY 50 SITES WITH ALL FEATURES
                if processed_sites % 50 == 0:
                    print(f"💾 INCREMENTAL SAVE at site {processed_sites}...")
                    
                    try:
                        if all_events:
                            # Create DataFrame with ALL features (optimized)
                            zc_df_temp = pd.DataFrame(all_events)
                            
                            # Add ALL derived classifications efficiently
                            zc_df_temp['intensity_category'] = pd.cut(
                                zc_df_temp['intensity_percentile'],
                                bins=[0, 0.25, 0.5, 0.75, 1.0],
                                labels=['weak', 'moderate', 'strong', 'extreme']
                            )
                            
                            zc_df_temp['duration_category'] = pd.cut(
                                zc_df_temp['duration_hours'],
                                bins=[0, 72, 168, 336, np.inf],
                                labels=['short', 'medium', 'long', 'extended']
                            )
                            
                            zc_df_temp['extent_category'] = pd.cut(
                                zc_df_temp['spatial_extent_meters'],
                                bins=[0, 0.3, 0.8, 1.5, np.inf],
                                labels=['shallow', 'moderate', 'deep', 'very_deep']
                            )
                            
                            # CryoGrid-specific classifications (if columns exist)
                            if 'cryogrid_thermal_conductivity' in zc_df_temp.columns:
                                zc_df_temp['thermal_conductivity_category'] = pd.cut(
                                    zc_df_temp['cryogrid_thermal_conductivity'],
                                    bins=[0, 0.5, 1.0, 2.0, np.inf],
                                    labels=['low', 'medium', 'high', 'very_high']
                                )
                            
                            if 'cryogrid_enthalpy_stability' in zc_df_temp.columns:
                                zc_df_temp['enthalpy_stability_category'] = pd.cut(
                                    zc_df_temp['cryogrid_enthalpy_stability'],
                                    bins=[0, 0.3, 0.6, 0.8, 1.0],
                                    labels=['unstable', 'moderately_stable', 'stable', 'very_stable']
                                )
                            
                            # SAVE WITH ALL FEATURES
                            temp_output_file = str(self.config.paths.output_dir / f"{self.config.output_prefix}_INCREMENTAL_site_{processed_sites}.parquet")
                            zc_df_temp.to_parquet(temp_output_file, index=False, compression='snappy')
                            
                            print(f"  ✅ Saved {len(all_events)} events with ALL features to: {temp_output_file}")
                            
                            # Verify the three main features are present
                            main_features = ['intensity_percentile', 'duration_hours', 'spatial_extent_meters']
                            missing_features = [f for f in main_features if f not in zc_df_temp.columns]
                            if missing_features:
                                print(f"  ⚠️  WARNING: Missing main features: {missing_features}")
                            else:
                                print(f"  ✅ All three main features confirmed: intensity, duration, spatial_extent")
                            
                            # ALSO SAVE SITE SUITABILITY
                            suitable_df_temp = pd.DataFrame(suitable_sites)
                            suitable_df_temp['is_suitable'] = True
                            
                            unsuitable_df_temp = pd.DataFrame(unsuitable_sites)
                            unsuitable_df_temp['is_suitable'] = False
                            
                            all_sites_df_temp = pd.concat([suitable_df_temp, unsuitable_df_temp], ignore_index=True)
                            suitability_temp_file = str(self.config.paths.output_dir / f"{self.config.output_prefix}_INCREMENTAL_suitability_site_{processed_sites}.parquet")
                            all_sites_df_temp.to_parquet(suitability_temp_file, index=False, compression='snappy')
                            
                            # Save progress info
                            progress_info = {
                                'total_sites_processed': processed_sites,
                                'sites_with_events': sites_with_events,
                                'total_events': len(all_events),
                                'detection_rate_percent': detection_rate,
                                'events_per_site': events_per_site,
                                'timestamp': pd.Timestamp.now(),
                                'total_suitable_sites': len(suitable_sites),
                                'total_unsuitable_sites': len(unsuitable_sites)
                            }
                            
                            progress_file = str(self.config.paths.output_dir / f"{self.config.output_prefix}_PROGRESS_site_{processed_sites}.txt")
                            with open(progress_file, 'w') as f:
                                f.write("INCREMENTAL SAVE PROGRESS REPORT\n")
                                f.write("=" * 40 + "\n")
                                for key, value in progress_info.items():
                                    f.write(f"{key}: {value}\n")
                                f.write(f"\nFeatures saved: {list(zc_df_temp.columns)}\n")
                                f.write(f"Main features verified: {main_features}\n")
                            
                            print(f"  ✅ Progress and suitability saved")
                            
                            # Memory cleanup to optimize performance
                            del zc_df_temp, suitable_df_temp, unsuitable_df_temp, all_sites_df_temp
                            import gc
                            gc.collect()
                            
                    except Exception as e:
                        print(f"  ⚠️  Incremental save failed: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Special notification for high-event sites
                if site_event_count >= 3:
                    print(f"🎯 HIGH-YIELD SITE: {site_event_count} events at {lat:.3f}°N, {lon:.3f}°E")
                    for i, event in enumerate(events[-site_event_count:]):
                        print(f"   Event {i+1}: {event['duration_hours']:.1f}h duration, "
                              f"intensity={event['intensity_percentile']:.3f}, "
                              f"extent={event['spatial_extent_meters']:.2f}m")
                    print()
                    
                    # EMERGENCY SAVE for high-yield sites (>100 events) with ALL features
                    if site_event_count >= 100:
                        print(f"💾 EMERGENCY SAVE - High-yield site with {site_event_count} events")
                        
                        try:
                            emergency_file = str(self.config.paths.output_dir / f"{self.config.output_prefix}_EMERGENCY_site_{processed_sites}_{site_event_count}events.parquet")
                            if all_events:
                                emergency_df = pd.DataFrame(all_events)
                                emergency_df.to_parquet(emergency_file, index=False, compression='snappy')
                                print(f"  ✅ Emergency save with ALL features: {emergency_file}")
                                
                                # Verify all three main features
                                main_features = ['intensity_percentile', 'duration_hours', 'spatial_extent_meters']
                                verified = all(f in emergency_df.columns for f in main_features)
                                print(f"  ✅ Three main features verified: {verified}")
                                
                                # Memory cleanup
                                del emergency_df
                                import gc
                                gc.collect()
                                
                        except Exception as e:
                            print(f"  ⚠️  Emergency save failed: {e}")
                    
            except Exception as e:
                print(f"⚠️  ERROR processing site {processed_sites} ({lat:.3f}, {lon:.3f}): {e}")
                continue
        
        # Create comprehensive dataset with CryoGrid enhancements
        if all_events:
            zc_df = pd.DataFrame(all_events)
            
            # Add derived classifications
            zc_df['intensity_category'] = pd.cut(
                zc_df['intensity_percentile'],
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['weak', 'moderate', 'strong', 'extreme']
            )
            
            zc_df['duration_category'] = pd.cut(
                zc_df['duration_hours'],
                bins=[0, 72, 168, 336, np.inf],
                labels=['short', 'medium', 'long', 'extended']
            )
            
            zc_df['extent_category'] = pd.cut(
                zc_df['spatial_extent_meters'],
                bins=[0, 0.3, 0.8, 1.5, np.inf],
                labels=['shallow', 'moderate', 'deep', 'very_deep']
            )
            
            # CryoGrid-specific classifications
            if 'cryogrid_thermal_conductivity' in zc_df.columns:
                zc_df['thermal_conductivity_category'] = pd.cut(
                    zc_df['cryogrid_thermal_conductivity'],
                    bins=[0, 0.5, 1.0, 2.0, np.inf],
                    labels=['low', 'medium', 'high', 'very_high']
                )
            
            if 'cryogrid_enthalpy_stability' in zc_df.columns:
                zc_df['enthalpy_stability_category'] = pd.cut(
                    zc_df['cryogrid_enthalpy_stability'],
                    bins=[0, 0.3, 0.6, 0.8, 1.0],
                    labels=['unstable', 'moderately_stable', 'stable', 'very_stable']
                )
            
            # Save comprehensive physics-informed results with CryoGrid enhancements
            zc_df.to_parquet(output_file, index=False, compression='snappy')
            
            # Save site suitability information
            suitability_file = output_file.replace('.parquet', '_site_suitability.parquet')
            
            suitable_df = pd.DataFrame(suitable_sites)
            suitable_df['is_suitable'] = True
            
            unsuitable_df = pd.DataFrame(unsuitable_sites)
            unsuitable_df['is_suitable'] = False
            
            all_sites_df = pd.concat([suitable_df, unsuitable_df], ignore_index=True)
            all_sites_df.to_parquet(suitability_file, index=False, compression='snappy')
            
            # Save CryoGrid configuration summary
            config_file = output_file.replace('.parquet', '_cryogrid_config.txt')
            with open(config_file, 'w') as f:
                f.write("CryoGrid Integration Configuration\n")
                f.write("=" * 40 + "\n")
                f.write(f"Use CryoGrid Enthalpy Formulation: {self.use_cryogrid_enthalpy}\n")
                f.write(f"Use Painter-Karra Freezing: {self.use_painter_karra_freezing}\n")
                f.write(f"Use Surface Energy Balance: {self.use_surface_energy_balance}\n")
                f.write(f"Use Adaptive Time-stepping: {self.use_adaptive_timestep}\n")
                f.write(f"Maximum Enthalpy Change: {self.MAX_ENTHALPY_CHANGE} J/m³\n")
                f.write(f"Stefan-Boltzmann Constant: {self.STEFAN_BOLTZMANN} W/m²/K⁴\n")
                f.write(f"Volumetric Latent Heat: {self.LVOL_SL} J/m³\n")
            
            # COMPREHENSIVE FINAL ANALYSIS
            if all_events:
                zc_df = pd.DataFrame(all_events)
                
                # Add derived classifications
                zc_df['intensity_category'] = pd.cut(
                    zc_df['intensity_percentile'],
                    bins=[0, 0.25, 0.5, 0.75, 1.0],
                    labels=['weak', 'moderate', 'strong', 'extreme']
                )
                
                zc_df['duration_category'] = pd.cut(
                    zc_df['duration_hours'],
                    bins=[0, 72, 168, 336, np.inf],
                    labels=['short', 'medium', 'long', 'extended']
                )
                
                zc_df['extent_category'] = pd.cut(
                    zc_df['spatial_extent_meters'],
                    bins=[0, 0.3, 0.8, 1.5, np.inf],
                    labels=['shallow', 'moderate', 'deep', 'very_deep']
                )
                
                # CryoGrid-specific classifications
                if 'cryogrid_thermal_conductivity' in zc_df.columns:
                    zc_df['thermal_conductivity_category'] = pd.cut(
                        zc_df['cryogrid_thermal_conductivity'],
                        bins=[0, 0.5, 1.0, 2.0, np.inf],
                        labels=['low', 'medium', 'high', 'very_high']
                    )
                
                if 'cryogrid_enthalpy_stability' in zc_df.columns:
                    zc_df['enthalpy_stability_category'] = pd.cut(
                        zc_df['cryogrid_enthalpy_stability'],
                        bins=[0, 0.3, 0.6, 0.8, 1.0],
                        labels=['unstable', 'moderately_stable', 'stable', 'very_stable']
                    )
                
                # Save comprehensive physics-informed results with CryoGrid enhancements
                zc_df.to_parquet(output_file, index=False, compression='snappy')
                
                # Save site suitability information
                suitability_file = output_file.replace('.parquet', '_site_suitability.parquet')
                
                suitable_df = pd.DataFrame(suitable_sites)
                suitable_df['is_suitable'] = True
                
                unsuitable_df = pd.DataFrame(unsuitable_sites)
                unsuitable_df['is_suitable'] = False
                
                all_sites_df = pd.concat([suitable_df, unsuitable_df], ignore_index=True)
                all_sites_df.to_parquet(suitability_file, index=False, compression='snappy')
                
                # Save CryoGrid configuration summary
                config_file = output_file.replace('.parquet', '_cryogrid_config.txt')
                with open(config_file, 'w') as f:
                    f.write("CryoGrid Integration Configuration\n")
                    f.write("=" * 40 + "\n")
                    f.write(f"Use CryoGrid Enthalpy Formulation: {self.use_cryogrid_enthalpy}\n")
                    f.write(f"Use Painter-Karra Freezing: {self.use_painter_karra_freezing}\n")
                    f.write(f"Use Surface Energy Balance: {self.use_surface_energy_balance}\n")
                    f.write(f"Use Adaptive Time-stepping: {self.use_adaptive_timestep}\n")
                    f.write(f"Maximum Enthalpy Change: {self.MAX_ENTHALPY_CHANGE} J/m³\n")
                    f.write(f"Stefan-Boltzmann Constant: {self.STEFAN_BOLTZMANN} W/m²/K⁴\n")
                    f.write(f"Volumetric Latent Heat: {self.LVOL_SL} J/m³\n")

                print(f"\n" + "="*120)
                print("ENHANCED PHYSICS-INFORMED ZERO-CURTAIN DETECTION WITH CRYOGRID INTEGRATION COMPLETE")
                print("="*120)

                # Detection rate analysis
                final_detection_rate = (sites_with_events / processed_sites) * 100 if processed_sites > 0 else 0
                final_events_per_site = len(all_events) / processed_sites if processed_sites > 0 else 0

                print(f"DETECTION PERFORMANCE SUMMARY:")
                print(f"  Total suitable permafrost sites: {len(suitable_sites):,}")
                print(f"  Successfully processed sites: {processed_sites:,}")
                print(f"  Sites with zero-curtain events: {sites_with_events:,}")
                print(f"  Sites without events: {processed_sites - sites_with_events:,}")
                print(f"  Overall detection rate: {final_detection_rate:.1f}% ({sites_with_events}/{processed_sites})")
                print(f"  Total zero-curtain events: {len(all_events):,}")
                print(f"  Events per site (average): {final_events_per_site:.2f}")
                print(f"  Events per successful site: {len(all_events)/sites_with_events:.2f}" if sites_with_events > 0 else "  Events per successful site: N/A")

                # Detection rate progression analysis
                if len(detection_history) >= 3:
                    print(f"\nDETECTION RATE PROGRESSION:")
                    milestones = [detection_history[i] for i in [0, len(detection_history)//2, -1]]
                    
                    for i, milestone in enumerate(milestones):
                        stage = ["Early", "Mid", "Final"][i]
                        print(f"  {stage} stage (site {milestone['sites_processed']}): "
                              f"{milestone['detection_rate_pct']:.1f}% detection rate, "
                              f"{milestone['events_per_site']:.2f} events/site")

                # Event characteristics summary
                print(f"\nZERO-CURTAIN EVENT CHARACTERISTICS:")
                print(f"  Mean intensity: {zc_df['intensity_percentile'].mean():.3f}")
                print(f"  Mean duration: {zc_df['duration_hours'].mean():.1f} hours")
                print(f"  Mean spatial extent: {zc_df['spatial_extent_meters'].mean():.2f} meters")
                
                # Detection method breakdown
                if 'detection_method' in zc_df.columns:
                    method_counts = zc_df['detection_method'].value_counts()
                    print(f"  Detection methods: {dict(method_counts)}")
                
                # Geographic distribution
                print(f"  Latitude range: {zc_df['latitude'].min():.1f}° to {zc_df['latitude'].max():.1f}°N")
                print(f"  Longitude range: {zc_df['longitude'].min():.1f}° to {zc_df['longitude'].max():.1f}°E")
                
                # Permafrost context
                print(f"  Permafrost zones represented: {zc_df['permafrost_zone'].value_counts().to_dict()}")
                print(f"  Depth zones: {zc_df['depth_zone'].value_counts().to_dict()}")

                # CryoGrid-specific statistics
                if 'cryogrid_thermal_conductivity' in zc_df.columns:
                    print(f"  Mean CryoGrid thermal conductivity: {zc_df['cryogrid_thermal_conductivity'].mean():.3f} W/m/K")
                if 'cryogrid_heat_capacity' in zc_df.columns:
                    print(f"  Mean CryoGrid heat capacity: {zc_df['cryogrid_heat_capacity'].mean():.0f} J/m³/K")
                if 'cryogrid_enthalpy_stability' in zc_df.columns:
                    print(f"  Mean enthalpy stability: {zc_df['cryogrid_enthalpy_stability'].mean():.3f}")

                # Processing efficiency
                unsuitable_rate = (len(unsuitable_sites) / (len(suitable_sites) + len(unsuitable_sites))) * 100
                print(f"\nPROCESSING EFFICIENCY:")
                print(f"  Unsuitable sites filtered: {len(unsuitable_sites):,} ({unsuitable_rate:.1f}%)")
                print(f"  Processing success rate: {(processed_sites/len(suitable_sites))*100:.1f}%")

                # Recommendations based on results
                print(f"\nRECOMMENDATIONS:")
                if final_detection_rate < 50:
                    print(f"  ⚠️  Low detection rate ({final_detection_rate:.1f}%) suggests criteria may still be too restrictive")
                    print(f"     Consider further loosening thresholds or adding more fallback detection pathways")
                elif final_detection_rate > 80:
                    print(f"  ✅ High detection rate ({final_detection_rate:.1f}%) indicates good parameter tuning")
                    print(f"     Current criteria successfully capture zero-curtain events")
                else:
                    print(f"  📊 Moderate detection rate ({final_detection_rate:.1f}%) suggests balanced criteria")
                    print(f"     Results appear reasonable for circumarctic analysis")

                if final_events_per_site < 1.0:
                    print(f"  📈 Low events per site ({final_events_per_site:.2f}) suggests brief or weak zero-curtain signatures")
                elif final_events_per_site > 3.0:
                    print(f"  🎯 High events per site ({final_events_per_site:.2f}) indicates strong zero-curtain activity")

                print(f"Freezing characteristics: {zc_df['soil_freezing_characteristic'].value_counts().to_dict()}")
                print(f"\nOUTPUT FILES:")
                print(f"  Zero-curtain results: {output_file}")
                print(f"  Site suitability results: {suitability_file}")
                print(f"  CryoGrid configuration: {config_file}")
                
                return zc_df
            
            else:
                print("No zero-curtain events detected from suitable permafrost sites.")
                return pd.DataFrame()

def main():
    """Main execution with comprehensive CryoGrid physics integration."""
    
    print("Initializing Enhanced Physics-Informed Zero-Curtain Detector with CryoGrid Integration...")
    print("=" * 90)
    
    detector = PhysicsInformedZeroCurtainDetector()
    
    print(f"CryoGrid Integration Status:")
    print(f"  - Enthalpy-based formulation: {detector.use_cryogrid_enthalpy}")
    print(f"  - Painter-Karra freezing: {detector.use_painter_karra_freezing}")
    print(f"  - Surface energy balance: {detector.use_surface_energy_balance}")
    print(f"  - Adaptive time-stepping: {detector.use_adaptive_timestep}")
    print()
    
    parquet_file = "/Users/bagay/Downloads/merged_compressed_corrected_final.parquet"
    output_file = "/Users/bagay/Downloads/zero_curtain_enhanced_cryogrid_physics_dataset.parquet"
    
    results = detector.process_circumarctic_dataset(parquet_file, output_file)
    
    if not results.empty:
        print("\nEnhanced Physics-Informed Analysis Summary with CryoGrid:")
        print(f"Permafrost zones represented: {results['permafrost_zone'].value_counts().to_dict()}")
        print(f"Depth zones: {results['depth_zone'].value_counts().to_dict()}")
        print(f"Mean permafrost probability: {results['permafrost_probability'].mean():.3f}")
        
        # CryoGrid-specific analysis
        if 'soil_freezing_characteristic' in results.columns:
            print(f"Soil freezing characteristics: {results['soil_freezing_characteristic'].value_counts().to_dict()}")
        
        if 'thermal_conductivity_category' in results.columns:
            print(f"Thermal conductivity distribution: {results['thermal_conductivity_category'].value_counts().to_dict()}")
        
        if 'enthalpy_stability_category' in results.columns:
            print(f"Enthalpy stability distribution: {results['enthalpy_stability_category'].value_counts().to_dict()}")
        
        print(f"\nCryoGrid Enhanced Features Successfully Integrated!")
        print(f"  - Advanced enthalpy-based thermodynamics")
        print(f"  - Sophisticated soil freezing characteristics")
        print(f"  - Surface energy balance coupling")
        print(f"  - Adaptive numerical time-stepping")
        print(f"  - Enhanced lateral thermal interactions")
        print(f"  - Multi-physics integration framework")

if __name__ == "__main__":
    main()
