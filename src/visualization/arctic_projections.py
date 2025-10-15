"""
Arctic projection and visualization utilities.
Functions for plotting Arctic data with proper polar projections.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
import geopandas as gpd
from shapely.geometry import Point


def create_spatial_df(df):
    """
    Convert DataFrame with lat/lon to GeoDataFrame.
    
    Args:
        df: DataFrame with 'longitude' and 'latitude' columns
    
    Returns:
        GeoDataFrame with geometry column
    """
    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def plot_arctic_projection(df, column, cmap='viridis', title=None, output_path=None,
                           vmin=None, vmax=None, percentile_clip=True):
    """
    Plot Arctic data using North Polar Stereographic projection.
    
    Args:
        df: DataFrame with 'longitude', 'latitude', and data column
        column: Column name to plot
        cmap: Matplotlib colormap name
        title: Plot title
        output_path: Path to save figure (optional)
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        percentile_clip: If True, clip to 5th-95th percentile
    
    Returns:
        Figure object
    """
    projection = ccrs.NorthPolarStereo(central_longitude=0.0)
    
    fig = plt.figure(figsize=(14, 14))
    ax = plt.axes(projection=projection)
    
    # Set extent to Arctic (50°N to 90°N)
    ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    
    # Add cartographic features
    ax.coastlines(resolution='50m', color='white', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6, edgecolor='white')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    
    # Set color normalization
    if vmin is None or vmax is None:
        if percentile_clip:
            vmin = np.percentile(df[column].dropna(), 5)
            vmax = np.percentile(df[column].dropna(), 95)
        else:
            vmin = df[column].min()
            vmax = df[column].max()
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot data
    sc = ax.scatter(
        df['longitude'], 
        df['latitude'], 
        c=df[column],
        s=3, 
        alpha=0.7,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm
    )
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.05, shrink=0.8)
    cbar.set_label(column.replace('_', ' ').title(), fontsize=14)
    
    # Add title
    if title:
        plt.title(title, fontsize=16, pad=20)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {output_path}")
        plt.close()
    else:
        plt.show()
    
    return fig


def analyze_data_coverage(df, output_dir="coverage_analysis", 
                          spatial_bins=(36, 8), lat_range=(50, 90)):
    """
    Analyze spatial and temporal data coverage.
    
    Args:
        df: DataFrame with 'longitude', 'latitude', and optionally 'datetime'
        output_dir: Directory to save analysis results
        spatial_bins: (lon_bins, lat_bins) for histogram
        lat_range: (min_lat, max_lat) for analysis
    
    Returns:
        Dictionary with coverage statistics
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nDATA COVERAGE ANALYSIS\n")
    print(f"Total observations: {len(df):,}")
    print(f"Longitude range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
    print(f"Latitude range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
    
    # Spatial coverage histogram
    hist, xedges, yedges = np.histogram2d(
        df['longitude'], 
        df['latitude'],
        bins=spatial_bins,
        range=[[-180, 180], lat_range]
    )
    
    total_bins = hist.size
    empty_bins = np.sum(hist == 0)
    coverage_pct = 100 * (1 - empty_bins / total_bins)
    
    print(f"Spatial coverage: {coverage_pct:.2f}% of Arctic grid cells")
    print(f"Empty cells: {empty_bins}/{total_bins}")
    
    # Temporal coverage
    if 'datetime' in df.columns:
        import pandas as pd
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"Temporal range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        years_covered = df['datetime'].dt.year.nunique()
        print(f"Years covered: {years_covered}")
    
    results = {
        "total_observations": len(df),
        "coverage_percentage": coverage_pct,
        "empty_cells": int(empty_bins),
        "total_cells": int(total_bins),
        "lon_range": (float(df['longitude'].min()), float(df['longitude'].max())),
        "lat_range": (float(df['latitude'].min()), float(df['latitude'].max()))
    }
    
    return results


if __name__ == "__main__":
    """Test visualization functions."""
    print("Visualization module loaded successfully")
    print("\nAvailable functions:")
    print("  - create_spatial_df(df)")
    print("  - plot_arctic_projection(df, column, ...)")
    print("  - analyze_data_coverage(df, ...)")
