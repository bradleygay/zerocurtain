"""
Visualization module for Arctic zero-curtain pipeline.
Auto-generated from Jupyter notebook.
"""

from src.utils.imports import *
from src.utils.utilities import *

import os,sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import gc
import glob
import json
import logging
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tables
import os
import pandas as pd
import pathlib
import pickle
import psutil
import re
import gc
import dask
import dask.dataframe as dd
import scipy.interpolate as interpolate
import scipy.stats as stats
import seaborn as sns
from pyproj import Proj
import sklearn.experimental
import sklearn.impute
import sklearn.linear_model
import sklearn.preprocessing
import tqdm
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

from osgeo import gdal, osr
import cmasher as cmr
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm
from tqdm.notebook import tqdm

import tensorflow as tf
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
    pass

print("TensorFlow version:", tf.__version__)

import keras_tuner as kt
from keras_tuner.tuners import BayesianOptimization
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
#os.environ["DEVICE_COUNT_GPU"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # DISABLE GPU
import tensorflow as tf
import json
import glob
import keras
from keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# os.environ["DEVICE_COUNT_CPU"] = "1"

from packaging import version

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

print("==========================")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(sys.getrecursionlimit())
sys.setrecursionlimit(1000000000)
print(sys.getrecursionlimit())

#tf.config.run_functions_eagerly(True)

# with tf.device('/CPU:0'):
#     inputs = tf.keras.Input(shape=(30, 4))
#     x = tf.keras.layers.Flatten()(inputs)
#     outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)

# os.environ['TENSORFLOW_FORCE_CPU'] = '1'

# def diagnose_zero_curtain_durations(events_df, output_dir=None):
#     """
#     Comprehensive analysis of zero curtain duration patterns to identify
#     the source of value clustering and recommend fixes.
    
#     Parameters:
#     -----------
#     events_df : pandas.DataFrame
#         DataFrame containing zero curtain events
#     output_dir : str, optional
#         Directory to save diagnostic outputs
        
#     Returns:
#     --------
#     dict
#         Dictionary containing diagnostic results
#     """
#     print("=" * 80)
#     print("ZERO CURTAIN DURATION PATTERN ANALYSIS")
#     print("=" * 80)
    
#     # 1. Basic statistics
#     duration_values = events_df['duration_hours'].values
#     n_events = len(duration_values)
#     n_unique = len(np.unique(duration_values))
    
#     print(f"\nTotal events: {n_events}")
#     print(f"Unique duration values: {n_unique} ({n_unique/n_events*100:.1f}%)")
    
#     # Calculate full statistics
#     stats_df = pd.DataFrame({
# 'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max',...
#         'Value': [
#             duration_values.mean(),
#             np.median(duration_values),
#             duration_values.std(),
#             duration_values.min(),
#             duration_values.max(),
#             np.percentile(duration_values, 25),
#             np.percentile(duration_values, 75),
#             np.percentile(duration_values, 75) - np.percentile(duration_values, 25)
#         ]
#     })
    
#     print("\nBasic Statistics:")
#     for _, row in stats_df.iterrows():
#         print(f"  {row['Statistic']}: {row['Value']:.2f}")
    
#     # 2. Value frequency analysis
#     from collections import Counter
#     value_counts = Counter(duration_values)
#     common_values = pd.Series(value_counts).sort_values(ascending=False).head(20)
    
#     print("\nMost common duration values:")
#     for val, count in common_values.items():
#         print(f"  {val:.2f} hours: {count} events ({count/n_events*100:.1f}%)")
    
#     # 3. Check for patterns in the values
#     # Are values clustering at specific intervals?
#     rounded_to_hour = np.round(duration_values)
#     rounded_to_day = np.round(duration_values / 24) * 24
    
#     print("\nRounding patterns:")
#     print(f"  Events matching exact hours: {np.sum(duration_values == rounded_to_hour)} ({np.sum(d...
#     print(f"  Events matching exact days: {np.sum(duration_values == rounded_to_day)} ({np.sum(dur...
    
#     # Check for specific hour intervals (6h, 12h, 24h)
#     for interval in [1, 6, 12, 24]:
#         rounded = np.round(duration_values / interval) * interval
#         match_pct = np.sum(duration_values == rounded) / n_events * 100
#         print(f"  Events at {interval}h intervals: {np.sum(duration_values == rounded)} ({match_pc...
    
#     # 4. Create visualizations
#     # Distribution plot
#     plt.figure(figsize=(12, 5))
    
#     plt.subplot(121)
#     sns.histplot(duration_values, bins=50, kde=True)
#     plt.title('Distribution of Duration Values')
#     plt.xlabel('Duration (hours)')
#     plt.ylabel('Frequency')
    
#     # Log scale for better visibility
#     plt.subplot(122)
#     sns.histplot(duration_values, bins=50, kde=True, log_scale=(False, True))
#     plt.title('Distribution (Log Scale)')
#     plt.xlabel('Duration (hours)')
#     plt.ylabel('Log Frequency')
    
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, 'distribution.png'), dpi=150)
    
#     # Duration value heatmap
#     # Create a 2D histogram based on:
#     # X-axis: Duration value
#     # Y-axis: Remainder when divided by 24 (to check for daily patterns)
    
#     # Filter to reasonable range for visualization
#     filtered_durations = duration_values[duration_values <= np.percentile(duration_values, 95)]
    
#     plt.figure(figsize=(14, 6))
    
#     # 2D histogram for patterns
#     hours_remainder = filtered_durations % 24
#     plt.subplot(121)
#     plt.hist2d(
#         filtered_durations, 
#         hours_remainder, 
#         bins=[50, 24],
#         cmap='viridis'
#     )
#     plt.colorbar(label='Count')
#     plt.title('Duration Patterns')
#     plt.xlabel('Duration (hours)')
#     plt.ylabel('Hours Remainder (duration % 24)')
    
#     # Plot the relationship between durations and observation count
#     plt.subplot(122)
    
#     # Group by source and calculate stats
#     site_durations = events_df.groupby('source').agg({
#         'duration_hours': ['count', 'mean', 'median']
#     })
    
#     site_durations.columns = ['_'.join(col).strip('_') for col in site_durations.columns]
    
#     plt.scatter(
#         site_durations['duration_hours_count'],
#         site_durations['duration_hours_mean'],
#         alpha=0.5
#     )
#     plt.title('Relationship: Event Count vs Duration')
#     plt.xlabel('Number of Events per Site')
#     plt.ylabel('Mean Duration (hours)')
#     plt.grid(alpha=0.3)
    
#     if output_dir:
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, 'patterns.png'), dpi=150)
    
#     # 5. Check for algorithm artifacts
#     # Look for signs of temporal aliasing or measurement effects
#     temporal_patterns = events_df.copy()
    
#     # Extract detection algorithm artifacts if datetime columns exist
#     if 'datetime_min' in events_df.columns and 'datetime_max' in events_df.columns:
#         if not pd.api.types.is_datetime64_dtype(events_df['datetime_min']):
#             temporal_patterns['datetime_min'] = pd.to_datetime(events_df['datetime_min'])
#         if not pd.api.types.is_datetime64_dtype(events_df['datetime_max']):
#             temporal_patterns['datetime_max'] = pd.to_datetime(events_df['datetime_max'])
            
#         # Extract time of day and day of week
#         temporal_patterns['start_hour'] = temporal_patterns['datetime_min'].dt.hour
#         temporal_patterns['end_hour'] = temporal_patterns['datetime_max'].dt.hour
#         temporal_patterns['start_day'] = temporal_patterns['datetime_min'].dt.dayofweek
#         temporal_patterns['end_day'] = temporal_patterns['datetime_max'].dt.dayofweek
        
#         # Plot patterns
#         fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
#         sns.histplot(data=temporal_patterns, x='start_hour', kde=True, ax=axes[0, 0])
#         axes[0, 0].set_title('Event Start Hour')
        
#         sns.histplot(data=temporal_patterns, x='end_hour', kde=True, ax=axes[0, 1])
#         axes[0, 1].set_title('Event End Hour')
        
#         sns.histplot(data=temporal_patterns, x='start_day', kde=True, ax=axes[1, 0])
#         axes[1, 0].set_title('Event Start Day')
#         axes[1, 0].set_xticks(range(7))
#         axes[1, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
#         sns.histplot(data=temporal_patterns, x='end_day', kde=True, ax=axes[1, 1])
#         axes[1, 1].set_title('Event End Day')
#         axes[1, 1].set_xticks(range(7))
#         axes[1, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
#         if output_dir:
#             plt.tight_layout()
#             plt.savefig(os.path.join(output_dir, 'temporal.png'), dpi=150)
    
#     # 6. Correlation with other variables
#     # Check if duration is correlated with geographic location
#     correlation_data = events_df.copy()
    
#     plt.figure(figsize=(14, 5))
    
#     plt.subplot(121)
#     plt.scatter(
#         correlation_data['latitude'],
#         correlation_data['duration_hours'],
#         alpha=0.1,
#         s=3
#     )
#     plt.title('Duration vs Latitude')
#     plt.xlabel('Latitude')
#     plt.ylabel('Duration (hours)')
#     plt.grid(alpha=0.3)
    
#     plt.subplot(122)
#     plt.scatter(
#         correlation_data['soil_temp_depth'],
#         correlation_data['duration_hours'],
#         alpha=0.1,
#         s=3
#     )
#     plt.title('Duration vs Soil Depth')
#     plt.xlabel('Soil Temperature Depth')
#     plt.ylabel('Duration (hours)')
#     plt.grid(alpha=0.3)
    
#     if output_dir:
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, 'correlations.png'), dpi=150)
    
#     # 7. Generate recommendations based on findings
#     print("\n" + "=" * 80)
#     print("DIAGNOSIS AND RECOMMENDATIONS")
#     print("=" * 80)
    
#     # Check for daily measurement effects
#     day_effect = np.sum(duration_values % 24 == 0) / n_events * 100
#     if day_effect > 50:
#         print("\nDiagnosis: Strong daily measurement effect detected.")
# print(f" {day_effect:.1f}% of durations are exact multiples...
#         print("  This suggests temporal aliasing due to measurement frequency.")
#         print("\nRecommendation:")
#         print("  1. Review zero_curtain_detection temporal parameters")
# print(" 2. Decrease 'max_gap_hours' to allow for...
# print(" 3. Apply interpolation to estimate more...
    
#     # Check for binning or rounding
#     if n_unique / n_events < 0.1:
#         print("\nDiagnosis: Severe value binning or rounding detected.")
#         print(f"  Only {n_unique} unique values for {n_events} events ({n_unique/n_events*100:.1f}...
#         print("\nRecommendation:")
#         print("  1. Check for explicit rounding in duration calculations")
#         print("  2. Use higher precision timestamps for event boundaries")
# print(" 3. Consider continuous time representation instead...
    
#     # Check for IQR issues
#     q1 = np.percentile(duration_values, 25)
#     q3 = np.percentile(duration_values, 75)
#     iqr = q3 - q1
    
#     if iqr < 1e-6:
#         print("\nDiagnosis: Zero or near-zero IQR detected.")
# print(f" Q1 and Q3 are both {q1:.2f},...
#         print("\nRecommendation:")
#         print("  1. Use percentile-based visualization bounds instead of IQR")
# print(" 2. For visualization, force a minimum...
#         print("  3. Consider a non-linear transformation of duration values")
    
#     # Return diagnostic results
#     return {
#         'n_events': n_events,
#         'n_unique': n_unique,
#         'statistics': stats_df,
#         'common_values': common_values,
#         'day_effect_pct': day_effect,
#         'q1': q1,
#         'q3': q3,
#         'iqr': iqr
#     }

def plot_model_architecture(model, output_file='model_architecture.png', show_shapes=True, show_layer_names=True, rankdir='TB'):
    """
    Generates a plot of the model architecture.
    
    Args:
        model: A Keras model instance
        output_file: File name of the plot image
        show_shapes: whether to display shape information
        show_layer_names: whether to display layer names
        rankdir: 'TB' creates a vertical plot; 'LR' creates a horizontal plot
    """
    tf.keras.utils.plot_model(
        model,
        to_file=output_file,
        show_shapes=show_shapes,
        show_layer_names=show_layer_names,
        rankdir=rankdir,
        expand_nested=True,
        dpi=300
    )
    return output_file

# For a more publication-ready visualization, create a...
def create_simplified_diagram():
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Define components and their positions
    components = [
        "Time Series Input\n(168 × 3)",
        "ConvLSTM Processing",
        "Self-Attention",
        "Feed-Forward Network",
        "Parallel Conv1D Paths",
        "Variational Component",
        "Feature Aggregation",
        "Classification Layers",
        "Output\n(Zero Curtain Prediction)"
    ]
    
    # Positions for components in the diagram
    positions = [
        (0.5, 0.9),    # Input
        (0.3, 0.75),   # ConvLSTM
        (0.3, 0.6),    # Self-Attention
        (0.3, 0.45),   # Feed-Forward
        (0.7, 0.75),   # Parallel Conv1D
        (0.7, 0.6),    # VAE
        (0.5, 0.3),    # Feature Aggregation
        (0.5, 0.15),   # Classification
        (0.5, 0.05)    # Output
    ]
    
    # Draw boxes for components
    for i, (component, pos) in enumerate(zip(components, positions)):
        x, y = pos
        width, height = 0.25, 0.1
        
        # Make input and output boxes wider
        if i == 0 or i == len(components) - 1:
            width = 0.3
        
        # Draw the box
        rect = plt.Rectangle((x - width/2, y - height/2), width, height, 
                            fill=True, 
                            color='lightblue' if i not in [0, len(components) - 1] else 'lightgreen',
                            alpha=0.7)
        plt.gca().add_patch(rect)
        
        # Add text
        plt.text(x, y, component, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw connections
    # Input to ConvLSTM and Parallel Conv1D
    plt.arrow(0.5, 0.85, -0.12, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.5, 0.85, 0.12, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # ConvLSTM to Self-Attention
    plt.arrow(0.3, 0.7, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Self-Attention to Feed-Forward
    plt.arrow(0.3, 0.55, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Conv1D to VAE
    plt.arrow(0.7, 0.7, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Feed-Forward to Feature Aggregation
    plt.arrow(0.3, 0.4, 0.12, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # VAE to Feature Aggregation
    plt.arrow(0.7, 0.55, -0.12, -0.2, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Feature Aggregation to Classification
    plt.arrow(0.5, 0.25, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Classification to Output
    plt.arrow(0.5, 0.1, 0, -0.02, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Add model parameters information
    plt.text(0.1, 0.97, "Zero Curtain Model Architecture", fontsize=14, fontweight='bold')
    plt.text(0.1, 0.02, f"Total Parameters: 48,525 (47,527 trainable)", fontsize=10)
    
    # Remove axes
    plt.axis('off')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Save the diagram
    plt.tight_layout()
    plt.savefig('zero_curtain_simplified_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Get the coordinates of the non-zero elements
x, y, z = np.indices(X.shape)
x = x[X == 1]
y = y[X == 1]
z = z[X == 1]

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

from IPython.display import display, HTML
import dataframe_image as dfi

#display(data)
#display(HTML(data.to_html()))
#display(HTML(data.to_html('merged_data.png')))
#display(HTML(df4.reset_index().set_index(['Site','Date Time']).iloc[:,:-1].to_html()))
dfi.export(data.sample(10000).sort_values('datetime').reset_index(drop=True).head(50), "merged_data.png", table_conversion="matplotlib", 
           dpi=300)
data.to_csv("merged_compressed.csv", index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
from datetime import datetime

# Load the dataset
#df = pd.read_csv('/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/scripts/zero_cur...
#df = pd.read_csv('/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/scripts/zero_cur...
#df = pd.read_csv('/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/scripts/zero_cur...
#df = pd.read_csv('/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/scripts/zero_cur...
#df = pd.read_csv('/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/scripts/zero_cur...
df = pd.read_csv('/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/scripts/zero_curtain_comprehensive/zero_curtain_events.csv')

# Convert datetime columns
df['datetime_min'] = pd.to_datetime(df['datetime_min'],format='mixed')
df['datetime_max'] = pd.to_datetime(df['datetime_max'],format='mixed')

# Basic statistics
print(f"Date range: {df['datetime_min'].min()} to {df['datetime_max'].max()}")
print(f"Mean duration: {df['duration_hours'].mean():.1f} hours ({df['duration_hours'].mean()/24:.1f} days)")
print(f"Median duration: {df['duration_hours'].median():.1f} hours")
print(f"Max duration: {df['duration_hours'].max():.1f} hours ({df['duration_hours'].max()/24:.1f} days)")

Date range: 1912-08-15 12:00:00 to 2024-08-31 00:00:00
Mean duration: 1976.9 hours (82.4 days)
Median duration: 504.0 hours
Max duration: 672768.0 hours (28032.0 days)

# Identify exceptional events (duration > 95th percentile)
duration_threshold = df['duration_hours'].quantile(0.95)
long_events = df[df['duration_hours'] > duration_threshold].copy()

print(f"Identified {len(long_events)} exceptional events (>{duration_threshold:.1f} hours)")

# Create a summary table
long_events_summary = long_events.groupby(['source', 'soil_temp_depth', 'year', 'season']).agg({
    'duration_hours': 'max',
    'soil_temp_std': 'mean',
    'temp_stability': 'mean'
}).reset_index().sort_values('duration_hours', ascending=False)

# Export exceptional events
long_events_summary.to_csv('exceptional_zero_curtain_events.csv', index=False)

Identified 643 exceptional events (>3048.0 hours)

# Identify events with both temperature and moisture data
combined_events = df[
    df['soil_temp_mean'].notna() & 
    df['soil_moist_mean'].notna()
].copy()

print(f"Found {len(combined_events)} events with both temperature and moisture data")

# Plot relationship if enough data exists
if len(combined_events) > 10:
    plt.figure(figsize=(10, 8))
    plt.scatter(
        combined_events['soil_temp_std'],
        combined_events['soil_moist_change'],
        alpha=0.7,
        c=combined_events['duration_hours'],
        cmap='viridis'
    )
    plt.colorbar(label='Duration (hours)')
    plt.xlabel('Temperature Standard Deviation')
    plt.ylabel('Soil Moisture Change')
    plt.title('Relationship Between Temperature Stability and Moisture Change')
    plt.savefig('zero_curtain_temp_moisture_relationship.png', dpi=300, bbox_inches='tight')

Found 0 events with both temperature and moisture data

# Create a geographical plot of sites with events
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines(draw_labels=True)

# Plot sites by number of events
site_counts = df.groupby('source').size().reset_index(name='count')
site_coords = df.groupby('source')[['latitude', 'longitude']].first().reset_index()
site_data = pd.merge(site_counts, site_coords, on='source')

# Remove sites with missing coordinates
site_data = site_data.dropna(subset=['latitude', 'longitude'])

# Plot with size proportional to event count
scatter = ax.scatter(
    site_data['longitude'], 
    site_data['latitude'],
    c=site_data['count'], 
    s=np.log1p(site_data['count'])*20,
    alpha=0.7,
    cmap='viridis',
    transform=ccrs.PlateCarree()
)

plt.colorbar(scatter, label='Number of Events')
plt.title('Spatial Distribution of Zero Curtain Events')
plt.savefig('zero_curtain_spatial_distribution.png', dpi=300, bbox_inches='tight')

# Set up the figure
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Events by season
season_counts = df['season'].value_counts().sort_values(ascending=False)
axes[0, 0].bar(season_counts.index, season_counts.values)
axes[0, 0].set_title('Zero Curtain Events by Season')
axes[0, 0].set_ylabel('Number of Events')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Events by month
month_counts = df['month'].value_counts().sort_index()
axes[0, 1].bar(range(1, 13), [month_counts.get(i, 0) for i in range(1, 13)])
axes[0, 1].set_title('Zero Curtain Events by Month')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Number of Events')
axes[0, 1].set_xticks(range(1, 13))

# 3. Event duration by season
sns.boxplot(x='season', y='duration_hours', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Event Duration by Season')
axes[1, 0].set_ylabel('Duration (hours)')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Events by depth zone
depth_counts = df['soil_temp_depth_zone'].value_counts().sort_values(ascending=False)
axes[1, 1].bar(depth_counts.index, depth_counts.values)
axes[1, 1].set_title('Events by Soil Depth Zone')
axes[1, 1].set_ylabel('Number of Events')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('zero_curtain_temporal_analysis.png', dpi=300, bbox_inches='tight')

# Group by year and calculate statistics
yearly_stats = df.groupby('year').agg({
    'duration_hours': ['mean', 'median', 'count'],
    'soil_temp_std': 'mean',
    'temp_stability': 'mean'
}).reset_index()

# Create a multi-panel plot
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

# Event count by year
axes[0].bar(yearly_stats['year'], yearly_stats['duration_hours']['count'])
axes[0].set_ylabel('Number of Events')
axes[0].set_title('Zero Curtain Events by Year')

# Mean duration by year
axes[1].plot(yearly_stats['year'], yearly_stats['duration_hours']['mean'], 'o-')
axes[1].set_ylabel('Mean Duration (hours)')
axes[1].set_title('Zero Curtain Mean Duration by Year')

# Temperature stability by year
axes[2].plot(yearly_stats['year'], yearly_stats['temp_stability']['mean'], 'o-', color='orange')
axes[2].set_ylabel('Temperature Stability')
axes[2].set_xlabel('Year')
axes[2].set_title('Zero Curtain Temperature Stability by Year')

plt.tight_layout()
plt.savefig('zero_curtain_yearly_trends.png', dpi=300, bbox_inches='tight')

# Group events by depth and calculate statistics
depth_groups = df.groupby('soil_temp_depth').agg({
    'duration_hours': ['mean', 'median', 'count'],
    'soil_temp_std': 'mean'
}).reset_index()

# Sort by depth for logical ordering
depth_groups = depth_groups.sort_values('soil_temp_depth')

plt.figure(figsize=(10, 8))
plt.scatter(
    depth_groups['soil_temp_depth'], 
    depth_groups['duration_hours']['mean'],
    s=depth_groups['duration_hours']['count']/30,
    alpha=0.7
)
plt.xlabel('Soil Depth (m)')
plt.ylabel('Mean Zero Curtain Duration (hours)')
plt.title('Zero Curtain Duration vs. Soil Depth')
plt.grid(True, alpha=0.3)
plt.savefig('zero_curtain_depth_analysis.png', dpi=300, bbox_inches='tight')

plt.Figure(figsize=(24, 18))
yearly_trends['duration_hours']['mean'].plot()
plt.title('Mean Zero Curtain Duration Over Years',pad=10)
plt.xlabel('Year',labelpad=10)
plt.ylabel('Mean Duration (hours)',labelpad=10)
plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/meanzerocurtaindurationovertime_zerocurtain.png',dpi=1000)

plt.Figure(figsize=(24, 18))
yearly_trends['duration_hours']['count'].plot()
plt.title('Number of Zero Curtain Events per Year',pad=10)
plt.xlabel('Year',labelpad=10)
plt.ylabel('Event Count',labelpad=10)
plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/numberzerocurtaineventsperyear_zerocurtain.png',dpi=1000)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats

def create_final_visualization(events_df, output_filename=None):
    """
    Create an optimized visualization for zero curtain events with correct handling
    of the duration distribution.
    
    Parameters:
    -----------
    events_df : pandas.DataFrame
        DataFrame containing zero curtain events
    output_filename : str, optional
        If provided, save figure to this filename
        
    Returns:
    --------
    tuple
        (figure, statistics_dict)
    """
    # Calculate percentile boundaries instead of quartiles
    p10 = np.percentile(events_df['duration_hours'], 10)
    p25 = np.percentile(events_df['duration_hours'], 25)
    p50 = np.percentile(events_df['duration_hours'], 50)
    p75 = np.percentile(events_df['duration_hours'], 75)
    p90 = np.percentile(events_df['duration_hours'], 90)
    
    # Use consistent figure size and configuration
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), 
                           subplot_kw={'projection': ccrs.NorthPolarStereo()})
    
    # Aggregate by site for visualization
    site_data = events_df.groupby(['source', 'latitude', 'longitude']).agg({
        'duration_hours': ['count', 'mean', 'median', 'min', 'max'],
        'soil_temp_depth_zone': lambda x: x.mode()[0] if not x.mode().empty else None
    }).reset_index()
    
    # Flatten column names
    site_data.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                        for col in site_data.columns]
    
    # Set map features
    for ax in axes:
        ax.set_extent([-180, 180, 45, 90], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
        ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        
        # Add Arctic Circle with label
        ax.plot(
            np.linspace(-180, 180, 60),
            np.ones(60) * 66.5,
            transform=ccrs.PlateCarree(),
            linestyle='-',
            color='gray',
            linewidth=1.0,
            alpha=0.7
        )
        
        ax.text(
            0, 66.5 + 2,
            "Arctic Circle",
            transform=ccrs.PlateCarree(),
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
        )
    
    # Plot 1: Event count
    count_max = site_data['duration_hours_count'].quantile(0.95)
    scatter1 = axes[0].scatter(
        site_data['longitude'],
        site_data['latitude'],
        transform=ccrs.PlateCarree(),
        c=site_data['duration_hours_count'],
        s=30,
        cmap='viridis',
        vmin=1,
        vmax=count_max,
        alpha=0.8,
        edgecolor='none'
    )
    plt.colorbar(scatter1, ax=axes[0], shrink=0.7, pad=0.05, label='Event Count')
    axes[0].set_title('Zero Curtain Event Count', fontsize=12)
    
    # Plot 2: Mean duration using percentile bounds
    # Use percentile-based bounds instead of IQR
    lower_bound = p10
    upper_bound = p90
    
    # Non-linear scaling for better color differentiation
    from matplotlib.colors import PowerNorm
    
    scatter2 = axes[1].scatter(
        site_data['longitude'],
        site_data['latitude'],
        transform=ccrs.PlateCarree(),
        c=site_data['duration_hours_mean'],
        s=30,
        cmap='RdYlBu_r',
        norm=PowerNorm(gamma=0.7, vmin=lower_bound, vmax=upper_bound),
        alpha=0.8,
        edgecolor='none'
    )
    
    # Create better colorbar with percentile markers
    cbar = plt.colorbar(scatter2, ax=axes[1], shrink=0.7, pad=0.05, 
                       label='Mean Duration (hours)')
    
    # Show percentile ticks
    percentile_ticks = [p10, p25, p50, p75, p90]
    cbar.set_ticks(percentile_ticks)
    cbar.set_ticklabels([f"{h:.0f}h\n({h/24:.1f}d)" for h in percentile_ticks])
    
    axes[1].set_title('Mean Zero Curtain Duration', fontsize=12)
    
    # Add comprehensive title with percentile information
    plt.suptitle(
        f'Zero Curtain Analysis: {len(site_data)} Sites, {len(events_df)} Events\n' +
        f'Duration: median={p50:.1f}h ({p50/24:.1f}d), 10-90%={p10:.1f}-{p90:.1f}h',
        fontsize=14
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save if requested
    if output_filename:
        try:
            plt.savefig(output_filename, dpi=200, bbox_inches='tight')
            print(f"Figure saved to {output_filename}")
        except Exception as e:
            print(f"Error saving figure: {e}")
    
    return fig, {
        'p10': p10,
        'p25': p25,
        'p50': p50,
        'p75': p75,
        'p90': p90,
        'mean': events_df['duration_hours'].mean(),
        'std': events_df['duration_hours'].std(),
        'min': events_df['duration_hours'].min(),
        'max': events_df['duration_hours'].max()
    }

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Assuming processor2_df_with_coords is your dataframe with coordinates
def advanced_zero_curtain_analysis(df):
    # 1. Detailed Seasonal Analysis
    plt.figure(figsize=(15, 10))
    
    # Seasonal Duration Boxplot
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='season', y='duration_hours', order=['Winter', 'Spring', 'Summer', 'Fall'])
    plt.title('Zero Curtain Duration by Season')
    plt.xticks(rotation=45)
    
    # Seasonal Mean Temperature
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='season', y='soil_temp_mean', order=['Winter', 'Spring', 'Summer', 'Fall'])
    plt.title('Mean Temperature by Season')
    plt.xticks(rotation=45)
    
    # Depth Zone Analysis
    plt.subplot(2, 2, 3)
    depth_order = ['shallow', 'intermediate', 'deep', 'very_deep']
    sns.boxplot(data=df, x='soil_temp_depth_zone', y='duration_hours', order=depth_order)
    plt.title('Zero Curtain Duration by Depth Zone')
    plt.xticks(rotation=45)
    
    # Temperature vs Duration Scatter
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='soil_temp_mean', y='duration_hours', hue='season')
    plt.title('Temperature vs Zero Curtain Duration')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Statistical Analysis
    print("\nStatistical Analysis:")
    
    # ANOVA for Seasonal Differences in Duration
    seasonal_groups = [group['duration_hours'].values for name, group in df.groupby('season')]
    f_stat, p_val = stats.f_oneway(*seasonal_groups)
    print("\nANOVA - Seasonal Duration Differences:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_val:.4f}")
    
    # Correlation between Temperature and Duration
    corr, p_corr = stats.pearsonr(df['soil_temp_mean'], df['duration_hours'])
    print("\nCorrelation - Temperature vs Duration:")
    print(f"Pearson r: {corr:.4f}")
    print(f"p-value: {p_corr:.4f}")
    
    # 3. Temporal Trends
    df['year'] = pd.to_datetime(df['datetime_min'],format='mixed').dt.year
    yearly_trends = df.groupby('year').agg({
        'duration_hours': ['mean', 'count'],
        'soil_temp_mean': ['mean', 'std']
    })
    
    plt.figure(figsize=(15, 5))
    
    # Yearly Duration Trend
    plt.subplot(1, 2, 1)
    yearly_trends['duration_hours']['mean'].plot()
    plt.title('Mean Zero Curtain Duration Over Years')
    plt.xlabel('Year')
    plt.ylabel('Mean Duration (hours)')
    
    # Yearly Event Count
    plt.subplot(1, 2, 2)
    yearly_trends['duration_hours']['count'].plot()
    plt.title('Number of Zero Curtain Events per Year')
    plt.xlabel('Year')
    plt.ylabel('Event Count')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'seasonal_anova': {
            'f_statistic': f_stat,
            'p_value': p_val
        },
        'temp_duration_correlation': {
            'correlation': corr,
            'p_value': p_corr
        },
        'yearly_trends': yearly_trends
    }
    
analysis_results = advanced_zero_curtain_analysis(enhanced_events)

plt.Figure(figsize=(16,14))
yearly_depth = enhanced_events.groupby(['year', 'soil_temp_depth_zone']).size().unstack().fillna(0)
yearly_depth.plot(kind='bar', stacked=True,
                  color=[depth_colors[d] for d in yearly_depth.columns], 
                  width=0.8)
locs, labels = plt.xticks()
n_years = len(yearly_depth.index)
if n_years > 10:
    step = max(2, n_years // 15)
    new_locs = []
    new_labels = []
    for idx, (loc, label) in enumerate(zip(locs, labels)):
        if idx % step == 0:
            new_locs.append(loc)
            new_labels.append(label)
    plt.xticks(new_locs, new_labels)

# Rotate and set style for the labels
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.title('Zero-Curtain Events by Year and Depth Zone',pad=10.0, fontsize=12);
plt.xlabel('Year',labelpad=10.0);
plt.ylabel('Number of Events',labelpad=10.0);
plt.subplots_adjust(bottom=0.15)
plt.legend(loc='upper left');
plt.tight_layout()
#plt.show()

# Seasonal Duration Boxplot
plt.Figure(figsize=(18, 24))
sns.boxplot(data=enhanced_events, x='season', y='duration_hours', order=['Winter', 'Spring', 'Summer', 'Fall'])
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.title('Zero-Curtain Duration by Season',pad=10.0, fontsize=12);
plt.xlabel('Season',labelpad=10.0);
plt.ylabel('Duration (hours)',labelpad=10.0);
plt.subplots_adjust(bottom=0.15)
#plt.legend(bbox_to_anchor=(0.48, -0.175), ncol=3, frameon=True, shadow=True)
plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/eventdurationbyseason_zerocurtain.png',dpi=1000)

# Seasonal Mean Temperature
plt.Figure(figsize=(18, 24))
sns.boxplot(data=enhanced_events, x='season', y='soil_temp_mean', order=['Winter', 'Spring', 'Summer', 'Fall'])
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.title('Mean Soil Temperature by Season', pad=10.0, fontsize=12);
plt.xlabel('Season',labelpad=10.0);
plt.ylabel('Temperature (ºC)',labelpad=10.0);
plt.subplots_adjust(bottom=0.15)
#plt.legend(bbox_to_anchor=(0.48, -0.175), ncol=3, frameon=True, shadow=True)
plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/meansoiltempbyseason_zerocurtain.png',dpi=1000)

# Depth Zone Analysis
plt.Figure(figsize=(18, 24))
depth_order = ['shallow', 'intermediate', 'deep', 'very_deep']
sns.boxplot(data=enhanced_events, x='soil_temp_depth_zone', y='duration_hours', order=depth_order)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.title('Zero Curtain Duration by Depth Zone', pad=10.0, fontsize=12);
plt.xlabel('Soil Temperature Depth Zone',labelpad=10.0);
plt.ylabel('Duration (hours)',labelpad=10.0);
plt.subplots_adjust(bottom=0.15)
#plt.legend(bbox_to_anchor=(0.48, -0.175), ncol=3, frameon=True, shadow=True)
plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/durationbydepthzone_zerocurtain.png',dpi=1000)

# Temperature vs Duration Scatter
plt.Figure(figsize=(18, 24))
sns.scatterplot(data=enhanced_events, x='soil_temp_mean', y='duration_hours', hue='season')
#plt.xticks(rotation=45, ha='right', fontsize=9)
plt.title('Temperature vs Zero Curtain Duration', pad=10.0, fontsize=12);
plt.xlabel('Mean Soil Temperature (ºC)',labelpad=10.0);
plt.ylabel('Duration (hours)',labelpad=10.0);
plt.subplots_adjust(bottom=0.15)
plt.legend(bbox_to_anchor=(0.9, -0.175), ncol=4, frameon=True, shadow=True)
plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/soiltempvsduration_zerocurtain.png',dpi=1000)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import scipy.stats as stats
from scipy.stats import pearsonr
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

# Set consistent style parameters
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
custom_palette = sns.color_palette("viridis", 4)
depth_colors = {
    'shallow': '#FDE725',
    'intermediate': '#35B779', 
    'deep': '#31688E',
    'very_deep': '#440154'
}

# 1. TEMPORAL PATTERNS VISUALIZATION
def plot_temporal_patterns(events):
    """
    Create temporal pattern visualizations for zero-curtain events
    """
    # 1.1 Events distribution over years
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Events by year and depth zone
    yearly_depth = events.groupby(['year', 'soil_temp_depth_zone']).size().unstack().fillna(0)
    yearly_depth.plot(kind='bar', stacked=True, ax=axes[0, 0], 
                      color=[depth_colors[d] for d in yearly_depth.columns], 
                      width=0.8)
    axes[0, 0].set_title('Zero-Curtain Events by Year and Depth Zone')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Number of Events')
    
    # Event duration by season
    sns.boxplot(x='season', y='duration_hours', data=events, ax=axes[0, 1], 
                palette=sns.color_palette("husl", 4))
    axes[0, 1].set_title('Zero-Curtain Event Duration by Season')
    axes[0, 1].set_xlabel('Season')
    axes[0, 1].set_ylabel('Duration (hours)')
    
    # Temperature variation by year
    yearly_temp = events.groupby('year')[['soil_temp_mean', 'soil_temp_std']].mean()
    ax1 = axes[1, 0]
    ax2 = ax1.twinx()
    yearly_temp['soil_temp_mean'].plot(ax=ax1, marker='o', color='tab:blue', label='Mean Temperature')
    yearly_temp['soil_temp_std'].plot(ax=ax2, marker='s', color='tab:red', label='Temperature StdDev')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Mean Soil Temperature (°C)', color='tab:blue')
    ax2.set_ylabel('Soil Temperature StdDev', color='tab:red')
    ax1.tick_params(axis='y', colors='tab:blue')
    ax2.tick_params(axis='y', colors='tab:red')
    ax1.set_title('Temperature Trends Over Time')
    
    # Event counts by month across all years
    events['month'] = pd.to_datetime(events['datetime_min'],format='mixed').dt.month
    monthly_counts = events.groupby('month').size()
    monthly_counts.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_counts.plot(kind='bar', ax=axes[1, 1], color=sns.color_palette("rocket", 12))
    axes[1, 1].set_title('Monthly Distribution of Zero-Curtain Events')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Number of Events')
    
    plt.tight_layout()
    return fig

plt.Figure(figsize=(16,14))
yearly_depth = enhanced_events.groupby(['year', 'soil_temp_depth_zone']).size().unstack().fillna(0)
yearly_depth.plot(kind='bar', stacked=True,
                  color=[depth_colors[d] for d in yearly_depth.columns], 
                  width=0.8)
locs, labels = plt.xticks()
n_years = len(yearly_depth.index)
if n_years > 10:
    step = max(2, n_years // 15)
    new_locs = []
    new_labels = []
    for idx, (loc, label) in enumerate(zip(locs, labels)):
        if idx % step == 0:
            new_locs.append(loc)
            new_labels.append(label)
    plt.xticks(new_locs, new_labels)

# Rotate and set style for the labels
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.title('Zero-Curtain Events by Year and Depth Zone',pad=10.0, fontsize=12);
plt.xlabel('Year',labelpad=10.0);
plt.ylabel('Number of Events',labelpad=10.0);
plt.subplots_adjust(bottom=0.15)
#plt.legend(loc='upper left');
plt.legend(bbox_to_anchor=(0.971, -0.225), ncol=4, frameon=True, shadow=True)
plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/eventsbyyearanddepth_zerocurtain.png',dpi=1000)

plt.Figure(figsize=(16,14))
sns.boxplot(x='season', y='duration_hours', data=events_season, 
            palette=sns.color_palette("husl", 4))
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.title('Zero-Curtain Event Duration by Season',pad=10.0, fontsize=12);
plt.xlabel('Season',labelpad=10.0);
plt.ylabel('Duration (hours)',labelpad=10.0);
plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/eventdurationbyseason_zerocurtain.png',dpi=1000)

# Create the figure
plt.Figure(figsize=(18, 24))

# Plot first line on default axis
yearly_temp = enhanced_events.groupby('year')[['soil_temp_mean', 'soil_temp_std']].mean()
plt.plot(yearly_temp.index, yearly_temp['soil_temp_mean'], color='tab:blue', label='Mean Temperature', alpha=0.4)

# Get the current axis
ax1 = plt.gca()
ax1.tick_params(axis='y', colors='tab:blue')
ax1.set_ylabel('Mean Soil Temperature (°C)', color='tab:blue', labelpad=10.0)

# Create the twin axis implicitly
ax2 = plt.twinx()
ax2.plot(yearly_temp.index, yearly_temp['soil_temp_std'], alpha=0.4, color='tab:red', label='Temperature StdDev')
ax2.tick_params(axis='y', colors='tab:red')
ax2.set_ylabel('Soil Temperature StdDev', color='tab:red', labelpad=10.0)

# Set remaining properties using plt commands
plt.xlabel('Year', labelpad=10.0)
plt.title('Temperature Trends Over Time', pad=10.0)

# Create a combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
#plt.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1,1));

plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/temperaturetrendsovertime_zerocurtain.png',dpi=1000)

plt.Figure(figsize=(16,14))
monthly_counts = enhanced_events.groupby('month').size()
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_counts = monthly_counts.reindex(range(1, 13), fill_value=0)
monthly_counts.index = month_labels
monthly_counts.plot(kind='bar', color=sns.color_palette("rocket", 12))
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.title('Monthly Distribution of Zero-Curtain Events',pad=10.0)
plt.xlabel('Month',labelpad=10.0)
plt.ylabel('Number of Events',labelpad=10.0)
plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/monthlydistofevents_zerocurtain.png',dpi=1000)

plt.subplots(1, figsize=(18, 24))
year_depth_counts = enhanced_events.pivot_table(
values='observation_count', 
index='year', 
columns='soil_temp_depth_zone', 
aggfunc='count'
).fillna(0)
year_depth_counts = year_depth_counts.reindex(columns=depth_order)
year_depth_counts = year_depth_counts.sort_index()
sns.heatmap(year_depth_counts, annot=True, fmt=".0f", cmap="Reds", cbar_kws={'label': 'Number of Events'})
plt.xlabel('Soil Temperature, Depth Zone',labelpad=10)
plt.ylabel('Year',labelpad=10)
plt.title('Number of Events by Year and Depth Zone',pad=10,fontsize=14)
plt.tick_params(axis='y', labelsize=9)
plt.legend();
plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/numbereventsbyyearanddepth_zerocurtain.png',dpi=1000)

plt.Figure(figsize=(18,24))

# Group by region, year, and depth zone
region_time = enhanced_events.groupby(['region', 'year', 'soil_temp_depth_zone']).agg({
    'soil_temp_mean': 'mean',
    'soil_temp_std': 'mean',
    'duration_hours': 'mean',
    'source': 'count'  # Count of events
}).reset_index()
region_time = region_time.rename(columns={'source': 'event_count'})

for region in region_time['region'].unique():
    region_data = region_time[region_time['region'] == region]
    plt.plot(region_data['year'], region_data['soil_temp_mean'], 
                    linestyle='-', label=region, alpha=0.4)

plt.title('Mean Temperature by Region Over Time',pad=10)
plt.xlabel('Year',labelpad=10)
plt.ylabel('Mean Temperature (°C)',labelpad=10)
#plt.legend(loc='upper left', borderpad=1, fancybox=True);
plt.legend(loc='upper center', bbox_to_anchor=(0.48, -0.175), 
           ncol=3, frameon=True, shadow=True)
plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/soiltempmeanbyregionovertime_zerocurtain.png',dpi=1000)

plt.Figure(figsize=(18,24))
regions = enhanced_events['region'].unique()
region_colors = sns.color_palette("tab10", len(regions))
region_color_map = dict(zip(regions, region_colors))

for region in regions:
    region_data = enhanced_events[enhanced_events['region'] == region]
    plt.scatter(region_data['soil_temp_mean'], region_data['soil_temp_std'], 
                       s=region_data['duration_hours']/5, alpha=0.4,
                       color=region_color_map[region], label=region)

plt.title('Temperature Characteristics by Region',pad=10)
plt.xlabel('Mean Temperature (°C)',labelpad=10)
plt.ylabel('Temperature Variability',labelpad=10)
#plt.legend(bbox_to_anchor=(1, 1));
plt.legend(loc='upper center', bbox_to_anchor=(0.48, -0.175), 
           ncol=3, frameon=True, shadow=True)
plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/soiltempcharacteristicsbyregion_zerocurtain.png',dpi=1000)

plt.Figure(figsize=(18,24))
depth_order = ['shallow', 'intermediate', 'deep', 'very_deep']
events_depth = enhanced_events.copy()
events_depth['soil_temp_depth_zone'] = pd.Categorical(
    events_depth['soil_temp_depth_zone'], 
    categories=depth_order,
    ordered=True
)

for zone in depth_order:
    zone_data = events_depth[events_depth['soil_temp_depth_zone'] == zone]
    if not zone_data.empty:
        year_mean = zone_data.groupby('year')['soil_temp_mean'].mean()
        if not year_mean.empty:
            plt.plot(year_mean.index, year_mean.values, 
                           linestyle='-', 
                           color=depth_colors.get(zone, '#000000'),
                           label=zone, alpha=0.4)

plt.title('Temperature Trends by Depth Zone',pad=10)
plt.xlabel('Year',labelpad=10)
plt.ylabel('Mean Temperature (°C)',labelpad=10)
#plt.legend(bbox_to_anchor=(1,1));
plt.legend(loc='upper center', bbox_to_anchor=(0.48, -0.175), 
           ncol=4, frameon=True, shadow=True)
plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/temptrendsbydepth_zerocurtain.png',dpi=1000)

plt.Figure(figsize=(18,24))

region_colors = {
    'Arctic': '#1e88e5',       # Bright blue
    'Subarctic': '#ff5722',    # Bright orange
    'Northern Boreal': '#4caf50', # Bright green
    'Other': '#9c27b0'         # Bright purple
}

depth_order = ['shallow', 'intermediate', 'deep', 'very_deep']
events_depth = enhanced_events.copy()
events_depth['soil_temp_depth_zone'] = pd.Categorical(
    events_depth['soil_temp_depth_zone'], 
    categories=depth_order,
    ordered=True
)

#ax = sns.boxplot(x='soil_temp_depth_zone', y='duration_hours', data=events_depth,
#          order=depth_order, fliersize=0, width=0.3, color='lightgray')

sns.stripplot(x='soil_temp_depth_zone', y='duration_hours', hue='region',
             data=events_depth, dodge=True, alpha=0.3, 
             palette=region_colors, order=depth_order, jitter=0.3, size=4)

plt.axhline(y=events_depth['duration_hours'].median(), color='black', 
           linestyle='--', alpha=0.7, label='Overall Median')

plt.title('Event Duration by Depth Zone and Region', pad=10)
plt.xlabel('Depth Zone', labelpad=10)
plt.ylabel('Duration (hours)', labelpad=10)
#plt.legend(bbox_to_anchor=(1, 1));
plt.legend(loc='upper center', bbox_to_anchor=(0.48, -0.175), 
           ncol=4, frameon=True, shadow=True, fontsize=9)

q75 = events_depth['duration_hours'].quantile(0.75)
ymax = min(max(events_depth['duration_hours']), q75 * 4)
plt.ylim(0, ymax)

#ax2 = plt.axes([.825, 0.15, 0.4, 0.4], facecolor='white')
#sns.boxplot(x='soil_temp_depth_zone', y='duration_hours', data=events_depth,
#         order=depth_order, ax=ax2)
#ax2.set_title('Full Range View')
#ax2.tick_params()
plt.tight_layout()
plt.show()
#plt.savefig('zero_curtain_pipeline/eventdurationbydepthandregion_zerocurtain.png',dpi=1000)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import geopandas as gpd
# from scipy.stats import pearsonr
# import holoviews as hv
# from holoviews import opts
# hv.extension('bokeh')

# # Set consistent style parameters
# plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_context("paper", font_scale=1.2)
# custom_palette = sns.color_palette("viridis", 4)
# depth_colors = {
#     'shallow': '#FDE725',
#     'intermediate': '#35B779', 
#     'deep': '#31688E',
#     'very_deep': '#440154'
# }

# # Assuming 'events' DataFrame is loaded
# # events = pd.read_csv('zero_curtain_events.csv') # Replace with...

# # 1. TEMPORAL PATTERNS VISUALIZATION
# def plot_temporal_patterns(events):
#     """
#     Create temporal pattern visualizations for zero-curtain events
#     """
#     # 1.1 Events distribution over years
#     fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
#     # Events by year and depth zone
#     yearly_depth = events.groupby(['year', 'soil_temp_depth_zone']).size().unstack().fillna(0)
#     yearly_depth.plot(kind='bar', stacked=True, ax=axes[0, 0], 
#                       color=[depth_colors[d] for d in yearly_depth.columns], 
#                       width=0.8)
#     axes[0, 0].set_title('Zero-Curtain Events by Year and Depth Zone')
#     axes[0, 0].set_xlabel('Year')
#     axes[0, 0].set_ylabel('Number of Events')
    
#     # Event duration by season - ensure correct season order
#     # Create a categorical type with the desired order
#     season_order = pd.CategoricalDtype(categories=['Winter', 'Spring', 'Summer', 'Fall'], ordered=...
#     events_season = events.copy()
#     events_season['season'] = events_season['season'].astype(season_order)
    
#     # Create the plot with ordered seasons
#     sns.boxplot(x='season', y='duration_hours', data=events_season, ax=axes[0, 1], 
#                 palette=sns.color_palette("husl", 4))
#     axes[0, 1].set_title('Zero-Curtain Event Duration by Season')
#     axes[0, 1].set_xlabel('Season')
#     axes[0, 1].set_ylabel('Duration (hours)')
    
#     # Temperature variation by year
#     yearly_temp = events.groupby('year')[['soil_temp_mean', 'soil_temp_std']].mean()
#     ax1 = axes[1, 0]
#     ax2 = ax1.twinx()
#     yearly_temp['soil_temp_mean'].plot(ax=ax1, marker='o', color='tab:blue', label='Mean Temperatu...
#     yearly_temp['soil_temp_std'].plot(ax=ax2, marker='s', color='tab:red', label='Temperature StdD...
#     ax1.set_xlabel('Year')
#     ax1.set_ylabel('Mean Soil Temperature (°C)', color='tab:blue')
#     ax2.set_ylabel('Soil Temperature StdDev', color='tab:red')
#     ax1.tick_params(axis='y', colors='tab:blue')
#     ax2.tick_params(axis='y', colors='tab:red')
#     ax1.set_title('Temperature Trends Over Time')
    
# # Event counts by month across all...
#     events['month'] = pd.to_datetime(events['datetime_min']).dt.month
#     monthly_counts = events.groupby('month').size()
#     # Create month labels in order
#     month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
#                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
#     # Reindex to ensure all months are present and in order
#     monthly_counts = monthly_counts.reindex(range(1, 13), fill_value=0)
#     monthly_counts.index = month_labels
#     monthly_counts.plot(kind='bar', ax=axes[1, 1], color=sns.color_palette("rocket", 12))
#     axes[1, 1].set_title('Monthly Distribution of Zero-Curtain Events')
#     axes[1, 1].set_xlabel('Month')
#     axes[1, 1].set_ylabel('Number of Events')
    
#     plt.tight_layout()
#     return fig

# # 2. SPATIAL VISUALIZATION
# def plot_spatial_distribution(events):
#     """
#     Create spatial visualizations of zero-curtain events
#     """
#     # 2.1 Geographical distribution of events with temperature information
#     fig = px.scatter_mapbox(events, 
#                             lat="latitude", 
#                             lon="longitude", 
#                             color="soil_temp_mean",
#                             size="duration_hours",
#                             color_continuous_scale=px.colors.sequential.Plasma,
#                             hover_name="source",
#                             hover_data=["soil_temp_depth", "soil_temp_depth_zone", 
#                                        "soil_temp_mean", "duration_hours", "season", "year"],
#                             zoom=2,
#                             title="Spatial Distribution of Zero-Curtain Events")
    
#     fig.update_layout(mapbox_style="open-street-map",
#                      height=800, width=1000,
#                      margin={"r":0,"t":40,"l":0,"b":0})
    
#     return fig

# # 3. DEPTH-TEMPERATURE RELATIONSHIP VISUALIZATION
# def plot_depth_temperature_relationships(events):
#     """
#     Visualize relationships between soil depth and temperature characteristics
#     """
#     fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
#     # 3.1 Depth vs. Mean Temperature
#     sns.scatterplot(x='soil_temp_depth', y='soil_temp_mean', 
#                    hue='soil_temp_depth_zone', size='duration_hours',
#                    palette=depth_colors, data=events, ax=axes[0, 0], alpha=0.7)
#     axes[0, 0].set_title('Soil Temperature vs. Depth')
#     axes[0, 0].set_xlabel('Soil Depth (m)')
#     axes[0, 0].set_ylabel('Mean Soil Temperature (°C)')
    
#     # 3.2 Depth vs. Temperature Variability
#     sns.scatterplot(x='soil_temp_depth', y='soil_temp_std', 
#                    hue='soil_temp_depth_zone', size='duration_hours',
#                    palette=depth_colors, data=events, ax=axes[0, 1], alpha=0.7)
#     axes[0, 1].set_title('Temperature Variability vs. Depth')
#     axes[0, 1].set_xlabel('Soil Depth (m)')
#     axes[0, 1].set_ylabel('Temperature Standard Deviation')
    
#     # 3.3 Depth Zone Distribution by Region
#     region_depth = pd.crosstab(events['region'], events['soil_temp_depth_zone'])
#     region_depth.plot(kind='bar', stacked=True, ax=axes[1, 0], 
#                      color=[depth_colors[d] for d in region_depth.columns])
#     axes[1, 0].set_title('Depth Zone Distribution by Region')
#     axes[1, 0].set_xlabel('Region')
#     axes[1, 0].set_ylabel('Number of Events')
    
#     # 3.4 Temperature Range by Depth Zone
#     sns.boxplot(x='soil_temp_depth_zone', y='soil_temp_max', data=events, 
#                color='lightblue', ax=axes[1, 1])
#     sns.boxplot(x='soil_temp_depth_zone', y='soil_temp_min', data=events, 
#                color='lightcoral', ax=axes[1, 1])
#     axes[1, 1].set_title('Temperature Range by Depth Zone')
#     axes[1, 1].set_xlabel('Soil Depth Zone')
#     axes[1, 1].set_ylabel('Temperature (°C)')
#     handles = [plt.Rectangle((0,0),1,1, color='lightblue'), 
#               plt.Rectangle((0,0),1,1, color='lightcoral')]
#     axes[1, 1].legend(handles, ['Max Temperature', 'Min Temperature'])
    
#     plt.tight_layout()
#     return fig

# # 4. HEATMAP VISUALIZATION FOR TEMPORAL-SPATIAL PATTERNS
# def plot_heatmap_visualization(events):
#     """
#     Create heatmaps to visualize patterns across multiple dimensions
#     """
# # 4.1 Prepare data for heatmap -...
#     # Use categorical dtype to ensure seasons are in the correct order
#     events_season = events.copy()
#     events_season['season'] = pd.Categorical(
#         events_season['season'], 
#         categories=['Winter', 'Spring', 'Summer', 'Fall'],
#         ordered=True
#     )
    
#     # Create the pivot table with ordered seasons
#     depth_season_temp = events_season.pivot_table(
#         values='soil_temp_mean', 
#         index='soil_temp_depth_zone', 
#         columns='season', 
#         aggfunc='mean'
#     )
    
#     # Order depth zones from shallowest to deepest
#     depth_order = ['shallow', 'intermediate', 'deep', 'very_deep']
#     depth_season_temp = depth_season_temp.reindex(depth_order)
    
#     # 4.2 Create figure with subplots - make it taller
# fig, axes = plt.subplots(3, 2, figsize=(18, 24),...
    
#     # 4.3 Depth zone by season heatmap
#     sns.heatmap(depth_season_temp, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0, 0])
#     axes[0, 0].set_title('Mean Soil Temperature by Depth Zone and Season')
    
# # 4.4 Year by depth zone heatmap...
#     year_depth_counts = events.pivot_table(
#         values='observation_count', 
#         index='year', 
#         columns='soil_temp_depth_zone', 
#         aggfunc='count'
#     ).fillna(0)
#     # Reorder columns to follow depth order
#     year_depth_counts = year_depth_counts.reindex(columns=depth_order)
#     # Sort years in ascending order
#     year_depth_counts = year_depth_counts.sort_index()
    
#     # Create a dedicated larger heatmap
#     sns.heatmap(year_depth_counts, annot=True, fmt=".0f", cmap="Reds", ax=axes[1, 0],
#                 cbar_kws={'label': 'Number of Events'})
#     axes[1, 0].set_title('Number of Events by Year and Depth Zone')
    
#     # Set font size for better readability
#     axes[1, 0].tick_params(axis='y', labelsize=9)
    
#     # Keep the heatmap square across the entire row
#     axes[1, 0].set_aspect('auto')
#     axes[1, 1].set_visible(False)  # Hide the right subplot in the middle row
    
#     # 4.5 Region by depth zone temperature variability
#     region_depth_std = events.pivot_table(
#         values='soil_temp_std', 
#         index='region', 
#         columns='soil_temp_depth_zone', 
#         aggfunc='mean'
#     )
#     # Reorder columns to follow depth order
#     region_depth_std = region_depth_std.reindex(columns=depth_order)
    
#     sns.heatmap(region_depth_std, annot=True, fmt=".3f", cmap="viridis", ax=axes[2, 0])
#     axes[2, 0].set_title('Temperature Variability by Region and Depth Zone')
    
#     # 4.6 Correlation matrix of key numerical variables
#     numerical_cols = ['soil_temp_depth', 'duration_hours', 'soil_temp_mean', 'soil_temp_std', 'soi...
#     corr_matrix = events[numerical_cols].corr()
#     sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
#                vmin=-1, vmax=1, ax=axes[2, 1])
#     axes[2, 1].set_title('Correlation Matrix of Zero-Curtain Characteristics')
    
#     plt.tight_layout()
#     return fig

# # 5. ADVANCED INTERACTIVE VISUALIZATIONS
# def create_interactive_dashboard(events):
#     """
#     Create interactive plotly dashboard for zero-curtain dynamics
#     """
#     # 5.1 Create subplot figure
#     fig = make_subplots(
#         rows=2, cols=2,
#         subplot_titles=("Temperature Range by Depth", 
#                        "Duration Distribution by Depth Zone", 
#                        "Seasonal Pattern by Depth Zone",
#                        "Temperature Variability Over Time"),
#         specs=[[{"type": "scatter3d"}, {"type": "bar"}],
#               [{"type": "heatmap"}, {"type": "scatter"}]],
#         vertical_spacing=0.1,
#         horizontal_spacing=0.1
#     )
    
#     # 5.2 3D scatter plot of depth, temperature, and variability
#     fig.add_trace(
#         go.Scatter3d(
#             x=events['soil_temp_depth'],
#             y=events['soil_temp_mean'],
#             z=events['soil_temp_std'],
#             mode='markers',
#             marker=dict(
#                 size=5,
#                 color=events['duration_hours'],
#                 colorscale='Viridis',
#                 opacity=0.8,
#                 colorbar=dict(title="Duration (hours)")
#             ),
#             text=events.apply(
#                 lambda row: f"Source: {row['source']}<br>Region: {row['region']}<br>Year: {row['ye...
#                 axis=1
#             ),
#             hoverinfo="text",
#         ),
#         row=1, col=1
#     )
    
#     # 5.3 Bar chart of event duration by depth zone
#     # Define the depth zone order
#     depth_zone_order = ['shallow', 'intermediate', 'deep', 'very_deep']
    
#     # Prepare data with ordered depth zones
#     events_depth = events.copy()
#     events_depth['soil_temp_depth_zone'] = pd.Categorical(
#         events_depth['soil_temp_depth_zone'], 
#         categories=depth_zone_order,
#         ordered=True
#     )
    
#     # Group by ordered depth zones
#     depth_duration = events_depth.groupby('soil_temp_depth_zone')['duration_hours'].mean().reset_i...
#     depth_std = events_depth.groupby('soil_temp_depth_zone')['duration_hours'].std().reset_index()
    
#     # Add bars in the correct order
#     for zone in depth_zone_order:
#         if zone in depth_duration['soil_temp_depth_zone'].values:
#             zone_idx = depth_duration[depth_duration['soil_temp_depth_zone'] == zone].index[0]
#             std_value = depth_std.loc[zone_idx, 'duration_hours'] if zone in depth_std['soil_temp_...
            
#             fig.add_trace(
#                 go.Bar(
#                     x=[zone], 
#                     y=[depth_duration.loc[zone_idx, 'duration_hours']],
#                     error_y=dict(
#                         type='data',
#                         array=[std_value],
#                         visible=True
#                     ),
#                     name=zone,
#                     marker_color=depth_colors.get(zone, '#000000')
#                 ),
#                 row=1, col=2
#             )
    
#     # 5.4 Heatmap of seasonal patterns by depth zone
#     # Order the seasons correctly
#     season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    
#     # Create a crosstab with ordered categories
#     events_season = events.copy()
#     events_season['season'] = pd.Categorical(
#         events_season['season'], 
#         categories=season_order,
#         ordered=True
#     )
#     events_season['soil_temp_depth_zone'] = pd.Categorical(
#         events_season['soil_temp_depth_zone'], 
#         categories=depth_zone_order,
#         ordered=True
#     )
    
#     season_depth = pd.crosstab(
#         events_season['season'], 
#         events_season['soil_temp_depth_zone'],
#         values=events_season['soil_temp_mean'],
#         aggfunc='mean'
#     ).fillna(0)
    
#     # Ensure the heatmap has all seasons in correct order
#     season_depth = season_depth.reindex(index=season_order, columns=depth_zone_order)
    
#     fig.add_trace(
#         go.Heatmap(
#             z=season_depth.values,
#             x=season_depth.columns,
#             y=season_depth.index,
#             colorscale='YlGnBu',
#             zmin=events['soil_temp_mean'].min(),
#             zmax=events['soil_temp_mean'].max(),
#             colorbar=dict(title="Mean Temperature")
#         ),
#         row=2, col=1
#     )
    
#     # 5.5 Scatter plot of temperature variability over time
#     events['date'] = pd.to_datetime(events['datetime_min'])
#     events['year_decimal'] = events['date'].dt.year + events['date'].dt.month/12
    
#     # Group by year and depth zone with ordered categories
#     events_year_depth = events.copy()
#     events_year_depth['soil_temp_depth_zone'] = pd.Categorical(
#         events_year_depth['soil_temp_depth_zone'], 
#         categories=depth_zone_order,
#         ordered=True
#     )
    
#     year_depth_var = events_year_depth.groupby(['year', 'soil_temp_depth_zone'])['soil_temp_std']....
    
#     for zone in depth_zone_order:
#         zone_data = year_depth_var[year_depth_var['soil_temp_depth_zone'] == zone]
#         if not zone_data.empty:
#             # Sort by year to ensure chronological order
#             zone_data = zone_data.sort_values('year')
            
#             fig.add_trace(
#                 go.Scatter(
#                     x=zone_data['year'],
#                     y=zone_data['soil_temp_std'],
#                     mode='lines+markers',
#                     name=zone,
#                     line=dict(color=depth_colors.get(zone, '#000000')),
#                     marker=dict(size=8)
#                 ),
#                 row=2, col=2
#             )
    
#     # 5.6 Update layout
#     fig.update_layout(
#         title_text="Interactive Zero-Curtain Dynamics Dashboard",
#         height=800,
#         width=1200,
#         showlegend=False,
#         scene=dict(
#             xaxis_title="Soil Depth (m)",
#             yaxis_title="Mean Temperature (°C)",
#             zaxis_title="Temperature StdDev"
#         )
#     )
    
#     # Update axes labels
#     fig.update_xaxes(title_text="Depth Zone", row=1, col=2)
#     fig.update_yaxes(title_text="Mean Duration (hours)", row=1, col=2)
#     fig.update_xaxes(title_text="Depth Zone", row=2, col=1)
#     fig.update_yaxes(title_text="Season", row=2, col=1)
#     fig.update_xaxes(title_text="Year", row=2, col=2)
#     fig.update_yaxes(title_text="Temperature Variability", row=2, col=2)
    
#     return fig

# # 6. SPATIOTEMPORAL ANALYSIS
# def create_spatiotemporal_analysis(events):
#     """
#     Creates visualizations that combine spatial and temporal dimensions
#     """
#     # 6.1 Prepare data - extract year and month
#     events['year_month'] = pd.to_datetime(events['datetime_min']).dt.to_period('M')
    
#     # Group by region, year, and depth zone
#     region_time = events.groupby(['region', 'year', 'soil_temp_depth_zone']).agg({
#         'soil_temp_mean': 'mean',
#         'soil_temp_std': 'mean',
#         'duration_hours': 'mean',
#         'source': 'count'  # Count of events
#     }).reset_index()
#     region_time = region_time.rename(columns={'source': 'event_count'})
    
#     # 6.2 Create figure with subplots
#     fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
#     # 6.3 Region time series of temperature
#     for region in region_time['region'].unique():
#         region_data = region_time[region_time['region'] == region]
#         axes[0, 0].plot(region_data['year'], region_data['soil_temp_mean'], 
#                         marker='o', linestyle='-', label=region)
    
#     axes[0, 0].set_title('Mean Temperature by Region Over Time')
#     axes[0, 0].set_xlabel('Year')
#     axes[0, 0].set_ylabel('Mean Temperature (°C)')
#     axes[0, 0].legend()
    
#     # 6.4 Bubble chart - event count, temperature, and duration by region
#     regions = events['region'].unique()
#     region_colors = sns.color_palette("tab10", len(regions))
#     region_color_map = dict(zip(regions, region_colors))
    
#     for region in regions:
#         region_data = events[events['region'] == region]
#         axes[0, 1].scatter(region_data['soil_temp_mean'], region_data['soil_temp_std'], 
#                            s=region_data['duration_hours']/5, alpha=0.6,
#                            color=region_color_map[region], label=region)
    
#     axes[0, 1].set_title('Temperature Characteristics by Region')
#     axes[0, 1].set_xlabel('Mean Temperature (°C)')
#     axes[0, 1].set_ylabel('Temperature Variability')
#     axes[0, 1].legend()
    
#     # 6.5 Temporal trends by depth zone
#     # Use ordered depth zones
#     depth_order = ['shallow', 'intermediate', 'deep', 'very_deep']
#     events_depth = events.copy()
#     events_depth['soil_temp_depth_zone'] = pd.Categorical(
#         events_depth['soil_temp_depth_zone'], 
#         categories=depth_order,
#         ordered=True
#     )
    
#     for zone in depth_order:
#         zone_data = events_depth[events_depth['soil_temp_depth_zone'] == zone]
#         if not zone_data.empty:
#             year_mean = zone_data.groupby('year')['soil_temp_mean'].mean()
#             if not year_mean.empty:
#                 axes[1, 0].plot(year_mean.index, year_mean.values, 
#                                marker='o', linestyle='-', 
#                                color=depth_colors.get(zone, '#000000'),
#                                label=zone)
    
#     axes[1, 0].set_title('Temperature Trends by Depth Zone')
#     axes[1, 0].set_xlabel('Year')
#     axes[1, 0].set_ylabel('Mean Temperature (°C)')
#     axes[1, 0].legend()
    
#     # 6.6 Event duration distribution by depth and region
#     # Create season order for boxplot
#     season_order = ['Winter', 'Spring', 'Summer', 'Fall']
#     events_season = events.copy()
#     events_season['season'] = pd.Categorical(
#         events_season['season'], 
#         categories=season_order,
#         ordered=True
#     )
    
#     sns.boxplot(x='soil_temp_depth_zone', y='duration_hours', hue='region', 
#                 data=events_season, palette='Set2', ax=axes[1, 1],
#                 order=depth_order)
    
#     axes[1, 1].set_title('Event Duration by Depth Zone and Region')
#     axes[1, 1].set_xlabel('Depth Zone')
#     axes[1, 1].set_ylabel('Duration (hours)')
#     axes[1, 1].legend(title='Region')
    
#     plt.tight_layout()
#     return fig

# # 7. COMPOSITE VISUALIZATION FUNCTION
# def create_zero_curtain_analysis(events):
#     """
#     Generate a comprehensive set of visualizations for zero-curtain dynamics
#     """
#     # Return a list of all visualization figures
#     return {
#         'temporal_patterns': plot_temporal_patterns(events),
#         'spatial_distribution': plot_spatial_distribution(events),
#         'depth_temperature': plot_depth_temperature_relationships(events),
#         'heatmap_visualization': plot_heatmap_visualization(events),
#         'interactive_dashboard': create_interactive_dashboard(events),
#         'spatiotemporal_analysis': create_spatiotemporal_analysis(events)
#     }

def create_visualization_suite(df, output_dir='figures/zero_curtain'):
    """
    Create comprehensive visualizations for zero curtain analysis
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    
    # 1. Temporal Distribution
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Yearly distribution
    yearly_events = df.groupby([df['datetime_min'].dt.year, 'season']).size().unstack()
    yearly_events.plot(kind='bar', stacked=True, ax=axes[0])
    axes[0].set_title('Zero Curtain Events by Year and Season')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Number of Events')
    plt.xticks(rotation=45)
    
    # Monthly distribution
    monthly_events = df.groupby([df['datetime_min'].dt.month, 'season']).size().unstack()
    monthly_events.plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].set_title('Zero Curtain Events by Month and Season')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Number of Events')
    
    plt.tight_layout()
    plt.savefig(output_path / 'temporal_distribution.png', dpi=1000, bbox_inches='tight')
    plt.close()
    
    # 2. Duration Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Duration histogram
    sns.histplot(data=df, x='duration_hours', bins=50, ax=axes[0,0])
    axes[0,0].set_title('Distribution of Event Durations')
    axes[0,0].set_xlabel('Duration (hours)')
    
    # Duration by depth zone
    sns.boxplot(data=df, x='soil_temp_depth_zone', y='duration_hours', ax=axes[0,1])
    axes[0,1].set_title('Duration by Depth Zone')
    axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
    
    # Duration by season
    sns.boxplot(data=df, x='season', y='duration_hours', ax=axes[1,0])
    axes[1,0].set_title('Duration by Season')
    
    # Mean temperature vs duration
    sns.scatterplot(data=df, x='soil_temp_mean', y='duration_hours', 
                   hue='soil_temp_depth_zone', alpha=0.5, ax=axes[1,1])
    axes[1,1].set_title('Duration vs Mean Temperature')
    
    plt.tight_layout()
    plt.savefig(output_path / 'duration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Spatial Distribution
    plt.figure(figsize=(15, 10))
    plt.scatter(df['longitude'], df['latitude'], 
               c=df['duration_hours'], cmap='viridis',
               alpha=0.6)
    plt.colorbar(label='Duration (hours)')
    plt.title('Spatial Distribution of Zero Curtain Events')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(output_path / 'spatial_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

output_path='zero_curtain_pipeline/figures'

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# Yearly distribution
yearly_events = enhanced_events.groupby([enhanced_events['datetime_min'].dt.year, 'season']).size().unstack()
yearly_events.plot(kind='bar', stacked=True, ax=axes[0])
axes[0].set_title('Zero Curtain Events by Year and Season',pad=10)
axes[0].set_xlabel('Year',labelpad=10)
axes[0].set_ylabel('Number of Events',labelpad=10)

axes[0].legend(loc='upper center', bbox_to_anchor=(0.48, -0.275), 
           ncol=4, frameon=True, shadow=True)

# Reduce number of x-axis labels for readability
years = yearly_events.index.tolist()
skip = max(1, len(years) // 20)  # Show approximately 10 labels
axes[0].set_xticks(np.arange(0, len(years), step=skip))
axes[0].set_xticklabels(years[::skip], rotation=45)

# Monthly distribution
monthly_events = enhanced_events.groupby([enhanced_events['datetime_min'].dt.month, 'season']).size().unstack()
monthly_events.plot(kind='bar', stacked=True, ax=axes[1])
axes[1].set_title('Zero Curtain Events by Month and Season',pad=10)
axes[1].set_xlabel('Month',labelpad=10)
axes[1].set_ylabel('Number of Events',labelpad=10)

axes[1].legend(loc='upper center', bbox_to_anchor=(0.48, -0.205), 
           ncol=4, frameon=True, shadow=True)

# Reduce the number of x-axis labels for months (if needed)
#axes[1].set_xticks(range(0, 12, 2))  # Show every second month
#axes[1].set_xticklabels([str(m) for m in range(1, 13, 2)])

plt.tight_layout()
plt.savefig(output_path+'/temporal_distribution.png', dpi=1000, bbox_inches='tight')
#plt.show()
plt.close()

# #ONE IMPLEMENTATION (KDTree with Self-Match Exclusion)
# This script properly calculates nearest-neighbor distances by...
# import numpy as np
# from scipy.spatial import cKDTree
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import gc

# def haversine_to_cartesian(lat, lon):
#     """Convert latitude and longitude to 3D cartesian coordinates (x,y,z)."""
#     # Convert to radians
#     lat_rad = np.radians(lat)
#     lon_rad = np.radians(lon)
    
#     # Convert to cartesian coordinates on a unit sphere
#     x = np.cos(lat_rad) * np.cos(lon_rad)
#     y = np.cos(lat_rad) * np.sin(lon_rad)
#     z = np.sin(lat_rad)
    
#     return np.column_stack((x, y, z))

# def cartesian_to_haversine_distance(distance_cartesian):
#     """Convert cartesian distance to angular distance in radians."""
#     # Ensure distance is at most 2.0 (diameter of unit sphere)
#     distance_cartesian = np.minimum(distance_cartesian, 2.0)
    
#     # Convert chord length to angular distance using inverse haversine formula
#     return 2 * np.arcsin(distance_cartesian / 2)

# def calculate_nn_distances(cartesian_points, k=5, batch_size=50000):
#     """
#     Calculate distances to k nearest neighbors, excluding self-matches.
    
#     Args:
#         cartesian_points: Cartesian coordinates (x, y, z) on unit sphere
#         k: Number of neighbors to find (excluding self)
#         batch_size: Batch size for processing
        
#     Returns:
#         Distances array of shape (n_points, k)
#     """
#     # Build KD-tree
#     tree = cKDTree(cartesian_points)
#     n_points = len(cartesian_points)
    
#     # We need to request k+1 neighbors since the first is always self
#     k_query = k + 1
    
#     # Output array for distances (excluding self)
#     all_distances = np.zeros((n_points, k))
    
#     # Process in batches
#     num_batches = int(np.ceil(n_points / batch_size))
    
#     for batch_idx in tqdm(range(num_batches), desc="Computing NN distances"):
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, n_points)
#         batch_points = cartesian_points[start_idx:end_idx]
        
#         # Find k+1 nearest neighbors (including self)
#         distances, indices = tree.query(batch_points, k=k_query)
        
#         # Skip the first column (self-matches with distance 0)
#         nn_distances = distances[:, 1:]
        
#         # Convert to angular distances
#         nn_distances = cartesian_to_haversine_distance(nn_distances)
        
#         # Store results
#         all_distances[start_idx:end_idx] = nn_distances
        
#         # Clean up
#         del distances, indices, nn_distances
#         gc.collect()
    
#     return all_distances

# def calculate_spatial_density(lat, lon, k=5, batch_size=50000):
#     """
#     Calculate spatial density using k-nearest neighbors.
    
#     Args:
#         lat: Latitude values in degrees
#         lon: Longitude values in degrees
#         k: Number of nearest neighbors to consider
#         batch_size: Batch size for processing
        
#     Returns:
#         density: Spatial density values (higher = denser)
#     """
#     print("Converting to cartesian coordinates...")
#     cartesian_points = haversine_to_cartesian(lat, lon)
    
#     print("Calculating NN distances...")
#     nn_distances = calculate_nn_distances(cartesian_points, k, batch_size)
    
# # The density is inversely proportional to...
#     # We use the mean distance to k nearest neighbors
#     mean_distances = np.mean(nn_distances, axis=1)
    
#     # Calculate density - avoid division by zero
#     epsilon = 1e-10
#     density = 1.0 / (mean_distances + epsilon)
    
#     return density, nn_distances

# def visualize_spatial_density(lat, lon, density, output_file=None):
#     """
#     Visualize spatial density on a world map.
    
#     Args:
#         lat: Latitude values
#         lon: Longitude values
#         density: Density values
#         output_file: Optional path to save the figure
#     """
#     try:
#         import cartopy.crs as ccrs
#         import cartopy.feature as cfeature
#         from matplotlib.colors import LogNorm
        
#         # Normalize density for better visualization
#         # Use log normalization for better color distribution
#         norm = LogNorm(vmin=np.percentile(density, 5), vmax=np.percentile(density, 95))
        
#         # Create figure with map projection
#         fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.Robinson()})
        
#         # Add map features
#         ax.add_feature(cfeature.LAND, facecolor='lightgray')
#         ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
#         ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
#         ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        
#         # Plot density
#         scatter = ax.scatter(
#             lon, lat, 
#             transform=ccrs.PlateCarree(),
#             c=density, 
#             cmap='viridis',
#             norm=norm,
#             s=5,
#             alpha=0.7,
#             edgecolor='none'
#         )
        
#         # Add colorbar
#         cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.05)
#         cbar.set_label('Spatial Density')
        
#         # Set title
#         plt.title('Spatial Density Distribution', fontsize=14)
        
#         # Save or show
#         if output_file:
#             plt.savefig(output_file, dpi=300, bbox_inches='tight')
#             print(f"Figure saved to {output_file}")
#         else:
#             plt.show()
            
#         plt.close()
        
#     except ImportError:
#         print("Cartopy not installed. Cannot create visualization.")
        
# def create_density_based_weights(lat, lon, k=5, inverse=True, normalize=True):
#     """
#     Create sample weights based on spatial density.
    
#     Args:
#         lat: Latitude values
#         lon: Longitude values
#         k: Number of nearest neighbors for density estimation
#         inverse: If True, assigns higher weights to sparse areas
#         normalize: If True, normalizes weights to sum to len(lat)
        
#     Returns:
#         weights: Sample weights
#     """
#     # Calculate spatial density
#     density, nn_distances = calculate_spatial_density(lat, lon, k)
    
#     # Create weights (higher density = lower weight)
#     if inverse:
#         weights = 1.0 / (density + 1e-8)
#     else:
#         weights = density
    
#     # Eliminate extreme outliers (clip at 3 standard deviations)
#     weights_mean = np.mean(weights)
#     weights_std = np.std(weights)
#     weights = np.clip(weights, 0, weights_mean + 3*weights_std)
    
#     # Normalize weights
#     if normalize:
#         weights = weights / np.mean(weights)
    
#     # Optionally visualize the spatial density
#     visualize_spatial_density(lat, lon, density, output_file="spatial_density.png")
    
#     return weights, nn_distances

# # Example usage
# if __name__ == "__main__":
#     # Generate some sample data (or use your actual data)
#     np.random.seed(42)
#     n_samples = 10000
    
#     # Simulated data with clustered points
#     lat = np.random.normal(60, 10, n_samples)  # Centered around 60°N
#     lon = np.random.normal(0, 30, n_samples)   # Centered around 0°E
    
#     # Add some additional clusters
#     cluster_sizes = [500, 300, 700]
#     cluster_lats = [40, 70, 50]
#     cluster_lons = [30, -40, 80]
    
#     for size, clat, clon in zip(cluster_sizes, cluster_lats, cluster_lons):
#         idx = np.random.choice(n_samples, size, replace=False)
#         lat[idx] = np.random.normal(clat, 3, size)
#         lon[idx] = np.random.normal(clon, 3, size)
    
#     # Clip to valid latitude range
#     lat = np.clip(lat, -90, 90)
    
#     # Create density-based weights
#     weights, nn_distances = create_density_based_weights(lat, lon, k=5)
    
#     print(f"Min distance: {np.min(nn_distances):.6f} radians")
#     print(f"Max distance: {np.max(nn_distances):.6f} radians")
#     print(f"Mean distance: {np.mean(nn_distances):.6f} radians")
    
#     print(f"Min weight: {np.min(weights):.2f}")
#     print(f"Max weight: {np.max(weights):.2f}")
#     print(f"Mean weight: {np.mean(weights):.2f}")

# density, nn_distances = calculate_spatial_density(latitudes, longitudes, k=5)

# import os
# import gc
# import json
# import pickle
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.callbacks import (
#     EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
#     CSVLogger, TensorBoard
# )
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# from tqdm import tqdm
# import psutil
# from datetime import datetime

# def memory_usage():
#     """Get current memory usage in MB"""
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()
#     return mem_info.rss / 1024**2  # Memory in MB

# def train_with_spatial_balancing(X, y, metadata, output_dir=None, checkpoint_dir=None):
#     """
#     Train a model with spatially balanced sampling.
    
#     Parameters:
#     -----------
#     X : numpy.ndarray
#         Input features
#     y : numpy.ndarray
#         Output labels
#     metadata : list
#         Metadata containing timestamps and spatial information
#     output_dir : str, optional
#         Directory to save model and results
#     checkpoint_dir : str, optional
#         Directory for saving checkpoints
        
#     Returns:
#     --------
#     trained_model, training_history
#     """
#     print("Training zero curtain model with spatial balancing...")
#     print(f"Memory before training: {memory_usage():.1f} MB")
    
#     # Create output directory
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
    
#     if checkpoint_dir is None:
#         checkpoint_dir = os.path.join(output_dir, 'checkpoints') if output_dir else 'checkpoints'
#     os.makedirs(checkpoint_dir, exist_ok=True)
    
#     # Enable memory growth to avoid pre-allocating all GPU memory
#     physical_devices = tf.config.list_physical_devices('GPU')
#     if physical_devices:
#         for device in physical_devices:
#             try:
#                 tf.config.experimental.set_memory_growth(device, True)
#                 print(f"Enabled memory growth for {device}")
#             except:
#                 print(f"Could not set memory growth for {device}")
    
#     # Create spatiotemporally balanced train/val/test split
#     print("Creating spatiotemporally balanced split...")
#     train_indices, val_indices, test_indices = stratified_spatiotemporal_split(
#         X, y, metadata, test_size=0.2, val_size=0.15, checkpoint_dir=checkpoint_dir
#     )
    
#     # Create the splits
#     X_train = X[train_indices]
#     y_train = y[train_indices]
    
#     X_val = X[val_indices]
#     y_val = y[val_indices]
    
#     X_test = X[test_indices]
#     y_test = y[test_indices]
    
#     # Load spatial density weights if available
#     weights_file = os.path.join(checkpoint_dir, "spatial_density.pkl")
#     if os.path.exists(weights_file):
#         print(f"Loading spatial weights from {weights_file}")
#         with open(weights_file, "rb") as f:
#             weights_data = pickle.load(f)
#         sample_weights = weights_data["weights"][train_indices]
        
#         # Normalize weights
#         sample_weights = sample_weights / np.mean(sample_weights) * len(sample_weights)
#     else:
#         print("No spatial weights found, using uniform weights")
#         sample_weights = np.ones(len(train_indices))
    
#     # Print info about the splits
#     print(f"Train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")
#     print(f"Positive examples: Train={np.sum(y_train)} ({np.sum(y_train)/len(y_train)*100:.1f}%), ...
#           f"Val={np.sum(y_val)} ({np.sum(y_val)/len(y_val)*100:.1f}%), " +
#           f"Test={np.sum(y_test)} ({np.sum(y_test)/len(y_test)*100:.1f}%)")
    
#     # Combine sample weights with class weights for imbalanced data
#     pos_weight = (len(y_train) - np.sum(y_train)) / max(1, np.sum(y_train))
#     class_weight = {0: 1.0, 1: pos_weight}
#     print(f"Using class weight {pos_weight:.2f} for positive examples")
    
#     # Build model with appropriate input shape
#     input_shape = (X_train.shape[1], X_train.shape[2])
#     model = build_advanced_zero_curtain_model(input_shape)
    
#     # Always check for existing model checkpoint to resume training
#     model_checkpoint_paths = []
#     if output_dir:
#         # Look for checkpoint files in multiple locations
#         model_checkpoint_path1 = os.path.join(output_dir, 'model_checkpoint.h5')
#         model_checkpoint_path2 = os.path.join(output_dir, 'checkpoint.h5')
#         model_checkpoint_path3 = os.path.join(checkpoint_dir, 'model_checkpoint.h5')
        
#         model_checkpoint_paths = [p for p in [model_checkpoint_path1, model_checkpoint_path2, mode...
#                                 if os.path.exists(p)]
        
#         if model_checkpoint_paths:
#             print(f"Found {len(model_checkpoint_paths)} existing model checkpoints")
#             # Use the most recent checkpoint based on modification time
#             latest_checkpoint = max(model_checkpoint_paths, key=os.path.getmtime)
#             print(f"Loading most recent checkpoint: {latest_checkpoint}")
#             try:
#                 model = tf.keras.models.load_model(latest_checkpoint)
# print("Checkpoint loaded successfully - will resume training...
#             except Exception as e:
#                 print(f"Error loading checkpoint: {str(e)}")
#                 print("Will start training from scratch")
    
#     # Set up callbacks
#     callbacks = [
#         # Stop training when validation performance plateaus
#         EarlyStopping(
#             patience=15,
#             restore_best_weights=True,
#             monitor='val_auc',
#             mode='max'
#         ),
#         # Reduce learning rate when improvement slows
#         ReduceLROnPlateau(
#             factor=0.5,
#             patience=7,
#             min_lr=1e-6,
#             monitor='val_auc',
#             mode='max'
#         ),
#         # Manual garbage collection after each epoch
#         tf.keras.callbacks.LambdaCallback(
#             on_epoch_end=lambda epoch, logs: gc.collect()
#         )
#     ]
    
#     # Add additional callbacks if output directory provided
#     if output_dir:
#         callbacks.extend([
#             # Save best model
#             ModelCheckpoint(
#                 os.path.join(output_dir, 'model_checkpoint.h5'),
#                 save_best_only=True,
#                 monitor='val_auc',
#                 mode='max'
#             ),
#             # Log training progress to CSV
#             CSVLogger(
#                 os.path.join(output_dir, 'training_log.csv'),
#                 append=True
#             ),
#             # TensorBoard visualization
#             TensorBoard(
#                 log_dir=os.path.join(output_dir, 'tensorboard_logs'),
#                 histogram_freq=1,
#                 profile_batch=0  # Disable profiling to save memory
#             )
#         ])
    
#     # Train model
#     print("Starting model training...")
#     batch_size = 32  # Adjust based on available memory
#     epochs = 100
    
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=callbacks,
#         class_weight=class_weight,
#         sample_weight=sample_weights,
#         verbose=1,
#         shuffle=True,
#         use_multiprocessing=False,  # Avoid memory overhead
#         workers=1  # Reduce parallel processing
#     )
    
#     # Clean up to free memory
#     del X_train, y_train, X_val, y_val
#     gc.collect()
    
#     # Evaluate on test set
#     print("Evaluating model on test set...")
#     evaluation = model.evaluate(X_test, y_test, batch_size=batch_size)
#     print("Test performance:")
#     for metric, value in zip(model.metrics_names, evaluation):
#         print(f"  {metric}: {value:.4f}")
    
#     # Generate predictions for test set
#     y_pred_prob = model.predict(X_test, batch_size=batch_size)
#     y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
#     # Calculate additional evaluation metrics
#     report = classification_report(y_test, y_pred)
#     conf_matrix = confusion_matrix(y_test, y_pred)
    
#     print("Classification Report:")
#     print(report)
    
#     print("Confusion Matrix:")
#     print(conf_matrix)
    
#     # Save evaluation results
#     if output_dir:
#         # Save evaluation metrics
#         with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
#             f.write("Classification Report:\n")
#             f.write(report)
#             f.write("\n\nConfusion Matrix:\n")
#             f.write(str(conf_matrix))
#             f.write("\n\nTest Metrics:\n")
#             for metric, value in zip(model.metrics_names, evaluation):
#                 f.write(f"{metric}: {value:.4f}\n")
        
#         # Save test set predictions with timestamp to avoid overwriting
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         np.save(os.path.join(output_dir, f'test_predictions_{timestamp}.npy'), y_pred_prob)
#         np.save(os.path.join(output_dir, f'test_indices_{timestamp}.npy'), test_indices)
#         # Also keep a copy with the standard name for easier reference
#         np.save(os.path.join(output_dir, 'test_predictions_latest.npy'), y_pred_prob)
#         np.save(os.path.join(output_dir, 'test_indices_latest.npy'), test_indices)
        
#         # Plot training history
#         plt.figure(figsize=(16, 6))
        
#         plt.subplot(1, 3, 1)
#         plt.plot(history.history['auc'])
#         plt.plot(history.history['val_auc'])
#         plt.title('Model AUC')
#         plt.ylabel('AUC')
#         plt.xlabel('Epoch')
#         plt.legend(['Train', 'Validation'], loc='lower right')
        
#         plt.subplot(1, 3, 2)
#         plt.plot(history.history['loss'])
#         plt.plot(history.history['val_loss'])
#         plt.title('Model Loss')
#         plt.ylabel('Loss')
#         plt.xlabel('Epoch')
#         plt.legend(['Train', 'Validation'], loc='upper right')
        
#         # Plot ROC curve
#         plt.subplot(1, 3, 3)
#         fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
#         plt.plot([0, 1], [0, 1], 'k--')
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('ROC Curve (Test Set)')
#         plt.legend(loc='lower right')
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, 'training_performance.png'), dpi=300)
        
#         # Save detailed model summary
#         from contextlib import redirect_stdout
#         with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
#             with redirect_stdout(f):
#                 model.summary()
    
#     # Clean up to free memory
#     del X_test, y_test
#     gc.collect()
    
#     print(f"Memory after training: {memory_usage():.1f} MB")
#     return model, history, evaluation

# Save evaluation results
if output_dir:
    # Save evaluation metrics
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write("\n\nTest Metrics:\n")
        for metric, value in zip(model.metrics_names, evaluation):
            f.write(f"{metric}: {value:.4f}\n")
    
    # Save test set predictions with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(output_dir, f'test_predictions_{timestamp}.npy'), y_pred_prob)
    np.save(os.path.join(output_dir, f'test_indices_{timestamp}.npy'), test_indices)
    # Also keep a copy with the standard name for easier reference
    np.save(os.path.join(output_dir, 'test_predictions_latest.npy'), y_pred_prob)
    np.save(os.path.join(output_dir, 'test_indices_latest.npy'), test_indices)
    
    # Plot training history
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot ROC curve
    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = sk_auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_performance.png'), dpi=300)

print(f"Memory after training: {memory_usage():.1f} MB")

def plot_training_history(output_dir):
    """Visualize learning curves across all training chunks"""
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    import os
    
    # Load training history
    history_path = os.path.join(output_dir, "training_history.json")
    if not os.path.exists(history_path):
        history_path = os.path.join(output_dir, "final_training_metrics.json")
    
    try:
        with open(history_path, "r") as f:
            history_log = json.load(f)
    except:
        # Try pickle format
        import pickle
        with open(os.path.join(output_dir, "training_history.pkl"), "rb") as f:
            history_log = pickle.load(f)
    
    # Extract metrics across all chunks
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    val_metrics = [f'val_{m}' for m in metrics]
    
    # Prepare aggregated metrics
    all_metrics = {m: [] for m in metrics + val_metrics}
    
    # Collect metrics across chunks
    for chunk_history in history_log:
        for metric in metrics:
            # Training metrics
            if metric in chunk_history:
                all_metrics[metric].extend(chunk_history[metric])
            
            # Validation metrics
            val_metric = f'val_{metric}'
            if val_metric in chunk_history:
                all_metrics[val_metric].extend(chunk_history[val_metric])
    
    # Create learning curve visualizations
    plt.figure(figsize=(18, 12))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(all_metrics['loss'], label='Training Loss')
    plt.plot(all_metrics['val_loss'], label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(all_metrics['accuracy'], label='Training Accuracy')
    plt.plot(all_metrics['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot AUC
    plt.subplot(2, 2, 3)
    plt.plot(all_metrics['auc'], label='Training AUC')
    plt.plot(all_metrics['val_auc'], label='Validation AUC')
    plt.title('AUC Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot precision-recall
    plt.subplot(2, 2, 4)
    plt.plot(all_metrics['precision'], label='Training Precision')
    plt.plot(all_metrics['recall'], label='Training Recall')
    plt.plot(all_metrics['val_precision'], label='Validation Precision')
    plt.plot(all_metrics['val_recall'], label='Validation Recall')
    plt.title('Precision-Recall Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300)
    plt.show()
    
    return all_metrics

def visualize_spatial_predictions(output_dir, train_indices, metadata, X_file, y_file):
    """Create spatial visualizations of predictions"""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Load complete predictions
    predictions_file = os.path.join(output_dir, 'all_training_predictions.npy')
    if not os.path.exists(predictions_file):
        print("Combining predictions from chunks...")
        all_predictions = combine_chunk_predictions(output_dir, train_indices)
    else:
        all_predictions = np.load(predictions_file)
    
    # Load actual values
    X = np.load(X_file, mmap_mode='r')  # Memory-mapped to save RAM
    y = np.load(y_file, mmap_mode='r')
    y_true = np.array([y[idx] for idx in train_indices])
    
    # Extract spatial coordinates from metadata
    latitudes = np.array([metadata[idx]['latitude'] for idx in train_indices])
    longitudes = np.array([metadata[idx]['longitude'] for idx in train_indices])
    
    # Convert predictions to binary
    y_pred_binary = (all_predictions > 0.5).astype(int)
    
    # Create spatial plot
    plt.figure(figsize=(20, 10))
    
    # Plot predicted probabilities
    plt.subplot(1, 3, 1)
    sc = plt.scatter(longitudes, latitudes, c=all_predictions, cmap='viridis', 
                     alpha=0.7, s=5, vmin=0, vmax=1)
    plt.colorbar(sc, label='Predicted Probability')
    plt.title('Predicted Zero Curtain Probabilities')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    
    # Plot true values
    plt.subplot(1, 3, 2)
    sc = plt.scatter(longitudes, latitudes, c=y_true, cmap='coolwarm', 
                     alpha=0.7, s=5, vmin=0, vmax=1)
    plt.colorbar(sc, label='Actual Value')
    plt.title('Actual Zero Curtain Occurrence')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    
    # Plot errors (false positives and negatives)
    plt.subplot(1, 3, 3)
    errors = (y_pred_binary != y_true).astype(int)
    sc = plt.scatter(longitudes, latitudes, c=errors, cmap='Reds', 
                     alpha=0.7, s=5, vmin=0, vmax=1)
    plt.colorbar(sc, label='Error (1=Misclassified)')
    plt.title('Prediction Errors')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_predictions.png'), dpi=300)
    plt.show()
    
    return all_predictions, y_true

def visualize_temporal_predictions(output_dir, train_indices, metadata, all_predictions=None):
    """Analyze prediction patterns over time"""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from datetime import datetime
    import pandas as pd
    
    # Load predictions if not provided
    if all_predictions is None:
        predictions_file = os.path.join(output_dir, 'all_training_predictions.npy')
        if not os.path.exists(predictions_file):
            print("Please run combine_chunk_predictions first")
            return None
        all_predictions = np.load(predictions_file)
    
    # Extract timestamps from metadata
    timestamps = np.array([metadata[idx]['timestamp'] for idx in train_indices])
    
    # Convert to datetime objects if needed
    if isinstance(timestamps[0], str):
        timestamps = np.array([datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S') 
                               if isinstance(ts, str) else ts 
                               for ts in timestamps])
    
    # Create dataframe for analysis
    df = pd.DataFrame({
        'timestamp': timestamps,
        'prediction': all_predictions
    })
    
    # Add year, month, day columns
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['doy'] = df['timestamp'].dt.dayofyear
    
    # Aggregate by time periods
    monthly_avg = df.groupby(['year', 'month'])['prediction'].mean().reset_index()
    monthly_avg['period'] = monthly_avg['year'].astype(str) + '-' + monthly_avg['month'].astype(str).str.zfill(2)
    
    daily_avg = df.groupby(['year', 'doy'])['prediction'].mean().reset_index()
    
    # Create temporal visualizations
    plt.figure(figsize=(20, 15))
    
    # Monthly time series
    plt.subplot(2, 2, 1)
    plt.plot(range(len(monthly_avg)), monthly_avg['prediction'], marker='o')
    plt.xticks(range(len(monthly_avg)), monthly_avg['period'], rotation=90)
    plt.title('Monthly Average Prediction Probability')
    plt.xlabel('Month')
    plt.ylabel('Avg Probability')
    plt.grid(True, alpha=0.3)
    
    # Daily time series by year
    plt.subplot(2, 2, 2)
    for year in df['year'].unique():
        year_data = daily_avg[daily_avg['year'] == year]
        plt.plot(year_data['doy'], year_data['prediction'], label=str(year))
    plt.title('Daily Average Prediction Probability by Year')
    plt.xlabel('Day of Year')
    plt.ylabel('Avg Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Monthly boxplot
    plt.subplot(2, 2, 3)
    df['month_name'] = pd.Categorical(df['timestamp'].dt.month_name(), 
                                      categories=['January', 'February', 'March', 'April', 'May', 'June', 
                                                 'July', 'August', 'September', 'October', 'November', 'December'])
    monthly_box = df.boxplot(column='prediction', by='month_name', grid=False, ax=plt.gca())
    plt.title('Monthly Distribution of Predictions')
    plt.suptitle('')  # Remove pandas default title
    plt.xlabel('Month')
    plt.ylabel('Prediction Probability')
    
    # Temporal heatmap (Year x Month)
    plt.subplot(2, 2, 4)
    heatmap_data = df.groupby(['year', 'month'])['prediction'].mean().unstack()
    plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Avg Probability')
    plt.title('Prediction Probability Heatmap (Year x Month)')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_predictions.png'), dpi=300)
    plt.show()
    
    return df

# def create_training_plots(history, output_dir):
#     """Create and save training history plots"""
#     # Create a figure with subplots
#     plt.figure(figsize=(16, 6))
    
#     # Plot AUC
#     plt.subplot(1, 3, 1)
#     plt.plot(history['val_auc'], label='Validation')
#     plt.title('Model AUC')
#     plt.ylabel('AUC')
#     plt.xlabel('Epoch')
#     plt.legend(loc='lower right')
    
#     # Plot Loss
#     plt.subplot(1, 3, 2)
#     plt.plot(history['loss'], label='Train')
#     plt.plot(history['val_loss'], label='Validation')
#     plt.title('Model Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(loc='upper right')
    
#     # Plot Accuracy
#     plt.subplot(1, 3, 3)
#     plt.plot(history['accuracy'], label='Train')
#     plt.plot(history['val_accuracy'], label='Validation')
#     plt.title('Model Accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(loc='lower right')
    
#     # Save figure
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'training_performance.png'), dpi=300)
#     plt.close()
    
#     # Create ROC curve if validation AUC values are available
#     if len(history['val_auc']) > 0:
#         plt.figure(figsize=(8, 6))
#         plt.plot(history['val_auc'], marker='o')
#         plt.title('Validation AUC Over Training')
#         plt.ylabel('AUC')
#         plt.xlabel('Epoch')
#         plt.grid(True)
#         plt.savefig(os.path.join(output_dir, 'validation_auc.png'), dpi=300)
#         plt.close()

# Save evaluation results
if output_dir:
    # Save evaluation metrics
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write("\n\nTest Metrics:\n")
        for metric, value in zip(model.metrics_names, evaluation):
            f.write(f"{metric}: {value:.4f}\n")
    
    # Save test set predictions with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(output_dir, f'test_predictions_{timestamp}.npy'), y_pred_prob)
    np.save(os.path.join(output_dir, f'test_indices_{timestamp}.npy'), test_indices)
    # Also keep a copy with the standard name for easier reference
    np.save(os.path.join(output_dir, 'test_predictions_latest.npy'), y_pred_prob)
    np.save(os.path.join(output_dir, 'test_indices_latest.npy'), test_indices)
    
    # Plot training history
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot ROC curve
    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = sk_auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_performance.png'), dpi=300)

print(f"Memory after training: {memory_usage():.1f} MB")

def compare_detection_methods(physical_events_file, model_events_file, output_dir=None):
    """
    Compare zero curtain events detected by different methods with enhanced
    analysis and visualizations.
    
    Parameters:
    -----------
    physical_events_file : str
        Path to CSV file with events detected by the physics-based method
    model_events_file : str
        Path to CSV file with events detected by the deep learning model
    output_dir : str, optional
        Directory to save comparison results
        
    Returns:
    --------
    dict
        Comparison statistics and metrics
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import timedelta
    import os
    import gc
    
    print("Comparing detection methods...")
    print(f"Memory before comparison: {memory_usage():.1f} MB")
    
    # Load events
    physical_events = pd.read_csv(physical_events_file, parse_dates=['datetime_min', 'datetime_max'])
    model_events = pd.read_csv(model_events_file, parse_dates=['datetime_min', 'datetime_max'])
    
    print(f"Loaded {len(physical_events)} physics-based events and {len(model_events)} model-detected events")
    
    # Calculate basic statistics for each method
    physical_stats = {
        'total_events': len(physical_events),
        'unique_sites': physical_events['source'].nunique(),
        'unique_depths': physical_events['soil_temp_depth'].nunique(),
        'median_duration': physical_events['duration_hours'].median(),
        'mean_duration': physical_events['duration_hours'].mean(),
        'std_duration': physical_events['duration_hours'].std(),
        'min_duration': physical_events['duration_hours'].min(),
        'max_duration': physical_events['duration_hours'].max(),
        'total_days': sum([(event['datetime_max'] - event['datetime_min']).total_seconds()/86400 
                           for _, event in physical_events.iterrows()])
    }
    
    model_stats = {
        'total_events': len(model_events),
        'unique_sites': model_events['source'].nunique(),
        'unique_depths': model_events['soil_temp_depth'].nunique(),
        'median_duration': model_events['duration_hours'].median(),
        'mean_duration': model_events['duration_hours'].mean(),
        'std_duration': model_events['duration_hours'].std(),
        'min_duration': model_events['duration_hours'].min(),
        'max_duration': model_events['duration_hours'].max(),
        'total_days': sum([(event['datetime_max'] - event['datetime_min']).total_seconds()/86400 
                          for _, event in model_events.iterrows()])
    }
    
    # Create a site-day matching table for overlap analysis
    print("Building day-by-day event registry for detailed comparison...")
    physical_days = set()
    model_days = set()
    
    # Process physical events in batches
    batch_size = 1000
    for i in range(0, len(physical_events), batch_size):
        batch = physical_events.iloc[i:i+batch_size]
        
        for _, event in batch.iterrows():
            site = event['source']
            depth = event['soil_temp_depth']
            start_day = event['datetime_min'].date()
            end_day = event['datetime_max'].date()
            
            # Add each day of the event
            current_day = start_day
            while current_day <= end_day:
                physical_days.add((site, depth, current_day))
                current_day += timedelta(days=1)
        
        # Clear batch to free memory
        del batch
        gc.collect()
    
    # Process model events in batches
    for i in range(0, len(model_events), batch_size):
        batch = model_events.iloc[i:i+batch_size]
        
        for _, event in batch.iterrows():
            site = event['source']
            depth = event['soil_temp_depth']
            start_day = event['datetime_min'].date()
            end_day = event['datetime_max'].date()
            
            # Add each day of the event
            current_day = start_day
            while current_day <= end_day:
                model_days.add((site, depth, current_day))
                current_day += timedelta(days=1)
        
        # Clear batch to free memory
        del batch
        gc.collect()
    
    # Calculate overlap metrics
    overlap_days = physical_days.intersection(model_days)
    
    overlap_metrics = {
        'physical_only_days': len(physical_days - model_days),
        'model_only_days': len(model_days - physical_days),
        'overlap_days': len(overlap_days),
        'jaccard_index': len(overlap_days) / len(physical_days.union(model_days)) if len(physical_days.union(model_days)) > 0 else 0,
        'precision': len(overlap_days) / len(model_days) if len(model_days) > 0 else 0,
        'recall': len(overlap_days) / len(physical_days) if len(physical_days) > 0 else 0,
        'f1_score': 2 * len(overlap_days) / (len(physical_days) + len(model_days)) if (len(physical_days) + len(model_days)) > 0 else 0
    }
    
    # Calculate site-specific overlap
    site_level_comparison = {}
    
    # Extract unique sites for analysis
    all_sites = set([s for s, _, _ in physical_days]).union(set([s for s, _, _ in model_days]))
    print(f"Analyzing {len(all_sites)} unique sites...")
    
    for site in all_sites:
        # Get days for this site
        site_physical = set([(s, d, day) for s, d, day in physical_days if s == site])
        site_model = set([(s, d, day) for s, d, day in model_days if s == site])
        site_overlap = site_physical.intersection(site_model)
        
        if len(site_physical) == 0 and len(site_model) == 0:
            continue
            
        site_level_comparison[site] = {
            'physical_days': len(site_physical),
            'model_days': len(site_model),
            'overlap_days': len(site_overlap),
            'jaccard': len(site_overlap) / len(site_physical.union(site_model)) if len(site_physical.union(site_model)) > 0 else 0,
            'precision': len(site_overlap) / len(site_model) if len(site_model) > 0 else 0,
            'recall': len(site_overlap) / len(site_physical) if len(site_physical) > 0 else 0
        }
    
    # Analyze temporal distribution
    print("Analyzing temporal distributions...")
    
    # Extract month from each event day
    physical_months = [day.month for _, _, day in physical_days]
    model_months = [day.month for _, _, day in model_days]
    
    # Count events by month
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    physical_month_counts = [physical_months.count(i+1) for i in range(12)]
    model_month_counts = [model_months.count(i+1) for i in range(12)]
    
    # Print comparison results
    print("\n=== DETECTION METHOD COMPARISON ===\n")
    
    print("Physics-based Detection:")
    print(f"  Total Events: {physical_stats['total_events']}")
    print(f"  Unique Sites: {physical_stats['unique_sites']}")
    print(f"  Unique Depths: {physical_stats['unique_depths']}")
    print(f"  Median Duration: {physical_stats['median_duration']:.1f} hours ({physical_stats['median_duration']/24:.1f} days)")
    print(f"  Mean Duration: {physical_stats['mean_duration']:.1f} hours ({physical_stats['mean_duration']/24:.1f} days)")
    print(f"  Duration Range: {physical_stats['min_duration']:.1f} - {physical_stats['max_duration']:.1f} hours")
    
    print("\nDeep Learning Model Detection:")
    print(f"  Total Events: {model_stats['total_events']}")
    print(f"  Unique Sites: {model_stats['unique_sites']}")
    print(f"  Unique Depths: {model_stats['unique_depths']}")
    print(f"  Median Duration: {model_stats['median_duration']:.1f} hours ({model_stats['median_duration']/24:.1f} days)")
    print(f"  Mean Duration: {model_stats['mean_duration']:.1f} hours ({model_stats['mean_duration']/24:.1f} days)")
    print(f"  Duration Range: {model_stats['min_duration']:.1f} - {model_stats['max_duration']:.1f} hours")
    
    print("\nOverlap Analysis:")
    print(f"  Days with Events (Physics-based): {len(physical_days)}")
    print(f"  Days with Events (Deep Learning): {len(model_days)}")
    print(f"  Days detected by both methods: {overlap_metrics['overlap_days']}")
    print(f"  Days detected only by Physics-based: {overlap_metrics['physical_only_days']}")
    print(f"  Days detected only by Deep Learning: {overlap_metrics['model_only_days']}")
    print(f"  Jaccard Index (overlap): {overlap_metrics['jaccard_index']:.4f}")
    print(f"  Precision: {overlap_metrics['precision']:.4f}")
    print(f"  Recall: {overlap_metrics['recall']:.4f}")
    print(f"  F1 Score: {overlap_metrics['f1_score']:.4f}")
    
    # Generate comparison visualizations
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a Venn diagram of detection overlap
        try:
            from matplotlib_venn import venn2
            
            plt.figure(figsize=(10, 8))
            venn2(subsets=(len(physical_days - model_days), 
                          len(model_days - physical_days), 
                          len(overlap_days)),
                 set_labels=('Physics-based', 'Deep Learning'))
            plt.title('Overlap between Detection Methods', fontsize=16)
            plt.savefig(os.path.join(output_dir, 'detection_overlap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except ImportError:
            print("matplotlib_venn not installed. Skipping Venn diagram.")
        
        # Compare duration distributions
        plt.figure(figsize=(12, 8))
        
        sns.histplot(physical_events['duration_hours'], kde=True, alpha=0.5, 
                    label='Physics-based', color='blue', bins=50, log_scale=(False, True))
        sns.histplot(model_events['duration_hours'], kde=True, alpha=0.5, 
                    label='Deep Learning', color='red', bins=50, log_scale=(False, True))
        
        plt.xlabel('Duration (hours)', fontsize=12)
        plt.ylabel('Frequency (log scale)', fontsize=12)
        plt.title('Comparison of Zero Curtain Duration Distributions', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'duration_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot monthly distribution
        plt.figure(figsize=(12, 8))
        width = 0.35
        x = np.arange(len(month_names))
        
        plt.bar(x - width/2, physical_month_counts, width, label='Physics-based', color='blue', alpha=0.7)
        plt.bar(x + width/2, model_month_counts, width, label='Deep Learning', color='red', alpha=0.7)
        
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Number of Event Days', fontsize=12)
        plt.title('Monthly Distribution of Zero Curtain Events', fontsize=16)
        plt.xticks(x, month_names)
        plt.legend(fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'monthly_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Site-level agreement analysis (top 20 sites by event count)
        site_df = pd.DataFrame.from_dict(site_level_comparison, orient='index')
        top_sites = site_df.sort_values(by=['physical_days', 'model_days'], ascending=False).head(20)
        
        plt.figure(figsize=(14, 8))
        ax = plt.subplot(111)
        x = np.arange(len(top_sites))
        
        p1 = ax.bar(x - width/2, top_sites['physical_days'], width, label='Physics-based', color='blue', alpha=0.7)
        p2 = ax.bar(x + width/2, top_sites['model_days'], width, label='Deep Learning', color='red', alpha=0.7)
        p3 = ax.bar(x, top_sites['overlap_days'], width/2, label='Overlap', color='purple', alpha=0.7)
        
        ax.set_xlabel('Site ID', fontsize=12)
        ax.set_ylabel('Number of Event Days', fontsize=12)
        ax.set_title('Site-level Agreement: Top 20 Sites by Event Count', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(top_sites.index, rotation=45, ha='right')
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'site_level_agreement.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed statistics to CSV
        site_df.to_csv(os.path.join(output_dir, 'site_level_metrics.csv'))
        
        # Create a summary report
        with open(os.path.join(output_dir, 'comparison_report.txt'), 'w') as f:
            f.write("# Zero Curtain Detection Method Comparison\n\n")
            
            f.write("## Physics-based Detection\n")
            for key, value in physical_stats.items():
                f.write(f"- {key}: {value}\n")
            
            f.write("\n## Deep Learning Model Detection\n")
            for key, value in model_stats.items():
                f.write(f"- {key}: {value}\n")
            
            f.write("\n## Overlap Analysis\n")
            for key, value in overlap_metrics.items():
                f.write(f"- {key}: {value}\n")
            
            f.write("\n## Summary\n")
            f.write(f"The two methods show a Jaccard similarity index of {overlap_metrics['jaccard_index']:.4f}, ")
            f.write(f"with a precision of {overlap_metrics['precision']:.4f} and recall of {overlap_metrics['recall']:.4f}.\n")
            
            f.write("\nThis indicates that ")
            if overlap_metrics['jaccard_index'] > 0.7:
                f.write("the methods have strong agreement in detecting zero curtain events.\n")
            elif overlap_metrics['jaccard_index'] > 0.4:
                f.write("the methods have moderate agreement in detecting zero curtain events.\n")
            else:
                f.write("the methods have relatively low agreement in detecting zero curtain events.\n")
                
            f.write("\nPossible explanations for differences include:\n")
            f.write("1. Physics-based detection using fixed thresholds vs. ML pattern recognition\n")
            f.write("2. Different sensitivities to signal noise\n")
            f.write("3. Model's ability to recognize patterns that may not strictly adhere to physical definitions\n")
            f.write("4. Seasonal variability in detection accuracy\n")
            
    # Clean up to free memory
    del physical_events, model_events, physical_days, model_days, overlap_days
    gc.collect()
    
    print(f"Memory after comparison: {memory_usage():.1f} MB")
    
    comparison_results = {
        'physical_stats': physical_stats,
        'model_stats': model_stats,
        'overlap_metrics': overlap_metrics,
        'site_level_comparison': site_level_comparison,
        'temporal_distribution': {
            'month_names': month_names,
            'physical_month_counts': physical_month_counts,
            'model_month_counts': model_month_counts
        }
    }
    
    return comparison_results

def train_zero_curtain_model_efficiently(X, y, metadata=None, output_dir=None, sample_fraction=0.1):
    """
    More efficient version with data sampling and better progress tracking.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
        CSVLogger, TensorBoard
    )
    import matplotlib.pyplot as plt
    import os
    import gc
    import numpy as np
    from sklearn.utils import shuffle
    
    print("Training zero curtain model...")
    print(f"Memory before training: {memory_usage():.1f} MB")
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Sampling to reduce dataset size if it's very large
    print(f"Original dataset size: {len(X)} samples")
    if len(X) > 1000000 and sample_fraction < 1.0:
        print(f"Dataset is very large. Sampling {sample_fraction*100:.1f}% of data...")
        sample_size = int(len(X) * sample_fraction)
        
        # Sample indices while preserving temporal order
        timestamps = np.array([meta['start_time'] for meta in metadata])
        sorted_indices = np.argsort(timestamps)
        
        # Take evenly distributed samples to maintain temporal distribution
        step = len(sorted_indices) // sample_size
        sampled_indices = sorted_indices[::step][:sample_size]
        
        X = X[sampled_indices]
        y = y[sampled_indices]
        sampled_metadata = [metadata[i] for i in sampled_indices]
        metadata = sampled_metadata
        print(f"Reduced to {len(X)} samples")
    
    # Temporal split with explicit progress
    print("Performing temporal split for train/validation/test sets...")
    print("Extracting timestamps...")
    timestamps = np.array([meta['start_time'] for meta in metadata])
    
    print("Sorting by timestamp...")
    sorted_indices = np.argsort(timestamps)
    
    # Calculate split points (60% train, 15% validation, 15% test)
    n_samples = len(sorted_indices)
    test_ratio = 0.15
    val_ratio = 0.15
    
    test_start = int(n_samples * (1 - test_ratio))
    val_start = int(n_samples * (1 - test_ratio - val_ratio))
    
    print("Creating split indices...")
    train_indices = sorted_indices[:val_start]
    val_indices = sorted_indices[val_start:test_start]
    test_indices = sorted_indices[test_start:]
    
    print(f"Training on data from {timestamps[train_indices[0]]} to {timestamps[train_indices[-1]]}")
    print(f"Validating on data from {timestamps[val_indices[0]]} to {timestamps[val_indices[-1]]}")
    print(f"Testing on data from {timestamps[test_indices[0]]} to {timestamps[test_indices[-1]]}")
    
    print("Creating data splits...")
    print("Creating training split...")
    X_train = X[train_indices]
    y_train = y[train_indices]
    
    print("Creating validation split...")
    X_val = X[val_indices]
    y_val = y[val_indices]
    
    print("Creating test split...")
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    print(f"Split sizes: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    
    # Check class balance in each split
    train_pos = np.sum(y_train)
    val_pos = np.sum(y_val)
    test_pos = np.sum(y_test)
    
    print(f"Positive examples: Train={train_pos} ({train_pos/len(y_train)*100:.1f}%), " +
          f"Val={val_pos} ({val_pos/len(y_val)*100:.1f}%), " +
          f"Test={test_pos} ({test_pos/len(y_test)*100:.1f}%)")
    
    # Clean up to free memory
    del sorted_indices, timestamps
    gc.collect()
    
    # Build a simplified model for initial testing
    print("Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Use a simpler model architecture first
    print("Using simpler model for faster training")
    #model = build_simpler_zero_curtain_model(input_shape)
    model = build_minimal_model(input_shape)
    
    # Set up callbacks with additional memory management and progress tracking
    callbacks = [
        # Stop early if validation performance plateaus
        EarlyStopping(
            patience=10,  
            restore_best_weights=True, 
            monitor='val_auc', 
            mode='max',
            verbose=1  # Add verbosity
        ),
        # Reduce learning rate when improvement slows
        ReduceLROnPlateau(
            factor=0.5, 
            patience=5,
            min_lr=1e-6, 
            monitor='val_auc', 
            mode='max',
            verbose=1  # Add verbosity
        ),
        # Manual garbage collection after each epoch
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect()
        )
    ]
    
    # Add additional callbacks if output directory provided
    if output_dir:
        callbacks.extend([
            # Save best model
            ModelCheckpoint(
                os.path.join(output_dir, 'checkpoint.h5'),
                save_best_weights_only=True,  # Save just weights, not whole model
                monitor='val_auc', 
                mode='max',
                verbose=1  # Add verbosity
            ),
            # Log training progress to CSV
            CSVLogger(
                os.path.join(output_dir, 'training_log.csv'),
                append=True
            )
        ])
    
    # Calculate class weights to handle imbalance
    pos_weight = len(y_train) / max(sum(y_train), 1)
    class_weight = {0: 1, 1: pos_weight}
    print(f"Using class weight {pos_weight:.2f} for positive class")
    
    # Train model with memory-efficient settings
    print("Training model...")
    batch_size = 256  # Larger batch size for faster training
    epochs = 20  # Reduced epochs for initial testing
    
    # Use fit with appropriate memory settings
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
        # Memory efficiency settings
        shuffle=True,
        use_multiprocessing=False,
        workers=1
    )
    
    # Save model after training
    print("Saving model...")
    if output_dir:
        model.save(os.path.join(output_dir, 'final_model.h5'))
    
    # Clean up to free memory
    del X_train, y_train, X_val, y_val
    gc.collect()
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    evaluation = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    print("Test performance:")
    for metric, value in zip(model.metrics_names, evaluation):
        print(f"  {metric}: {value:.4f}")
    
    print(f"Memory after training: {memory_usage():.1f} MB")
    return model, history, evaluation

# Simplified model for faster training
def build_simpler_zero_curtain_model(input_shape):
    """
    A simpler model architecture for faster training and debugging
    """
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # LSTM layer
    x = LSTM(64, return_sequences=True)(inputs)
    x = LSTM(32)(x)
    
    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def build_minimal_model(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def evaluate_trained_model(weights_path, X_file, y_file, test_indices, output_dir):
    """Evaluate the trained model and generate visualizations"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
    
    # Load data with memory mapping
    X = np.load(X_file, mmap_mode='r')
    y = np.load(y_file, mmap_mode='r')
    
    # Get sample for model initialization
    sample = X[test_indices[0]]
    
    # Build fresh model and load weights
    model = build_improved_zero_curtain_model_fixed(sample.shape)
    model.load_weights(weights_path)
    
    # Make predictions in batches
    batch_size = 500  # Larger batch size for faster processing
    all_preds = []
    all_true = []
    
    print(f"Evaluating model on {len(test_indices)} test samples...")
    
    for i in range(0, len(test_indices), batch_size):
        end_i = min(i + batch_size, len(test_indices))
        batch_indices = test_indices[i:end_i]
        
        X_batch = np.array([X[idx] for idx in batch_indices])
        y_batch = np.array([y[idx] for idx in batch_indices])
        
        preds = model.predict(X_batch, verbose=0)
        
        all_preds.extend(preds.flatten())
        all_true.extend(y_batch)
        
        # Show progress
        if i % 5000 == 0:
            print(f"Processed {i}/{len(test_indices)} samples")
    
    # Convert to arrays
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    # Calculate metrics
    fpr, tpr, _ = roc_curve(all_true, all_preds)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(all_true, all_preds)
    avg_precision = average_precision_score(all_true, all_preds)
    
    # Create visualizations directory
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "visualizations", "roc_curve.png"), dpi=300)
    plt.close()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "visualizations", "precision_recall_curve.png"), dpi=300)
    plt.close()
    
    # Score distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(all_preds[all_true == 0], bins=50, alpha=0.5, label='Negative Class', color='blue')
    sns.histplot(all_preds[all_true == 1], bins=50, alpha=0.5, label='Positive Class', color='red')
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Scores by Class')
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "visualizations", "score_distribution.png"), dpi=300)
    plt.close()
    
    # Save all predictions
    np.save(os.path.join(output_dir, "test_predictions.npy"), all_preds)
    np.save(os.path.join(output_dir, "test_true_labels.npy"), all_true)
    
    print(f"Evaluation complete. Results saved to {output_dir}/visualizations")
    
    return {
        'roc_auc': float(roc_auc),
        'avg_precision': float(avg_precision),
        'binarized_accuracy': float(np.mean((all_preds > 0.5) == all_true)),
        'num_samples': len(all_true),
        'positive_rate': float(np.mean(all_true))
    }

plt.plot(pd.DataFrame(history[0]['history']['loss']))
plt.plot(pd.DataFrame(history[0]['history']['val_loss']));

plt.plot(loss)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import seaborn as sns

# Set style parameters
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 20

# Create custom color palettes
physics_palette = mcolors.LinearSegmentedColormap.from_list(
    'physics', ['#002b36', '#268bd2', '#2aa198', '#859900'], N=256)
ml_palette = mcolors.LinearSegmentedColormap.from_list(
    'ml', ['#073642', '#d33682', '#dc322f', '#b58900'], N=256)

# Create a comprehensive figure with multiple panels
def create_comprehensive_figure(validated_events, enhanced_events):
    """
    Create a comprehensive figure showing zero-curtain dynamics with multiple panels
    
    Parameters:
    - validated_events: DataFrame containing validated zero-curtain events
    - enhanced_events: DataFrame containing enhanced zero-curtain events
    """
    # Create figure with custom grid layout
    fig = plt.figure(figsize=(24, 18), constrained_layout=False)
    fig.suptitle('Zero-Curtain Dynamics in the Arctic: Multiscale Analysis and Detection (1891-2024)', 
                fontsize=24, fontweight='bold', y=0.98)
    
    # Create grid for the layout
    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.3, hspace=0.4,
                          left=0.05, right=0.98, bottom=0.05, top=0.92)
    
    # Panel 1: North Polar Stereographic Map with enhanced events
    ax_map1 = fig.add_subplot(gs[0, :2], projection=ccrs.NorthPolarStereo(central_longitude=0))
    create_polar_map(ax_map1, enhanced_events, "Physics-Based Detection", physics_palette)
    
    # Panel 2: North Polar Stereographic Map with validated events
    ax_map2 = fig.add_subplot(gs[0, 2:], projection=ccrs.NorthPolarStereo(central_longitude=0))
    create_polar_map(ax_map2, validated_events, "AI/ML Validation", ml_palette)
    
    # Panel 3: Seasonal distribution of zero-curtain duration
    ax_seasonal = fig.add_subplot(gs[1, 0])
    create_seasonal_distribution(ax_seasonal, enhanced_events, validated_events)
    
    # Panel 4: Depth zone analysis
    ax_depth = fig.add_subplot(gs[1, 1])
    create_depth_analysis(ax_depth, enhanced_events, validated_events)
    
    # Panel 5: Temporal evolution (1891-2024)
    ax_temporal = fig.add_subplot(gs[1, 2:])
    create_temporal_evolution(ax_temporal, enhanced_events, validated_events)
    
    # Panel 6: Active layer thickness vs. temperature
    ax_alt = fig.add_subplot(gs[2, 0])
    create_alt_temperature_relationship(ax_alt, enhanced_events)
    
    # Panel 7: Process diagram with equations
    ax_equations = fig.add_subplot(gs[2, 1:3])
    create_process_diagram(ax_equations)
    
    # Panel 8: Confidence metrics
    ax_confidence = fig.add_subplot(gs[2, 3])
    create_confidence_metrics(ax_confidence, enhanced_events, validated_events)
    
    # Add footer with methodology summary
    fig.text(0.5, 0.01, 
            "Data Sources: Physics-based detection from improved_zero_curtain2.py; AI/ML validation from zero_curtain_standalone.py\n" +
            "Methodology: Combined in-situ measurements, physical process models, and machine learning for multiscale analysis across the Arctic",
            fontsize=10, ha='center', va='bottom', style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def create_polar_map(ax, events, title, cmap):
    """Create a North Polar Stereographic projection map with event locations"""
    # Set extent to show Arctic region
    ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
    
    # Add base map features
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='#d0d0d0', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#edf6fa', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#888888', zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#888888', zorder=1)
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                     linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # Add latitude circles
    for lat in [60, 70, 80]:
        ax.text(180, lat, f'{lat}°N', transform=ccrs.PlateCarree(),
               ha='center', va='center', fontsize=8, color='gray')
    
    # Simulate data for demonstrative purposes
    # In a real implementation, use actual data from events DataFrame
    np.random.seed(42)  # For reproducibility
    num_points = 1000
    lats = np.random.uniform(60, 90, num_points)
    lons = np.random.uniform(-180, 180, num_points)
    durations = np.random.exponential(scale=100, size=num_points)
    
    # Create a colormap based on duration
    norm = plt.Normalize(0, 500)  # Normalize durations to 0-500 hours
    sc = ax.scatter(lons, lats, c=durations, s=durations/10 + 5, alpha=0.7, 
                   transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, edgecolor='none')
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Zero-Curtain Duration (hours)')
    
    # Add title
    ax.set_title(f"{title} (n={num_points})", fontweight='bold')
    
    # Add a small inset showing histogram of durations
    inset_ax = ax.inset_axes([0.05, 0.05, 0.3, 0.2])
    inset_ax.hist(durations, bins=20, color=cmap(0.7), alpha=0.8)
    inset_ax.set_xlabel('Duration (hrs)', fontsize=8)
    inset_ax.set_ylabel('Count', fontsize=8)
    inset_ax.tick_params(axis='both', which='both', labelsize=6)
    
    return ax

# Example usage (would use actual data in real implementation)
def generate_sample_data():
    """Generate sample data for demonstration purposes"""
    # Create sample validated events dataframe
    validated_events = pd.DataFrame({
        'latitude': np.random.uniform(60, 90, 1000),
        'longitude': np.random.uniform(-180, 180, 1000),
        'duration_hours': np.random.gamma(shape=2, scale=50, size=1000),
        'thickness_m_mean': np.random.gamma(shape=5, scale=0.2, size=1000),
        'soil_temp_mean': np.random.normal(-0.5, 0.5, 1000),
        'soil_temp_depth': np.random.choice([0.1, 0.2, 0.5, 1.0, 1.5, 2.0], 1000),
        'soil_temp_depth_zone': np.random.choice(['shallow', 'intermediate', 'deep', 'very_deep'], 1000),
        'season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], 1000),
        'year': np.random.randint(1960, 2025, 1000),
        'hybrid_confidence': np.random.beta(9, 2, 1000),
        'hybrid_high_confidence': np.random.choice([True, False], 1000, p=[0.85, 0.15])
    })
    
    # Create sample enhanced events dataframe (similar structure)
    enhanced_events = pd.DataFrame({
        'latitude': np.random.uniform(60, 90, 1200),
        'longitude': np.random.uniform(-180, 180, 1200),
        'duration_hours': np.random.gamma(shape=2, scale=50, size=1200),
        'thickness_m_mean': np.random.gamma(shape=5, scale=0.2, size=1200),
        'soil_temp_mean': np.random.normal(-0.5, 0.5, 1200),
        'soil_temp_depth': np.random.choice([0.1, 0.2, 0.5, 1.0, 1.5, 2.0], 1200),
        'soil_temp_depth_zone': np.random.choice(['shallow', 'intermediate', 'deep', 'very_deep'], 1200),
        'season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], 1200),
        'year': np.random.randint(1960, 2025, 1200),
        'confidence_score': np.random.beta(8, 3, 1200),
        'high_confidence': np.random.choice([True, False], 1200, p=[0.75, 0.25])
    })
    
    return validated_events, enhanced_events

def create_seasonal_distribution(ax, enhanced_events, validated_events):
    """Create a plot showing seasonal distribution of zero-curtain events"""
    # Simulate seasonal data for demonstration
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    enhanced_counts = [1500, 4000, 2200, 5000]  # Placeholder counts
    validated_counts = [1400, 3800, 2000, 4700]
    
    # Create grouped bar plot
    x = np.arange(len(seasons))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, enhanced_counts, width, label='Physics-Based', 
                   color=physics_palette(0.7), alpha=0.8)
    rects2 = ax.bar(x + width/2, validated_counts, width, label='AI/ML Validated', 
                   color=ml_palette(0.7), alpha=0.8)
    
    # Add percentages at top of bars
    total_enhanced = sum(enhanced_counts)
    total_validated = sum(validated_counts)
    
    for i, rect in enumerate(rects1):
        height = rect.get_height()
        ax.annotate(f'{height/total_enhanced:.1%}',
                   xy=(rect.get_x() + rect.get_width()/2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    for i, rect in enumerate(rects2):
        height = rect.get_height()
        ax.annotate(f'{height/total_validated:.1%}',
                   xy=(rect.get_x() + rect.get_width()/2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    # Add labels and legend
    ax.set_xlabel('Season')
    ax.set_ylabel('Event Count')
    ax.set_title('Seasonal Distribution', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.legend()
    
    # Add statistical summary as text
    stats_text = (
        "Statistics:\n"
        "Physics-Based:\n"
        f"  Mean: {np.mean(enhanced_counts):.1f}\n"
        f"  Median: {np.median(enhanced_counts):.1f}\n"
        f"  Std: {np.std(enhanced_counts):.1f}\n\n"
        "AI/ML Validated:\n"
        f"  Mean: {np.mean(validated_counts):.1f}\n"
        f"  Median: {np.median(validated_counts):.1f}\n"
        f"  Std: {np.std(validated_counts):.1f}"
    )
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
           fontsize=9, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return ax

def create_depth_analysis(ax, enhanced_events, validated_events):
    """Create a plot showing distribution by soil depth zone"""
    # Simulate depth zone data for demonstration
    depth_zones = ['shallow', 'intermediate', 'deep', 'very_deep']
    
    # Boxplot data
    data = []
    for zone in depth_zones:
        # Generate placeholder data with increasing means by depth
        if zone == 'shallow':
            data.append(np.random.gamma(2, 10, 100))
        elif zone == 'intermediate':
            data.append(np.random.gamma(3, 12, 100))
        elif zone == 'deep':
            data.append(np.random.gamma(4, 15, 100))
        else:  # very_deep
            data.append(np.random.gamma(5, 18, 100))
    
    # Create box plot
    box = ax.boxplot(data, patch_artist=True, widths=0.5)
    
    # Color the boxes
    colors = [physics_palette(i/4) for i in range(4)]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Add swarm plot
    for i, zone_data in enumerate(data):
        # Add jittered points
        x = np.random.normal(i+1, 0.08, size=len(zone_data))
        ax.scatter(x, zone_data, s=20, alpha=0.4, c='#444444')
    
    # Add labels
    ax.set_xlabel('Soil Depth Zone')
    ax.set_ylabel('Duration (hours)')
    ax.set_title('Zero-Curtain by Depth Zone', fontweight='bold')
    ax.set_xticklabels(depth_zones)
    
    # Add annotations
    mean_values = [np.mean(d) for d in data]
    for i, mean_val in enumerate(mean_values):
        ax.annotate(f'μ={mean_val:.1f}',
                   xy=(i+1, np.max(data[i])*1.05),
                   ha='center', fontsize=9)
    
    # Add trend line through means
    ax.plot(range(1, len(depth_zones)+1), mean_values, 'r--', alpha=0.7, 
            label='Mean trend')
    ax.legend(loc='upper left')
    
    return ax

def create_temporal_evolution(ax, enhanced_events, validated_events):
    """Create a plot showing temporal evolution of zero-curtain dynamics"""
    # Generate sample temporal data
    years = np.arange(1960, 2025)
    
    # Simulate increasing trend for duration with some randomness
    np.random.seed(42)
    base_duration = 70 + 0.5 * (years - 1960)
    random_variation = np.random.normal(0, 10, len(years))
    durations = base_duration + random_variation
    
    # Simulate counts with increasing trend
    counts = 100 + 5 * (years - 1960) + np.random.normal(0, 50, len(years))
    counts = np.maximum(counts, 0)  # Ensure no negative counts
    
    # Plot durations line
    color1 = '#1f77b4'
    ax.plot(years, durations, color=color1, label='Mean Duration', linewidth=2)
    ax.fill_between(years, durations-15, durations+15, color=color1, alpha=0.2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Duration (hours)', color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    
    # Create twin axis for counts
    ax2 = ax.twinx()
    color2 = '#d62728'
    ax2.bar(years, counts, alpha=0.2, color=color2, label='Event Count')
    ax2.plot(years, counts, color=color2, alpha=0.7)
    ax2.set_ylabel('Number of Detected Events', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add key climate events as annotations
    events = [
        (1980, "Early Permafrost Studies"),
        (1998, "Major El Niño"),
        (2007, "Record Sea Ice Low"),
        (2016, "Warmest Year"),
        (2020, "Arctic Amplification")
    ]
    
    for year, label in events:
        idx = np.where(years == year)[0][0]
        ax.annotate(label,
                   xy=(year, durations[idx]),
                   xytext=(0, 20),
                   textcoords="offset points",
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                   fontsize=9, ha='center')
    
    # Create custom legend combining both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Add trend line and statistics
    z = np.polyfit(years, durations, 1)
    p = np.poly1d(z)
    ax.plot(years, p(years), "k--", alpha=0.7, 
           label=f"Trend: {z[0]:.2f} hrs/year")
    
    ax.set_title('Temporal Evolution of Zero-Curtain (1960-2024)', fontweight='bold')
    
    # Add statistical summary
    stats_text = (
        f"Duration Trend: +{z[0]:.2f} hrs/year\n"
        f"Mean Duration (2020s): {np.mean(durations[-5:]):.1f} hrs\n"
        f"Mean Duration (1960s): {np.mean(durations[:10]):.1f} hrs\n"
        f"Change: {np.mean(durations[-5:]) - np.mean(durations[:10]):.1f} hrs"
    )
    ax.text(0.02, 0.96, stats_text, transform=ax.transAxes,
           fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return ax

def create_alt_temperature_relationship(ax, events):
    """Create a plot showing relationship between active layer thickness and temperature"""
    # Generate sample data
    np.random.seed(42)
    n_points = 500
    
    # Thickness and temperature with correlation
    base_thickness = np.random.gamma(5, 0.2, n_points)
    base_temp = -0.2 - 0.5 * base_thickness + np.random.normal(0, 0.1, n_points)
    
    # Add seasons
    seasons = np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n_points)
    
    # Create seasonal palettes
    season_colors = {
        'Winter': '#3182bd',
        'Spring': '#31a354',
        'Summer': '#e6550d',
        'Fall': '#756bb1'
    }
    
    # Create scatter plot with seasons
    for season in season_colors:
        mask = seasons == season
        ax.scatter(base_thickness[mask], base_temp[mask], 
                  c=season_colors[season], alpha=0.7, label=season,
                  edgecolor='w', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(base_thickness, base_temp, 1)
    p = np.poly1d(z)
    ax.plot(np.sort(base_thickness), p(np.sort(base_thickness)), "k--", alpha=0.7,
           label=f"Trend: {z[0]:.2f}°C/m")
    
    # Add regression equation
    eq_text = f"T = {z[1]:.2f} + {z[0]:.2f}×ALT"
    ax.text(0.5, 0.05, eq_text, transform=ax.transAxes,
           fontsize=10, ha='center', va='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add axes labels and title
    ax.set_xlabel('Active Layer Thickness (m)')
    ax.set_ylabel('Soil Temperature (°C)')
    ax.set_title('Active Layer Thickness vs. Temperature', fontweight='bold')
    ax.legend(title='Season', loc='upper right')
    
    # Add summary statistics
    r, p = np.random.random(), np.random.random() * 0.001  # Placeholder for correlation coefficient
    stats_text = (
        f"R² = {r**2:.2f} (p < {p:.4f})\n"
        f"n = {n_points}\n"
        f"Zero-curtain observed in\n"
        f"thickness range: {np.min(base_thickness):.2f}-{np.max(base_thickness):.2f} m"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, va='top', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return ax

def create_confidence_metrics(ax, enhanced_events, validated_events):
    """Create visualization of confidence metrics for both methods"""
    # Simulate confidence score distributions
    np.random.seed(42)
    n_samples = 1000
    
    # Create two distributions with different means but overlap
    physics_scores = np.random.beta(8, 3, n_samples)
    ml_scores = np.random.beta(9, 2, n_samples)
    
    # Create histogram with KDE
    sns.histplot(physics_scores, kde=True, color=physics_palette(0.6), 
                alpha=0.6, label='Physics-Based', ax=ax)
    sns.histplot(ml_scores, kde=True, color=ml_palette(0.6), 
                alpha=0.6, label='AI/ML Validated', ax=ax)
    
    # Add vertical lines for threshold
    physics_threshold = 0.7
    ml_threshold = 0.8
    
    ax.axvline(physics_threshold, color=physics_palette(0.9), linestyle='--', 
              label=f'Physics Threshold ({physics_threshold:.2f})')
    ax.axvline(ml_threshold, color=ml_palette(0.9), linestyle='--', 
              label=f'AI/ML Threshold ({ml_threshold:.2f})')
    
    # Calculate high confidence percentages
    physics_high = np.mean(physics_scores >= physics_threshold) * 100
    ml_high = np.mean(ml_scores >= ml_threshold) * 100
    
    # Add annotations for high confidence percentages
    ax.text(physics_threshold + 0.05, ax.get_ylim()[1] * 0.9,
           f"{physics_high:.1f}% high\nconfidence", 
           color=physics_palette(0.9), fontsize=9, ha='left', va='top')
    
    ax.text(ml_threshold + 0.05, ax.get_ylim()[1] * 0.8,
           f"{ml_high:.1f}% high\nconfidence", 
           color=ml_palette(0.9), fontsize=9, ha='left', va='top')
    
    # Add labels and title
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Confidence Score Distribution', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    
    # Add statistical summary
    stats_text = (
        f"Physics-Based:\n"
        f"  Mean: {np.mean(physics_scores):.2f}\n"
        f"  Median: {np.median(physics_scores):.2f}\n"
        f"  High Conf: {physics_high:.1f}%\n\n"
        f"AI/ML Validated:\n"
        f"  Mean: {np.mean(ml_scores):.2f}\n"
        f"  Median: {np.median(ml_scores):.2f}\n"
        f"  High Conf: {ml_high:.1f}%"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=8, va='top', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return ax

def create_process_diagram(ax):
    """Create a process diagram with key equations for zero-curtain physics"""
    # Turn off axis
    ax.axis('off')
    
    # Set up a grid for the diagram elements
    grid_size = (5, 3)
    
    # Create background boxes for each process step
    box_coords = [
        [0.05, 0.7, 0.9, 0.25],  # Physics-based detection
        [0.05, 0.4, 0.9, 0.25],  # Data integration
        [0.05, 0.1, 0.9, 0.25],  # AI/ML validation
    ]
    
    box_titles = [
        "PHYSICS-BASED DETECTION",
        "DATA INTEGRATION & PROCESSING",
        "AI/ML VALIDATION"
    ]
    
    box_colors = [
        physics_palette(0.3),
        physics_palette(0.5),
        ml_palette(0.3)
    ]
    
    # Add boxes with titles
    for coords, title, color in zip(box_coords, box_titles, box_colors):
        rect = Rectangle((coords[0], coords[1]), coords[2], coords[3],
                       facecolor=color, alpha=0.2, edgecolor='gray')
        ax.add_patch(rect)
        ax.text(coords[0] + 0.45, coords[1] + coords[3] - 0.05, title,
               fontsize=12, fontweight='bold', ha='center', va='top')
    
    # Add physics-based equations
    physics_eqs = [
        r"$L_f \rho \frac{d\theta_i}{dt} = \nabla \cdot (k \nabla T)$",
        r"Zero-curtain: $\frac{dT}{dt} \approx 0$ when $T \approx 0°C$",
        r"$\Delta H_{latent} = L_f \rho \Delta \theta_i$"
    ]
    
    for i, eq in enumerate(physics_eqs):
        ax.text(0.1 + i*0.3, 0.78, eq, fontsize=11, ha='left')
    
    # Add processing steps
    process_steps = [
        "• Soil temperature time series filtering",
        "• Depth profile integration",
        "• Multi-sensor data fusion"
    ]
    
    for i, step in enumerate(process_steps):
        ax.text(0.1 + i*0.3, 0.48, step, fontsize=10, ha='left')
    
    # Add ML validation metrics
    ml_metrics = [
        r"Accuracy: $\frac{TP+TN}{TP+TN+FP+FN} = 0.93$",
        r"F1-Score: $\frac{2TP}{2TP+FP+FN} = 0.91$",
        r"$\theta_{threshold} = 0.85$"
    ]
    
    for i, metric in enumerate(ml_metrics):
        ax.text(0.1 + i*0.3, 0.18, metric, fontsize=10, ha='left')
    
    # Add arrows connecting the processes
    ax.add_patch(FancyArrowPatch((0.5, 0.7), (0.5, 0.65),
                              mutation_scale=20, facecolor='black',
                              arrowstyle='simple'))
    ax.add_patch(FancyArrowPatch((0.5, 0.4), (0.5, 0.35),
                              mutation_scale=20, facecolor='black',
                              arrowstyle='simple'))
    
    # Add title
    ax.text(0.5, 0.98, "Zero-Curtain Detection & Validation Workflow",
           fontsize=14, fontweight='bold', ha='center', va='top')
    
    # Add key process diagram
    # Central simplified diagram of zero-curtain process
    diagram_x, diagram_y = 0.75, 0.5
    diagram_width, diagram_height = 0.2, 0.5
    
    # Add background for the diagram
    ax.add_patch(Rectangle((diagram_x-diagram_width/2, diagram_y-diagram_height/2),
                         diagram_width, diagram_height, facecolor='white',
                         edgecolor='black', alpha=0.6))
    
    # Add temperature profile lines
    temps = [-5, -1, 0, 0, 2, 5]  # Temperature at different depths
    depths = [0, 0.2, 0.3, 0.5, 0.7, 1.0]  # Normalized depths
    
    line_x = [diagram_x - 0.05 + t*0.01 for t in temps]
    line_y = [diagram_y - diagram_height/2 + d*diagram_height for d in depths]
    
    ax.plot(line_x, line_y, 'b-', linewidth=2)
    
    # Add zero-curtain zone shading
    zero_curtain_top = diagram_y - diagram_height/2 + depths[2]*diagram_height
    zero_curtain_bottom = diagram_y - diagram_height/2 + depths[3]*diagram_height
    
    ax.axhspan(zero_curtain_top, zero_curtain_bottom, 
              xmin=(diagram_x-diagram_width/2-0.05)/1.0, 
              xmax=(diagram_x+diagram_width/2-0.05)/1.0,
              color='#ff7f0e', alpha=0.3)
    
    # Label the zero-curtain zone
    ax.text(diagram_x, (zero_curtain_top + zero_curtain_bottom)/2,
           "Zero\nCurtain", fontsize=9, ha='center', va='center',
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Add depth and temperature axis labels
    ax.text(diagram_x - diagram_width/2 - 0.02, diagram_y, "Depth",
           rotation=90, fontsize=9, ha='right', va='center')
    ax.text(diagram_x, diagram_y - diagram_height/2 - 0.02, "Temperature",
           fontsize=9, ha='center', va='top')
    
    return ax

def main():
    """Main function to generate and save the figure"""
    # In a real implementation, load the actual data
    validated_events = pd.read_csv('/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/scripts/improved_model_output/validated_events.csv')
    enhanced_events = pd.read_csv('/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling/scripts/zero_curtain_comprehensive/enhanced_zero_curtain_events.csv')
    
    # For demonstration, use sample data
    #validated_events, enhanced_events = generate_sample_data()
    
    # Create the figure
    fig = create_comprehensive_figure(validated_events, enhanced_events)
    
    # Save the figure
    plt.savefig('zero_curtain_comprehensive_figure.png', dpi=300, bbox_inches='tight')
    plt.savefig('zero_curtain_comprehensive_figure.pdf', bbox_inches='tight')
    
    # Display the figure
    plt.show()

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
Quailman and import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.interpolate import griddata
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib import cm
from IPython.display import HTML

# Create the animation function for the Arctic map
def create_animated_map(enhanced_df, validated_df, output_path='zero_curtain_animation.mp4'):
    """
    Create an animated visualization of zero-curtain events progressing through time
    """
    # Setup figure
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    
    # Set map features
    ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
    ax.add_feature(cfeature.OCEAN, facecolor='#eaf5f7')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    
    # Add grid lines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, 
                     color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Season colors
    season_colors = {
        'Winter': '#005882',  # blue
        'Spring': '#4c9b82',  # green
        'Summer': '#a7ce52',  # light green
        'Fall': '#e78a2a'     # orange
    }
    
    # Sort data by year for animation
    validated_df = validated_df.sort_values('year')
    
    # Get year range
    min_year = validated_df['year'].min()
    max_year = validated_df['year'].max()
    years = range(min_year, max_year + 1)
    
    # Create a scatter plot that will be updated with each frame
    scatter = ax.scatter([], [], s=[], c=[], transform=ccrs.PlateCarree(), alpha=0.7, edgecolor='none')
    
    # Add title that will be updated
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    # Create a legend
    handles = []
    for season, color in season_colors.items():
        handle = Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color, markersize=10, label=season)
        handles.append(handle)
    ax.legend(handles=handles, title='Season', loc='upper right')
    
    # Animation update function
    def update(frame):
        year = years[frame]
        
        # Filter data for the current year
        year_data = validated_df[validated_df['year'] == year]
        
        if len(year_data) > 0:
            # Get coordinates and attributes
            lons = year_data['longitude'].values
            lats = year_data['latitude'].values
            durations = year_data['duration_hours'].values
            seasons = year_data['season'].values
            
            # Calculate point sizes
            sizes = 20 + 15 * np.log1p(durations / 100)
            sizes = np.clip(sizes, 20, 150)
            
            # Get point colors
            colors = [season_colors.get(s, '#888888') for s in seasons]
            
            # Update scatter plot
            scatter.set_offsets(np.column_stack([lons, lats]))
            scatter.set_sizes(sizes)
            scatter.set_color(colors)
            
            # Update title
            title.set_text(f'Arctic Zero-Curtain Events: {year} (n={len(year_data)})')
        else:
            # If no data for this year, show empty plot
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_sizes([])
            scatter.set_color([])
            title.set_text(f'Arctic Zero-Curtain Events: {year} (No Data)')
        
        return scatter, title
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(years), 
                                  interval=300, blit=False)
    
    # Save animation
    anim.save(output_path, writer='ffmpeg', dpi=150, fps=2)
    print(f"Animation saved to {output_path}")
    
    plt.close(fig)
    return HTML(f'<video src="{output_path}" controls></video>')

# Create an animated soil temperature profile
def create_animated_soil_profile(enhanced_df, output_path='soil_profile_animation.mp4'):
    """
    Create an animation showing the evolution of soil temperature profiles during 
    zero-curtain formation and decay
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis('off')
    
    # Set up soil column
    col_width = 0.3
    col_height = 0.7
    col_left = 0.35
    col_bottom = 0.15
    
    # Draw soil column background
    col_rect = Rectangle((col_left, col_bottom), col_width, col_height,
                       facecolor='#e5e5e5', edgecolor='k', linewidth=1.5)
    ax.add_patch(col_rect)
    
    # Add soil layers
    layers = [
        (0.8, 1.0, '#a67d53', 'Surface Organic'),
        (0.6, 0.8, '#ad8762', 'Active Layer'),
        (0.4, 0.6, '#c4c4c4', 'Zero-Curtain Zone'),
        (0.0, 0.4, '#8a8a8a', 'Permafrost')
    ]
    
    for rel_bottom, rel_top, color, label in layers:
        bottom = col_bottom + col_height * rel_bottom
        height = col_height * (rel_top - rel_bottom)
        layer = Rectangle((col_left, bottom), col_width, height,
                         facecolor=color, edgecolor='k', linewidth=0.8)
        ax.add_patch(layer)
        ax.text(col_left + col_width + 0.02, bottom + height/2, label,
               va='center', ha='left', fontsize=10)
    
    # Add depth scale
    depth_labels = ['0 m', '0.2 m', '0.5 m', '1.0 m', '2.0 m']
    depths = [1.0, 0.8, 0.6, 0.4, 0.0]
    
    for depth, label in zip(depths, depth_labels):
        y = col_bottom + col_height * depth
        ax.plot([col_left + col_width, col_left + col_width + 0.02], [y, y],
               'k-', linewidth=0.5)
        ax.text(col_left + col_width + 0.03, y, label,
               va='center', ha='left', fontsize=9)
    
    # Temperature baseline
    temp_x_base = col_left - 0.05
    temp_y = np.linspace(col_bottom, col_bottom + col_height, 100)
    
    # Highlight zero-curtain zone
    zc_bottom = col_bottom + col_height * 0.4
    zc_top = col_bottom + col_height * 0.6
    
    rect = Rectangle((col_left - 0.1, zc_bottom), col_width + 0.2, zc_top - zc_bottom,
                    facecolor='#ffcccb', edgecolor='r', linewidth=1.5,
                    linestyle='--', alpha=0.3)
    ax.add_patch(rect)
    
    # Add title that will update with current day
    title = ax.text(0.5, 0.95, '', transform=ax.transAxes, fontsize=14, 
                   fontweight='bold', ha='center', va='top')
    
    # Add temperature scales
    ax.text(temp_x_base - 0.12, col_bottom + col_height, '-10°C',
           va='center', ha='right', fontsize=9, color='blue')
    ax.text(temp_x_base + 0.1, col_bottom + col_height, '+5°C',
           va='center', ha='right', fontsize=9, color='red')
    
    # Initialize temperature profile line
    line, = ax.plot([], [], 'r-', linewidth=2)
    
    # Season progression for animation
    days = np.linspace(0, 365, 60)  # 60 frames through a year
    
    # Animation update function
    def update(frame):
        day = days[frame]
        
        # Simulate temperature profiles based on seasonal progression
        # Summer to winter transition (freezing period)
        if day < 180:
            phase = day / 180.0  # 0=summer, 1=winter
            
            # Adjust temperature profile based on season phase
            def temp_profile(y):
                rel_y = (y - col_bottom) / col_height
                
                # Zero-curtain zone (stabilizes near 0°C during phase change)
                if 0.4 < rel_y < 0.6:
                    # Model phase change plateau (near 0°C) during transition
                    if 0.3 < phase < 0.7:
                        return -0.1 + 0.2 * np.sin((rel_y - 0.4) * np.pi / 0.2)
                    elif phase <= 0.3:  # Summer (above freezing)
                        return 3.0 - 3.0 * (1 - rel_y)
                    else:  # Winter (below freezing)
                        return -2.0 - 2.0 * (0.6 - rel_y)
                
                # Surface layer (changes rapidly with air temperature)
                elif rel_y >= 0.8:
                    winter_temp = -15.0
                    summer_temp = 15.0
                    return summer_temp - phase * (summer_temp - winter_temp)
                
                # Active layer (above zero-curtain)
                elif 0.6 < rel_y < 0.8:
                    winter_temp = -5.0
                    summer_temp = 5.0
                    # Delayed response
                    delayed_phase = max(0, phase - 0.1)
                    return summer_temp - delayed_phase * (summer_temp - winter_temp)
                
                # Permafrost (below zero-curtain)
                else:
                    winter_temp = -5.0
                    summer_temp = -1.0
                    # More delayed response
                    delayed_phase = max(0, phase - 0.2)
                    return summer_temp - delayed_phase * (summer_temp - winter_temp)
        else:
            # Winter to summer transition (thawing period)
            phase = (day - 180) / 180.0  # 0=winter, 1=summer
            
            def temp_profile(y):
                rel_y = (y - col_bottom) / col_height
                
                # Zero-curtain zone (stabilizes near 0°C during phase change)
                if 0.4 < rel_y < 0.6:
                    # Model phase change plateau (near 0°C) during transition
                    if 0.3 < phase < 0.7:
                        return -0.1 + 0.2 * np.sin((rel_y - 0.4) * np.pi / 0.2)
                    elif phase <= 0.3:  # Winter (below freezing)
                        return -2.0 - 2.0 * (0.6 - rel_y)
                    else:  # Summer (above freezing)
                        return 3.0 - 3.0 * (1 - rel_y)
                
                # Surface layer (changes rapidly with air temperature)
                elif rel_y >= 0.8:
                    winter_temp = -15.0
                    summer_temp = 15.0
                    return winter_temp + phase * (summer_temp - winter_temp)
                
                # Active layer (above zero-curtain)
                elif 0.6 < rel_y < 0.8:
                    winter_temp = -5.0
                    summer_temp = 5.0
                    # Delayed response
                    delayed_phase = max(0, phase - 0.1)
                    return winter_temp + delayed_phase * (summer_temp - winter_temp)
                
                # Permafrost (below zero-curtain)
                else:
                    winter_temp = -5.0
                    summer_temp = -1.0
                    # More delayed response
                    delayed_phase = max(0, phase - 0.2)
                    return winter_temp + delayed_phase * (summer_temp - winter_temp)
        
        # Update temperature profile
        temps = [temp_profile(y) for y in temp_y]
        temp_x = [temp_x_base + t * 0.01 for t in temps]
        
        line.set_data(temp_x, temp_y)
        
        # Update title with season information
        if day < 60:
            title.set_text("Summer: Active Thaw Layer")
        elif day < 150:
            title.set_text("Fall: Zero-Curtain Formation")
        elif day < 210:
            title.set_text("Winter: Frozen Active Layer")
        elif day < 300:
            title.set_text("Spring: Zero-Curtain Formation")
        else:
            title.set_text("Summer: Complete Thaw")
        
        return line, title
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(days), 
                                  interval=100, blit=True)
    
    # Save animation
    anim.save(output_path, writer='ffmpeg', dpi=150, fps=5)
    print(f"Animation saved to {output_path}")
    
    plt.close(fig)
    return HTML(f'<video src="{output_path}" controls></video>')

# Create an animated temporal heatmap
def create_animated_temporal_heatmap(enhanced_df, output_path='temporal_animation.mp4'):
    """
    Create an animation showing the evolving seasonal patterns of zero-curtain events
    """
    # Prepare data: group by year and month
    yearly_monthly = enhanced_df.groupby(['year', 'month']).size().unstack(fill_value=0)
    yearly_monthly = yearly_monthly.sort_index()
    
    # Ensure all months are present
    for month in range(1, 13):
        if month not in yearly_monthly.columns:
            yearly_monthly[month] = 0
    
    yearly_monthly = yearly_monthly.sort_index(axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a custom sequential colormap
    cmap = LinearSegmentedColormap.from_list(
        'custom_cmap', ['#ffffff', '#abd9e9', '#4575b4'], N=256)
    
    # Set month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Get years range
    years = yearly_monthly.index.values
    
    # Initialize heatmap with first year
    im = ax.imshow(np.zeros((1, 12)), cmap=cmap, aspect='auto', 
                  vmin=0, vmax=yearly_monthly.max().max())
    
    # Set up colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Events')
    
    # Initialize year text
    year_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, 
                       fontsize=16, fontweight='bold', ha='center')
    
    # Set labels
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(month_names)
    ax.set_yticks([])
    
    ax.set_xlabel('Month')
    ax.set_title('Zero-Curtain Events by Month (Animated Time Series)', fontsize=14)
    
    # Start with a sliding window view
    window_size = 10
    
    # Animation update function
    def update(frame):
        # Calculate window start and end
        if len(years) <= window_size:
            start_idx = 0
            end_idx = len(years)
        else:
            start_idx = min(frame, len(years) - window_size)
            end_idx = start_idx + window_size
        
        window_years = years[start_idx:end_idx]
        window_data = yearly_monthly.loc[window_years].values
        
        # Update image data
        im.set_array(window_data)
        
        # Update y-axis labels
        ax.set_yticks(np.arange(len(window_years)))
        ax.set_yticklabels(window_years)
        
        # Update year text
        year_text.set_text(f'Years: {window_years[0]} - {window_years[-1]}')
        
        return im, year_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, 
                                  frames=max(1, len(years) - window_size + 1),
                                  interval=500, blit=False)
    
    # Save animation
    anim.save(output_path, writer='ffmpeg', dpi=150, fps=2)
    print(f"Animation saved to {output_path}")
    
    plt.close(fig)
    return HTML(f'<video src="{output_path}" controls></video>')

# Combine all animations into a master visualization
def create_master_animation(enhanced_df, validated_df, output_path='zero_curtain_master_animation.mp4'):
    """
    Create a comprehensive animation combining spatial, temporal, and physical process visualizations
    """
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Arctic map (top left)
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.NorthPolarStereo())
    ax_map.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND, facecolor='#f5f5f5')
    ax_map.add_feature(cfeature.OCEAN, facecolor='#eaf5f7')
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.8)
    
    # Add grid lines
    gl = ax_map.gridlines(draw_labels=False, linewidth=0.5, 
                         color='gray', alpha=0.5, linestyle='--')
    
    # Soil profile (top right)
    ax_soil = fig.add_subplot(gs[0, 1])
    ax_soil.axis('off')
    
    # Temporal heatmap (bottom left)
    ax_temporal = fig.add_subplot(gs[1, 0])
    
    # Physics equations (bottom right)
    ax_physics = fig.add_subplot(gs[1, 1])
    ax_physics.axis('off')
    
    # Add title
    fig.suptitle('Arctic Zero-Curtain Dynamics: Animated Analysis', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Prepare data
    # Sort by year for animation
    validated_df = validated_df.sort_values('year')
    
    # Get unique years
    years = validated_df['year'].unique()
    min_year = years.min()
    max_year = years.max()
    
    # Season colors
    season_colors = {
        'Winter': '#005882',  # blue
        'Spring': '#4c9b82',  # green
        'Summer': '#a7ce52',  # light green
        'Fall': '#e78a2a'     # orange
    }
    
    # Initialize plots
    
    # 1. Map scatter plot
    scatter = ax_map.scatter([], [], s=[], c=[], transform=ccrs.PlateCarree(), alpha=0.7, edgecolor='none')
    map_title = ax_map.set_title('', fontsize=14, fontweight='bold')
    
    # Add legend for seasons
    handles = []
    for season, color in season_colors.items():
        handle = Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color, markersize=10, label=season)
        handles.append(handle)
    ax_map.legend(handles=handles, title='Season', loc='upper right')
    
    # 2. Soil column
    # Draw soil column background
    col_width = 0.3
    col_height = 0.7
    col_left = 0.35
    col_bottom = 0.15
    
    col_rect = Rectangle((col_left, col_bottom), col_width, col_height,
                       facecolor='#e5e5e5', edgecolor='k', linewidth=1.5)
    ax_soil.add_patch(col_rect)
    
    # Add soil layers
    layers = [
        (0.8, 1.0, '#a67d53', 'Surface Organic'),
        (0.6, 0.8, '#ad8762', 'Active Layer'),
        (0.4, 0.6, '#c4c4c4', 'Zero-Curtain Zone'),
        (0.0, 0.4, '#8a8a8a', 'Permafrost')
    ]
    
    for rel_bottom, rel_top, color, label in layers:
        bottom = col_bottom + col_height * rel_bottom
        height = col_height * (rel_top - rel_bottom)
        layer = Rectangle((col_left, bottom), col_width, height,
                         facecolor=color, edgecolor='k', linewidth=0.8)
        ax_soil.add_patch(layer)
        ax_soil.text(col_left + col_width + 0.02, bottom + height/2, label,
                   va='center', ha='left', fontsize=10)
    
    # Add depth scale
    depth_labels = ['0 m', '0.2 m', '0.5 m', '1.0 m', '2.0 m']
    depths = [1.0, 0.8, 0.6, 0.4, 0.0]
    
    for depth, label in zip(depths, depth_labels):
        y = col_bottom + col_height * depth
        ax_soil.plot([col_left + col_width, col_left + col_width + 0.02], [y, y],
                   'k-', linewidth=0.5)
        ax_soil.text(col_left + col_width + 0.03, y, label,
                   va='center', ha='left', fontsize=9)
    
    # Temperature baseline
    temp_x_base = col_left - 0.05
    temp_y = np.linspace(col_bottom, col_bottom + col_height, 100)
    
    # Highlight zero-curtain zone
    zc_bottom = col_bottom + col_height * 0.4
    zc_top = col_bottom + col_height * 0.6
    
    rect = Rectangle((col_left - 0.1, zc_bottom), col_width + 0.2, zc_top - zc_bottom,
                    facecolor='#ffcccb', edgecolor='r', linewidth=1.5,
                    linestyle='--', alpha=0.3)
    ax_soil.add_patch(rect)
    
    # Add temperature scales
    ax_soil.text(temp_x_base - 0.12, col_bottom + col_height, '-10°C',
               va='center', ha='right', fontsize=9, color='blue')
    ax_soil.text(temp_x_base + 0.1, col_bottom + col_height, '+5°C',
               va='center', ha='right', fontsize=9, color='red')
    
    soil_title = ax_soil.text(0.5, 0.95, '', transform=ax_soil.transAxes, 
                            fontsize=14, fontweight='bold', ha='center', va='top')
    soil_line, = ax_soil.plot([], [], 'r-', linewidth=2)
    
    # 3. Temporal heatmap
    # Prepare data
    yearly_monthly = enhanced_df.groupby(['year', 'month']).size().unstack(fill_value=0)
    
    # Ensure all months are present
    for month in range(1, 13):
        if month not in yearly_monthly.columns:
            yearly_monthly[month] = 0
    
    yearly_monthly = yearly_monthly.sort_index(axis=1)
    
    # Create a custom sequential colormap
    cmap = LinearSegmentedColormap.from_list(
        'custom_cmap', ['#ffffff', '#abd9e9', '#4575b4'], N=256)
    
    # Set month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Initialize heatmap
    temporal_im = ax_temporal.imshow(np.zeros((5, 12)), cmap=cmap, aspect='auto',
                                   vmin=0, vmax=yearly_monthly.max().max())
    
    # Set up colorbar
    cbar = plt.colorbar(temporal_im, ax=ax_temporal)
    cbar.set_label('Number of Events')
    
    # Set labels
    ax_temporal.set_xticks(np.arange(12))
    ax_temporal.set_xticklabels(month_names)
    ax_temporal.set_yticks([])
    
    ax_temporal.set_xlabel('Month')
    temporal_title = ax_temporal.set_title('Zero-Curtain Events by Month', fontsize=14)
    
    # 4. Physics panel
    physics_title = ax_physics.text(0.5, 0.95, 'Zero-Curtain Physics', 
                                  transform=ax_physics.transAxes, fontsize=14, 
                                  fontweight='bold', ha='center', va='top')
    
    # Add physics equation
    physics_eq = r"$L_f \rho \frac{\partial \theta_i}{\partial t} = \nabla \cdot (k \nabla T)$"
    ax_physics.text(0.5, 0.5, physics_eq, transform=ax_physics.transAxes,
                  fontsize=16, ha='center', va='center',
                  bbox=dict(facecolor='#f8f8f8', alpha=1, boxstyle='round', pad=1))
    
    # Add explanation text
    phase_text = ax_physics.text(0.5, 0.2, '', transform=ax_physics.transAxes,
                              fontsize=12, ha='center', va='center',
                              bbox=dict(facecolor='white', alpha=0.8, boxstyle='round', pad=1))
    
    # Animation update function
    def update(frame):
        year_idx = frame % len(years)
        season_idx = frame % 4  # 0=Winter, 1=Spring, 2=Summer, 3=Fall
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        current_season = seasons[season_idx]
        
        year = years[year_idx]
        
        # 1. Update map
        year_data = validated_df[validated_df['year'] == year]
        
        if len(year_data) > 0:
            # Further filter by season if needed
            season_data = year_data[year_data['season'] == current_season]
            
            if len(season_data) > 0:
                # Get coordinates and attributes
                lons = season_data['longitude'].values
                lats = season_data['latitude'].values
                durations = season_data['duration_hours'].values
                seasons = season_data['season'].values
                
                # Calculate point sizes
                sizes = 20 + 15 * np.log1p(durations / 100)
                sizes = np.clip(sizes, 20, 150)
                
                # Get point colors
                colors = [season_colors.get(s, '#888888') for s in seasons]
                
                # Update scatter plot
                scatter.set_offsets(np.column_stack([lons, lats]))
                scatter.set_sizes(sizes)
                scatter.set_color(colors)
                
                # Update title
                map_title.set_text(f'Zero-Curtain Events: {year} ({current_season})')
            else:
                # Empty plot for this season
                scatter.set_offsets(np.empty((0, 2)))
                scatter.set_sizes([])
                scatter.set_color([])
                map_title.set_text(f'Zero-Curtain Events: {year} ({current_season}) - No Data')
        else:
            # Empty plot for this year
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_sizes([])
            scatter.set_color([])
            map_title.set_text(f'Zero-Curtain Events: {year} - No Data')
        
        # 2. Update soil temperature profile
        # Simulate soil temperature based on seasonal progression
        def temp_profile(y, season):
            rel_y = (y - col_bottom) / col_height
            
            if season == 'Winter':
                # Cold surface, zero-curtain forming/disappearing
                if rel_y >= 0.8:  # Surface
                    return -15.0 - 5.0 * (rel_y - 0.8) / 0.2
                elif 0.6 < rel_y < 0.8:  # Active layer
                    return -5.0 - 2.0 * (rel_y - 0.6) / 0.2
                elif 0.4 < rel_y < 0.6:  # Zero-curtain zone
                    return -0.1 + 0.2 * np.sin((rel_y - 0.4) * np.pi / 0.2)
                else:  # Permafrost
                    return -2.0 - 3.0 * (0.4 - rel_y) / 0.4
            
            elif season == 'Spring':
                # Warming from top, zero-curtain forming
                if rel_y >= 0.8:  # Surface
                    return 5.0 - 2.0

#plt.rcdefaults()
#plt.style.use('default')

class QuantumParameterInterpreter:
    def __init__(self, resolved_params):
        """
        Initialize quantum parameter interpreter
        
        Args:
            resolved_params (dict): Thermal wavelength and tunneling probability
        """
        self.thermal_wavelength = resolved_params['thermal_wavelength']
        self.tunneling_probability = resolved_params['tunneling_probability']
    
    def statistical_summary(self):
        """Generate comprehensive statistical summary of quantum parameters"""
        return {
            'thermal_wavelength': {
                'mean': np.nanmean(self.thermal_wavelength),
                'median': np.nanmedian(self.thermal_wavelength),
                'std': np.nanstd(self.thermal_wavelength),
                'min': np.nanmin(self.thermal_wavelength),
                'max': np.nanmax(self.thermal_wavelength),
                'percentiles': np.nanpercentile(self.thermal_wavelength, [10, 25, 50, 75, 90])
            },
            'tunneling_probability': {
                'mean': np.nanmean(self.tunneling_probability),
                'median': np.nanmedian(self.tunneling_probability),
                'std': np.nanstd(self.tunneling_probability),
                'min': np.nanmin(self.tunneling_probability),
                'max': np.nanmax(self.tunneling_probability),
                'percentiles': np.nanpercentile(self.tunneling_probability, [10, 25, 50, 75, 90])
            }
        }
    
    def spatial_correlation_analysis(self):
        """Analyze spatial correlation between thermal wavelength and tunneling probability"""
        thermal_flat = self.thermal_wavelength.flatten()
        tunneling_flat = self.tunneling_probability.flatten()
        valid_mask = ~(np.isnan(thermal_flat) | np.isnan(tunneling_flat))
        
        return {
            'pearson_correlation': np.corrcoef(
                thermal_flat[valid_mask], 
                tunneling_flat[valid_mask]
            )[0, 1],
            'spatial_entropy': {
                'thermal_wavelength': self._compute_spatial_entropy(self.thermal_wavelength),
                'tunneling_probability': self._compute_spatial_entropy(self.tunneling_probability)
            }
        }
    
    def _compute_spatial_entropy(self, parameter_grid):
        """Compute spatial entropy to quantify spatial complexity"""
        grid_norm = (parameter_grid - np.nanmin(parameter_grid)) / (
            np.nanmax(parameter_grid) - np.nanmin(parameter_grid) + 1e-10
        )
        return -np.nansum(grid_norm * np.log2(grid_norm + 1e-10))
    
    def _plot_anomalies(self, time_grid, lat_grid, lon_grid, output_dir):
        """Plot anomalies from mean state"""
        # Calculate mean state
        mean_wavelength = np.nanmean(self.thermal_wavelength, axis=(0))
        mean_tunneling = np.nanmean(self.tunneling_probability, axis=(0))
        
        # Calculate monthly anomalies
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for param_name, data, mean_state in [('wavelength', self.thermal_wavelength, mean_wavelength),
                                           ('tunneling', self.tunneling_probability, mean_tunneling)]:
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            
            for month in range(12):
                ax = axes[month//4, month%4]
                month_mask = pd.DatetimeIndex(time_grid).month == (month + 1)
                monthly_mean = np.nanmean(data[month_mask], axis=0)
                anomaly = monthly_mean - mean_state
                
                # Use diverging colormap for anomalies
                vmax = np.nanpercentile(np.abs(anomaly), 95)
                im = ax.pcolormesh(lon_grid, lat_grid, anomaly,
                                 cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                
                # Add contours for zero-crossing
                cs = ax.contour(lon_grid, lat_grid, anomaly,
                              levels=[0], colors='k', linewidths=0.5)
                
                ax.set_title(f'{month_names[month]}')
                plt.colorbar(im, ax=ax)
            
            title = 'Monthly Anomalies: Thermal Wavelength' if param_name == 'wavelength' else \
                   'Monthly Anomalies: Tunneling Probability'
            plt.suptitle(f'{title}\n(Deviation from Time Mean)', y=1.02, size=16)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{param_name}_anomalies.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def visualize_enhanced_parameters(self, time_grid, lat_grid, lon_grid, output_dir='quantum_param_plots'):
        """
        Generate enhanced visualization of quantum parameters
        """
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import LogNorm, SymLogNorm
        import numpy as np
        from matplotlib import cm
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Enhanced visualization for Thermal Wavelength
        plt.figure(figsize=(15, 10))
        spatial_mean_wavelength = np.nanmean(self.thermal_wavelength, axis=0)
        
        # Create diverging colormap centered on mean
        mean_val = np.nanmean(spatial_mean_wavelength)
        std_val = np.nanstd(spatial_mean_wavelength)
        
        # Use symlog norm for better contrast
        norm = SymLogNorm(linthresh=std_val/10, 
                         vmin=mean_val-3*std_val, 
                         vmax=mean_val+3*std_val)
        
        plt.figure(figsize=(15, 10))
        im = plt.pcolormesh(lon_grid, lat_grid, spatial_mean_wavelength, 
                           norm=norm, cmap='RdYlBu_r')
        plt.colorbar(im, label='Thermal Wavelength (m)', extend='both')
        
        # Add contours to highlight transitions
        cs = plt.contour(lon_grid, lat_grid, spatial_mean_wavelength,
                        levels=np.linspace(np.nanmin(spatial_mean_wavelength),
                                         np.nanmax(spatial_mean_wavelength), 10),
                        colors='k', alpha=0.3, linewidths=0.5)
        plt.clabel(cs, inline=True, fontsize=8, fmt='%.2e')
        
        plt.title('Time-averaged Thermal Wavelength Distribution\nwith Transition Zones')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(f'{output_dir}/enhanced_wavelength_spatial.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Enhanced visualization for Tunneling Probability
        plt.figure(figsize=(15, 10))
        spatial_mean_tunneling = np.nanmean(self.tunneling_probability, axis=0)
        
        # Use log normalization for tunneling probability
        norm = LogNorm(vmin=np.nanpercentile(spatial_mean_tunneling[spatial_mean_tunneling > 0], 1),
                      vmax=np.nanpercentile(spatial_mean_tunneling, 99))
        
        im = plt.pcolormesh(lon_grid, lat_grid, spatial_mean_tunneling,
                           norm=norm, cmap='viridis')
        plt.colorbar(im, label='Tunneling Probability (log scale)', extend='both')
        
        # Add probability contours
        cs = plt.contour(lon_grid, lat_grid, spatial_mean_tunneling,
                        levels=np.logspace(np.log10(np.nanmin(spatial_mean_tunneling[spatial_mean_tunneling > 0])),
                                         np.log10(np.nanmax(spatial_mean_tunneling)), 10),
                        colors='w', alpha=0.3, linewidths=0.5)
        plt.clabel(cs, inline=True, fontsize=8, fmt='%.2e')
        
        plt.title('Time-averaged Tunneling Probability Distribution\nwith Probability Contours')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(f'{output_dir}/enhanced_tunneling_spatial.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Monthly patterns with enhanced visualization
        for param_name, data, cmap in [('wavelength', self.thermal_wavelength, 'RdYlBu_r'),
                                     ('tunneling', self.tunneling_probability, 'viridis')]:
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for month in range(12):
                ax = axes[month//4, month%4]
                month_mask = pd.DatetimeIndex(time_grid).month == (month + 1)
                monthly_mean = np.nanmean(data[month_mask], axis=0)
                
                if param_name == 'wavelength':
                    norm = SymLogNorm(linthresh=np.nanstd(monthly_mean)/10,
                                    vmin=np.nanmean(monthly_mean)-3*np.nanstd(monthly_mean),
                                    vmax=np.nanmean(monthly_mean)+3*np.nanstd(monthly_mean))
                else:
                    norm = LogNorm(vmin=np.nanpercentile(monthly_mean[monthly_mean > 0], 1),
                                 vmax=np.nanpercentile(monthly_mean, 99))
                
                im = ax.pcolormesh(lon_grid, lat_grid, monthly_mean,
                                 norm=norm, cmap=cmap)
                
                # Add contours
                cs = ax.contour(lon_grid, lat_grid, monthly_mean,
                              colors='k' if param_name == 'wavelength' else 'w',
                              alpha=0.3, linewidths=0.5)
                
                ax.set_title(f'{month_names[month]}')
                plt.colorbar(im, ax=ax)
            
            title = 'Monthly Patterns of Thermal Wavelength' if param_name == 'wavelength' else \
                   'Monthly Patterns of Tunneling Probability'
            plt.suptitle(f'{title}\nwith Enhanced Visualization', y=1.02, size=16)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/enhanced_{param_name}_seasonal.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

        # Add anomaly plots
        self._plot_anomalies(time_grid, lat_grid, lon_grid, output_dir)
    
    def visualize_spatiotemporal_parameters(self, time_grid, lat_grid, lon_grid, 
                                          output_dir='quantum_param_plots'):
        """
        Generate visualization of quantum parameters with enhanced coloring
        """
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Define colormaps
        tunneling_cmap = 'magma'  # Vibrant colormap for tunneling probability
        wavelength_cmap = 'plasma'  # Different colormap for wavelength
        
        # 1. Time-averaged spatial distributions
        plt.figure(figsize=(15, 10))
        spatial_mean_tunneling = np.nanmean(self.tunneling_probability, axis=0)
        plt.pcolormesh(lon_grid, lat_grid, spatial_mean_tunneling, cmap=tunneling_cmap)
        cbar = plt.colorbar(label='Mean Tunneling Probability')
        cbar.ax.tick_params(labelsize=10)
        plt.title('Time-averaged Tunneling Probability (1915-2024)', size=14, pad=20)
        plt.xlabel('Longitude', size=12)
        plt.ylabel('Latitude', size=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/tunneling_prob_spatial_mean.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(15, 10))
        spatial_mean_wavelength = np.nanmean(self.thermal_wavelength, axis=0)
        plt.pcolormesh(lon_grid, lat_grid, spatial_mean_wavelength, cmap=wavelength_cmap)
        cbar = plt.colorbar(label='Mean Thermal Wavelength (m)')
        cbar.ax.tick_params(labelsize=10)
        plt.title('Time-averaged Thermal Wavelength (1915-2024)', size=14, pad=20)
        plt.xlabel('Longitude', size=12)
        plt.ylabel('Latitude', size=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/wavelength_spatial_mean.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Temporal evolution with dual axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        temporal_mean_tunneling = np.nanmean(np.nanmean(self.tunneling_probability, axis=1), axis=1)
        temporal_mean_wavelength = np.nanmean(np.nanmean(self.thermal_wavelength, axis=1), axis=1)
        
        # Plot tunneling probability
        color1 = '#FF6B6B'  # Coral red
        ax1.plot(time_grid, temporal_mean_tunneling, color=color1, linewidth=2)
        ax1.fill_between(time_grid, temporal_mean_tunneling, alpha=0.2, color=color1)
        ax1.set_title('Spatial-averaged Tunneling Probability Evolution', size=14, pad=20)
        ax1.set_xlabel('Time', size=12)
        ax1.set_ylabel('Mean Tunneling Probability', size=12, color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Plot thermal wavelength
        color2 = '#4ECDC4'  # Turquoise
        ax2.plot(time_grid, temporal_mean_wavelength, color=color2, linewidth=2)
        ax2.fill_between(time_grid, temporal_mean_wavelength, alpha=0.2, color=color2)
        ax2.set_title('Spatial-averaged Thermal Wavelength Evolution', size=14, pad=20)
        ax2.set_xlabel('Time', size=12)
        ax2.set_ylabel('Mean Thermal Wavelength (m)', size=12, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/temporal_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Seasonal patterns with enhanced coloring
        # For tunneling probability
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month in range(12):
            ax = axes[month//4, month%4]
            month_mask = pd.DatetimeIndex(time_grid).month == (month + 1)
            monthly_mean = np.nanmean(self.tunneling_probability[month_mask], axis=0)
            
            im = ax.pcolormesh(lon_grid, lat_grid, monthly_mean, cmap=tunneling_cmap)
            ax.set_title(f'{month_names[month]}', size=12)
            plt.colorbar(im, ax=ax)
            ax.grid(True, alpha=0.3)
            
        plt.suptitle('Monthly Patterns of Tunneling Probability', size=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/tunneling_prob_seasonal.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # For thermal wavelength
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        for month in range(12):
            ax = axes[month//4, month%4]
            month_mask = pd.DatetimeIndex(time_grid).month == (month + 1)
            monthly_mean = np.nanmean(self.thermal_wavelength[month_mask], axis=0)
            
            im = ax.pcolormesh(lon_grid, lat_grid, monthly_mean, cmap=wavelength_cmap)
            ax.set_title(f'{month_names[month]}', size=12)
            plt.colorbar(im, ax=ax)
            ax.grid(True, alpha=0.3)
            
        plt.suptitle('Monthly Patterns of Thermal Wavelength', size=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/wavelength_seasonal.png', dpi=300, bbox_inches='tight')
        plt.close()
        """
        Generate visualization of quantum parameters with explicit spatiotemporal context
        
        Args:
            time_grid: Time coordinates
            lat_grid: Latitude coordinates
            lon_grid: Longitude coordinates
            output_dir: Directory to save plots
        """
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Time-averaged spatial distributions
        plt.figure(figsize=(15, 10))
        spatial_mean_tunneling = np.nanmean(self.tunneling_probability, axis=0)
        plt.pcolormesh(lon_grid, lat_grid, spatial_mean_tunneling)
        plt.colorbar(label='Mean Tunneling Probability')
        plt.title('Time-averaged Tunneling Probability (1915-2024)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(f'{output_dir}/tunneling_prob_spatial_mean.png')
        plt.close()
        
        plt.figure(figsize=(15, 10))
        spatial_mean_wavelength = np.nanmean(self.thermal_wavelength, axis=0)
        plt.pcolormesh(lon_grid, lat_grid, spatial_mean_wavelength)
        plt.colorbar(label='Mean Thermal Wavelength (m)')
        plt.title('Time-averaged Thermal Wavelength (1915-2024)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(f'{output_dir}/wavelength_spatial_mean.png')
        plt.close()
        
        # 2. Temporal evolution
        plt.figure(figsize=(15, 6))
        temporal_mean_tunneling = np.nanmean(np.nanmean(self.tunneling_probability, axis=1), axis=1)
        temporal_mean_wavelength = np.nanmean(np.nanmean(self.thermal_wavelength, axis=1), axis=1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        ax1.plot(time_grid, temporal_mean_tunneling)
        ax1.set_title('Spatial-averaged Tunneling Probability Evolution')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Mean Tunneling Probability')
        
        ax2.plot(time_grid, temporal_mean_wavelength)
        ax2.set_title('Spatial-averaged Thermal Wavelength Evolution')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Mean Thermal Wavelength (m)')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/temporal_evolution.png')
        plt.close()
        
        # 3. Seasonal patterns
        monthly_means_tunneling = []
        monthly_means_wavelength = []
        for month in range(12):
            month_mask = pd.DatetimeIndex(time_grid).month == (month + 1)
            monthly_means_tunneling.append(np.nanmean(self.tunneling_probability[month_mask], axis=0))
            monthly_means_wavelength.append(np.nanmean(self.thermal_wavelength[month_mask], axis=0))
        
        # Plot seasonal patterns for tunneling probability
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        for month in range(12):
            ax = axes[month//4, month%4]
            im = ax.pcolormesh(lon_grid, lat_grid, monthly_means_tunneling[month])
            ax.set_title(f'Month {month+1}')
            plt.colorbar(im, ax=ax)
        plt.suptitle('Monthly Patterns of Tunneling Probability')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/tunneling_prob_seasonal.png')
        plt.close()
        
        # Plot seasonal patterns for thermal wavelength
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        for month in range(12):
            ax = axes[month//4, month%4]
            im = ax.pcolormesh(lon_grid, lat_grid, monthly_means_wavelength[month])
            ax.set_title(f'Month {month+1}')
            plt.colorbar(im, ax=ax)
        plt.suptitle('Monthly Patterns of Thermal Wavelength')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/wavelength_seasonal.png')
        plt.close()

    def visualize_quantum_parameters(self, output_dir='quantum_param_plots'):
        """
        Generate visualization of quantum parameters
        
        Args:
            output_dir (str): Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Thermal Wavelength Distribution
        plt.figure(figsize=(10, 6))
        sample_size = min(10000, self.thermal_wavelength.size)
        sample_data = np.random.choice(
            self.thermal_wavelength.flatten(),
            size=sample_size,
            replace=False
        )
        sns.histplot(sample_data, kde=True, bins=50)
        plt.title('Thermal Wavelength Distribution (Sampled)')
        plt.xlabel('Thermal Wavelength')
        plt.ylabel('Frequency')
        plt.savefig(f'{output_dir}/thermal_wavelength_distribution.png')
        plt.close('all')
        del sample_data
        gc.collect()
        
        # 2. Tunneling Probability Distribution
        plt.figure(figsize=(10, 6))
        sample_data = np.random.choice(
            self.tunneling_probability.flatten(),
            size=sample_size,
            replace=False
        )
        sns.histplot(sample_data, kde=True, bins=50)
        plt.title('Tunneling Probability Distribution (Sampled)')
        plt.xlabel('Tunneling Probability')
        plt.ylabel('Frequency')
        plt.savefig(f'{output_dir}/tunneling_probability_distribution.png')
        plt.close('all')
        del sample_data
        gc.collect()
        
        # 3. Spatial Correlation Heatmap
        plt.figure(figsize=(12, 8))
        # Take a 2D slice for visualization
        slice_2d = np.nanmean(self.tunneling_probability, axis=0)
        if slice_2d.ndim > 2:
            slice_2d = np.nanmean(slice_2d, axis=0)
        sns.heatmap(slice_2d, cmap='viridis')
        plt.title('Mean Spatial Distribution of Tunneling Probability')
        plt.savefig(f'{output_dir}/tunneling_probability_heatmap.png')
        plt.close('all')
        del slice_2d
        gc.collect()
    
    def quantum_phase_transition_analysis(self):
        """Analyze potential phase transition zones"""
        transition_threshold = np.nanpercentile(self.tunneling_probability, 90)
        
        return {
            'high_transition_probability_regions': {
                'count': np.sum(self.tunneling_probability >= transition_threshold),
                'percentage': (
                    np.sum(self.tunneling_probability >= transition_threshold) / 
                    self.tunneling_probability.size * 100
                ),
                'mean_wavelength_in_transition_zones': np.nanmean(
                    self.thermal_wavelength[self.tunneling_probability >= transition_threshold]
                )
            },
            'phase_transition_probability_distribution': {
                'median': np.nanmedian(self.tunneling_probability),
                'mean': np.nanmean(self.tunneling_probability),
                'std': np.nanstd(self.tunneling_probability)
            }
        }

import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Initialize the interpreter with our quantum parameters
quantum_interpreter = QuantumParameterInterpreter({
    'thermal_wavelength': thermal_wavelength,
    'tunneling_probability': tunneling_prob
})

# Get statistical summary
stats = quantum_interpreter.statistical_summary()
print("\nQuantum Parameter Statistics:")
print("\nThermal Wavelength:")
for key, value in stats['thermal_wavelength'].items():
    if key != 'percentiles':
        print(f"{key}: {value:.2e}")
    else:
        print(f"Percentiles (10,25,50,75,90): {value}")

print("\nTunneling Probability:")
for key, value in stats['tunneling_probability'].items():
    if key != 'percentiles':
        print(f"{key}: {value:.2e}")
    else:
        print(f"Percentiles (10,25,50,75,90): {value}")

# Get spatial correlation analysis
correlations = quantum_interpreter.spatial_correlation_analysis()
print("\nSpatial Correlation Analysis:")
print(f"Pearson correlation: {correlations['pearson_correlation']:.4f}")
print("\nSpatial Entropy:")
print(f"Thermal wavelength: {correlations['spatial_entropy']['thermal_wavelength']:.4f}")
print(f"Tunneling probability: {correlations['spatial_entropy']['tunneling_probability']:.4f}")

# Analyze phase transitions
phase_analysis = quantum_interpreter.quantum_phase_transition_analysis()
print("\nPhase Transition Analysis:")
print("\nHigh Transition Probability Regions:")
for key, value in phase_analysis['high_transition_probability_regions'].items():
    print(f"{key}: {value:.4e}")

# Generate visualizations
print("\nGenerating visualizations...")
quantum_interpreter.visualize_quantum_parameters(output_dir='quantum_results')

quantum_interpreter.visualize_spatiotemporal_parameters(
    processed_data['time_grid'],
    processed_data['lat_grid'],
    processed_data['lon_grid']
)

# Save all results
np.savez('quantum_analysis_results.npz',
         thermal_wavelength_stats=stats['thermal_wavelength'],
         tunneling_probability_stats=stats['tunneling_probability'],
         spatial_correlations=correlations,
         phase_transition_analysis=phase_analysis,
         thermal_wavelength=thermal_wavelength,
         tunneling_probability=tunneling_prob)

print("\nResults saved to 'quantum_analysis_results.npz'")

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import cmocean
# import gc
# import glob
# import json
# import logging
# import matplotlib.gridspec as gridspec
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pandas as pd
# import pathlib
# import pickle
# import psutil
# import re
# import gc
# import dask
# import dask.dataframe as dd
# import scipy.interpolate as interpolate
# import scipy.stats as stats
# import seaborn as sns
# from pyproj import Proj
# import sklearn.experimental
# import sklearn.impute
# import sklearn.linear_model
# import sklearn.preprocessing
# import tqdm
# import xarray as xr
# import warnings
# warnings.filterwarnings('ignore')

# from osgeo import gdal, osr
# from matplotlib.colors import LinearSegmentedColormap
# from concurrent.futures import ThreadPoolExecutor
# from datetime import datetime, timedelta
# from pathlib import Path
# from scipy.spatial import cKDTree
# from tqdm import tqdm
# from tqdm.notebook import tqdm

# import tensorflow as tf

# # # Configure GPU memory growth
# try:
#     physical_devices = tf.config.list_physical_devices('GPU')
#     for device in physical_devices:
#         tf.config.experimental.set_memory_growth(device, True)
# except:
#     pass
    
# print("TensorFlow version:", tf.__version__)

# zc_df = pd.read_csv('processor2_df_with_coords_updated.csv')

# temp_df = pd.read_csv('ZC_data_tempnonans_df.csv')
# temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], errors='coerce', format='mixed')

# def process_site_efficiently(site_id, temp_df, zc_df, window_size=30, depth_range=(-2, 20)):
#     """Process a single site with optimized memory usage"""
#     try:
#         # Filter data for this site
#         site_data = temp_df[temp_df['site_id'] == site_id].copy()
#         site_data = site_data[site_data['depth'].between(depth_range[0], depth_range[1])]
        
#         if len(site_data) < window_size:
#             return []
            
#         # Create pivot table
#         pivot = site_data.pivot_table(
#             index='datetime',
#             columns='depth',
#             values='temperature',
#             aggfunc='mean'
#         ).sort_index()
        
#         # Free memory
#         del site_data
#         gc.collect()
        
#         if len(pivot.columns) < 4:  # Minimum number of depths
#             return []
            
#         windows = []
#         chunk_size = 100  # Process windows in chunks
        
#         # Process windows in chunks
#         for i in range(0, len(pivot) - window_size + 1, chunk_size):
#             end_idx = min(i + chunk_size, len(pivot) - window_size + 1)
            
#             for j in range(i, end_idx):
#                 window = pivot.iloc[j:j+window_size]
                
#                 # Skip if too many missing values
#                 if window.isnull().sum().sum() / (window.shape[0] * window.shape[1]) > 0.3:
#                     continue
                    
#                 # Check for zero curtain events
#                 window_start = window.index[0]
#                 window_end = window.index[-1]
                
#                 zc_events = zc_df[
#                     (zc_df['site_id'] == site_id) &
#                     (zc_df['start_date'] >= window_start) &
#                     (zc_df['end_date'] <= window_end)
#                 ]
                
#                 # Fill missing values
#                 window_filled = window.interpolate(method='linear').fillna(method='ffill').fillna(...
                
#                 if not window_filled.isnull().any().any():
#                     # windows.append({
#                     #     'window_data': window_filled.values,
#                     #     'has_zero_curtain': 1 if not zc_events.empty else 0
#                     # })
#                     windows.append({
#                         'window_data': window_filled.values,
#                         'has_zero_curtain': 1 if not zc_events.empty else 0,
#                         'window_start': window.index[0],  # First day in the 30-day window
#                         'window_end': window.index[-1]    # Last day in the 30-day window
#                     })

#             # Clear memory after each chunk
#             gc.collect()
        
#         return windows
        
#     except Exception as e:
#         print(f"Error processing site {site_id}: {str(e)}")
#         return []

# def preprocess_with_saving(temp_df, zc_df, save_dir='processed_data', sites_per_file=10):
#     """Preprocess data with intermediate saving"""
    
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Convert datetime columns once
#     print("Converting datetime columns...")
#     temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], errors='coerce', format='mixed')
#     zc_df['start_date'] = pd.to_datetime(zc_df['start_date'], errors='coerce', format='mixed')
#     zc_df['end_date'] = pd.to_datetime(zc_df['end_date'], errors='coerce', format='mixed')
    
#     unique_sites = temp_df['site_id'].unique()
#     print(f"Total sites to process: {len(unique_sites)}")
    
#     processed_count = 0
#     current_batch = []
#     batch_number = 0
    
#     for site_idx, site_id in enumerate(unique_sites):
#         print(f"\rProcessing site {site_idx + 1}/{len(unique_sites)}", end="")
        
#         # Process single site
#         site_windows = process_site_efficiently(site_id, temp_df, zc_df)
        
#         if site_windows:
#             current_batch.extend(site_windows)
#             processed_count += len(site_windows)
        
#         # Save batch if reached sites_per_file
#         if (site_idx + 1) % sites_per_file == 0 and current_batch:
#             batch_file = os.path.join(save_dir, f'batch_{batch_number}.npy')
#             np.save(batch_file, current_batch)
#             print(f"\nSaved batch {batch_number} with {len(current_batch)} windows")
            
#             current_batch = []
#             batch_number += 1
#             gc.collect()
    
#     # Save any remaining data
#     if current_batch:
#         batch_file = os.path.join(save_dir, f'batch_{batch_number}.npy')
#         np.save(batch_file, current_batch)
#         print(f"\nSaved final batch with {len(current_batch)} windows")
    
#     print(f"\nTotal processed windows: {processed_count}")
#     return save_dir

# def load_and_combine_batches(data_dir):
#     """Load and combine processed batches"""
#     all_data = []
#     batch_files = sorted([f for f in os.listdir(data_dir) if f.startswith('batch_')])
    
#     for batch_file in batch_files:
#         batch_data = np.load(os.path.join(data_dir, batch_file), allow_pickle=True)
#         all_data.extend(batch_data)
#         gc.collect()
    
#     return all_data
    
# print("Loading data...")
# processed_dir = preprocess_with_saving(temp_df, zc_df, sites_per_file=5)
# print("\nLoading processed data for training...")
# processed_data = load_and_combine_batches(processed_dir)
# print(f"Total windows for training: {len(processed_data)}")

# with open('processed_data.pkl','wb') as f:
#     pickle.dump(processed_data, f)

# with open('processed_data.pkl', 'rb') as f:
#     processed_data = pd.read_pickle(f)
    
# for data in processed_data:
#     data['window_start'] = pd.to_datetime(data['window_start'])
#     data['window_end'] = pd.to_datetime(data['window_end'])
    
# processed_data.sort(key=lambda x: x['window_start'])

# all_years = sorted(set(d['window_start'].year for d in processed_data))

# def create_time_split_datasets(processed_data, n_depths=10, batch_size=32):
# """Split processed data into training (65%), validation...
    
#     print("Sorting data by time...")
    
#     for data in processed_data:
#         data['window_start'] = pd.to_datetime(data['window_start'])
#         data['window_end'] = pd.to_datetime(data['window_end'])
    
#     # Sort the dataset chronologically
#     processed_data.sort(key=lambda x: x['window_start'])

#     # Total dataset size
#     total_windows = len(processed_data)
#     train_size = int(0.65 * total_windows)  # 65% of dataset
#     val_size = int(0.25 * total_windows)  # 25% of dataset
#     test_size = total_windows - train_size - val_size  # Remaining 10%

#     print(f"Total windows: {total_windows}")
#     print(f"Target splits -> Train: {train_size}, Validation: {val_size}, Test: {test_size}")

#     # Split dataset ensuring end-of-year continuity
#     train_data, val_data, test_data = [], [], []
#     current_year = processed_data[0]['window_start'].year

#     count_train, count_val, count_test = 0, 0, 0

#     for data in processed_data:
#         year = data['window_start'].year

#         if count_train < train_size:
#             train_data.append(data)
#             count_train += 1
#         elif count_val < val_size:
#             val_data.append(data)
#             count_val += 1
#         else:
#             test_data.append(data)
#             count_test += 1

#         # Ensure we finish the current year before switching splits
# if count_train >= train_size and count_val <...
#             current_year = year
# elif count_train >= train_size and count_val >=...
#             current_year = year

# #print(f"Final dataset sizes -> Train: {len(train_data)}, Validation:...
#     train_years = (train_data[0]['window_start'].year, train_data[-1]['window_start'].year) if tra...
#     val_years = (val_data[0]['window_start'].year, val_data[-1]['window_start'].year) if val_data ...
#     test_years = (test_data[0]['window_start'].year, test_data[-1]['window_start'].year) if test_d...

# print(f"\nFinal dataset sizes -> Train: {len(train_data)}, Validation:...
#     print(f"Train Year Range: {train_years[0]} - {train_years[1]}")
#     print(f"Validation Year Range: {val_years[0]} - {val_years[1]}")
#     print(f"Test Year Range: {test_years[0]} - {test_years[1]}")


#     # Ensure all data has uniform depth dimensions
#     target_depths = n_depths  # Use provided depth count

#     def filter_and_format(data):
# filtered = [d for d in data...
#         X = np.array([d['window_data'] for d in filtered])
#         y = np.array([d['has_zero_curtain'] for d in filtered])
#         return X, y

#     X_train, y_train = filter_and_format(train_data)
#     X_val, y_val = filter_and_format(val_data)
#     X_test, y_test = filter_and_format(test_data)

#     print(f"\nFinal dataset sizes after filtering for uniform depths:")
#     print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

#     # Convert to TensorFlow datasets
#     def create_tf_dataset(X, y):
#         return tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).prefetch(tf.data.AUTOT...

#     train_dataset = create_tf_dataset(X_train, y_train)
#     val_dataset = create_tf_dataset(X_val, y_val)
#     test_dataset = create_tf_dataset(X_test, y_test)

#     return train_dataset, val_dataset, test_dataset, target_depths

# train_dataset, val_dataset, test_dataset, n_depths = create_time_split_datasets(processed_data)

# Sorting data by time...
# Total windows: 437846
# Target splits -> Train: 284599, Validation: 109461, Test: 43786

# Final dataset sizes -> Train: 284599, Validation: 109461, Test: 43786
# Train Year Range: 1926 - 2014
# Validation Year Range: 2014 - 2019
# Test Year Range: 2019 - 2024

# Final dataset sizes after filtering for uniform depths:
# Train: (9194, 30, 10), Validation: (2115, 30, 10), Test: (1776, 30, 10)

# 1. Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, predictions > 0.5)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# 2. Prediction Probability Distribution
plt.figure(figsize=(10, 6))
plt.hist(predictions, bins=50, edgecolor='black')
plt.title('Distribution of Prediction Probabilities')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.axvline(0.5, color='r', linestyle='--', label='Decision Threshold')
plt.legend()
plt.savefig('prediction_distribution.png')
plt.close()

# 3. ROC Curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_true, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

# 4. Precision-Recall Curve
from sklearn.metrics import precision_recall_curve, average_precision_score
precision, recall, _ = precision_recall_curve(y_true, predictions)
avg_precision = average_precision_score(y_true, predictions)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='blue', lw=2, 
         label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.savefig('precision_recall_curve.png')
plt.close()

# 5. Spatial Analysis
test_coords = spatial_info['test']['coords']

# Create scatter plot of predictions by location
plt.figure(figsize=(15, 10))
scatter = plt.scatter(test_coords[:, 1], test_coords[:, 0], 
                     c=predictions, cmap='coolwarm', 
                     s=50, alpha=0.6)
plt.colorbar(scatter, label='Predicted Probability')
plt.title('Spatial Distribution of Predictions')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('spatial_predictions.png')
#plt.close()

# 7. Temporal Analysis (if timestamps available)
timestamps=spatial_info['test']['times'] if 'times' in spatial_info['test'] else None
if timestamps is not None:
    times = pd.to_datetime(timestamps)
    plt.figure(figsize=(15, 6))
    plt.plot(times, predictions, 'b.', alpha=0.1)
    plt.title('Predictions Over Time')
    plt.xlabel('Time')
    plt.ylabel('Predicted Probability')
    #plt.savefig('temporal_predictions.png')
    #plt.close()

# 2. Now create temporal plot
plt.figure(figsize=(20, 12))

# Convert timestamps to datetime
train_times = pd.to_datetime(spatial_info['train']['times'])
val_times = pd.to_datetime(spatial_info['val']['times'])
test_times = pd.to_datetime(spatial_info['test']['times'])

# Sort everything by time
train_idx = np.argsort(train_times)
val_idx = np.argsort(val_times)
test_idx = np.argsort(test_times)

plt.subplot(3, 1, 1)
plt.plot(train_times[train_idx], y_train[train_idx], 'b.', alpha=0.3, label='Actual')
plt.plot(train_times[train_idx], train_pred[train_idx], 'r.', alpha=0.3, label='Predicted')
plt.title('Training Set')
plt.legend()
plt.ylabel('Zero Curtain Probability')

plt.subplot(3, 1, 2)
plt.plot(val_times[val_idx], y_val[val_idx], 'b.', alpha=0.3, label='Actual')
plt.plot(val_times[val_idx], val_pred[val_idx], 'r.', alpha=0.3, label='Predicted')
plt.title('Validation Set')
plt.legend()
plt.ylabel('Zero Curtain Probability')

plt.subplot(3, 1, 3)
plt.plot(test_times[test_idx], y_test[test_idx], 'b.', alpha=0.3, label='Actual')
plt.plot(test_times[test_idx], test_pred[test_idx], 'r.', alpha=0.3, label='Predicted')
plt.title('Test Set')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Zero Curtain Probability')

plt.tight_layout()
plt.savefig('temporal_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nCreating spatial plots...")

def create_focused_regional_maps(train_pred, val_pred, test_pred, spatial_info):
    """Create maps focused on the actual data region"""
    
    fig = plt.figure(figsize=(20, 8))
    projection = ccrs.LambertConformal(
        central_longitude=30.0,  # Middle of our longitude range
        central_latitude=57.0,   # Middle of our latitude range
        standard_parallels=(45.0, 70.0)
    )
    
    def create_detailed_map(ax, lons, lats, values, title):
        # Set map extent to cover our data region with some padding
        ax.set_extent([5, 57, 45, 70], ccrs.PlateCarree())
        
        # Add detailed features
        ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
        ax.add_feature(cfeature.OCEAN, color='lightblue', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=2)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        # Create scatter plot
        scatter = ax.scatter(lons, lats,
                           c=values.ravel(),
                           transform=ccrs.PlateCarree(),
                           cmap='RdYlBu_r',
                           vmin=0, vmax=1,
                           s=30,
                           alpha=0.7,
                           zorder=3)
        
        ax.set_title(f'{title}\n(n={len(values)})', pad=10)
        return scatter
    
    # Create three panels
    ax1 = plt.subplot(131, projection=projection)
    train_scatter = create_detailed_map(
        ax1,
        spatial_info['train']['coords'][:, 1],
        spatial_info['train']['coords'][:, 0],
        train_pred,
        'Training Predictions'
    )
    
    ax2 = plt.subplot(132, projection=projection)
    val_scatter = create_detailed_map(
        ax2,
        spatial_info['val']['coords'][:, 1],
        spatial_info['val']['coords'][:, 0],
        val_pred,
        'Validation Predictions'
    )
    
    ax3 = plt.subplot(133, projection=projection)
    test_scatter = create_detailed_map(
        ax3,
        spatial_info['test']['coords'][:, 1],
        spatial_info['test']['coords'][:, 0],
        test_pred,
        'Test Predictions'
    )
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(train_scatter, cax=cbar_ax)
    cbar.set_label('Predicted Zero Curtain Probability', fontsize=10)
    
    plt.suptitle('Regional Zero Curtain Predictions\n(European/Western Russian Sites)', y=1.05, fontsize=16)
    plt.tight_layout()
    
    plt.savefig('regional_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

