"""
Processing module for Arctic zero-curtain pipeline.
Auto-generated from Jupyter notebook.
"""

from src.common.imports import *
from src.common.utilities import *

# Archived
import pandas as pd
import numpy as np

def standardize_thickness(df):
    df_copy = df.copy()
    df_copy['thickness'] = pd.to_numeric(df_copy['thickness'], errors='coerce')
    cm_mask = df_copy['thickness'].abs() >= 1.0
    df_copy.loc[cm_mask, 'thickness'] = df_copy.loc[cm_mask, 'thickness'] / 100.0
    df_copy.loc[df_copy['thickness'] > 0, 'thickness'] = -df_copy.loc[df_copy['thickness'] > 0, 'thickness']
    return df_copy

standardized_df = standardize_thickness(newdf3)
standardized_df = standardized_df.sort_values('datetime').reset_index(drop=True)
standardized_df.to_csv('standardizedalt_cleaned_all_insitu_df.csv', index=False)
standardized_df.sample(10000).sort_values('datetime').reset_index(drop=True).to_csv('standardized_df_sample_10000.csv', index=False)

def add_depth_zones(df):
    conditions = [
        (df['depth'] <= 0.25),
        (df['depth'] > 0.25) & (df['depth'] <= 0.5),
        (df['depth'] > 0.5) & (df['depth'] <= 1.0),
        (df['depth'] > 1.0)
    ]
    choices = ['shallow', 'intermediate', 'deep', 'very_deep']
    df['depth_zone'] = np.select(conditions, choices, default='unknown')
    return df

df_with_dates_zones = add_depth_zones(standardized_df)
df_with_dates_zones = df_with_dates_zones[df_with_dates_zones.latitude >= 30]
df_with_dates_zones.to_csv('new_df_dates_zones.csv', index=False)

print(df_with_dates_zones.columns.tolist())

required_columns = ['datetime', 'latitude', 'longitude', 'temperature', 'source', 'year', 'depth', 'season', 'depth_zone', 'site_id']

for col in required_columns:
    if col not in df_with_dates_zones.columns:
        if col == 'site_id':
            df_with_dates_zones['site_id'] = df_with_dates_zones['source']
        elif col == 'season':
            df_with_dates_zones['season'] = pd.to_datetime(df_with_dates_zones['datetime']).dt.quarter.map({
                1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'
            })

df_with_dates_zones.to_csv('ZC_df_data.csv', index=False)

data = pd.read_csv('ZC_df_data.csv')
data = data[~data.thickness.isna()]
data = data.sort_values('datetime').reset_index(drop=True)
alt = data
del data
alt.to_csv('ZC_data_thicknessnonans_df.csv', index=False)

data = pd.read_csv('ZC_df_data.csv')
data = data[~data.temperature.isna()]
data = data.sort_values('datetime').reset_index(drop=True)
temp = data
del data
temp.to_csv('ZC_data_tempnonans_df.csv', index=False)

data = pd.read_csv('ZC_data_tempnonans_df.csv')

# Derived ALT from GTNP ST
def identify_active_layer_from_temperature(borehole_df, output_file=None):
    grouped = borehole_df.groupby(['site_name', 'year', 'datetime'])
    alt_data = []
    for (site, year, date), group in grouped:
        if pd.isna(year):
            continue
        profile = group.sort_values('depth_m')
        if len(profile) < 2:
            continue
        if profile['depth_m'].min() > 10 or profile['depth_m'].max() > 50:
            continue
        positive_temps = profile[profile['temperature'] > 0]
        negative_temps = profile[profile['temperature'] <= 0]
        if len(positive_temps) > 0 and len(negative_temps) > 0:
            max_positive_depth = positive_temps['depth_m'].max()
            min_negative_depth = negative_temps['depth_m'].min()
            if min_negative_depth > max_positive_depth:
                positive_idx = positive_temps['depth_m'].idxmax()
                positive_depth = positive_temps.loc[positive_idx, 'depth_m']
                positive_temp = positive_temps.loc[positive_idx, 'temperature']
                negative_idx = negative_temps['depth_m'].idxmin()
                negative_depth = negative_temps.loc[negative_idx, 'depth_m']
                negative_temp = negative_temps.loc[negative_idx, 'temperature']
                if positive_temp != negative_temp:
                    zero_depth = positive_depth + (negative_depth - positive_depth) * (0 - positive_temp) / (negative_temp - positive_temp)
                    if 0 <= zero_depth <= 20:
                        alt_data.append({
                            'site_name': site,
                            'year': year,
                            'datetime': date,
                            'alt_m': zero_depth,
                            'derived_from': 'temperature_zero_crossing',
                            'min_depth_m': profile['depth_m'].min(),
                            'max_depth_m': profile['depth_m'].max(),
                            'pos_temp': positive_temp,
                            'neg_temp': negative_temp,
                            'latitude': profile['latitude'].iloc[0] if 'latitude' in profile.columns else None,
                            'longitude': profile['longitude'].iloc[0] if 'longitude' in profile.columns else None
                        })
    if alt_data:
        alt_df = pd.DataFrame(alt_data)
        max_alt_df = alt_df.loc[alt_df.groupby(['site_name', 'year'])['alt_m'].idxmax()]
        max_alt_df['processing_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        max_alt_df['processing_version'] = '1.0'
        max_alt_df['data_source'] = 'Derived_from_GTNP_Borehole'
        if output_file:
            max_alt_df.to_csv(output_file, index=False)
            print(f"Saved derived ALT data to {output_file}")
        return max_alt_df
    else:
        print("No active layer thickness could be derived from the temperature data")
        return pd.DataFrame()

alt_df = identify_active_layer_from_temperature(gtnp_st, 'derived_alt_from_cleaned_valid_gtnp_borehole_st_df.csv')
alt_df = alt_df.sort_values('datetime').reset_index(drop=True)
alt_df = alt_df[alt_df.latitude>=49].sort_values('datetime').reset_index(drop=True)

def standardize_active_layer_thickness(df, thickness_col='alt_m', output_col='thickness_m_standardized'):
    result_df = df.copy()
    if 'standardization_note' not in result_df.columns:
        result_df['standardization_note'] = ""
    result_df[output_col] = np.nan
    negative_mask = result_df[thickness_col] < 0
    result_df.loc[negative_mask, output_col] = -1 * result_df.loc[negative_mask, thickness_col]
    result_df.loc[negative_mask, 'standardization_note'] = "Negative depth converted to positive thickness (meters)"
    mm_mask = (result_df[thickness_col] > 1000) & (~negative_mask)
    result_df.loc[mm_mask, output_col] = result_df.loc[mm_mask, thickness_col] / 1000
    result_df.loc[mm_mask, 'standardization_note'] = "Converted from mm to m"
    cm_mask = (result_df[thickness_col] >= 10) & (result_df[thickness_col] <= 1000) & (~negative_mask)
    result_df.loc[cm_mask, output_col] = result_df.loc[cm_mask, thickness_col] / 100
    result_df.loc[cm_mask, 'standardization_note'] = "Converted from cm to m"
    m_mask = (result_df[thickness_col] >= 0) & (result_df[thickness_col] < 10) & (~negative_mask)
    result_df.loc[m_mask, output_col] = result_df.loc[m_mask, thickness_col]
    result_df.loc[m_mask, 'standardization_note'] = "Already in meters"
    if 'thickness_m' in result_df.columns:
        result_df['discrepancy'] = np.abs(result_df[output_col] - result_df['thickness_m'])
        significant_diff = result_df['discrepancy'] > 0.01
        print(f"Found {significant_diff.sum()} rows with significant discrepancies")
        if significant_diff.sum() > 0:
            diagnostic_df = result_df.loc[significant_diff, [thickness_col, 'thickness_m', output_col, 'discrepancy', 'standardization_note', 'latitude', 'longitude', 'datetime', 'source']]
            diagnostic_df.sort_values('discrepancy', ascending=False).to_csv('thickness_standardization_issues.csv', index=False)
    return result_df

def enhanced_standardize_thickness(df, thickness_col='alt_m', output_col='thickness_m_standardized'):
    result_df = standardize_active_layer_thickness(df, thickness_col, output_col)
    conditions = [
        (result_df[output_col] > 4),
        (result_df['discrepancy'] > 0.1) if 'discrepancy' in result_df.columns else False,
        (result_df[output_col] > 2) & (result_df[output_col] <= 4),
        (result_df[output_col] <= 2)
    ]
    confidence_levels = ['low', 'low', 'medium', 'high']
    result_df['standardization_confidence'] = np.select(conditions, confidence_levels, default='high')
    result_df['is_physical_outlier'] = False
    result_df.loc[result_df[output_col] > 4, 'is_physical_outlier'] = True
    if 'latitude' in result_df.columns and 'longitude' in result_df.columns:
        result_df['lat_bin'] = np.floor(result_df['latitude'])
        result_df['lon_bin'] = np.floor(result_df['longitude'])
        spatial_groups = result_df.groupby(['lat_bin', 'lon_bin'])
        result_df['thickness_zscore'] = np.nan
        for (lat, lon), group in spatial_groups:
            if len(group) >= 5:
                bin_mean = group[output_col].mean()
                bin_std = group[output_col].std()
                if bin_std > 0:
                    idx = group.index
                    result_df.loc[idx, 'thickness_zscore'] = (result_df.loc[idx, output_col] - bin_mean) / bin_std
        result_df['is_spatial_outlier'] = False
        result_df.loc[abs(result_df['thickness_zscore']) > 3, 'is_spatial_outlier'] = True
    extreme_mask = (result_df[output_col] > 5) & (result_df['standardization_confidence'] == 'low')
    result_df.loc[extreme_mask, output_col] = 5.0
    result_df.loc[extreme_mask, 'standardization_note'] += " (Capped at 5m due to implausible value)"
    return result_df

standardized_df = enhanced_standardize_thickness(alt_df)
print("\nSummary Statistics for Standardized Thickness (meters):")
print(standardized_df['thickness_m_standardized'].describe())
if 'thickness_m' in standardized_df.columns:
    print("\nComparison with existing thickness_m column:")
    print(f"Correlation: {standardized_df['thickness_m_standardized'].corr(standardized_df['thickness_m']):.4f}")
    print(f"Mean absolute difference: {standardized_df['discrepancy'].mean():.4f} m")
    print(f"Median absolute difference: {standardized_df['discrepancy'].median():.4f} m")
    print(f"Max absolute difference: {standardized_df['discrepancy'].max():.4f} m")
    if standardized_df['discrepancy'].max() > 0.01:
        source_groups = standardized_df.groupby('source')['discrepancy'].agg(['mean', 'max', 'count'])
        print("\nDiscrepancies by source:")
        print(source_groups.sort_values('mean', ascending=False))
standardized_df.to_csv('derivedaltfromgtnpst_matched_processed_cleaned_validated_borehole_gtnp_insitu_df.csv', index=False)
derived_alt = standardized_df
del standardized_df

# CALM (R files)
combined_df = pd.read_csv('alt_temp_smc_siberian_ismn_withoutcalmrfiles_combined_df.csv',index=False)

#Load merged_df

merged_df = pd.read_csv('alt_temp_smc_siberian_ismn_calmrfiles_combined_df.csv')
merged_df = merged_df.reset_index(drop=True)
merged_df['datetime'] = pd.to_datetime(merged_df['datetime'],format='mixed')
merged_df = merged_df[(merged_df['datetime'] <= '2024-12-31 00:00:00') & (merged_df['latitude'] >= 49)]
merged_df['longitude'] = merged_df.apply(
    lambda row: row['longitude'] - 360 if row['longitude'] > 180 else row['longitude'],
    axis=1
)

print("New longitude range:")
print(f"Min longitude: {merged_df.longitude.min():.6f}")
print(f"Max longitude: {merged_df.longitude.max():.6f}")
merged_df['datetime'] = pd.to_datetime(merged_df['datetime'],format='mixec')
merged_df = merged_df.sort_values('datetime').reset_index(drop=True)

merged_df.to_feather('merged_compressed.feather', compression='zstd')

merged_df

# First, let's analyze the distribution to confirm our suspicion
print("Summary statistics:")
print(merged_df['soil_moist'].describe())
# Check the proportion of values above 1.0 (likely percentage format)
pct_above_1 = (merged_df['soil_moist'] > 1.0).mean() * 100
print(f"Percentage of values above 1.0: {pct_above_1:.2f}%")
# Create a standardized soil moisture column
merged_df['soil_moist_standardized'] = merged_df['soil_moist'].copy()
# Convert percentage values (>1.0) to decimal form (0-1 range)
mask_percentage = merged_df['soil_moist'] > 1.0
merged_df.loc[mask_percentage, 'soil_moist_standardized'] = merged_df.loc[mask_percentage, 'soil_moist'] / 100.0
# Verify the conversion worked
print("\nAfter standardization:")
print(merged_df['soil_moist_standardized'].describe())
# Double-check that all values are now between 0 and 1
above_1_after = (merged_df['soil_moist_standardized'] > 1.0).sum()
print(f"Values still above 1.0: {above_1_after}")
# Final verification - maximum should be close to but not exceed 1.0
print(f"Maximum standardized value: {merged_df['soil_moist_standardized'].max():.4f}")

merged_df = merged_df[['datetime', 'year', 'season', 
                       'latitude', 'longitude', 
                       'thickness_m', 'thickness_m_standardized', 'soil_temp', 'soil_temp_standardized', 'soil_temp_depth', 'soil_temp_depth_zone', 'soil_moist', 'soil_moist_standardized', 'soil_moist_depth', 
                       'source', 'data_type']]

merged_df['datetime'] = pd.to_datetime(merged_df['datetime'],format='mixec')
merged_df = merged_df.sort_values('datetime').reset_index(drop=True)

merged_df.to_feather('merged_compressed.feather', compression='zstd')

# Load updated merged_df

merged_df = pd.read_feather('merged_compressed.feather')
merged_df

#LIST OF COLUMNS
#['datetime', 'year', 'season', 'latitude', 'longitude', 'thickness_m', 'thickness_m_standardized', ...

# #TEMP
# # - GTNP ST (Borehole_*)
# #gtnp_st = pd.read_csv('updated_cleaned_validated_processed_borehole_gtnp_soiltemp_insitu_df.csv')...
# #gtnp_st = pd.read_csv('cleaned_df_final_gtnp_st_w_depth_insitu_df.csv') #GTNP ST (Borehole*), cle...
# gtnp_st = pd.read_csv('ZC_df_data.csv').sort_values('datetime').reset_index(drop=True) #Just anoth...
# # - RuMeteo (RU meteorological data)
# ru_dst = pd.read_csv('ru_dst.csv').sort_values('datetime').reset_index(drop=True)

temp = merged_df[merged_df.soil_temp_standardized.isna()!=True][['datetime', 'year', 'season', 'latitude', 'longitude', 'soil_temp', 'soil_temp_standardized', 'soil_temp_depth', 'soil_temp_depth_zone', 'source', 'data_type']]

# #ALT
# # - External ALT (CALM, ABoVE, etc.)
# #external_alt = pd.read_csv('external_alt_final_insitu_df.csv').sort_values('datetime').reset_inde...
# standardized_external_alt = pd.read_csv('external_alt_final_insitu_df_standardized.csv').sort_valu...
# # - GTNP ALT (Activelayer_*)
# gtnp_alt = pd.read_csv('cleaned_df_final_gtnp_alt_insitu_df.csv'.sort_values('datetime').reset_ind...
# #derived_alt = pd.read_csv('matched_processed_cleaned_validated_borehole_gtnp_derivedaltfromst_ins...

alt = merged_df[merged_df.thickness_m_standardized.isna()!=True][['datetime', 'year', 'season', 'latitude', 'longitude', 'thickness_m', 'thickness_m_standardized', 'source', 'data_type']]

#SMC

smc = merged_df[merged_df.soil_moist.isna()!=True][['datetime', 'year', 'season', 'latitude', 'longitude', 'soil_moist', 'soil_moist_standardized', 'soil_moist_depth', 'source', 'data_type']]



# merged_df = pd.read_feather('merged_compressed.feather')
# merged_df
# datetime	year	season	latitude	longitude	thickness_m	thickness_m_standardized	soil_temp	soil_temp_s...
# 0	1891-01-15 12:00:00	1891.0	Winter	52.283299	104.300003	NaN	NaN	-10.5	-10.5	0.8	deep	NaN	NaN	NaN	...
# 1	1891-01-15 12:00:00	1891.0	Winter	52.283299	104.300003	NaN	NaN	2.4	2.4	3.2	very_deep	NaN	NaN	NaN...
# 2	1891-01-15 12:00:00	1891.0	Winter	52.283299	104.300003	NaN	NaN	-0.5	-0.5	1.6	very_deep	NaN	NaN	N...
# 3	1891-01-15 12:00:00	1891.0	Winter	52.283299	104.300003	NaN	NaN	-13.4	-13.4	0.4	intermediate	NaN	...
# 4	1891-02-14 00:00:00	1891.0	Winter	52.283299	104.300003	NaN	NaN	-11.7	-11.7	0.8	deep	NaN	NaN	NaN	...
# ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
# 62708663	2024-12-31 00:00:00	2024.0	Winter	65.154010	-147.502580	NaN	NaN	NaN	NaN	NaN	None	7.59	0.0...
# 62708664	2024-12-31 00:00:00	2024.0	Winter	63.875800	-149.213350	NaN	NaN	NaN	NaN	NaN	None	0.00	0.0...
# 62708665	2024-12-31 00:00:00	2024.0	Winter	65.154010	-147.502580	NaN	NaN	NaN	NaN	NaN	None	4.69	0.0...
# 62708666	2024-12-31 00:00:00	2024.0	Winter	71.282410	-156.619360	NaN	NaN	NaN	NaN	NaN	None	1.66	0.0...
# 62708667	2024-12-31 00:00:00	2024.0	Winter	65.154010	-147.502580	NaN	NaN	NaN	NaN	NaN	None	28.13	0....
# 62708668 rows × 16 columns











# # zero_curtain_pipeline.py
# import os
# import numpy as np
# import pandas as pd
# import gc
# import time
# from tqdm.auto import tqdm

# def run_memory_efficient_pipeline(feather_path, output_dir=None, 
#                                   site_batch_size=20, checkpoint_interval=5, 
#                                   max_gap_hours=6, interpolation_method='cubic', 
#                                   force_restart=False):
#     """
#     Run the complete memory-efficient zero curtain detection pipeline
    
#     Parameters:
#     -----------
#     feather_path : str
#         Path to the feather file
#     output_dir : str
#         Directory to save results and checkpoints
#     site_batch_size : int
#         Number of sites to process in each batch (smaller = less memory)
#     checkpoint_interval : int
#         Number of sites between saving checkpoints
#     max_gap_hours : float
#         Maximum gap hours for interpolation
#     interpolation_method : str
#         Interpolation method ('linear', 'cubic')
        
#     Returns:
#     --------
#     pandas.DataFrame
#         DataFrame with detected zero curtain events
#     """
#     print("=" * 80)
#     print("TRULY MEMORY-EFFICIENT ZERO CURTAIN DETECTION")
#     print("=" * 80)
#     print(f"Initial memory usage: {memory_usage():.1f} MB")
    
#     # Set up directories
#     if output_dir is not None:
#         os.makedirs(output_dir, exist_ok=True)
#         checkpoint_dir = os.path.join(output_dir, 'checkpoints')
#         os.makedirs(checkpoint_dir, exist_ok=True)
#     else:
#         checkpoint_dir = None
    
#     # Start timing
#     start_time = time.time()

#     def save_checkpoint(data, name):
#         if checkpoint_dir:
#             backup_path = os.path.join(checkpoint_dir, f'{name}_backup.pkl')
#             target_path = os.path.join(checkpoint_dir, f'{name}.pkl')
            
#             # First save to backup file
#             with open(backup_path, 'wb') as f:
#                 pickle.dump(data, f)
            
#             # Then rename to target (atomic operation)
#             import shutil
#             shutil.move(backup_path, target_path)
            
#             print(f"Saved checkpoint to {target_path}")
    
#     # Function to load checkpoint
#     def load_checkpoint(name):
#         if checkpoint_dir:
#             try:
#                 with open(os.path.join(checkpoint_dir, f'{name}.pkl'), 'rb') as f:
#                     data = pickle.load(f)
#                 print(f"Loaded checkpoint from {name}.pkl")
#                 return data
#             except:
#                 print(f"No checkpoint found for {name}.pkl")
#         return None
    
#     # Step 1: Get site-depth combinations
#     site_depths = None
#     if not force_restart:
#         site_depths = load_checkpoint('site_depths')
    
#     if site_depths is None:
#         print("\nStep 1: Finding unique site-depth combinations...")
#         site_depths = get_unique_site_depths(feather_path)
#         save_checkpoint(site_depths, 'site_depths')
    
#     total_combinations = len(site_depths)
#     print(f"Found {total_combinations} unique site-depth combinations")
    
#     # Step 2: Initialize results
#     all_events = []
#     processed_indices = set()
    
#     if not force_restart:
#         saved_events = load_checkpoint('all_events')
#         if saved_events is not None:
#             if isinstance(saved_events, list):
#                 all_events = saved_events
#             else:
#                 # If it's a DataFrame, convert to list of dicts
#                 all_events = saved_events.to_dict('records') if len(saved_events) > 0 else []
        
#         saved_indices = load_checkpoint('processed_indices')
#         if saved_indices is not None:
#             processed_indices = set(saved_indices)
    
#     print(f"Starting from {len(processed_indices)}/{total_combinations} processed sites")
#     print(f"Current event count: {len(all_events)}")

#     # Step 3: Process in batches
#     print("\nProcessing site-depth combinations in batches")
    
#     # For tracking new events in this run
#     new_events_count = 0
    
#     # Create batches for processing
#     total_batches = (total_combinations + site_batch_size - 1) // site_batch_size
    
#     for batch_idx in range(total_batches):
#         batch_start = batch_idx * site_batch_size
#         batch_end = min(batch_start + site_batch_size, total_combinations)
        
#         # Skip if already processed
#         batch_indices = set(range(batch_start, batch_end))
#         if batch_indices.issubset(processed_indices):
#             print(f"Batch {batch_idx+1}/{total_batches} (sites {batch_start+1}-{batch_end}/{total_...
#             continue
        
#         print(f"\nProcessing batch {batch_idx+1}/{total_batches} (sites {batch_start+1}-{batch_end...
#         print(f"Memory before batch: {memory_usage():.1f} MB")
        
#         # Force garbage collection
#         gc.collect()
        
#         # Process each site in batch
#         for site_idx in range(batch_start, batch_end):
#             # Skip if already processed
#             if site_idx in processed_indices:
#                 continue
            
#             site = site_depths.iloc[site_idx]['source']
#             temp_depth = site_depths.iloc[site_idx]['soil_temp_depth']
            
#             print(f"\nProcessing site {site_idx+1}/{total_combinations}: {site}, depth: {temp_dept...
            
#             try:
#                 # Load site data efficiently
#                 site_df = load_site_depth_data(feather_path, site, temp_depth)
                
#                 # Skip if insufficient data
#                 if len(site_df) < 10:
#                     print(f"  Insufficient data ({len(site_df)} rows), skipping")
#                     processed_indices.add(site_idx)
#                     continue
                
#                 # Process for zero curtain
#                 site_events = process_site_for_zero_curtain(
#                     site_df, site, temp_depth, 
#                     max_gap_hours, interpolation_method
#                 )
                
#                 # Add to all events
#                 new_events = len(site_events)
#                 all_events.extend(site_events)
#                 new_events_count += new_events
                
#                 print(f"  Found {new_events} events, new total: {len(all_events)}")
                
#                 # Mark as processed
#                 processed_indices.add(site_idx)
                
#                 # Clean up
#                 #del site_df, site_events
#                 #gc.collect()
            
#                 # Save checkpoint periodically
#                 if site_idx % checkpoint_interval == 0:
#                     save_checkpoint(all_events, 'all_events')
#                     save_checkpoint(list(processed_indices), 'processed_indices')
                    
#                 # Clean up
#                 del site_df, site_events
#                 gc.collect()

#             except Exception as e:
#                 print(f"  Error processing site {site}, depth {temp_depth}: {str(e)}")
#                 continue # Continue with next site
        
#         # Save checkpoint after each batch
#         print(f"Saving checkpoint after batch {batch_idx+1}/{total_batches}")
#         save_checkpoint(all_events, 'all_events')
#         save_checkpoint(list(processed_indices), 'processed_indices')
        
#         # Also save intermediate results as CSV
#         if len(all_events) > 0:
#             interim_df = pd.DataFrame(all_events)
#             if output_dir is not None:
#                 interim_path = os.path.join(output_dir, 'events_checkpoint.csv')
#                 interim_df.to_csv(interim_path, index=False)
#                 print(f"Saved interim results to {interim_path}")
        
#         print(f"Memory after batch: {memory_usage():.1f} MB")
    
#     # Step 4: Create final dataframe
#     print("\nCreating final events dataframe")
    
#     if len(all_events) > 0:
#         events_df = pd.DataFrame(all_events)
#         print(f"Created events dataframe with {len(events_df)} total events ({new_events_count} ne...
#     else:
#         # Create empty dataframe with correct columns
#         events_df = pd.DataFrame(columns=[
#             'source', 'soil_temp_depth', 'soil_temp_depth_zone', 
#             'datetime_min', 'datetime_max', 'duration_hours',
#             'observation_count', 'observations_per_day',
#             'soil_temp_mean', 'soil_temp_min', 'soil_temp_max', 'soil_temp_std',
#             'season', 'latitude', 'longitude', 'year', 'month',
#             'soil_moist_mean', 'soil_moist_std', 'soil_moist_min', 'soil_moist_max', 
#             'soil_moist_change', 'soil_moist_depth',
#             'temp_gradient_mean', 'temp_stability',
#             'year_month', 'region', 'lat_band'
#         ])
#         print("No events found")
    
#     # Save final results
#     if output_dir is not None:
#         final_path = os.path.join(output_dir, 'zero_curtain_events.csv')

#         temp_path = os.path.join(output_dir, 'zero_curtain_events_temp.csv')
#         events_df.to_csv(temp_path, index=False)

#         import shutil
#         shutil.move(temp_path, final_path)
        
#         print(f"Saved final results to {final_path}")
    
#     # Report timing
#     total_time = time.time() - start_time
#     print(f"\nPipeline completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
#     print(f"Final memory usage: {memory_usage():.1f} MB")
    
#     return events_df



# FIX ALL DPIS to 300!!!!!!!!!!!

import sys
sys.path.append("/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling/scripts")

from zero_curtain_standalone import build_zero_curtain_model, BatchNorm5D

import numpy as np

X = np.load("zero_curtain_pipeline/modeling/scripts/hybrid_results/ml_data/X_features.npy")  # shape: (12608, 168, 3)
input_shape = X.shape[1:]      # (168, 3)

# Instantiate the model
model = build_zero_curtain_model(input_shape=input_shape)

# Load pretrained weights
model.load_weights("/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling/scripts/hybrid_results/ml_model/checkpoints/final_model_weights")

model.summary()

# Run inference on the first sample (batch size 1)
y_pred = model.predict(X[:1])
print("Prediction shape:", y_pred.shape)
print(y_pred)

# Simplified model diagram (vertical layout)
plot_model_architecture(model, 'zero_curtain_model_vertical.png', rankdir='TB')

# Horizontal layout for potentially better visibility of complex architectures
plot_model_architecture(model, 'zero_curtain_model_horizontal.png', rankdir='LR')

create_simplified_diagram()

create_flowchart_visualization()

X = np.load("zero_curtain_pipeline/modeling/scripts/hybrid_results/ml_data/X_features.npy")
y = np.load("zero_curtain_pipeline/modeling/scripts/hybrid_results/ml_data/y_labels.npy")

X.shape

y.shape















with open('/Users/bgay/Desktop/Research/Code/merged_compressed.feather','rb') as f:
    data = pd.read_feather(f)

data

#Filter out the exceptional/extreme values for zero-curtain periods derived from preliminary evaluat...
data = data.drop(index=long_events_summary.index.values.tolist()).sort_values('datetime').reset_index(drop=True)

data

moisture_data = data[data['soil_moist_depth'].notna() & data['soil_moist'].notna() & data['soil_moist_standardized'].notna()]\
[['datetime', 'year', 'season', 'latitude', 'longitude', 'soil_moist', 'soil_moist_standardized', 'soil_moist_depth', 'source', \
  'data_type']]\
.sort_values('datetime').reset_index(drop=True)
#moisture_data['datetime'] = moisture_data['datetime'].dt.strftime('%Y-%m-%d')
moisture_data['datetime'] = pd.to_datetime(moisture_data['datetime'], format='%Y-%m-%d %H:%M:%S')
moisture_data.datetime = pd.to_datetime(moisture_data.datetime,format='mixed')
moisture_data.year=moisture_data.year.astype(int)
moisture_data = moisture_data.sort_values('datetime').reset_index(drop=True)
moisture_data

datetime	year	season	latitude	longitude	soil_moist	soil_moist_standardized	soil_moist_depth	source	data_type
0	1952-06-08	1952	Summer	49.049999	51.869999	0.1811	0.1811	1.00	KALMYKOVO	soil_moisture
1	1952-06-18	1952	Summer	49.049999	51.869999	0.1831	0.1831	1.00	KALMYKOVO	soil_moisture
2	1952-06-28	1952	Summer	49.049999	51.869999	0.1841	0.1841	1.00	KALMYKOVO	soil_moisture
3	1952-07-08	1952	Summer	49.049999	51.869999	0.1861	0.1861	1.00	KALMYKOVO	soil_moisture
4	1952-07-18	1952	Summer	49.049999	51.869999	0.1981	0.1981	1.00	KALMYKOVO	soil_moisture
...	...	...	...	...	...	...	...	...	...	...
21487216	2024-12-31	2024	Winter	50.514900	6.375590	0.4350	0.4350	0.05	Schoeneseiffen	soil_moisture
21487217	2024-12-31	2024	Winter	50.930302	6.297470	0.3370	0.3370	0.50	Merzenhausen	soil_moisture
21487218	2024-12-31	2024	Winter	50.930302	6.297470	0.3380	0.3380	0.50	Merzenhausen	soil_moisture
21487219	2024-12-31	2024	Winter	50.989201	6.323550	0.3050	0.3050	0.20	Gevenich	soil_moisture
21487220	2024-12-31	2024	Winter	50.869099	6.449540	0.3270	0.3270	0.20	Selhausen	soil_moisture
21487221 rows × 10 columns

moisture_data.isna().sum()

datetime                   0
year                       0
season                     0
latitude                   0
longitude                  0
soil_moist                 0
soil_moist_standardized    0
soil_moist_depth           0
source                     0
data_type                  0
dtype: int64

alt_data = data[data['thickness_m'].notna() & data['thickness_m_standardized'].notna()]\
[['datetime', 'year', 'season', 'latitude', 'longitude', 'thickness_m', 'thickness_m_standardized', 'data_type']]\
.sort_values('datetime').reset_index(drop=True)
alt_data['datetime'] = pd.to_datetime(alt_data['datetime'], format='%Y-%m-%d %H:%M:%S')
alt_data.year=alt_data.year.astype(int)
alt_data = alt_data.sort_values('datetime').reset_index(drop=True)
alt_data

datetime	year	season	latitude	longitude	thickness_m	thickness_m_standardized	data_type
0	1903-05-15 12:00:00	1903	Spring	52.283299	104.300003	1.466667	1.466667	active_layer
1	1904-05-15 12:00:00	1904	Spring	52.283299	104.300003	1.490909	1.490909	active_layer
2	1905-05-15 12:00:00	1905	Spring	52.283299	104.300003	1.527273	1.527273	active_layer
3	1906-05-15 12:00:00	1906	Spring	52.283299	104.300003	1.522581	1.522581	active_layer
4	1907-06-15 00:00:00	1907	Summer	52.283299	104.300003	1.582979	1.582979	active_layer
...	...	...	...	...	...	...	...	...
1032339	2024-08-31 00:00:00	2024	Summer	72.369775	126.480632	0.490000	0.490000	active_layer
1032340	2024-08-31 00:00:00	2024	Summer	72.369775	126.480632	0.500000	0.500000	active_layer
1032341	2024-08-31 00:00:00	2024	Summer	72.369775	126.480632	0.570000	0.570000	active_layer
1032342	2024-08-31 00:00:00	2024	Summer	72.369775	126.480632	0.620000	0.620000	active_layer
1032343	2024-08-31 00:00:00	2024	Summer	72.369775	126.480632	0.600000	0.600000	active_layer
1032344 rows × 8 columns

alt_data.isna().sum()

datetime                    0
year                        0
season                      0
latitude                    0
longitude                   0
thickness_m                 0
thickness_m_standardized    0
data_type                   0
dtype: int64

temp_data = data[data['soil_temp_depth'].notna() & data['soil_temp'].notna() & data['soil_temp_standardized'].notna() & \
data['soil_temp_depth_zone'].notna()]\
[['datetime', 'year', 'season', 'latitude', 'longitude', 'soil_temp', 'soil_temp_standardized', 'soil_temp_depth', 'soil_temp_depth_zone',\
  'source', 'data_type']]\
.sort_values('datetime').reset_index(drop=True)
temp_data['datetime'] = pd.to_datetime(temp_data['datetime'], format='%Y-%m-%d %H:%M:%S')
temp_data.year=temp_data.year.astype(int)
temp_data = temp_data.sort_values('datetime').reset_index(drop=True)
temp_data

datetime	year	season	latitude	longitude	soil_temp	soil_temp_standardized	soil_temp_depth	soil_temp_depth_zone	source	data_type
0	1903-05-15 12:00:00	1903	Spring	52.283299	104.300003	-0.60	-0.60	1.60	very_deep	GTNP_Borehole_1695	soil_temperature
1	1903-05-15 12:00:00	1903	Spring	52.283299	104.300003	6.50	6.50	0.40	intermediate	GTNP_Borehole_1695	soil_temperature
2	1903-06-15 00:00:00	1903	Summer	52.283299	104.300003	5.10	5.10	0.80	deep	GTNP_Borehole_1653	soil_temperature
3	1903-06-15 00:00:00	1903	Summer	52.283299	104.300003	0.90	0.90	3.20	very_deep	GTNP_Borehole_1695	soil_temperature
4	1903-06-15 00:00:00	1903	Summer	52.283299	104.300003	-0.30	-0.30	1.60	very_deep	GTNP_Borehole_1695	soil_temperature
...	...	...	...	...	...	...	...	...	...	...	...
34370768	2023-07-30 00:00:00	2023	Summer	56.905454	118.281306	15.67	15.67	0.01	shallow	GTNP_Borehole_225	soil_temperature
34370769	2023-07-31 00:00:00	2023	Summer	56.905454	118.281306	14.86	14.86	0.01	shallow	GTNP_Borehole_225	soil_temperature
34370770	2023-07-31 00:00:00	2023	Summer	56.901670	118.081930	14.46	14.46	0.01	shallow	GTNP_Borehole_1117	soil_temperature
34370771	2023-07-31 00:00:00	2023	Summer	56.901670	118.081930	-0.02	-0.02	1.00	deep	GTNP_Borehole_1117	soil_temperature
34370772	2023-08-01 00:00:00	2023	Summer	56.905454	118.281306	14.70	14.70	0.01	shallow	GTNP_Borehole_225	soil_temperature
34370773 rows × 11 columns

temp_data.isna().sum()

datetime                  0
year                      0
season                    0
latitude                  0
longitude                 0
soil_temp                 0
soil_temp_standardized    0
soil_temp_depth           0
soil_temp_depth_zone      0
source                    0
data_type                 0
dtype: int64

data.to_csv('data_exceptionalzerocurtainperiodfilteredout.csv', index=False)

temp_data.to_csv('temp_data_exceptionalzerocurtainperiodfilteredout.csv', index=False)

moisture_data.to_csv('moisture_data_exceptionalzerocurtainperiodfilteredout.csv', index=False)

alt_data.to_csv('alt_data_exceptionalzerocurtainperiodfilteredout.csv', index=False)











































data = pd.read_feather('/Users/bgay/Desktop/Research/Code/merged_compressed.feather')

data['soil_moist']

#soil_moist column values from entire feather dataframe 
0             NaN
1             NaN
2             NaN
3             NaN
4             NaN
            ...  
62708663     7.59
62708664     0.00
62708665     4.69
62708666     1.66
62708667    28.13
Name: soil_moist, Length: 62708668, dtype: float64

data['soil_moist'].dropna()

#soil_moist column values from NaN-filtered/dropped NaNs in feather dataframe
40825        0.1811
41082        0.1831
41083        0.1841
41086        0.1861
41366        0.1981
             ...   
62708663     7.5900
62708664     0.0000
62708665     4.6900
62708666     1.6600
62708667    28.1300
Name: soil_moist, Length: 27304593, dtype: float64

#indices to determine other column values for where soil_moist values are present in feather datafra...

data['soil_moist'].dropna().index

Index([   40825,    41082,    41083,    41086,    41366,    41367,    41368,
          41671,    41672,    41673,
       ...
       62708658, 62708659, 62708660, 62708661, 62708662, 62708663, 62708664,
       62708665, 62708666, 62708667],
      dtype='int64', length=27304593)

data.columns



import os
import pandas as pd
import pickle
import gc

# 1. Define paths
feather_path = '/Users/bgay/Desktop/Research/Code/merged_compressed.feather'
output_dir = 'zero_curtain_moisture_run'
new_site_depths_path = '/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling/scripts/zero_curtain_new/checkpoints/site_depths.pkl'

# 2. Create directory structure
os.makedirs(output_dir, exist_ok=True)
events_dir = os.path.join(output_dir, 'events')
ml_dir = os.path.join(output_dir, 'ml_features')
checkpoint_dir = os.path.join(events_dir, 'checkpoints')
os.makedirs(events_dir, exist_ok=True)
os.makedirs(ml_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# 3. Copy the correctly-detected site_depths to the new checkpoint directory
print(f"Copying site_depths with moisture detection to {checkpoint_dir}")
with open(new_site_depths_path, 'rb') as f:
    site_depths = pickle.load(f)

# Check moisture coverage
moisture_sites = site_depths['has_moisture_data'].sum()
total_sites = len(site_depths)
print(f"Site depths with moisture data: {moisture_sites}/{total_sites} ({moisture_sites/total_sites*100:.1f}%)")

# Save to the pipeline checkpoint directory
with open(os.path.join(checkpoint_dir, 'site_depths.pkl'), 'wb') as f:
    pickle.dump(site_depths, f)

# 4. Initialize empty events and processed_indices
print("Initializing empty checkpoint files")
all_events = []
processed_indices = []

with open(os.path.join(checkpoint_dir, 'all_events.pkl'), 'wb') as f:
    pickle.dump(all_events, f)
    
with open(os.path.join(checkpoint_dir, 'processed_indices.pkl'), 'wb') as f:
    pickle.dump(processed_indices, f)

print("\nSetup complete. Now run the pipeline below.")
# print("\nSetup complete. Now run the pipeline using:")
# print("\nrun_memory_efficient_pipeline_fixed(")
# print(f"    feather_path='{feather_path}',")
# print(f"    output_dir='{events_dir}',")
# print("    site_batch_size=10,")
# print("    checkpoint_interval=5,")
# print("    max_gap_hours=8,")
# print("    interpolation_method='cubic',")
# print("    force_restart=False,  # IMPORTANT: must be False to use checkpoints")
# print("    include_moisture=True,")
# print("    verbose=True")
# print(")")

# print(f"  Starting memory: {memory_usage():.1f} MB")

# events = run_memory_efficient_pipeline_fixed(
#     feather_path='/Users/bgay/Desktop/Research/Code/merged_compressed.feather',
#     output_dir='zero_curtain_moisture_run/events',
#     site_batch_size=10,
#     checkpoint_interval=5,
#     max_gap_hours=8,
#     interpolation_method='cubic',
#     force_restart=False,  # IMPORTANT: must be False to use checkpoints
#     include_moisture=True,
#     verbose=True
#     )

# print(f"Detected {len(events)} zero curtain events with integrated soil moisture")

# print(f"  Final memory: {memory_usage():.1f} MB")

# del events
# for _ in range(3):
#     gc.collect()

## IF PIPELINE IS INTERRUPTED, RE-RUN THIS PIPELINE CODE:

# print(f"  Starting memory: {memory_usage():.1f} MB")

# events = run_memory_efficient_pipeline_fixed(
#     feather_path='/Users/bgay/Desktop/Research/Code/merged_compressed.feather',
#     output_dir='zero_curtain_moisture_run/events',
#     site_batch_size=10,
#     checkpoint_interval=5,
#     max_gap_hours=8,
#     interpolation_method='cubic',
#     force_restart=False,  # KEEP THIS FALSE TO RESUME PIPELINE WORKFLOW!
#     include_moisture=True,
#     verbose=True
#     )

# print(f"Detected {len(events)} zero curtain events with integrated soil moisture")

# print(f"  Final memory: {memory_usage():.1f} MB")

# del events
# for _ in range(3):
#     gc.collect()

# print(f"  Starting memory: {memory_usage():.1f} MB")

# events = run_ultra_robust_pipeline(
#     feather_path='/Users/bgay/Desktop/Research/Code/merged_compressed.feather',
#     output_dir='zero_curtain_moisture_run/events',
#     site_batch_size=10,
#     checkpoint_interval=5
# )

# print(f"Detected {len(events)} zero curtain events with integrated soil moisture")

# print(f"  Final memory: {memory_usage():.1f} MB")

# del events
# for _ in range(3):
#     gc.collect()











# Check intermediate results for moisture data
import pandas as pd

events_checkpoint = pd.read_csv('zero_curtain_moisture_run/events/events_checkpoint.csv')

# Check moisture columns
moisture_cols = [col for col in events_checkpoint.columns if 'moist' in col.lower()]
print(f"Found {len(moisture_cols)} moisture columns")

# Check data presence in each column
for col in moisture_cols:
    non_null = pd.to_numeric(events_checkpoint[col], errors='coerce').notna().sum()
    pct = non_null / len(events_checkpoint) * 100
    print(f"  {col}: {non_null}/{len(events_checkpoint)} non-null values ({pct:.2f}%)")



events_df = pd.read_csv('/Users/bgay/Desktop/Research/Code/zero_curtain_results_3/events/zero_curtain_events.csv')

output_dir='zero_curtain_results_3'







if len(events_df) > 0:
    try:
        print("\nRunning diagnostics on detected events...")
        diagnostics = diagnose_zero_curtain_durations(
            events_df,
            output_dir=os.path.join(output_dir, 'diagnostics')
        )
        
        # Save diagnostic results
        if isinstance(diagnostics, dict):
            import json
            with open(os.path.join(output_dir, 'diagnostics.json'), 'w') as f:
                json.dump({k: str(v) for k, v in diagnostics.items() if k not in ['statistics', 'common_values']}, f, indent=2)
            
            print("Diagnostic results saved to diagnostics.json")
    except ImportError:
        print("Diagnostics module not available. Skipping diagnostics.")
    except Exception as e:
        print(f"Error running diagnostics: {str(e)}")

# Final cleanup
del events_df
gc.collect()

print(f"Final memory usage: {memory_usage():.1f} MB")
print("Done!")



enhanced_events = pd.read_csv('/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/zero_curtain_events.csv')

print(enhanced_events.columns)
print(enhanced_events.head())

enhanced_events.datetime_min[0]

#print(enhanced_events.columns.tolist())
enhanced_events = enhanced_events[['datetime_min', 'datetime_max', 'duration_hours', 'year', 'month', 'year_month', 'season', 
                                   'latitude', 'longitude', 'lat_band', 'region', 
                                   'observation_count', 'observations_per_day', 
                                   'soil_temp_mean', 'soil_temp_min', 'soil_temp_max', 'soil_temp_std', 'soil_temp_depth', 
                                   'soil_temp_depth_zone', 'temp_gradient_mean', 'temp_stability', 
                                   'soil_moist_mean', 'soil_moist_std', 'soil_moist_min', 'soil_moist_max', 
                                   'soil_moist_change', 'soil_moist_depth', 
                                   'source']]

enhanced_events = enhanced_events.sort_values('datetime_min').reset_index(drop=True)

enhanced_events.to_csv('/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/zero_curtain_events_sorted.csv', index=False)

enhanced_events = pd.read_csv('/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/zero_curtain_events_sorted.csv')

enhanced_events.datetime_min = pd.to_datetime(enhanced_events.datetime_min,format='mixed')
enhanced_events.datetime_max = pd.to_datetime(enhanced_events.datetime_max,format='mixed')

from scipy import stats
import scipy.stats as stats
from scipy.stats import pearsonr

print("\nSpatial Analysis Summary:")

unique_sites = enhanced_events['source'].nunique()
print(f"Total Unique Sites: {unique_sites}")

print("\nSpatial Extent:")
print(f"Latitude Range: {enhanced_events['latitude'].min():.2f} to {enhanced_events['latitude'].max():.2f}")
print(f"Longitude Range: {enhanced_events['longitude'].min():.2f} to {enhanced_events['longitude'].max():.2f}")

top_sites = enhanced_events.groupby('source').size().nlargest(5)
print("\nTop 5 Sites by Event Count:")
print(top_sites)

depth_distribution = enhanced_events.groupby('soil_temp_depth_zone').size()
print("\nDepth Zone Distribution:")
print(depth_distribution)

print("\nStatistical Analysis:")

seasonal_groups = [group['duration_hours'].values for name, group in enhanced_events.groupby('season')]
f_stat, p_val = stats.f_oneway(*seasonal_groups)
print("\nANOVA - Seasonal Duration Differences:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_val:.4f}")

corr, p_corr = stats.pearsonr(enhanced_events['soil_temp_mean'], enhanced_events['duration_hours'])
print("\nCorrelation - Temperature vs Duration:")
print(f"Pearson r: {corr:.4f}")
print(f"p-value: {p_corr:.4f}")

yearly_trends = enhanced_events.groupby('year').agg({
    'duration_hours': ['mean', 'count'],
    'soil_temp_mean': ['mean', 'std']
})

def compute_zero_curtain_statistics(events_df, output_file=None):
    """
    Compute comprehensive statistics and metrics from zero curtain events dataframe.
    
    Parameters:
    -----------
    events_df : pandas.DataFrame
        DataFrame containing zero curtain events (output from enhanced_zero_curtain_detection)
    output_file : str, optional
        If provided, save statistics to this file path
        
    Returns:
    --------
    dict
        Dictionary containing multiple statistical summaries and metrics
    """
    import pandas as pd
    import numpy as np
    from collections import Counter
    
    stats = {}
    
    # Basic counts and summary
    stats['total_events'] = len(events_df)
    stats['unique_sites'] = events_df['source'].nunique()
    stats['mean_events_per_site'] = len(events_df) / events_df['source'].nunique() if events_df['source'].nunique() > 0 else 0
    
    # Duration statistics overall
    duration_stats = events_df['duration_hours'].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    stats['duration'] = {
        'mean': events_df['duration_hours'].mean(),
        'median': events_df['duration_hours'].median(),
        'std': events_df['duration_hours'].std(),
        'min': events_df['duration_hours'].min(),
        'max': events_df['duration_hours'].max(),
        'q1': duration_stats['25%'],
        'q3': duration_stats['75%'],
        'p5': duration_stats['5%'],
        'p95': duration_stats['95%'],
        'iqr': duration_stats['75%'] - duration_stats['25%'],
        'most_common': Counter(events_df['duration_hours']).most_common(5)
    }
    
    # Duration statistics in days
    stats['duration_days'] = {
        'mean': stats['duration']['mean'] / 24,
        'median': stats['duration']['median'] / 24,
        'min': stats['duration']['min'] / 24,
        'max': stats['duration']['max'] / 24,
        'q1': stats['duration']['q1'] / 24,
        'q3': stats['duration']['q3'] / 24
    }
    
    # Temperature statistics
    if 'soil_temp_mean' in events_df.columns:
        temp_stats = events_df['soil_temp_mean'].describe()
        stats['soil_temperature'] = {
            'mean': events_df['soil_temp_mean'].mean(),
            'median': events_df['soil_temp_mean'].median(),
            'std': events_df['soil_temp_mean'].std(),
            'min': events_df['soil_temp_mean'].min(),
            'max': events_df['soil_temp_mean'].max()
        }
    
    # Soil moisture statistics (if available)
    if 'soil_moist_mean' in events_df.columns and not events_df['soil_moist_mean'].isna().all():
        # Filter out NaN values
        soil_moist_data = events_df['soil_moist_mean'].dropna()
        if len(soil_moist_data) > 0:
            moist_stats = soil_moist_data.describe()
            stats['soil_moisture'] = {
                'data_availability': len(soil_moist_data) / len(events_df) * 100,  # % of events with moisture data
                'mean': soil_moist_data.mean(),
                'median': soil_moist_data.median(),
                'std': soil_moist_data.std(),
                'min': soil_moist_data.min(),
                'max': soil_moist_data.max()
            }
            
            if 'soil_moist_change' in events_df.columns:
                moist_change = events_df['soil_moist_change'].dropna()
                if len(moist_change) > 0:
                    stats['soil_moisture']['mean_change'] = moist_change.mean()
                    stats['soil_moisture']['median_change'] = moist_change.median()
    
    # Regional statistics
    if 'region' in events_df.columns:
        region_stats = events_df.groupby('region').agg({
            'duration_hours': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'source': 'nunique'
        })
        
        # Flatten the column names
        region_stats.columns = ['_'.join(col).strip('_') for col in region_stats.columns]
        
        # Convert to dictionary for easier access
        stats['by_region'] = region_stats.to_dict(orient='index')
        
        # Calculate percentage of events by region
        region_counts = events_df['region'].value_counts()
        region_percentages = (region_counts / len(events_df) * 100).to_dict()
        stats['region_distribution'] = region_percentages
    
    # Latitude band statistics
    if 'lat_band' in events_df.columns:
        lat_stats = events_df.groupby('lat_band').agg({
            'duration_hours': ['count', 'mean', 'median', 'std'],
            'source': 'nunique'
        })
        
        # Flatten the column names
        lat_stats.columns = ['_'.join(col).strip('_') for col in lat_stats.columns]
        
        # Sort by latitude band
        lat_order = ['<55°N', '55-60°N', '60-66.5°N', '66.5-70°N', '70-75°N', '75-80°N', '>80°N']
        lat_stats = lat_stats.reindex(lat_order)
        
        # Convert to dictionary for easier access
        stats['by_latitude'] = lat_stats.to_dict(orient='index')
        
        # Calculate percentage of events by latitude band
        lat_counts = events_df['lat_band'].value_counts()
        lat_percentages = (lat_counts / len(events_df) * 100).to_dict()
        stats['latitude_distribution'] = {k: lat_percentages.get(k, 0) for k in lat_order}
    
    # Depth zone statistics
    if 'soil_temp_depth_zone' in events_df.columns:
        depth_stats = events_df.groupby('soil_temp_depth_zone').agg({
            'duration_hours': ['count', 'mean', 'median', 'std'],
            'source': 'nunique'
        })
        
        # Flatten the column names
        depth_stats.columns = ['_'.join(col).strip('_') for col in depth_stats.columns]
        
        # Sort by depth zone
        depth_order = ['shallow', 'intermediate', 'deep', 'very_deep']
        depth_stats = depth_stats.reindex(depth_order)
        
        # Convert to dictionary for easier access
        stats['by_depth'] = depth_stats.to_dict(orient='index')
        
        # Calculate percentage of events by depth zone
        depth_counts = events_df['soil_temp_depth_zone'].value_counts()
        depth_percentages = (depth_counts / len(events_df) * 100).to_dict()
        stats['depth_distribution'] = {k: depth_percentages.get(k, 0) for k in depth_order}
    
    # Seasonal statistics
    if 'season' in events_df.columns:
        season_stats = events_df.groupby('season').agg({
            'duration_hours': ['count', 'mean', 'median', 'std'],
            'source': 'nunique'
        })
        
        # Flatten the column names
        season_stats.columns = ['_'.join(col).strip('_') for col in season_stats.columns]
        
        # Sort by season
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        season_stats = season_stats.reindex([s for s in season_order if s in season_stats.index])
        
        # Convert to dictionary for easier access
        stats['by_season'] = season_stats.to_dict(orient='index')
        
        # Calculate percentage of events by season
        season_counts = events_df['season'].value_counts()
        season_percentages = (season_counts / len(events_df) * 100).to_dict()
        stats['season_distribution'] = {k: season_percentages.get(k, 0) for k in season_order if k in season_percentages}
    
    # Annual statistics
    if 'year' in events_df.columns:
        year_stats = events_df.groupby('year').agg({
            'duration_hours': ['count', 'mean', 'median'],
            'source': 'nunique'
        })
        
        # Flatten the column names
        year_stats.columns = ['_'.join(col).strip('_') for col in year_stats.columns]
        
        # Sort by year
        year_stats = year_stats.sort_index()
        
        # Convert to dictionary for easier access
        stats['by_year'] = year_stats.to_dict(orient='index')
    
    # Monthly statistics
    if 'month' in events_df.columns:
        month_stats = events_df.groupby('month').agg({
            'duration_hours': ['count', 'mean', 'median'],
            'source': 'nunique'
        })
        
        # Flatten the column names
        month_stats.columns = ['_'.join(col).strip('_') for col in month_stats.columns]
        
        # Convert to dictionary for easier access
        stats['by_month'] = month_stats.to_dict(orient='index')
        
        # Get month names for better readability
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        
        # Calculate percentage of events by month
        month_counts = events_df['month'].value_counts()
        month_percentages = (month_counts / len(events_df) * 100).to_dict()
        stats['month_distribution'] = {month_names[k]: v for k, v in month_percentages.items()}
    
    # Cross-tabulation statistics
    # Region by depth zone
    if 'region' in events_df.columns and 'soil_temp_depth_zone' in events_df.columns:
        region_depth_counts = pd.crosstab(
            events_df['region'], 
            events_df['soil_temp_depth_zone'], 
            values=events_df['duration_hours'], 
            aggfunc='count'
        ).fillna(0)
        
        stats['region_by_depth_counts'] = region_depth_counts.to_dict(orient='index')
        
        # Mean duration by region and depth
        region_depth_duration = pd.crosstab(
            events_df['region'], 
            events_df['soil_temp_depth_zone'], 
            values=events_df['duration_hours'], 
            aggfunc='mean'
        ).fillna(0)
        
        stats['region_by_depth_duration'] = region_depth_duration.to_dict(orient='index')
    
    # Format everything as a clean summary report
    summary_text = "ZERO CURTAIN EVENTS SUMMARY\n"
    summary_text += "========================\n\n"
    
    summary_text += f"Total Events: {stats['total_events']}\n"
    summary_text += f"Unique Sites: {stats['unique_sites']}\n"
    summary_text += f"Mean Events per Site: {stats['mean_events_per_site']:.2f}\n\n"
    
    summary_text += "DURATION STATISTICS\n"
    summary_text += "-----------------\n"
    summary_text += f"Mean: {stats['duration']['mean']:.2f} hours ({stats['duration_days']['mean']:.2f} days)\n"
    summary_text += f"Median: {stats['duration']['median']:.2f} hours ({stats['duration_days']['median']:.2f} days)\n"
    summary_text += f"Standard Deviation: {stats['duration']['std']:.2f} hours\n"
    summary_text += f"Range: {stats['duration']['min']:.2f} - {stats['duration']['max']:.2f} hours\n"
    summary_text += f"IQR (Q1-Q3): {stats['duration']['q1']:.2f} - {stats['duration']['q3']:.2f} hours\n"
    summary_text += f"5th-95th Percentile: {stats['duration']['p5']:.2f} - {stats['duration']['p95']:.2f} hours\n\n"
    
    summary_text += "Most Common Duration Values:\n"
    for value, count in stats['duration']['most_common']:
        summary_text += f"  {value:.2f} hours: {count} events ({count/stats['total_events']*100:.2f}%)\n"
    summary_text += "\n"
    
    # Add regional statistics
    if 'by_region' in stats:
        summary_text += "REGIONAL STATISTICS\n"
        summary_text += "------------------\n"
        for region, region_stats in stats['by_region'].items():
            summary_text += f"{region}:\n"
            summary_text += f"  Events: {region_stats['duration_hours_count']} ({stats['region_distribution'][region]:.1f}%)\n"
            summary_text += f"  Sites: {region_stats['source_nunique']}\n"
            summary_text += f"  Mean Duration: {region_stats['duration_hours_mean']:.2f} hours ({region_stats['duration_hours_mean']/24:.2f} days)\n"
            summary_text += f"  Median Duration: {region_stats['duration_hours_median']:.2f} hours\n"
        summary_text += "\n"
    
    # Add latitude band statistics
    if 'by_latitude' in stats:
        summary_text += "LATITUDE BAND STATISTICS\n"
        summary_text += "-----------------------\n"
        for band, band_stats in stats['by_latitude'].items():
            if band in stats['latitude_distribution']:
                summary_text += f"{band}:\n"
                summary_text += f"  Events: {band_stats['duration_hours_count']} ({stats['latitude_distribution'][band]:.1f}%)\n"
                summary_text += f"  Sites: {band_stats['source_nunique']}\n"
                summary_text += f"  Mean Duration: {band_stats['duration_hours_mean']:.2f} hours ({band_stats['duration_hours_mean']/24:.2f} days)\n"
                summary_text += f"  Median Duration: {band_stats['duration_hours_median']:.2f} hours\n"
        summary_text += "\n"
    
    # Add depth zone statistics
    if 'by_depth' in stats:
        summary_text += "DEPTH ZONE STATISTICS\n"
        summary_text += "--------------------\n"
        for depth, depth_stats in stats['by_depth'].items():
            if depth in stats['depth_distribution']:
                summary_text += f"{depth.capitalize()}:\n"
                summary_text += f"  Events: {depth_stats['duration_hours_count']} ({stats['depth_distribution'][depth]:.1f}%)\n"
                summary_text += f"  Sites: {depth_stats['source_nunique']}\n"
                summary_text += f"  Mean Duration: {depth_stats['duration_hours_mean']:.2f} hours ({depth_stats['duration_hours_mean']/24:.2f} days)\n"
                summary_text += f"  Median Duration: {depth_stats['duration_hours_median']:.2f} hours\n"
        summary_text += "\n"
    
    # Add seasonal statistics
    if 'by_season' in stats:
        summary_text += "SEASONAL STATISTICS\n"
        summary_text += "------------------\n"
        for season, season_stats in stats['by_season'].items():
            if season in stats['season_distribution']:
                summary_text += f"{season}:\n"
                summary_text += f"  Events: {season_stats['duration_hours_count']} ({stats['season_distribution'][season]:.1f}%)\n"
                summary_text += f"  Sites: {season_stats['source_nunique']}\n"
                summary_text += f"  Mean Duration: {season_stats['duration_hours_mean']:.2f} hours ({season_stats['duration_hours_mean']/24:.2f} days)\n"
                summary_text += f"  Median Duration: {season_stats['duration_hours_median']:.2f} hours\n"
        summary_text += "\n"
    
    # Add soil temperature statistics if available
    if 'soil_temperature' in stats:
        summary_text += "SOIL TEMPERATURE STATISTICS\n"
        summary_text += "-------------------------\n"
        summary_text += f"Mean: {stats['soil_temperature']['mean']:.4f}°C\n"
        summary_text += f"Median: {stats['soil_temperature']['median']:.4f}°C\n"
        summary_text += f"Standard Deviation: {stats['soil_temperature']['std']:.4f}°C\n"
        summary_text += f"Range: {stats['soil_temperature']['min']:.4f} - {stats['soil_temperature']['max']:.4f}°C\n\n"
    
    # Add soil moisture statistics if available
    if 'soil_moisture' in stats:
        summary_text += "SOIL MOISTURE STATISTICS\n"
        summary_text += "-----------------------\n"
        summary_text += f"Data Availability: {stats['soil_moisture']['data_availability']:.1f}% of events\n"
        summary_text += f"Mean: {stats['soil_moisture']['mean']:.4f}\n"
        summary_text += f"Median: {stats['soil_moisture']['median']:.4f}\n"
        summary_text += f"Standard Deviation: {stats['soil_moisture']['std']:.4f}\n"
        summary_text += f"Range: {stats['soil_moisture']['min']:.4f} - {stats['soil_moisture']['max']:.4f}\n"
        if 'mean_change' in stats['soil_moisture']:
            summary_text += f"Mean Moisture Change During Events: {stats['soil_moisture']['mean_change']:.4f}\n"
        summary_text += "\n"
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(summary_text)
        print(f"Statistics saved to {output_file}")
    
    # Print summary
    print(summary_text)
    
    return stats

stats = compute_zero_curtain_statistics(
    enhanced_events, 
    output_file='zero_curtain_pipeline/zero_curtain_statistics.txt'
)

print(f"Median duration: {stats['duration']['median']} hours")
print(f"Events in the Arctic: {stats['by_region']['Arctic']['duration_hours_count']}")
print(f"Mean duration at shallow depths: {stats['by_depth']['shallow']['duration_hours_mean']}")

fig, stats = create_final_visualization(
    enhanced_events,
    'zero_curtain_pipeline/enhanced_zero_curtain_visualization.png'
)



plot_temporal_patterns(enhanced_events)

season_order = pd.CategoricalDtype(categories=['Winter', 'Spring', 'Summer', 'Fall'], ordered=True)
events_season = enhanced_events.copy()
events_season['season'] = events_season['season'].astype(season_order)

events_season = enhanced_events.copy()
events_season['season'] = pd.Categorical(
events_season['season'], 
categories=['Winter', 'Spring', 'Summer', 'Fall'],
ordered=True
)

depth_season_temp = events_season.pivot_table(
values='soil_temp_mean', 
index='soil_temp_depth_zone', 
columns='season', 
aggfunc='mean'
)

depth_order = ['shallow', 'intermediate', 'deep', 'very_deep']
depth_season_temp = depth_season_temp.reindex(depth_order)

#Modified code above because 'Other' was designated for latitudes >= 50 but in reality, we already f...
#enhanced_events[enhanced_events.region=='Other'].region

enhanced_events.loc[enhanced_events.region == 'Other', 'region'] = 'Northern Boreal'

enhanced_events.region.unique()

# visualizations = create_zero_curtain_analysis(enhanced_events)





enhanced_events = pd.read_csv('/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/zero_curtain_events_sorted.csv')
enhanced_events.datetime_min = pd.to_datetime(enhanced_events.datetime_min,format='mixed')
enhanced_events.datetime_max = pd.to_datetime(enhanced_events.datetime_max,format='mixed')

merged_df = pd.read_feather('merged_compressed.feather')

enhanced_events

create_visualization_suite(enhanced_events, output_dir='zero_curtain_pipeline/figures/zero_curtain')











print("Testing Metal GPU with a simple operation...")
with tf.device('/GPU:0'):
    start_time = time.time()
    a = tf.random.normal((1000, 1000))
    b = tf.random.normal((1000, 1000))
    c = tf.matmul(a, b)
    # Force execution
    result = c.numpy()
    execution_time = time.time() - start_time
    print(f"Test completed in {execution_time:.4f}s")

tf.config.list_physical_devices('GPU')

diagnose_tf_environment()

tf.config.optimizer.set_jit(True)  # Enable XLA

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')



#low = pd.to_datetime(events_df.datetime_min, format='mixed').min(); high = pd.to_datetime(events_df...
#high-low

# len(pd.to_datetime(events_df['datetime_min'],format='mixed')[
# pd.to_datetime(events_df['datetime_min'],format='mixed').dt.strftime('%Y') == '1909'])

# pd.to_datetime(events_df['datetime_min'],format='mixed')[
# pd.to_datetime(events_df['datetime_min'],format='mixed').dt.strftime('%Y') == '1907']

#21397132*.66666
#training data: 14264612.01912

# model.fit(
#     X_train, y_train,
#     sample_weight=weights,  # Use the density-based weights
#     validation_data=(X_val, y_val),
# )



# events_df = pd.read_csv('zero_curtain_pipeline/zero_curtain_events.csv', parse_dates=['datetime_mi...
# print(f"Loaded {len(events_df)} events from existing file")



merged_df

result = prepare_data_for_deep_learning_efficiently(
    feather_path='merged_compressed.feather',
    events_df=pd.read_csv('zero_curtain_pipeline/zero_curtain_events.csv', parse_dates=['datetime_min', 'datetime_max']),
    sequence_length=6,
    output_dir='zero_curtain_pipeline/modeling/ml_data',
    batch_size=50
)

# #If the code is interrupted above, use the following code to pick up where it left off:
# import os
# import numpy as np
# import pandas as pd
# import glob

# # Load the original events dataframe
# events_df = pd.read_csv('zero_curtain_pipeline/zero_curtain_events.csv', parse_dates=['datetime_mi...
# print(f"Loaded {len(events_df)} events from existing file")

# # Output directory where batch files are stored
# output_dir = 'zero_curtain_pipeline/modeling/ml_data'

# # Find the highest batch that was already processed
# batch_files = glob.glob(os.path.join(output_dir, 'X_batch_*_*.npy'))
# if batch_files:
#     # Extract batch numbers from filenames
#     batch_ends = [int(f.split('_')[-1].split('.')[0]) for f in batch_files]
#     last_processed_batch = max(batch_ends)
#     start_batch = last_processed_batch
#     print(f"Found {len(batch_files)} processed batches, resuming from batch {start_batch}")
# else:
#     # No batches found, start from the beginning
#     start_batch = 0
#     print("No processed batches found, starting from the beginning")

# # Continue data preparation from where it left off
# batch_size = 50  # Same as before
# X, y, metadata = prepare_data_for_deep_learning_efficiently(
#     feather_path='merged_compressed.feather',
#     events_df=events_df,
#     sequence_length=6,
#     output_dir=output_dir,
#     batch_size=batch_size,
#     start_batch=start_batch  # Add this parameter to your function
# )

merge_result = merge_batch_files('zero_curtain_pipeline/modeling/ml_data')



# from tqdm import tqdm
# import numpy as np
# from sklearn.neighbors import KernelDensity
# from sklearn.preprocessing import StandardScaler

# def spatiotemporal_train_test_split(X, y, metadata, test_fraction=0.2, val_fraction=0.15):
#     """
#     Spatiotemporally-aware data splitting for Earth science applications.
#     """
    
#     # Extract spatiotemporal features
#     timestamps = np.array([meta['start_time'] for meta in metadata])
    
#     latitudes = np.array([meta.get('latitude', 0) if meta.get('latitude') is not None else 0 for m...
#     longitudes = np.array([meta.get('longitude', 0) if meta.get('longitude') is not None else 0 fo...
#     depths = np.array([meta.get('soil_temp_depth', 0) if meta.get('soil_temp_depth') is not None e...
    
#     has_geo_info = (np.count_nonzero(latitudes) > 0 and np.count_nonzero(longitudes) > 0)
    
#     # Temporal ordering - sort everything by time
#     sorted_time_indices = np.argsort(timestamps)
#     timestamps = timestamps[sorted_time_indices]
#     latitudes = latitudes[sorted_time_indices]
#     longitudes = longitudes[sorted_time_indices]
#     depths = depths[sorted_time_indices]
#     X = X[sorted_time_indices]
#     y = y[sorted_time_indices]
    
#     # Reserve the most recent data as a true test set
#     n_samples = len(X)
#     test_start = int(n_samples * (1 - test_fraction))
    
#     X_test = X[test_start:]
#     y_test = y[test_start:]
#     test_metadata = [metadata[i] for i in sorted_time_indices[test_start:]]
    
#     # Remaining data for training/validation
#     train_val_indices = sorted_time_indices[:test_start]
#     X_remaining = X[:test_start]
#     y_remaining = y[:test_start]
#     latitudes_remaining = latitudes[:test_start]
#     longitudes_remaining = longitudes[:test_start]
#     depths_remaining = depths[:test_start]
    
#     # Compute spatial density
#     if has_geo_info:
#         lat_lon_points = np.radians(np.vstack([latitudes_remaining, longitudes_remaining]).T)
#         geo_kde = KernelDensity(bandwidth=0.05, metric='haversine')
#         geo_kde.fit(lat_lon_points)
#         geo_log_density = geo_kde.score_samples(lat_lon_points)
#         geo_density = np.exp(geo_log_density)
        
#         depth_kde = KernelDensity(bandwidth=0.5, metric='euclidean')
#         depth_kde.fit(depths_remaining.reshape(-1, 1))
#         depth_log_density = depth_kde.score_samples(depths_remaining.reshape(-1, 1))
#         depth_density = np.exp(depth_log_density)
        
#         density = geo_density * depth_density
#     else:
#         depth_kde = KernelDensity(bandwidth=0.5, metric='euclidean')
#         depth_kde.fit(depths_remaining.reshape(-1, 1))
#         depth_log_density = depth_kde.score_samples(depths_remaining.reshape(-1, 1))
#         density = np.exp(depth_log_density)
    
#     weights = 1.0 / (density + 0.01)
#     weights = weights / np.sum(weights) * len(weights)
    
#     months = np.array([ts.month if hasattr(ts, 'month') else ts.to_pydatetime().month for ts in ti...
#     seasons = np.digitize(months, bins=[3, 6, 9, 12])
    
#     if has_geo_info:
#         regions = np.zeros_like(latitudes_remaining, dtype=int)
#         regions[(latitudes_remaining >= 66.5)] = 3
#         regions[(latitudes_remaining >= 60) & (latitudes_remaining < 66.5)] = 2
#         regions[(latitudes_remaining >= 50) & (latitudes_remaining < 60)] = 1
#     else:
#         regions = np.zeros_like(depths_remaining, dtype=int)
    
#     depth_zones = np.digitize(depths_remaining, bins=[0.2, 0.5, 1.0, 2.0])
#     density_quantiles = np.digitize(density, bins=np.percentile(density, [20, 40, 60, 80]))
#     strata = seasons * 1000 + regions * 100 + depth_zones * 10 + density_quantiles
#     unique_strata = np.unique(strata)
    
#     val_size = int(n_samples * val_fraction)
#     train_size = test_start - val_size
    
#     train_indices = []
#     val_indices = []
    
#     for stratum in tqdm(unique_strata, desc='Stratified Sampling'):
#         stratum_indices = np.where(strata == stratum)[0]
        
#         if len(stratum_indices) == 0:
#             continue
        
#         stratum_weights = weights[stratum_indices]
#         stratum_weight_sum = np.sum(stratum_weights)
#         target_val_size = int(val_size * stratum_weight_sum / np.sum(weights))
#         target_val_size = max(target_val_size, 1)
#         target_val_size = min(target_val_size, len(stratum_indices) - 1)
        
#         stratum_val_indices = stratum_indices[-target_val_size:]
#         stratum_train_indices = stratum_indices[:-target_val_size]
        
#         train_indices.extend(stratum_train_indices)
#         val_indices.extend(stratum_val_indices)
    
#     X_train = X_remaining[train_indices]
#     y_train = y_remaining[train_indices]
#     train_metadata = [metadata[i] for i in train_val_indices[train_indices]]
    
#     X_val = X_remaining[val_indices]
#     y_val = y_remaining[val_indices]
#     val_metadata = [metadata[i] for i in train_val_indices[val_indices]]
    
#     print(f"Split sizes: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
#     return X_train, X_val, X_test, y_train, y_val, y_test, weights[train_indices]

#X #(21397132, 6, 3)
#y.shape #(21397132,)
#len(metadata) #21397132

# # Check environment
# print(f"Checkpoint directory exists: {os.path.exists(CHECKPOINT_DIR)}")
# print(f"Checkpoint directory is writable: {os.access(CHECKPOINT_DIR, os.W_OK)}")

# # Determine optimal number of jobs based on available CPU cores and memory
# n_cpus = os.cpu_count()
# available_memory_gb = psutil.virtual_memory().available / (1024*1024*1024)
# recommended_jobs = max(1, min(n_cpus - 1, int(available_memory_gb / 8)))  # More conservative
# print(f"Available CPUs: {n_cpus}, Available memory: {available_memory_gb:.2f} GB")
# print(f"Recommended parallel jobs: {recommended_jobs}")

# # Process the data with improved parameters
# distances = process_spatial_data(
#     latitudes=latitudes,
#     longitudes=longitudes,
#     k=5,
#     leaf_size=40,
#     batch_size=100000,  # Much larger batch size for fewer batches
#     n_jobs=recommended_jobs  # Use parallel processing
# )

# print(f"Processed {len(distances)} points with {distances.shape[1]} nearest neighbors each")
# print(f"Average nearest neighbor distance: {np.mean(distances[:, 1]):.6f} radians")





# WELL LETS TRY THIS AGAIN

output_base_dir = '/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling'
data_dir = os.path.join(output_base_dir, 'ml_data')
X = np.load(os.path.join(data_dir, 'X_features.npy'))
y = np.load(os.path.join(data_dir, 'y_labels.npy'))
with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)
# y labels are binary, i.e., 0 is False for has_moisture_data and 1 is True for has_moisture_data

#OLD WAY OF SPLITTING; WE ARE LOOKING INTO SPATIOTEMPORAL SPLITTING NOW
# from tqdm import tqdm
# from sklearn.neighbors import KernelDensity
# from sklearn.preprocessing import StandardScaler

# test_fraction=0.2
# val_fraction=0.15
# timestamps = np.array([meta['start_time'] for meta in metadata])

# latitudes = np.array([meta.get('latitude', 0) if meta.get('latitude') is not None else 0 for meta ...
# longitudes = np.array([meta.get('longitude', 0) if meta.get('longitude') is not None else 0 for me...
# depths = np.array([meta.get('soil_temp_depth', 0) if meta.get('soil_temp_depth') is not None else ...
# has_geo_info = (np.count_nonzero(latitudes) > 0 and np.count_nonzero(longitudes) > 0)
# sorted_time_indices = np.argsort(timestamps, kind='mergesort')
# timestamps = timestamps[sorted_time_indices]
# latitudes = latitudes[sorted_time_indices]
# longitudes = longitudes[sorted_time_indices]
# depths = depths[sorted_time_indices]
# X = X[sorted_time_indices]
# y = y[sorted_time_indices]

# n_samples = len(X)
# test_start = int(n_samples * (1 - test_fraction))

# X_test = X[test_start:]
# y_test = y[test_start:]
# test_metadata = [metadata[i] for i in sorted_time_indices[test_start:]]

# train_val_indices = sorted_time_indices[:test_start]
# X_remaining = X[:test_start]
# y_remaining = y[:test_start]
# latitudes_remaining = latitudes[:test_start]
# longitudes_remaining = longitudes[:test_start]
# depths_remaining = depths[:test_start]

# Determine optimal number of jobs based on available CPU cores and memory
n_cpus = os.cpu_count()
available_memory_gb = psutil.virtual_memory().available / (1024*1024*1024)
recommended_jobs = max(1, min(n_cpus - 1, int(available_memory_gb / 8)))  # More conservative
print(f"Available CPUs: {n_cpus}, Available memory: {available_memory_gb:.2f} GB")
print(f"Recommended parallel jobs: {recommended_jobs}")

densities, weights = calculate_spatial_density(
    latitudes=latitudes, 
    longitudes=longitudes, 
    k=5,
    leaf_size=40,
    batch_size=100000,
    n_jobs=recommended_jobs
    )
print(f"Average density: {np.mean(densities):.6f}")
print(f"Average weight: {np.mean(weights):.6f}")

train_indices, val_indices, test_indices = stratified_spatiotemporal_split(
    X=X, 
    y=y, 
    metadata=metadata,
    test_size=0.2, 
    val_size=0.15
)

with open("zero_curtain_pipeline/modeling/checkpoints/spatiotemporal_split.pkl","rb") as f:
    split = pickle.load(f)

#split.get('train_indices')
#split.get('valid_indices')
#split.get('test_indices')

with open("zero_curtain_pipeline/modeling/checkpoints/spatial_density.pkl","rb") as f:
    spatialdensity = pickle.load(f)

#spatialdensity.get('density')
#spatialdensity.get('weights')



# ENABLE GPU
# # Try explicit memory limiting:
# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
#     # Limit to 4GB
#     tf.config.set_logical_device_configuration(
#         gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
# import tensorflow

# Add at the beginning of your code
def configure_tensorflow_memory():
    """Configure TensorFlow to use memory growth and limit GPU memory allocation"""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                # Allow memory growth - prevents TF from allocating all GPU memory at once
                tf.config.experimental.set_memory_growth(device, True)
                
                # Optional: Set memory limit (e.g., 4GB)
                # tf.config.experimental.set_virtual_device_configuration(
                #     device,
                #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                # )
                print(f"Memory growth enabled for {device}")
            except Exception as e:
                print(f"Error configuring GPU: {e}")
    
    # Limit CPU threads
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    
    # Set soft device placement
    tf.config.set_soft_device_placement(True)

configure_tensorflow_memory()

# Load split indices
with open("zero_curtain_pipeline/modeling/checkpoints/spatiotemporal_split.pkl", "rb") as f:
    split_data = pickle.load(f)
train_indices = split_data["train_indices"]
val_indices = split_data["val_indices"]
test_indices = split_data["test_indices"]

# Load spatial weights
with open("zero_curtain_pipeline/modeling/checkpoints/spatial_density.pkl", "rb") as f:
    weights_data = pickle.load(f)

data_dir = os.path.join('/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling', 'ml_data')

# Initialize generators
X_file = os.path.join(data_dir, 'X_features.npy')
y_file = os.path.join(data_dir, 'y_labels.npy')

# Get sample weights for training set
sample_weights = weights_data["weights"][train_indices]
sample_weights = sample_weights / np.mean(sample_weights) * len(sample_weights)

# Load data and metadata; use memory mapping to reduce memory usage
print("Loading data...")
X = np.load(X_file, mmap_mode='r')
y = np.load(y_file, mmap_mode='r')
print("Loading metadata...")
with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)
print("Done.")

train_y = y[train_indices]
val_y = y[val_indices]
test_y = y[test_indices]

print(f"Train/val/test sizes: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")
print(f"Positive examples: Train={np.sum(train_y)} ({np.sum(train_y)/len(train_y)*100:.1f}%), " +
      f"Val={np.sum(val_y)} ({np.sum(val_y)/len(val_y)*100:.1f}%), " +
      f"Test={np.sum(test_y)} ({np.sum(test_y)/len(test_y)*100:.1f}%)")

# model.summary()

'/Users/bradleygay/new2/'

# # Create generators
# train_gen = DataGenerator(X, y, train_indices, batch_size=1024, shuffle=True, weights=sample_weigh...
# val_gen = DataGenerator(X, y, val_indices, batch_size=1024, shuffle=False)
# test_gen = DataGenerator(X, y, test_indices, batch_size=1024, shuffle=False)

# # Create output signatures
# features_signature = tf.TensorSpec(shape=input_shape, dtype=tf.float32)
# label_signature = tf.TensorSpec(shape=(), dtype=tf.int32)
# weight_signature = tf.TensorSpec(shape=(), dtype=tf.float32)

# # Create datasets from existing generators
# train_ds = create_tf_dataset_from_generator(
#     train_gen,
#     output_signature=(features_signature, label_signature, weight_signature)
# )

# val_ds = create_tf_dataset_from_generator(
#     val_gen,
#     output_signature=(features_signature, label_signature)
# )

# test_ds = create_tf_dataset_from_generator(
#     test_gen,
#     output_signature=(features_signature, label_signature)
# )



#LETS CREATE A SAMPLE FIRST
################################################
################################################
################################################
################################################

# When creating subset indices
subset_indices = np.random.choice(train_indices, subset_size, replace=False)

# Get only the weights for the subset
subset_weights = np.array([sample_weights[np.where(train_indices == idx)[0][0]] 
                          for idx in subset_indices])

# Create generator with the subset and its corresponding weights
train_gen = DataGenerator(X, y, subset_indices, batch_size=512, shuffle=True, weights=subset_weights)

subset_fraction = 0.001
subset_size = int(len(val_indices) * subset_fraction)
subset_indices = np.random.choice(val_indices, subset_size, replace=False)
val_gen = DataGenerator(X, y, subset_indices, batch_size=1024, shuffle=False)

subset_fraction = 0.001
subset_size = int(len(test_indices) * subset_fraction)
subset_indices = np.random.choice(test_indices, subset_size, replace=False)
test_gen = DataGenerator(X, y, test_indices, batch_size=1024, shuffle=False)

# Create datasets from existing generators - specify if weights are expected
train_ds = create_tf_dataset_from_generator(
    train_gen,
    output_signature=(features_signature, label_signature, weight_signature),
    has_weights=True  # Train generator has weights
)

val_ds = create_tf_dataset_from_generator(
    val_gen,
    output_signature=(features_signature, label_signature),
    has_weights=False  # Validation generator doesn't have weights
)

test_ds = create_tf_dataset_from_generator(
    test_gen,
    output_signature=(features_signature, label_signature),
    has_weights=False  # Test generator doesn't have weights
)

# Train model
print("Starting model training...")
epochs = 100

# Note: class_weight is handled by the sample weights in train_gen
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# Clean up to free memory
del train_ds, val_ds
gc.collect()

# Evaluate on test set in batches
print("Evaluating model on test set...")
evaluation = model.evaluate(test_gen, verbose=1)

print("Test performance:")
for metric, value in zip(model.metrics_names, evaluation):
    print(f"  {metric}: {value:.4f}")

# Generate predictions in batches
print("Generating predictions...")

all_preds = []
all_true = []

for i in range(len(test_gen)):
    X_batch, y_batch = test_gen[i]
    pred_batch = model.predict(X_batch, verbose=0)
    all_preds.append(pred_batch)
    all_true.append(y_batch)

# Calculate additional evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc as sk_auc

report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Classification Report:")
print(report)

print("Confusion Matrix:")
print(conf_matrix)

################################################
################################################
################################################
################################################

model.summary()

# train_gen = OptimizedDataGenerator(X_file, y_file, train_indices, 
#                                   batch_size=256, weights=sample_weights)
# val_gen = OptimizedDataGenerator(X_file, y_file, val_indices, 
#                                 batch_size=256)

# # Create output signatures
# features_signature = tf.TensorSpec(shape=input_shape, dtype=tf.float32)
# label_signature = tf.TensorSpec(shape=(), dtype=tf.int32)
# weight_signature = tf.TensorSpec(shape=(), dtype=tf.float32)

# # Create datasets from existing generators
# train_ds = create_tf_dataset_from_generator(
#     train_gen,
#     output_signature=(features_signature, label_signature, weight_signature)
# )

# val_ds = create_tf_dataset_from_generator(
#     val_gen,
#     output_signature=(features_signature, label_signature)
# )

# test_ds = create_tf_dataset_from_generator(
#     test_gen,
#     output_signature=(features_signature, label_signature)
# )

# # Train model
# print("Starting model training...")
# epochs = 100

# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=epochs,
#     callbacks=callbacks,
#     class_weight=class_weight,
#     verbose=1,
#     use_multiprocessing=False,
#     workers=1  # Reduce parallel processing
# )

# # Clean up to free memory
# del train_gen, val_gen
# gc.collect()

def create_optimized_tf_dataset(X_file, y_file, indices, batch_size=256, shuffle=True, 
                               weights=None, cache=False, prefetch_factor=tf.data.AUTOTUNE):
    """
    Create an optimized TensorFlow dataset with detailed progress reporting.
    """
    import time
    
    # Log start time for loading data
    load_start = time.time()
    print(f"  Loading memory-mapped arrays...")
    
    # Load as memory-mapped arrays
    X = np.load(X_file, mmap_mode='r')
    y = np.load(y_file, mmap_mode='r')
    
    print(f"  Arrays loaded in {time.time() - load_start:.2f} seconds")
    
    # Get input shape from first sample
    sample_start = time.time()
    input_shape = X[indices[0]].shape
    print(f"  Input shape: {input_shape}, obtained in {time.time() - sample_start:.2f} seconds")
    
    # Define generator function with progress reporting
    def generator():
        total = len(indices)
        start_time = time.time()
        last_report = start_time
        
        for i, idx in enumerate(indices):
            # Report progress every 10000 samples or 10 seconds
            current_time = time.time()
            if i % 10000 == 0 or current_time - last_report > 10:
                elapsed = current_time - start_time
                if i > 0:
                    rate = i / elapsed
                    eta = (total - i) / rate if rate > 0 else 0
                    print(f"  Generator progress: {i}/{total} ({i/total*100:.1f}%), "
                          f"Rate: {rate:.1f} samples/sec, ETA: {int(eta)} seconds")
                last_report = current_time
            
            if weights is not None:
                # Find position of idx in original indices array
                pos = np.where(indices == idx)[0][0]
                yield X[idx], y[idx], weights[pos]
            else:
                yield X[idx], y[idx]
    
    # Create dataset from generator
    create_start = time.time()
    print(f"  Creating dataset from generator...")
    
    if weights is not None:
        output_signature = (
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    print(f"  Dataset created in {time.time() - create_start:.2f} seconds")
    
    # Apply dataset optimizations
    print(f"  Applying dataset optimizations...")
    opt_start = time.time()
    
    if shuffle:
        buffer_size = min(len(indices), 10000)
        print(f"  Shuffling with buffer size {buffer_size}...")
        dataset = dataset.shuffle(buffer_size)
    
    print(f"  Batching with size {batch_size}...")
    dataset = dataset.batch(batch_size)
    
    if cache:
        print(f"  Caching dataset...")
        dataset = dataset.cache()
    
    print(f"  Setting prefetch to {prefetch_factor}...")
    dataset = dataset.prefetch(prefetch_factor)
    
    print(f"  Optimizations applied in {time.time() - opt_start:.2f} seconds")
    
    return dataset

# Configuration
# Configure TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
        except Exception as e:
            print(f"Error configuring GPU: {e}")

# Get input shape for the model
input_shape = X[train_indices[0]].shape
print(f"Input shape: {input_shape}")

# Build model
model = build_advanced_zero_curtain_model(input_shape)

model.summary()



################################################################################
################################################################################
################################################################################
################################################################################

# # For initial run:
# model, final_model_path = resumable_efficient_training(
#     model, X_file, y_file,
#     train_indices, val_indices, test_indices,
#     output_dir=output_dir,
#     batch_size=256,
#     chunk_size=25000,
#     epochs_per_chunk=2,
#     save_frequency=5,
#     class_weight=class_weight,
#     start_chunk=0  # Start from beginning
# )

# To resume after a freeze (e.g., after chunk #):
model, final_model_path = resumable_efficient_training(
    model, X_file, y_file,
    train_indices, val_indices, test_indices,
    output_dir=output_dir,
    batch_size=256,
    chunk_size=25000,
    epochs_per_chunk=2,
    save_frequency=5,
    class_weight=class_weight,
    #start_chunk=360
    #start_chunk=405
    #start_chunk=450
    #start_chunk=495
    start_chunk=540
)

################################################################################
################################################################################
################################################################################
################################################################################

def combine_chunk_predictions(output_dir, train_indices):
    """Combine predictions from all chunks into a single array"""
    import os
    import numpy as np
    
    predictions_dir = os.path.join(output_dir, "predictions")
    
    # Get all prediction files
    all_files = os.listdir(predictions_dir)
    pred_files = [f for f in all_files if f.startswith('chunk_') and f.endswith('_predictions.npy')]
    indices_files = [f for f in all_files if f.startswith('chunk_') and f.endswith('_indices.npy')]
    
    # Sort by chunk number
    pred_files.sort(key=lambda x: int(x.split('_')[1]))
    indices_files.sort(key=lambda x: int(x.split('_')[1]))
    
    # Create full predictions array
    all_predictions = np.zeros(len(train_indices))
    covered_indices = np.zeros(len(train_indices), dtype=bool)
    
    # Load and map predictions
    print(f"Found {len(pred_files)} prediction chunks")
    
    for pred_file, idx_file in zip(pred_files, indices_files):
        # Extract chunk number for reporting
        chunk_num = int(pred_file.split('_')[1])
        
        # Load predictions and indices
        chunk_preds = np.load(os.path.join(predictions_dir, pred_file))
        chunk_indices = np.load(os.path.join(predictions_dir, idx_file))
        
        # Ensure predictions are flattened
        if len(chunk_preds.shape) > 1:
            chunk_preds = chunk_preds.flatten()
        
        # Find positions in the original array
        for i, idx in enumerate(chunk_indices):
            pos = np.where(train_indices == idx)[0]
            if len(pos) > 0:
                pos = pos[0]
                all_predictions[pos] = chunk_preds[i]
                covered_indices[pos] = True
        
        print(f"Processed chunk {chunk_num}, total covered indices: {np.sum(covered_indices)}/{len(train_indices)}")
    
    # Check for any missing predictions
    missing = len(train_indices) - np.sum(covered_indices)
    if missing > 0:
        print(f"Warning: {missing} indices have no predictions")
    
    # Save combined predictions
    np.save(os.path.join(output_dir, 'all_training_predictions.npy'), all_predictions)
    
    return all_predictions

# Define output directory
output_dir = '/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling/efficient_model'

# Data paths
data_dir = os.path.join('/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling', 'ml_data')
X_file = os.path.join(data_dir, 'X_features.npy')
y_file = os.path.join(data_dir, 'y_labels.npy')

# Load metadata for spatial analysis
metadata_file = os.path.join(data_dir, 'metadata.pkl')
with open(metadata_file, 'rb') as f:
    metadata = pickle.load(f)

# Run all analyses
print("1. Analyzing learning curves...")
learning_metrics = plot_training_history(output_dir)

print("2. Combining predictions from all chunks...")
all_predictions = combine_chunk_predictions(output_dir, train_indices)

print("3. Creating spatial visualizations...")
predictions, true_values = visualize_spatial_predictions(output_dir, train_indices, metadata, X_file, y_file)

print("4. Analyzing temporal patterns...")
temporal_df = visualize_temporal_predictions(output_dir, train_indices, metadata, all_predictions)

print("5. Analyzing feature importance...")
# Define your feature names appropriate to your model
feature_names = ["Temperature", "Precipitation", "Solar Radiation"]  # Replace with your actual feature names
importance_df = analyze_feature_importance(model, output_dir, X_file, train_indices, feature_names)

print("6. Creating comprehensive model evaluation report...")
evaluation_results = create_model_evaluation_report(output_dir, train_indices, test_indices, X_file, y_file)

print("Analysis complete! All visualizations and reports saved to:", output_dir)



















################################################################################
################################################################################
################################################################################
################################################################################

# Evaluate the model using a TF dataset
test_dataset = create_optimized_tf_dataset(
    X_file, y_file, test_indices,
    batch_size=256, shuffle=False
)





evaluation = model.evaluate(test_dataset)
print("Test performance:")
for metric, value in zip(model.metrics_names, evaluation):
    print(f"  {metric}: {value:.4f}")



# # Train with extreme memory efficiency 
# history, final_model_path = extreme_memory_efficient_training(
#     model, X_file, y_file,
#     train_indices, val_indices, test_indices,
#     output_dir=output_dir,
#     mini_batch_size=8,           # Process only 8 samples at a time in memory
#     virtual_batch_size=256,      # Accumulate gradients to simulate batch of 256
#     chunk_size=2000,             # Process 2000 samples per chunk
#     epochs_per_chunk=2,          # Train each chunk for 2 epochs
#     class_weight=class_weight
# )

# print(f"Training complete. Final model saved to: {final_model_path}")











#SAMPLE DATA WITH THE ADVANCED ARCHITECTURE FIRST - THEN WE CAN RUN THE WHOLE DATASET ABOVE
#SAMPLE DATA WITH THE ADVANCED ARCHITECTURE FIRST - THEN WE CAN RUN THE WHOLE DATASET ABOVE
#SAMPLE DATA WITH THE ADVANCED ARCHITECTURE FIRST - THEN WE CAN RUN THE WHOLE DATASET ABOVE
#SAMPLE DATA WITH THE ADVANCED ARCHITECTURE FIRST - THEN WE CAN RUN THE WHOLE DATASET ABOVE

subset_fraction = 0.001
subset_size = int(len(train_indices) * subset_fraction)

# When creating subset indices
subset_indices = np.random.choice(train_indices, subset_size, replace=False)

# Get only the weights for the subset
subset_weights = np.array([sample_weights[np.where(train_indices == idx)[0][0]] 
                          for idx in subset_indices])

# Create generator with the subset and its corresponding weights
train_gen = DataGenerator(X, y, subset_indices, batch_size=512, shuffle=True, weights=subset_weights)

subset_fraction = 0.001
subset_size = int(len(val_indices) * subset_fraction)
subset_indices = np.random.choice(val_indices, subset_size, replace=False)
val_gen = DataGenerator(X, y, subset_indices, batch_size=1024, shuffle=False)

subset_fraction = 0.001
subset_size = int(len(test_indices) * subset_fraction)
subset_indices = np.random.choice(test_indices, subset_size, replace=False)
test_gen = DataGenerator(X, y, test_indices, batch_size=1024, shuffle=False)

# Create datasets from existing generators - specify if weights are expected
train_ds = create_tf_dataset_from_generator(
    train_gen,
    output_signature=(features_signature, label_signature, weight_signature),
    has_weights=True  # Train generator has weights
)

val_ds = create_tf_dataset_from_generator(
    val_gen,
    output_signature=(features_signature, label_signature),
    has_weights=False  # Validation generator doesn't have weights
)

test_ds = create_tf_dataset_from_generator(
    test_gen,
    output_signature=(features_signature, label_signature),
    has_weights=False  # Test generator doesn't have weights
)

# Train model
print("Starting model training...")
epochs = 100

# Note: class_weight is handled by the sample weights in train_gen
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# Clean up to free memory
del train_ds, val_ds
gc.collect()

# Evaluate on test set in batches
print("Evaluating model on test set...")
evaluation = model.evaluate(test_gen, verbose=1)

print("Test performance:")
for metric, value in zip(model.metrics_names, evaluation):
    print(f"  {metric}: {value:.4f}")

# Generate predictions in batches
print("Generating predictions...")

all_preds = []
all_true = []

for i in range(len(test_gen)):
    X_batch, y_batch = test_gen[i]
    pred_batch = model.predict(X_batch, verbose=0)
    all_preds.append(pred_batch)
    all_true.append(y_batch)

# Calculate additional evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc as sk_auc

report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Classification Report:")
print(report)

print("Confusion Matrix:")
print(conf_matrix)





# Start timing
start_time = time.time()

# 1. Create an efficient lookup structure for physical events
print("Creating efficient lookup structure...")
lookup_start = time.time()

# Create a dictionary with (site, depth) as key
# Each value is a sorted list of (start_time, end_time) tuples
physical_lookup = {}

for _, event in physical_events.iterrows():
    key = (event['source'], event['soil_temp_depth'])
    
    if key not in physical_lookup:
        physical_lookup[key] = []
    
    # Make sure timestamps are pandas Timestamps
    start = pd.Timestamp(event['datetime_min'])
    end = pd.Timestamp(event['datetime_max'])
    
    physical_lookup[key].append((start, end))

# Sort each list by start time for binary search later
for key in physical_lookup:
    physical_lookup[key].sort(key=lambda x: x[0])

print(f"Created lookup for {len(physical_lookup)} site-depth combinations")
print(f"Lookup creation time: {time.time() - lookup_start:.2f} seconds")

# 2. Function to efficiently find overlapping events using binary search
def find_overlapping_event(site, depth, start_time, end_time, min_overlap_ratio=0.5):
    """
    Find if a sequence overlaps with any physical event using binary search
    for more efficient lookup.
    """
    if (site, depth) not in physical_lookup:
        return 0
    
    events = physical_lookup[(site, depth)]
    
    # Ensure start_time and end_time are pandas Timestamps
    if not isinstance(start_time, pd.Timestamp):
        start_time = pd.Timestamp(start_time)
    if not isinstance(end_time, pd.Timestamp):
        end_time = pd.Timestamp(end_time)
    
    # Instead of binary search, use linear search for now to debug
    # This is less efficient but more reliable
    for event_start, event_end in events:
        # Calculate overlap
        if event_start <= end_time and event_end >= start_time:
            overlap_start = max(start_time, event_start)
            overlap_end = min(end_time, event_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds()
            sequence_duration = (end_time - start_time).total_seconds()
            
            # Check if overlap is significant
            if overlap_duration > min_overlap_ratio * sequence_duration:
                return 1
    
    return 0

# Start timing
start_time = time.time()

# 3. Process test sequences in batches for memory efficiency
print("Processing test sequences...")
batch_size = 10000
comparison_data = []
test_labels = np.zeros(len(test_indices), dtype=int)

processing_start = time.time()
chunks_file = 'zero_curtain_pipeline/modeling/comparison_chunks.csv'

# Clear the chunks file if it exists
if os.path.exists(chunks_file):
    os.remove(chunks_file)

for batch_start in tqdm(range(0, len(test_indices), batch_size)):
    batch_end = min(batch_start + batch_size, len(test_indices))
    batch_indices = test_indices[batch_start:batch_end]
    batch_meta = [metadata[i] for i in batch_indices]
    
    for i, meta in enumerate(batch_meta):
        global_idx = batch_start + i
        
        # Ensure datetime objects
        try:
            seq_start_time = pd.to_datetime(meta['start_time'])
            seq_end_time = pd.to_datetime(meta['end_time'])
        except Exception as e:
            print(f"Error converting timestamps: {e}")
            print(f"start_time: {meta.get('start_time')}, type: {type(meta.get('start_time'))}")
            print(f"end_time: {meta.get('end_time')}, type: {type(meta.get('end_time'))}")
            # Skip sequences with invalid timestamps
            continue
        
        # Check for overlap with physical events
        try:
            is_zero_curtain = find_overlapping_event(
                meta['source'], 
                meta['soil_temp_depth'],
                seq_start_time,
                seq_end_time
            )
        except Exception as e:
            print(f"Error finding overlap: {e}")
            # Skip in case of errors
            continue
        
        test_labels[global_idx] = is_zero_curtain
        
        # Store comparison data
        comparison_data.append({
            'source': meta['source'],
            'soil_temp_depth': meta['soil_temp_depth'],
            'start_time': seq_start_time,
            'end_time': seq_end_time,
            'latitude': meta.get('latitude'),
            'longitude': meta.get('longitude'),
            'physical_detected': is_zero_curtain,
            'model_probability': float(predictions.flatten()[global_idx]),
            'model_detected': 1 if predictions.flatten()[global_idx] > 0.5 else 0,
            'dataset': 'test'
        })
        
        # Free memory by processing in smaller chunks
        if len(comparison_data) >= 100000:
            temp_df = pd.DataFrame(comparison_data)
            temp_df.to_csv(chunks_file, 
                           mode='a', header=not os.path.exists(chunks_file),
                           index=False)
            comparison_data = []

# Save any remaining data
if comparison_data:
    temp_df = pd.DataFrame(comparison_data)
    temp_df.to_csv(chunks_file, 
                   mode='a', header=not os.path.exists(chunks_file),
                   index=False)

print(f"Processing time: {time.time() - processing_start:.2f} seconds")

# Start timing
start_time = time.time()

# 4. Load the saved chunks for analysis if the file exists
print("Loading comparison data for analysis...")
if os.path.exists(chunks_file):
    comparison_df = pd.read_csv(chunks_file, parse_dates=['start_time', 'end_time'])
    
    # 5. Compute metrics
    print("Computing metrics...")
    comparison_df['agreement'] = comparison_df['physical_detected'] == comparison_df['model_detected']
    comparison_df['false_positive'] = (comparison_df['model_detected'] == 1) & (comparison_df['physical_detected'] == 0)
    comparison_df['false_negative'] = (comparison_df['model_detected'] == 0) & (comparison_df['physical_detected'] == 1)
    
    # Calculate evaluation metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        comparison_df['physical_detected'], 
        comparison_df['model_detected'], 
        average='binary'
    )
    auc = roc_auc_score(comparison_df['physical_detected'], comparison_df['model_probability'])
    
    # Print summary metrics
    print(f"Sequence-level metrics on test set:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Count agreement and disagreement instances
    agreement_count = comparison_df['agreement'].sum()
    disagreement_count = len(comparison_df) - agreement_count
    agreement_percentage = 100 * agreement_count / len(comparison_df)
    
    print(f"\nAgreement analysis:")
    print(f"Total sequences: {len(comparison_df)}")
    print(f"Agreement count: {agreement_count} ({agreement_percentage:.1f}%)")
    print(f"Disagreement count: {disagreement_count} ({100-agreement_percentage:.1f}%)")
    print(f"False positives: {comparison_df['false_positive'].sum()}")
    print(f"False negatives: {comparison_df['false_negative'].sum()}")
    
    # Save the finalized comparison dataframe
    comparison_df.to_csv('zero_curtain_pipeline/modeling/sequence_comparison_final.csv', index=False)
else:
    print("No comparison data was saved, something went wrong during processing.")

print(f"Total execution time: {time.time() - start_time:.2f} seconds")

comparison_df.start_time = pd.to_datetime(comparison_df.start_time,format='mixed')

comparison_df.end_time = pd.to_datetime(comparison_df.end_time,format='mixed')

comparison_df['duration'] = [comparison_df.end_time[i]-comparison_df.start_time[i] for i in range(len(comparison_df))]

comparison_df['duration'] = [(comparison_df.duration[i].total_seconds() / 3600) for i in range(len(comparison_df))]

comparison_df2 = comparison_df[comparison_df.agreement==True][comparison_df[comparison_df.agreement==True].false_positive!=True]\
[comparison_df[comparison_df.agreement==True][comparison_df[comparison_df.agreement==True].false_positive!=True].false_negative!=True]

comparison_df2 = comparison_df2.sort_values('start_time').reset_index(drop=True)

comparison_df2.to_csv('sequence_comparison_final_nofalsepos_nofalseneg_agreement_sorted.csv',index=False)

comparison_df = comparison_df2; del comparison_df2



comparison_df





































# Start timing
start_time = time.time()

# 1. Create an efficient lookup structure for physical events
print("Creating efficient lookup structure...")
lookup_start = time.time()

# Create a dictionary with (site, depth) as key
# Each value is a sorted list of (start_time, end_time) tuples
physical_lookup = {}

for _, event in physical_events.iterrows():
    key = (event['source'], event['soil_temp_depth'])
    
    if key not in physical_lookup:
        physical_lookup[key] = []
    
    # Make sure timestamps are pandas Timestamps
    start = pd.Timestamp(event['datetime_min'])
    end = pd.Timestamp(event['datetime_max'])
    
    physical_lookup[key].append((start, end))

# Sort each list by start time for binary search later
for key in physical_lookup:
    physical_lookup[key].sort(key=lambda x: x[0])

print(f"Created lookup for {len(physical_lookup)} site-depth combinations")
print(f"Lookup creation time: {time.time() - lookup_start:.2f} seconds")

# 2. Function to efficiently find overlapping events using binary search
def find_overlapping_event(site, depth, start_time, end_time, min_overlap_ratio=0.5):
    """
    Find if a sequence overlaps with any physical event using binary search
    for more efficient lookup.
    """
    if (site, depth) not in physical_lookup:
        return 0
    
    events = physical_lookup[(site, depth)]
    
    # Ensure start_time and end_time are pandas Timestamps
    if not isinstance(start_time, pd.Timestamp):
        start_time = pd.Timestamp(start_time)
    if not isinstance(end_time, pd.Timestamp):
        end_time = pd.Timestamp(end_time)
    
    # Instead of binary search, use linear search for now to debug
    # This is less efficient but more reliable
    for event_start, event_end in events:
        # Calculate overlap
        if event_start <= end_time and event_end >= start_time:
            overlap_start = max(start_time, event_start)
            overlap_end = min(end_time, event_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds()
            sequence_duration = (end_time - start_time).total_seconds()
            
            # Check if overlap is significant
            if overlap_duration > min_overlap_ratio * sequence_duration:
                return 1
    
    return 0

# Start timing
start_time = time.time()

# 3. Process test sequences in batches for memory efficiency
print("Processing test sequences...")
batch_size = 10000
comparison_data = []
test_labels = np.zeros(len(test_indices), dtype=int)

processing_start = time.time()
chunks_file = '/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling/comparison_chunks.csv'

# Clear the chunks file if it exists
if os.path.exists(chunks_file):
    os.remove(chunks_file)

for batch_start in tqdm(range(0, len(test_indices), batch_size)):
    batch_end = min(batch_start + batch_size, len(test_indices))
    batch_indices = test_indices[batch_start:batch_end]
    batch_meta = [metadata[i] for i in batch_indices]
    
    for i, meta in enumerate(batch_meta):
        global_idx = batch_start + i
        
        # Ensure datetime objects
        try:
            seq_start_time = pd.to_datetime(meta['start_time'])
            seq_end_time = pd.to_datetime(meta['end_time'])
        except Exception as e:
            print(f"Error converting timestamps: {e}")
            print(f"start_time: {meta.get('start_time')}, type: {type(meta.get('start_time'))}")
            print(f"end_time: {meta.get('end_time')}, type: {type(meta.get('end_time'))}")
            # Skip sequences with invalid timestamps
            continue
        
        # Check for overlap with physical events
        try:
            is_zero_curtain = find_overlapping_event(
                meta['source'], 
                meta['soil_temp_depth'],
                seq_start_time,
                seq_end_time
            )
        except Exception as e:
            print(f"Error finding overlap: {e}")
            # Skip in case of errors
            continue
        
        test_labels[global_idx] = is_zero_curtain
        
        # Store comparison data
        comparison_data.append({
            'source': meta['source'],
            'soil_temp_depth': meta['soil_temp_depth'],
            'start_time': seq_start_time,
            'end_time': seq_end_time,
            'latitude': meta.get('latitude'),
            'longitude': meta.get('longitude'),
            'physical_detected': is_zero_curtain,
            'model_probability': float(predictions.flatten()[global_idx]),
            'model_detected': 1 if predictions.flatten()[global_idx] > 0.5 else 0,
            'dataset': 'test'
        })
        
        # Free memory by processing in smaller chunks
        if len(comparison_data) >= 100000:
            temp_df = pd.DataFrame(comparison_data)
            temp_df.to_csv(chunks_file, 
                           mode='a', header=not os.path.exists(chunks_file),
                           index=False)
            comparison_data = []

# Save any remaining data
if comparison_data:
    temp_df = pd.DataFrame(comparison_data)
    temp_df.to_csv(chunks_file, 
                   mode='a', header=not os.path.exists(chunks_file),
                   index=False)

print(f"Processing time: {time.time() - processing_start:.2f} seconds")

# Start timing
start_time = time.time()

# 4. Load the saved chunks for analysis if the file exists
print("Loading comparison data for analysis...")
if os.path.exists(chunks_file):
    comparison_df = pd.read_csv(chunks_file, parse_dates=['start_time', 'end_time'])
    
    # 5. Compute metrics
    print("Computing metrics...")
    comparison_df['agreement'] = comparison_df['physical_detected'] == comparison_df['model_detected']
    comparison_df['false_positive'] = (comparison_df['model_detected'] == 1) & (comparison_df['physical_detected'] == 0)
    comparison_df['false_negative'] = (comparison_df['model_detected'] == 0) & (comparison_df['physical_detected'] == 1)
    
    # Calculate evaluation metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        comparison_df['physical_detected'], 
        comparison_df['model_detected'], 
        average='binary'
    )
    auc = roc_auc_score(comparison_df['physical_detected'], comparison_df['model_probability'])
    
    # Print summary metrics
    print(f"Sequence-level metrics on test set:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Count agreement and disagreement instances
    agreement_count = comparison_df['agreement'].sum()
    disagreement_count = len(comparison_df) - agreement_count
    agreement_percentage = 100 * agreement_count / len(comparison_df)
    
    print(f"\nAgreement analysis:")
    print(f"Total sequences: {len(comparison_df)}")
    print(f"Agreement count: {agreement_count} ({agreement_percentage:.1f}%)")
    print(f"Disagreement count: {disagreement_count} ({100-agreement_percentage:.1f}%)")
    print(f"False positives: {comparison_df['false_positive'].sum()}")
    print(f"False negatives: {comparison_df['false_negative'].sum()}")
    
    # Save the finalized comparison dataframe
    comparison_df.to_csv('/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling/sequence_comparison_final.csv', index=False)
else:
    print("No comparison data was saved, something went wrong during processing.")

print(f"Total execution time: {time.time() - start_time:.2f} seconds")

comparison_df.start_time = pd.to_datetime(comparison_df.start_time,format='mixed')

comparison_df.end_time = pd.to_datetime(comparison_df.end_time,format='mixed')

comparison_df['duration'] = [comparison_df.end_time[i]-comparison_df.start_time[i] for i in range(len(comparison_df))]

comparison_df['duration'] = [(comparison_df.duration[i].total_seconds() / 3600) for i in range(len(comparison_df))]

comparison_df2 = comparison_df[comparison_df.agreement==True][comparison_df[comparison_df.agreement==True].false_positive!=True]\
[comparison_df[comparison_df.agreement==True][comparison_df[comparison_df.agreement==True].false_positive!=True].false_negative!=True]

comparison_df2 = comparison_df2.sort_values('start_time').reset_index(drop=True)

comparison_df2.to_csv('/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling/sequence_comparison_final_nofalsepos_nofalseneg_agreement_sorted.csv',index=False)

comparison_df = comparison_df2; del comparison_df2









# Example: Analyze disagreement patterns by depth
depth_analysis = comparison_df.groupby('soil_temp_depth').agg({
    'agreement': 'mean',
    'false_positive': 'sum',
    'false_negative': 'sum',
    'physical_detected': 'sum',
    'model_detected': 'sum'
}).reset_index()

# Example: Analyze seasonal patterns in disagreement
comparison_df['month'] = comparison_df['start_time'].dt.month
seasonal_analysis = comparison_df.groupby('month').agg({
    'agreement': 'mean',
    'false_positive': 'sum', 
    'false_negative': 'sum',
    'physical_detected': 'sum',
    'model_detected': 'sum'
}).reset_index()

# Statistical test for spatial clustering of disagreements
from scipy.stats import moran

# Group disagreements by location and test for spatial autocorrelation
# (implementation would depend on spatial distribution of your data)

























model.summary()

# Create generators
train_gen = DataGenerator(X, y, train_indices, batch_size=256, shuffle=True, weights=sample_weights)
val_gen = DataGenerator(X, y, val_indices, batch_size=256, shuffle=False)
test_gen = DataGenerator(X, y, test_indices, batch_size=256, shuffle=False)

print(train_gen.X[train_gen.indices[0]].shape)
print(input_shape)

# Train model
print("Starting model training...")
epochs = 100

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1,
    use_multiprocessing=False,
    workers=1  # Reduce parallel processing
)

# Clean up to free memory
del train_gen, val_gen
gc.collect()

























# Train model with spatial balancing
output_dir = os.path.join('/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling', 'spatial_model')
os.makedirs(output_dir, exist_ok=True)

model, history, evaluation = train_with_spatial_balancing(
    X=X,
    y=y,
    metadata=metadata,
    output_dir=output_dir
)

print("Training complete!")

print(f"X shape (features): {X.shape}, y shape (labels): {y.shape}, metadata: {len(metadata)}")

input_shape = (X.shape[1], X.shape[2])
model = build_advanced_zero_curtain_model(input_shape)
model.summary()



os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Train with spatial balancing using the pre-calculated weights
output_dir = '/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling/spatial_model'
checkpoint_dir = 'zero_curtain_pipeline/modeling/checkpoints'  # Where your spatial density weights are

batch_size = 8

model, history, evaluation = train_with_spatial_balancing(
    X=X,
    y=y,
    metadata=metadata,
    output_dir=output_dir,
    checkpoint_dir=checkpoint_dir
)








data_dir = '/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling/ml_data'
X = np.load(os.path.join(data_dir, 'X_features.npy'))
y = np.load(os.path.join(data_dir, 'y_labels.npy'))
with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)

# Train with spatial balancing using the pre-calculated weights
#output_dir = '/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling/spatial_model'
output_dir = os.path.join('/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling', 'spatial_model')
os.makedirs(output_dir, exist_ok=True)
checkpoint_dir = 'zero_curtain_pipeline/modeling/checkpoints' #spatial density weights are here

model, history, evaluation = train_with_spatial_balancing(
    X=X,
    y=y,
    metadata=metadata,
    output_dir=output_dir,
    checkpoint_dir=checkpoint_dir
)





























# # Compute spatial density
# if has_geo_info:
#     lat_lon_points = np.radians(np.vstack([latitudes_remaining, longitudes_remaining]).T)
#     geo_kde = KernelDensity(bandwidth=0.05, metric='haversine')
#     geo_kde.fit(lat_lon_points)
#     geo_log_density = geo_kde.score_samples(lat_lon_points)
#     geo_density = np.exp(geo_log_density)
    
#     depth_kde = KernelDensity(bandwidth=0.5, metric='euclidean')
#     depth_kde.fit(depths_remaining.reshape(-1, 1))
#     depth_log_density = depth_kde.score_samples(depths_remaining.reshape(-1, 1))
#     depth_density = np.exp(depth_log_density)
    
#     density = geo_density * depth_density
# else:
#     depth_kde = KernelDensity(bandwidth=0.5, metric='euclidean')
#     depth_kde.fit(depths_remaining.reshape(-1, 1))
#     depth_log_density = depth_kde.score_samples(depths_remaining.reshape(-1, 1))
#     density = np.exp(depth_log_density)

# from sklearn.neighbors import BallTree
# tree = BallTree(lat_lon_points, metric='haversine')

# from sklearn.neighbors import KDTree

# def latlon_to_cartesian(lat, lon):
#     lat, lon = np.radians(lat), np.radians(lon)
#     x = np.cos(lat) * np.cos(lon)
#     y = np.cos(lat) * np.sin(lon)
#     z = np.sin(lat)
#     return np.column_stack((x, y, z))

# distances, _ = tree.query(lat_lon_points, k=5)  # Using k=5 nearest neighbors
# #distances, _ = tree.query(lat_lon_points, k=3)  # Using k=3 instead of k=5

# cartesian_points = latlon_to_cartesian(latitudes_remaining, longitudes_remaining)

# tree = KDTree(cartesian_points, metric='euclidean')

# distances, _ = tree.query(cartesian_points, k=5)

# batch_size = 5000  # Adjust based on memory constraints
# num_batches = int(np.ceil(len(lat_lon_points) / batch_size))
# all_distances = []

# for i in range(num_batches):
#     start, end = i * batch_size, min((i + 1) * batch_size, len(lat_lon_points))
#     distances, _ = tree.query(lat_lon_points[start:end], k=5)
#     all_distances.append(distances)

# distances = np.vstack(all_distances)  # Combine results

import hnswlib
import numpy as np
from tqdm import tqdm

def latlon_to_cartesian(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack((x, y, z))

cartesian_points = latlon_to_cartesian(latitudes_remaining, longitudes_remaining)

lat_lon_points = np.radians(np.vstack([latitudes_remaining, longitudes_remaining]).T)

#!pip install hnswlib

# Initialize HNSW index for fast Approximate Nearest Neighbors
num_elements = len(cartesian_points)
hnsw_index = hnswlib.Index(space='l2', dim=3)  # Use 'l2' for Euclidean distance

# Efficient index parameters: 
hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=32)

# Add points to index in batches
batch_size = 5000
for i in tqdm(range(0, num_elements, batch_size), desc="Adding points to HNSW index"):
    hnsw_index.add_items(cartesian_points[i:i + batch_size])

# Set query parameters (higher `ef` improves accuracy, but increases computation)
hnsw_index.set_ef(100)

# Query in batches to prevent excessive memory usage
k = 5  # Number of nearest neighbors
all_distances = []

for i in tqdm(range(0, num_elements, batch_size), desc="Querying nearest neighbors"):
    batch_points = cartesian_points[i:i + batch_size]
    distances, _ = hnsw_index.knn_query(batch_points, k=k)
    all_distances.append(distances)

# Stack all distances
distances = np.vstack(all_distances)

distances.shape





# Advanced spatiotemporal density estimation with computational optimization
from sklearn.neighbors import KernelDensity
from scipy.spatial import cKDTree
from tqdm import tqdm

# 1. Optimize depth KDE with adaptive bandwidth selection
# Calculate interquartile range for adaptive bandwidth
depth_q75, depth_q25 = np.percentile(depths_remaining, [75, 25])
depth_iqr = depth_q75 - depth_q25
depth_bandwidth = 0.9 * depth_iqr * len(depths_remaining)**(-1/5)  # Silverman's rule of thumb
print(f"Using adaptive depth bandwidth: {depth_bandwidth:.4f}")

depth_kde = KernelDensity(bandwidth=max(0.1, depth_bandwidth), metric='euclidean')
depth_kde.fit(depths_remaining.reshape(-1, 1))

# # # Incorporates adaptive bandwidth selection for the KDE based on distribution characteristics
# # # Adds Arctic-specific geographic weighting to compensate for sparse high-latitude data
# # # Implements temporal density considerations to balance seasonal representation
# # # Uses more detailed stratification relevant to permafrost and active layer dynamics
# # # Applies physically informed weighting for depths typical of zero-curtain formation
# # # Maintains computational efficiency through batched processing
# # # Preserves spatiotemporal integrity for your 130-year record

# # # Incorporates adaptive bandwidth selection for the KDE based on distribution characteristics
# # # Adds Arctic-specific geographic weighting to compensate for sparse high-latitude data
# # # Implements temporal density considerations to balance seasonal representation
# # # Uses more detailed stratification relevant to permafrost and active layer dynamics
# # # Applies physically informed weighting for depths typical of zero-curtain formation
# # # Maintains computational efficiency through batched processing
# # # Preserves spatiotemporal integrity for your 130-year record

# # Process in batches with enhanced error handling
# print("Calculating depth kernel density estimates...")
# batch_size = 10000
# depth_log_density = np.zeros(len(depths_remaining))

# for i in tqdm(range(0, len(depths_remaining), batch_size), desc="KDE"):
#     end_idx = min(i + batch_size, len(depths_remaining))
#     batch = depths_remaining[i:end_idx].reshape(-1, 1)
#     depth_log_density[i:end_idx] = depth_kde.score_samples(batch)



# # 2. Geographic density with spherical harmonics for Arctic-focused weighting
# # Emphasize high-latitude regions
# if np.any(geo_density == 0):
#     print("Enhancing geographic density estimation...")
#     # Add latitude-based weighting (Arctic amplification factor)
#     arctic_weight = np.ones_like(latitudes_remaining)
#     arctic_weight[latitudes_remaining >= 66.5] *= 0.8  # Reduce density weight for Arctic
#     arctic_weight[(latitudes_remaining >= 60) & (latitudes_remaining < 66.5)] *= 0.9  # Reduce sli...
    
#     # Combine with existing geo_density or use as replacement if geo_density is all zeros
#     if np.all(geo_density == 0):
#         geo_density = arctic_weight
#     else:
#         geo_density = geo_density * arctic_weight

# # 3. Temporal density based on record completeness and seasonal patterns
# # Calculate per-month data density to account for seasonal sampling bias
# timestamps_months = np.array([ts.month if hasattr(ts, 'month') else 
#                               ts.to_pydatetime().month if hasattr(ts, 'to_pydatetime') else 1 
#                               for ts in timestamps[:test_start]])
# month_counts = np.bincount(timestamps_months, minlength=13)[1:]  # Skip index 0
# month_density = month_counts / np.max(month_counts)  # Normalize

# # Apply monthly weighting to balance seasonal representation
# temporal_weights = np.array([1.0/max(0.1, month_density[m-1]) for m in timestamps_months])
# temporal_weights = temporal_weights / np.mean(temporal_weights)  # Normalize

# # 4. Combine multi-dimensional densities with physical understanding of zero-curtain processes
# depth_density = np.exp(depth_log_density)
# combined_density = geo_density * depth_density

# # Apply multi-scale density normalization for balance across dimensions
# density = np.sqrt(combined_density)  # Use sqrt to moderate extreme values
# density = density / np.mean(density) * temporal_weights  # Apply temporal weighting

# # 5. Generate physically informed weights
# weights = 1.0 / (density + 0.01)  # Inverse density with stabilizer
# weights = weights / np.sum(weights) * len(weights)  # Normalize

# # 6. Enhanced stratification with climate zone consideration
# # More detailed seasonal definition (early/late winter, spring thaw, summer, fall freeze)
# months = np.array([ts.month if hasattr(ts, 'month') else 
#                    ts.to_pydatetime().month if hasattr(ts, 'to_pydatetime') else 1 
#                    for ts in timestamps[:test_start]])
# seasons = np.zeros_like(months, dtype=int)
# seasons[(months == 12) | (months == 1) | (months == 2)] = 1  # Winter
# seasons[(months == 3) | (months == 4) | (months == 5)] = 2   # Spring
# seasons[(months == 6) | (months == 7) | (months == 8)] = 3   # Summer
# seasons[(months == 9) | (months == 10) | (months == 11)] = 4 # Fall

# # Arctic-specific regions with permafrost classification
# regions = np.zeros_like(latitudes_remaining, dtype=int)
# regions[(latitudes_remaining >= 75)] = 4                      # High Arctic
# regions[(latitudes_remaining >= 66.5) & (latitudes_remaining < 75)] = 3  # Arctic
# regions[(latitudes_remaining >= 60) & (latitudes_remaining < 66.5)] = 2  # Subarctic
# regions[(latitudes_remaining >= 50) & (latitudes_remaining < 60)] = 1    # Northern Boreal

# # More detailed depth zones relevant to active layer dynamics
# depth_zones = np.zeros_like(depths_remaining, dtype=int)
# depth_zones[depths_remaining <= 0.2] = 1  # Surface zone
# depth_zones[(depths_remaining > 0.2) & (depths_remaining <= 0.5)] = 2  # Upper active layer
# depth_zones[(depths_remaining > 0.5) & (depths_remaining <= 1.0)] = 3  # Lower active layer
# depth_zones[(depths_remaining > 1.0) & (depths_remaining <= 2.0)] = 4  # Top permafrost
# depth_zones[depths_remaining > 2.0] = 5  # Deep permafrost

# # Density stratification with emphasis on sparse regions
# density_quantiles = np.zeros_like(density, dtype=int)
# try:
#     percentiles = np.percentile(density, [20, 40, 60, 80])
#     density_quantiles = np.digitize(density, bins=percentiles)
# except:
#     print("Using simplified density quantiles due to computational constraints")
#     # Simplified density quantiles if percentile calculation fails
#     density_sorted = np.sort(density)
#     step = len(density_sorted) // 5
#     if step > 0:
#         bins = [density_sorted[i*step] for i in range(1, 5)]
#         density_quantiles = np.digitize(density, bins=bins)

# # Combined strata with permafrost-focused weighting
# strata = seasons * 10000 + regions * 1000 + depth_zones * 100 + density_quantiles
# unique_strata = np.unique(strata)
# print(f"Created {len(unique_strata)} unique strata for sampling")

# # 7. Stratified sampling with consideration for sparse temporal coverage
# val_size = int(n_samples * val_fraction)
# train_size = test_start - val_size

# train_indices = []
# val_indices = []

# for stratum in tqdm(unique_strata, desc='Stratified Sampling'):
#     stratum_indices = np.where(strata == stratum)[0]
    
#     if len(stratum_indices) == 0:
#         continue
    
#     stratum_weights = weights[stratum_indices]
#     stratum_weight_sum = np.sum(stratum_weights)
    
#     # Adaptive validation size based on stratum importance for zero-curtain detection
#     # Larger validation samples for depths typical of zero-curtain formation
#     importance_factor = 1.0
#     stratum_depth_zone = (stratum // 100) % 10
#     if stratum_depth_zone in [2, 3]:  # Active layer depths
#         importance_factor = 1.2  # Increase validation representation
    
#     target_val_size = int(val_size * stratum_weight_sum / np.sum(weights) * importance_factor)
#     target_val_size = max(target_val_size, 1)
#     target_val_size = min(target_val_size, len(stratum_indices) - 1)
    
#     # Chronological splitting within strata to maintain temporal patterns
#     stratum_val_indices = stratum_indices[-target_val_size:]
#     stratum_train_indices = stratum_indices[:-target_val_size]
    
#     train_indices.extend(stratum_train_indices)
#     val_indices.extend(stratum_val_indices)

from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors

# Convert to float32 for efficiency
depths_remaining = depths_remaining.astype(np.float32)

# Step 1: Compute global bandwidth using Silverman’s rule
global_bandwidth = 1.06 * np.std(depths_remaining) * (len(depths_remaining) ** -0.2)

# Step 2: Estimate local density using Nearest Neighbors
nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
nn.fit(depths_remaining.reshape(-1, 1))
distances, _ = nn.kneighbors(depths_remaining.reshape(-1, 1))

# Adaptive bandwidth: Scale global bandwidth based on local density
adaptive_bandwidths = global_bandwidth * (np.median(distances, axis=1) / np.median(distances))

# Step 3: Compute KDE with Adaptive Bandwidth
depth_kde_values = np.zeros_like(depths_remaining)

for i in tqdm(range(len(depths_remaining))):
    kde = gaussian_kde(depths_remaining, bw_method=adaptive_bandwidths[i])
    depth_kde_values[i] = kde(depths_remaining[i])

# Convert to log-density for better numerical stability
depth_log_density = np.log(depth_kde_values + 1e-10)  # Add small value to avoid log(0)
depth_density = np.exp(depth_log_density)



























# Depth KDE
depth_kde = KernelDensity(bandwidth=0.5, metric='euclidean', n_jobs=-1)
depth_kde.fit(depths_remaining.reshape(-1, 1))
depth_log_density = depth_kde.score_samples(depths_remaining.reshape(-1, 1))
depth_density = np.exp(depth_log_density)

density = geo_density * depth_density









arctic_weight = np.ones_like(latitudes_remaining)
arctic_weight[latitudes_remaining >= 66.5] *= 0.8  # High Arctic reduction
arctic_weight[(latitudes_remaining >= 60) & (latitudes_remaining < 66.5)] *= 0.9  # Subarctic

# Efficiently apply weighting
geo_density = geo_density * arctic_weight if np.any(geo_density) else arctic_weight




# Apply Arctic-specific geographic weighting
geo_density = geo_density * arctic_weight if np.any(geo_density) else arctic_weight

# Apply seasonal weighting
density = np.sqrt(geo_density * depth_density)
density = density / np.mean(density) * temporal_weights  # Normalize

# Generate stratified sampling weights
weights = 1.0 / (density + 0.01)
weights = weights / np.sum(weights) * len(weights)  # Normalize







from scipy.stats import gaussian_kde

# Convert to float32 for memory efficiency
depths_remaining = depths_remaining.astype(np.float32)

# Fit Gaussian KDE (much faster than sklearn's KernelDensity)
depth_kde = gaussian_kde(depths_remaining, bw_method=0.5)  # Bandwidth = 0.5

# Compute KDE in one go (no need for batch processing)
depth_log_density = depth_kde(depths_remaining)
depth_density = np.exp(depth_log_density)














# Highly optimized cKDTree implementation
from scipy.spatial import cKDTree
import numpy as np
from tqdm import tqdm

print("Using optimized cKDTree for depth density estimation...")

# 1. Reduce dimensionality by working with unique depths only
unique_depths, inverse_indices, counts = np.unique(depths_remaining, 
                                                  return_inverse=True, 
                                                  return_counts=True)
print(f"Reduced from {len(depths_remaining)} points to {len(unique_depths)} unique depths")

# 2. Build tree on unique values only (much smaller)
depth_tree = cKDTree(unique_depths.reshape(-1, 1))

# 3. Query with smaller k for speed
k = min(5, len(unique_depths)-1)  # Use smaller k, but ensure it's valid
print(f"Using k={k} nearest neighbors")

# 4. Query all at once (no batching needed for small unique set)
distances, _ = depth_tree.query(unique_depths.reshape(-1, 1), k=k+1)

# 5. Process distances for unique depths only
mean_distances = np.mean(distances[:, 1:], axis=1)  # Skip first (self)
unique_depth_density = 1.0 / (mean_distances + 0.01)

# 6. Map back to original indices via lookup
depth_density = unique_depth_density[inverse_indices]

# 7. Normalize
depth_density = depth_density / np.sum(depth_density) * len(depth_density)

print("Depth density estimation completed!")

# Continue with your stratification as before
density = depth_density

# Create weights
weights = 1.0 / (density + 0.01)
weights = weights / np.sum(weights) * len(weights)

print("Density estimation completed successfully")

geo_density = np.exp(-distances[:, 1:].mean(axis=1))  # Approximate density

min_dist = np.min(distances[:, 1:])
max_dist = np.max(distances[:, 1:])
normalized_distances = (distances[:, 1:] - min_dist) / (max_dist - min_dist + 1e-10)
scaled_distances = normalized_distances * 5  # Adjust the multiplier to get good density distribution
geo_density = np.exp(-scaled_distances.mean(axis=1))

geo_density.mean(), geo_density.max(), geo_density.min()

# Calculate density based on mean distance to neighbors
neighbor_distances = np.abs(depths_remaining.reshape(-1, 1) - depths_remaining[indices])

months = np.array([ts.month if hasattr(ts, 'month') else ts.to_pydatetime().month for ts in timestamps[:test_start]])
seasons = np.digitize(months, bins=[3, 6, 9, 12])

if has_geo_info:
    regions = np.zeros_like(latitudes_remaining, dtype=int)
    regions[(latitudes_remaining >= 66.5)] = 3
    regions[(latitudes_remaining >= 60) & (latitudes_remaining < 66.5)] = 2
    regions[(latitudes_remaining >= 50) & (latitudes_remaining < 60)] = 1
else:
    regions = np.zeros_like(depths_remaining, dtype=int)

geo_density*depth_density

















np.unique(np.exp(-distances[:, 1:].mean(axis=1)))

# Query in batches to prevent excessive memory usage
k = 5  # Number of nearest neighbors
all_distances = []

for i in tqdm(range(0, num_elements, batch_size), desc="Querying nearest neighbors"):
    batch_points = cartesian_points[i:i + batch_size]
    distances, _ = hnsw_index.knn_query(batch_points, k=k)
    all_distances.append(distances)

# Stack all distances
distances = np.vstack(all_distances)

distances

geo_density = np.exp(-distances[:, 1:].mean(axis=1))  # Approximate density

geo_density

min_dist = np.min(distances[:, 1:])
max_dist = np.max(distances[:, 1:])
normalized_distances = (distances[:, 1:] - min_dist) / (max_dist - min_dist + 1e-10)
scaled_distances = normalized_distances * 5  # Adjust the multiplier to get good density distribution
geo_density = np.exp(-scaled_distances.mean(axis=1))
geo_density

#pd.DataFrame(geo_density).plot();

geo_density.mean(), geo_density.max(), geo_density.min()

np.unique(geo_density)

#OR
# # Use inverse distance for density estimation
# geo_density = 1.0 / (distances[:, 1:].mean(axis=1) + 1)  # Add 1 to avoid division by zero
# # Normalize to 0-1 range
# geo_density = (geo_density - np.min(geo_density)) / (np.max(geo_density) - np.min(geo_density) + 1...
# geo_density
#pd.DataFrame(geo_density).plot();
#geo_density.mean(), geo_density.max(), geo_density.min()

# Remove the n_jobs parameter
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm

# Correct initialization without n_jobs
depth_kde = KernelDensity(bandwidth=0.5, metric='euclidean')  # Remove n_jobs parameter
depth_kde.fit(depths_remaining.reshape(-1, 1))

depth_kde.fit(depths_remaining.reshape(-1, 1))

# depth_log_density = depth_kde.score_samples(depths_remaining.reshape(-1, 1))
# depth_density = np.exp(depth_log_density)

from scipy.spatial import cKDTree

# Create a KD-tree for quick nearest neighbor lookups
depth_tree = cKDTree(depths_remaining.reshape(-1, 1))



# Calculate density based on mean distance to neighbors
neighbor_distances = np.abs(depths_remaining.reshape(-1, 1) - depths_remaining[indices])

mean_distances = np.mean(neighbor_distances, axis=1)

# Convert distances to density (smaller distance = higher density)
depth_density = 1.0 / (mean_distances + 0.01)

density = geo_density * depth_density

# Ensure density is not zero to avoid division by zero
if np.all(density == 0):
    print("Warning: Combined density is zero. Using uniform density.")
    density = np.ones_like(depth_density)

# Create weights
weights = 1.0 / (density + 0.01)  # Add small constant to avoid division by zero
weights = weights / np.sum(weights) * len(weights)  # Normalize

months = np.array([ts.month if hasattr(ts, 'month') else ts.to_pydatetime().month for ts in timestamps[:test_start]])
seasons = np.digitize(months, bins=[3, 6, 9, 12])

if has_geo_info:
    regions = np.zeros_like(latitudes_remaining, dtype=int)
    regions[(latitudes_remaining >= 66.5)] = 3
    regions[(latitudes_remaining >= 60) & (latitudes_remaining < 66.5)] = 2
    regions[(latitudes_remaining >= 50) & (latitudes_remaining < 60)] = 1
else:
    regions = np.zeros_like(depths_remaining, dtype=int)

depth_zones = np.digitize(depths_remaining, bins=[0.2, 0.5, 1.0, 2.0])

# Check if density is suitable for percentile calculation
if len(np.unique(density)) > 5:
    density_quantiles = np.digitize(density, bins=np.percentile(density, [20, 40, 60, 80]))
else:
    print("Warning: Not enough unique density values. Using uniform density quantiles.")
    density_quantiles = np.zeros_like(density, dtype=int)

strata = seasons * 1000 + regions * 100 + depth_zones * 10 + density_quantiles

unique_strata = np.unique(strata)
val_size = int(n_samples * val_fraction)
val_indices = []
train_size = test_start - val_size
train_indices = []

for stratum in tqdm(unique_strata, desc='Stratified Sampling'):
    stratum_indices = np.where(strata == stratum)[0]
    
    if len(stratum_indices) == 0:
        continue
    
    stratum_weights = weights[stratum_indices]
    stratum_weight_sum = np.sum(stratum_weights)
    
    # Avoid division by zero if weights sum is zero
    target_val_size = int(val_size * stratum_weight_sum / np.sum(weights)) if np.sum(weights) > 0 else 1
    target_val_size = max(target_val_size, 1)
    target_val_size = min(target_val_size, len(stratum_indices) - 1)
    
    # Ensure there are enough indices for both train and validation
    if len(stratum_indices) <= 1:
        train_indices.extend(stratum_indices)
    else:
        stratum_val_indices = stratum_indices[-target_val_size:]
        stratum_train_indices = stratum_indices[:-target_val_size]
        train_indices.extend(stratum_train_indices)
        val_indices.extend(stratum_val_indices)

# Create final splits
X_train = X_remaining[train_indices]
y_train = y_remaining[train_indices]
train_metadata = [metadata[i] for i in train_val_indices[train_indices]]

X_val = X_remaining[val_indices]
y_val = y_remaining[val_indices]
val_metadata = [metadata[i] for i in train_val_indices[val_indices]]

print(f"Split sizes: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")



















#Spatiotemporally-aware data splitting


# 1. Temporal Stratification First
# Reserve the most recent 15-20% of data as a true hold-out test set
# This preserves temporal ordering and prevents data leakage

# 2. Spatial Density Analysis
# For the remaining 75-80% of data:
# Compute spatial density using Gaussian kernel density estimation
# Use varying bandwidths (e.g., 10km, 50km, 100km) to capture different scales
# Consider both horizontal (lat/long) and vertical (depth) dimensions

# 3. Stratified Spatiotemporal Sampling
# Create strata based on:
# Time periods (seasons, years)
# Geographic regions (Arctic, Subarctic, etc.)
# Measurement depth zones
# Density quantiles (to balance dense vs. sparse areas)

# 4. Block Cross-Validation Design
# Implement block CV with spatiotemporal constraints:
# Blocks defined by both time windows and spatial regions
# Ensure validation blocks are separated from training blocks by both space and time buffers

# 5. Variable Importance Weighting
# Weight samples inversely to their spatial density
# This prevents dense regions from dominating the learning process

#----#

# Density-Aware Sampling: Prevents dense regions from overwhelming the model
# Multi-dimensional Stratification: Ensures representation across seasons, regions, depths, and dens...
# Sample Weighting: Provides inverse density weighting to balance spatial representation
# Temporal Coherence: Maintains temporal progression within strata
# Comprehensive Reporting: Provides detailed statistics on spatial and temporal coverage

# This approach addresses the specific challenges you mentioned:

# Prevents Data Leakage: True temporal hold-out test set
# Limits Extrapolation: Ensures representation across spatiotemporal strata
# Reduces Overfitting: Balanced representation prevents memorization of dense regions
# Minimizes Error: Weighting scheme improves representation of sparse regions

#----#

# Separated geographic and depth density estimation
# Using radians for the haversine calculation
# Added robust error handling for missing coordinates
# Improved timestamp formatting
# Added more comprehensive reporting
# Fixed the unpacking syntax in your call to the function

X_train, X_val, X_test, y_train, y_val, y_test, train_weights = spatiotemporal_train_test_split(
    X, y, metadata, test_fraction=0.2, val_fraction=0.15
)

model.fit(
    X_train, y_train,
    sample_weight=train_weights,
    validation_data=(X_val, y_val),
    # Other parameters
)

X_test











import time
import os
import gc
import numpy as np
import pickle

output_base_dir='zero_curtain_pipeline/modeling'
os.makedirs(output_base_dir, exist_ok=True)

print("Starting model training process...")
data_dir = os.path.join(output_base_dir, 'ml_data')
print(f"Loading X data from {data_dir}...")
X = np.load(os.path.join(data_dir, 'X_features.npy'))
print(f"Loading y data from {data_dir}...")
y = np.load(os.path.join(data_dir, 'y_labels.npy'))
print(f"Loading metadata from {data_dir}...")
with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)

print(f"Data shapes: X={X.shape}, y={y.shape}, metadata={len(metadata)}")

print("Starting model training with 10% of data for initial test...")
start_time = time.time()

model, history, evaluation = train_zero_curtain_model_efficiently(
    X=X, 
    y=y,
    metadata=metadata,
    #output_dir=os.path.join(output_base_dir, 'model'),
    output_dir=os.path.join(output_base_dir, 'sample'),
    sample_fraction=0.001  # Use 0.1% of data for faster testing
)

# Save results
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

print("Saving results...")
# Save training time
#with open(os.path.join(output_base_dir, 'model', 'training_time.txt'), 'w') as f:
with open(os.path.join(output_base_dir, 'sample', 'training_time.txt'), 'w') as f:
    f.write(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")

# Save history and evaluation
#with open(os.path.join(output_base_dir, 'model', 'history.pkl'), 'wb') as f:
with open(os.path.join(output_base_dir, 'sample', 'history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

#with open(os.path.join(output_base_dir, 'model', 'evaluation.pkl'), 'wb') as f:
with open(os.path.join(output_base_dir, 'sample', 'evaluation.pkl'), 'wb') as f:
    pickle.dump(evaluation, f)

print("Done!")



# import tensorflow as tf
# tf.keras.mixed_precision.set_global_policy('mixed_float16')









# Safe F1 score implementation with error handling
def f1_score(y_true, y_pred):
    """Compute F1 score with numerical stability safeguards"""
    precision = tf.keras.metrics.Precision()(y_true, y_pred)
    recall = tf.keras.metrics.Recall()(y_true, y_pred)
    # Add epsilon to avoid division by zero
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

def load_model_from_weights(weights_path, input_shape, build_model_func=None):
    """Helper function to load a model from weights."""
    # Use the provided or default model building function
    if build_model_func is None:
        build_model_func = build_improved_zero_curtain_model_fixed
    
    # Build model architecture
    model = build_model_func(input_shape)
    
    # Load weights if they exist
    if os.path.exists(weights_path + ".index"):
        model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: Could not find weights at {weights_path}")
    
    return model



























model_dir="/Users/bgay/Desktop/Research/Code/zero_curtain_pipeline/modeling/improved_model"
history_file=os.path.join(model_dir, "training_history.json")

with open(history_file, 'rb') as f:
    history=json.load(f)

train_loss = []
val_loss = []
train_auc = []
val_auc = []

for chunk in history:
    if 'history' in chunk and chunk['history']:
        history = chunk['history']
        
        # Check if keys exist
        if 'loss' in history:
            train_loss.extend(history['loss'])
        if 'val_loss' in history:
            val_loss.extend(history['val_loss'])
        if 'auc' in history:
            train_auc.extend(history['auc'])
        if 'val_auc' in history:
            val_auc.extend(history['val_auc'])

#np.array(val_loss).max(), np.array(val_loss).min()
#(2615.30517578125, 0.07140758633613586)
#np.array(loss).max(), np.array(loss).min()
#(5.939471244812012, 7.869218279665802e-06)



loss=[]
loss.extend([history[i]['history']['loss'] for i in range(2784)])























import pandas as pd
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree

# Load merged_df
merged_df = pd.read_feather('/Users/bgay/Desktop/Research/Code/merged_compressed.feather')

# Load ERA5-Land data (Assuming you downloaded a NetCDF file)
era5_file = "era5_land_soil.nc"  # Change if using separate files for different years/months
ds = xr.open_dataset(era5_file)

# Extract ERA5 grid coordinates
era5_lats = ds.latitude.values
era5_lons = ds.longitude.values

# Convert ERA5 lat/lon into a 2D array of coordinate pairs
era5_coords = np.array([(lat, lon) for lat in era5_lats for lon in era5_lons])

# Create a k-d tree for quick nearest-neighbor lookup
tree = cKDTree(era5_coords)

# Extract unique lat/lon pairs from merged_df to reduce computation
unique_coords = merged_df[['latitude', 'longitude']].drop_duplicates().values

# Query the nearest ERA5 grid points for each unique coordinate
_, nearest_indices = tree.query(unique_coords)

# Create a mapping from original coordinates to nearest ERA5 grid points
coord_mapping = {tuple(unique_coords[i]): tuple(era5_coords[idx]) for i, idx in enumerate(nearest_indices)}

# Function to map a dataframe row to nearest ERA5 coordinate
def map_nearest_era5(row):
    return coord_mapping.get((row['latitude'], row['longitude']), (np.nan, np.nan))

# Apply mapping to get nearest ERA5 coordinates for each row in merged_df
merged_df[['era5_lat', 'era5_lon']] = merged_df.apply(map_nearest_era5, axis=1, result_type='expand')

# Extract ERA5 time variable
era5_time = pd.to_datetime(ds.time.values)

# Function to match nearest timestamp
def match_nearest_time(timestamp):
    return era5_time[np.argmin(np.abs(era5_time - timestamp))]

# Convert merged_df datetime to pandas datetime
merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])

# Find the closest ERA5 timestamp for each row
merged_df['era5_time'] = merged_df['datetime'].apply(match_nearest_time)

# Extract ERA5 values at matched lat/lon/time
def extract_era5_value(row, variable):
    try:
        return ds[variable].sel(
            latitude=row['era5_lat'],
            longitude=row['era5_lon'],
            time=row['era5_time'],
            method="nearest"
        ).values.item()
    except Exception:
        return np.nan

# List of ERA5 variables
era5_vars = [
    "soil_temperature_level_1", "soil_temperature_level_2", "soil_temperature_level_3", "soil_temperature_level_4",
    "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2", "volumetric_soil_water_layer_3", "volumetric_soil_water_layer_4",
    "snow_depth", "soil_type", "2m_temperature", "surface_latent_heat_flux",
    "surface_sensible_heat_flux", "total_precipitation", "land_sea_mask"
]

# Apply function to extract data for each variable
for var in era5_vars:
    merged_df[var] = merged_df.apply(lambda row: extract_era5_value(row, var), axis=1)

# Save the updated dataframe
merged_df.to_feather('/Users/bgay/Desktop/Research/Code/merged_with_era5.feather')

# Print a sample
print(merged_df.head())




# merged_df = pd.read_csv('merged_insitu_dataset.csv')
# processor2_df_with_coords = pd.read_csv('processor2_df_with_coords_updated.csv')

# del processor2_df_with_coords

zc = pd.read_csv('processor2_df_with_coords_updated.csv')

merged_df = pd.read_csv('merged_insitu_dataset.csv')

temp_ds = pd.read_csv('ZC_data_tempnonans_df.csv')

temp_ds['datetime'] = pd.to_datetime(temp_ds['datetime'], errors='coerce', format='mixed')

alt_ds = pd.read_csv('ZC_data_thicknessnonans_df.csv')

alt_ds['datetime'] = pd.to_datetime(alt_ds['datetime'], errors='coerce', format='mixed')

#temp_ds.isna().sum()

# #print(temp_ds.columns.tolist())
# temp_ds = temp_ds[['datetime', 'latitude', 'longitude', 'source', 'depth', 'depth_zone', 
#          'site_id', 'temperature']]

#alt_ds.isna().sum()

# #print(alt_ds.columns.tolist())
# alt_ds = alt_ds[['datetime', 'latitude', 'longitude', 'source', 'year', 'thickness', 
#         'depth_zone', 'site_id']]

# processor = InSituDataProcessor()

# try:
#     # Process the data
#     processed_data = processor.prepare_for_model(temp_ds, alt_ds)
    
#     # Save processed data
#     np.savez('arctic_data_processed.npz',
#              temperature=processed_data['temperature'],
#              alt=processed_data['alt'],
#              zero_curtain_mask=processed_data['zero_curtain_mask'],
#              time_grid=processed_data['time_grid'].astype('datetime64[ns]'),
#              lat_grid=processed_data['lat_grid'],
#              lon_grid=processed_data['lon_grid'])
    
#     print("\nData saved successfully!")
    
# except Exception as e:
#     print(f"Processing error: {str(e)}")
#     import traceback
#     traceback.print_exc()

processor = InSituDataProcessor()

processed_data = processor.prepare_for_model(temp_ds, alt_ds)

thermal_wavelength = processed_data['thermal_wavelength']

tunneling_prob = processed_data['tunneling_probability']

print("\nQuantum Parameter Statistics:")
print(f"Thermal wavelength shape: {thermal_wavelength.shape}")
print(f"Valid thermal wavelength points: {np.sum(~np.isnan(thermal_wavelength))}")
print(f"Valid tunneling probability points: {np.sum(~np.isnan(tunneling_prob))}")

print("Thermal Wavelength (m):")
print(f"Mean: {np.nanmean(thermal_wavelength):.2e}")
print(f"Min: {np.nanmin(thermal_wavelength):.2e}")
print(f"Max: {np.nanmax(thermal_wavelength):.2e}")

print("\nTunneling Probability:")
print(f"Mean: {np.nanmean(tunneling_prob):.2e}")
print(f"Min: {np.nanmin(tunneling_prob):.2e}")
print(f"Max: {np.nanmax(tunneling_prob):.2e}")

zero_curtain_mask = processed_data['zero_curtain_mask']
zc_wavelength = np.nanmean(thermal_wavelength[zero_curtain_mask])
zc_tunneling = np.nanmean(tunneling_prob[zero_curtain_mask])

print("\nMean values during zero curtain periods:")
print(f"Thermal wavelength: {zc_wavelength:.2e} m")
print(f"Tunneling probability: {zc_tunneling:.2e}")

non_zc_wavelength = np.nanmean(thermal_wavelength[~zero_curtain_mask])
non_zc_tunneling = np.nanmean(tunneling_prob[~zero_curtain_mask])

print("\nMean values during non-zero curtain periods:")
print(f"Thermal wavelength: {non_zc_wavelength:.2e} m")
print(f"Tunneling probability: {non_zc_tunneling:.2e}")

np.savez('arctic_processed_data.npz',
         temperature=processed_data['temperature'],
         alt=processed_data['alt'],
         zero_curtain_mask=processed_data['zero_curtain_mask'],
         thermal_wavelength=processed_data['thermal_wavelength'],
         tunneling_probability=processed_data['tunneling_probability'],
         time_grid=processed_data['time_grid'].astype('datetime64[ns]'),
         lat_grid=processed_data['lat_grid'],
         lon_grid=processed_data['lon_grid'])

print("Data saved to 'arctic_processed_data.npz'")

loaded_data = np.load('arctic_processed_data.npz')

# Access the arrays
temperature = loaded_data['temperature']
alt = loaded_data['alt']
zero_curtain_mask = loaded_data['zero_curtain_mask']
thermal_wavelength = loaded_data['thermal_wavelength']
tunneling_probability = loaded_data['tunneling_probability']
time_grid = loaded_data['time_grid'].astype('datetime64[ns]')
lat_grid = loaded_data['lat_grid']
lon_grid = loaded_data['lon_grid']



quantum_interpreter = QuantumParameterInterpreter({
    'thermal_wavelength': thermal_wavelength,
    'tunneling_probability': tunneling_prob
})

quantum_interpreter.visualize_enhanced_parameters(
    processed_data['time_grid'],
    processed_data['lat_grid'],
    processed_data['lon_grid']
)

# Quantum Effects at Various Temperatures
    # Smaller wavelength → Higher temperature, more energetic molecular motion
    # Larger wavelength → Lower temperature, less molecular kinetic energy

# Tunneling: quantifies the likelihood of quantum particles (water molecules) passing through an ene...
    # Probabilistic Phase Transitions
        # Tunneling = 0 | Lower probability of quantum phase transition
    # Temperature Dependence
        # Tunneling probability decreases exponentially with temperature
        # Critically important near 0°C (zero curtain effect)

#Interpretations

# For Soil Temperature (temp):

# Distribution Pattern:
# The thermal wavelength shows a highly skewed distribution with most values concentrated near zero ...
# The tunneling probability is strongly biased towards 1.0 (mean ≈ 0.90, median = 1.0)

# Spatial Correlations:
# There's a strong negative correlation (pearson ≈ -0.96) between thermal wavelength and tunneling p...
# The spatial heterogeneity is high for thermal wavelength (entropy ≈ 7.68e7) but very low for tunne...


# For Active Layer Thickness (alt):

# Distribution Pattern:
# The thermal wavelength is tightly clustered around 1.60e-05 (very small standard deviation ≈ 2.14e...
# The tunneling probability is consistently zero across all measurements

# Spatial Characteristics:
# No correlation coefficient could be calculated (NaN) due to zero variance in tunneling probability
# High spatial entropy for thermal wavelength (≈ 4.25e8) suggests complex spatial patterns


# Key Interpretations:

# Temperature varies more significantly across space than ALT does
# The ALT measurements show remarkably consistent behavior (tight clustering)
# The strong negative correlation in temperature data suggests that areas with higher thermal wavele...
# The spatial patterns suggest that soil temperature has more localized variability while ALT shows ...

# Temperature Dynamics (temp):
    # Spatial Distribution: The heatmap (Image 1) shows a distinct spatial pattern with a notable bl...
    # The bimodal distribution in the tunneling probability (Image 2) suggests two dominant thermal ...
    # Most locations show very high tunneling probability (~1.0)
    # A smaller but significant set of locations show near-zero tunneling probability
    # Thermal wavelength distribution (Image 3) is heavily right-skewed, with most values concentrat...

# Active Layer Thickness (alt):
    # Much more uniform spatial pattern (Image 4) with consistent thermal wavelength values
    # The tunneling probability distribution (Image 5) shows an extremely concentrated peak at zero
    # Thermal wavelength (Image 6) shows a sharp peak around 1.6e-5, indicating very consistent ALT ...

# Key Correlations and Relationships:
    # For temperature: The strong negative correlation (-0.96) between thermal wavelength and tunnel...
        # Areas with higher thermal wavelengths consistently show lower tunneling probabilities
        # This suggests potential thermal barriers or phase transitions in these regions
    # For ALT: The lack of correlation (NaN) and zero tunneling probability suggests:
        # ALT behavior is more mechanically constrained
        # Less thermal variability in the active layer processes

# Environmental Implications:
    # Temperature shows more spatial heterogeneity than ALT
    # ALT appears more regionally consistent, suggesting it might be controlled by broader environme...
    # The high spatial entropy in both measurements indicates complex spatial patterns, but:
    # Temperature shows more localized variability (76.8M entropy)
    # ALT shows broader regional patterns (425.1M entropy)

# Phase Transition Analysis:
    # Temperature:
        # 90.08% of regions show high transition probability
        # Mean wavelength in transition zones is effectively zero
        # Suggests widespread but stable thermal conditions
    # ALT:
        # 100% of regions are in transition zones
        # Consistent mean wavelength of 1.6e-5
        # Indicates uniform active layer behavior

# These patterns suggest that while soil temperature varies significantly across space and shows dis...
# Different controlling factors (local vs regional)
# Different response times to environmental changes
# Different sensitivity to local conditions vs broader climate patterns

# Potential Drivers of Observed Patterns:

# Temperature Patterns:
    # Local topography affecting solar radiation receipt
    # Vegetation cover influencing ground thermal regime
    # Snow cover distribution and duration
    # Surface water bodies and drainage patterns
    # Soil composition and organic layer thickness

# ALT Uniformity:
    # Regional climate patterns
    # Ground ice content
    # Soil texture/composition homogeneity
    # Permafrost continuity
    # Historical permafrost evolution

# Integration Value of Additional Data:
    # MODIS/ASTER Land Surface Temperature:
        # Would provide temporal dynamics of surface temperature
        # Could reveal lag effects between surface and subsurface temperatures
        # Help identify areas of thermal anomalies
        # Bridge scale gap between point measurements and regional patterns
        # Value added: Understanding surface-subsurface thermal coupling
    # SMAP Soil Moisture:
        # Critical for understanding thermal conductivity variations
        # Could explain some of the tunneling probability patterns
        # Help identify areas of potential phase change
        # Link to active layer dynamics
        # Value added: Explain thermal property variations
    # InSAR (Sentinel-1, UAVSAR):
        # Surface deformation patterns could link to subsurface processes
        # Seasonal thaw subsidence patterns
        # Long-term permafrost degradation trends
        # Spatial continuity of active layer processes
        # Value added: Physical manifestation of thermal processes

# Expected Insights:
    # Process Understanding:
        # Better quantification of surface-subsurface coupling
        # Identification of key controlling factors
        # Understanding of scale-dependent processes
    # Pattern Validation:
        # Confirm if observed patterns are representative
        # Identify potential sampling biases
        # Understand temporal stability of patterns
    # Future Predictions:
        # Develop more robust predictive models
        # Better understand system sensitivity
        # Identify key monitoring parameters

# Initialize analyzers
quantum_analyzer = QuantumEnvironmentalAnalyzer(
    quantum_data={
        'thermal_wavelength': thermal_wavelength,
        'tunneling_probability': tunneling_prob
    },
    environmental_data={
        'temp': temperature_data,
        'alt': alt_data
    }
)

# Get comprehensive analysis
phase_transitions = quantum_analyzer.analyze_phase_transitions()
correlations = quantum_analyzer.compute_environmental_correlations()
seasonal_patterns = quantum_analyzer.analyze_seasonal_patterns(time_grid)
hotspots = quantum_analyzer.identify_hotspots(lat_grid, lon_grid)

# Generate visualizations
quantum_analyzer.visualize_relationships()

# For integrating additional data sources
integrator = MultiSourceIntegrator()

# Add remote sensing data
integrator.add_remote_sensing_source(
    'MODIS_LST', 
    modis_data, 
    modis_metadata
)
integrator.add_remote_sensing_source(
    'SMAP_SM',
    smap_data,
    smap_metadata
)

# Harmonize all data to common grid
harmonized_data = integrator.harmonize_data(
    target_grid=(lat_grid, lon_grid),
    target_times=time_grid
)









temp_df = pd.read_csv('ZC_data_tempnonans_df.csv')
temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], errors='coerce', format='mixed')

zc_df = pd.read_csv('processor2_df_with_coords_updated.csv')

# alt_df = pd.read_csv('ZC_data_thicknessnonans_df.csv')
# alt_df['datetime'] = pd.to_datetime(alt_df['datetime'], errors='coerce', format='mixed')

# merged_df = pd.read_csv('merged_insitu_dataset.csv')



# zc_df = pd.read_csv('processor2_df_with_coords_updated.csv')

# site_id	site_name	depth	year	season	depth_zone	start_date	end_date	duration_hours	mean_temp	std_te...
# 0	Borehole_1657-Eniseisk-Dataset_1900-Average-Mo...	NaN	0.8	1920	Spring	deep	1920-04-15 00:00:00	1...
# 1	Borehole_1605-Ishim-Dataset_1848-Average-Month...	NaN	0.4	1942	Spring	intermediate	1942-04-15 00...
# 2	Borehole_1681-Kazachinskoe_expfield-Dataset_19...	NaN	1.6	1945	Spring	very_deep	1945-04-15 00:00...
# 3	Borehole_1630-Norsk__Norskii_sklad_-Dataset_18...	NaN	0.6	1962	Spring	deep	1962-04-15 00:00:00	1...
# 4	Borehole_1652-Turukhansk-Dataset_1895-Average-...	NaN	0.2	1965	Spring	shallow	1965-04-15 00:00:0...
# ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
# 1047663	Borehole_1867-Kitzsteinhorn_7-Dataset_2146-Ave...	NaN	5.0	2024	Fall	very_deep	2024-11-12 0...
# 1047664	Borehole_1866-Kitzsteinhorn_6-Dataset_2144-Ave...	NaN	10.0	2024	Fall	very_deep	2024-11-13 ...
# 1047665	Borehole_1867-Kitzsteinhorn_7-Dataset_2146-Ave...	NaN	5.0	2024	Fall	very_deep	2024-11-13 0...
# 1047666	Borehole_1866-Kitzsteinhorn_6-Dataset_2144-Ave...	NaN	3.0	2024	Fall	very_deep	2024-11-14 0...
# 1047667	Borehole_1866-Kitzsteinhorn_6-Dataset_2144-Ave...	NaN	20.0	2024	Fall	very_deep	2024-11-14 ...
# 1047668 rows × 16 columns

# temp_df = pd.read_csv('ZC_data_tempnonans_df.csv')
# temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], errors='coerce', format='mixed')

# Surface temperature average weight	Number of cloud free observations	datetime	temperature	latitude...
# 0	NaN	NaN	1915-07-01	17.00	62.000000	129.700000	siberia_Yakutsk_jul	10.0	1915.0	Yakutsk	...	NaN	Na...
# 1	NaN	NaN	1915-07-01	16.00	62.000000	129.700000	siberia_Yakutsk_jul	11.0	1915.0	Yakutsk	...	NaN	Na...
# 2	NaN	NaN	1915-07-15	-48.00	62.000000	129.700000	siberia_Yakutsk_jan	11.0	1915.0	Yakutsk	...	NaN	N...
# 3	NaN	NaN	1915-07-15	-46.00	62.000000	129.700000	siberia_Yakutsk_jan	10.0	1915.0	Yakutsk	...	NaN	N...
# 4	NaN	NaN	1915-07-15	1.00	62.000000	129.700000	siberia_Yakutsk_jan	0.0	1915.0	Yakutsk	...	NaN	NaN	...
# ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
# 12700489	NaN	NaN	2024-11-15	-4.48	47.189343	12.685295	Borehole_1119-Kitzsteinhorn_1-Dataset_2145-A...
# 12700490	NaN	NaN	2024-11-15	-0.02	47.189343	12.685295	Borehole_1119-Kitzsteinhorn_1-Dataset_2145-A...
# 12700491	NaN	NaN	2024-11-15	-2.48	47.189343	12.685295	Borehole_1119-Kitzsteinhorn_1-Dataset_2145-A...
# 12700492	NaN	NaN	2024-11-15	0.04	47.189343	12.685295	Borehole_1119-Kitzsteinhorn_1-Dataset_2145-Av...
# 12700493	NaN	NaN	2024-11-15	-6.46	47.189343	12.685295	Borehole_1119-Kitzsteinhorn_1-Dataset_2145-A...
# 12700494 rows × 29 columns

# del alt_ds, temp_ds, zc

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from datetime import datetime, timedelta
# import gc
# import os

# def process_single_site(site_id, site_data, zc_df, window_size=30, min_depths=4):
#     """Process a single site to reduce memory usage"""
#     try:
#         # Create pivot table
#         pivot = site_data.pivot_table(
#             index='datetime',
#             columns='depth',
#             values='temperature',
#             aggfunc='mean'
#         ).sort_index()
        
#         # Ensure regular time intervals
#         pivot = pivot.resample('D').mean()
        
#         # Skip if insufficient depths
#         if len(pivot.columns) < min_depths:
#             return []
            
#         site_windows = []
#         dates = pivot.index
        
#         # Process in smaller window chunks
#         for j in range(0, len(dates) - window_size + 1, window_size):
#             window = pivot.iloc[j:j+window_size]
            
#             # Skip windows with too many missing values
#             if window.isnull().sum().sum() / (window.shape[0] * window.shape[1]) > 0.3:
#                 continue
            
#             # Check for zero curtain events
#             window_start = dates[j]
#             window_end = dates[j + window_size - 1]
            
#             zc_events = zc_df[
#                 (zc_df['site_id'] == site_id) &
#                 (zc_df['start_date'] >= window_start) &
#                 (zc_df['end_date'] <= window_end)
#             ]
            
#             # Fill missing values
#             window_filled = window.interpolate(method='linear', axis=0)
#             window_filled = window_filled.fillna(method='ffill').fillna(method='bfill')
            
#             if not window_filled.isnull().any().any():
#                 site_windows.append({
#                     'site_id': site_id,
#                     'window_data': window_filled.values,
#                     'window_start': window_start,
#                     'window_end': window_end,
#                     'latitude': site_data['latitude'].iloc[0],
#                     'longitude': site_data['longitude'].iloc[0],
#                     'depths': window_filled.columns.values,
#                     'has_zero_curtain': 1 if not zc_events.empty else 0,
#                     'zc_duration': zc_events['duration_hours'].iloc[0] if not zc_events.empty else...
#                 })
            
#             # Clear some memory
#             del window, window_filled
#             gc.collect()
            
#         return site_windows
        
#     except Exception as e:
#         print(f"Error processing site {site_id}: {str(e)}")
#         return []

# def preprocess_in_chunks(temp_df, zc_df, chunk_size=10, window_size=30, min_depths=4, depth_range=...
#     """Process the data in smaller chunks with better memory management"""
    
#     # Convert to datetime
#     temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
#     zc_df['start_date'] = pd.to_datetime(zc_df['start_date'])
#     zc_df['end_date'] = pd.to_datetime(zc_df['end_date'])
    
#     # Get unique sites
#     unique_sites = temp_df['site_id'].unique()
#     print(f"Total unique sites: {len(unique_sites)}")
    
#     all_processed_data = []
    
#     for i in range(0, len(unique_sites), chunk_size):
#         chunk_sites = unique_sites[i:i + chunk_size]
#         print(f"\nProcessing sites {i} to {i + len(chunk_sites)-1}")
        
#         # Filter data for current chunk and relevant depths
#         chunk_temp_df = temp_df[
#             (temp_df['site_id'].isin(chunk_sites)) &
#             (temp_df['depth'].between(depth_range[0], depth_range[1]))
#         ].copy()
        
#         n_measurements = len(chunk_temp_df)
#         print(f"Processing chunk with {n_measurements:,} measurements")
        
#         # Handle duplicate measurements by averaging
#         chunk_temp_df = chunk_temp_df.groupby(
#             ['site_id', 'datetime', 'depth']
#         ).agg({
#             'temperature': 'mean',
#             'latitude': 'first',
#             'longitude': 'first'
#         }).reset_index()
        
#         # Process each site individually
#         chunk_processed = []
#         for site_id in chunk_sites:
#             site_data = chunk_temp_df[chunk_temp_df['site_id'] == site_id]
            
#             if len(site_data) < window_size:
#                 continue
                
#             site_windows = process_single_site(
#                 site_id, site_data, zc_df, 
#                 window_size=window_size, 
#                 min_depths=min_depths
#             )
            
#             chunk_processed.extend(site_windows)
            
#             # Clear site data
#             del site_data
#             gc.collect()
        
#         # Clear chunk data
#         del chunk_temp_df
#         gc.collect()
        
#         all_processed_data.extend(chunk_processed)
#         print(f"Processed windows so far: {len(all_processed_data)}")
        
#         # Optional: Save intermediate results
#         if len(all_processed_data) > 0 and len(all_processed_data) % 1000 == 0:
#             print("Saving intermediate results...")
#             np.save(f'processed_data_{len(all_processed_data)}.npy', all_processed_data)
    
#     return all_processed_data





def process_site_efficiently(site_id, temp_df, zc_df, window_size=30, depth_range=(-2, 20)):
    """Process a single site with optimized memory usage"""
    try:
        # Filter data for this site
        site_data = temp_df[temp_df['site_id'] == site_id].copy()
        site_data = site_data[site_data['depth'].between(depth_range[0], depth_range[1])]
        
        if len(site_data) < window_size:
            return []

        site_coords = site_data[['latitude', 'longitude']].iloc[0]
            
        # Create pivot table
        pivot = site_data.pivot_table(
            index='datetime',
            columns='depth',
            values='temperature',
            aggfunc='mean'
        ).sort_index()
        
        # Free memory
        del site_data
        gc.collect()
        
        if len(pivot.columns) < 4:  # Minimum number of depths
            return []
            
        windows = []
        chunk_size = 100  # Process windows in chunks
        
        # Process windows in chunks
        for i in range(0, len(pivot) - window_size + 1, chunk_size):
            end_idx = min(i + chunk_size, len(pivot) - window_size + 1)
            
            for j in range(i, end_idx):
                window = pivot.iloc[j:j+window_size]
                
                # Skip if too many missing values
                if window.isnull().sum().sum() / (window.shape[0] * window.shape[1]) > 0.3:
                    continue
                    
                # Check for zero curtain events
                window_start = window.index[0]
                window_end = window.index[-1]
                
                zc_events = zc_df[
                    (zc_df['site_id'] == site_id) &
                    (zc_df['start_date'] >= window_start) &
                    (zc_df['end_date'] <= window_end)
                ]
                
                # Fill missing values
                window_filled = window.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
                
                if not window_filled.isnull().any().any():
                    windows.append({
                        'window_data': window_filled.values,
                        'has_zero_curtain': 1 if not zc_events.empty else 0,
                        'window_start': window.index[0],
                        'window_end': window.index[-1],
                        'latitude': float(site_coords['latitude']),
                        'longitude': float(site_coords['longitude'])
                    })

            # Clear memory after each chunk
            gc.collect()
        
        return windows
        
    except Exception as e:
        print(f"Error processing site {site_id}: {str(e)}")
        return []

def preprocess_with_saving(temp_df, zc_df, save_dir='processed_data', sites_per_file=10):
    """Preprocess data with intermediate saving"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert datetime columns once
    print("Converting datetime columns...")
    temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], errors='coerce', format='mixed')
    zc_df['start_date'] = pd.to_datetime(zc_df['start_date'], errors='coerce', format='mixed')
    zc_df['end_date'] = pd.to_datetime(zc_df['end_date'], errors='coerce', format='mixed')
    
    unique_sites = temp_df['site_id'].unique()
    print(f"Total sites to process: {len(unique_sites)}")
    
    processed_count = 0
    current_batch = []
    batch_number = 0
    
    for site_idx, site_id in enumerate(unique_sites):
        print(f"\rProcessing site {site_idx + 1}/{len(unique_sites)}", end="")
        
        # Process single site
        site_windows = process_site_efficiently(site_id, temp_df, zc_df)
        
        if site_windows:
            current_batch.extend(site_windows)
            processed_count += len(site_windows)
        
        # Save batch if reached sites_per_file
        if (site_idx + 1) % sites_per_file == 0 and current_batch:
            batch_file = os.path.join(save_dir, f'batch_{batch_number}.npy')
            np.save(batch_file, current_batch)
            print(f"\nSaved batch {batch_number} with {len(current_batch)} windows")
            
            current_batch = []
            batch_number += 1
            gc.collect()
    
    # Save any remaining data
    if current_batch:
        batch_file = os.path.join(save_dir, f'batch_{batch_number}.npy')
        np.save(batch_file, current_batch)
        print(f"\nSaved final batch with {len(current_batch)} windows")
    
    print(f"\nTotal processed windows: {processed_count}")
    return save_dir

def load_and_combine_batches(data_dir):
    """Load and combine processed batches"""
    all_data = []
    batch_files = sorted([f for f in os.listdir(data_dir) if f.startswith('batch_')])
    
    for batch_file in batch_files:
        batch_data = np.load(os.path.join(data_dir, batch_file), allow_pickle=True)
        all_data.extend(batch_data)
        gc.collect()
    
    return all_data

# # #DO NOT RUN - TAKES FOREVER - ALREADY HAVE DATA! LOAD IT AFTER HERE!

# print("Loading data...")
# processed_dir = preprocess_with_saving(temp_df, zc_df, sites_per_file=5)
# print("\nLoading processed data for training...")
# processed_data = load_and_combine_batches(processed_dir)
# print(f"Total windows for training: {len(processed_data)}")

# # SAME WITH THIS!!!
# with open('processed_data.pkl','wb') as f:
#     pickle.dump(processed_data, f)

# Final Structure of processed_data
# After all steps, processed_data is a list of dictionaries, where:

# window_data is a 2D NumPy array (30-day × n_depths temperature matrix).
# has_zero_curtain is a binary label (1 if a zero curtain event is present, otherwise 0).
# Each dictionary in processed_data corresponds to one time window from a specific site.

# # UPDATE

# Each entry in processed_data is a dictionary with the following keys:

# processed_data[i] = {
#     'window_data': np.array([...]),   # 30-day x n_depths temperature matrix
#     'has_zero_curtain': 0 or 1,       # Label: 1 = Zero Curtain detected, 0 = No Zero Curtain
#     'window_start': Timestamp(...),   # Start datetime of the window (first day in 30-day period)
#     'window_end': Timestamp(...)      # End datetime of the window (last day in 30-day period)
# }

# Assume a 30-day temperature window from January 1, 1950, to January 30, 1950, at a specific site w...

# processed_data[0] = {
#     'window_data': np.array([
#         [-3.0, -3.1, -2.8, -3.2, -2.9],  # Day 1: temperatures at 5 depths
#         [-3.5, -3.2, -3.0, -3.3, -3.1],  # Day 2
#         [-3.7, -3.4, -3.2, -3.6, -3.4],  # Day 3
#         ...
#         [-2.9, -3.0, -2.7, -3.1, -2.8]   # Day 30
#     ]),  
#     'has_zero_curtain': 1,  # Zero curtain event detected
#     'window_start': Timestamp('1950-01-01 00:00:00'),
#     'window_end': Timestamp('1950-01-30 00:00:00')
# }

# Impact of These Changes
# What’s Improved?
# Time-Based Splitting is Now Possible

# We can now sort and segment the dataset by year instead of shuffling randomly.
# Enables chronological train-validation-test splitting.
# Preserves Temporal Order

# The dataset is structured in a way that reflects real-world time evolution.
# Better Model Generalization

# Since the model trains on earlier years and validates/tests on future years, it mimics real-world ...

# Once processed_data is regenerated with this structure, we sort it by window_start and apply the t...

# This ensures that training only uses past data, validation uses intermediate years, and testing ev...

# This structured approach results in a realistic, time-aware dataset that properly reflects the phy...



with open('processed_data.pkl', 'rb') as f:
    processed_data = pd.read_pickle(f)

for data in processed_data:
    data['window_start'] = pd.to_datetime(data['window_start'])
    data['window_end'] = pd.to_datetime(data['window_end'])

processed_data.sort(key=lambda x: x['window_start'])

all_years = sorted(set(d['window_start'].year for d in processed_data))



# def create_uniform_datasets(processed_data, n_depths=10, batch_size=32, val_split=0.2):
#     """Create datasets with uniform dimensions by selecting most common depths"""
#     print("Creating uniform datasets...")
    
#     # First, analyze depth distributions
#     all_depths = []
#     for data in processed_data:
#         window_shape = data['window_data'].shape
#         if len(window_shape) != 2:
#             continue
#         n_depths_window = window_shape[1]
#         all_depths.append(n_depths_window)
    
#     depth_counts = pd.Series(all_depths).value_counts()
#     print("\nDepth distribution:")
#     print(depth_counts.head())
    
#     # Select most common depth count
#     target_depths = depth_counts.index[0]
#     print(f"\nUsing {target_depths} depths (most common)")
    
#     # Filter and reshape data
#     valid_data = []
#     for data in processed_data:
#         window_shape = data['window_data'].shape
#         if len(window_shape) != 2:
#             continue
#         if window_shape[1] == target_depths:
#             valid_data.append(data)
    
#     print(f"\nValid windows after filtering: {len(valid_data)}")
    
#     # Create numpy arrays
#     X = np.array([d['window_data'] for d in valid_data])
#     y = np.array([d['has_zero_curtain'] for d in valid_data])
    
#     print("\nFinal data shape:", X.shape)
    
#     # Split indices
#     n_val = int(len(X) * val_split)
#     indices = np.random.permutation(len(X))
#     train_idx, val_idx = indices[n_val:], indices[:n_val]
    
#     # Create datasets
#     train_dataset = tf.data.Dataset.from_tensor_slices(
#         (X[train_idx], y[train_idx])
#     ).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
#     val_dataset = tf.data.Dataset.from_tensor_slices(
#         (X[val_idx], y[val_idx])
#     ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
#     return train_dataset, val_dataset, target_depths

# print("Creating training datasets...")
# train_dataset, val_dataset, n_depths = create_uniform_datasets(processed_data)



# def create_time_split_datasets(processed_data, n_depths=10, batch_size=32):
#     """Split processed data into training, validation, and test sets based on time sequence."""
    
#     print("Sorting data by time...")
#     for data in processed_data:
#         data['window_start'] = pd.to_datetime(data['window_start'])
#         data['window_end'] = pd.to_datetime(data['window_end'])
    
#     processed_data.sort(key=lambda x: x['window_start'])  # Sort chronologically
    
#     # Determine year boundaries
#     all_years = sorted(set(d['window_start'].year for d in processed_data))
#     total_years = len(all_years)
    
#     train_years = int(0.65 * total_years)
#     val_years = int(0.25 * total_years)
#     test_years = total_years - train_years - val_years

#     train_cutoff = all_years[train_years - 1]  # Last year included in training
#     val_cutoff = all_years[train_years + val_years - 1]  # Last year included in validation

#     # Assign data into splits
#     train_data = [d for d in processed_data if d['window_start'].year <= train_cutoff]
#     val_data = [d for d in processed_data if train_cutoff < d['window_start'].year <= val_cutoff]
#     test_data = [d for d in processed_data if d['window_start'].year > val_cutoff]

#     print(f"Train: {len(train_data)} windows (<= {train_cutoff})")
#     print(f"Validation: {len(val_data)} windows ({train_cutoff} - {val_cutoff})")
#     print(f"Test: {len(test_data)} windows (> {val_cutoff})")

#     # Ensure all data has uniform depth dimensions
#     target_depths = n_depths  # Use provided depth count

#     def filter_and_format(data):
#         filtered = [d for d in data if d['window_data'].shape[1] == target_depths]
#         X = np.array([d['window_data'] for d in filtered])
#         y = np.array([d['has_zero_curtain'] for d in filtered])
#         return X, y

#     X_train, y_train = filter_and_format(train_data)
#     X_val, y_val = filter_and_format(val_data)
#     X_test, y_test = filter_and_format(test_data)

#     print(f"\nFinal dataset sizes: Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_tes...

#     # Convert to TensorFlow datasets
#     def create_tf_dataset(X, y):
#         return tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).prefetch(tf.data.AUTOT...

#     train_dataset = create_tf_dataset(X_train, y_train)
#     val_dataset = create_tf_dataset(X_val, y_val)
#     test_dataset = create_tf_dataset(X_test, y_test)

#     return train_dataset, val_dataset, test_dataset, target_depths

# train_dataset, val_dataset, test_dataset, n_depths = create_time_split_datasets(processed_data)

def create_time_split_datasets(processed_data, n_depths=10, batch_size=32):
    """Split processed data into training (65%), validation (25%), and test (10%) ensuring correct weighting by windows."""
    
    print("Sorting data by time...")
    
    for data in processed_data:
        data['window_start'] = pd.to_datetime(data['window_start'])
        data['window_end'] = pd.to_datetime(data['window_end'])
    
    # Sort the dataset chronologically
    processed_data.sort(key=lambda x: x['window_start'])

    # Total dataset size
    total_windows = len(processed_data)
    train_size = int(0.65 * total_windows)  # 65% of dataset
    val_size = int(0.25 * total_windows)  # 25% of dataset
    test_size = total_windows - train_size - val_size  # Remaining 10%

    print(f"Total windows: {total_windows}")
    print(f"Target splits -> Train: {train_size}, Validation: {val_size}, Test: {test_size}")

    # Split dataset ensuring end-of-year continuity
    train_data, val_data, test_data = [], [], []
    current_year = processed_data[0]['window_start'].year

    count_train, count_val, count_test = 0, 0, 0

    for data in processed_data:
        year = data['window_start'].year

        if count_train < train_size:
            train_data.append(data)
            count_train += 1
        elif count_val < val_size:
            val_data.append(data)
            count_val += 1
        else:
            test_data.append(data)
            count_test += 1

        # Ensure we finish the current year before switching splits
        if count_train >= train_size and count_val < val_size and year > current_year:
            current_year = year
        elif count_train >= train_size and count_val >= val_size and count_test < test_size and year > current_year:
            current_year = year

    #print(f"Final dataset sizes -> Train: {len(train_data)}, Validation: {len(val_data)}, Test: {le...
    train_years = (train_data[0]['window_start'].year, train_data[-1]['window_start'].year) if train_data else (None, None)
    val_years = (val_data[0]['window_start'].year, val_data[-1]['window_start'].year) if val_data else (None, None)
    test_years = (test_data[0]['window_start'].year, test_data[-1]['window_start'].year) if test_data else (None, None)

    print(f"\nFinal dataset sizes -> Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    print(f"Train Year Range: {train_years[0]} - {train_years[1]}")
    print(f"Validation Year Range: {val_years[0]} - {val_years[1]}")
    print(f"Test Year Range: {test_years[0]} - {test_years[1]}")

    # Ensure all data has uniform depth dimensions
    target_depths = n_depths  # Use provided depth count

    def filter_and_format(data):
        filtered = [d for d in data if d['window_data'].shape[1] == target_depths]
        X = np.array([d['window_data'] for d in filtered])
        y = np.array([d['has_zero_curtain'] for d in filtered])
        coords = np.array([[d['latitude'], d['longitude']] for d in filtered])
        timestamps = np.array([d['window_start'] for d in filtered])
        return X, y, coords, timestamps

    X_train, y_train, train_coords, train_times = filter_and_format(train_data)
    X_val, y_val, val_coords, val_times = filter_and_format(val_data)
    X_test, y_test, test_coords, test_times = filter_and_format(test_data)

    print(f"\nFinal dataset sizes after filtering for uniform depths:")
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    # Convert to TensorFlow datasets
    def create_tf_dataset(X, y):
        return tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_dataset = create_tf_dataset(X_train, y_train)
    val_dataset = create_tf_dataset(X_val, y_val)
    test_dataset = create_tf_dataset(X_test, y_test)

    spatial_info = {
        'train': {'coords': train_coords, 'times': train_times},
        'val': {'coords': val_coords, 'times': val_times},
        'test': {'coords': test_coords, 'times': test_times}
    }

    return train_dataset, val_dataset, test_dataset, target_depths, spatial_info

train_dataset, val_dataset, test_dataset, n_depths, spatial_info = create_time_split_datasets(processed_data)



####

####



model_params = {
    'temporal_window': 30,
    'n_depths': n_depths,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10
}

print("\nInitializing model with dimensions:")
print(f"- Temporal window: {model_params['temporal_window']}")
print(f"- Number of depths: {model_params['n_depths']}")

training_config = {
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10
}



# for x, y in train_dataset.take(1):
#     print("Input shape:", x.shape)
#     print("Label shape:", y.shape)
#     input_shape = x.shape[1:]
#     break

for x, y in train_dataset.take(1):
    print("Input shape:", x.shape)
    print("Label shape:", y.shape)
    input_shape = x.shape[1:]

model.summary()











print(f"TensorFlow version: {tf.__version__}")
print("\nDevice configuration:")
print(tf.config.list_physical_devices())

tf.keras.backend.clear_session()

# x_small = x_train[:32]  # Just one batch
# y_small = y_train[:32]  # Just one batch

# print("\nTesting with single batch:")
# print(f"x shape: {x_small.shape}")
# print(f"y shape: {y_small.shape}")



#TROUBLESHOOTING

# print("\nChecking TensorFlow installation:")
# !python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

# print("Testing basic TensorFlow operations...")

# a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
# b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# print("Matrix multiplication:", tf.matmul(a, b).numpy())
# print("Addition:", (a + b).numpy())

# print("\nTesting dataset iteration...")
# small_dataset = tf.data.Dataset.from_tensor_slices(
#     (tf.random.normal((10, 30, 4)), tf.random.uniform((10,), maxval=2, dtype=tf.int32))
# ).batch(2)

# for batch_idx, (x, y) in enumerate(small_dataset):
#     print(f"Batch {batch_idx} shapes: x={x.shape}, y={y.shape}")

# test_x = tf.random.normal((32, 30, 4))
# test_y = tf.random.uniform((32,), maxval=2, dtype=tf.int32)



# with tf.device('/CPU:0'):
#     # Create mini-batches manually
#     batch_size = 16
#     n_samples = len(x_train)
    
#     # Remove @tf.function decorator
#     def train_step(x, y):
#         with tf.GradientTape() as tape:
#             predictions = model(x, training=True)
#             loss = loss_fn(y, predictions)
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         return loss, predictions
    
#     print("Starting training with manual mini-batches...")
    
#     for epoch in range(5):
#         print(f"\nEpoch {epoch + 1}")
        
#         # Reset metrics
#         train_acc_metric.reset_state()
#         train_precision.reset_state()
#         train_recall.reset_state()
#         train_auc.reset_state()
        
#         # Process each mini-batch
#         for start_idx in range(0, n_samples, batch_size):
#             end_idx = min(start_idx + batch_size, n_samples)
            
#             x_batch = x_train[start_idx:end_idx]
#             y_batch = y_train[start_idx:end_idx]
            
#             loss, predictions = train_step(x_batch, y_batch)
            
#             # Update metrics
#             train_acc_metric.update_state(y_batch, predictions)
#             train_precision.update_state(y_batch, predictions)
#             train_recall.update_state(y_batch, predictions)
#             train_auc.update_state(y_batch, predictions)
        
#         # Print epoch metrics
#         print(f"Loss: {loss.numpy():.4f}")
#         print(f"Accuracy: {train_acc_metric.result().numpy():.4f}")
#         print(f"Precision: {train_precision.result().numpy():.4f}")
#         print(f"Recall: {train_recall.result().numpy():.4f}")
#         print(f"AUC: {train_auc.result().numpy():.4f}")

# print("\nTraining complete!")

# Full Model with training subset (10 epochs)

# with tf.device('/CPU:0'):
#     # Training configuration
#     batch_size = 32
#     n_samples = len(x_train)
#     n_epochs = 10
    
#     def train_step(x, y):
#         with tf.GradientTape() as tape:
#             predictions = model(x, training=True)
#             loss = loss_fn(y, predictions)
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         return loss, predictions
    
#     print(f"Starting training on {n_samples} samples...")
#     best_auc = 0
#     best_weights = None
    
#     for epoch in range(n_epochs):
#         print(f"\nEpoch {epoch + 1}/{n_epochs}")
#         epoch_loss = 0
#         n_batches = int(np.ceil(n_samples / batch_size))
        
#         # Reset metrics
#         train_acc_metric.reset_state()
#         train_precision.reset_state()
#         train_recall.reset_state()
#         train_auc.reset_state()
        
#         # Shuffle data
#         perm = np.random.permutation(n_samples)
#         x_train_shuffled = x_train[perm]
#         y_train_shuffled = y_train[perm]
        
#         # Process batches
#         for batch_idx in range(n_batches):
#             start_idx = batch_idx * batch_size
#             end_idx = min(start_idx + batch_size, n_samples)
            
#             x_batch = x_train_shuffled[start_idx:end_idx]
#             y_batch = y_train_shuffled[start_idx:end_idx]
            
#             loss, predictions = train_step(x_batch, y_batch)
#             epoch_loss += loss.numpy()
            
#             # Update metrics
#             train_acc_metric.update_state(y_batch, predictions)
#             train_precision.update_state(y_batch, predictions)
#             train_recall.update_state(y_batch, predictions)
#             train_auc.update_state(y_batch, predictions)
            
#             # Print batch progress
#             print(f"\rBatch {batch_idx + 1}/{n_batches}", end='')
        
#         # Calculate epoch metrics
#         epoch_loss = epoch_loss / n_batches
#         epoch_acc = train_acc_metric.result().numpy()
#         epoch_prec = train_precision.result().numpy()
#         epoch_recall = train_recall.result().numpy()
#         epoch_auc = train_auc.result().numpy()
        
#         # Print epoch results
#         print(f"\nLoss: {epoch_loss:.4f}")
#         print(f"Accuracy: {epoch_acc:.4f}")
#         print(f"Precision: {epoch_prec:.4f}")
#         print(f"Recall: {epoch_recall:.4f}")
#         print(f"AUC: {epoch_auc:.4f}")
        
#         # Save best model
#         if epoch_auc > best_auc:
#             best_auc = epoch_auc
#             best_weights = model.get_weights()
#             print("New best model saved!")

#     # Restore best weights
#     if best_weights is not None:
#         model.set_weights(best_weights)
#         print(f"\nRestored best model with AUC: {best_auc:.4f}")

# print("\nTraining complete!")

# train_dataset.element_spec[0].shape
# TensorShape([None, 30, 4])

# with tf.device('/CPU:0'):
#     # Training configuration
#     batch_size = 32
#     n_samples = len(x_train)
#     n_epochs = 10
    
#     def train_step(x, y):
#         with tf.GradientTape() as tape:
#             predictions = model(x, training=True)
#             loss = loss_fn(y, predictions)
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         return loss, predictions
    
#     print(f"Starting training on {n_samples} samples...")
#     best_auc = 0
#     best_weights = None
    
#     for epoch in range(n_epochs):
#         print(f"\nEpoch {epoch + 1}/{n_epochs}")
#         epoch_loss = 0
#         n_batches = int(np.ceil(n_samples / batch_size))
        
#         # Reset metrics
#         train_acc_metric.reset_state()
#         train_precision.reset_state()
#         train_recall.reset_state()
#         train_auc.reset_state()
        
#         # Shuffle data
#         perm = np.random.permutation(n_samples)
#         x_train_shuffled = x_train[perm]
#         y_train_shuffled = y_train[perm]
        
#         # Process batches
#         for batch_idx in range(n_batches):
#             start_idx = batch_idx * batch_size
#             end_idx = min(start_idx + batch_size, n_samples)
            
#             x_batch = x_train_shuffled[start_idx:end_idx]
#             y_batch = y_train_shuffled[start_idx:end_idx]
            
#             loss, predictions = train_step(x_batch, y_batch)
#             epoch_loss += loss.numpy()
            
#             # Update metrics
#             train_acc_metric.update_state(y_batch, predictions)
#             train_precision.update_state(y_batch, predictions)
#             train_recall.update_state(y_batch, predictions)
#             train_auc.update_state(y_batch, predictions)
            
#             # Print batch progress
#             print(f"\rBatch {batch_idx + 1}/{n_batches}", end='')
        
#         # Calculate epoch metrics
#         epoch_loss = epoch_loss / n_batches
#         epoch_acc = train_acc_metric.result().numpy()
#         epoch_prec = train_precision.result().numpy()
#         epoch_recall = train_recall.result().numpy()
#         epoch_auc = train_auc.result().numpy()
        
#         # Print epoch results
#         print(f"\nLoss: {epoch_loss:.4f}")
#         print(f"Accuracy: {epoch_acc:.4f}")
#         print(f"Precision: {epoch_prec:.4f}")
#         print(f"Recall: {epoch_recall:.4f}")
#         print(f"AUC: {epoch_auc:.4f}")
        
#         # Save best model
#         if epoch_auc > best_auc:
#             best_auc = epoch_auc
#             best_weights = model.get_weights()
#             print("New best model saved!")

#     # Restore best weights
#     if best_weights is not None:
#         model.set_weights(best_weights)
#         print(f"\nRestored best model with AUC: {best_auc:.4f}")

# print("\nTraining complete!")



# Input shape: (32, 30, 10)
# Label shape: (32,)
# Model: "functional_5"
# 
#  Layer (type)                     Output Shape                  Param # 
# 
#  input_layer_5 (InputLayer)       (None, 30, 10)                      0 
# 
#  reshape_10 (Reshape)             (None, 30, 10, 1)                   0 
# 
#  conv2d_10 (Conv2D)               (None, 30, 10, 32)                320 
# 
#  batch_normalization_5            (None, 30, 10, 32)                128 
#  (BatchNormalization)                                                   
# 
#  conv2d_11 (Conv2D)               (None, 30, 10, 64)             18,496 
# 
#  physics_layer_5 (PhysicsLayer)   (None, 30, 10, 64)                  1 
# 
#  reshape_11 (Reshape)             (None, 30, 2, 5, 64)                0 
# 
#  conv_lstm2d_5 (ConvLSTM2D)       (None, 2, 5, 32)               49,280 
# 
#  flatten_5 (Flatten)              (None, 320)                         0 
# 
#  dense_10 (Dense)                 (None, 32)                     10,272 
# 
#  dense_11 (Dense)                 (None, 1)                          33 
# 
#  Total params: 78,530 (306.76 KB)
#  Trainable params: 78,466 (306.51 KB)
#  Non-trainable params: 64 (256.00 B)
# Starting training on 9194 samples...

# Epoch 1/10
# Batch 288/288
# Train Loss: 0.1703, Val Loss: 0.2885
# Accuracy: 0.9538
# Precision: 0.9939
# Recall: 0.9594
# AUC: 0.2815
# New best model saved!

# Epoch 2/10
# Batch 288/288
# Train Loss: 0.0411, Val Loss: 0.2315
# Accuracy: 0.9934
# Precision: 0.9941
# Recall: 0.9992
# AUC: 0.7496
# New best model saved!

# Epoch 3/10
# Batch 288/288
# Train Loss: 0.0397, Val Loss: 0.2044
# Accuracy: 0.9941
# Precision: 0.9941
# Recall: 1.0000
# AUC: 0.7072

# Epoch 4/10
# Batch 288/288
# Train Loss: 0.0339, Val Loss: 0.2048
# Accuracy: 0.9941
# Precision: 0.9941
# Recall: 1.0000
# AUC: 0.7705
# New best model saved!

# Epoch 5/10
# Batch 288/288
# Train Loss: 0.0297, Val Loss: 0.2891
# Accuracy: 0.9941
# Precision: 0.9941
# Recall: 1.0000
# AUC: 0.7423

# Epoch 6/10
# Batch 288/288
# Train Loss: 0.0278, Val Loss: 0.2040
# Accuracy: 0.9941
# Precision: 0.9941
# Recall: 1.0000
# AUC: 0.7819
# New best model saved!

# Epoch 7/10
# Batch 288/288
# Train Loss: 0.0310, Val Loss: 0.1816
# Accuracy: 0.9941
# Precision: 0.9941
# Recall: 1.0000
# AUC: 0.7044

# Epoch 8/10
# Batch 288/288
# Train Loss: 0.0263, Val Loss: 0.2146
# Accuracy: 0.9941
# Precision: 0.9941
# Recall: 1.0000
# AUC: 0.7659

# Epoch 9/10
# Batch 288/288
# Train Loss: 0.0264, Val Loss: 0.2619
# Accuracy: 0.9941
# Precision: 0.9941
# Recall: 1.0000
# AUC: 0.7787

# Epoch 10/10
# Batch 288/288
# Train Loss: 0.0284, Val Loss: 0.2422
# Accuracy: 0.9949
# Precision: 0.9949
# Recall: 1.0000
# AUC: 0.7847
# New best model saved!

# Restored best model with AUC: 0.7847

# Training complete!

# model.save_weights('model.weights.h5')
# #model.load_weights('model.weights.h5')



def convert_to_serializable(obj):
    """Convert numpy/tensorflow types to Python native types"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (tf.Tensor, tf.Variable)):
        return float(obj.numpy())
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

best_model, best_hyperparameters, results = train_hyperparameter_optimization(
    x_train, y_train, 
    x_val, y_val, 
    input_shape=(30, 10),
    spatial_info=spatial_info
)



hyperparameter_files = glob.glob('best_hyperparameters_*.json')
latest_hp_file = max(hyperparameter_files, key=os.path.getctime)

with open(latest_hp_file, 'r') as f:
    best_hyperparameters = json.load(f)

print("Loading hyperparameters from:", latest_hp_file)
print("\nBest hyperparameters found:")
for param, value in best_hyperparameters.items():
    print(f"{param}: {value}")

best_model = build_final_model_from_hyperparameters(best_hyperparameters)

model_files = glob.glob('best_model_intermediate_*.h5')

if model_files:
    latest_model_file = max(model_files, key=os.path.getctime)
    print("\nLoading weights from:", latest_model_file)
    best_model.load_weights(latest_model_file)

#best_model.weights

batch_size = 32
n_batches = int(np.ceil(len(x_test) / batch_size))
print(f"Processing {len(x_test)} samples in {n_batches} batches...")





# 6. Performance Metrics by Latitude Band
lat_bands = pd.qcut(test_coords[:, 0], q=5, duplicates='drop')
lat_performance = []

for band in lat_bands.unique():
    mask = lat_bands == band
    band_pred = predictions[mask]
    band_true = y_true[mask]
    
    lat_performance.append({
        'band': f"{band.left:.1f} to {band.right:.1f}",
        'accuracy': accuracy_score(band_true, band_pred > 0.5),
        'precision': precision_score(band_true, band_pred > 0.5),
        'recall': recall_score(band_true, band_pred > 0.5),
        'auc': roc_auc_score(band_true, band_pred)
    })

lat_perf_df = pd.DataFrame(lat_performance)
print("\nPerformance by Latitude Band:")
print(lat_perf_df.to_string(index=False))

best_model.input.shape

predictions.shape

# First, let's check what we're actually working with
print("Training coordinates shape:", spatial_info['train']['coords'].shape)
print("Sample of training coordinates:")
print(spatial_info['train']['coords'][:5])
print("\nSample of training predictions:")
print(train_pred[:5])

# Let's also verify our prediction arrays
print("\nPrediction array shapes:")
print(f"Training: {train_pred.shape}")
print(f"Validation: {val_pred.shape}")
print(f"Test: {test_pred.shape}")

# Check for any NaN or invalid values in coordinates
print("\nChecking for NaN values in coordinates:")
print("Training NaN:", np.isnan(spatial_info['train']['coords']).any())
print("Validation NaN:", np.isnan(spatial_info['val']['coords']).any())
print("Test NaN:", np.isnan(spatial_info['test']['coords']).any())

# Check coordinate ranges
print("\nCoordinate ranges:")
print("Training lat range:", np.min(spatial_info['train']['coords'][:,0]), "-", np.max(spatial_info['train']['coords'][:,0]))
print("Training lon range:", np.min(spatial_info['train']['coords'][:,1]), "-", np.max(spatial_info['train']['coords'][:,1]))

Training coordinates shape: (9194, 2)
Sample of training coordinates:
[[46.4964    9.931076]
 [46.4964    9.931076]
 [46.4964    9.931076]
 [46.4964    9.931076]
 [46.4964    9.931076]]

Sample of training predictions:
[[0.9986811 ]
 [0.9986395 ]
 [0.99859613]
 [0.99855095]
 [0.9985066 ]]

Prediction array shapes:
Training: (9194, 1)
Validation: (2115, 1)
Test: (1776, 1)

Checking for NaN values in coordinates:
Training NaN: False
Validation NaN: False
Test NaN: False

Coordinate ranges:
Training lat range: 46.083705 - 68.2875
Training lon range: 7.302472 - 54.498611

create_focused_regional_maps(train_pred, val_pred, test_pred, spatial_info)





test_metrics = best_model.evaluate(x_test, y_test, batch_size=32, verbose=1)

predictions = best_model.predict(x_test)





















 #UAVSAR

import json

def polygon_within_bounds(coord_str):
    try:
        coords = ast.literal_eval(coord_str)  # Convert from string to list
        if isinstance(coords, list):
            for polygon in coords:  # Iterate through polygons
                for point in polygon:  # Iterate through individual points
                    lon, lat = point  # GeoJSON is [longitude, latitude]
                    if 49 <= lat <= 90 and -180 <= lon <= 180:
                        return True  # At least one point is within bounds
    except Exception as e:
        print(f"Error processing coordinates: {e}")
        return False
    return False

geojson_path = "/Users/bgay/Downloads/AllFlownUAVSARSwaths.geojson"
csv_path = "/Users/bgay/Downloads/AllFlownUAVSARSwaths.csv"

with open(geojson_path, "r") as file:
    geojson_data = json.load(file)

features = geojson_data.get("features", [])

data = []
for feature in features:
    properties = feature.get("properties", {})
    geometry = feature.get("geometry", {})
    geometry_type = geometry.get("type", "")
    coordinates = json.dumps(geometry.get("coordinates", ""))

    properties["geometry_type"] = geometry_type
    properties["coordinates"] = coordinates
    data.append(properties)

df = pd.DataFrame(data)
#df.to_csv(csv_path, index=False)

filtered_df = df[df["coordinates"].apply(polygon_within_bounds)]
print(filtered_df)
filtered_df[filtered_df.Band=='L'].sort_values('Line ID').reset_index(drop=True).drop(columns=['Disclaimer']).to_csv\
('uavsar_lines.csv',index=False)

asf_datapool_file = "/Users/bgay/Downloads/asf-datapool-results-2025-02-19_20-15-57.csv"
uavsar_lines_file = "/Users/bgay/Downloads/uavsar_lines.csv"
asf_df = pd.read_csv(asf_datapool_file)
uavsar_df = pd.read_csv(uavsar_lines_file)
asf_df.head(), uavsar_df.head()



asf_df["Line ID"] = asf_df["GroupID"].str.split("_").str[2]
asf_df["Line ID"] = asf_df["Line ID"].astype(str)
uavsar_df["Line ID"] = uavsar_df["Line ID"].astype(str)
merged_df = uavsar_df.merge(asf_df, on="Line ID", how="inner")

merged_df

if "Processing Level" in merged_df.columns:
    filtered_df = merged_df[merged_df["Processing Level"] == "INTERFEROMETRY"]
else:
    filtered_df = merged_df

filtered_df

print(merged_df['Granule Name'][:30].tolist())

['UA_Grnlnd_00002_09029-007_09030-008_0001d_s01_L090_01', 
 'UA_Grnlnd_00002_09029-007_09030-008_0001d_s01_L090_01', 
 'UA_Grnlnd_00002_09029-007_09030-008_0001d_s01_L090_01', 
 'UA_Grnlnd_00002_09029-007_09030-008_0001d_s01_L090_01', 
 'UA_Grnlnd_00002_09029-007_09030-008_0001d_s01_L090_01', 
 'UA_Grnlnd_00002_09029-007_09030-008_0001d_s01_L090_01', 
 'UA_Grnlnd_00002_09029-007_09030-008_0001d_s01_L090_01', 
 'UA_Grnlnd_00002_09029-007_09032-008_0006d_s01_L090_01', 
 'UA_Grnlnd_00002_09029-007_09032-008_0006d_s01_L090_01', 
 'UA_Grnlnd_00002_09029-007_09032-008_0006d_s01_L090_01', 
 'UA_Grnlnd_00002_09029-007_09032-008_0006d_s01_L090_01', 
 'UA_Grnlnd_00002_09029-007_09032-008_0006d_s01_L090_01', 
 'UA_Grnlnd_00002_09029-007_09032-008_0006d_s01_L090_01', 
 'UA_Grnlnd_00002_09029-007_09032-008_0006d_s01_L090_01', 
 'UA_Grnlnd_00002_09030-008_09032-008_0005d_s01_L090_01', 
 'UA_Grnlnd_00002_09030-008_09032-008_0005d_s01_L090_01', 
 'UA_Grnlnd_00002_09030-008_09032-008_0005d_s01_L090_01', 
 'UA_Grnlnd_00002_09030-008_09032-008_0005d_s01_L090_01', 
 'UA_Grnlnd_00002_09030-008_09032-008_0005d_s01_L090_01', 
 'UA_Grnlnd_00002_09030-008_09032-008_0005d_s01_L090_01', 
 'UA_Grnlnd_00002_09030-008_09032-008_0005d_s01_L090_01', 
 'UA_Grnlnd_00003_09029-005_09030-006_0001d_s01_L090_01', 
 'UA_Grnlnd_00003_09029-005_09030-006_0001d_s01_L090_01', 
 'UA_Grnlnd_00003_09029-005_09030-006_0001d_s01_L090_01', 
 'UA_Grnlnd_00003_09029-005_09030-006_0001d_s01_L090_01', 
 'UA_Grnlnd_00003_09029-005_09030-006_0001d_s01_L090_01', 
 'UA_Grnlnd_00003_09029-005_09030-006_0001d_s01_L090_01', 
 'UA_Grnlnd_00003_09029-005_09030-006_0001d_s01_L090_01', 
 'UA_Grnlnd_00004_09029-008_09030-009_0001d_s01_L090_01', 
 'UA_Grnlnd_00004_09029-008_09030-009_0001d_s01_L090_01']

print(filtered_df['Granule Name'][:30].tolist())

['UA_Grnlnd_00002_09029-007_09030-008_0001d_s01_L090_01', 
 'UA_Grnlnd_00002_09029-007_09032-008_0006d_s01_L090_01', 
 'UA_Grnlnd_00002_09030-008_09032-008_0005d_s01_L090_01', 
 'UA_Grnlnd_00003_09029-005_09030-006_0001d_s01_L090_01', 
 'UA_Grnlnd_00004_09029-008_09030-009_0001d_s01_L090_01', 
 'UA_Grnlnd_00005_09030-007_09032-007_0005d_s01_L090_01']


#merged_df['Granule Name'].nunique() #nunique()=201
#merged_df['Granule Name'].unique()

merged_df.nunique()

filtered_df.nunique()









































































































