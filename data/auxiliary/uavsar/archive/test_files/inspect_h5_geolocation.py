import h5py
import numpy as np

h5_file = "/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar/nisar_downloads/chevak_01811_17055_004_170527_PL09043020_30_CX_02.h5"

print("="*80)
print("H5 GEOLOCATION INSPECTION")
print("="*80)

with h5py.File(h5_file, 'r') as f:
    print("\n1. TOP-LEVEL STRUCTURE:")
    print(list(f.keys()))
    
    # Recursively find all datasets
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name} | shape: {obj.shape} | dtype: {obj.dtype}")
    
    print("\n2. ALL DATASETS:")
    f.visititems(print_structure)
    
    # Look for latitude/longitude
    print("\n3. SEARCHING FOR COORDINATE DATA:")
    
    for key in ['latitude', 'lat', 'Latitude', 'Lat']:
        if key in f:
            print(f"\nFound '{key}':")
            lat_data = f[key][:]
            print(f"  Shape: {lat_data.shape}")
            print(f"  Min: {np.nanmin(lat_data):.4f}, Max: {np.nanmax(lat_data):.4f}")
            print(f"  Mean: {np.nanmean(lat_data):.4f}")
            print(f"  Sample values:\n{lat_data[:5, :5]}")
    
    for key in ['longitude', 'lon', 'Longitude', 'Lon']:
        if key in f:
            print(f"\nFound '{key}':")
            lon_data = f[key][:]
            print(f"  Shape: {lon_data.shape}")
            print(f"  Min: {np.nanmin(lon_data):.4f}, Max: {np.nanmax(lon_data):.4f}")
            print(f"  Mean: {np.nanmean(lon_data):.4f}")
            print(f"  Sample values:\n{lon_data[:5, :5]}")
    
    # Check for metadata
    print("\n4. GLOBAL ATTRIBUTES:")
    for attr in f.attrs:
        print(f"  {attr}: {f.attrs[attr]}")
    
    # Check SLC group
    if 'SLC' in f:
        print("\n5. SLC GROUP ATTRIBUTES:")
        for attr in f['SLC'].attrs:
            print(f"  {attr}: {f['SLC'].attrs[attr]}")

print("="*80)
