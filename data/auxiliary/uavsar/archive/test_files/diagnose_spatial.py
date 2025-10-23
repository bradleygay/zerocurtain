import pandas as pd
import pyarrow.parquet as pq

# Load displacement bounds
print("DISPLACEMENT DATA:")
print("  Lat: [61.11, 61.94]")
print("  Lon: [-165.78, -164.98]")

# Sample SMAP data in this region
smap_path = "/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/smap/smap.parquet"
parquet_file = pq.ParquetFile(smap_path)

# Read first few batches and check coordinates
for i in range(5):
    table = parquet_file.read_row_group(i)
    df = table.to_pandas()
    
    # Filter to Alaska region
    mask = (
        (df['lat'] >= 60) & (df['lat'] <= 63) &
        (df['lon'] >= -167) & (df['lon'] <= -163)
    )
    
    alaska = df[mask]
    
    if len(alaska) > 0:
        print(f"\nSMAP BATCH {i} (Alaska region):")
        print(f"  Records: {len(alaska)}")
        print(f"  Lat range: [{alaska['lat'].min():.2f}, {alaska['lat'].max():.2f}]")
        print(f"  Lon range: [{alaska['lon'].min():.2f}, {alaska['lon'].max():.2f}]")
        print(f"  Date range: {alaska['datetime'].min()} to {alaska['datetime'].max()}")
        
        # Check overlap with displacement
        overlap = (
            (alaska['lat'] >= 61.11) & (alaska['lat'] <= 61.94) &
            (alaska['lon'] >= -165.78) & (alaska['lon'] <= -164.98)
        )
        print(f"  Records in displacement bounds: {overlap.sum()}")
        break

print("\nDISPLACEMENT DATES:")
print("  May 27, 2017 to August 14, 2017")
