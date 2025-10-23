import pyarrow.parquet as pq
import pandas as pd

# Load SMAP file
smap_path = "/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/smap/smap.parquet"

print("="*80)
print("SMAP PARQUET FILE INSPECTION")
print("="*80)

# Method 1: PyArrow schema
print("\n1. PyArrow Schema:")
parquet_file = pq.ParquetFile(smap_path)
print(parquet_file.schema)

# Method 2: Read first row group
print("\n2. First Row Group Sample:")
table = parquet_file.read_row_group(0)
df_sample = table.to_pandas()
print(f"\nShape: {df_sample.shape}")
print(f"\nColumns: {df_sample.columns.tolist()}")
print(f"\nData types:\n{df_sample.dtypes}")
print(f"\nFirst 3 rows:\n{df_sample.head(3)}")

# Method 3: Check for nested structures
print("\n3. Checking for nested structures...")
for col in df_sample.columns:
    print(f"  {col}: {type(df_sample[col].iloc[0]) if len(df_sample) > 0 else 'N/A'}")

print("\n" + "="*80)
