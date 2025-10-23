# Add this at the top of zero_curtain_ml_model_update.py
# Or create as separate file and import

class MemoryEfficientDataset(Dataset):
    """Ultra-lightweight dataset with no caching."""
    
    def __init__(self, parquet_file, sequence_length=30):
        self.parquet_file = parquet_file
        self.sequence_length = sequence_length
        
        # Only load metadata, not actual data
        import pyarrow.parquet as pq
        self.parquet_file_obj = pq.ParquetFile(parquet_file)
        self.num_rows = self.parquet_file_obj.metadata.num_rows
        
    def __len__(self):
        return max(0, self.num_rows - self.sequence_length)
    
    def __getitem__(self, idx):
        # Read ONLY the rows needed for this sequence
        # This is the key: no loading entire dataset into memory
        start_row = idx
        end_row = idx + self.sequence_length
        
        # Read just these rows from disk
        table = self.parquet_file_obj.read_row_group(0, columns=['all'])
        df = table.to_pandas().iloc[start_row:end_row]
        
        # Process and return
        # ... feature extraction ...
        
        return features, targets, sequence_length
