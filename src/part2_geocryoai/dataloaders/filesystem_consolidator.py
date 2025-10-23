import torch
#!/usr/bin/env python3
"""
Filesystem-Optimized Batch Consolidation for [RESEARCHER]
Consolidates 19 small files into larger chunks optimized for parallel filesystem
"""

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from typing import List, Dict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import psutil


class FilesystemOptimizedConsolidator:
    """
    Consolidates fragmented sequence files into large blocks optimized
    for Discover's Lustre parallel filesystem.
    
    Key optimizations:
    1. Consolidate 19 files → 4-8 large files (>10GB each)
    2. Align to Lustre stripe size (typically 1-4MB)
    3. Sequential read patterns for HDD-based storage
    4. Parallel consolidation using multiple processes
    """
    
    def __init__(self,
                 source_dir: Path,
                 output_dir: Path,
                 target_file_size_gb: float = 15.0,
                 lustre_stripe_size_mb: float = 4.0,
                 num_processes: int = 8):
        """
        Initialize filesystem consolidator.
        
        Args:
            source_dir: Directory with fragmented sequence files
            output_dir: Output directory for consolidated files
            target_file_size_gb: Target size for each consolidated file
            lustre_stripe_size_mb: Lustre filesystem stripe size
            num_processes: Number of parallel consolidation processes
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_file_size_bytes = int(target_file_size_gb * 1024**3)
        self.lustre_stripe_size = int(lustre_stripe_size_mb * 1024**2)
        self.num_processes = num_processes
        
        print(f"\n Filesystem-Optimized Consolidation")
        print(f"{'='*60}")
        print(f"  Source: {source_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Target file size: {target_file_size_gb:.1f} GB")
        print(f"  Lustre stripe size: {lustre_stripe_size_mb:.1f} MB")
        print(f"  Processes: {num_processes}")
    
    def consolidate_sequences(self) -> List[Path]:
        """
        Consolidate fragmented sequence files.
        
        Returns:
            List of consolidated file paths
        """
        
        # Find all source files
        source_files = sorted(self.source_dir.glob('**/*sequence*.parquet'))
        
        if not source_files:
            raise ValueError(f"No sequence files found in {self.source_dir}")
        
        print(f"\n  Found {len(source_files)} source files")
        
        # Analyze source file sizes
        total_size = sum(f.stat().st_size for f in source_files)
        print(f"  Total data size: {total_size / 1024**3:.2f} GB")
        
        # Calculate optimal number of output files
        num_output_files = max(1, int(np.ceil(total_size / self.target_file_size_bytes)))
        print(f"  Creating {num_output_files} consolidated files")
        
        # Group source files into chunks
        file_groups = self._group_files(source_files, num_output_files)
        
        # Consolidate each group in parallel
        print(f"\n  Consolidating files...")
        
        consolidated_files = []
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = []
            
            for group_id, file_group in enumerate(file_groups):
                output_file = self.output_dir / f'consolidated_sequences_{group_id:03d}.parquet'
                
                future = executor.submit(
                    self._consolidate_group,
                    file_group,
                    output_file,
                    group_id
                )
                futures.append((future, output_file))
            
            # Collect results
            for future, output_file in futures:
                try:
                    success = future.result()
                    if success:
                        consolidated_files.append(output_file)
                        print(f"     Created: {output_file.name} ({output_file.stat().st_size / 1024**3:.2f} GB)")
                except Exception as e:
                    print(f"     Error consolidating {output_file.name}: {e}")
        
        print(f"\n  Consolidation complete!")
        print(f"  Output files: {len(consolidated_files)}")
        print(f"  Total size: {sum(f.stat().st_size for f in consolidated_files) / 1024**3:.2f} GB")
        print(f"{'='*60}\n")
        
        return consolidated_files
    
    def _group_files(self, source_files: List[Path], num_groups: int) -> List[List[Path]]:
        """
        Group source files to balance output file sizes.
        """
        # Calculate file sizes
        file_sizes = [(f, f.stat().st_size) for f in source_files]
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize groups
        groups = [[] for _ in range(num_groups)]
        group_sizes = [0] * num_groups
        
        # Greedy bin packing - assign largest files first
        for file_path, file_size in file_sizes:
            # Find group with smallest current size
            min_group_idx = np.argmin(group_sizes)
            
            groups[min_group_idx].append(file_path)
            group_sizes[min_group_idx] += file_size
        
        return groups
    
    def _consolidate_group(self, 
                          file_group: List[Path],
                          output_file: Path,
                          group_id: int) -> bool:
        """
        Consolidate a group of files into a single output file.
        
        Args:
            file_group: List of files to consolidate
            output_file: Output file path
            group_id: Group identifier for logging
        
        Returns:
            True if successful
        """
        try:
            # Read all tables
            tables = []
            
            for file_path in file_group:
                table = pq.read_table(file_path)
                tables.append(table)
            
            # Concatenate tables
            combined_table = pa.concat_tables(tables)
            
            # Write with optimized settings for Lustre
            pq.write_table(
                combined_table,
                output_file,
                compression='snappy',  # Fast compression
                row_group_size=131072,  # 128K rows per row group
                data_page_size=self.lustre_stripe_size,  # Align to stripe size
                use_dictionary=True,
                write_statistics=True,
                version='2.6'
            )
            
            # Verify write
            if output_file.exists():
                # Clear memory
                del tables
                del combined_table
                gc.collect()
                
                return True
            else:
                return False
        
        except Exception as e:
            print(f"    Error in group {group_id}: {e}")
            return False
    
    def set_lustre_striping(self, file_path: Path, stripe_count: int = 4):
        """
        Set Lustre striping parameters for optimal I/O.
        
        Args:
            file_path: File to stripe
            stripe_count: Number of OSTs to stripe across
        """
        import subprocess
        
        try:
            # Set stripe count and size using lfs setstripe
            subprocess.run([
                'lfs', 'setstripe',
                '-c', str(stripe_count),
                '-S', f'{self.lustre_stripe_size}',
                str(file_path)
            ], check=True)
            
            print(f"    Set Lustre striping: {stripe_count} stripes × {self.lustre_stripe_size / 1024**2:.1f} MB")
        
        except Exception as e:
            print(f"    Warning: Could not set Lustre striping: {e}")


class ConsolidatedSequenceDataset(torch.utils.data.Dataset):
    """
    Dataset for consolidated sequence files.
    """
    
    def __init__(self, consolidated_files: List[Path]):
        """
        Initialize dataset from consolidated files.
        
        Args:
            consolidated_files: List of consolidated parquet files
        """
        super().__init__()
        
        self.consolidated_files = [Path(f) for f in consolidated_files]
        
        # Build index
        self.file_index = self._build_index()
        self.total_sequences = sum(count for _, count in self.file_index.values())
        
        # Memory-map files
        self.tables = {}
        for file_path in self.consolidated_files:
            self.tables[file_path] = pq.ParquetFile(file_path)
        
        print(f"  ConsolidatedSequenceDataset initialized:")
        print(f"    Files: {len(self.consolidated_files)}")
        print(f"    Total sequences: {self.total_sequences:,}")
    
    def _build_index(self):
        """
        Build index for fast lookup.
        """
        file_index = {}
        cumulative = 0
        
        for file_path in self.consolidated_files:
            parquet_file = pq.ParquetFile(file_path)
            num_rows = parquet_file.metadata.num_rows
            
            for i in range(num_rows):
                file_index[cumulative + i] = (file_path, i)
            
            cumulative += num_rows
        
        return file_index
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        """
        Get sequence by index.
        """
        file_path, local_idx = self.file_index[idx]
        
        # Read row from memory-mapped table
        table = self.tables[file_path]
        row_group = local_idx // table.metadata.row_group(0).num_rows
        local_row = local_idx % table.metadata.row_group(0).num_rows
        
        # Read specific row group
        batch = table.read_row_group(row_group)
        row = batch.slice(local_row, 1).to_pandas().iloc[0]
        
        # Extract data
        features = torch.from_numpy(np.array(row['features'], dtype=np.float32))
        targets = torch.from_numpy(np.array(row['targets'], dtype=np.float32))
        metadata = row['metadata'] if 'metadata' in row else {}
        
        return {
            'features': features,
            'targets': targets,
            'metadata': metadata
        }


def consolidate_training_data(config: Dict) -> tuple:
    """
    Consolidate fragmented training data into optimized files.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Paths to consolidated train/val/test files
    """
    
    import torch
    
    # Paths
    cache_dir = Path(config['data'].get('cache_dir', 'outputs/part2_geocryoai/cache'))
    consolidated_dir = cache_dir / 'consolidated'
    
    # Initialize consolidator
    consolidator = FilesystemOptimizedConsolidator(
        source_dir=cache_dir,
        output_dir=consolidated_dir,
        target_file_size_gb=15.0,
        lustre_stripe_size_mb=4.0,
        num_processes=8
    )
    
    # Consolidate each split
    train_files = consolidator.consolidate_sequences()
    
    return train_files, [], []  # Return consolidated files
