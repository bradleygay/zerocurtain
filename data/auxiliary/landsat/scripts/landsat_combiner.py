#!/usr/bin/env python3
"""
Landsat Dataset Combiner
Merges original, gap-filled, and swath gap-filled datasets
"""

import sys
import logging
from pathlib import Path
import pandas as pd

def combine_datasets(original_file: Path, gap_file: Path, 
                    swath_gap_file: Path, output_file: Path):
    """
    Combine all Landsat datasets and remove duplicates
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Loading datasets...")
    original_df = pd.read_parquet(original_file)
    gap_df = pd.read_parquet(gap_file)
    swath_gap_df = pd.read_parquet(swath_gap_file)
    
    logger.info(f"Original: {len(original_df):,} rows")
    logger.info(f"Gap-filled: {len(gap_df):,} rows")
    logger.info(f"Swath gap-filled: {len(swath_gap_df):,} rows")
    
    # Combine
    combined_df = pd.concat([original_df, gap_df, swath_gap_df], ignore_index=True)
    logger.info(f"Combined (with duplicates): {len(combined_df):,} rows")
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['longitude', 'latitude', 'scene_id'])
    logger.info(f"Final dataset: {len(combined_df):,} rows")
    
    # Save
    combined_df.to_parquet(output_file, index=False)
    logger.info(f"Saved to {output_file}")
    
    return combined_df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine Landsat datasets')
    parser.add_argument('--original', type=str, required=True)
    parser.add_argument('--gaps', type=str, required=True)
    parser.add_argument('--swath-gaps', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s [%(levelname)s] %(message)s')
    
    combine_datasets(
        Path(args.original),
        Path(args.gaps),
        Path(args.swath_gaps),
        Path(args.output)
    )

if __name__ == "__main__":
    main()