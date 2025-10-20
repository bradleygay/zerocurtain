"""
Teacher forcing with COLD SEASON filtering.
Excludes summer months (June, July, August) - zero-curtain only in Sept-May.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split


def prepare_teacher_forcing_dataset(
    input_path,
    output_path,
    train_ratio=0.70,
    val_ratio=0.20,
    test_ratio=0.10,
    stratify_columns=None,
    random_state=42,
    exclude_months=None,
    min_duration_hours=6,
    max_duration_hours=4500
):
    """
    Prepare stratified teacher forcing dataset with seasonal filtering.
    
    Args:
        exclude_months: Months to exclude (e.g., [6,7,8] for June-Aug)
    """
    print("=" * 80)
    print("COLD SEASON STRATIFIED DATASET PREPARATION")
    print("=" * 80)
    
    if exclude_months:
        print(f"\n  SEASONAL FILTER ACTIVE")
        print(f"  Excluding months: {exclude_months} (summer - no zero-curtain)")
        print(f"  Zero-curtain only occurs Sept-May (freeze-thaw transitions)")
    
    print(f"\nSplits: {train_ratio*100:.0f}% / {val_ratio*100:.0f}% / {test_ratio*100:.0f}%")
    print(f"Stratification: {stratify_columns}")
    
    print(f"\nLoading: {input_path}")
    df = pd.read_parquet(input_path)
    
    print(f"Total observations (before filtering): {len(df):,}")
    
    # CRITICAL: Filter out summer months
    if exclude_months:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['month'] = df['datetime'].dt.month
        
        initial_count = len(df)
        df = df[~df['month'].isin(exclude_months)].copy()
        filtered_count = initial_count - len(df)
        
        print(f"\n SUMMER FILTER APPLIED:")
        print(f"  Removed: {filtered_count:,} summer observations")
        print(f"  Remaining: {len(df):,} cold-season observations")
        print(f"  Reduction: {filtered_count/initial_count*100:.1f}%")
        
        df = df.drop('month', axis=1)
    
    print(f"\nDate range (cold season): {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Sort
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Stratification
    if stratify_columns:
        stratify_key = df[stratify_columns].astype(str).agg('_'.join, axis=1)
        print(f"\nStratification groups (cold season): {stratify_key.nunique()}")
    else:
        stratify_key = None
    
    # Split test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=stratify_key if stratify_columns else None
    )
    
    # Split validation
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    
    if stratify_columns:
        stratify_key_train_val = train_val_df[stratify_columns].astype(str).agg('_'.join, axis=1)
    else:
        stratify_key_train_val = None
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_key_train_val if stratify_columns else None
    )
    
    print(f"\nStratified Splits (COLD SEASON ONLY):")
    print(f"  Training:   {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:       {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify stratification
    if stratify_columns:
        print(f"\nStratification verification (cold season):")
        for col in stratify_columns:
            print(f"  {col}:")
            train_dist = train_df[col].value_counts(normalize=True).sort_index()
            val_dist = val_df[col].value_counts(normalize=True).sort_index()
            test_dist = test_df[col].value_counts(normalize=True).sort_index()
            for category in train_dist.index:
                print(f"    {category}: Train={train_dist.get(category, 0):.3f}, "
                      f"Val={val_dist.get(category, 0):.3f}, "
                      f"Test={test_dist.get(category, 0):.3f}")
    
    # Save
    output_base = Path(output_path).parent / Path(output_path).stem
    
    train_path = f"{output_base}_train.parquet"
    val_path = f"{output_base}_val.parquet"
    test_path = f"{output_base}_test.parquet"
    
    train_df.to_parquet(train_path, engine='pyarrow', compression='snappy')
    val_df.to_parquet(val_path, engine='pyarrow', compression='snappy')
    test_df.to_parquet(test_path, engine='pyarrow', compression='snappy')
    
    # Metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'methodology': 'Cold season only (Sept-May), stratified splits',
        'seasonal_filter': {
            'excluded_months': exclude_months,
            'reason': 'Zero-curtain only occurs during freeze-thaw (cold season)'
        },
        'total_observations_before_filter': initial_count if exclude_months else len(df),
        'total_observations_after_filter': len(df),
        'observations_removed': filtered_count if exclude_months else 0,
        'split_method': 'stratified',
        'stratify_columns': stratify_columns,
        'splits': {
            'train': {'count': len(train_df), 'fraction': len(train_df)/len(df), 'file': train_path},
            'validation': {'count': len(val_df), 'fraction': len(val_df)/len(df), 'file': val_path},
            'test': {'count': len(test_df), 'fraction': len(test_df)/len(df), 'file': test_path}
        }
    }
    
    metadata_path = f"{output_base}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")
    print(f"  Meta:  {metadata_path}")
    
    print("\n" + "=" * 80)
    print("COLD SEASON DATASET COMPLETE")
    print("=" * 80)
    
    return metadata


if __name__ == "__main__":
    print("Cold season teacher forcing module ready")
