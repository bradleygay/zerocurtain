#!/usr/bin/env python3
"""Complete the 8.0% threshold CV RMSE computation."""
import json
import logging
import sys
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                    handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)

def main():
    output_dir = Path('/discover/nobackup/bagay/supplementary_tables_results')
    
    logger.info("=" * 60)
    logger.info("COMPLETING 8.0% THRESHOLD CV RMSE")
    logger.info("=" * 60)
    
    # Load PINSZC
    logger.info("Loading PINSZC...")
    pinszc = pd.read_parquet(
        '/discover/nobackup/bagay/gdrive_sync/arctic_zero_curtain_pipeline/outputs/part1_pinszc/consolidated_datasets/physics_informed_zero_curtain_events_COMPLETE.parquet',
        engine='pyarrow'
    )
    logger.info(f"Loaded {len(pinszc):,} events")
    
    # Apply 8.0% threshold mask
    cutoff = np.percentile(pinszc['duration_hours'].dropna(), 100 - 8.0)
    masked = pinszc[pinszc['duration_hours'] <= cutoff]
    logger.info(f"8.0% threshold: {len(masked):,} events retained")
    
    # CV RMSE
    logger.info("Computing 5-fold CV RMSE...")
    feature_cols = ['latitude', 'longitude', 'intensity_percentile', 'year']
    available = [c for c in feature_cols if c in masked.columns]
    
    X = masked[available].values.astype(np.float32)
    y = masked['duration_hours'].values.astype(np.float32)
    
    valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X, y = X[valid], y[valid]
    n = len(y)
    logger.info(f"Valid: {n:,}")
    
    perm = np.random.RandomState(42).permutation(n)
    X, y = X[perm], y[perm]
    
    n_folds = 5
    fold_size = n // n_folds
    errors = []
    
    for fold in range(n_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < n_folds - 1 else n
        
        test_idx = slice(start, end)
        train_idx = np.r_[:start, end:n]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        
        model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
        model.fit(X_tr, y[train_idx])
        
        errors.extend((y[test_idx] - model.predict(X_te)) ** 2)
        logger.info(f"  Fold {fold + 1}/{n_folds} done")
        gc.collect()
    
    rmse = np.sqrt(np.mean(errors))
    logger.info(f"CV RMSE = {rmse:.2f} hrs")
    
    # Save result
    result = {
        'threshold_pct': 8.0,
        'events_retained': int(len(masked)),
        'cv_rmse': float(rmse)
    }
    
    with open(output_dir / 'supplementary_thresh_8.0_rmse.json', 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {output_dir / 'supplementary_thresh_8.0_rmse.json'}")
    
    # Also update the complete checkpoint
    complete = {
        'threshold_pct': 8.0,
        'events_retained': 50076951,
        'events_excluded': 4341166,
        'detection_cv': 0.000067,
        'morans_i_masked': 0.180435,
        'morans_i_z': 1276.85,
        'cv_rmse': float(rmse)
    }
    with open(output_dir / 'supplementary_thresh_8.0_complete.json', 'w') as f:
        json.dump(complete, f, indent=2)
    logger.info(f"Saved: {output_dir / 'supplementary_thresh_8.0_complete.json'}")
    
    logger.info("COMPLETE")

if __name__ == '__main__':
    main()
