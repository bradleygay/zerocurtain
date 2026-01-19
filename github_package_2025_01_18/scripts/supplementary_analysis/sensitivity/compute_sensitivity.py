#!/usr/bin/env python3
"""
RESUMABLE Supplementary Tables - Saves after EACH computation.
Skips already-completed work. Uses pre-computed values from failed run.
"""

import argparse
import json
import logging
import gc
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                    handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)

# PRE-COMPUTED VALUES FROM PREVIOUS 12-HOUR RUN
PRECOMPUTED = {
    'raw_morans_i': {'morans_i': 0.344417, 'z_score': 2540.72, 'p_value': 0.0, 'n_points': 54418117},
    'threshold_4.0': {
        'bootstrap_cv': 0.000064,
        'morans_i': {'morans_i': 0.279946, 'z_score': 2023.96, 'p_value': 0.0}
    }
}


def save_checkpoint(results: dict, output_dir: str, checkpoint_name: str = "checkpoint"):
    """Save results immediately after each computation."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f'supplementary_{checkpoint_name}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"CHECKPOINT SAVED: {path}")


def compute_morans_i_fast(df: pd.DataFrame, value_col: str, k: int = 8, chunk_size: int = 500000) -> dict:
    """Moran's I with analytical p-value."""
    from scipy.spatial import cKDTree
    from scipy import stats
    
    n_total = len(df)
    logger.info(f"  Moran's I on {n_total:,} points")
    
    coords = df[['latitude', 'longitude']].values.astype(np.float32)
    values = df[value_col].values.astype(np.float32)
    
    valid = ~np.isnan(values) & ~np.isnan(coords[:, 0]) & ~np.isnan(coords[:, 1])
    coords, values = coords[valid], values[valid]
    n = len(values)
    logger.info(f"    Valid: {n:,}")
    
    if n < 100:
        return {'morans_i': np.nan, 'p_value': np.nan, 'z_score': np.nan, 'n_points': n}
    
    mean_val, std_val = np.mean(values), np.std(values)
    z = (values - mean_val) / std_val
    
    logger.info(f"    Building KD-tree...")
    tree = cKDTree(coords)
    
    n_chunks = (n + chunk_size - 1) // chunk_size
    spatial_lag = np.zeros(n, dtype=np.float32)
    
    logger.info(f"    Spatial lag ({n_chunks} chunks)...")
    for i in range(n_chunks):
        start, end = i * chunk_size, min((i + 1) * chunk_size, n)
        if i % 10 == 0:
            logger.info(f"      Chunk {i+1}/{n_chunks}")
        _, idx = tree.query(coords[start:end], k=k+1, workers=-1)
        spatial_lag[start:end] = np.mean(z[idx[:, 1:]], axis=1)
        del idx
        gc.collect()
    
    I = np.dot(z, spatial_lag) / np.dot(z, z)
    E_I = -1.0 / (n - 1)
    Var_I = 1.0 / n
    z_score = (I - E_I) / np.sqrt(Var_I)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    logger.info(f"    Moran's I = {I:.6f}, z = {z_score:.2f}, p = {p_value:.2e}")
    
    del coords, values, z, spatial_lag, tree
    gc.collect()
    
    return {'morans_i': float(I), 'p_value': float(p_value), 'z_score': float(z_score), 'n_points': int(n)}


def compute_bootstrap_cv(df: pd.DataFrame, n_iter: int = 100) -> float:
    n = len(df)
    logger.info(f"  Bootstrap CV ({n_iter} iter, {n:,} events)")
    counts = [len(np.unique(np.random.randint(0, n, n))) for _ in range(n_iter)]
    cv = np.std(counts) / np.mean(counts)
    logger.info(f"    CV = {cv:.6f}")
    return float(cv)


def compute_cv_rmse_fast(df: pd.DataFrame, target_col: str, n_folds: int = 5) -> dict:
    """Faster CV with 5 folds instead of 10."""
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
    
    logger.info(f"  {n_folds}-fold CV RMSE on {len(df):,} events")
    
    feature_cols = ['latitude', 'longitude', 'intensity_percentile', 'year']
    available = [c for c in feature_cols if c in df.columns]
    
    X = df[available].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    
    valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X, y = X[valid], y[valid]
    n = len(y)
    logger.info(f"    Valid: {n:,}")
    
    perm = np.random.permutation(n)
    X, y = X[perm], y[perm]
    
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
        logger.info(f"    Fold {fold + 1}/{n_folds} done")
        gc.collect()
    
    rmse = np.sqrt(np.mean(errors))
    logger.info(f"    RMSE = {rmse:.2f}")
    return {'rmse': float(rmse), 'n_samples': int(n)}


def compute_mann_kendall(df: pd.DataFrame, var_col: str) -> dict:
    from scipy.stats import kendalltau
    annual = df.groupby('year')[var_col].median().dropna()
    if len(annual) < 5:
        return {'tau': np.nan, 'p_value': np.nan, 'trend': 'insufficient', 'n_years': len(annual)}
    
    years, values = annual.index.values.astype(float), annual.values
    tau, p = kendalltau(years, values)
    
    slopes = [(values[j] - values[i]) / (years[j] - years[i]) 
              for i in range(len(years)) for j in range(i+1, len(years)) if years[j] != years[i]]
    sen = np.median(slopes) if slopes else np.nan
    
    trend = 'increasing' if p < 0.05 and tau > 0 else ('decreasing' if p < 0.05 else 'not_significant')
    return {'tau': float(tau), 'p_value': float(p), 'slope_per_decade': float(sen * 10) if not np.isnan(sen) else None,
            'trend': trend, 'n_years': int(len(annual)), 'total_events': int(len(df))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pinszc', required=True)
    parser.add_argument('--pirszc', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("RESUMABLE SUPPLEMENTARY TABLES - INCREMENTAL SAVES")
    logger.info("=" * 80)
    
    # Load data
    logger.info(f"Loading PINSZC...")
    pinszc = pd.read_parquet(args.pinszc, engine='pyarrow')
    logger.info(f"PINSZC: {len(pinszc):,} events")
    
    logger.info(f"Loading PIRSZC...")
    pirszc = pd.read_parquet(args.pirszc, engine='pyarrow')
    logger.info(f"PIRSZC: {len(pirszc):,} events")
    
    results = {
        'metadata': {'generated': datetime.now().isoformat(), 'pinszc': len(pinszc), 'pirszc': len(pirszc)},
        'table_s2_4': []
    }
    
    # Use pre-computed RAW Moran's I
    logger.info("Using PRE-COMPUTED raw Moran's I from previous run...")
    raw_moran = PRECOMPUTED['raw_morans_i']
    logger.info(f"  Raw Moran's I = {raw_moran['morans_i']:.6f}")
    
    thresholds = [4.0, 5.0, 5.95, 7.0, 8.0]
    
    for thresh in thresholds:
        logger.info(f"\n{'='*60}")
        logger.info(f"THRESHOLD: {thresh}%")
        logger.info(f"{'='*60}")
        
        cutoff = np.percentile(pinszc['duration_hours'].dropna(), 100 - thresh)
        masked = pinszc[pinszc['duration_hours'] <= cutoff]
        
        retained, excluded = len(masked), len(pinszc) - len(masked)
        logger.info(f"  Retained: {retained:,} | Excluded: {excluded:,}")
        
        # Check for pre-computed values
        precomp_key = f'threshold_{thresh}'
        if precomp_key in PRECOMPUTED:
            logger.info(f"  Using PRE-COMPUTED values for threshold {thresh}%")
            precomp = PRECOMPUTED[precomp_key]
            cv = precomp['bootstrap_cv']
            moran = precomp['morans_i']
            logger.info(f"    Bootstrap CV = {cv:.6f}")
            logger.info(f"    Moran's I = {moran['morans_i']:.6f}")
        else:
            cv = compute_bootstrap_cv(masked)
            save_checkpoint(results, args.output, f"thresh_{thresh}_cv")
            
            moran = compute_morans_i_fast(masked, 'duration_hours')
            save_checkpoint(results, args.output, f"thresh_{thresh}_moran")
        
        # Always compute CV RMSE (was incomplete for 4.0%)
        rmse_result = compute_cv_rmse_fast(masked, 'duration_hours', n_folds=5)
        save_checkpoint(results, args.output, f"thresh_{thresh}_rmse")
        
        violations = (masked['duration_hours'] > 4380).sum()
        
        results['table_s2_4'].append({
            'threshold_pct': thresh,
            'events_retained': int(retained),
            'events_excluded': int(excluded),
            'detection_cv': cv if isinstance(cv, float) else cv,
            'morans_i_raw': raw_moran['morans_i'],
            'morans_i_masked': moran['morans_i'] if isinstance(moran, dict) else moran,
            'morans_i_z': moran.get('z_score', moran) if isinstance(moran, dict) else 0,
            'cv_rmse': rmse_result['rmse'],
            'physical_violations': int(violations)
        })
        
        save_checkpoint(results, args.output, f"thresh_{thresh}_complete")
        gc.collect()
    
    # Mann-Kendall
    logger.info("\n" + "=" * 60)
    logger.info("TABLE S3.5: MANN-KENDALL")
    logger.info("=" * 60)
    
    results['table_s3_5'] = {}
    for var in ['spatial_extent_meters', 'duration_hours', 'intensity_percentile']:
        if var in pinszc.columns:
            logger.info(f"  PINSZC {var}...")
            results['table_s3_5'][f'pinszc_{var}'] = compute_mann_kendall(pinszc, var)
        if var in pirszc.columns:
            logger.info(f"  PIRSZC {var}...")
            results['table_s3_5'][f'pirszc_{var}'] = compute_mann_kendall(pirszc, var)
    
    save_checkpoint(results, args.output, "mann_kendall")
    
    # Power Analysis
    logger.info("\n" + "=" * 60)
    logger.info("TABLE S3.6: POWER ANALYSIS")
    logger.info("=" * 60)
    
    results['table_s3_6'] = []
    for var in ['spatial_extent_meters', 'duration_hours']:
        if var not in pinszc.columns:
            continue
        sigma = pinszc.groupby('year')[var].median().std()
        logger.info(f"  {var}: σ = {sigma:.4f}")
        for n in [10, 20, 30, 50]:
            mde = 2.8 * sigma * np.sqrt(12 / (n**3 - n)) * 10
            cap = 'Insufficient' if n <= 10 else ('Marginal' if n <= 20 else ('Recommended' if n <= 30 else 'Robust'))
            results['table_s3_6'].append({'variable': var, 'record_length_years': n, 
                                         'mde_per_decade': float(mde), 'sigma_interannual': float(sigma), 
                                         'detection_capability': cap})
    
    # Final save
    save_checkpoint(results, args.output, "FINAL")
    
    # Also save as the expected filename
    final_path = Path(args.output) / 'supplementary_table_results_FULL.json'
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nFINAL SAVED: {final_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
