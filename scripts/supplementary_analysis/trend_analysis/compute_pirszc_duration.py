#!/usr/bin/env python3
"""
Fix PIRSZC Mann-Kendall - extract year from start_time column.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import dask.dataframe as dd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                    handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)

def compute_mann_kendall_optimized(years: np.ndarray, values: np.ndarray) -> dict:
    from scipy.stats import kendalltau
    
    n = len(years)
    if n < 5:
        return {'tau': np.nan, 'p_value': np.nan, 'sen_slope': np.nan, 
                'trend': 'insufficient_data', 'n_years': n}
    
    tau, p = kendalltau(years, values)
    
    i, j = np.triu_indices(n, k=1)
    slopes = (values[j] - values[i]) / (years[j] - years[i])
    sen_slope = np.median(slopes)
    
    if p < 0.05:
        trend = 'significant_increasing' if tau > 0 else 'significant_decreasing'
    else:
        trend = 'not_significant'
    
    return {
        'tau': float(tau),
        'p_value': float(p),
        'sen_slope_per_year': float(sen_slope),
        'sen_slope_per_decade': float(sen_slope * 10),
        'trend': trend,
        'n_years': int(n)
    }


def main():
    output_dir = Path('/discover/nobackup/bagay/supplementary_tables_results')
    pirszc_path = '/discover/nobackup/bagay/gdrive_sync/arctic_zero_curtain_pipeline/outputs/part3_pirszc/remote_sensing_physics_informed_comprehensive.parquet'
    
    logger.info("=" * 80)
    logger.info("PIRSZC MANN-KENDALL FIX - Extracting year from start_time")
    logger.info("=" * 80)
    
    # Load with Dask
    logger.info("Loading PIRSZC with Dask...")
    ddf = dd.read_parquet(pirszc_path, engine='pyarrow')
    
    # Extract year from start_time
    logger.info("Extracting year from start_time...")
    ddf['year'] = ddf['start_time'].dt.year
    
    n_total = len(ddf)
    logger.info(f"Total events: {n_total:,}")
    
    variables = ['spatial_extent_meters', 'duration_hours', 'intensity_percentile']
    
    results = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'pirszc_events': n_total,
            'year_source': 'start_time'
        },
        'table_s3_5_mann_kendall': {},
        'table_s3_6_power_analysis': {}
    }
    
    # Compute annual medians
    for var in variables:
        if var not in ddf.columns:
            logger.warning(f"Variable '{var}' not found, skipping")
            continue
        
        logger.info(f"Computing annual medians for {var}...")
        annual = ddf.groupby('year')[var].median().compute()
        annual = annual.dropna().sort_index()
        
        years = annual.index.values.astype(float)
        values = annual.values.astype(float)
        
        logger.info(f"  {var}: {len(annual)} years ({int(years.min())}-{int(years.max())})")
        
        # Mann-Kendall
        mk_result = compute_mann_kendall_optimized(years, values)
        mk_result['total_events'] = n_total
        mk_result['year_range'] = f"{int(years.min())}-{int(years.max())}"
        results['table_s3_5_mann_kendall'][f'pirszc_{var}'] = mk_result
        
        logger.info(f"    τ = {mk_result['tau']:.4f}, p = {mk_result['p_value']:.4f}, "
                   f"slope = {mk_result['sen_slope_per_decade']:.4f}/decade, "
                   f"trend = {mk_result['trend']}")
        
        # Power analysis
        if var in ['spatial_extent_meters', 'duration_hours']:
            sigma = np.std(values)
            power_results = []
            for n_years in [10, 20, 30, 50]:
                mde = 2.8 * sigma * np.sqrt(12.0 / (n_years**3 - n_years)) * 10
                cap = 'Insufficient' if n_years <= 10 else ('Marginal' if n_years <= 20 else ('Recommended' if n_years <= 30 else 'Robust'))
                power_results.append({
                    'record_length_years': n_years,
                    'mde_per_decade': float(mde),
                    'detection_capability': cap
                })
            results['table_s3_6_power_analysis'][f'pirszc_{var}'] = {
                'sigma_interannual': float(sigma),
                'unit': 'm' if 'extent' in var else 'hours',
                'analyses': power_results
            }
            logger.info(f"    Power analysis: σ = {sigma:.4f}")
    
    # Save PIRSZC-specific results
    pirszc_path = output_dir / 'mann_kendall_pirszc_FIXED.json'
    with open(pirszc_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved: {pirszc_path}")
    
    # Load and merge with existing results
    existing_path = output_dir / 'mann_kendall_power_analysis_COMPLETE.json'
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        
        # Merge PIRSZC results into existing
        existing['table_s3_5_mann_kendall'].update(results['table_s3_5_mann_kendall'])
        existing['table_s3_6_power_analysis'].update(results['table_s3_6_power_analysis'])
        existing['metadata']['pirszc_events'] = n_total
        existing['metadata']['pirszc_year_source'] = 'start_time'
        
        merged_path = output_dir / 'mann_kendall_power_analysis_MERGED.json'
        with open(merged_path, 'w') as f:
            json.dump(existing, f, indent=2, default=str)
        logger.info(f"Merged: {merged_path}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PIRSZC MANN-KENDALL RESULTS")
    logger.info("=" * 80)
    for key, val in results['table_s3_5_mann_kendall'].items():
        logger.info(f"{key}: τ={val['tau']:.4f}, p={val['p_value']:.4f}, "
                   f"slope={val['sen_slope_per_decade']:.4f}/decade, {val['trend']}")


if __name__ == '__main__':
    main()
