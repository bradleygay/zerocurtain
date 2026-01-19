#!/usr/bin/env python3
"""
INDEPENDENT Mann-Kendall Trend Analysis (Table S3.5) and Power Analysis (Table S3.6)
Optimized with Dask for parallel processing on large datasets.
NO DEPENDENCY on sensitivity thresholds - can run concurrently.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                    handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)

def compute_mann_kendall_optimized(years: np.ndarray, values: np.ndarray) -> dict:
    """
    Optimized Mann-Kendall using vectorized operations.
    Input: annual aggregates (small arrays ~10 elements)
    """
    from scipy.stats import kendalltau
    
    n = len(years)
    if n < 5:
        return {'tau': np.nan, 'p_value': np.nan, 'sen_slope': np.nan, 
                'trend': 'insufficient_data', 'n_years': n}
    
    # Kendall's tau
    tau, p = kendalltau(years, values)
    
    # Sen's slope (vectorized)
    i, j = np.triu_indices(n, k=1)
    slopes = (values[j] - values[i]) / (years[j] - years[i])
    sen_slope = np.median(slopes)
    
    # Trend interpretation
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


def compute_annual_medians_dask(parquet_path: str, variables: list) -> dict:
    """
    Compute annual medians using Dask for memory-efficient parallel processing.
    Returns dict of {variable: {year: median_value}}
    """
    import dask.dataframe as dd
    
    logger.info(f"  Loading with Dask: {parquet_path}")
    
    # Load with Dask - lazy evaluation
    ddf = dd.read_parquet(parquet_path, engine='pyarrow')
    
    # Get total count
    n_total = len(ddf)
    logger.info(f"  Total events: {n_total:,}")
    
    results = {}
    
    for var in variables:
        if var not in ddf.columns:
            logger.warning(f"  Variable '{var}' not in dataset, skipping")
            continue
        
        if 'year' not in ddf.columns:
            logger.warning(f"  'year' column not found, skipping")
            continue
        
        logger.info(f"  Computing annual medians for {var}...")
        
        # Dask groupby median - parallel computation
        annual = ddf.groupby('year')[var].median().compute()
        annual = annual.dropna().sort_index()
        
        results[var] = {
            'years': annual.index.values.astype(float),
            'medians': annual.values.astype(float),
            'n_years': len(annual)
        }
        logger.info(f"    {var}: {len(annual)} years of data")
    
    return results, n_total


def compute_power_analysis(sigma: float, variable: str) -> list:
    """
    Compute minimum detectable effect for various record lengths.
    Based on linear trend detection power (alpha=0.05, power=0.80).
    """
    results = []
    
    for n_years in [10, 20, 30, 50]:
        # MDE formula for linear trend (Weatherhead et al. 1998)
        # MDE = t_crit * sigma * sqrt(12 / (n^3 - n))
        # Using t_crit ≈ 2.8 for alpha=0.05, power=0.80
        mde_per_year = 2.8 * sigma * np.sqrt(12.0 / (n_years**3 - n_years))
        mde_per_decade = mde_per_year * 10
        
        if n_years <= 10:
            capability = 'Insufficient'
        elif n_years <= 20:
            capability = 'Marginal'
        elif n_years <= 30:
            capability = 'Recommended'
        else:
            capability = 'Robust'
        
        results.append({
            'record_length_years': n_years,
            'mde_per_decade': float(mde_per_decade),
            'detection_capability': capability
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Mann-Kendall and Power Analysis')
    parser.add_argument('--pinszc', required=True, help='Path to PINSZC parquet')
    parser.add_argument('--pirszc', required=True, help='Path to PIRSZC parquet')
    parser.add_argument('--output', required=True, help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("MANN-KENDALL TREND ANALYSIS & POWER ANALYSIS")
    logger.info("Independent computation - no threshold dependencies")
    logger.info("=" * 80)
    
    variables = ['spatial_extent_meters', 'duration_hours', 'intensity_percentile']
    
    results = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'script': 'compute_mann_kendall_power.py'
        },
        'table_s3_5_mann_kendall': {},
        'table_s3_6_power_analysis': {}
    }
    
    # =========================================================================
    # PINSZC Mann-Kendall
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PINSZC ANALYSIS")
    logger.info("=" * 60)
    
    pinszc_annuals, pinszc_n = compute_annual_medians_dask(args.pinszc, variables)
    results['metadata']['pinszc_events'] = pinszc_n
    
    for var, data in pinszc_annuals.items():
        logger.info(f"  Mann-Kendall: PINSZC {var}")
        mk_result = compute_mann_kendall_optimized(data['years'], data['medians'])
        mk_result['total_events'] = pinszc_n
        results['table_s3_5_mann_kendall'][f'pinszc_{var}'] = mk_result
        
        logger.info(f"    τ = {mk_result['tau']:.4f}, p = {mk_result['p_value']:.4f}, "
                   f"slope = {mk_result['sen_slope_per_decade']:.4f}/decade, "
                   f"trend = {mk_result['trend']}")
    
    # PINSZC Power Analysis
    logger.info("\n  Power Analysis: PINSZC")
    for var in ['spatial_extent_meters', 'duration_hours']:
        if var in pinszc_annuals:
            sigma = np.std(pinszc_annuals[var]['medians'])
            power_results = compute_power_analysis(sigma, var)
            results['table_s3_6_power_analysis'][f'pinszc_{var}'] = {
                'sigma_interannual': float(sigma),
                'unit': 'm' if 'extent' in var else 'hours',
                'analyses': power_results
            }
            logger.info(f"    {var}: σ = {sigma:.4f}")
    
    # Save checkpoint
    checkpoint_path = output_dir / 'mann_kendall_pinszc_complete.json'
    with open(checkpoint_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"CHECKPOINT: {checkpoint_path}")
    
    # =========================================================================
    # PIRSZC Mann-Kendall
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PIRSZC ANALYSIS")
    logger.info("=" * 60)
    
    pirszc_annuals, pirszc_n = compute_annual_medians_dask(args.pirszc, variables)
    results['metadata']['pirszc_events'] = pirszc_n
    
    for var, data in pirszc_annuals.items():
        logger.info(f"  Mann-Kendall: PIRSZC {var}")
        mk_result = compute_mann_kendall_optimized(data['years'], data['medians'])
        mk_result['total_events'] = pirszc_n
        results['table_s3_5_mann_kendall'][f'pirszc_{var}'] = mk_result
        
        logger.info(f"    τ = {mk_result['tau']:.4f}, p = {mk_result['p_value']:.4f}, "
                   f"slope = {mk_result['sen_slope_per_decade']:.4f}/decade, "
                   f"trend = {mk_result['trend']}")
    
    # PIRSZC Power Analysis
    logger.info("\n  Power Analysis: PIRSZC")
    for var in ['spatial_extent_meters', 'duration_hours']:
        if var in pirszc_annuals:
            sigma = np.std(pirszc_annuals[var]['medians'])
            power_results = compute_power_analysis(sigma, var)
            results['table_s3_6_power_analysis'][f'pirszc_{var}'] = {
                'sigma_interannual': float(sigma),
                'unit': 'm' if 'extent' in var else 'hours',
                'analyses': power_results
            }
            logger.info(f"    {var}: σ = {sigma:.4f}")
    
    # =========================================================================
    # FINAL SAVE
    # =========================================================================
    final_path = output_dir / 'mann_kendall_power_analysis_COMPLETE.json'
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"COMPLETE: {final_path}")
    logger.info("=" * 80)
    
    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("TABLE S3.5 SUMMARY - MANN-KENDALL RESULTS")
    logger.info("=" * 80)
    logger.info(f"{'Dataset':<20} {'Variable':<25} {'τ':>8} {'p-value':>10} {'Slope/decade':>15} {'Trend':<20}")
    logger.info("-" * 100)
    for key, val in results['table_s3_5_mann_kendall'].items():
        dataset, var = key.split('_', 1)
        logger.info(f"{dataset.upper():<20} {var:<25} {val['tau']:>8.4f} {val['p_value']:>10.4f} "
                   f"{val['sen_slope_per_decade']:>15.4f} {val['trend']:<20}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TABLE S3.6 SUMMARY - POWER ANALYSIS")
    logger.info("=" * 80)
    for key, val in results['table_s3_6_power_analysis'].items():
        logger.info(f"\n{key}: σ = {val['sigma_interannual']:.4f} {val['unit']}")
        for a in val['analyses']:
            logger.info(f"  {a['record_length_years']:>3} years: MDE = ±{a['mde_per_decade']:.4f}/decade ({a['detection_capability']})")


if __name__ == '__main__':
    main()
