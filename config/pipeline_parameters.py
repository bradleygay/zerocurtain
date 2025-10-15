"""
Pipeline parameters - EXACT METHODOLOGY

CRITICAL: Zero-curtain only occurs in COLD SEASON (Sept-May)
Summer months (June, July, August) are EXCLUDED - no freeze-thaw dynamics during growing season.
"""

# SEASONAL FILTERING - CRITICAL FOR ZERO-CURTAIN RESEARCH
SEASONAL_FILTER = {
    'exclude_months': [6, 7, 8],  # June, July, August (growing season)
    'include_months': [9, 10, 11, 12, 1, 2, 3, 4, 5],  # Sept-May (cold season)
    'reason': 'Zero-curtain dynamics only occur during freeze-thaw transitions in cold season'
}

# Data Splitting - STRATIFIED (on cold season data only)
SPLIT_RATIOS = {
    'train': 0.70,
    'validation': 0.20,
    'test': 0.10
}

# Stratification (after summer filtering)
STRATIFICATION = {
    'use_stratified': True,
    'stratify_columns': ['season', 'data_type'],  # Fall, Winter, Spring only
    'random_state': 42
}

# NO SAMPLING
SAMPLING = {
    'use_sampling': False,
    'note': 'Full cold-season dataset after summer exclusion'
}

# Duration Thresholds
SEQUENCE_PARAMS = {
    'min_duration_hours': 6,
    'max_duration_hours': 4500,
}

USE_FULL_DATASET = True  # All cold-season observations

if __name__ == "__main__":
    print("=" * 80)
    print("PIPELINE PARAMETERS - COLD SEASON ONLY")
    print("=" * 80)
    print(f"\nSeasonal Filter:")
    print(f"  EXCLUDE: {SEASONAL_FILTER['exclude_months']} (June, July, August)")
    print(f"  INCLUDE: {SEASONAL_FILTER['include_months']} (Sept-May)")
    print(f"  Reason: {SEASONAL_FILTER['reason']}")
    print(f"\nData Splits (Stratified on cold season):")
    print(f"  Training:   {SPLIT_RATIOS['train']*100:.0f}%")
    print(f"  Validation: {SPLIT_RATIOS['validation']*100:.0f}%")
    print(f"  Test:       {SPLIT_RATIOS['test']*100:.0f}%")
    print("=" * 80)
