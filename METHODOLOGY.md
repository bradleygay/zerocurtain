# Arctic Zero-Curtain Pipeline - Methodology

## CRITICAL: Cold Season Only

Zero-curtain dynamics **only occur during freeze-thaw transitions** in the cold season.

**Seasonal Filter Applied:**
- **EXCLUDED**: June, July, August (summer/growing season)
- **INCLUDED**: September, October, November, December, January, February, March, April, May
- **Rationale**: No freeze-thaw dynamics occur during summer growing season

**Impact:**
- Total dataset: 62,708,668 observations (1891-2024)
- Summer observations removed: 18,414,166 (29.4%)
- Cold season observations: 44,294,502 (70.6%)

## Data Splits

**Method:** Stratified sampling on cold season data

**Split Ratios:**
- Training: 70% (31,006,150 observations)
- Validation: 20% (8,858,901 observations)
- Test: 10% (4,429,451 observations)

## Stratification

Data is stratified by:
1. **Season** (Fall, Spring, Winter) - Summer excluded
2. **Data Type** (soil_temperature, soil_moisture, active_layer)

### Stratification Verification (Cold Season)

Identical distributions maintained across all splits:

**By Season:**
- Fall: 36.7%
- Spring: 32.8%
- Winter: 30.5%

**By Data Type:**
- Soil Temperature: 56.9%
- Soil Moisture: 42.8%
- Active Layer: 0.3%

## Duration Thresholds

- **Minimum:** 6 hours (zero-curtain event detection threshold)
- **Maximum:** 4500 hours

## Sampling

**No random sampling.** Full cold-season dataset (44.3M observations) used.

## Reproducibility

- Random state: 42
- Stratified splits ensure consistent distributions
- Seasonal filter documented in `config/pipeline_parameters.py`

## Usage
```bash
# Run complete pipeline (with cold season filter)
python scripts/run_full_pipeline.py
```

**Outputs:**
- `outputs/teacher_forcing_in_situ_database_train.parquet` (31.0M obs)
- `outputs/teacher_forcing_in_situ_database_val.parquet` (8.9M obs)
- `outputs/teacher_forcing_in_situ_database_test.parquet` (4.4M obs)
- `outputs/teacher_forcing_in_situ_database_metadata.json`

## Scientific Rationale

Zero-curtain is a period during freeze-thaw transitions when soil temperature remains near 0°C due to latent heat exchange. This phenomenon:

1. **Occurs only during cold season** (Sept-May) when freeze-thaw transitions happen
2. **Does not occur during summer** (June-Aug) when soil remains thawed
3. **Is critical for permafrost dynamics** and carbon cycling in Arctic regions

Therefore, summer months are excluded from the training dataset to focus the model on relevant freeze-thaw dynamics.

## Reference

This methodology implements the exact approach from Dr. Gay's original research on zero-curtain detection and prediction in Arctic permafrost regions.
