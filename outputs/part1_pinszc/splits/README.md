# Train/Validation/Test Splits

This directory contains stratified data splits for GeoCryoAI machine learning.

## Files

- `physics_informed_events_train.parquet`: Training set (70%)
- `physics_informed_events_val.parquet`: Validation set (15%)
- `physics_informed_events_test.parquet`: Testing set (15%)

## Stratification

Splits are stratified by `intensity_category` to ensure:
- Balanced representation of weak/moderate/strong/extreme events
- Consistent distribution across training, validation, and test sets
- Prevention of data leakage between sets

## Usage

**CRITICAL**: 
- Use ONLY training set for model training
- Use validation set for hyperparameter tuning
- Use test set ONLY ONCE for final model evaluation
- Never mix or combine these datasets during training

## Statistics

Training set: ~38 million events
Validation set: ~8 million events
Testing set: ~8 million events

All splits contain identical feature sets and distributions.
