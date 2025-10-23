#!/usr/bin/env python3

"""
PART IV: COMPREHENSIVE TEST SCRIPT
===================================
Validates all components without full training execution

Tests:
1. Environment and dependencies
2. File path verification
3. Model architecture loading
4. PIRSZC data loading and processing
5. Dataset creation with minimal samples
6. Forward pass validation
7. Training loop dry run (1 batch)
8. Output format verification

[RESEARCHER] Gay
Arctic Zero-Curtain Detection Research
[RESEARCHER] Sciences Laboratory
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import traceback

print("="*80)
print("PART IV TRANSFER LEARNING - COMPREHENSIVE TEST SCRIPT")
print("="*80)
print(f"Timestamp: {datetime.now()}")
print()

# ============================================================================
# TEST 1: ENVIRONMENT AND DEPENDENCIES
# ============================================================================

def test_environment():
    """Test Python environment and required packages"""
    print("="*80)
    print("TEST 1: ENVIRONMENT AND DEPENDENCIES")
    print("="*80)
    
    tests_passed = []
    
    # Python version
    print(f" Python version: {sys.version.split()[0]}")
    tests_passed.append(True)
    
    # Critical imports
    packages = {
        'numpy': 'np',
        'pandas': 'pd',
        'torch': 'torch',
        'pyarrow': 'pyarrow',
        'numba': 'numba',
        'tqdm': 'tqdm'
    }
    
    for package, alias in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f" {package}: {version}")
            tests_passed.append(True)
        except ImportError as e:
            print(f" {package}: NOT FOUND - {e}")
            tests_passed.append(False)
    
    # PyTorch device check
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" PyTorch device: {device}")
        if device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        tests_passed.append(True)
    except Exception as e:
        print(f" PyTorch device check failed: {e}")
        tests_passed.append(False)
    
    return all(tests_passed)

# ============================================================================
# TEST 2: FILE PATH VERIFICATION
# ============================================================================

def test_file_paths(base_dir=None):
    """Verify all required files exist"""
    print("\n" + "="*80)
    print("TEST 2: FILE PATH VERIFICATION")
    print("="*80)
    
    if base_dir is None:
        base_dir = Path.home() / "arctic_zero_curtain_pipeline"
    else:
        base_dir = Path(base_dir)
    
    print(f"Base directory: {base_dir}")
    
    files_to_check = {
        'Part II Model': base_dir / 'outputs' / 'part2_geocryoai' / 'models' / 'best_model_tf.pth',
        'Part III PIRSZC Data': base_dir / 'outputs' / 'part3_pirszc' / 'remote_sensing_physics_zero_curtain_comprehensive.parquet'
    }
    
    all_exist = True
    
    for name, path in files_to_check.items():
        if path.exists():
            size_gb = path.stat().st_size / 1e9
            print(f" {name}:")
            print(f"  Path: {path}")
            print(f"  Size: {size_gb:.2f} GB")
        else:
            print(f" {name}: NOT FOUND")
            print(f"  Expected: {path}")
            all_exist = False
    
    # Check/create output directories
    output_dirs = {
        'Part IV Base': base_dir / 'outputs' / 'part4_transfer_learning',
        'Checkpoints': base_dir / 'outputs' / 'part4_transfer_learning' / 'checkpoints',
        'Predictions': base_dir / 'outputs' / 'part4_transfer_learning' / 'predictions',
        'Metrics': base_dir / 'outputs' / 'part4_transfer_learning' / 'metrics',
        'Logs': base_dir / 'outputs' / 'part4_transfer_learning' / 'logs'
    }
    
    print("\nOutput directories:")
    for name, path in output_dirs.items():
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f" {name}: Created - {path}")
        else:
            print(f" {name}: Exists - {path}")
    
    return all_exist

# ============================================================================
# TEST 3: MODEL ARCHITECTURE LOADING
# ============================================================================

def test_model_loading(model_path):
    """Test loading GeoCryoAI model architecture"""
    print("\n" + "="*80)
    print("TEST 3: MODEL ARCHITECTURE LOADING")
    print("="*80)
    
    try:
        import torch
        import torch.nn as nn
        
        print("Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print(f" Checkpoint loaded successfully")
        
        # Analyze checkpoint structure
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  Using 'model_state_dict' key")
        else:
            state_dict = checkpoint
            print(f"  Using direct state dict")
        
        print(f"  Total parameters: {len(state_dict)}")
        
        # Check for critical components
        critical_components = [
            'feature_embedding.0.weight',
            'positional_encoding',
            'attention_layers.0.spatial_attention',
            'liquid_layers.0.input_layer.weight',
            'intensity_head.0.weight',
            'duration_head.0.weight',
            'extent_head.0.weight'
        ]
        
        print("\nCritical component check:")
        all_present = True
        for component in critical_components:
            # Check if component or partial match exists
            matches = [k for k in state_dict.keys() if component in k]
            if matches:
                print(f"   {component}: Found ({len(matches)} matches)")
            else:
                print(f"   {component}: Missing")
                all_present = False
        
        # Display first few keys for verification
        print("\nFirst 10 state dict keys:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {i+1}. {key}")
        
        # Check metadata
        if 'epoch' in checkpoint:
            print(f"\nCheckpoint metadata:")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
            print(f"  Phase: {checkpoint.get('phase', 'N/A')}")
        
        return all_present
        
    except Exception as e:
        print(f" Model loading failed: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# TEST 4: PIRSZC DATA LOADING
# ============================================================================

def test_pirszc_data_loading(data_path):
    """Test PIRSZC data loading and schema validation"""
    print("\n" + "="*80)
    print("TEST 4: PIRSZC DATA LOADING")
    print("="*80)
    
    try:
        import pandas as pd
        import pyarrow.parquet as pq
        
        print(f"Loading PIRSZC data: {data_path}")
        
        # Read metadata without loading full data
        pf = pq.ParquetFile(str(data_path))
        
        print(f" Parquet file accessible")
        print(f"  Total rows: {pf.metadata.num_rows:,}")
        print(f"  Total columns: {pf.metadata.num_columns}")
        print(f"  Row groups: {pf.metadata.num_row_groups}")
        
        # Check schema
        print("\nSchema validation:")
        required_columns = [
            'cluster_id', 'start_time', 'duration_hours', 'intensity_percentile',
            'spatial_extent_meters', 'latitude', 'longitude', 'mean_temperature',
            'temperature_variance', 'mean_moisture', 'moisture_variance',
            'mean_displacement', 'consensus_confidence', 'permafrost_probability',
            'phase_change_energy', 'freeze_penetration_depth', 'thermal_diffusivity',
            'snow_insulation_factor', 'cryogrid_thermal_conductivity',
            'cryogrid_heat_capacity', 'surface_energy_balance',
            'van_genuchten_alpha', 'van_genuchten_n'
        ]
        
        schema = pf.schema_arrow
        available_columns = [field.name for field in schema]
        
        all_present = True
        for col in required_columns:
            if col in available_columns:
                print(f"   {col}")
            else:
                print(f"   {col}: MISSING")
                all_present = False
        
        if not all_present:
            print("\n Warning: Some required columns missing")
            print("Available columns:")
            for col in available_columns[:10]:
                print(f"  - {col}")
        
        # Load small sample
        print("\nLoading sample data (1000 rows)...")
        df_sample = pd.read_parquet(data_path, engine='pyarrow').head(1000)
        
        print(f" Sample loaded: {len(df_sample)} rows")
        print(f"  Clusters in sample: {df_sample['cluster_id'].nunique()}")
        print(f"  Date range: {df_sample['start_time'].min()} to {df_sample['start_time'].max()}")
        print(f"  Lat range: {df_sample['latitude'].min():.2f} to {df_sample['latitude'].max():.2f}")
        print(f"  Lon range: {df_sample['longitude'].min():.2f} to {df_sample['longitude'].max():.2f}")
        
        return all_present
        
    except Exception as e:
        print(f" PIRSZC data loading failed: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# TEST 5: DATASET CREATION WITH MINIMAL SAMPLE
# ============================================================================

def test_dataset_creation(data_path):
    """Test dataset creation with minimal sample"""
    print("\n" + "="*80)
    print("TEST 5: DATASET CREATION (MINIMAL SAMPLE)")
    print("="*80)
    
    try:
        import pandas as pd
        import numpy as np
        from numba import jit
        
        print("Loading minimal sample for dataset test...")
        
        # Load only 2 clusters for testing
        df_full = pd.read_parquet(data_path, columns=['cluster_id'])
        cluster_counts = df_full['cluster_id'].value_counts()
        test_clusters = cluster_counts.index.tolist()[:2]
        
        print(f"Test clusters: {test_clusters}")
        
        # Load data for test clusters
        df = pd.read_parquet(data_path)
        df = df[df['cluster_id'].isin(test_clusters)]
        df = df.sort_values(['cluster_id', 'start_time'])
        
        print(f" Loaded {len(df)} records for test clusters")
        
        # Test NUMBA functions
        @jit(nopython=True)
        def test_numba_normalize(features):
            n_samples, n_features = features.shape
            normalized = np.empty_like(features)
            for j in range(n_features):
                col = features[:, j]
                normalized[:, j] = (col - np.mean(col)) / (np.std(col) + 1e-8)
            return normalized
        
        test_features = np.random.randn(100, 21).astype(np.float32)
        normalized = test_numba_normalize(test_features)
        
        print(f" NUMBA normalization working")
        print(f"  Input shape: {test_features.shape}")
        print(f"  Output shape: {normalized.shape}")
        
        # Test sequence creation
        @jit(nopython=True)
        def test_create_sequences(features, sequence_length=24):
            n_samples, n_features = features.shape
            n_sequences = max(0, n_samples - sequence_length + 1)
            if n_sequences == 0:
                return np.empty((0, sequence_length, n_features), dtype=np.float32)
            sequences = np.empty((n_sequences, sequence_length, n_features), dtype=np.float32)
            for i in range(n_sequences):
                for j in range(sequence_length):
                    for k in range(n_features):
                        sequences[i, j, k] = features[i + j, k]
            return sequences
        
        sequences = test_create_sequences(test_features, sequence_length=24)
        
        print(f" Sequence creation working")
        print(f"  Input: {test_features.shape} → Output: {sequences.shape}")
        print(f"  Sequences created: {len(sequences)}")
        
        return True
        
    except Exception as e:
        print(f" Dataset creation test failed: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# TEST 6: MODEL FORWARD PASS
# ============================================================================

def test_model_forward_pass(model_path):
    """Test model forward pass with dummy data"""
    print("\n" + "="*80)
    print("TEST 6: MODEL FORWARD PASS")
    print("="*80)
    
    try:
        import torch
        import torch.nn as nn
        
        # Import model architecture from part4_transfer_learning.py
        print("Importing model architecture...")
        sys.path.insert(0, str(Path(__file__).parent))
        from part4_transfer_learning import GeoCryoAIModel
        
        print(" Model architecture imported")
        
        # Initialize model
        device = torch.device('cpu')  # Use CPU for testing
        model = GeoCryoAIModel().to(device)
        
        print(f" Model initialized on {device}")
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f" Model weights loaded")
        
        # Test forward pass with dummy data
        batch_size = 4
        sequence_length = 24
        n_features = 21
        
        dummy_input = torch.randn(batch_size, sequence_length, n_features).to(device)
        
        print(f"\nTesting forward pass...")
        print(f"  Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f" Forward pass successful")
        print(f"\nOutput shapes:")
        for key, value in outputs.items():
            print(f"  {key}: {value.shape}")
        
        # Validate output ranges
        print(f"\nOutput value ranges:")
        print(f"  Intensity: [{outputs['intensity'].min():.4f}, {outputs['intensity'].max():.4f}]")
        print(f"  Duration: [{outputs['duration'].min():.4f}, {outputs['duration'].max():.4f}]")
        print(f"  Extent: [{outputs['extent'].min():.4f}, {outputs['extent'].max():.4f}]")
        
        # Check if outputs are in expected ranges
        intensity_valid = (outputs['intensity'] >= 0).all() and (outputs['intensity'] <= 1).all()
        duration_valid = (outputs['duration'] >= 0).all() and (outputs['duration'] <= 10).all()
        extent_valid = (outputs['extent'] >= 0).all()
        
        print(f"\nOutput validation:")
        print(f"  Intensity [0,1]: {'' if intensity_valid else ''}")
        print(f"  Duration [0,~9]: {'' if duration_valid else ''}")
        print(f"  Extent ≥0: {'' if extent_valid else ''}")
        
        return intensity_valid and duration_valid and extent_valid
        
    except Exception as e:
        print(f" Forward pass test failed: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# TEST 7: TRAINING LOOP DRY RUN
# ============================================================================

def test_training_dry_run(model_path, data_path):
    """Test training loop with single batch"""
    print("\n" + "="*80)
    print("TEST 7: TRAINING LOOP DRY RUN (1 BATCH)")
    print("="*80)
    
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        import pandas as pd
        
        # Import components
        sys.path.insert(0, str(Path(__file__).parent))
        from part4_transfer_learning import GeoCryoAIModel, PIRSZCDataset
        
        print("Creating minimal dataset (1 cluster)...")
        
        # Get one cluster
        df_clusters = pd.read_parquet(data_path, columns=['cluster_id'])
        test_cluster = [df_clusters['cluster_id'].value_counts().index[0]]
        
        # Create minimal dataset
        dataset = PIRSZCDataset(data_path, test_cluster, sequence_length=24)
        
        if len(dataset) == 0:
            print(" Warning: Dataset empty, trying second cluster...")
            test_cluster = [df_clusters['cluster_id'].value_counts().index[1]]
            dataset = PIRSZCDataset(data_path, test_cluster, sequence_length=24)
        
        print(f" Dataset created: {len(dataset)} sequences")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        
        print(f" DataLoader created")
        
        # Initialize model
        device = torch.device('cpu')
        model = GeoCryoAIModel().to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.train()
        
        print(f" Model initialized in training mode")
        
        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        mse_loss = nn.MSELoss()
        
        print(f" Optimizer and loss function ready")
        
        # Test single batch
        print(f"\nRunning training on single batch...")
        
        features, targets = next(iter(dataloader))
        features = features.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        print(f"  Batch features shape: {features.shape}")
        print(f"  Batch targets:")
        for k, v in targets.items():
            print(f"    {k}: {v.shape}")
        
        # Forward pass
        predictions = model(features)
        
        print(f"   Forward pass completed")
        
        # Compute loss
        intensity_loss = mse_loss(predictions['intensity'], targets['intensity'])
        duration_loss = mse_loss(predictions['duration'], targets['duration'])
        extent_loss = mse_loss(predictions['extent'], targets['extent'])
        total_loss = (intensity_loss + duration_loss + extent_loss) / 3.0
        
        print(f"   Loss computed:")
        print(f"    Total: {total_loss.item():.6f}")
        print(f"    Intensity: {intensity_loss.item():.6f}")
        print(f"    Duration: {duration_loss.item():.6f}")
        print(f"    Extent: {extent_loss.item():.6f}")
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"   Backward pass and optimization step completed")
        
        return True
        
    except Exception as e:
        print(f" Training dry run failed: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# TEST 8: OUTPUT FORMAT VERIFICATION
# ============================================================================

def test_output_format():
    """Test output format for predictions"""
    print("\n" + "="*80)
    print("TEST 8: OUTPUT FORMAT VERIFICATION")
    print("="*80)
    
    try:
        import pandas as pd
        import numpy as np
        
        print("Creating sample prediction dataframe...")
        
        # Create sample predictions
        n_samples = 100
        
        sample_df = pd.DataFrame({
            'cluster_id': np.random.randint(0, 10, n_samples),
            'start_time': pd.date_range('2015-01-01', periods=n_samples, freq='D'),
            'latitude': np.random.uniform(50, 75, n_samples),
            'longitude': np.random.uniform(-180, 180, n_samples),
            'predicted_intensity_percentile': np.random.uniform(0, 1, n_samples),
            'predicted_duration_hours': np.random.uniform(6, 6570, n_samples),
            'predicted_spatial_extent_meters': np.random.uniform(0.1, 10, n_samples)
        })
        
        # Add temporal features
        sample_df['year'] = sample_df['start_time'].dt.year
        sample_df['month'] = sample_df['start_time'].dt.month
        sample_df['season'] = sample_df['month'].apply(
            lambda m: 'Winter' if m in [12,1,2] else 'Spring' if m in [3,4,5] 
                     else 'Summer' if m in [6,7,8] else 'Fall'
        )
        
        # Add validation flags
        sample_df['duration_valid'] = (
            (sample_df['predicted_duration_hours'] >= 6) & 
            (sample_df['predicted_duration_hours'] <= 6570)
        )
        sample_df['intensity_valid'] = (
            (sample_df['predicted_intensity_percentile'] >= 0) & 
            (sample_df['predicted_intensity_percentile'] <= 1)
        )
        sample_df['extent_valid'] = sample_df['predicted_spatial_extent_meters'] >= 0.001
        sample_df['prediction_valid'] = (
            sample_df['duration_valid'] & 
            sample_df['intensity_valid'] & 
            sample_df['extent_valid']
        )
        
        print(f" Sample dataframe created: {len(sample_df)} rows")
        print(f"\nColumns ({len(sample_df.columns)}):")
        for col in sample_df.columns:
            print(f"  - {col}")
        
        print(f"\nValidation summary:")
        print(f"  Valid predictions: {sample_df['prediction_valid'].sum()} ({sample_df['prediction_valid'].mean()*100:.1f}%)")
        print(f"  Years: {sorted(sample_df['year'].unique())}")
        print(f"  Seasons: {sample_df['season'].unique()}")
        
        # Test saving
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            test_parquet = Path(tmpdir) / 'test_predictions.parquet'
            test_csv = Path(tmpdir) / 'test_predictions.csv'
            
            sample_df.to_parquet(test_parquet, index=False)
            sample_df.to_csv(test_csv, index=False)
            
            print(f"\n Test file saving:")
            print(f"  Parquet: {test_parquet.stat().st_size / 1024:.1f} KB")
            print(f"  CSV: {test_csv.stat().st_size / 1024:.1f} KB")
            
            # Test loading back
            df_loaded = pd.read_parquet(test_parquet)
            print(f"   Parquet reload: {len(df_loaded)} rows")
        
        return True
        
    except Exception as e:
        print(f" Output format test failed: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

def main():
    """Execute all tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Part IV Pipeline Test Script')
    parser.add_argument('--base-dir', type=str, default=None,
                       help='Base pipeline directory')
    
    args = parser.parse_args()
    
    # Initialize paths
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path.home() / "arctic_zero_curtain_pipeline"
    
    model_path = base_dir / 'outputs' / 'part2_geocryoai' / 'models' / 'best_model_tf.pth'
    data_path = base_dir / 'outputs' / 'part3_pirszc' / 'remote_sensing_physics_zero_curtain_comprehensive.parquet'
    
    # Run tests
    test_results = {}
    
    try:
        test_results['environment'] = test_environment()
        test_results['file_paths'] = test_file_paths(base_dir)
        
        if test_results['file_paths']:
            test_results['model_loading'] = test_model_loading(model_path)
            test_results['pirszc_loading'] = test_pirszc_data_loading(data_path)
            test_results['dataset_creation'] = test_dataset_creation(data_path)
            test_results['forward_pass'] = test_model_forward_pass(model_path)
            test_results['training_dry_run'] = test_training_dry_run(model_path, data_path)
            test_results['output_format'] = test_output_format()
        else:
            print("\n Skipping remaining tests due to missing files")
            test_results['model_loading'] = False
            test_results['pirszc_loading'] = False
            test_results['dataset_creation'] = False
            test_results['forward_pass'] = False
            test_results['training_dry_run'] = False
            test_results['output_format'] = False
    
    except Exception as e:
        print(f"\n CRITICAL ERROR: {e}")
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in test_results.items():
        status = " PASSED" if passed else " FAILED"
        print(f"  {status}: {test_name}")
    
    print()
    
    if all(test_results.values()):
        print("="*80)
        print(" ALL TESTS PASSED")
        print("="*80)
        print()
        print("System is ready for Part IV transfer learning.")
        print()
        print("To run full training:")
        print(f"  cd {base_dir / 'scripts'}")
        print("  python part4_transfer_learning.py --epochs 10 --batch-size 64")
        print()
        return 0
    else:
        print("="*80)
        print(" SOME TESTS FAILED")
        print("="*80)
        print()
        print("Please resolve errors above before running full training.")
        print()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)