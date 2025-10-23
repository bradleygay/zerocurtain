"""
Modeling module for Arctic zero-curtain pipeline.
Auto-generated from Jupyter notebook.
"""

from src.utils.imports import *
from src.utils.utilities import *

from tensorflow.keras.utils import plot_model

tf.keras.utils.plot_model(model, to_file='model_architecture.png',
           show_shapes=True, show_layer_names=True,
           expand_nested=True, dpi=300)

#MODEL DEVELOPMENT PART I

import os
import tensorflow as tf

# Configure memory growth to avoid pre-allocating all GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Optional: Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0=debug, 1=info, 2=warning, 3=error

# TensorFlow Performance Troubleshooting Guide for Apple Silicon (M1/M2)

"""
This guide contains functions and tips to resolve common TensorFlow performance issues,
especially on Apple Silicon (M1/M2) Macs.

Functions:
- diagnose_tf_environment: Print detailed information about TensorFlow and system
- enable_metal_gpu: Configure TensorFlow to use Metal GPU acceleration
- optimize_model_build: Test and optimize model building with progressive complexity
- check_memory_leak: Test for memory leaks during model training
- benchmark_performance: Benchmark model performance with different configurations
"""

import os
import sys
import time
import platform
import tensorflow as tf
import numpy as np

def diagnose_tf_environment():
    """
    Print detailed information about the TensorFlow environment
    and system configuration to help diagnose performance issues.
    """
    print("\n===== TensorFlow Environment Diagnosis =====")
    
    # System information
    print("\n----- System Information -----")
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Check if running on Apple Silicon
    is_apple_silicon = (platform.system() == 'Darwin' and 
                       ('arm64' in platform.machine() or 'ARM64' in platform.machine()))
    print(f"Apple Silicon detected: {is_apple_silicon}")
    
    # Check if Metal plugin is available (for Apple Silicon)
    try:
        from tensorflow.python.framework.errors_impl import NotFoundError
        try:
            tf.config.list_physical_devices('GPU')
            metal_available = True
        except NotFoundError:
            metal_available = False
        print(f"Metal plugin available: {metal_available}")
    except:
        print("Could not determine Metal plugin availability")
    
    # Available devices
    print("\n----- Available Devices -----")
    devices = tf.config.list_physical_devices()
    if not devices:
        print("No devices found.")
    else:
        for device in devices:
            print(f"Device name: {device.name}, type: {device.device_type}")
    
    # Memory settings
    print("\n----- Memory Settings -----")
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        try:
            for gpu in gpu_devices:
                memory_details = tf.config.experimental.get_memory_info(gpu.name)
                print(f"Device: {gpu.name}")
                print(f"  Current memory allocation: {memory_details['current'] / 1e9:.2f} GB")
                print(f"  Peak memory allocation: {memory_details['peak'] / 1e9:.2f} GB")
        except:
            print("Memory information not available")
    
    # Environment variables
    print("\n----- TensorFlow-Related Environment Variables -----")
    tf_env_vars = [var for var in os.environ if 'TF_' in var]
    if tf_env_vars:
        for var in tf_env_vars:
            print(f"{var}={os.environ[var]}")
    else:
        print("No TensorFlow-specific environment variables set")
    
    return is_apple_silicon

def enable_metal_gpu(debug_level=2):
    """
    Configure TensorFlow to use Metal GPU acceleration on Apple Silicon.
    
    Parameters:
    -----------
    debug_level : int
        0 = minimal logging
        1 = info logging
        2 = verbose logging
    
    Returns:
    --------
    bool
        Whether Metal GPU was successfully enabled
    """
    # Check if running on MacOS with Apple Silicon
    is_apple_silicon = (platform.system() == 'Darwin' and 
                       ('arm64' in platform.machine() or 'ARM64' in platform.machine()))
    
    if not is_apple_silicon:
        print("Not running on Apple Silicon - skipping Metal GPU configuration")
        return False
    
    # Set environment variables for TensorFlow Metal plugin
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Set debug level
    if debug_level == 0:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs
    elif debug_level == 1:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Show INFO logs
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs
        os.environ['TF_METAL_DEVICE_DELEGATE_DEBUG'] = '1'  # Enable Metal debug
    
    # Try to detect Metal GPU
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if not physical_devices:
            print("No Metal GPU found. TensorFlow will use CPU.")
            return False
        
        print(f"Found {len(physical_devices)} Metal GPU(s)")
        
        # Configure memory growth
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {gpu}")
        
        # Run a simple test on GPU
        print("Testing Metal GPU with a simple operation...")
        with tf.device('/GPU:0'):
            start_time = time.time()
            a = tf.random.normal((1000, 1000))
            b = tf.random.normal((1000, 1000))
            c = tf.matmul(a, b)
            # Force execution
            result = c.numpy()
            execution_time = time.time() - start_time
            print(f"Test completed in {execution_time:.4f}s")
        
        return True
    
    except Exception as e:
        print(f"Error configuring Metal GPU: {str(e)}")
        print("Falling back to CPU execution")
        return False

# def build_simplified_model(input_shape):
#     inputs = Input(shape=input_shape)
#     x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
#     x = GlobalAveragePooling1D()(x)
#     x = Dense(32, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     outputs = Dense(1, activation='sigmoid')(x)
#     model = Model(inputs, outputs)
#     model.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
#     )
#     return model

# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Layer
# input_shape = X[train_indices[0]].shape
# model = build_simplified_model(input_shape)

# from tensorflow.keras.utils import plot_model

# plot_model(model, to_file='zero_curtain_pipeline/modeling/spatial_model/simplified_insitu_model_pl...
#            show_layer_names=True, expand_nested=True, dpi=300, layer_range=None, show_layer_activa...

# def create_tf_dataset_from_generator(data_generator, output_signature, buffer_size=10000):
#     """
#     Create a TensorFlow Dataset from a Keras Sequence generator.
    
#     Parameters:
#     -----------
#     data_generator : Sequence
#         Keras Sequence generator
#     output_signature : tuple
#         Output signature for the dataset
#     buffer_size : int
#         Size of shuffle buffer
        
#     Returns:
#     --------
#     tf.data.Dataset
#         Dataset ready for model training/evaluation
#     """
#     # Define generator function that wraps the Keras Sequence
#     def tf_generator():
#         for batch_index in range(len(data_generator)):
#             batch = data_generator[batch_index]
#             # If batch is a tuple, yield elements individually
#             if isinstance(batch, tuple):
# for i in range(len(batch[0])): # For each...
#                     # Extract individual items from the batch
#                     if len(batch) == 2:  # (x, y)
#                         yield batch[0][i], batch[1][i]
#                     elif len(batch) == 3:  # (x, y, weights)
#                         yield batch[0][i], batch[1][i], batch[2][i]
    
#     # Create dataset
#     dataset = tf.data.Dataset.from_generator(
#         tf_generator,
#         output_signature=output_signature
#     )
    
#     # Determine if shuffling is needed based on the generator
#     if data_generator.shuffle:
#         dataset = dataset.shuffle(buffer_size)
    
#     # Batch and prefetch
#     dataset = dataset.batch(data_generator.batch_size)
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
#     return dataset

# output_dir = os.path.join('/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling', 'spa...

# # Set up callbacks
# callbacks = [
#     # Stop training when validation performance plateaus
#     tf.keras.callbacks.EarlyStopping(
#         patience=15,
#         restore_best_weights=True,
#         monitor='val_auc',
#         mode='max'
#     ),
#     # Reduce learning rate when improvement slows
#     tf.keras.callbacks.ReduceLROnPlateau(
#         factor=0.5,
#         patience=7,
#         min_lr=1e-6,
#         monitor='val_auc',
#         mode='max'
#     ),
#     # Manual garbage collection after each epoch
#     tf.keras.callbacks.LambdaCallback(
#         on_epoch_end=lambda epoch, logs: gc.collect()
#     )
# ]

# # Add additional callbacks if output directory provided
# if output_dir:
#     callbacks.extend([
#         # Save best model
#         tf.keras.callbacks.ModelCheckpoint(
#             os.path.join(output_dir, 'model_checkpoint.h5'),
#             save_best_only=True,
#             monitor='val_auc',
#             mode='max'
#         ),
#         # Log training progress to CSV
#         tf.keras.callbacks.CSVLogger(
#             os.path.join(output_dir, 'training_log.csv'),
#             append=True
#         )
#     ])

def create_tf_dataset_from_generator(data_generator, output_signature, has_weights=True, buffer_size=10000):
    """
    Create a TensorFlow Dataset from a Keras Sequence generator.
    
    Parameters:
    -----------
    data_generator : Sequence
        Keras Sequence generator
    output_signature : tuple
        Output signature for the dataset
    has_weights : bool
        Whether the generator produces sample weights
    buffer_size : int
        Size of shuffle buffer
        
    Returns:
    --------
    tf.data.Dataset
        Dataset ready for model training/evaluation
    """
    # Define generator function that wraps the Keras Sequence
    def tf_generator():
        for batch_index in range(len(data_generator)):
            batch = data_generator[batch_index]
            # Process the batch based on its structure
            if isinstance(batch, tuple):
                if len(batch) == 2:  # (x, y) format
                    X_batch, y_batch = batch
                    for i in range(len(X_batch)):
                        # If weights are expected but not provided, use dummy weights
                        if has_weights:
                            yield X_batch[i], y_batch[i], 1.0
                        else:
                            yield X_batch[i], y_batch[i]
                elif len(batch) == 3:  # (x, y, weights) format
                    X_batch, y_batch, w_batch = batch
                    for i in range(len(X_batch)):
                        yield X_batch[i], y_batch[i], w_batch[i]
            else:
                raise ValueError("Unexpected batch format")
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        tf_generator,
        output_signature=output_signature
    )
    
    # Determine if shuffling is needed based on the generator
    if data_generator.shuffle:
        dataset = dataset.shuffle(buffer_size)
    
    # Batch and prefetch
    dataset = dataset.batch(data_generator.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

#Now try a more sophisticated architecture from before
from tensorflow.keras.optimizers import Adam
input_shape = X[train_indices[0]].shape
model = build_advanced_zero_curtain_model(input_shape)

from tensorflow.keras.utils import plot_model

plot_model(model, to_file='zero_curtain_pipeline/modeling/spatial_model/insitu_model_plot.png', show_shapes=True, show_dtype=True, \
           show_layer_names=True, expand_nested=True, dpi=300, layer_range=None, show_layer_activations=True);#, rankdir='TB')

# def create_tf_dataset_from_generator(data_generator, output_signature, buffer_size=10000):
#     """
#     Create a TensorFlow Dataset from a Keras Sequence generator.
    
#     Parameters:
#     -----------
#     data_generator : Sequence
#         Keras Sequence generator
#     output_signature : tuple
#         Output signature for the dataset
#     buffer_size : int
#         Size of shuffle buffer
        
#     Returns:
#     --------
#     tf.data.Dataset
#         Dataset ready for model training/evaluation
#     """
#     # Define generator function that wraps the Keras Sequence
#     def tf_generator():
#         for batch_index in range(len(data_generator)):
#             batch = data_generator[batch_index]
#             # If batch is a tuple, yield elements individually
#             if isinstance(batch, tuple):
# for i in range(len(batch[0])): # For each...
#                     # Extract individual items from the batch
#                     if len(batch) == 2:  # (x, y)
#                         yield batch[0][i], batch[1][i]
#                     elif len(batch) == 3:  # (x, y, weights)
#                         yield batch[0][i], batch[1][i], batch[2][i]
    
#     # Create dataset
#     dataset = tf.data.Dataset.from_generator(
#         tf_generator,
#         output_signature=output_signature
#     )
    
#     # Determine if shuffling is needed based on the generator
#     if data_generator.shuffle:
#         dataset = dataset.shuffle(buffer_size)
    
#     # Batch and prefetch
#     dataset = dataset.batch(data_generator.batch_size)
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
#     return dataset

# def train_in_chunks(model, X_file, y_file, train_indices, val_indices, 
#                     batch_size=1024, chunks=10, epochs_per_chunk=10, 
#                     callbacks=None, class_weight=None, sample_weights=None):
#     """
#     Train model sequentially in chunks to minimize memory usage.
    
#     Parameters:
#     -----------
#     model : tf.keras.Model
#         Pre-compiled model
#     X_file : str
#         Path to features file (memory-mapped)
#     y_file : str
#         Path to labels file (memory-mapped)
#     train_indices, val_indices : array
#         Training and validation indices
#     batch_size : int
#         Batch size for training
#     chunks : int
#         Number of chunks to split indices into
#     epochs_per_chunk : int
#         Number of epochs to train each chunk
#     """
#     X = np.load(X_file, mmap_mode='r')
#     y = np.load(y_file, mmap_mode='r')
    
#     # Split indices into chunks
#     train_chunks = np.array_split(train_indices, chunks)
    
#     history_aggregate = None
    
#     for chunk_idx, chunk_indices in enumerate(train_chunks):
#         print(f"Training on chunk {chunk_idx+1}/{len(train_chunks)} with {len(chunk_indices)} samp...
        
#         # Create generators for this chunk only
#         train_gen = DataGenerator(X, y, chunk_indices, batch_size=batch_size, 
#                                   shuffle=True, weights=sample_weights[chunk_indices] if sample_we...
#         val_gen = DataGenerator(X, y, val_indices, batch_size=batch_size, shuffle=False)
        
#         # Train on this chunk
#         history = model.fit(
#             train_gen,
#             validation_data=val_gen,
#             epochs=epochs_per_chunk,
#             callbacks=callbacks,
#             class_weight=class_weight,
#             verbose=1,
#             use_multiprocessing=False,
#             workers=1
#         )
        
#         # Aggregate history
#         if history_aggregate is None:
#             history_aggregate = {k: v for k, v in history.history.items()}
#         else:
#             for k, v in history.history.items():
#                 history_aggregate[k].extend(v)
        
#         # Force garbage collection
#         del train_gen
#         gc.collect()
    
#     return history_aggregate

output_dir = os.path.join('/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling', 'spatial_model')

# Set up callbacks
callbacks = [
    # Stop training when validation performance plateaus
    tf.keras.callbacks.EarlyStopping(
        patience=15,
        restore_best_weights=True,
        monitor='val_auc',
        mode='max'
    ),
    # Reduce learning rate when improvement slows
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        monitor='val_auc',
        mode='max'
    ),
    # Manual garbage collection after each epoch
    tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: gc.collect()
    )
]

# Add additional callbacks if output directory provided
if output_dir:
    callbacks.extend([
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'model_checkpoint.h5'),
            save_best_only=True,
            monitor='val_auc',
            mode='max'
        ),
        # Log training progress to CSV
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_log.csv'),
            append=True
        )
    ])

def train_in_chunks_with_tf_datasets(model, X_file, y_file, train_indices, val_indices,
                                     batch_size=256, chunks=20, epochs_per_chunk=5,
                                     callbacks=None, class_weight=None, sample_weights=None):
    """
    Train model sequentially in chunks using TensorFlow datasets with detailed progress logging.
    """
    import time
    from datetime import timedelta
    
    # Log start time
    start_time = time.time()
    print(f"Starting chunked training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total samples: {len(train_indices)}, Chunks: {chunks}, Batch size: {batch_size}")
    
    # Create validation dataset once
    print(f"Creating validation dataset with {len(val_indices)} samples...")
    val_dataset_start = time.time()
    val_dataset = create_optimized_tf_dataset(
        X_file, y_file, val_indices, 
        batch_size=batch_size, shuffle=False
    )
    print(f"Validation dataset created in {time.time() - val_dataset_start:.2f} seconds")
    
    # Split training indices into chunks
    train_chunks = np.array_split(train_indices, chunks)
    
    history_aggregate = None
    
    # Train on each chunk sequentially
    for chunk_idx, chunk_indices in enumerate(train_chunks):
        chunk_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"Training on chunk {chunk_idx+1}/{len(train_chunks)} with {len(chunk_indices)} samples")
        print(f"Memory usage before creating dataset: {memory_usage():.1f} MB")
        
        # Create TF dataset for this chunk only
        print(f"Creating training dataset for chunk {chunk_idx+1}...")
        dataset_start = time.time()
        
        if sample_weights is not None:
            # Get weights for current chunk
            print(f"Extracting sample weights for {len(chunk_indices)} samples...")
            weights_start = time.time()
            chunk_weights = np.array([sample_weights[np.where(train_indices == idx)[0][0]] 
                                     for idx in chunk_indices])
            print(f"Weights extracted in {time.time() - weights_start:.2f} seconds")
        else:
            chunk_weights = None
            
        train_dataset = create_optimized_tf_dataset(
            X_file, y_file, chunk_indices,
            batch_size=batch_size, shuffle=True,
            weights=chunk_weights
        )
        print(f"Training dataset created in {time.time() - dataset_start:.2f} seconds")
        print(f"Memory usage after creating dataset: {memory_usage():.1f} MB")
        
        # Count batches for progress reporting
        steps_per_epoch = len(chunk_indices) // batch_size + (1 if len(chunk_indices) % batch_size > 0 else 0)
        print(f"Steps per epoch: {steps_per_epoch}")
        
        # Create a custom callback to log batch progress
        class BatchProgressCallback(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                if batch % 10 == 0:  # Log every 10 batches
                    print(f"  Batch {batch}/{steps_per_epoch}, Loss: {logs['loss']:.4f}")
        
        # Add our progress callback
        chunk_callbacks = callbacks.copy() if callbacks else []
        chunk_callbacks.append(BatchProgressCallback())
        
        # Train on this chunk
        print(f"Starting training for {epochs_per_chunk} epochs on chunk {chunk_idx+1}...")
        train_start = time.time()
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs_per_chunk,
            callbacks=chunk_callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        train_time = time.time() - train_start
        print(f"Chunk {chunk_idx+1} training completed in {timedelta(seconds=int(train_time))}")
        print(f"Average time per epoch: {train_time/epochs_per_chunk:.2f} seconds")
        
        # Aggregate history
        if history_aggregate is None:
            history_aggregate = {k: v for k, v in history.history.items()}
        else:
            for k, v in history.history.items():
                history_aggregate[k].extend(v)
        
        # Force garbage collection
        del train_dataset
        if chunk_weights is not None:
            del chunk_weights
        gc.collect()
        
        # Log memory usage after training
        print(f"Memory usage after training: {memory_usage():.1f} MB")
        print(f"Chunk {chunk_idx+1} total time: {timedelta(seconds=int(time.time() - chunk_start_time))}")
        
        # Estimate remaining time
        elapsed_time = time.time() - start_time
        avg_chunk_time = elapsed_time / (chunk_idx + 1)
        remaining_chunks = chunks - (chunk_idx + 1)
        estimated_remaining = avg_chunk_time * remaining_chunks
        print(f"Estimated remaining time: {timedelta(seconds=int(estimated_remaining))}")
        print(f"{'='*80}")
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {timedelta(seconds=int(total_time))}")
    
    # Create a History object with the aggregated metrics
    agg_history = type('History', (), {'history': history_aggregate})
    
    return agg_history

import time

# Diagnostic test with proper tensor conversion
print("Testing batch processing...")
X = np.load(X_file, mmap_mode='r')
y = np.load(y_file, mmap_mode='r')

# Get a small sample batch and convert to tensors
sample_indices = train_indices[:32]
X_sample = tf.convert_to_tensor(np.array([X[i] for i in sample_indices]), dtype=tf.float32)
y_sample = tf.convert_to_tensor(np.array([y[i] for i in sample_indices]), dtype=tf.float32)

print(f"Sample batch shapes: X={X_sample.shape}, y={y_sample.shape}")
print(f"Testing forward pass...")
start = time.time()
y_pred = model(X_sample)
print(f"Forward pass completed in {time.time() - start:.2f} seconds")

# Reshape y_sample if needed
if len(y_sample.shape) == 1 and len(y_pred.shape) == 2:
    y_sample = tf.reshape(y_sample, (-1, 1))
    print(f"Reshaped y_sample to {y_sample.shape} to match y_pred {y_pred.shape}")

print(f"Testing backward pass...")
start = time.time()
with tf.GradientTape() as tape:
    y_pred = model(X_sample, training=True)
    # Use tf.keras.losses directly instead of model.compiled_loss
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    loss = loss_fn(y_sample, y_pred)
gradients = tape.gradient(loss, model.trainable_variables)
model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
print(f"Backward pass completed in {time.time() - start:.2f} seconds")

output_dir = os.path.join('/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling', 'efficient_model')

# Set up callbacks
callbacks = [
    # Stop training when validation performance plateaus
    tf.keras.callbacks.EarlyStopping(
        patience=15,
        restore_best_weights=True,
        monitor='val_auc',
        mode='max'
    ),
    # Reduce learning rate when improvement slows
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        monitor='val_auc',
        mode='max'
    ),
    # Manual garbage collection after each epoch
    tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: gc.collect()
    )
]

# Add additional callbacks if output directory provided
if output_dir:
    callbacks.extend([
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'model_checkpoint.h5'),
            save_best_only=True,
            monitor='val_auc',
            mode='max'
        ),
        # Log training progress to CSV
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_log.csv'),
            append=True
        )
    ])

# def resumable_efficient_training(model, X_file, y_file, train_indices, val_indices, test_indices,
#                                output_dir, batch_size=256, chunk_size=25000, epochs_per_chunk=2, 
#                                save_frequency=5, class_weight=None, start_chunk=360):
# def resumable_efficient_training(model, X_file, y_file, train_indices, val_indices, test_indices,
#                                output_dir, batch_size=256, chunk_size=25000, epochs_per_chunk=2, 
#                                save_frequency=5, class_weight=None, start_chunk=405):
# def resumable_efficient_training(model, X_file, y_file, train_indices, val_indices, test_indices,
#                                output_dir, batch_size=256, chunk_size=25000, epochs_per_chunk=2, 
#                                save_frequency=5, class_weight=None, start_chunk=450):
# def resumable_efficient_training(model, X_file, y_file, train_indices, val_indices, test_indices,
#                                output_dir, batch_size=256, chunk_size=25000, epochs_per_chunk=2, 
#                                save_frequency=5, class_weight=None, start_chunk=495):
def resumable_efficient_training(model, X_file, y_file, train_indices, val_indices, test_indices,
                               output_dir, batch_size=256, chunk_size=25000, epochs_per_chunk=2, 
                               save_frequency=5, class_weight=None, start_chunk=540):
    """
    Training function with built-in resume capability.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Pre-compiled model
    X_file, y_file : str
        Paths to feature and label files
    train_indices, val_indices, test_indices : array
        Training, validation, and test indices
    output_dir : str
        Directory to save results
    batch_size : int
        Batch size for training
    chunk_size : int
        Number of samples to process at once
    epochs_per_chunk : int
        Epochs to train each chunk
    save_frequency : int
        Save model every N chunks
    class_weight : dict, optional
        Class weights for handling imbalanced data
    start_chunk : int, optional
        Chunk index to start/resume from
    """
    import os
    import gc
    import json
    import numpy as np
    import tensorflow as tf
    from datetime import datetime, timedelta
    import time
    import psutil
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Process in chunks
    num_chunks = int(np.ceil(len(train_indices) / chunk_size))
    print(f"Processing {len(train_indices)} samples in {num_chunks} chunks of {chunk_size}")
    print(f"Starting from chunk {start_chunk+1}")
    
    # Create validation set once
    val_limit = min(2000, len(val_indices))
    val_indices_subset = val_indices[:val_limit]
    
    # Open data files
    X_mmap = np.load(X_file, mmap_mode='r')
    y_mmap = np.load(y_file, mmap_mode='r')
    
    # Load validation data once
    val_X = np.array([X_mmap[idx] for idx in val_indices_subset])
    val_y = np.array([y_mmap[idx] for idx in val_indices_subset])
    print(f"Loaded {len(val_X)} validation samples")
    
    # Load existing history if resuming
    history_log = []
    history_path = os.path.join(output_dir, "training_history.json")
    if start_chunk > 0 and os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history_log = json.load(f)
        except Exception as e:
            print(f"Could not load existing history: {e}")
            # Try pickle format
            pickle_path = os.path.join(output_dir, "training_history.pkl")
            if os.path.exists(pickle_path):
                import pickle
                with open(pickle_path, "rb") as f:
                    history_log = pickle.load(f)
    
    # If resuming, load latest model
    if start_chunk > 0:
        # Find the most recent checkpoint before start_chunk
        checkpoint_indices = []
        for filename in os.listdir(checkpoints_dir):
            if filename.startswith("model_checkpoint_") and filename.endswith(".h5"):
                try:
                    idx = int(filename.split("_")[-1].split(".")[0])
                    if idx < start_chunk:
                        checkpoint_indices.append(idx)
                except ValueError:
                    continue
        
        if checkpoint_indices:
            latest_idx = max(checkpoint_indices)
            model_path = os.path.join(checkpoints_dir, f"model_checkpoint_{latest_idx}.h5")
            if os.path.exists(model_path):
                print(f"Loading model from checkpoint {model_path}")
                model = tf.keras.models.load_model(model_path)
            else:
                print(f"Warning: Could not find model checkpoint for chunk {latest_idx}")
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=3, 
            restore_best_weights=True,
            monitor='val_loss',
            min_delta=0.01
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            monitor='val_loss'
        ),
        # Memory cleanup after each epoch
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect()
        )
    ]
    
    # Track metrics across chunks
    start_time = time.time()
    
    # For safe recovery
    recovery_file = os.path.join(output_dir, "last_completed_chunk.txt")
    
    # Process each chunk
    for chunk_idx in range(start_chunk, num_chunks):
        # Get chunk indices
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(train_indices))
        chunk_indices = train_indices[start_idx:end_idx]
        
        # Report memory
        memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print(f"\n{'='*50}")
        print(f"Processing chunk {chunk_idx+1}/{num_chunks} with {len(chunk_indices)} samples")
        print(f"Memory before: {memory_before:.1f} MB")
        
        # Force garbage collection before loading new data
        gc.collect()
        
        # Load chunk data
        chunk_X = np.array([X_mmap[idx] for idx in chunk_indices])
        chunk_y = np.array([y_mmap[idx] for idx in chunk_indices])
        
        print(f"Data loaded. Memory: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.1f} MB")
        
        # Train on chunk
        print(f"Training for {epochs_per_chunk} epochs...")
        history = model.fit(
            chunk_X, chunk_y,
            validation_data=(val_X, val_y),
            epochs=epochs_per_chunk,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store serializable metrics
        chunk_metrics = {}
        for k, v in history.history.items():
            chunk_metrics[k] = [float(val) for val in v]
        history_log.append(chunk_metrics)
        
        # Save model periodically instead of after every chunk
        if (chunk_idx + 1) % save_frequency == 0 or chunk_idx == num_chunks - 1:
            model_path = os.path.join(checkpoints_dir, f"model_checkpoint_{chunk_idx+1}.h5")
            model.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Also save history
            try:
                with open(history_path, "w") as f:
                    json.dump(history_log, f)
            except Exception as e:
                print(f"Warning: Could not save history to JSON: {e}")
                # Fallback - save as pickle
                import pickle
                with open(os.path.join(output_dir, "training_history.pkl"), "wb") as f:
                    pickle.dump(history_log, f)
        
        # Generate predictions only for selected chunks to save time
        if (chunk_idx + 1) % save_frequency == 0 or chunk_idx == num_chunks - 1:
            chunk_preds = model.predict(chunk_X, batch_size=batch_size)
            np.save(os.path.join(predictions_dir, f"chunk_{chunk_idx+1}_predictions.npy"), chunk_preds)
            np.save(os.path.join(predictions_dir, f"chunk_{chunk_idx+1}_indices.npy"), chunk_indices)
            del chunk_preds
            
        # Explicitly delete everything from memory
        del chunk_X, chunk_y
        
        # Force garbage collection
        gc.collect()
        
        # Report memory after cleanup
        memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print(f"Memory after: {memory_after:.1f} MB (Change: {memory_after - memory_before:.1f} MB)")
        
        # Write recovery file with last completed chunk
        with open(recovery_file, "w") as f:
            f.write(str(chunk_idx + 1))
        
        # Estimate time
        elapsed = time.time() - start_time
        avg_time_per_chunk = elapsed / (chunk_idx - start_chunk + 1)
        remaining = avg_time_per_chunk * (num_chunks - chunk_idx - 1)
        print(f"Estimated remaining time: {timedelta(seconds=int(remaining))}")
        
        # If we're approaching the problematic chunk, reset tensorflow session
        if (chunk_idx + 1) % 40 == 0:
            print("Approaching potential freeze point - resetting TensorFlow session")
            # Save model for this chunk
            temp_model_path = os.path.join(checkpoints_dir, f"temp_reset_point_{chunk_idx+1}.h5")
            model.save(temp_model_path)
            
            # Clear session
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Reload model
            model = tf.keras.models.load_model(temp_model_path)
            print("TensorFlow session reset complete")
    
    # Training complete - save final model
    final_model_path = os.path.join(output_dir, "final_model.h5")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save all metrics
    try:
        with open(os.path.join(output_dir, "final_training_metrics.json"), "w") as f:
            json.dump(history_log, f)
    except Exception as e:
        print(f"Error saving metrics: {e}")
        # Save as pickle instead
        import pickle
        with open(os.path.join(output_dir, "final_training_metrics.pkl"), "wb") as f:
            pickle.dump(history_log, f)
    
    # Clean up validation data
    del val_X, val_y
    gc.collect()
    
    # Final evaluation on test set
    print("\nPerforming final evaluation on test set...")
    # [Rest of evaluation code remains the same]
    
    return model, final_model_path

# # Define output directory
# output_dir = os.path.join('/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling', 'mem...

# # Configure initial TensorFlow memory settings
# configure_tensorflow_memory()

# # Use your existing model architecture - no changes needed
# input_shape = X[train_indices[0]].shape
# model = build_advanced_zero_curtain_model(input_shape)

def create_tf_dataset_from_generator(data_generator, output_signature, has_weights=True, buffer_size=10000):
    """
    Create a TensorFlow Dataset from a Keras Sequence generator.
    
    Parameters:
    -----------
    data_generator : Sequence
        Keras Sequence generator
    output_signature : tuple
        Output signature for the dataset
    has_weights : bool
        Whether the generator produces sample weights
    buffer_size : int
        Size of shuffle buffer
        
    Returns:
    --------
    tf.data.Dataset
        Dataset ready for model training/evaluation
    """
    # Define generator function that wraps the Keras Sequence
    def tf_generator():
        for batch_index in range(len(data_generator)):
            batch = data_generator[batch_index]
            # Process the batch based on its structure
            if isinstance(batch, tuple):
                if len(batch) == 2:  # (x, y) format
                    X_batch, y_batch = batch
                    for i in range(len(X_batch)):
                        # If weights are expected but not provided, use dummy weights
                        if has_weights:
                            yield X_batch[i], y_batch[i], 1.0
                        else:
                            yield X_batch[i], y_batch[i]
                elif len(batch) == 3:  # (x, y, weights) format
                    X_batch, y_batch, w_batch = batch
                    for i in range(len(X_batch)):
                        yield X_batch[i], y_batch[i], w_batch[i]
            else:
                raise ValueError("Unexpected batch format")
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        tf_generator,
        output_signature=output_signature
    )
    
    # Determine if shuffling is needed based on the generator
    if data_generator.shuffle:
        dataset = dataset.shuffle(buffer_size)
    
    # Batch and prefetch
    dataset = dataset.batch(data_generator.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

#Now try a more sophisticated architecture from before
from tensorflow.keras.optimizers import Adam
input_shape = X[train_indices[0]].shape
model = build_advanced_zero_curtain_model(input_shape)

from tensorflow.keras.utils import plot_model

plot_model(model, to_file='zero_curtain_pipeline/modeling/spatial_model/insitu_model_plot.png', show_shapes=True, show_dtype=True, \
           show_layer_names=True, expand_nested=True, dpi=300, layer_range=None, show_layer_activations=True);#, rankdir='TB')

output_dir = os.path.join('/Users/[USER]/Desktop/Research/Code/zero_curtain_pipeline/modeling', 'spatial_model')

# Set up callbacks
callbacks = [
    # Stop training when validation performance plateaus
    tf.keras.callbacks.EarlyStopping(
        patience=15,
        restore_best_weights=True,
        monitor='val_auc',
        mode='max'
    ),
    # Reduce learning rate when improvement slows
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        monitor='val_auc',
        mode='max'
    ),
    # Manual garbage collection after each epoch
    tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: gc.collect()
    )
]

# Add additional callbacks if output directory provided
if output_dir:
    callbacks.extend([
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'model_checkpoint.h5'),
            save_best_only=True,
            monitor='val_auc',
            mode='max'
        ),
        # Log training progress to CSV
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_log.csv'),
            append=True
        )
    ])

# Cyclical Learning Rate Callback for smoother training
class CyclicLR(tf.keras.callbacks.Callback):
    """
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    a constant frequency.
    
    Arguments:
        base_lr: Initial learning rate which is the lower boundary in the cycle.
        max_lr: Upper boundary in the cycle. The learning rate at any cycle is the
                maximum of base_lr and the value calculated for that cycle.
        step_size: Number of training iterations per half cycle.
        mode: One of {triangular, triangular2, exp_range}.
              Default 'triangular2'.
              - triangular: A basic triangular cycle.
              - triangular2: A triangle cycle that scales initial amplitude by half each cycle.
              - exp_range: A cycle that scales initial amplitude by gamma^cycle.
        gamma: Constant used for 'exp_range' mode.
                cycles initial amplitude by gamma each cycle.
    """
    def __init__(
        self,
        base_lr=0.0001,
        max_lr=0.001,
        step_size=2000,
        mode='triangular2',
        gamma=1.0
    ):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        
        if self.mode == 'triangular':
            self.scale_fn = lambda x: 1.
        elif self.mode == 'triangular2':
            self.scale_fn = lambda x: 1 / (2. ** (x - 1))
        elif self.mode == 'exp_range':
            self.scale_fn = lambda x: gamma ** (x)
        else:
            raise ValueError(f"Mode {self.mode} not supported.")
        
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

    def clr(self):
        """Calculate the current learning rate"""
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        
    def on_train_begin(self, logs=None):
        """Initialize learning rate to base_lr at the start of training"""
        logs = logs or {}
        tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)
            
    def on_batch_end(self, batch, logs=None):
        """Update learning rate after each batch"""
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        # Set the learning rate
        lr = self.clr()
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

# Function to create optimized callbacks
def get_enhanced_callbacks(output_dir, fold_idx=None):
    """
    Create enhanced callbacks with better stability and monitoring
    
    Parameters:
    -----------
    output_dir : str
        Directory to save model checkpoints and logs
    fold_idx : int or None
        Fold index for cross-validation
        
    Returns:
    --------
    list
        List of callback objects
    """
    import os
    
    # Create subdirectories
    sub_dir = f"fold_{fold_idx}" if fold_idx is not None else ""
    checkpoint_dir = os.path.join(output_dir, sub_dir, "checkpoints")
    tensorboard_dir = os.path.join(output_dir, sub_dir, "tensorboard")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    callbacks = [
        # Early stopping with appropriate patience
        tf.keras.callbacks.EarlyStopping(
            patience=15,  # Moderate patience
            restore_best_weights=True,
            monitor='val_auc',
            mode='max',
            min_delta=0.001  # Reduced from 0.005 for more sensitivity
        ),
        # Learning rate reduction with moderate patience
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            monitor='val_auc',
            mode='max',
            verbose=1
        ),
        # Model checkpoint to save best model
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, "best_model.h5"),
            save_best_only=True,
            monitor='val_auc',
            mode='max',
            verbose=1
        ),
        # TensorBoard for monitoring
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=False,
            update_freq='epoch'
        ),
        # Cyclical learning rate for better convergence
        CyclicLR(
            base_lr=0.0001,
            max_lr=0.001,
            step_size=2000,
            mode='triangular2'
        ),
        # Memory cleanup after each epoch
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: tf.keras.backend.clear_session()
        )
    ]
    
    return callbacks

# Function to safely process data in chunks for training
def process_chunk_with_batch_safety(model, chunk_X, chunk_y, val_data, batch_size=256, epochs=1):
    """
    Process a data chunk with batch-level error handling to prevent training failures
    
    Parameters:
    -----------
    model : tf.keras.Model
        The model to train
    chunk_X : numpy.ndarray
        Input features
    chunk_y : numpy.ndarray
        Target labels
    val_data : tuple
        Validation data (X_val, y_val)
    batch_size : int
        Batch size for training
    epochs : int
        Number of epochs to train
        
    Returns:
    --------
    dict
        Training history
    """
    import numpy as np
    import gc
    
    # Store metrics across all batches and epochs
    combined_metrics = {
        "loss": [], "accuracy": [], "auc": [], 
        "precision": [], "recall": [], "val_loss": [],
        "val_accuracy": [], "val_auc": [], "val_precision": [],
        "val_recall": []
    }
    
    # Calculate number of batches
    num_batches = len(chunk_X) // batch_size + (1 if len(chunk_X) % batch_size > 0 else 0)
    
    # Progressively try smaller batch sizes in case of memory issues
    effective_batch_sizes = [batch_size, batch_size//2, batch_size//4, 32]
    
    # For each epoch
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_metrics = {k: [] for k in combined_metrics.keys()}
        
        # Process each batch with failure recovery
        for batch_idx in range(num_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(chunk_X))
            
            # Extract batch data
            batch_X = chunk_X[start_idx:end_idx]
            batch_y = chunk_y[start_idx:end_idx]
            
            # Try progressively smaller batch sizes
            training_success = False
            
            for attempt, mini_batch_size in enumerate(effective_batch_sizes):
                if training_success:
                    break
                    
                try:
                    # Train on single batch
                    batch_history = model.fit(
                        batch_X, batch_y,
                        epochs=1,
                        batch_size=mini_batch_size,
                        verbose=0,  # Silent mode to reduce output
                        validation_data=val_data if (batch_idx % 10 == 0) else None
                    )
                    
                    # Success - store metrics
                    training_success = True
                    
                    # Store training metrics
                    for key in epoch_metrics:
                        if key in batch_history.history:
                            epoch_metrics[key].append(batch_history.history[key][0])
                            
                except Exception as e:
                    print(f"Error processing batch {batch_idx+1}/{num_batches} "
                          f"with batch size {mini_batch_size}: {e}")
                    
                    # If this was the last attempt
                    if attempt == len(effective_batch_sizes) - 1:
                        print(f"Skipping batch {batch_idx+1}")
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                metrics_str = " - ".join([
                    f"{k}: {np.mean(v):.4f}" for k, v in epoch_metrics.items() 
                    if v and not k.startswith('val_')
                ])
                print(f"Batch {batch_idx+1}/{num_batches} - {metrics_str}")
            
            # Force garbage collection
            gc.collect()
        
        # Validate after each epoch
        try:
            val_X, val_y = val_data
            val_metrics = model.evaluate(val_X, val_y, verbose=0)
            val_dict = {f"val_{name}": value for name, value in zip(model.metrics_names, val_metrics)}
            
            # Add validation metrics to epoch metrics
            for k, v in val_dict.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)
                
            # Print validation metrics
            val_metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in val_dict.items()])
            print(f"Validation: {val_metrics_str}")
        except Exception as val_e:
            print(f"Error during validation: {val_e}")
        
        # Add epoch metrics to combined metrics
        for key in combined_metrics:
            if key in epoch_metrics and epoch_metrics[key]:
                combined_metrics[key].extend(epoch_metrics[key])
    
    # Return combined metrics as a history-like object
    return {"history": combined_metrics}

# import tensorflow as tf
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from datetime import datetime
# import gc
# import os

# # 1. PHYSICS-INFORMED MODEL DEFINITION
# class PhysicsLayer(tf.keras.layers.Layer):
#     def __init__(self):
#         super().__init__()
#         # Physical constants
#         self.L_fusion = 334000.0  # J/kg, latent heat of fusion for water
#         self.k_ice = 2.22  # W/m/K, thermal conductivity of ice
#         self.k_water = 0.58  # W/m/K, thermal conductivity of water
#         self.c_ice = 2090.0  # J/kg/K, specific heat capacity of ice
#         self.c_water = 4186.0  # J/kg/K, specific heat capacity of water
#         self.rho_ice = 917.0  # kg/m³, density of ice
#         self.rho_water = 1000.0  # kg/m³, density of water
        
#     def build(self, input_shape):
#         self.porosity = self.add_weight(
#             name="porosity",
#             shape=[1],
#             initializer=tf.keras.initializers.Constant(0.4),
#             constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.9)
#         )
#         super().build(input_shape)
    
#     def call(self, inputs):
#         temperatures, time_steps = inputs
        
#         # Use TF gradient operation instead of numpy
#         # Calculate gradients along time axis
#         temp_padded = tf.pad(temperatures, [[0, 0], [1, 1], [0, 0], [0, 0]])
#         temp_gradients = (temp_padded[:, 2:] - temp_padded[:, :-2]) / 2.0
        
#         # Phase transition calculation
#         phase_fraction = tf.sigmoid((temperatures + 0.1) * 100)
        
#         # Thermal properties
# k_eff = (phase_fraction * self.k_water + (1...
        
#         return temperatures + k_eff * temp_gradients

# class ZeroCurtainModel(tf.keras.Model):
#     def __init__(self, temporal_window=30, n_depths=4):
#         super().__init__()
#         self.temporal_window = temporal_window
#         self.n_depths = n_depths
        
#         # Model layers
#         self.encoder = tf.keras.Sequential([
#             tf.keras.layers.Reshape((temporal_window, n_depths, 1)),
#             tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')
#         ])
        
#         self.physics_layer = PhysicsLayer()
#         self.lstm = tf.keras.layers.LSTM(128, return_sequences=True)
        
#         self.classifier = tf.keras.Sequential([
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dropout(0.3),
#             tf.keras.layers.Dense(1, activation='sigmoid')
#         ])
    
#     def call(self, inputs, training=False):
#         # Process temperature data
#         temp_features = self.encoder(inputs)
        
#         # Apply physics constraints
#         time_steps = tf.range(self.temporal_window, dtype=tf.float32)
#         physics_features = self.physics_layer([temp_features, time_steps])
        
#         # Reshape for LSTM: (batch, time, features)
#         reshaped_features = tf.reshape(physics_features, 
#             (-1, self.temporal_window, self.n_depths * 64))
        
#         # Process temporal dynamics
#         temporal_features = self.lstm(reshaped_features)
        
#         # Final classification
#         output = self.classifier(temporal_features)
#         return output

# # 2. DATA PREPROCESSING
# def preprocess_data(temp_df, zc_df, window_size=30, depth_range=(-2, 20)):
#     """Main preprocessing function"""
#     print("Starting preprocessing...")
    
#     # Convert datetime columns
#     temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], errors='coerce', format='mixed')
#     zc_df['start_date'] = pd.to_datetime(zc_df['start_date'], errors='coerce', format='mixed')
#     zc_df['end_date'] = pd.to_datetime(zc_df['end_date'], errors='coerce', format='mixed')
    
#     # Filter depths
#     temp_df = temp_df[temp_df['depth'].between(depth_range[0], depth_range[1])]
    
#     # Process each site
#     processed_data = []
#     unique_sites = temp_df['site_id'].unique()
    
#     for site_idx, site_id in enumerate(unique_sites):
#         if site_idx % 10 == 0:
#             print(f"Processing site {site_idx}/{len(unique_sites)}")
            
#         site_data = temp_df[temp_df['site_id'] == site_id].copy()
        
#         # Create temperature profile
#         pivot = site_data.pivot_table(
#             index='datetime',
#             columns='depth',
#             values='temperature',
#             aggfunc='mean'
#         ).sort_index()
        
#         # Create windows
#         for i in range(len(pivot) - window_size + 1):
#             window = pivot.iloc[i:i+window_size]
            
#             # Check for zero curtain events
#             window_start = window.index[0]
#             window_end = window.index[-1]
            
#             zc_events = zc_df[
#                 (zc_df['site_id'] == site_id) &
#                 (zc_df['start_date'] >= window_start) &
#                 (zc_df['end_date'] <= window_end)
#             ]
            
#             if not window.isnull().sum().sum() / (window.shape[0] * window.shape[1]) > 0.3:
#                 window_filled = window.interpolate(method='linear').fillna(method='ffill').fillna(...
                
#                 if not window_filled.isnull().any().any():
#                     processed_data.append({
#                         'window_data': window_filled.values,
#                         'has_zero_curtain': 1 if not zc_events.empty else 0
#                     })
        
#         # Clear memory
#         del site_data
#         gc.collect()
    
#     return processed_data

# # 3. DATASET CREATION
# def create_datasets(processed_data, batch_size=32, val_split=0.2):
#     """Create TensorFlow datasets for training"""
#     # Prepare features and labels
#     X = np.array([d['window_data'] for d in processed_data])
#     y = np.array([d['has_zero_curtain'] for d in processed_data])
    
#     # Split indices
#     n_val = int(len(X) * val_split)
#     indices = np.random.permutation(len(X))
#     train_idx, val_idx = indices[n_val:], indices[:n_val]
    
#     # Create datasets
#     train_dataset = tf.data.Dataset.from_tensor_slices(
#         (X[train_idx], y[train_idx])
#     ).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
#     val_dataset = tf.data.Dataset.from_tensor_slices(
#         (X[val_idx], y[val_idx])
#     ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
#     return train_dataset, val_dataset

# class ZeroCurtainModel(tf.keras.Model):
#     def __init__(self, temporal_window=30, n_depths=4):
#         super().__init__()
        
#         # Model layers - simplified for debugging
#         self.lstm = tf.keras.layers.LSTM(64, return_sequences=False)
        
#         self.dense = tf.keras.Sequential([
#             tf.keras.layers.Dense(32, activation='relu'),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Dense(1, activation='sigmoid')
#         ])
    
#     def call(self, inputs):
#         # Reshape if needed - inputs shape is (batch, time, depths)
#         x = self.lstm(inputs)
#         output = self.dense(x)
#         return output

# class ZeroCurtainModel(tf.keras.Model):
#     def __init__(self, temporal_window=30, n_depths=4):
#         super().__init__()
#         self.temporal_window = temporal_window
#         self.n_depths = n_depths
        
#         # Simplified layers
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
    
#     def call(self, inputs):
#         x = self.flatten(inputs)
#         return self.dense(x)

# class ZeroCurtainModel(tf.keras.Model):
#     def __init__(self, temporal_window=30, n_depths=4):
#         super().__init__()
#         self.temporal_window = temporal_window
#         self.n_depths = n_depths
        
#         # Add encoder back
#         self.encoder = tf.keras.Sequential([
#             tf.keras.layers.Reshape((temporal_window, n_depths, 1)),
#             tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')
#         ])
        
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
    
#     def call(self, inputs):
#         x = self.encoder(inputs)
#         x = self.flatten(x)
#         return self.dense(x)

# class PhysicsLayer(tf.keras.layers.Layer):
#     def __init__(self):
#         super().__init__()
#         # Physical constants
#         self.L_fusion = 334000.0  # J/kg, latent heat of fusion for water
#         self.k_ice = 2.22  # W/m/K, thermal conductivity of ice
#         self.k_water = 0.58  # W/m/K, thermal conductivity of water
#         self.c_ice = 2090.0  # J/kg/K, specific heat capacity of ice
#         self.c_water = 4186.0  # J/kg/K, specific heat capacity of water
        
#     def build(self, input_shape):
#         self.porosity = self.add_weight(
#             name="porosity",
#             shape=[1],
#             initializer=tf.keras.initializers.Constant(0.4),
#             constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.9)
#         )
#         self.output_shape = input_shape[0]
#         super().build(input_shape)

#     def compute_output_shape(self, input_shape):
#         return self.output_shape
    
#     def call(self, inputs):
#         temperatures, time_steps = inputs
#         # Manual gradient calculation
#         temp_padded = tf.pad(temperatures, [[0, 0], [1, 1], [0, 0], [0, 0]])
#         temp_gradients = (temp_padded[:, 2:] - temp_padded[:, :-2]) / 2.0
        
#         # Phase transition calculation
#         phase_fraction = tf.sigmoid((temperatures + 0.1) * 100)
        
#         # Thermal properties
# k_eff = (phase_fraction * self.k_water + (1...
        
#         return temperatures + k_eff * temp_gradients
        
# class ZeroCurtainModel(tf.keras.Model):
#     def __init__(self, temporal_window=30, n_depths=4):
#         super().__init__()
#         self.temporal_window = temporal_window
#         self.n_depths = n_depths
        
#         # Add encoder back
#         self.encoder = tf.keras.Sequential([
#             tf.keras.layers.Reshape((temporal_window, n_depths, 1)),
#             tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')
#         ])
#         self.physics_layer = PhysicsLayer()
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
    
#     def call(self, inputs):
#         x = self.encoder(inputs)
#         time_steps = tf.range(self.temporal_window, dtype=tf.float32)
#         x = self.physics_layer([x, time_steps])
#         x = self.flatten(x)
#         return self.dense(x)

# class ZeroCurtainModel(tf.keras.Model):
#     def __init__(self, temporal_window=30, n_depths=4):
#         super().__init__()
#         self.temporal_window = temporal_window
#         self.n_depths = n_depths
        
#         # Encoder
#         self.encoder = tf.keras.Sequential([
#             tf.keras.layers.Reshape((temporal_window, n_depths, 1)),
#             tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')
#         ])
        
#         # Physics layer
#         self.physics_layer = PhysicsLayer()
        
#         # Simple output
#         self.flatten = tf.keras.layers.Flatten()
#         self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    
#     def call(self, inputs):
#         x = self.encoder(inputs)
#         time_steps = tf.range(self.temporal_window, dtype=tf.float32)
#         x = self.physics_layer([x, time_steps])
#         x = self.flatten(x)
#         return self.output_layer(x)

# class ZeroCurtainModel(tf.keras.Model):
#     def __init__(self, temporal_window=30, n_depths=4):
#         super().__init__()
#         self.temporal_window = temporal_window
#         self.n_depths = n_depths
        
#         # Encoder
#         self.encoder = tf.keras.Sequential([
#             tf.keras.layers.Reshape((temporal_window, n_depths, 1)),
#             tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')
#         ])
        
#         # Physics layer
#         self.physics_layer = PhysicsLayer()
        
#         # Temporal processing with Conv1D instead of LSTM
#         self.temporal = tf.keras.Sequential([
#             tf.keras.layers.Reshape((temporal_window, -1)),  # Reshape for Conv1D
#             tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.GlobalAveragePooling1D()
#         ])
        
#         # Output
#         self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    
#     def call(self, inputs):
#         x = self.encoder(inputs)
#         time_steps = tf.range(self.temporal_window, dtype=tf.float32)
#         x = self.physics_layer([x, time_steps])
#         x = self.temporal(x)
#         return self.output_layer(x)

class ZeroCurtainModel(tf.keras.Model):
    def __init__(self, temporal_window=30, n_depths=4):
        super().__init__()
        self.temporal_window = temporal_window
        self.n_depths = n_depths
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Reshape((temporal_window, n_depths, 1)),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')
        ])
        
        # Physics layer
        self.physics_layer = PhysicsLayer()
        
        # Define reshape dimensions based on input size
        channels = 64  # from previous Conv2D layer
        h = 2
        w = 2
        self.reshape = tf.keras.layers.Reshape(
            (temporal_window, h, w, channels * n_depths // (h * w))
        )
        
        # ConvLSTM2D layer
        self.temporal = tf.keras.layers.ConvLSTM2D(
            filters=32,
            kernel_size=(2, 2),
            padding='same',
            return_sequences=False,
            activation='relu'
        )
        
        # Output layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def build(self, input_shape):
        # Build the model with a sample input
        super().build(input_shape)
    
    def call(self, inputs):
        x = self.encoder(inputs)
        time_steps = tf.range(self.temporal_window, dtype=tf.float32)
        x = self.physics_layer([x, time_steps])
        x = self.reshape(x)
        x = self.temporal(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.output_layer(x)

# with tf.device('/CPU:0'):
#     model = ZeroCurtainModel(
#         temporal_window=30, 
#         n_depths=n_depths
#     )
    
#     dummy_input = tf.zeros((1, 30, n_depths))
#     _ = model(dummy_input)
    
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(training_config['learning_rate']),
#         loss='binary_crossentropy',
#         metrics=['accuracy', 
#                  tf.keras.metrics.Precision(name='precision'),
#                  tf.keras.metrics.Recall(name='recall'),
#                  tf.keras.metrics.AUC(name='auc')]
#     )

# with tf.device('/CPU:0'):
#     model = ZeroCurtainModel(
#         temporal_window=30, 
#         n_depths=n_depths
#     )
    
#     for x, y in train_dataset.take(1):
#         print("Input shape:", x.shape)
#         print("Label shape:", y.shape)
# dummy_input = tf.zeros_like(x) # Create dummy input...
    
#     _ = model(dummy_input)
    
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(training_config['learning_rate']),
#         loss='binary_crossentropy',
#         metrics=['accuracy', 
#                  tf.keras.metrics.Precision(name='precision'),
#                  tf.keras.metrics.Recall(name='recall'),
#                  tf.keras.metrics.AUC(name='auc')]
#     )

# with tf.device('/CPU:0'):
#     inputs = tf.keras.Input(shape=input_shape)
#     # Encoder
#     x = tf.keras.layers.Reshape((30, 4, 1))(inputs)
#     x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
#     # Physics layer
#     time_steps = tf.range(30, dtype=tf.float32)
#     physics_layer = PhysicsLayer()
#     x = physics_layer([x, time_steps])
#     # Reshape for ConvLSTM2D
#     x = tf.keras.layers.Reshape((30, 2, 2, -1))(x)
#     # ConvLSTM2D
#     x = tf.keras.layers.ConvLSTM2D(
#         filters=32,
#         kernel_size=(2, 2),
#         padding='same',
#         return_sequences=False,
#         activation='relu'
#     )(x)
#     # Output layers
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(32, activation='relu')(x)
#     outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
#     # Create model
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     # Compile model
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(training_config['learning_rate']),
#         loss='binary_crossentropy',
#         metrics=['accuracy', 
#                 tf.keras.metrics.Precision(name='precision'),
#                 tf.keras.metrics.Recall(name='recall'),
#                 tf.keras.metrics.AUC(name='auc')]
#     )

with tf.device('/CPU:0'):
    # Get input shape from actual data
    for x, y in train_dataset.take(1):
        print("Input shape:", x.shape)
        print("Label shape:", y.shape)
        input_shape = x.shape[1:]
        break
    
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    x = tf.keras.layers.Reshape((30, 10, 1))(inputs)  # Changed to 10 depths
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    
    # Physics layer stays the same
    time_steps = tf.range(30, dtype=tf.float32)
    physics_layer = PhysicsLayer()
    x = physics_layer([x, time_steps])
    
    # Reshape for ConvLSTM2D - adjust for 10 depths
    x = tf.keras.layers.Reshape((30, 2, 5, -1))(x)  # Changed dimensions to work with 10 depths
    
    # ConvLSTM2D
    x = tf.keras.layers.ConvLSTM2D(
        filters=32,
        kernel_size=(2, 2),
        padding='same',
        return_sequences=False,
        activation='relu'
    )(x)
    
    # Output layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create and compile model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=training_config['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    )

callbacks = [
    # Early stopping
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=training_config['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    ),
    
    # Model checkpoint
    tf.keras.callbacks.ModelCheckpoint(
        'best_zero_curtain_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    
    # Learning rate reduction
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-6
    ),
    
    # TensorBoard logging
    tf.keras.callbacks.TensorBoard(
        log_dir=f'logs/zero_curtain_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        histogram_freq=1,
        update_freq='epoch'
    )
]

# with tf.device('/CPU:0'):
#     test_model = tf.keras.Sequential([
#         tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(120,))
#     ])
    
#     test_input = tf.random.normal((1, 120))
#     prediction = test_model(test_input)
#     print("\nTest prediction shape:", prediction.shape)
#     print("Test prediction value:", prediction.numpy())

# with tf.device('/CPU:0'):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Flatten(input_shape=(30, 4)),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
    
#     optimizer = tf.keras.optimizers.Adam(1e-4)
#     loss_fn = tf.keras.losses.BinaryCrossentropy()
    
#     print("Starting manual training loop...")
#     for epoch in range(2):
#         print(f"\nEpoch {epoch + 1}")
        
#         with tf.GradientTape() as tape:
#             predictions = model(test_x)
#             loss = loss_fn(test_y, predictions)
        
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
#         print(f"Loss: {loss.numpy():.4f}")

# with tf.device('/CPU:0'):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Flatten(input_shape=(30, 4)),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
    
#     optimizer = tf.keras.optimizers.Adam(1e-4)
#     loss_fn = tf.keras.losses.BinaryCrossentropy()
    
#     x_train = x_small
#     y_train = y_small
    
#     print("Starting training with actual data...")
#     for epoch in range(5):
#         print(f"\nEpoch {epoch + 1}")
        
#         with tf.GradientTape() as tape:
#             predictions = model(x_train)
#             loss = loss_fn(y_train, predictions)
            
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
#         accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(predictions), y_train), tf.float32))
#         print(f"Loss: {loss.numpy():.4f}, Accuracy: {accuracy.numpy():.4f}")

# print("\nTraining complete!")

# #Full model with training subset (5 epochs)

# with tf.device('/CPU:0'):
#     # Create exact model architecture using functional API
#     inputs = tf.keras.Input(shape=(30, 4))
    
#     # Encoder
#     x = tf.keras.layers.Reshape((30, 4, 1))(inputs)
#     x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    
#     # Physics layer
#     time_steps = tf.range(30, dtype=tf.float32)
#     physics_layer = PhysicsLayer()
#     x = physics_layer([x, time_steps])
    
#     # Reshape for ConvLSTM2D
#     x = tf.keras.layers.Reshape((30, 2, 2, -1))(x)
    
#     # ConvLSTM2D
#     x = tf.keras.layers.ConvLSTM2D(
#         filters=32,
#         kernel_size=(2, 2),
#         padding='same',
#         return_sequences=False,
#         activation='relu'
#     )(x)
    
#     # Output layers
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(32, activation='relu')(x)
#     outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
#     # Create model
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
#     # Setup training components
#     optimizer = tf.keras.optimizers.Adam(training_config['learning_rate'])
#     loss_fn = tf.keras.losses.BinaryCrossentropy()
    
#     # Metrics
#     train_acc_metric = tf.keras.metrics.BinaryAccuracy()
#     train_precision = tf.keras.metrics.Precision()
#     train_recall = tf.keras.metrics.Recall()
#     train_auc = tf.keras.metrics.AUC()

#     model.summary()
    
#     print("Starting training with full physics-informed model...")
    
#     # Manual training loop
#     for epoch in range(5):
#         print(f"\nEpoch {epoch + 1}")
        
#         with tf.GradientTape() as tape:
#             predictions = model(x_train, training=True)
#             loss = loss_fn(y_train, predictions)
        
#         # Compute gradients and update weights
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
#         # Update metrics
#         train_acc_metric.update_state(y_train, predictions)
#         train_precision.update_state(y_train, predictions)
#         train_recall.update_state(y_train, predictions)
#         train_auc.update_state(y_train, predictions)
        
#         print(f"Loss: {loss.numpy():.4f}")
#         print(f"Accuracy: {train_acc_metric.result().numpy():.4f}")
#         print(f"Precision: {train_precision.result().numpy():.4f}")
#         print(f"Recall: {train_recall.result().numpy():.4f}")
#         print(f"AUC: {train_auc.result().numpy():.4f}")
        
#         # Reset metrics for next epoch
#         train_acc_metric.reset_state()  # Changed from reset_states()
#         train_precision.reset_state()   # Changed from reset_states()
#         train_recall.reset_state()      # Changed from reset_states()
#         train_auc.reset_state()         # Changed from reset_states()

# print("\nTraining complete!")

# with tf.device('/CPU:0'):
#     for x, y in train_dataset.take(1):
#         print("Input shape:", x.shape)
#         print("Label shape:", y.shape)
#         input_shape = x.shape[1:]
#         break
        
#     inputs = tf.keras.Input(shape=(30, 10))  # Updated for new depth dimension
    
#     # Encoder
#     x = tf.keras.layers.Reshape((30, 10, 1))(inputs)
#     x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    
#     # Physics layer
#     time_steps = tf.range(30, dtype=tf.float32)
#     physics_layer = PhysicsLayer()
#     x = physics_layer([x, time_steps])
    
#     # Reshape for ConvLSTM2D
#     x = tf.keras.layers.Reshape((30, 2, 5, -1))(x)  # Adjusted for 10 depths
    
#     # ConvLSTM2D
#     x = tf.keras.layers.ConvLSTM2D(
#         filters=32,
#         kernel_size=(2, 2),
#         padding='same',
#         return_sequences=False,
#         activation='relu'
#     )(x)
    
#     # Output layers
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(32, activation='relu')(x)
#     outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
#     # Create optimizer and compile model
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#     model.compile(
#         optimizer=optimizer,
#         loss='binary_crossentropy',
#         metrics=['accuracy', 
#                 tf.keras.metrics.Precision(name='precision'),
#                 tf.keras.metrics.Recall(name='recall'),
#                 tf.keras.metrics.AUC(name='auc')]
#     )
    
#     # Print model summary
#     model.summary()
    
#     # Initialize metrics for custom training
#     train_acc_metric = tf.keras.metrics.BinaryAccuracy()
#     train_precision = tf.keras.metrics.Precision()
#     train_recall = tf.keras.metrics.Recall()
#     train_auc = tf.keras.metrics.AUC()
    
#     # Training configuration
#     batch_size = 32
#     n_samples = len(x_train)
#     n_epochs = 10
    
#     def train_step(x, y):
#         # Ensure consistent data types
#         x = tf.cast(x, tf.float32)
#         y = tf.cast(y, tf.float32)
#         y = tf.reshape(y, (-1, 1))
        
#         with tf.GradientTape() as tape:
#             predictions = model(x, training=True)
#             # Use tf.keras.losses.BinaryCrossentropy instead of manual calculation
#             loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
#             loss = loss_fn(y, predictions)
            
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         return loss, predictions

#     def validation_step(x, y):
#         x = tf.cast(x, tf.float32)
#         y = tf.cast(y, tf.float32)
#         y = tf.reshape(y, (-1, 1))
        
#         predictions = model(x, training=False)
#         loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
#         loss = loss_fn(y, predictions)
#         return loss, predictions
    
#     print(f"Starting training on {n_samples} samples...")
#     best_auc = 0
#     best_weights = None

#     for epoch in range(n_epochs):
#         print(f"\nEpoch {epoch + 1}/{n_epochs}")
#         epoch_loss = 0
#         val_loss = 0
#         n_batches = int(np.ceil(n_samples / batch_size))
#         n_val_batches = int(np.ceil(len(x_val) / batch_size))
        
#         # Reset metrics
#         train_acc_metric.reset_state()
#         train_precision.reset_state()
#         train_recall.reset_state()
#         train_auc.reset_state()
        
#         # Ensure training data is properly cast before shuffling
#         x_train = tf.cast(x_train, tf.float32)
#         y_train = tf.cast(y_train, tf.float32)
        
#         # Shuffle training data
#         #indices = tf.random.shuffle(tf.range(n_samples))
#         #x_train_shuffled = tf.gather(x_train, indices)
#         #y_train_shuffled = tf.gather(y_train, indices)
        
#         # Training loop
#         for batch_idx in range(n_batches):
#             start_idx = batch_idx * batch_size
#             end_idx = min(start_idx + batch_size, n_samples)
            
#             x_batch = x_train[start_idx:end_idx]
#             y_batch = y_train[start_idx:end_idx]
            
#             loss, predictions = train_step(x_batch, y_batch)
#             epoch_loss += loss.numpy()
            
#             # Update metrics
#             train_acc_metric.update_state(y_batch, predictions)
#             train_precision.update_state(y_batch, predictions)
#             train_recall.update_state(y_batch, predictions)
#             train_auc.update_state(y_batch, predictions)
            
#             # Print batch progress
#             print(f"\rBatch {batch_idx + 1}/{n_batches}", end='')
        
#         # Validation loop
#         for batch_idx in range(n_val_batches):
#             start_idx = batch_idx * batch_size
#             end_idx = min(start_idx + batch_size, len(x_val))
            
#             x_batch = x_val[start_idx:end_idx]
#             y_batch = y_val[start_idx:end_idx]
            
#             loss, _ = validation_step(x_batch, y_batch)
#             val_loss += loss.numpy()
        
#         # Calculate metrics
#         epoch_loss = epoch_loss / n_batches
#         val_loss = val_loss / n_val_batches
#         epoch_acc = train_acc_metric.result().numpy()
#         epoch_prec = train_precision.result().numpy()
#         epoch_recall = train_recall.result().numpy()
#         epoch_auc = train_auc.result().numpy()
        
#         # Print results
#         print(f"\nTrain Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
#         print(f"Accuracy: {epoch_acc:.4f}")
#         print(f"Precision: {epoch_prec:.4f}")
#         print(f"Recall: {epoch_recall:.4f}")
#         print(f"AUC: {epoch_auc:.4f}")
        
#         # Save best model
#         if epoch_auc > best_auc:
#             best_auc = epoch_auc
#             best_weights = model.get_weights()
#             print("New best model saved!")

#     # Restore best weights
#     if best_weights is not None:
#         model.set_weights(best_weights)
#         print(f"\nRestored best model with AUC: {best_auc:.4f}")

# print("\nTraining complete!")

# model.save('model.keras')
# #model = load_model('model.keras')

class PermafrostHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        with tf.device('/CPU:0'):
            inputs = tf.keras.Input(shape=self.input_shape)
            
            # Encoder
            x = tf.keras.layers.Reshape((30, 10, 1))(inputs)
            
            # First Conv2D layer
            filters1 = hp.Int('conv1_filters', 16, 64, step=16)
            x = tf.keras.layers.Conv2D(
                filters=filters1,
                kernel_size=(3, 3),
                activation='relu',
                padding='same',
                # Add dilation for better spatial feature capture
                dilation_rate=hp.Choice('dilation_rate', values=[1, 2])
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            
            # Optional second Conv2D
            if hp.Boolean('use_second_conv'):
                filters2 = hp.Int('conv2_filters', 32, 96, step=32)
                x = tf.keras.layers.Conv2D(
                    filters=filters2,
                    kernel_size=(3, 3),
                    activation='relu',
                    padding='same'
                )(x)
            
            # Physics layer
            time_steps = tf.range(30, dtype=tf.float32)
            physics_layer = PhysicsLayer()
            x = physics_layer([x, time_steps])
            
            # Reshape for ConvLSTM2D
            x = tf.keras.layers.Reshape((30, 2, 5, -1))(x)
            
            # ConvLSTM2D
            lstm_filters = hp.Int('lstm_filters', 16, 48, step=16)
            kernel_size = hp.Choice('lstm_kernel_size', values=[2, 3])
            use_second_lstm = hp.Boolean('use_second_lstm')
            x = tf.keras.layers.ConvLSTM2D(
                filters=lstm_filters,
                kernel_size=(kernel_size, kernel_size),
                padding='same',
                return_sequences=use_second_lstm,
                activation='relu',
                recurrent_dropout=hp.Float('recurrent_dropout', 0, 0.3, step=0.1) #for better generalizability
            )(x)

            if use_second_lstm:
                x = tf.keras.layers.ConvLSTM2D(
                    filters=lstm_filters//2,
                    kernel_size=(2, 2),
                    padding='same',
                    return_sequences=False,  # Always false for last layer
                    activation='relu'
                )(x)

            # # Optional second ConvLSTM layer for deeper spatiotemporal patterns
            # if hp.Boolean('use_second_lstm'):
            #     x = tf.keras.layers.ConvLSTM2D(
            #         filters=lstm_filters//2,
            #         kernel_size=(2,2),
            #         padding='same',
            #         return_sequences=False,
            #         activation='relu'
            #     )(x)
            # else:
            # x = tf.keras.layers.Lambda(lambda x: x[:, -1, ...])(x)...

            # Spatial attention mechanism
            if hp.Boolean('use_spatial_attention'):
                attention = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(x)
                x = tf.keras.layers.Multiply()([x, attention])
            
            x = tf.keras.layers.Flatten()(x)
            dense_units = hp.Int('dense_units', 16, 64, step=16)
            x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
            
            if hp.Boolean('use_dropout'):
                x = tf.keras.layers.Dropout(hp.Float('dropout_rate', 0.1, 0.3, step=0.1))(x)
            
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            learning_rate = hp.Float('learning_rate', 1e-4, 1e-3, sampling='log')
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
            return model, optimizer

def train_model_with_hyperparameters(model, optimizer, x_train, y_train, x_val, y_val, batch_size=32, epochs=10):
    with tf.device('/CPU:0'):
        # Initialize metrics
        train_acc_metric = tf.keras.metrics.BinaryAccuracy()
        train_precision = tf.keras.metrics.Precision()
        train_recall = tf.keras.metrics.Recall()
        train_auc = tf.keras.metrics.AUC()
        
        # Initialize history dictionary
        history = {
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'auc': []
        }
        
        n_samples = len(x_train)
        best_val_auc = 0
        best_weights = None
        
        def train_step(x, y):
            x = tf.cast(x, tf.float32)
            y = tf.cast(y, tf.float32)
            y = tf.reshape(y, (-1, 1))
            
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
                loss = loss_fn(y, predictions)
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss, predictions
        
        def validation_step(x, y):
            x = tf.cast(x, tf.float32)
            y = tf.cast(y, tf.float32)
            y = tf.reshape(y, (-1, 1))
            
            predictions = model(x, training=False)
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            loss = loss_fn(y, predictions)
            return loss, predictions
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            val_loss = 0
            
            # Reset metrics
            train_acc_metric.reset_state()
            train_precision.reset_state()
            train_recall.reset_state()
            train_auc.reset_state()
            
            # Training batches - NO SHUFFLING to maintain temporal order
            n_batches = int(np.ceil(n_samples / batch_size))
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                loss, predictions = train_step(x_batch, y_batch)
                epoch_loss += loss.numpy()
                
                # Update metrics
                train_acc_metric.update_state(y_batch, predictions)
                train_precision.update_state(y_batch, predictions)
                train_recall.update_state(y_batch, predictions)
                train_auc.update_state(y_batch, predictions)
            
            # Validation
            n_val_batches = int(np.ceil(len(x_val) / batch_size))
            for batch_idx in range(n_val_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(x_val))
                
                x_batch = x_val[start_idx:end_idx]
                y_batch = y_val[start_idx:end_idx]
                
                loss, _ = validation_step(x_batch, y_batch)
                val_loss += loss.numpy()
            
            # Calculate metrics
            epoch_loss = epoch_loss / n_batches
            val_loss = val_loss / n_val_batches
            epoch_acc = train_acc_metric.result().numpy()
            epoch_prec = train_precision.result().numpy()
            epoch_recall = train_recall.result().numpy()
            epoch_auc = train_auc.result().numpy()
            
            # Store metrics in history
            history['train_loss'].append(float(epoch_loss))
            history['val_loss'].append(float(val_loss))
            history['accuracy'].append(float(epoch_acc))
            history['precision'].append(float(epoch_prec))
            history['recall'].append(float(epoch_recall))
            history['auc'].append(float(epoch_auc))
            
            if epoch_auc > best_val_auc:
                best_val_auc = epoch_auc
                best_weights = model.get_weights()
            
            # Print results
            print(f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Accuracy: {epoch_acc:.4f}")
            print(f"Precision: {epoch_prec:.4f}")
            print(f"Recall: {epoch_recall:.4f}")
            print(f"AUC: {epoch_auc:.4f}")
        
        # Restore best weights
        if best_weights is not None:
            model.set_weights(best_weights)
        
        return best_val_auc, history

def train_hyperparameter_optimization(x_train, y_train, x_val, y_val, input_shape, spatial_info):
    """Run hyperparameter optimization with spatiotemporal awareness"""
    
    # Global tracking
    global_best_auc = 0
    global_best_hyperparameters = None
    global_best_weights = None
    
    # Tracking all trials
    all_trials_history = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Hyperparameter tuning setup
    tuner = kt.BayesianOptimization(
        PermafrostHyperModel(input_shape),
        objective=kt.Objective('val_auc', direction='max'),
        max_trials=15,
        directory=f'permafrost_hyperopt_{timestamp}',
        project_name='zero_curtain_optimization',
        max_consecutive_failed_trials=10
    )

    #spatial_info = np.load('spatial_info.npy', allow_pickle=True).item()
    
    # Manual hyperparameter search with custom training
    for trial in range(15):
        # Create a new trial
        trial_hp = tuner.oracle.create_trial(tuner.tuner_id)
        
        try:
            print(f"\nTrial {trial + 1}/15")
            print("Current hyperparameters:")
            for param, value in trial_hp.hyperparameters.values.items():
                print(f"{param}: {value}")
            
            # Build model and optimizer for this trial
            model, optimizer = tuner.hypermodel.build(trial_hp.hyperparameters)
            
            # Train model using custom training function
            best_trial_auc, trial_history = train_model_with_hyperparameters(
                model, optimizer, 
                x_train, y_train, 
                x_val, y_val, 
                batch_size=32, 
                epochs=10
            )
            
            # Update global best if needed
            if best_trial_auc > global_best_auc:
                global_best_auc = best_trial_auc
                global_best_hyperparameters = trial_hp.hyperparameters.values
                global_best_weights = model.get_weights()

                model.save(f'best_model_intermediate_{timestamp}.h5')
                with open(f'best_hyperparameters_intermediate_{timestamp}.json', 'w') as f:
                    json.dump(convert_to_serializable(global_best_hyperparameters), f, indent=4)
            
            # Store trial history
            all_trials_history[f'trial_{trial+1}'] = {
                'hyperparameters': trial_hp.hyperparameters.values,
                'history': convert_to_serializable(trial_history),
                'trial_auc': best_trial_auc,
                'spatial_config': {
                    'use_spatial_attention': trial_hp.hyperparameters.values.get('use_spatial_attention', False),
                    'lstm_kernel_size': trial_hp.hyperparameters.values.get('lstm_kernel_size', 2),
                    'dilation_rate': trial_hp.hyperparameters.values.get('dilation_rate', 1)
                }
            }
            
            # Update the trial
            tuner.oracle.update_trial(
                trial_id=trial_hp.trial_id, 
                metrics={'val_auc': float(best_trial_auc)}, 
                step=0
            )
            
            # End the trial
            tuner.oracle.end_trial(trial_hp.trial_id, status='COMPLETED')
            
            # Save trial history
            with open(f'training_history_{timestamp}.json', 'w') as f:
                json.dump(convert_to_serializable(all_trials_history), f, indent=4)
            
            print(f"Trial {trial + 1} completed. AUC: {best_trial_auc:.4f}")
            print(f"Global Best AUC: {global_best_auc:.4f}")
            
        except Exception as e:
            print(f"Trial {trial + 1} failed: {str(e)}")
            tuner.oracle.end_trial(trial_hp.trial_id, status='FAILED')
            continue

    print("\nHyperparameter optimization completed. Building final model...")
    
    # Retrain best model with more epochs
    try:
        best_model, best_optimizer = tuner.hypermodel.build(global_best_hyperparameters)
        best_model.set_weights(global_best_weights)
        
        print("\nRetraining best model with extended epochs...")
        final_auc, final_history = train_model_with_hyperparameters(
            best_model, best_optimizer, 
            x_train, y_train, 
            x_val, y_val, 
            batch_size=32, 
            epochs=50  # Extended training for final model
        )
        
        # Save final model and configurations
        best_model.save(f'final_model_{timestamp}.h5')
        #from keras.models import load_model
        #new_model = load_model(filepath)'
        
        final_results = {
            'hyperparameters': convert_to_serializable(global_best_hyperparameters),
            'final_auc': float(final_auc),
            'training_history': convert_to_serializable(final_history),
            'spatial_info': {
                'train_coords': convert_to_serializable(spatial_info['train']['coords']),
                'train_coords': convert_to_serializable(spatial_info['val']['coords']),
                'timestamp': timestamp
            }
        }
        
        with open(f'final_results_{timestamp}.json', 'w') as f:
            json.dump(final_results, f, indent=4)
        
        print(f"\nFinal model AUC: {final_auc:.4f}")
        print(f"All results saved with timestamp: {timestamp}")
        
        return best_model, global_best_hyperparameters, final_results
        
    except Exception as e:
        print(f"Error in final model training: {str(e)}")
        
        # Return best model from trials if final training fails
        return tuner.hypermodel.build(global_best_hyperparameters)[0], global_best_hyperparameters, all_trials_history

def build_final_model_from_hyperparameters(best_hyperparameters, input_shape=(30, 10)):
    """Build the final model using the dictionary of best hyperparameters"""
    with tf.device('/CPU:0'):
        inputs = tf.keras.Input(shape=input_shape)
        
        # Encoder
        x = tf.keras.layers.Reshape((30, 10, 1))(inputs)
        
        # First Conv2D layer
        x = tf.keras.layers.Conv2D(
            filters=best_hyperparameters['conv1_filters'],
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            dilation_rate=best_hyperparameters['dilation_rate']
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Optional second Conv2D
        if best_hyperparameters.get('use_second_conv', False):
            x = tf.keras.layers.Conv2D(
                filters=best_hyperparameters['conv2_filters'],
                kernel_size=(3, 3),
                activation='relu',
                padding='same'
            )(x)
        
        # Physics layer
        time_steps = tf.range(30, dtype=tf.float32)
        physics_layer = PhysicsLayer()
        x = physics_layer([x, time_steps])
        
        # Reshape for ConvLSTM2D
        x = tf.keras.layers.Reshape((30, 2, 5, -1))(x)
        
        # First ConvLSTM layer
        use_second_lstm = best_hyperparameters.get('use_second_lstm', False)
        x = tf.keras.layers.ConvLSTM2D(
            filters=best_hyperparameters['lstm_filters'],
            kernel_size=(best_hyperparameters['lstm_kernel_size'], 
                        best_hyperparameters['lstm_kernel_size']),
            padding='same',
            return_sequences=use_second_lstm,
            activation='relu',
            recurrent_dropout=best_hyperparameters.get('recurrent_dropout', 0)
        )(x)
        
        if use_second_lstm:
            x = tf.keras.layers.ConvLSTM2D(
                filters=best_hyperparameters['lstm_filters']//2,
                kernel_size=(2, 2),
                padding='same',
                return_sequences=False,
                activation='relu'
            )(x)
        
        # Spatial attention
        if best_hyperparameters.get('use_spatial_attention', False):
            attention = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(x)
            x = tf.keras.layers.Multiply()([x, attention])
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(best_hyperparameters['dense_units'], activation='relu')(x)
        
        if best_hyperparameters.get('use_dropout', False):
            x = tf.keras.layers.Dropout(best_hyperparameters['dropout_rate'])(x)
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=best_hyperparameters['learning_rate']
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        return model

keras.utils.plot_model(best_model, to_file='best_model.png', show_shapes=True, dpi=300, show_trainable=True, 
                       show_layer_activations=True, rankdir=True, show_dtype=True, expand_nested=True, 
                       show_layer_names=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#best_model.save(f'final_model_{timestamp}.h5')
best_model.save(f'final_model_{timestamp}.keras')

def build_final_model(best_hyperparameters, input_shape=(30, 10)):
    """Build the final model with the best hyperparameters"""
    with tf.device('/CPU:0'):
        inputs = tf.keras.Input(shape=input_shape)
        
        # Encoder
        x = tf.keras.layers.Reshape((30, 10, 1))(inputs)
        
        # First Conv2D layer
        x = tf.keras.layers.Conv2D(
            filters=best_hyperparameters['conv1_filters'],
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            dilation_rate=best_hyperparameters.get('dilation_rate', 1)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Optional second Conv2D
        if best_hyperparameters['use_second_conv', False]:
            x = tf.keras.layers.Conv2D(
                filters=best_hyperparameters['conv2_filters'],
                kernel_size=(3, 3),
                activation='relu',
                padding='same'
            )(x)
        
        # Physics layer
        time_steps = tf.range(30, dtype=tf.float32)
        physics_layer = PhysicsLayer()
        x = physics_layer([x, time_steps])
        
        # Reshape for ConvLSTM2D
        x = tf.keras.layers.Reshape((30, 2, 5, -1))(x)
        
        # ConvLSTM2D
        x = tf.keras.layers.ConvLSTM2D(
            filters=best_hyperparameters['lstm_filters'],
            kernel_size=best_hyperparameters.get('lstm_kernel', (2,2)),
            padding='same',
            return_sequences=best_hyperparameters.get('use_second_lstm', False),
            activation='relu',
            recurrent_dropout=best_hyperparameters.get('recurrent_dropout', 0)
        )(x)

        # Optional second ConvLSTM
        if best_hyperparameters.get('use_second_lstm', False):
            x = tf.keras.layers.ConvLSTM2D(
                filters=best_hyperparameters['lstm_filters']//2,
                kernel_size=(2,2),
                padding='same',
                return_sequences=False,
                activation='relu'
            )(x)

        # Spatial attention if used
        if best_hyperparameters.get('use_spatial_attention', False):
            attention = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(x)
            x = tf.keras.layers.Multiply()([x, attention])
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(best_hyperparameters['dense_units'], activation='relu')(x)
        
        if best_hyperparameters['use_dropout', False]:
            x = tf.keras.layers.Dropout(best_hyperparameters['dropout_rate'])(x)
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with best learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=best_hyperparameters['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        return model

def train_final_model(model, x_train, y_train, x_val, y_val, spatial_info, batch_size=32, epochs=100):
    """Train the final model with the best hyperparameters"""
    with tf.device('/CPU:0'):
        # Initialize metrics
        train_acc_metric = tf.keras.metrics.BinaryAccuracy()
        train_precision = tf.keras.metrics.Precision()
        train_recall = tf.keras.metrics.Recall()
        train_auc = tf.keras.metrics.AUC()
        
        n_samples = len(x_train)
        best_val_auc = 0
        best_weights = None
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'auc': [],
            'spatial_metrics': []  # Track metrics by spatial region
        }
        
        print("Starting final model training...")
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            val_loss = 0
            
            # Reset metrics
            train_acc_metric.reset_state()
            train_precision.reset_state()
            train_recall.reset_state()
            train_auc.reset_state()
            
            # Training batches - NO SHUFFLING
            n_batches = int(np.ceil(n_samples / batch_size))
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                # Training step
                with tf.GradientTape() as tape:
                    predictions = model(x_batch, training=True)
                    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
                    loss = loss_fn(y_batch, predictions)
                    epoch_loss += tf.reduce_mean(loss)
                
                # Gradient update
                grads = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                
                #epoch_loss += loss.numpy()
                
                # Update metrics
                train_acc_metric.update_state(y_batch, predictions)
                train_precision.update_state(y_batch, predictions)
                train_recall.update_state(y_batch, predictions)
                train_auc.update_state(y_batch, predictions)
            
            # Validation
            # n_val_batches = int(np.ceil(len(x_val) / batch_size))
            # for batch_idx in range(n_val_batches):
            #     start_idx = batch_idx * batch_size
            #     end_idx = min(start_idx + batch_size, len(x_val))
                
            #     x_batch = x_val[start_idx:end_idx]
            #     y_batch = y_val[start_idx:end_idx]
                
            #     val_predictions = model(x_batch, training=False)
            #     val_loss += loss_fn(y_batch, val_predictions).numpy()

            val_predictions = model.predict(x_val)
            val_loss = tf.keras.losses.binary_crossentropy(y_val, val_predictions)
            val_auc = tf.keras.metrics.AUC()(y_val, val_predictions)
            
            # Calculate metrics
            epoch_loss = epoch_loss / n_batches
            val_loss = val_loss / n_val_batches
            epoch_acc = train_acc_metric.result().numpy()
            epoch_prec = train_precision.result().numpy()
            epoch_recall = train_recall.result().numpy()
            epoch_auc = train_auc.result().numpy()
            
            # Store metrics
            history['train_loss'].append(float(epoch_loss))
            history['val_loss'].append(float(val_loss))
            history['accuracy'].append(float(epoch_acc))
            history['precision'].append(float(epoch_prec))
            history['recall'].append(float(epoch_recall))
            history['auc'].append(float(epoch_auc))
            
            if epoch_auc > best_val_auc:
                best_val_auc = epoch_auc
                best_weights = model.get_weights()
                print("New best model saved!")
            
            print(f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Accuracy: {epoch_acc:.4f}")
            print(f"Precision: {epoch_prec:.4f}")
            print(f"Recall: {epoch_recall:.4f}")
            print(f"AUC: {epoch_auc:.4f}")
        
        # Restore best weights
        if best_weights is not None:
            model.set_weights(best_weights)
        
        # Save model and history
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model.save(f'final_model_{timestamp}.h5')
        
        with open(f'final_training_history_{timestamp}.json', 'w') as f:
            json.dump(history, f, indent=4)
        
        return model, history

# Final prediction function
def make_predictions(model, x_test):
    """Make predictions using the final model"""
    with tf.device('/CPU:0'):
        predictions = model.predict(x_test)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        np.save(f'predictions_{timestamp}.npy', predictions)
        
        return predictions

# Usage:
# 1. Build final model with best hyperparameters
final_model = build_final_model(best_hyperparameters)

# 2. Train final model
trained_model, training_history = train_final_model(
    final_model, x_train, y_train, x_val, y_val, 
    epochs=100  # More epochs for final training
)

# 3. Make predictions
final_predictions = make_predictions(trained_model, x_test)

