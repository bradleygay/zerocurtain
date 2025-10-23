#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --job-name=geocryoai_part2
#SBATCH --output=logs/geocryoai_part2_%j.log
#SBATCH --error=logs/geocryoai_part2_%j.err
#SBATCH --mem=120G
#SBATCH --qos=alla100
#SBATCH --account=j1101

echo "=========================================="
echo "Part II GeoCryoAI Training on NCCS Discover"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Environment setup
module purge
module load python/GEOSpyD/24.11.3-0/3.12

# CUDA configuration for dual A100s
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048
export TORCH_CUDA_ARCH_LIST="8.0"

# Filesystem optimization
export TMPDIR=/discover/nobackup/bagay/tmp
mkdir -p $TMPDIR
export PYTHONUNBUFFERED=1

# PyTorch optimization
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

# Enable cuDNN benchmarking for A100s
export CUDNN_BENCHMARK=1

# Working directory
WORK_DIR="/discover/nobackup/bagay/part2_discover_package"
cd $WORK_DIR || exit 1

echo "Working directory: $(pwd)"

# Verify GPU availability
echo ""
echo "GPU Configuration:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
"

# Verify dataset
echo ""
echo "Verifying dataset..."
if [ -f "outputs/part1_pinszc/zero_curtain_enhanced_cryogrid_physics_dataset.parquet" ]; then
    echo "  Dataset found: $(du -h outputs/part1_pinszc/zero_curtain_enhanced_cryogrid_physics_dataset.parquet | cut -f1)"
else
    echo "  ERROR: Dataset not found!"
    exit 1
fi

# Create output directories
mkdir -p outputs/part2_geocryoai/models
mkdir -p logs

# Run training
echo ""
echo "Starting GeoCryoAI training..."
echo "Configuration: config/part2_config_discover.yaml"
echo "=========================================="
echo ""

# Add project root to PYTHONPATH
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH}"

# Run directly instead of as module
python src/part2_geocryoai/zero_curtain_ml_model.py config/part2_config_discover.yaml

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""
echo "Output files:"
ls -lh outputs/part2_geocryoai/models/ 2>/dev/null || echo "  No models saved"
echo "=========================================="

exit $EXIT_CODE