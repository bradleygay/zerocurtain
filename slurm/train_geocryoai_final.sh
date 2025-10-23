#!/bin/bash
#SBATCH --job-name=geocryoai_final
#SBATCH --output=logs/geocryoai_final_%j.log
#SBATCH --error=logs/geocryoai_final_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=480G
#SBATCH --qos=alla100
#SBATCH --account=j1101

cd /discover/nobackup/bagay/part2_discover_package

echo "================================================================"
echo "GeoCryoAI Training - STABLE MODE"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

module purge
module load python/GEOSpyD/24.11.3-0/3.12

# AGGRESSIVE memory management
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export TORCH_CUDA_ARCH_LIST="8.0"
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

python3 -u src/part2_geocryoai/zero_curtain_ml_model.py \
    --config config/part2_config_discover.yaml \
    --output outputs/part2_geocryoai \
    --device cuda

echo ""
echo "================================================================"
echo "Training complete: $(date)"
echo "================================================================"
