#!/bin/bash
#SBATCH --job-name=abl_FULL
#SBATCH --output=/discover/nobackup/bagay/ablation_FULL_%j.out
#SBATCH --error=/discover/nobackup/bagay/ablation_FULL_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --constraint=rome
#SBATCH --qos=alla100
#SBATCH --account=j1101

module purge
module load python/GEOSpyD/24.11.3-0/3.12

cd /discover/nobackup/bagay
mkdir -p /discover/nobackup/bagay/ablation_results

echo "============================================================"
echo "FULL-DATASET ABLATION STUDY - NO SAMPLING"
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_NODELIST}"
echo "Start: $(date)"
echo "============================================================"
nvidia-smi

python run_ablation_study_FULL.py \
    --pinszc /discover/nobackup/bagay/gdrive_sync/arctic_zero_curtain_pipeline/outputs/part1_pinszc/consolidated_datasets/physics_informed_zero_curtain_events_COMPLETE.parquet \
    --output /discover/nobackup/bagay/ablation_results

echo "End: $(date)"
