#!/bin/bash
#SBATCH --job-name=supp_24h
#SBATCH --output=/discover/nobackup/bagay/supplementary_24h_%j.out
#SBATCH --error=/discover/nobackup/bagay/supplementary_24h_%j.err
#SBATCH --time=24:00:00
#SBATCH --qos=long
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --partition=compute
#SBATCH --account=j1101

module purge
module load python/GEOSpyD/24.11.3-0/3.12

cd /discover/nobackup/bagay

echo "============================================================"
echo "SUPPLEMENTARY TABLES - 24 HOUR QOS - FULL DATASET"
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_NODELIST}"
echo "Start: $(date)"
echo "============================================================"

python compute_supplementary_RESUME.py \
    --pinszc /discover/nobackup/bagay/gdrive_sync/arctic_zero_curtain_pipeline/outputs/part1_pinszc/consolidated_datasets/physics_informed_zero_curtain_events_COMPLETE.parquet \
    --pirszc /discover/nobackup/bagay/gdrive_sync/arctic_zero_curtain_pipeline/outputs/part3_pirszc/remote_sensing_physics_informed_comprehensive.parquet \
    --output /discover/nobackup/bagay/supplementary_tables_results

echo "End: $(date)"
