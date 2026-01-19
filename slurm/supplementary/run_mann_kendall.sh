#!/bin/bash
#SBATCH --job-name=mk_power
#SBATCH --output=/discover/nobackup/bagay/mann_kendall_power_%j.out
#SBATCH --error=/discover/nobackup/bagay/mann_kendall_power_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --account=j1101

module purge
module load python/GEOSpyD/24.11.3-0/3.12

cd /discover/nobackup/bagay

echo "============================================================"
echo "MANN-KENDALL & POWER ANALYSIS - INDEPENDENT RUN"
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_NODELIST}"
echo "Start: $(date)"
echo "============================================================"

python compute_mann_kendall_power.py \
    --pinszc /discover/nobackup/bagay/gdrive_sync/arctic_zero_curtain_pipeline/outputs/part1_pinszc/consolidated_datasets/physics_informed_zero_curtain_events_COMPLETE.parquet \
    --pirszc /discover/nobackup/bagay/gdrive_sync/arctic_zero_curtain_pipeline/outputs/part3_pirszc/remote_sensing_physics_informed_comprehensive.parquet \
    --output /discover/nobackup/bagay/supplementary_tables_results

echo "End: $(date)"
