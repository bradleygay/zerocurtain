#!/bin/bash
#SBATCH --job-name=rmse_8pct
#SBATCH --output=/discover/nobackup/bagay/rmse_8pct_%j.out
#SBATCH --error=/discover/nobackup/bagay/rmse_8pct_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute
#SBATCH --account=j1101

module purge
module load python/GEOSpyD/24.11.3-0/3.12

cd /discover/nobackup/bagay

echo "Start: $(date)"
python3 /discover/nobackup/bagay/complete_8pct_rmse.py
echo "End: $(date)"
