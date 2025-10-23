#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=part5_mapping_viz
#SBATCH --output=part5_mapping_viz_%j.log
#SBATCH --error=part5_mapping_viz_%j.err
#SBATCH --mem=192G
#SBATCH --qos=alla100
#SBATCH --account=j1101

# ============================================================================
# PART V: CIRCUMARCTIC MAPPING AND VISUALIZATION
# ============================================================================
# Generates comprehensive geospatial analysis from Part IV predictions
# 
# [RESEARCHER] Gay
# [RESEARCHER] Sciences Laboratory
# ============================================================================

echo "================================================================"
echo "PART V: CIRCUMARCTIC MAPPING AND VISUALIZATION PIPELINE"
echo "================================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

# Environment setup
module load python/GEOSpyD/24.11.3-0/3.12
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048

# Matplotlib configuration for SLURM
export MPLBACKEND=Agg
export MPLCONFIGDIR=/discover/nobackup/bagay/.matplotlib
mkdir -p $MPLCONFIGDIR

# Working directory setup
export TMPDIR=/discover/nobackup/bagay/tmp
mkdir -p $TMPDIR
export PYTHONUNBUFFERED=1

# System optimization
ulimit -n 65536
export OMP_NUM_THREADS=12

# Base directory configuration
BASE_DIR="${HOME}/arctic_zero_curtain_pipeline"
SCRIPT_DIR="${BASE_DIR}/scripts"

echo "Configuration:"
echo "  Base directory: ${BASE_DIR}"
echo "  Script directory: ${SCRIPT_DIR}"
echo "  Available CPUs: ${SLURM_CPUS_PER_TASK}"
echo "  Allocated memory: ${SLURM_MEM_PER_NODE} MB"
echo ""

# Change to script directory
cd ${SCRIPT_DIR}

# Verify Part IV outputs exist
PREDICTIONS_FILE="${BASE_DIR}/outputs/part4_transfer_learning/predictions/circumarctic_zero_curtain_predictions_complete.parquet"

echo "================================================================"
echo "VALIDATING PART IV OUTPUTS"
echo "================================================================"

if [ ! -f "${PREDICTIONS_FILE}" ]; then
    echo " Part IV predictions not found: ${PREDICTIONS_FILE}"
    echo ""
    echo "Please run Part IV first:"
    echo "  sbatch run_part4_transfer_learning.sh"
    echo ""
    exit 1
else
    FILE_SIZE=$(du -h "${PREDICTIONS_FILE}" | cut -f1)
    echo " Predictions file found: ${PREDICTIONS_FILE}"
    echo "  Size: ${FILE_SIZE}"
    echo ""
fi

# Verify Natural Earth data
NE_DIR="${HOME}/natural_earth_data"
echo "================================================================"
echo "CHECKING NATURAL EARTH DATA"
echo "================================================================"

if [ -d "${NE_DIR}" ]; then
    echo " Natural Earth directory found: ${NE_DIR}"
    
    if [ -f "${NE_DIR}/ne_10m_land/ne_10m_land.shp" ]; then
        echo "   Land polygons available"
    else
        echo "   Land polygons not found (will download automatically)"
    fi
    
    if [ -f "${NE_DIR}/ne_10m_lakes/ne_10m_lakes.shp" ]; then
        echo "   Lake polygons available"
    else
        echo "   Lake polygons not found (will download automatically)"
    fi
else
    echo " Natural Earth directory not found"
    echo "  Creating: ${NE_DIR}"
    mkdir -p "${NE_DIR}"
    echo "  Data will be downloaded automatically during mapping"
fi
echo ""

# Create output directories
echo "================================================================"
echo "PREPARING OUTPUT DIRECTORIES"
echo "================================================================"

OUTPUT_BASE="${BASE_DIR}/outputs/part5_mapping_visualization"
mkdir -p "${OUTPUT_BASE}/maps/annual"
mkdir -p "${OUTPUT_BASE}/maps/seasonal"
mkdir -p "${OUTPUT_BASE}/maps/monthly"
mkdir -p "${OUTPUT_BASE}/time_series"
mkdir -p "${OUTPUT_BASE}/regional_analysis"
mkdir -p "${OUTPUT_BASE}/statistics"
mkdir -p "${OUTPUT_BASE}/explainability"
mkdir -p "${OUTPUT_BASE}/logs"

echo " Output directories created at: ${OUTPUT_BASE}"
echo ""

# Memory monitoring function
monitor_memory() {
    USED_MEM=$(free -g | awk '/^Mem:/{print $3}')
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    echo "  Memory: ${USED_MEM}GB / ${TOTAL_MEM}GB used"
}

# Execute Part V pipeline
echo "================================================================"
echo "EXECUTING PART V MAPPING PIPELINE"
echo "================================================================"
echo ""

monitor_memory

python3 ${SCRIPT_DIR}/part5_mapping_visualization.py \
    --base-dir ${BASE_DIR} \
    2>&1 | tee "${OUTPUT_BASE}/logs/part5_execution_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================"
echo "PIPELINE EXECUTION COMPLETED"
echo "================================================================"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo " Part V completed successfully"
    echo ""
    
    # Summary statistics
    echo "Generated outputs:"
    
    ANNUAL_MAPS=$(find "${OUTPUT_BASE}/maps/annual" -name "*.png" 2>/dev/null | wc -l)
    echo "  Annual maps: ${ANNUAL_MAPS}"
    
    SEASONAL_MAPS=$(find "${OUTPUT_BASE}/maps/seasonal" -name "*.png" 2>/dev/null | wc -l)
    echo "  Seasonal maps: ${SEASONAL_MAPS}"
    
    MONTHLY_MAPS=$(find "${OUTPUT_BASE}/maps/monthly" -name "*.png" 2>/dev/null | wc -l)
    echo "  Monthly maps: ${MONTHLY_MAPS}"
    
    TIME_SERIES=$(find "${OUTPUT_BASE}/time_series" -name "*.png" 2>/dev/null | wc -l)
    echo "  Time series: ${TIME_SERIES}"
    
    REGIONAL=$(find "${OUTPUT_BASE}/regional_analysis" -name "*.png" 2>/dev/null | wc -l)
    echo "  Regional analyses: ${REGIONAL}"
    
    STATS=$(find "${OUTPUT_BASE}/statistics" -name "*.csv" 2>/dev/null | wc -l)
    echo "  Statistical summaries: ${STATS}"
    
    echo ""
    echo "Output location: ${OUTPUT_BASE}"
    echo ""
    
    # Next steps
    echo "Next steps:"
    echo "  1. Review maps in: ${OUTPUT_BASE}/maps/"
    echo "  2. Examine time series: ${OUTPUT_BASE}/time_series/"
    echo "  3. Check statistics: ${OUTPUT_BASE}/statistics/"
    echo ""
    
else
    echo " Part V failed with exit code: ${EXIT_CODE}"
    echo ""
    echo "Check logs:"
    echo "  Execution log: ${OUTPUT_BASE}/logs/part5_execution_${SLURM_JOB_ID}.log"
    echo "  SLURM error: part5_mapping_viz_${SLURM_JOB_ID}.err"
    echo ""
fi

monitor_memory

echo "End time: $(date)"
echo "================================================================"

exit ${EXIT_CODE}
