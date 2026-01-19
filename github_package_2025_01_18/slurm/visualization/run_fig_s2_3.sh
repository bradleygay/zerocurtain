#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=fig_s2_3_uncertainty
#SBATCH --output=fig_s2_3_uncertainty_%j.log
#SBATCH --error=fig_s2_3_uncertainty_%j.err
#SBATCH --mem=192G
#SBATCH --qos=allA100
#SBATCH --account=j1101

# ============================================================================
# FIGURE S2.3: UNCERTAINTY FRAMEWORK QUANTIFICATION AND VALIDATION
# ============================================================================
# Generates comprehensive uncertainty framework visualization for the
# GeoCryoAI Physics-Informed Zero-Curtain Detection System
#
# Manuscript: Nature Communications Earth & Environment
# Author: Dr. Bradley Gay
# Affiliation: NASA GSFC Cryospheric Sciences Laboratory / ESSIC-UMD
#
# Output:
#   - Figure_S2.3_Uncertainty_Framework.png (300 DPI)
#   - Figure_S2.3_Uncertainty_Framework.pdf (vector)
#
# Expected runtime: ~30-60 minutes
# ============================================================================

echo "================================================================"
echo "FIGURE S2.3: UNCERTAINTY FRAMEWORK GENERATION"
echo "================================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "Setting up environment..."

# Load Python module - CRITICAL: Use 3.12, NOT 3.11
module load python/GEOSpyD/24.11.3-0/3.12

# Matplotlib configuration for SLURM (non-interactive)
export MPLBACKEND=Agg
export MPLCONFIGDIR=/discover/nobackup/bagay/.matplotlib
mkdir -p $MPLCONFIGDIR

# Cartopy data directory (Natural Earth shapefiles)
export CARTOPY_DATA_DIR=/home/bagay/.local/share/cartopy

# Working directory setup
export TMPDIR=/discover/nobackup/bagay/tmp
mkdir -p $TMPDIR
export PYTHONUNBUFFERED=1

# System optimization
ulimit -n 65536
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "  Python: $(python3 --version)"
echo "  MPLBACKEND: ${MPLBACKEND}"
echo "  CARTOPY_DATA_DIR: ${CARTOPY_DATA_DIR}"
echo "  Available CPUs: ${SLURM_CPUS_PER_TASK}"
echo "  Allocated memory: ${SLURM_MEM_PER_NODE:-180G}"
echo ""

# ============================================================================
# DIRECTORY CONFIGURATION
# ============================================================================

BASE_DIR="/discover/nobackup/bagay/arctic_zero_curtain_pipeline"
SCRIPT_DIR="${BASE_DIR}/scripts"
OUTPUT_DIR="${BASE_DIR}/outputs/figures"

echo "Configuration:"
echo "  Base directory: ${BASE_DIR}"
echo "  Script directory: ${SCRIPT_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Change to script directory
cd ${SCRIPT_DIR}

# ============================================================================
# INPUT DATA VALIDATION
# ============================================================================

echo "================================================================"
echo "VALIDATING INPUT DATA"
echo "================================================================"

# PIRSZC predictions file
PIRSZC_FILE="${BASE_DIR}/outputs/part3_pirszc/remote_sensing_physics_informed_comprehensive.parquet"
PIRSZC_ALT="/discover/nobackup/bagay/gdrive_sync/arctic_zero_curtain_pipeline/outputs/part3_pirszc/remote_sensing_physics_informed_comprehensive.parquet"

if [ -f "${PIRSZC_FILE}" ]; then
    PIRSZC_SIZE=$(du -h "${PIRSZC_FILE}" | cut -f1)
    echo "✓ PIRSZC file found: ${PIRSZC_FILE}"
    echo "  Size: ${PIRSZC_SIZE}"
elif [ -f "${PIRSZC_ALT}" ]; then
    PIRSZC_SIZE=$(du -h "${PIRSZC_ALT}" | cut -f1)
    echo "✓ PIRSZC file found (alternate): ${PIRSZC_ALT}"
    echo "  Size: ${PIRSZC_SIZE}"
else
    echo "✗ ERROR: PIRSZC file not found!"
    echo "  Checked: ${PIRSZC_FILE}"
    echo "  Checked: ${PIRSZC_ALT}"
    echo ""
    echo "Please run Part III pipeline first to generate PIRSZC predictions."
    exit 1
fi
echo ""

# ArcticDEM file
ARCTICDEM_FILE="${BASE_DIR}/data/auxiliary/arcticdem/arcticdem.parquet"
ARCTICDEM_ALT="/discover/nobackup/bagay/gdrive_sync/arctic_zero_curtain_pipeline/data/auxiliary/arcticdem/arcticdem.parquet"

if [ -f "${ARCTICDEM_FILE}" ]; then
    DEM_SIZE=$(du -h "${ARCTICDEM_FILE}" | cut -f1)
    echo "✓ ArcticDEM file found: ${ARCTICDEM_FILE}"
    echo "  Size: ${DEM_SIZE}"
elif [ -f "${ARCTICDEM_ALT}" ]; then
    DEM_SIZE=$(du -h "${ARCTICDEM_ALT}" | cut -f1)
    echo "✓ ArcticDEM file found (alternate): ${ARCTICDEM_ALT}"
    echo "  Size: ${DEM_SIZE}"
else
    echo "✗ ERROR: ArcticDEM file not found!"
    echo "  Checked: ${ARCTICDEM_FILE}"
    echo "  Checked: ${ARCTICDEM_ALT}"
    exit 1
fi
echo ""

# Natural Earth data
NE_DIR="${HOME}/.local/share/cartopy"
if [ -d "${NE_DIR}" ]; then
    echo "✓ Natural Earth data directory found: ${NE_DIR}"
    
    # Check for specific shapefiles
    if [ -f "${NE_DIR}/shapefiles/natural_earth/physical/ne_50m_coastline.shp" ]; then
        echo "  ✓ Coastline shapefiles available"
    else
        echo "  ⚠ Coastline shapefiles may need to be downloaded"
    fi
else
    echo "⚠ Natural Earth directory not found"
    echo "  Creating: ${NE_DIR}"
    mkdir -p "${NE_DIR}"
    echo "  Data will be downloaded automatically during figure generation"
fi
echo ""

# ============================================================================
# MEMORY MONITORING FUNCTION
# ============================================================================

monitor_memory() {
    if command -v free &> /dev/null; then
        USED_MEM=$(free -g | awk '/^Mem:/{print $3}')
        TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
        echo "  Memory usage: ${USED_MEM}GB / ${TOTAL_MEM}GB"
    fi
}

# ============================================================================
# EXECUTE FIGURE GENERATION
# ============================================================================

echo "================================================================"
echo "EXECUTING FIGURE S2.3 GENERATION"
echo "================================================================"
echo ""

monitor_memory

# Run the Python script
python3 -u ${SCRIPT_DIR}/figure_s2_3_uncertainty_framework.py 2>&1 | tee "${OUTPUT_DIR}/fig_s2_3_execution_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================"
echo "EXECUTION COMPLETED"
echo "================================================================"

monitor_memory

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ Figure S2.3 generated successfully!"
    echo ""
    
    # Check output files
    PNG_FILE="${OUTPUT_DIR}/Figure_S2.3_Uncertainty_Framework.png"
    PDF_FILE="${OUTPUT_DIR}/Figure_S2.3_Uncertainty_Framework.pdf"
    
    if [ -f "${PNG_FILE}" ]; then
        PNG_SIZE=$(du -h "${PNG_FILE}" | cut -f1)
        echo "  PNG output: ${PNG_FILE}"
        echo "  Size: ${PNG_SIZE}"
    else
        echo "  ⚠ PNG file not found at expected location"
    fi
    
    if [ -f "${PDF_FILE}" ]; then
        PDF_SIZE=$(du -h "${PDF_FILE}" | cut -f1)
        echo "  PDF output: ${PDF_FILE}"
        echo "  Size: ${PDF_SIZE}"
    else
        echo "  ⚠ PDF file not found at expected location"
    fi
    
    echo ""
    echo "Figure contains:"
    echo "  (a) Variance decomposition bar chart"
    echo "  (b1-b3) Circumarctic uncertainty maps (NorthPolarStereo)"
    echo "  (c1-c3) Uncertainty relationship scatter plots"
    echo ""
    
else
    echo ""
    echo "✗ Figure generation FAILED with exit code: ${EXIT_CODE}"
    echo ""
    echo "Check logs for details:"
    echo "  Execution log: ${OUTPUT_DIR}/fig_s2_3_execution_${SLURM_JOB_ID}.log"
    echo "  SLURM output: fig_s2_3_uncertainty_${SLURM_JOB_ID}.log"
    echo "  SLURM error: fig_s2_3_uncertainty_${SLURM_JOB_ID}.err"
    echo ""
fi

echo "End time: $(date)"
echo "================================================================"

exit ${EXIT_CODE}
