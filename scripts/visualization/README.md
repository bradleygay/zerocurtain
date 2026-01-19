# Visualization Scripts

## Figure S2.3: Uncertainty Framework Visualization

Generates the uncertainty quantification framework figure (Figure S2.3) for the manuscript.

### Script
- `figure_s2_3_uncertainty_framework.py` - Main visualization script implementing PCHIP spline interpolation with QuantileNorm for Arctic uncertainty data

### SLURM Submission
```bash
cd /discover/nobackup/$USER/zerocurtain
sbatch slurm/visualization/run_fig_s2_3.sh
```

### Output
- `Figure_S2.3_Uncertainty_Framework.pdf` - Publication-ready PDF
- `Figure_S2.3_Uncertainty_Framework.png` - PNG version
- `Fig_S2.3_b1_intensity_uncertainty.png` - Panel b1 (intensity)
- `Fig_S2.3_b2_duration_uncertainty.png` - Panel b2 (duration)
- `Fig_S2.3_b3_extent_uncertainty.png` - Panel b3 (spatial extent)

### Computational Requirements
- Python 3.12 (GEOSpyD/24.11.3-0/3.12)
- Memory: 64GB RAM
- Time: ~45 minutes

### Verified Execution
- Job ID: 53466245
- Status: Complete (January 14, 2026)
