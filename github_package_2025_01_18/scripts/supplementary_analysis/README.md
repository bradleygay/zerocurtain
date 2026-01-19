# Supplementary Table Computation Scripts

Scripts for computing Tables S2.4, S2.6, S3.5, and S3.6 for the manuscript
"Resolving Circumarctic Zero-Curtain Phenomena with AI-Integrated Earth Observations"

## Directory Structure
supplementary_analysis/
├── ablation/
│   └── run_ablation_study.py      # Table S2.6: Ablation study
├── sensitivity/
│   ├── compute_sensitivity.py     # Table S2.4: Sensitivity analysis
│   └── complete_threshold.py      # Helper for threshold completion
└── trend_analysis/
├── compute_mann_kendall.py    # Tables S3.5, S3.6: Mann-Kendall & Power
└── compute_pirszc_duration.py # PIRSZC duration fix (mean not median)
## Scripts Overview

| Script | Table | Description | Job ID |
|--------|-------|-------------|--------|
| ablation/run_ablation_study.py | S2.6 | 6-configuration ablation study | 53173025 |
| sensitivity/compute_sensitivity.py | S2.4 | 5-threshold sensitivity analysis | 53195573 |
| sensitivity/complete_threshold.py | S2.4 | 8.0% threshold completion | 53289999 |
| trend_analysis/compute_mann_kendall.py | S3.5, S3.6 | Mann-Kendall trends & power | 53200119 |
| trend_analysis/compute_pirszc_duration.py | S3.5 | PIRSZC duration fix | — |

## Requirements

### Software
- Python 3.12 (module: python/GEOSpyD/24.11.3-0/3.12 on NCCS Discover)
- PyTorch with CUDA (for ablation study)
- NumPy, Pandas, SciPy, Dask, scikit-learn

### Hardware Requirements
| Script | GPU | RAM | Time |
|--------|-----|-----|------|
| Ablation | A100 (40GB) | 180GB | ~8.6 hrs |
| Sensitivity | — | 180GB | ~24 hrs |
| Mann-Kendall | — | 64GB | <1 min |

## Usage on NCCS Discover
```bash
# Load environment
module purge
module load python/GEOSpyD/24.11.3-0/3.12

# Submit ablation study (requires gpu_a100 partition)
sbatch slurm/supplementary/run_ablation.sh

# Submit sensitivity analysis (24hr QOS)
sbatch slurm/supplementary/run_sensitivity.sh

# Submit Mann-Kendall (fast, can run interactively)
sbatch slurm/supplementary/run_mann_kendall.sh
```

## Results Summary

### Table S2.4: Sensitivity Analysis
| Threshold | Events | Bootstrap CV | Moran's I | CV RMSE |
|-----------|--------|--------------|-----------|---------|
| 4.0% | 52,270,267 | 0.000064 | 0.2799 | 1085.53 |
| 5.0% | 51,703,501 | 0.000073 | 0.1960 | 980.98 |
| **5.95%** | **51,189,851** | **0.000071** | **0.1635** | **884.84** |
| 7.0% | 50,628,286 | 0.000073 | 0.2026 | 776.71 |
| 8.0% | 50,076,951 | 0.000067 | 0.1804 | 665.44 |

### Table S2.6: Ablation Study
| Configuration | Accuracy | Physics Compliance |
|--------------|----------|-------------------|
| Full GeoCryoAI | 93.4% | 94.0% |
| − Physics | 95.0% | 87.1% |
| − LNN | 94.0% | 91.9% |
| Baseline MLP | 81.6% | 89.0% |

### Table S3.5: Mann-Kendall Trends
| Dataset | Variable | τ | p-value | Sen's Slope |
|---------|----------|---|---------|-------------|
| PINSZC | Extent | +0.524 | <0.0001 | +0.017 m/dec |
| PINSZC | Duration | −0.430 | <0.0001 | −2.35 hrs/dec |
| PIRSZC | Extent | +0.309 | 0.218 | Not significant |

## Citation

Gay, B.A. et al. (2025). Resolving Circumarctic Zero-Curtain Phenomena 
with AI-Integrated Earth Observations. Scientific Reports. Under Review.
