# Arctic Zero-Curtain Detection Pipeline
**A physics-informed machine learning framework for detecting and analyzing zero-curtain events in permafrost regions**

[![DOI](https://zenodo.org/badge/1069438493.svg)](https://doi.org/10.5281/zenodo.17407929)

---

## Overview
Zero-curtain events—periods when soil temperature remains near 0°C during freeze-thaw transitions—are critical indicators of permafrost dynamics and climate change impacts in Arctic ecosystems. This pipeline combines physics-based modeling with advanced deep learning to detect, characterize, and predict these events across spatiotemporal scales.
### Key Features
- **Physics-Informed Neural Networks (PINSZC)**: Integrates thermodynamic constraints from CryoGrid and LPJ-EOSIM models
- **GeoCryoAI Spatiotemporal Graphs**: Captures geographic connectivity and temporal dependencies between events
- **Teacher Forcing with Ground Truth**: Leverages high-fidelity in-situ observations for curriculum learning
- **Multi-Scale Analysis**: From point measurements to regional satellite observations
- **Explainable AI**: SHAP and LIME interpretability for physical validation

---

## Scientific Motivation
Arctic warming at twice the global rate drives rapid permafrost degradation, altering:
- Carbon cycling (release of stored organic carbon)
- Ecosystem structure (vegetation shifts, thermokarst formation)
- Hydrological regimes (active layer deepening, drainage changes)
- Infrastructure stability (thaw settlement, ground ice loss)
Zero-curtain duration and intensity reflect the balance between latent heat exchange during phase change and environmental energy fluxes, making them sensitive proxies for permafrost vulnerability.

---

## Pipeline Architecture
The framework operates in three sequential stages:

```
┌─────────────────────────────────────────────────────────────────┐
│  PART I: Physics-Informed Neural Zero-Curtain Detection (PINSZC) │
│  • Thermodynamic constraint integration                          │
│  • Multi-sensor data fusion (in-situ + reanalysis)               │
│  • Event classification: rapid, consecutive, extended, composite │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  PART II: GeoCryoAI Teacher Forcing Model                        │
│  • Spatiotemporal graph neural networks                          │
│  • Liquid neural networks for ecological memory                  │
│  • Curriculum learning with ground truth                         │
│  • Multi-task prediction: intensity, duration, spatial extent    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  PART III: Physics-Informed Remote Sensing Zero-Curtain (PIRSZC) │
│  • Satellite data integration (Landsat, SMAP, UAVSAR, NISAR)     │
│  • Upscaling from site to regional predictions                   │
│  • Uncertainty quantification                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start
### Prerequisites
- Python 3.9 or higher
- CUDA 11.8+ (for GPU acceleration)
- 32GB+ RAM recommended
- 200GB+ storage for full dataset
### Installation
```bash
# Clone repository
git clone https://github.com/username/arctic-zero-curtain-pipeline.git
cd arctic-zero-curtain-pipeline
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
# Configure Google Drive integration for large files
python setup_gdrive_placeholders.py
```
### Large File Management
Due to GitHub's size constraints, datasets exceeding 100MB are managed via Google Drive placeholders:
```python
# Automatic download on first use
from src.utils.data_access import get_data_file
# Load dataset (downloads if not cached locally)
df = pd.read_parquet(get_data_file('data/auxiliary/arcticdem/arcticdem.parquet'))
```
**Optional**: Pre-download all files to local cache:
```bash
python predownload_all_data.py
```

---

## Usage
### Part I: Zero-Curtain Event Detection
```bash
# Run PINSZC detection on in-situ data
python scripts/run_part1_pinszc.py --config config/part1_config.yaml
# Output: outputs/part1_pinszc/physics_informed_events.parquet
```
### Part II: GeoCryoAI Training
```bash
# Train spatiotemporal prediction model
python scripts/run_part2_geocryoai.py --config config/part2_config.yaml
# For HPC environments (SLURM)
sbatch slurm/run_geocryoai_training.sh
```
### Part III: Remote Sensing Integration
```bash
# Scale predictions to satellite observations
python scripts/run_part3_pirszc.py --config config/part3_config.yaml
```

---

## Configuration
All pipeline parameters are defined in YAML configuration files:
```yaml
# config/part2_config.yaml
data:
  parquet_file: outputs/part1_pinszc/physics_informed_events.parquet
  batch_size: 512
  sequence_length: 90
  temporal_coverage: seasonal
model:
  d_model: 256
  n_heads: 8
  n_layers: 6
  geocryoai:
    enabled: true
    spatial_threshold_km: 50.0
teacher_forcing:
  initial_ratio: 0.9
  curriculum_schedule: exponential
  decay_rate: 0.95
training:
  epochs: 25
  learning_rate: 0.0002
  use_amp: true
```

---

## Data Sources
The pipeline integrates diverse geophysical datasets:

| Dataset | Coverage | Resolution | Purpose |
|---------|----------|------------|---------|
| **In-Situ Sensors** | Point measurements | Hourly | Ground truth for model training |
| **ERA5 Reanalysis** | Global | 0.25° / hourly | Meteorological forcing |
| **ArcticDEM** | Pan-Arctic | 2m | Topographic context |
| **SMAP** | Global | 9km / 3-day | Soil moisture |
| **Landsat 8/9** | Global | 30m / 16-day | Surface temperature, NDVI |
| **UAVSAR** | Alaska | 5m / seasonal | Ground displacement |
| **Permafrost Zones** | Circumpolar | 1km | Permafrost probability |

---

## Model Architecture
### Part II: Hybrid Neural Architecture
The GeoCryoAI model combines multiple neural paradigms:
**Transformer Backbone**: Multi-scale attention for spatiotemporal dependencies
**Liquid Neural Networks**: Continuous-time dynamics for ecological memory, capturing long-term thermal inertia
**Graph Neural Networks**: 
- Spatial GCN: Geographic connectivity (50km threshold)
- Temporal GAT: Event sequence relationships
**Physics-Informed Loss**:
```
L_total = L_MSE + α_physics·L_physics + α_temporal·L_temporal + α_pattern·L_pattern
```
Where:
- `L_physics`: Stefan problem energy conservation
- `L_temporal`: Smoothness constraints
- `L_pattern`: Event type classification

---

## Results
Performance metrics on held-out test sites (2020-2023):

| Target Variable | R² Score | RMSE | MAE |
|----------------|----------|------|-----|
| **Event Intensity** | 0.847 | 0.089 | 0.065 |
| **Duration (hours)** | 0.792 | 18.3 | 12.7 |
| **Spatial Extent (m)** | 0.756 | 0.43 | 0.31 |

**Key Findings**:
- Teacher forcing improves generalization by 23% over baseline
- GeoCryoAI spatial graphs reduce prediction error in data-sparse regions
- Physics constraints prevent non-physical predictions (e.g., negative durations)

---

## Reproducibility
All experiments are fully reproducible:
```bash
# Exact package versions
pip install -r requirements.lock
# Set random seeds
export PYTHONHASHSEED=42
# Run with deterministic algorithms
python scripts/run_part2_geocryoai.py --config config/part2_config.yaml \
  --seed 42 --deterministic
```

---

## Roadmap
### Version 1.0 (Current)
- ✅ PINSZC event detection
- ✅ GeoCryoAI teacher forcing model
- ✅ Remote sensing integration
### Version 1.1 (Planned)
- 🔄 Real-time event monitoring dashboard
- 🔄 Cloud-based inference API
- 🔄 Extended temporal coverage (1980-present)
### Version 2.0 (Future)
- 📋 Multi-model ensemble predictions
- 📋 Causal inference for feedback mechanisms
- 📋 Integration with Earth System Models (ESMs)

---

## Contributing
This is research code. For questions or collaboration inquiries, please open an issue and/or contact the PI ([RESEARCHER]) at: [RESEARCHER_EMAIL].

## License
This project is licensed under the AGPL-3.0 GNU AFFERO GENERAL PUBLIC LICENSE - see LICENSE file for details.

## Citation
If you use any of this code in your research, please cite:
[RESEARCHER]., Miner, K., Rietze, N., Pastick, N., Poulter, B., & Miller, C. (2025). zerocurtain (Version 1.0.0) [Computer software]. https://doi.org/Nature Machine Intelligence (TBD)
```bibtex
@software{Gay_zerocurtain_2025,
author = {[RESEARCHER] and Miner, Kimberley and Rietze, Nils and Pastick, Neal and Poulter, Ben and Miller, Charles},
doi = {Nature Machine Intelligence (TBD)},
month = oct,
title = {{zerocurtain}},
url = {https://github.com/bradleygay/zerocurtain},
version = {1.0.0},
year = {2025}
}
```

## Acknowledgments
Acknowledgment and gratitude are owed to the [REDACTED_AFFILIATION] Postdoctoral Program and proposal reviewers whose contributions and
feedback elevated this research. Much appreciation is owed and extended to the following institutions, networks, teams, and
individuals responsible for the viability, momentum, and success of this work: National Aeronautics and Space Administration
([REDACTED_AFFILIATION]), [REDACTED_AFFILIATION] ([REDACTED_AFFILIATION]), Goddard Space Flight Center (GSFC), [REDACTED_AFFILIATION] Arctic Boreal Vulnerability Experiment
(ABoVE) field and science support services, National Science Foundation (NSF), Woodwell Climate Research Center, University of
Alaska ‐ Fairbanks, Institute of Arctic Biology (IAB), California Institute of Technology, [REDACTED_AFFILIATION] UAVSAR airborne, algorithm, and
inSAR/polSAR processing teams, [REDACTED_AFFILIATION] ABoVE Airborne Working Group, CALM, GTNP, AmeriFlux, National Ecological
Observatory Network (NEON), Toolik Field Station, Peter Griffith (GSFC), Elizabeth Hoy (GSFC), Naiara Pinto ([REDACTED_AFFILIATION]), Yang Zheng
([REDACTED_AFFILIATION]), Nikolay Shiklomanov (GWU), Julia Boike (UCI), Pavel Groisman (NOAA), and Oliver Frauenfeld (TAMU). BG’s
participation and research was sponsored by the National Aeronautics and Space Administration ([REDACTED_AFFILIATION]) and supported by an
appointment to the [REDACTED_AFFILIATION] Postdoctoral Program (NPP) at the [REDACTED_AFFILIATION] ([REDACTED_AFFILIATION]) and the California Institute of
Technology, administered by Oak Ridge Associated Universities (ORAU) under contract with [REDACTED_AFFILIATION] (80NM0018D0004). The views
and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies,
either expressed or implied, of [REDACTED_AFFILIATION], [REDACTED_AFFILIATION], the California Institute of Technology, or the U.S. Government. The U.S. Government is
authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein. Any use of
trade, firm, or product names is for descriptive purposes only and does not imply endorsement by the U.S. Government.

## Further Acknowledgments
- LPJ-EOSIM permafrost physics model
- CryoGrid community model
- University of Oslo Permafrost CCI datasets
- ERA5-Land snow data (Copernicus Climate Data Store)
- Circumarctic in situ measurement networks

---

## Contact
For questions about this research:
- GitHub Issues: https://github.com/bradleygay/zerocurtain/issues
- Email: bradley.a.gay@nasa.gov
For questions or collaboration inquiries, please open a GitHub issue or contact via institutional email.
