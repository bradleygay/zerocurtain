# Arctic Zero-Curtain Analysis Pipeline

**Research pipeline for constructing teacher forcing datasets from Arctic in situ measurements, remote sensing data, and physics-based models for zero-curtain detection and analysis.**

## Overview

This pipeline integrates multi-source Arctic datasets to create high-quality training data for machine learning models focused on zero-curtain phenomenon detection and prediction.

### Data Sources

- **In Situ Measurements**: Ground-based soil temperature and moisture sensors
- **Remote Sensing**: NISAR, UAVSAR, and SMAP satellite observations  
- **Physics Models**: Thermodynamic and phase-change detection algorithms

## Project Structure
```
arctic_zero_curtain_pipeline/
├── config/              # Configuration files
│   ├── paths.py        # Data paths (template - update for your system)
│   └── parameters.py   # Pipeline parameters
├── src/                # Source code modules
│   ├── common/         # Shared utilities
│   ├── data_ingestion/ # Data loading and inspection
│   ├── transformations/# Data transformations
│   ├── processing/     # Core processing logic
│   ├── visualization/  # Plotting and visualization
│   └── modeling/       # ML model utilities
├── orchestration/      # Pipeline orchestration
├── scripts/            # Executable scripts
├── tests/              # Unit tests
└── outputs/            # Generated outputs (not in git)
```

## Installation

### Prerequisites

- Python 3.10+
- conda or mamba (recommended)
- 16GB+ RAM recommended
- External storage for large datasets

### Setup
```bash
# Clone repository
git clone https://github.com/bradleygay/zerocurtain.git
cd zerocurtain

# Create conda environment
conda env create -f environment.yml
conda activate arctic_pipeline

# Or install with pip
pip install -r requirements.txt

# Configure data paths
cp config/paths.py.template config/paths.py
# Edit config/paths.py with your data locations
```

## Usage

### Quick Start
```bash
# Check configuration
python scripts/check_external_drive.py

# Test data access
python scripts/test_imports.py

# Inspect data
python src/data_ingestion/inspect_parquet.py

# Run full pipeline
python scripts/run_pipeline.py
```

### Data Configuration

Update `config/paths.py` with your data locations:
```python
# Example configuration
DATA_DIR = Path('/path/to/your/data')
INPUT_PATHS = {
    'in_situ': DATA_DIR / 'merged_in_situ.parquet',
    'arctic_consolidated': DATA_DIR / 'nisar_smap_consolidated.parquet',
}
```

## Pipeline Stages

1. **Data Inspection**: Validate and inspect input datasets
2. **Quality Control**: Remove outliers and ensure data quality
3. **Feature Engineering**: Create temporal and spatial features
4. **Teacher Forcing Preparation**: Format data for ML training
5. **Visualization**: Generate analysis figures and reports

## Development

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test
python tests/test_inspect_parquet.py
```

### Code Style

This project follows PEP 8 guidelines. Before committing:
```bash
# Format code
black src/ scripts/ tests/

# Check style
flake8 src/ scripts/ tests/
```

## Contributing

This is research code. For questions or collaboration inquiries, please open an issue.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```bibtex
@software{arctic_zero_curtain_pipeline,
  title={Arctic Zero-Curtain Analysis Pipeline},
  author={Gay, Bradley A.},
  year={2025},
  url={https://github.com/bradleygay/zerocurtain}
}
```

## Acknowledgments

- NASA Goddard Space Flight Center
- Arctic research community
- Open-source scientific Python community

## Contact

For questions about this research:
- GitHub Issues: https://github.com/bradleygay/zerocurtain/issues
- Email: your.email@nasa.gov
