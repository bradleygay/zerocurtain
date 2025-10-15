#!/usr/bin/env python3

"""
Automated extraction of Jupyter notebook cells into standalone Python modules.
Designed for large-scale scientific computing notebooks (95K+ lines).

Author: Arctic Data Pipeline Team
Purpose: Zero-Curtain Research Infrastructure
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import nbformat
from nbformat import NotebookNode


class NotebookExtractor:
    """Extract and organize Jupyter notebook cells into modular Python files."""
    
    def __init__(self, notebook_path: str, output_dir: str = "arctic_zero_curtain_pipeline"):
        self.notebook_path = Path(notebook_path)
        self.output_dir = Path(output_dir)
        self.notebook = None
        self.cell_metadata = []
        
    def load_notebook(self):
        """Load notebook with nbformat."""
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            self.notebook = nbformat.read(f, as_version=4)
        print(f"Loaded notebook with {len(self.notebook.cells)} cells")
        
    def analyze_cell_structure(self) -> List[Dict]:
        """Analyze cells to determine functional categories."""
        analysis = []
        
        for idx, cell in enumerate(self.notebook.cells):
            if cell.cell_type != 'code':
                continue
                
            source = cell.source
            
            # Classify cell by content patterns
            classification = self._classify_cell(source, idx)
            
            analysis.append({
                'index': idx,
                'classification': classification,
                'line_count': len(source.split('\n')),
                'has_imports': 'import ' in source,
                'has_functions': 'def ' in source,
                'has_classes': 'class ' in source,
                'source': source
            })
            
        return analysis
    
    def _classify_cell(self, source: str, idx: int) -> str:
        """Classify cell based on content patterns."""
        source_lower = source.lower()
        
        # Pattern matching for classification
        if idx == 0 or (source.count('import ') > 5 and len(source) < 2000):
            return 'imports'
        elif 'def inspect_parquet' in source or 'def analyze_' in source:
            return 'data_inspection'
        elif 'def transform_' in source:
            return 'transformation'
        elif 'def merge_' in source or 'concat' in source_lower:
            return 'merging'
        elif 'def plot_' in source or 'plt.' in source or 'matplotlib' in source_lower:
            return 'visualization'
        elif 'model' in source_lower and ('keras' in source_lower or 'tensorflow' in source_lower):
            return 'modeling'
        elif 'def main' in source or '__main__' in source:
            return 'orchestration'
        elif 'class ' in source:
            return 'classes'
        else:
            return 'processing'
    
    def extract_imports(self, analysis: List[Dict]) -> str:
        """Consolidate all imports into a single module."""
        all_imports = set()
        
        for cell_info in analysis:
            if cell_info['has_imports']:
                lines = cell_info['source'].split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        # Remove comments
                        line = line.split('#')[0].strip()
                        if line:
                            all_imports.add(line)
        
        # Organize imports by category
        stdlib = []
        thirdparty = []
        local = []
        
        for imp in sorted(all_imports):
            if imp.startswith('from .') or imp.startswith('import .'):
                local.append(imp)
            elif any(lib in imp for lib in ['os', 'sys', 'json', 're', 'glob', 'pathlib', 
                                             'datetime', 'logging', 'warnings', 'pickle',
                                             'concurrent', 'packaging']):
                stdlib.append(imp)
            else:
                thirdparty.append(imp)
        
        import_str = '"""Common imports for Arctic zero-curtain pipeline."""\n\n'
        import_str += '# Standard library\n' + '\n'.join(stdlib) + '\n\n'
        import_str += '# Third-party libraries\n' + '\n'.join(thirdparty) + '\n\n'
        if local:
            import_str += '# Local imports\n' + '\n'.join(local) + '\n'
            
        return import_str
    
    def extract_by_classification(self, analysis: List[Dict]) -> Dict[str, List[Dict]]:
        """Group cells by classification."""
        classified = {}
        
        for cell_info in analysis:
            classification = cell_info['classification']
            if classification not in classified:
                classified[classification] = []
            classified[classification].append(cell_info)
            
        return classified
    
    def create_module_structure(self):
        """Create directory structure for modular pipeline."""
        directories = [
            'src/common',
            'src/data_ingestion',
            'src/transformations',
            'src/processing',
            'src/visualization',
            'src/modeling',
            'orchestration',
            'config',
            'scripts',
            'tests',
            'data/raw',
            'data/processed',
            'data/intermediate',
            'outputs/figures',
            'outputs/models',
            'logs'
        ]
        
        for directory in directories:
            dir_path = self.output_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py for Python packages
            if directory.startswith('src/') or directory == 'orchestration':
                init_file = dir_path / '__init__.py'
                if not init_file.exists():
                    init_file.write_text('"""' + directory.replace('src/', '').replace('/', ' ').title() + ' module."""\n')
        
        print(f"Created module structure in {self.output_dir}")
    
    def write_module_file(self, classification: str, cells: List[Dict], module_name: str):
        """Write cells to appropriate module file."""
        # Determine output directory based on classification
        dir_mapping = {
            'imports': 'src/common',
            'data_inspection': 'src/data_ingestion',
            'transformation': 'src/transformations',
            'merging': 'src/processing',
            'visualization': 'src/visualization',
            'modeling': 'src/modeling',
            'orchestration': 'orchestration',
            'processing': 'src/processing',
            'classes': 'src/common'
        }
        
        output_dir = self.output_dir / dir_mapping.get(classification, 'src/processing')
        output_file = output_dir / f"{module_name}.py"
        
        # Build module content
        content = f'"""\n{classification.replace("_", " ").title()} module for Arctic zero-curtain pipeline.\n'
        content += 'Auto-generated from Jupyter notebook.\n"""\n\n'
        
        # Add imports from common module
        if classification != 'imports':
            content += 'from src.common.imports import *\n'
            content += 'from src.common.utilities import *\n\n'
        
        # Add cell contents
        for cell in cells:
            source = cell['source']
            # Remove notebook-specific magic commands
            source = re.sub(r'^%.*$', '', source, flags=re.MULTILINE)
            source = re.sub(r'^!.*$', '', source, flags=re.MULTILINE)
            content += source + '\n\n'
        
        output_file.write_text(content)
        print(f"Created module: {output_file}")
        
        return output_file
    
    def create_config_files(self, analysis: List[Dict]):
        """Create configuration files for paths and parameters."""
        # Extract paths from notebook
        paths = {}
        for cell_info in analysis:
            source = cell_info['source']
            # Look for path definitions
            path_matches = re.findall(r"['\"]([/\w]+\.parquet)['\"]", source)
            for match in path_matches:
                key = Path(match).stem
                paths[key] = match
        
        # Write paths.py
        paths_file = self.output_dir / 'config' / 'paths.py'
        paths_content = '"""Path configuration for Arctic pipeline."""\n\n'
        paths_content += 'from pathlib import Path\n\n'
        paths_content += 'BASE_DIR = Path(__file__).parent.parent\n\n'
        paths_content += 'PATHS = {\n'
        for key, value in paths.items():
            paths_content += f"    '{key}': '{value}',\n"
        paths_content += '}\n'
        paths_file.write_text(paths_content)
        
        # Write parameters.py
        params_file = self.output_dir / 'config' / 'parameters.py'
        params_content = '"""Pipeline parameters."""\n\n'
        params_content += 'PARAMETERS = {\n'
        params_content += "    'n_preview_rows': 10,\n"
        params_content += "    'percentile_min': 5,\n"
        params_content += "    'percentile_max': 95,\n"
        params_content += "    'compression': 'snappy',\n"
        params_content += "    'engine': 'pyarrow',\n"
        params_content += '}\n'
        params_file.write_text(params_content)
        
        print("Created configuration files")
    
    def extract_pipeline(self):
        """Main extraction pipeline."""
        print("Starting notebook extraction...")
        
        # Load and analyze
        self.load_notebook()
        analysis = self.analyze_cell_structure()
        
        print(f"\nCell classification summary:")
        classified = self.extract_by_classification(analysis)
        for classification, cells in classified.items():
            print(f"  {classification}: {len(cells)} cell(s), "
                  f"{sum(c['line_count'] for c in cells)} lines")
        
        # Create structure
        self.create_module_structure()
        
        # Extract imports
        imports_content = self.extract_imports(analysis)
        imports_file = self.output_dir / 'src' / 'common' / 'imports.py'
        imports_file.write_text(imports_content)
        print(f"\nConsolidated imports: {imports_file}")
        
        # Write modules
        print("\nCreating modules:")
        for classification, cells in classified.items():
            if classification == 'imports':
                continue
            module_name = f"{classification}_module"
            self.write_module_file(classification, cells, module_name)
        
        # Create config files
        self.create_config_files(analysis)
        
        # Create requirements.txt
        self.create_requirements_file()
        
        # Create orchestration runner
        self.create_pipeline_runner()
        
        print("\n" + "="*60)
        print("Extraction complete!")
        print(f"Output directory: {self.output_dir.absolute()}")
        print("="*60)
    
    def create_requirements_file(self):
        """Generate requirements.txt from imports."""
        requirements = [
            'numpy>=1.24.0',
            'pandas>=2.0.0',
            'dask[complete]>=2023.1.0',
            'pyarrow>=11.0.0',
            'polars>=0.20.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
            'cartopy>=0.21.0',
            'geopandas>=0.14.0',
            'shapely>=2.0.0',
            'folium>=0.14.0',
            'rasterio>=1.3.0',
            'contextily>=1.3.0',
            'scikit-learn>=1.3.0',
            'scipy>=1.10.0',
            'tensorflow>=2.13.0',
            'keras>=2.13.0',
            'keras-tuner>=1.3.0',
            'xarray>=2023.1.0',
            'tables>=3.8.0',
            'cmocean>=3.0.0',
            'cmasher>=1.6.0',
            'tqdm>=4.65.0',
            'tabulate>=0.9.0',
            'jupyterlab>=4.0.0',
            'nbformat>=5.9.0',
            'geodatasets>=2023.12.0',
            'pyproj>=3.5.0'
        ]
        
        req_file = self.output_dir / 'requirements.txt'
        req_file.write_text('\n'.join(requirements))
        print(f"Created: {req_file}")
    
    def create_pipeline_runner(self):
        """Create main pipeline orchestration script."""
        runner_content = '''#!/usr/bin/env python3
"""
Main pipeline runner for Arctic zero-curtain in situ database construction.
Orchestrates synchronous execution of all pipeline stages.
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.paths import PATHS
from config.parameters import PARAMETERS
from data_ingestion.data_inspection_module import inspect_parquet
from transformations.transformation_module import transform_uavsar_nisar, transform_smap
from processing.merging_module import merge_arctic_datasets
from visualization.visualization_module import plot_arctic_projection, analyze_data_coverage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ArcticPipelineRunner:
    """Orchestrate the complete Arctic in situ database pipeline."""
    
    def __init__(self, paths: dict, params: dict):
        self.paths = paths
        self.params = params
        self.results = {}
        
    def run_stage_1_inspection(self):
        """Stage 1: Data inspection and validation."""
        logger.info("="*60)
        logger.info("STAGE 1: DATA INSPECTION")
        logger.info("="*60)
        
        datasets = [
            ('in_situ', self.paths.get('in_situ')),
            ('uavsar_nisar', self.paths.get('uavsar_nisar_raw')),
            ('smap', self.paths.get('smap_raw'))
        ]
        
        for name, path in datasets:
            if path and Path(path).exists():
                logger.info(f"Inspecting {name}...")
                df = inspect_parquet(path, n_rows=self.params['n_preview_rows'])
                self.results[f'{name}_inspected'] = True
            else:
                logger.warning(f"Path not found: {path}")
        
        return True
    
    def run_stage_2_transformation(self):
        """Stage 2: Data transformation and harmonization."""
        logger.info("="*60)
        logger.info("STAGE 2: DATA TRANSFORMATION")
        logger.info("="*60)
        
        # Transform UAVSAR/NISAR
        uavsar_output = self.paths.get('uavsar_nisar_transformed')
        if not Path(uavsar_output).exists():
            logger.info("Transforming UAVSAR/NISAR data...")
            transform_uavsar_nisar(
                self.paths['uavsar_nisar_raw'],
                uavsar_output
            )
        
        # Transform SMAP
        smap_output = self.paths.get('smap_transformed')
        if not Path(smap_output).exists():
            logger.info("Transforming SMAP data...")
            transform_smap(
                self.paths['smap_raw'],
                smap_output
            )
        
        return True
    
    def run_stage_3_merging(self):
        """Stage 3: Dataset merging and consolidation."""
        logger.info("="*60)
        logger.info("STAGE 3: DATASET MERGING")
        logger.info("="*60)
        
        merge_datasets = {
            'in_situ': self.paths['in_situ'],
            'uavsar_nisar': self.paths['uavsar_nisar_transformed'],
            'smap': self.paths['smap_transformed']
        }
        
        output_path = self.paths.get('merged_final')
        if not Path(output_path).exists():
            logger.info("Merging Arctic datasets...")
            df_merged = merge_arctic_datasets(merge_datasets, output_path)
            self.results['merged_dataset'] = output_path
        else:
            logger.info(f"Merged dataset already exists: {output_path}")
        
        return True
    
    def run_stage_4_quality_control(self):
        """Stage 4: Quality control and coverage analysis."""
        logger.info("="*60)
        logger.info("STAGE 4: QUALITY CONTROL")
        logger.info("="*60)
        
        # Implement quality control checks here
        logger.info("Running quality control checks...")
        
        return True
    
    def run_stage_5_teacher_forcing_prep(self):
        """Stage 5: Prepare dataset for teacher forcing."""
        logger.info("="*60)
        logger.info("STAGE 5: TEACHER FORCING PREPARATION")
        logger.info("="*60)
        
        # Implement teacher forcing preparation here
        logger.info("Preparing dataset for teacher forcing...")
        
        return True
    
    def run_full_pipeline(self):
        """Execute complete pipeline synchronously."""
        start_time = datetime.now()
        logger.info("\\n" + "="*60)
        logger.info("ARCTIC IN SITU DATABASE PIPELINE - START")
        logger.info("="*60 + "\\n")
        
        try:
            # Execute stages in sequence
            stages = [
                ('Inspection', self.run_stage_1_inspection),
                ('Transformation', self.run_stage_2_transformation),
                ('Merging', self.run_stage_3_merging),
                ('Quality Control', self.run_stage_4_quality_control),
                ('Teacher Forcing Prep', self.run_stage_5_teacher_forcing_prep)
            ]
            
            for stage_name, stage_func in stages:
                logger.info(f"\\nExecuting: {stage_name}")
                success = stage_func()
                if not success:
                    logger.error(f"Stage failed: {stage_name}")
                    return False
                logger.info(f"Completed: {stage_name}")
            
            elapsed = datetime.now() - start_time
            logger.info("\\n" + "="*60)
            logger.info(f"PIPELINE COMPLETE - Elapsed time: {elapsed}")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            return False


def main():
    """Main entry point."""
    runner = ArcticPipelineRunner(PATHS, PARAMETERS)
    success = runner.run_full_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
'''
        
        runner_file = self.output_dir / 'scripts' / 'run_pipeline.py'
        runner_file.write_text(runner_content)
        runner_file.chmod(0o755)
        print(f"Created pipeline runner: {runner_file}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract Jupyter notebook into modular Python pipeline'
    )
    parser.add_argument(
        'notebook_path',
        help='Path to input Jupyter notebook (.ipynb)'
    )
    parser.add_argument(
        '--output-dir',
        default='arctic_zero_curtain_pipeline',
        help='Output directory for extracted modules'
    )
    
    args = parser.parse_args()
    
    extractor = NotebookExtractor(args.notebook_path, args.output_dir)
    extractor.extract_pipeline()


if __name__ == "__main__":
    main()
