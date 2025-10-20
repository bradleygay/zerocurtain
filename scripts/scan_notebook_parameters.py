#!/usr/bin/env python3
"""
Scan Jupyter notebook to extract exact parameters and methodology.
Identifies splits, sampling, and any numeric parameters used.
"""

import json
import re
from pathlib import Path
import sys


def scan_notebook(notebook_path):
    """Extract parameters from Jupyter notebook."""
    
    print("=" * 80)
    print("JUPYTER NOTEBOOK PARAMETER SCANNER")
    print("=" * 80)
    print(f"\nScanning: {notebook_path}\n")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    findings = {
        'splits': [],
        'sampling': [],
        'sequence_params': [],
        'validation_params': [],
        'test_params': [],
        'numeric_constants': [],
        'sort_operations': [],
        'stratification': []
    }
    
    # Patterns to search for
    patterns = {
        'train_split': r'(train.*split|validation_split|test_split)\s*=\s*([\d.]+)',
        'sample': r'\.sample\s*\(\s*n\s*=\s*(\d+)',
        'frac_sample': r'\.sample\s*\(\s*frac\s*=\s*([\d.]+)',
        'sequence_length': r'sequence_length\s*=\s*(\d+)',
        'prediction_horizon': r'(prediction_horizon|forecast_horizon|horizon)\s*=\s*(\d+)',
        'random_state': r'random_state\s*=\s*(\d+)',
        'test_size': r'test_size\s*=\s*([\d.]+)',
        'train_size': r'train_size\s*=\s*([\d.]+)',
        'stratify': r'stratify\s*=',
        'sort_values': r'\.sort_values\s*\(',
        'train_test_split': r'train_test_split\s*\(',
    }
    
    print("Searching for parameters in code cells...\n")
    
    for cell_idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        
        # Check each pattern
        for param_name, pattern in patterns.items():
            matches = re.finditer(pattern, source, re.IGNORECASE)
            for match in matches:
                finding = {
                    'cell': cell_idx,
                    'parameter': param_name,
                    'match': match.group(0),
                    'value': match.groups() if match.groups() else None,
                    'context': source[max(0, match.start()-50):min(len(source), match.end()+50)]
                }
                
                # Categorize finding
                if 'split' in param_name or 'size' in param_name:
                    findings['splits'].append(finding)
                elif 'sample' in param_name:
                    findings['sampling'].append(finding)
                elif 'sequence' in param_name or 'horizon' in param_name:
                    findings['sequence_params'].append(finding)
                elif 'stratify' in param_name:
                    findings['stratification'].append(finding)
                elif 'sort' in param_name:
                    findings['sort_operations'].append(finding)
        
        # Look for literal split operations
        if 'train_test_split' in source or 'train_val_test' in source:
            findings['splits'].append({
                'cell': cell_idx,
                'parameter': 'split_operation',
                'match': 'Found split operation',
                'context': source[:200]
            })
        
        # Look for numeric constants that might be parameters
        numeric_assignments = re.finditer(r'(\w+)\s*=\s*(\d+\.?\d*)', source)
        for match in numeric_assignments:
            var_name = match.group(1)
            if any(keyword in var_name.lower() for keyword in ['split', 'size', 'frac', 'length', 'horizon', 'window', 'step']):
                findings['numeric_constants'].append({
                    'cell': cell_idx,
                    'variable': var_name,
                    'value': match.group(2),
                    'context': source[max(0, match.start()-30):min(len(source), match.end()+30)]
                })
    
    # Report findings
    print("-" * 80)
    print("FINDINGS:")
    print("-" * 80)
    
    for category, items in findings.items():
        if items:
            print(f"\n{category.upper().replace('_', ' ')}:")
            print("-" * 40)
            for item in items:
                print(f"  Cell {item['cell']}:")
                if 'match' in item:
                    print(f"    Found: {item['match']}")
                if 'value' in item and item['value']:
                    print(f"    Value: {item['value']}")
                if 'variable' in item:
                    print(f"    Variable: {item['variable']} = {item['value']}")
                print(f"    Context: ...{item['context']}...")
                print()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if findings['splits']:
        print(f" Found {len(findings['splits'])} split-related parameters")
    else:
        print("  No explicit split parameters found")
    
    if findings['sampling']:
        print(f" Found {len(findings['sampling'])} sampling operations")
    else:
        print(" No sampling operations found (using full dataset)")
    
    if findings['sequence_params']:
        print(f" Found {len(findings['sequence_params'])} sequence parameters")
    else:
        print("  No sequence parameters found")
    
    print("\n" + "=" * 80)
    
    return findings


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Scan Jupyter notebook for parameters and methodology'
    )
    parser.add_argument(
        'notebook',
        help='Path to Jupyter notebook (.ipynb)'
    )
    
    args = parser.parse_args()
    
    notebook_path = Path(args.notebook)
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found: {notebook_path}")
        sys.exit(1)
    
    findings = scan_notebook(notebook_path)
    
    # Optionally save findings
    output_path = Path('notebook_parameters_extracted.json')
    with open(output_path, 'w') as f:
        json.dump(findings, f, indent=2, default=str)
    
    print(f"\nDetailed findings saved to: {output_path}")


if __name__ == "__main__":
    main()
