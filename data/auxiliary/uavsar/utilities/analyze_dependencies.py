#!/usr/bin/env python3
"""
Analyze file dependencies in UAVSAR directory
"""

import os
import re
from pathlib import Path
from collections import defaultdict

uavsar_dir = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar")

print("="*80)
print("DEPENDENCY ANALYSIS")
print("="*80)

# Files to analyze
python_files = list(uavsar_dir.glob("*.py"))
shell_files = list(uavsar_dir.glob("*.sh"))
yaml_files = list(uavsar_dir.glob("*.yaml"))

all_files = python_files + shell_files + yaml_files

# Track imports and references
dependencies = defaultdict(set)
references = defaultdict(set)

# Analyze Python files
for py_file in python_files:
    try:
        with open(py_file, 'r') as f:
            content = f.read()
            
        # Find imports from modules/
        module_imports = re.findall(r'from modules\.(\w+) import', content)
        for mod in module_imports:
            dependencies[py_file.name].add(f"modules/{mod}.py")
        
        # Find file references
        file_refs = re.findall(r'["\']([^"\']+\.(?:py|sh|yaml|json|parquet))["\']', content)
        for ref in file_refs:
            if not ref.startswith('/'):  # Relative paths only
                dependencies[py_file.name].add(ref)
    except:
        pass

# Analyze shell files
for sh_file in shell_files:
    try:
        with open(sh_file, 'r') as f:
            content = f.read()
        
        # Find Python script calls
        py_calls = re.findall(r'python3?\s+([^\s]+\.py)', content)
        for py in py_calls:
            dependencies[sh_file.name].add(py)
    except:
        pass

# Print critical files (have dependencies)
print("\n CRITICAL FILES (have dependencies):")
critical = []
for file, deps in sorted(dependencies.items()):
    if deps:
        print(f"\n  {file}:")
        for dep in sorted(deps):
            print(f"    → {dep}")
        critical.append(file)

# Identify standalone files (no dependencies)
all_names = {f.name for f in all_files}
standalone = all_names - set(critical)

print("\n" + "="*80)
print(" FILE CATEGORIZATION")
print("="*80)

categories = {
    'PRODUCTION': [
        'run_pipeline.py',
        'consolidate_all_data.py',
        'production/pipeline_config.yaml'
    ],
    'TESTING': [
        'simple_consolidation.py',
        'simple_consolidate.py',
        'diagnose_spatial.py'
    ],
    'UTILITIES': [
        'generate_curl_script.py',
        'cross_ref_sh_txt.py',
        'inventory_existing_data.py'
    ],
    'DOWNLOADS': [
        'uavsar_download_curl.sh',
        'uavsar_download_consolidated.sh',
        'nisar_download.sh'
    ]
}

# Categorize files
categorized = set()
for category, files in categories.items():
    found = [f for f in files if f in all_names]
    if found:
        print(f"\n{category}:")
        for f in found:
            print(f"  • {f}")
            categorized.add(f)

# Uncategorized
uncategorized = all_names - categorized
if uncategorized:
    print(f"\nUNCATEGORIZED:")
    for f in sorted(uncategorized):
        print(f"  • {f}")

print("\n" + "="*80)
