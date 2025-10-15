#!/usr/bin/env python3
"""Test that imports work correctly."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project Root: {PROJECT_ROOT}")
print(f"Python Path: {sys.path[0]}\n")

# Test imports
try:
    from config.paths import INPUT_PATHS, PATHS, BASE_DIR
    print(" Successfully imported config.paths")
    print(f"  Base Directory: {BASE_DIR}")
    print(f"  Number of paths configured: {len(PATHS)}")
except ImportError as e:
    print(f" Failed to import config.paths: {e}")
    sys.exit(1)

try:
    from config.parameters import PARAMETERS
    print(" Successfully imported config.parameters")
    print(f"  Number of parameters: {len(PARAMETERS)}")
except ImportError as e:
    print(f" Failed to import config.parameters: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("All imports successful!")
print("="*50)

# Show which data files exist
print("\nData Files Status:")
for name, path in INPUT_PATHS.items():
    exists = Path(path).exists()
    status = "" if exists else ""
    size = f"({Path(path).stat().st_size / 1024**2:.1f} MB)" if exists else ""
    print(f"  {status} {name}: {path} {size}")
