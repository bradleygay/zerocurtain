#!/usr/bin/env python3
"""
Test single H5 file structure
"""

import h5py
import numpy as np
from pathlib import Path

h5_file = Path("/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar/nisar_downloads").rglob("*.h5").__next__()

print(f"Testing: {h5_file}")

with h5py.File(h5_file, 'r') as f:
    print("\nTop-level groups:")
    for key in f.keys():
        print(f"  /{key}")
    
    # Look for SLC data
    if 'science' in f:
        print("\n/science:")
        for key in f['science'].keys():
            print(f"  /science/{key}")
        
        if 'LSAR' in f['science']:
            print("\n/science/LSAR:")
            for key in f['science/LSAR'].keys():
                print(f"  /science/LSAR/{key}")
            
            # Check for SLC
            if 'SLC' in f['science/LSAR']:
                print("\n/science/LSAR/SLC:")
                for key in f['science/LSAR/SLC'].keys():
                    print(f"  /science/LSAR/SLC/{key}")
                
                # Check frequency
                if 'frequencyA' in f['science/LSAR/SLC']:
                    print("\n/science/LSAR/SLC/frequencyA:")
                    for key in f['science/LSAR/SLC/frequencyA'].keys():
                        print(f"  {key}")
                        
                        # Check for r/i components
                        pol_path = f"/science/LSAR/SLC/frequencyA/{key}"
                        if pol_path in f:
                            for component in ['r', 'i']:
                                comp_path = f"{pol_path}/{component}"
                                if comp_path in f:
                                    shape = f[comp_path].shape
                                    dtype = f[comp_path].dtype
                                    print(f"    {component}: shape={shape}, dtype={dtype}")
