#!/usr/bin/env python3
"""
Dedicated script to run physics-informed zero-curtain detection.
Can be executed standalone or as part of the full pipeline.
"""

import sys
from pathlib import Path

# Add parent directories to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "orchestration"))
sys.path.insert(0, str(project_root / "src"))

from orchestration.physics_detection_orchestrator import PhysicsDetectionOrchestrator
from src.physics_detection.physics_config import DetectionConfig, DataPaths, PhysicsParameters


def main():
    """Execute physics-informed detection with default configuration."""
    
    print("\n" + "="*90)
    print("STANDALONE PHYSICS-INFORMED ZERO-CURTAIN DETECTION")
    print("="*90)
    print()
    
    # Create configuration
    # Modify these parameters as needed
    config = DetectionConfig(
        paths=DataPaths(
            base_dir=Path("/path/to/user/Downloads")
        ),
        physics=PhysicsParameters(
            temp_threshold=3.0,
            min_duration_hours=12,
            use_cryogrid_enthalpy=True
        )
    )
    
    # Initialize and run orchestrator
    orchestrator = PhysicsDetectionOrchestrator(config=config)
    
    try:
        results, summary = orchestrator.execute_complete_workflow()
        
        print("\n" + "="*90)
        print("DETECTION COMPLETED SUCCESSFULLY")
        print("="*90)
        print(f"Results saved to: {config.paths.output_dir}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())