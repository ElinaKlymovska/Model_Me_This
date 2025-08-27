#!/usr/bin/env python3
"""Master script to run the complete Massy processing pipeline."""
import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script."""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    
    result = subprocess.run([sys.executable, script_name], cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        sys.exit(1)
    
    print(f"SUCCESS: {description} completed!")

def main():
    """Run the complete pipeline."""
    print("Starting Massy Face Processing Pipeline")
    
    scripts = [
        ("scripts/deploy_to_vast.py", "Deploy code to vast.ai"),
        ("scripts/run_massy_processing.py", "Process Massy faces"),
        ("scripts/download_results.py", "Download results")
    ]
    
    for script, description in scripts:
        if not Path(script).exists():
            print(f"ERROR: Script {script} not found!")
            sys.exit(1)
        
        run_script(script, description)
    
    print(f"\n{'='*50}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Results available in ./results/ directory")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()