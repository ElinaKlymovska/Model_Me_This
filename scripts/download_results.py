#!/usr/bin/env python3
"""Download processed results from vast.ai."""
import subprocess
import sys
from pathlib import Path

def download_results():
    """Download processed results from vast.ai."""
    
    # Create local results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Rsync command to download results
    download_cmd = """rsync -avz --progress \
-e "ssh -p 39430 -o StrictHostKeyChecking=no" \
root@ssh3.vast.ai:/workspace/adetailer_2cn/output/ \
./results/"""
    
    print("Downloading results from vast.ai...")
    
    result = subprocess.run(download_cmd, shell=True)
    
    if result.returncode == 0:
        print("Results downloaded successfully to ./results/")
    else:
        print("Failed to download results")
        sys.exit(1)

if __name__ == "__main__":
    download_results()