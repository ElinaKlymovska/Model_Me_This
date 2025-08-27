#!/usr/bin/env python3
"""Deploy code to vast.ai server."""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run shell command with error handling."""
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    else:
        print(f"Success: {result.stdout}")
        return True

def deploy_to_vast():
    """Deploy codebase to vast.ai."""
    
    # Rsync command to upload code
    rsync_cmd = """rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
-e "ssh -p 39430 -o StrictHostKeyChecking=no" \
./ root@ssh3.vast.ai:/workspace/adetailer_2cn/"""
    
    print("Deploying codebase to vast.ai...")
    
    if not run_command(rsync_cmd, "Uploading code to vast.ai"):
        print("Failed to deploy code")
        sys.exit(1)
    
    print("Deployment completed successfully!")

if __name__ == "__main__":
    deploy_to_vast()