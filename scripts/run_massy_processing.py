#!/usr/bin/env python3
"""Process Massy faces on vast.ai server."""
import subprocess
import sys

def run_remote_command(cmd, description):
    """Run command on vast.ai server."""
    print(f"Running on vast.ai: {description}")
    
    ssh_cmd = f"""ssh -p 39430 -o StrictHostKeyChecking=no root@ssh3.vast.ai \
"cd /workspace/adetailer_2cn && {cmd}" """
    
    print(f"Command: {ssh_cmd}")
    
    result = subprocess.run(ssh_cmd, shell=True)
    return result.returncode == 0

def process_massy_faces():
    """Process all Massy face images."""
    
    commands = [
        ("ls -la data/samples/Massy/", "Check Massy samples"),
        ("mkdir -p output/massy_results", "Create output directory"),
        ("pip3 install -r adetailer_2cn_plus/requirements.txt", "Install Python dependencies"),
        ("""python3 adetailer_2cn_plus/scripts/enhance_simple.py \
data/samples/Massy \
output/enhanced_massy \
adetailer_2cn_plus/config.yaml""", "Detect and enhance Massy faces")
    ]
    
    for cmd, desc in commands:
        if not run_remote_command(cmd, desc):
            print(f"Failed: {desc}")
            sys.exit(1)
    
    print("Massy processing completed!")

if __name__ == "__main__":
    process_massy_faces()