#!/usr/bin/env python3
"""Monitor vast.ai processing and download results."""
import subprocess
import sys
import time
import os
from pathlib import Path

class VastMonitor:
    """Monitor and manage vast.ai processing tasks."""
    
    def __init__(self, host="ssh3.vast.ai", port=39430, username="root"):
        self.host = host
        self.port = port
        self.username = username
        self.base_ssh_cmd = f"ssh -p {port} -o StrictHostKeyChecking=no {username}@{host}"
        
    def run_remote_command(self, cmd, description="", timeout=300):
        """Run command on vast.ai server with timeout."""
        print(f"🚀 {description}")
        print(f"Command: {cmd}")
        
        full_cmd = f'{self.base_ssh_cmd} "cd /workspace/adetailer_2cn && {cmd}"'
        
        try:
            result = subprocess.run(
                full_cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Success: {description}")
                if result.stdout.strip():
                    print(f"Output: {result.stdout.strip()}")
                return True, result.stdout
            else:
                print(f"❌ Failed: {description}")
                if result.stderr.strip():
                    print(f"Error: {result.stderr.strip()}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout: {description} took longer than {timeout}s")
            return False, "Timeout"
        except Exception as e:
            print(f"💥 Exception: {e}")
            return False, str(e)
    
    def check_vast_status(self):
        """Check if vast.ai server is accessible."""
        success, output = self.run_remote_command("whoami && pwd", "Checking vast.ai connection")
        return success
    
    def install_dependencies(self):
        """Install required dependencies on vast.ai."""
        commands = [
            ("apt update && apt install -y python3-pip", "Update package manager"),
            ("pip3 install mediapipe opencv-python pydantic", "Install core dependencies"),
            ("pip3 install numpy pillow", "Install image processing libraries")
        ]
        
        for cmd, desc in commands:
            success, _ = self.run_remote_command(cmd, desc, timeout=600)
            if not success:
                print(f"Failed to install dependencies: {desc}")
                return False
        return True
    
    def deploy_code(self):
        """Deploy code to vast.ai server."""
        print("📤 Deploying code to vast.ai...")
        
        # Create rsync command
        rsync_cmd = f"""rsync -avz --progress \
--exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
--exclude='output' --exclude='*.log' \
-e "ssh -p {self.port} -o StrictHostKeyChecking=no" \
./ {self.username}@{self.host}:/workspace/adetailer_2cn/"""
        
        try:
            result = subprocess.run(rsync_cmd, shell=True, timeout=300)
            if result.returncode == 0:
                print("✅ Code deployment successful")
                return True
            else:
                print("❌ Code deployment failed")
                return False
        except Exception as e:
            print(f"💥 Deployment error: {e}")
            return False
    
    def run_face_processing(self, person="Massy", config_file="adetailer_2cn_plus/config.yaml"):
        """Run face processing for specific person."""
        print(f"🎭 Processing {person} faces...")
        
        # Commands for processing
        commands = [
            ("ls -la data/samples/", "Check available samples"),
            (f"mkdir -p output/{person.lower()}_results", "Create output directory"),
            (f"ls -la data/samples/{person}/", f"Check {person} samples"),
            (f"""python3 adetailer_2cn_plus/scripts/enhance_simple.py \
data/samples/{person} \
output/{person.lower()}_enhanced \
{config_file}""", f"Process {person} faces with contouring")
        ]
        
        for cmd, desc in commands:
            success, output = self.run_remote_command(cmd, desc, timeout=600)
            if not success:
                print(f"❌ Processing failed at: {desc}")
                return False
                
            # Check if processing completed successfully
            if "PROCESSING COMPLETE" in output:
                print(f"🎉 {person} processing completed successfully!")
                
        return True
    
    def download_results(self, person="Massy", local_output="output/vast_results"):
        """Download processing results from vast.ai."""
        print(f"📥 Downloading {person} results...")
        
        # Create local output directory
        Path(local_output).mkdir(parents=True, exist_ok=True)
        
        # Download command
        download_cmd = f"""rsync -avz --progress \
-e "ssh -p {self.port} -o StrictHostKeyChecking=no" \
{self.username}@{self.host}:/workspace/adetailer_2cn/output/ \
{local_output}/"""
        
        try:
            result = subprocess.run(download_cmd, shell=True, timeout=600)
            if result.returncode == 0:
                print(f"✅ Results downloaded to {local_output}")
                return True
            else:
                print("❌ Download failed")
                return False
        except Exception as e:
            print(f"💥 Download error: {e}")
            return False
    
    def full_processing_pipeline(self, person="Massy"):
        """Run complete processing pipeline."""
        print(f"🚀 Starting full pipeline for {person}")
        
        steps = [
            (self.check_vast_status, "Check vast.ai connection"),
            (self.deploy_code, "Deploy code"),
            (self.install_dependencies, "Install dependencies"),
            (lambda: self.run_face_processing(person), f"Process {person} faces"),
            (lambda: self.download_results(person), "Download results")
        ]
        
        for step_func, step_desc in steps:
            print(f"\n📋 Step: {step_desc}")
            success = step_func()
            
            if not success:
                print(f"💥 Pipeline failed at: {step_desc}")
                return False
                
        print(f"\n🎉 Full pipeline completed successfully for {person}!")
        return True
    
    def monitor_processing(self, check_interval=30, max_checks=20):
        """Monitor processing status."""
        print(f"👁️ Monitoring processing (checking every {check_interval}s)")
        
        for i in range(max_checks):
            print(f"\n🔍 Check {i+1}/{max_checks}")
            
            # Check if processing is still running
            success, output = self.run_remote_command(
                "ps aux | grep python3 | grep -v grep", 
                "Check running processes"
            )
            
            if success and "enhance_simple.py" in output:
                print("⏳ Processing still running...")
                time.sleep(check_interval)
            else:
                print("✅ Processing appears to be finished")
                break
                
        # Final status check
        success, output = self.run_remote_command(
            "ls -la output/ && echo '---' && tail -20 output/*/enhanced_*.jpg 2>/dev/null | wc -l",
            "Check final results"
        )


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python3 monitor_vast.py <command> [person]")
        print("Commands: status, deploy, process, download, full, monitor")
        print("Persons: Massy, Orbi, Yana")
        sys.exit(1)
    
    command = sys.argv[1]
    person = sys.argv[2] if len(sys.argv) > 2 else "Massy"
    
    monitor = VastMonitor()
    
    if command == "status":
        success = monitor.check_vast_status()
        sys.exit(0 if success else 1)
    
    elif command == "deploy":
        success = monitor.deploy_code()
        sys.exit(0 if success else 1)
    
    elif command == "process":
        success = monitor.run_face_processing(person)
        sys.exit(0 if success else 1)
    
    elif command == "download":
        success = monitor.download_results(person)
        sys.exit(0 if success else 1)
    
    elif command == "full":
        success = monitor.full_processing_pipeline(person)
        sys.exit(0 if success else 1)
        
    elif command == "monitor":
        monitor.monitor_processing()
        
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()