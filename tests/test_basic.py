"""
Basic tests for core functionality
"""
import unittest
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestBasicFunctionality(unittest.TestCase):
    """Test basic project functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.project_root = Path(__file__).parent.parent
        
    def test_project_structure(self):
        """Test that project has required directories"""
        required_dirs = [
            'config',
            'docker',
            'scripts',
            'portrait-enhancer',
            'adetailer_2cn_plus'
        ]
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            self.assertTrue(dir_path.exists(), f"Directory {dir_name} should exist")
            self.assertTrue(dir_path.is_dir(), f"{dir_name} should be a directory")
    
    def test_required_files(self):
        """Test that required files exist"""
        required_files = [
            'README.md',
            'Makefile',
            'config/config.yaml',
            'config/models.yaml',
            'docker/Dockerfile',
            'docker/docker-compose.yml',
            'scripts/bootstrap.sh',
            'scripts/deploy_vast.sh'
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            self.assertTrue(full_path.exists(), f"File {file_path} should exist")
    
    def test_python_modules(self):
        """Test that Python modules can be imported"""
        try:
            # Test basic imports
            import yaml
            import requests
            import numpy
            import cv2
        except ImportError as e:
            self.fail(f"Required Python modules should be available: {e}")
    
    def test_config_loading(self):
        """Test that configuration can be loaded"""
        try:
            import yaml
            config_file = self.project_root / "config" / "config.yaml"
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            self.assertIsInstance(config, dict, "Configuration should load as dictionary")
        except Exception as e:
            self.fail(f"Configuration loading failed: {e}")

if __name__ == '__main__':
    unittest.main()
