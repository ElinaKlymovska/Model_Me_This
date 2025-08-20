"""
Tests for configuration files
"""
import unittest
import yaml
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestConfiguration(unittest.TestCase):
    """Test configuration files loading and validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_dir = Path(__file__).parent.parent / "config"
        self.config_file = self.config_dir / "config.yaml"
        self.models_file = self.config_dir / "models.yaml"
    
    def test_config_files_exist(self):
        """Test that configuration files exist"""
        self.assertTrue(self.config_file.exists(), "config.yaml should exist")
        self.assertTrue(self.models_file.exists(), "models.yaml should exist")
    
    def test_config_yaml_valid(self):
        """Test that config.yaml is valid YAML"""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            self.assertIsInstance(config, dict, "config.yaml should be valid YAML")
        except yaml.YAMLError as e:
            self.fail(f"config.yaml is not valid YAML: {e}")
    
    def test_models_yaml_valid(self):
        """Test that models.yaml is valid YAML"""
        try:
            with open(self.models_file, 'r') as f:
                models = yaml.safe_load(f)
            self.assertIsInstance(models, dict, "models.yaml should be valid YAML")
        except yaml.YAMLError as e:
            self.fail(f"models.yaml is not valid YAML: {e}")
    
    def test_config_structure(self):
        """Test config.yaml has required structure"""
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required top-level keys
        required_keys = ['general', 'webui', 'adetailer_2cn', 'processing']
        for key in required_keys:
            self.assertIn(key, config, f"config.yaml should contain '{key}' section")
        
        # Check webui section
        webui_keys = ['host', 'port', 'base_url']
        for key in webui_keys:
            self.assertIn(key, config['webui'], f"webui section should contain '{key}'")
    
    def test_models_structure(self):
        """Test models.yaml has required structure"""
        with open(self.models_file, 'r') as f:
            models = yaml.safe_load(f)
        
        # Check required top-level keys
        required_keys = ['stable_diffusion', 'controlnet', 'adetailer']
        for key in required_keys:
            self.assertIn(key, models, f"models.yaml should contain '{key}' section")
        
        # Check stable_diffusion section
        self.assertIn('models', models['stable_diffusion'], 
                     "stable_diffusion section should contain 'models'")
        self.assertIsInstance(models['stable_diffusion']['models'], list,
                            "stable_diffusion models should be a list")

if __name__ == '__main__':
    unittest.main()
