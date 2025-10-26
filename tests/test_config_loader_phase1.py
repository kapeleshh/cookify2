"""
Unit tests for Phase 1 improvements in configuration loader
"""

import unittest
import tempfile
import os
import yaml
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config_loader import load_config


class TestConfigLoaderPhase1(unittest.TestCase):
    """Test cases for Phase 1 configuration loader improvements."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_config_with_valid_file(self):
        """Test loading configuration from a valid file."""
        config_data = {
            "general": {
                "output_dir": "test_output",
                "log_level": "INFO"
            },
            "logging": {
                "log_dir": "logs",
                "level": "DEBUG"
            }
        }
        
        config_file = os.path.join(self.temp_dir, "test_config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(config_file)
        
        self.assertEqual(config["general"]["output_dir"], "test_output")
        self.assertEqual(config["general"]["log_level"], "INFO")
        self.assertEqual(config["logging"]["log_dir"], "logs")
        self.assertEqual(config["logging"]["level"], "DEBUG")
    
    def test_load_config_with_nonexistent_file(self):
        """Test loading configuration with non-existent file."""
        config_file = os.path.join(self.temp_dir, "nonexistent.yaml")
        
        with self.assertRaises(FileNotFoundError):
            load_config(config_file)
    
    def test_load_config_with_invalid_yaml(self):
        """Test loading configuration with invalid YAML."""
        config_file = os.path.join(self.temp_dir, "invalid.yaml")
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with self.assertRaises(yaml.YAMLError):
            load_config(config_file)
    
    def test_load_config_with_default(self):
        """Test loading configuration with default values."""
        # Test with None (should use default config)
        with patch('utils.config_loader.os.path.exists', return_value=True):
            with patch('builtins.open', unittest.mock.mock_open(read_data='')):
                config = load_config(None)
                
                # Should have default structure
                self.assertIn("general", config)
                self.assertIn("logging", config)
                self.assertIn("preprocessing", config)
                self.assertIn("object_detection", config)
    
    def test_load_config_with_environment_variable(self):
        """Test loading configuration with environment variable."""
        config_data = {
            "general": {
                "output_dir": "env_output",
                "log_level": "DEBUG"
            }
        }
        
        config_file = os.path.join(self.temp_dir, "env_config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        with patch.dict(os.environ, {'COOKIFY_CONFIG': config_file}):
            config = load_config()
            
            self.assertEqual(config["general"]["output_dir"], "env_output")
            self.assertEqual(config["general"]["log_level"], "DEBUG")
    
    def test_config_validation(self):
        """Test basic configuration validation."""
        config_data = {
            "general": {
                "output_dir": "test_output",
                "log_level": "INFO"
            },
            "logging": {
                "log_dir": "logs",
                "level": "DEBUG"
            },
            "preprocessing": {
                "frame_rate": 1,
                "high_res_frames": True
            },
            "object_detection": {
                "confidence_threshold": 0.25,
                "filter_cooking_objects": True
            }
        }
        
        config_file = os.path.join(self.temp_dir, "valid_config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(config_file)
        
        # Validate required sections exist
        required_sections = ["general", "logging", "preprocessing", "object_detection"]
        for section in required_sections:
            self.assertIn(section, config, f"Missing required section: {section}")
        
        # Validate log level is valid
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self.assertIn(config["general"]["log_level"], valid_log_levels)
        self.assertIn(config["logging"]["level"], valid_log_levels)
        
        # Validate numeric values
        self.assertIsInstance(config["preprocessing"]["frame_rate"], int)
        self.assertIsInstance(config["object_detection"]["confidence_threshold"], float)
        self.assertGreater(config["object_detection"]["confidence_threshold"], 0)
        self.assertLessEqual(config["object_detection"]["confidence_threshold"], 1)


if __name__ == '__main__':
    unittest.main()
