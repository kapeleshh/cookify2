"""
Config Loader - Load configuration from a YAML file
"""

import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path=None):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str, optional): Path to the configuration file. Defaults to None.
        
    Returns:
        dict: Configuration dictionary.
    """
    # Default configuration
    default_config = {
        "general": {
            "output_dir": "data/output",
            "temp_dir": "data/temp",
            "log_level": "INFO",
            "save_intermediate": True
        },
        "logging": {
            "log_dir": "logs",
            "level": "INFO",
            "enable_performance_logging": True,
            "enable_debug_logging": True,
            "max_file_size": 10,
            "backup_count": 5
        },
        "preprocessing": {
            "frame_rate": 1,
            "high_res_frames": True,
            "audio_quality": 0,
            "use_scene_detection": True
        },
        "object_detection": {
            "confidence_threshold": 0.25,
            "filter_cooking_objects": True,
            "max_objects": 20,
            "model": "yolov8x"
        },
        "scene_detection": {
            "threshold": 30.0,
            "min_scene_len": 15
        },
        "transcription": {
            "model": "base",
            "language": None,
            "timestamps": True,
            "translate": False
        }
    }
    
    # If no config path specified, look in default locations
    if config_path is None:
        # Try common config locations
        possible_paths = [
            "config.yaml",
            "config.yml",
            "configs/config.yaml",
            os.path.join(os.path.dirname(__file__), "../../config.yaml"),
            os.path.expanduser("~/.config/cookify/config.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                logger.info(f"Found configuration at {path}")
                break
    
    # If config path exists, load it
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                
                # Update default config with user config
                update_nested_dict(default_config, user_config)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.warning("Using default configuration")
    else:
        if config_path:
            logger.warning(f"Configuration file not found: {config_path}")
        logger.info("Using default configuration")
    
    return default_config

def update_nested_dict(d, u):
    """Update a nested dictionary with another nested dictionary."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d
