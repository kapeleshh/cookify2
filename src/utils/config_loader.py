"""
Configuration Loader - Utility to load and validate configuration
"""

import os
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "general": {
        "output_dir": "cookify/data/output",
        "temp_dir": "cookify/data/temp",
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
    "text_recognition": {
        "confidence_threshold": 0.5,
        "languages": ["en"],
        "enhance_text": True
    },
    "action_recognition": {
        "confidence_threshold": 0.5,
        "frame_window": 16,
        "use_optical_flow": False
    },
    "transcription": {
        "model": "base",
        "language": None,
        "timestamps": True,
        "translate": False
    },
    "nlp": {
        "model": "en_core_web_lg",
        "use_custom_ner": True,
        "entity_confidence": 0.7
    },
    "recipe_extraction": {
        "infer_missing_ingredients": True,
        "normalize_quantities": True,
        "group_similar_steps": False,
        "min_confidence": 0.6
    },
    "output_formatting": {
        "format": "json",
        "include_confidence": False,
        "include_timestamps": True,
        "include_frame_refs": False
    }
}

def load_config(config_path=None):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str, optional): Path to the configuration file. Defaults to None.
        
    Returns:
        dict: Configuration dictionary.
    """
    config = DEFAULT_CONFIG.copy()
    
    # If no config path is provided, look for config.yaml in standard locations
    if config_path is None:
        # Look in current directory
        if os.path.exists("config.yaml"):
            config_path = "config.yaml"
        # Look in the cookify directory
        elif os.path.exists("cookify/config.yaml"):
            config_path = "cookify/config.yaml"
        # Look relative to this file
        else:
            base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            potential_path = base_dir / "config.yaml"
            if os.path.exists(potential_path):
                config_path = potential_path
    
    # Load configuration from file if it exists
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                
            # Update configuration with values from file
            if file_config:
                _update_config_recursive(config, file_config)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.warning("Using default configuration")
    else:
        logger.warning(f"Configuration file not found: {config_path}")
        logger.warning("Using default configuration")
    
    # Validate configuration
    _validate_config(config)
    
    # Create directories if they don't exist
    os.makedirs(config["general"]["output_dir"], exist_ok=True)
    os.makedirs(config["general"]["temp_dir"], exist_ok=True)
    
    return config

def _update_config_recursive(base_config, new_config):
    """
    Recursively update a nested configuration dictionary.
    
    Args:
        base_config (dict): Base configuration to update.
        new_config (dict): New configuration values.
    """
    for key, value in new_config.items():
        if key in base_config:
            if isinstance(value, dict) and isinstance(base_config[key], dict):
                _update_config_recursive(base_config[key], value)
            else:
                base_config[key] = value
        else:
            logger.warning(f"Unknown configuration key: {key}")

def _validate_config(config):
    """
    Validate configuration values with comprehensive checks.
    
    Args:
        config (dict): Configuration dictionary.
    """
    validation_errors = []
    validation_warnings = []
    
    # Validate log level
    log_level = config["general"]["log_level"].upper()
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_log_levels:
        validation_warnings.append(f"Invalid log level: {log_level}. Using INFO.")
        config["general"]["log_level"] = "INFO"
    
    # Validate logging configuration
    if "logging" in config:
        logging_config = config["logging"]
        if "level" in logging_config:
            log_level = logging_config["level"].upper()
            if log_level not in valid_log_levels:
                validation_warnings.append(f"Invalid logging level: {log_level}. Using INFO.")
                logging_config["level"] = "INFO"
        
        if "max_file_size" in logging_config:
            max_size = logging_config["max_file_size"]
            if not isinstance(max_size, (int, float)) or max_size <= 0:
                validation_warnings.append(f"Invalid max_file_size: {max_size}. Using 10.")
                logging_config["max_file_size"] = 10
        
        if "backup_count" in logging_config:
            backup_count = logging_config["backup_count"]
            if not isinstance(backup_count, int) or backup_count < 0:
                validation_warnings.append(f"Invalid backup_count: {backup_count}. Using 5.")
                logging_config["backup_count"] = 5
    
    # Validate confidence thresholds
    for section in ["object_detection", "text_recognition", "action_recognition"]:
        if section in config:
            threshold = config[section].get("confidence_threshold")
            if threshold is not None:
                if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
                    validation_warnings.append(f"Invalid confidence threshold in {section}: {threshold}. Using default.")
                    config[section]["confidence_threshold"] = DEFAULT_CONFIG[section]["confidence_threshold"]
    
    # Validate frame rate
    frame_rate = config["preprocessing"]["frame_rate"]
    if not isinstance(frame_rate, (int, float)) or frame_rate <= 0:
        validation_warnings.append(f"Invalid frame rate: {frame_rate}. Using default.")
        config["preprocessing"]["frame_rate"] = DEFAULT_CONFIG["preprocessing"]["frame_rate"]
    
    # Validate Whisper model
    whisper_model = config["transcription"]["model"]
    valid_whisper_models = ["tiny", "base", "small", "medium", "large"]
    if whisper_model not in valid_whisper_models:
        validation_warnings.append(f"Invalid Whisper model: {whisper_model}. Using base.")
        config["transcription"]["model"] = "base"
    
    # Validate output format
    output_format = config["output_formatting"]["format"]
    valid_formats = ["json", "yaml", "markdown"]
    if output_format not in valid_formats:
        validation_warnings.append(f"Invalid output format: {output_format}. Using json.")
        config["output_formatting"]["format"] = "json"
    
    # Validate paths exist and are writable
    output_dir = config["general"]["output_dir"]
    temp_dir = config["general"]["temp_dir"]
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            validation_errors.append(f"Output directory is not writable: {output_dir}")
    except Exception as e:
        validation_errors.append(f"Cannot create output directory {output_dir}: {e}")
    
    try:
        os.makedirs(temp_dir, exist_ok=True)
        if not os.access(temp_dir, os.W_OK):
            validation_errors.append(f"Temp directory is not writable: {temp_dir}")
    except Exception as e:
        validation_errors.append(f"Cannot create temp directory {temp_dir}: {e}")
    
    # Log validation results
    if validation_errors:
        logger.error("Configuration validation errors:")
        for error in validation_errors:
            logger.error(f"  - {error}")
        raise ValueError("Configuration validation failed")
    
    if validation_warnings:
        logger.warning("Configuration validation warnings:")
        for warning in validation_warnings:
            logger.warning(f"  - {warning}")
    
    logger.info("Configuration validation completed successfully")

def get_config_value(config, path, default=None):
    """
    Get a value from the configuration using a dot-separated path.
    
    Args:
        config (dict): Configuration dictionary.
        path (str): Dot-separated path to the configuration value.
        default: Default value to return if the path doesn't exist.
        
    Returns:
        The configuration value, or the default if not found.
    """
    keys = path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

def validate_config_file(config_path):
    """
    Validate a configuration file without loading it into the pipeline.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: Validation results with errors and warnings.
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "config": None
    }
    
    try:
        # Load and validate the configuration
        config = load_config(config_path)
        results["config"] = config
        logger.info(f"Configuration file {config_path} is valid")
    except Exception as e:
        results["valid"] = False
        results["errors"].append(str(e))
        logger.error(f"Configuration file {config_path} is invalid: {e}")
    
    return results

def create_sample_config(output_path="config_sample.yaml"):
    """
    Create a sample configuration file with all available options.
    
    Args:
        output_path (str): Path where to save the sample configuration.
    """
    sample_config = DEFAULT_CONFIG.copy()
    
    # Add comments and descriptions
    sample_config_with_comments = {
        "# Cookify Configuration File": None,
        "# This is a sample configuration file with all available options": None,
        "": None,
        "general": sample_config["general"],
        "logging": sample_config["logging"],
        "preprocessing": sample_config["preprocessing"],
        "object_detection": sample_config["object_detection"],
        "scene_detection": sample_config["scene_detection"],
        "text_recognition": sample_config["text_recognition"],
        "action_recognition": sample_config["action_recognition"],
        "transcription": sample_config["transcription"],
        "nlp": sample_config["nlp"],
        "recipe_extraction": sample_config["recipe_extraction"],
        "output_formatting": sample_config["output_formatting"]
    }
    
    try:
        with open(output_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False, indent=2)
        logger.info(f"Sample configuration created: {output_path}")
    except Exception as e:
        logger.error(f"Failed to create sample configuration: {e}")
        raise
