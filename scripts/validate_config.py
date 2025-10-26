#!/usr/bin/env python3
"""
Configuration validation utility for Cookify
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config_loader import validate_config_file, create_sample_config, load_config

def main():
    """Main function for configuration validation."""
    parser = argparse.ArgumentParser(description="Validate Cookify configuration files")
    parser.add_argument("config_path", nargs="?", help="Path to configuration file to validate")
    parser.add_argument("--create-sample", action="store_true", help="Create a sample configuration file")
    parser.add_argument("--output", "-o", help="Output path for sample configuration")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.create_sample:
        output_path = args.output or "config_sample.yaml"
        try:
            create_sample_config(output_path)
            print(f"Sample configuration created: {output_path}")
            return 0
        except Exception as e:
            print(f"Error creating sample configuration: {e}")
            return 1
    
    if not args.config_path:
        # Try to find config.yaml in common locations
        possible_paths = [
            "config.yaml",
            "cookify/config.yaml",
            os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        ]
        
        config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if not config_path:
            print("No configuration file specified and none found in common locations")
            print("Use --create-sample to create a sample configuration file")
            return 1
    else:
        config_path = args.config_path
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return 1
    
    print(f"Validating configuration file: {config_path}")
    
    try:
        results = validate_config_file(config_path)
        
        if results["valid"]:
            print("✅ Configuration is valid")
            if args.verbose and results["warnings"]:
                print("\nWarnings:")
                for warning in results["warnings"]:
                    print(f"  ⚠️  {warning}")
            return 0
        else:
            print("❌ Configuration is invalid")
            print("\nErrors:")
            for error in results["errors"]:
                print(f"  ❌ {error}")
            return 1
            
    except Exception as e:
        print(f"Error validating configuration: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
