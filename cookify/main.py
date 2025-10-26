#!/usr/bin/env python3
"""
Cookify - Recipe Extraction from Cooking Videos
Main entry point for the recipe extraction pipeline.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from pipeline import Pipeline
from utils.model_downloader import download_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cookify.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract structured recipe from cooking video")
    parser.add_argument("video_path", type=str, help="Path to the cooking video file")
    parser.add_argument("--output", "-o", type=str, default=None, 
                        help="Output file path for the extracted recipe (default: auto-generated)")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to configuration file (default: auto-detected)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--skip-download", action="store_true", 
                        help="Skip downloading models (use if models are already downloaded)")
    return parser.parse_args()

def ensure_models_downloaded():
    """Ensure all required models are downloaded."""
    logger.info("Checking for pre-trained models...")
    download_models()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure models are downloaded
    if not args.skip_download:
        ensure_models_downloaded()
    
    try:
        # Initialize pipeline
        logger.info("Initializing recipe extraction pipeline...")
        pipeline = Pipeline(config_path=args.config)
        
        # Process the video
        logger.info(f"Processing video: {args.video_path}")
        recipe = pipeline.process(args.video_path, args.output)
        
        # Print summary
        pipeline.print_summary(recipe)
        
        # Print output path
        if args.output:
            output_path = args.output
        else:
            video_name = Path(args.video_path).stem
            output_path = os.path.join(pipeline.config["general"]["output_dir"], f"{video_name}_recipe.json")
        
        print(f"Full recipe saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
