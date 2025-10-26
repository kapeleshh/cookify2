#!/usr/bin/env python3
"""
Example script for extracting a recipe from a cooking video.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import Pipeline

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract a recipe from a cooking video")
    parser.add_argument("video_path", type=str, help="Path to the cooking video file")
    parser.add_argument("--output", "-o", type=str, default=None, 
                        help="Output file path for the extracted recipe (default: auto-generated)")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to configuration file (default: auto-detected)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("cookify.log"),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Initialize pipeline
        print("Initializing recipe extraction pipeline...")
        pipeline = Pipeline(config_path=args.config)
        
        # Process the video
        print(f"Processing video: {args.video_path}")
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
        logging.error(f"Error: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
