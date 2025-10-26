#!/usr/bin/env python3
"""
Test script for Phase 1 of the Cookify project.
This script tests the basic functionality of the project structure and components.
"""

import os
import sys
import unittest
import importlib.util
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestPhase1(unittest.TestCase):
    """Test cases for Phase 1 of the Cookify project."""

    def test_project_structure(self):
        """Test that the project structure is correctly set up."""
        # Check main directories
        directories = [
            "src",
            "src/utils",
            "src/preprocessing",
            "src/frame_analysis",
            "src/audio_analysis",
            "src/integration",
            "src/recipe_extraction",
            "src/output_formatting",
            "documentation",
            "examples"
        ]
        
        for directory in directories:
            self.assertTrue(os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), directory)),
                           f"Directory {directory} does not exist")
        
        # Check key files
        files = [
            "main.py",
            "setup.py",
            "requirements.txt",
            "config.yaml",
            "README.md"
        ]
        
        for file in files:
            self.assertTrue(os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(__file__)), file)),
                           f"File {file} does not exist")

    def test_module_imports(self):
        """Test that all modules can be imported without errors."""
        modules = [
            "src.utils.config_loader",
            "src.utils.model_downloader",
            "src.preprocessing.video_processor",
            "src.frame_analysis.object_detector",
            "src.frame_analysis.scene_detector",
            "src.frame_analysis.text_recognizer",
            "src.frame_analysis.action_recognizer",
            "src.audio_analysis.transcriber",
            "src.audio_analysis.nlp_processor",
            "src.integration.multimodal_integrator",
            "src.recipe_extraction.recipe_extractor",
            "src.output_formatting.formatter",
            "src.pipeline"
        ]
        
        for module in modules:
            try:
                importlib.import_module(module)
                logger.info(f"Successfully imported {module}")
            except ImportError as e:
                self.fail(f"Failed to import {module}: {e}")

    def test_config_loader(self):
        """Test that the configuration loader works correctly."""
        from src.utils.config_loader import load_config
        
        # Test with default config
        config = load_config()
        self.assertIsNotNone(config)
        self.assertIn("general", config)
        
        # Test with non-existent config
        with self.assertRaises(FileNotFoundError):
            load_config("non_existent_config.yaml")

    def test_pipeline_initialization(self):
        """Test that the pipeline can be initialized."""
        from src.pipeline import Pipeline
        
        try:
            pipeline = Pipeline()
            self.assertIsNotNone(pipeline)
            logger.info("Successfully initialized pipeline")
        except Exception as e:
            self.fail(f"Failed to initialize pipeline: {e}")

    def test_documentation(self):
        """Test that all documentation files exist."""
        doc_files = [
            "00_documentation_index.md",
            "01_architecture_overview.md",
            "02_preprocessing_phase.md",
            "03_frame_analysis_phase.md",
            "04_audio_analysis_phase.md",
            "05_multimodal_integration_phase.md",
            "06_recipe_extraction_phase.md",
            "07_output_formatting_phase.md",
            "08_future_directions.md"
        ]
        
        doc_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documentation")
        
        for file in doc_files:
            self.assertTrue(os.path.isfile(os.path.join(doc_dir, file)),
                           f"Documentation file {file} does not exist")

if __name__ == "__main__":
    unittest.main()
