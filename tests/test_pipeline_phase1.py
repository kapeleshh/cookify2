"""
Unit tests for Phase 1 improvements in Cookify pipeline
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import Pipeline, PipelineError, VideoProcessingError, AudioProcessingError, RecipeExtractionError
from utils.logger import CookifyLogger


class TestPipelinePhase1(unittest.TestCase):
    """Test cases for Phase 1 pipeline improvements."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "general": {
                "output_dir": self.temp_dir,
                "temp_dir": self.temp_dir,
                "log_level": "INFO"
            },
            "logging": {
                "log_dir": self.temp_dir,
                "level": "INFO"
            },
            "preprocessing": {
                "frame_rate": 1,
                "high_res_frames": True,
                "audio_quality": 0
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
                "timestamps": True
            }
        }
        
        # Create a mock video file
        self.test_video_path = os.path.join(self.temp_dir, "test_video.mp4")
        with open(self.test_video_path, 'w') as f:
            f.write("fake video content")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with proper error handling."""
        with patch('pipeline.load_config', return_value=self.config):
            with patch('pipeline.VideoProcessor') as mock_vp:
                with patch('pipeline.ObjectDetector') as mock_od:
                    with patch('pipeline.SceneDetector') as mock_sd:
                        pipeline = Pipeline()
                        
                        # Verify components are initialized
                        self.assertIsNotNone(pipeline.video_processor)
                        self.assertIsNotNone(pipeline.object_detector)
                        self.assertIsNotNone(pipeline.scene_detector)
                        self.assertIsNotNone(pipeline.cookify_logger)
    
    def test_input_validation_valid_file(self):
        """Test input validation with a valid video file."""
        with patch('pipeline.load_config', return_value=self.config):
            with patch('pipeline.VideoProcessor'):
                with patch('pipeline.ObjectDetector'):
                    with patch('pipeline.SceneDetector'):
                        pipeline = Pipeline()
                        
                        # Should not raise an exception
                        pipeline._validate_input(self.test_video_path)
    
    def test_input_validation_empty_path(self):
        """Test input validation with empty path."""
        with patch('pipeline.load_config', return_value=self.config):
            with patch('pipeline.VideoProcessor'):
                with patch('pipeline.ObjectDetector'):
                    with patch('pipeline.SceneDetector'):
                        pipeline = Pipeline()
                        
                        with self.assertRaises(VideoProcessingError):
                            pipeline._validate_input("")
    
    def test_input_validation_nonexistent_file(self):
        """Test input validation with non-existent file."""
        with patch('pipeline.load_config', return_value=self.config):
            with patch('pipeline.VideoProcessor'):
                with patch('pipeline.ObjectDetector'):
                    with patch('pipeline.SceneDetector'):
                        pipeline = Pipeline()
                        
                        with self.assertRaises(VideoProcessingError):
                            pipeline._validate_input("nonexistent.mp4")
    
    def test_input_validation_empty_file(self):
        """Test input validation with empty file."""
        empty_file = os.path.join(self.temp_dir, "empty.mp4")
        with open(empty_file, 'w') as f:
            pass  # Create empty file
        
        with patch('pipeline.load_config', return_value=self.config):
            with patch('pipeline.VideoProcessor'):
                with patch('pipeline.ObjectDetector'):
                    with patch('pipeline.SceneDetector'):
                        pipeline = Pipeline()
                        
                        with self.assertRaises(VideoProcessingError):
                            pipeline._validate_input(empty_file)
    
    def test_input_validation_invalid_extension(self):
        """Test input validation with invalid file extension."""
        invalid_file = os.path.join(self.temp_dir, "test.txt")
        with open(invalid_file, 'w') as f:
            f.write("not a video")
        
        with patch('pipeline.load_config', return_value=self.config):
            with patch('pipeline.VideoProcessor'):
                with patch('pipeline.ObjectDetector'):
                    with patch('pipeline.SceneDetector'):
                        pipeline = Pipeline()
                        
                        # Should not raise exception, just log warning
                        pipeline._validate_input(invalid_file)
    
    def test_recipe_extraction_with_valid_data(self):
        """Test recipe extraction with valid data."""
        with patch('pipeline.load_config', return_value=self.config):
            with patch('pipeline.VideoProcessor'):
                with patch('pipeline.ObjectDetector'):
                    with patch('pipeline.SceneDetector'):
                        pipeline = Pipeline()
                        
                        # Mock integrated data
                        integrated_data = {
                            "object_detections": [
                                {
                                    "frame_idx": 0,
                                    "detections": [
                                        {"class": "apple", "confidence": 0.9},
                                        {"class": "knife", "confidence": 0.8}
                                    ]
                                }
                            ],
                            "scenes": [
                                {"start": 0, "end": 10, "type": "cooking"},
                                {"start": 10, "end": 20, "type": "cooking"}
                            ],
                            "transcription": {
                                "text": "First, chop the apple with a knife",
                                "segments": []
                            }
                        }
                        
                        metadata = {"duration": 20, "fps": 25}
                        
                        recipe = pipeline._extract_recipe_from_data(integrated_data, metadata)
                        
                        # Verify recipe structure
                        self.assertIn("title", recipe)
                        self.assertIn("servings", recipe)
                        self.assertIn("ingredients", recipe)
                        self.assertIn("tools", recipe)
                        self.assertIn("steps", recipe)
                        
                        # Verify ingredients were extracted
                        self.assertGreater(len(recipe["ingredients"]), 0)
    
    def test_recipe_extraction_with_no_data(self):
        """Test recipe extraction with no data."""
        with patch('pipeline.load_config', return_value=self.config):
            with patch('pipeline.VideoProcessor'):
                with patch('pipeline.ObjectDetector'):
                    with patch('pipeline.SceneDetector'):
                        pipeline = Pipeline()
                        
                        # Mock empty integrated data
                        integrated_data = {
                            "object_detections": [],
                            "scenes": [],
                            "transcription": {}
                        }
                        
                        metadata = {"duration": 20, "fps": 25}
                        
                        with self.assertRaises(RecipeExtractionError):
                            pipeline._extract_recipe_from_data(integrated_data, metadata)
    
    def test_recipe_extraction_with_partial_data(self):
        """Test recipe extraction with partial data (graceful degradation)."""
        with patch('pipeline.load_config', return_value=self.config):
            with patch('pipeline.VideoProcessor'):
                with patch('pipeline.ObjectDetector'):
                    with patch('pipeline.SceneDetector'):
                        pipeline = Pipeline()
                        
                        # Mock partial integrated data
                        integrated_data = {
                            "object_detections": [
                                {
                                    "frame_idx": 0,
                                    "detections": [
                                        {"class": "apple", "confidence": 0.9}
                                    ]
                                }
                            ],
                            "scenes": [],
                            "transcription": {}
                        }
                        
                        metadata = {"duration": 20, "fps": 25}
                        
                        recipe = pipeline._extract_recipe_from_data(integrated_data, metadata)
                        
                        # Should still create a recipe with available data
                        self.assertIn("title", recipe)
                        self.assertIn("ingredients", recipe)
                        self.assertGreater(len(recipe["ingredients"]), 0)
    
    def test_error_handling_in_recipe_extraction(self):
        """Test error handling in recipe extraction."""
        with patch('pipeline.load_config', return_value=self.config):
            with patch('pipeline.VideoProcessor'):
                with patch('pipeline.ObjectDetector'):
                    with patch('pipeline.SceneDetector'):
                        pipeline = Pipeline()
                        
                        # Mock data that will cause an error in ingredient extraction
                        integrated_data = {
                            "object_detections": "invalid_data",  # This should cause an error
                            "scenes": [],
                            "transcription": {}
                        }
                        
                        metadata = {"duration": 20, "fps": 25}
                        
                        # Should handle error gracefully and return empty ingredients
                        recipe = pipeline._extract_recipe_from_data(integrated_data, metadata)
                        self.assertEqual(recipe["ingredients"], [])
    
    def test_processing_metadata_inclusion(self):
        """Test that processing metadata is included in output."""
        with patch('pipeline.load_config', return_value=self.config):
            with patch('pipeline.VideoProcessor') as mock_vp:
                with patch('pipeline.ObjectDetector') as mock_od:
                    with patch('pipeline.SceneDetector') as mock_sd:
                        # Mock the process method to return test data
                        mock_vp.return_value.process.return_value = (
                            [np.zeros((100, 100, 3))],  # frames
                            "audio.wav",  # audio_path
                            {"duration": 20, "fps": 25}  # metadata
                        )
                        mock_od.return_value.detect.return_value = []
                        mock_sd.return_value.detect.return_value = []
                        
                        pipeline = Pipeline()
                        
                        # Mock the recipe extraction to return test recipe
                        with patch.object(pipeline, '_extract_recipe_from_data') as mock_extract:
                            mock_extract.return_value = {
                                "title": "Test Recipe",
                                "servings": 4,
                                "ingredients": [],
                                "tools": [],
                                "steps": []
                            }
                            
                            result = pipeline.process(self.test_video_path)
                            
                            # Verify processing metadata is included
                            self.assertIn("_processing_metadata", result)
                            metadata = result["_processing_metadata"]
                            self.assertIn("processing_errors", metadata)
                            self.assertIn("processing_warnings", metadata)
                            self.assertIn("components_used", metadata)


class TestCookifyLogger(unittest.TestCase):
    """Test cases for CookifyLogger."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = CookifyLogger("test_logger", self.temp_dir, "INFO")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        self.assertEqual(self.logger.name, "test_logger")
        self.assertEqual(self.logger.log_dir, Path(self.temp_dir))
        self.assertIsNotNone(self.logger.logger)
    
    def test_timer_functionality(self):
        """Test timer start/end functionality."""
        import time
        
        self.logger.start_timer("test_operation")
        time.sleep(0.1)  # Sleep for 100ms
        duration = self.logger.end_timer("test_operation")
        
        self.assertGreater(duration, 0.05)  # Should be at least 50ms
        self.assertLess(duration, 0.2)  # Should be less than 200ms
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        self.logger.start_timer("op1")
        self.logger.end_timer("op1")
        
        self.logger.start_timer("op2")
        self.logger.end_timer("op2")
        
        summary = self.logger.log_performance_summary()
        
        self.assertIn("op1", summary)
        self.assertIn("op2", summary)
        self.assertEqual(summary["op1"]["count"], 1)
        self.assertEqual(summary["op2"]["count"], 1)
    
    def test_processing_step_logging(self):
        """Test processing step logging."""
        # This should not raise an exception
        self.logger.log_processing_step("test_step", {"status": "success"})
    
    def test_error_logging_with_context(self):
        """Test error logging with context."""
        # This should not raise an exception
        try:
            raise ValueError("Test error")
        except ValueError as e:
            self.logger.log_error_with_context(e, {"step": "test", "data": "test_data"})
    
    def test_recipe_extraction_logging(self):
        """Test recipe extraction logging."""
        recipe_data = {
            "title": "Test Recipe",
            "ingredients": [{"name": "apple", "quantity": "1"}],
            "tools": [{"name": "knife"}],
            "steps": [{"description": "Chop apple"}],
            "servings": 4
        }
        
        # This should not raise an exception
        self.logger.log_recipe_extraction(recipe_data)


if __name__ == '__main__':
    unittest.main()
