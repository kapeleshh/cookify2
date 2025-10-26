"""
Integration tests for Phase 2 of the Cookify project.
"""

import os
import sys
import unittest
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.video_processor import VideoProcessor
from src.frame_analysis.object_detector import ObjectDetector
from src.frame_analysis.scene_detector import SceneDetector, DetectorType
from src.frame_analysis.text_recognizer import TextRecognizer
from src.frame_analysis.action_recognizer import ActionRecognizer
from src.audio_analysis.speech_recognizer import SpeechRecognizer
from src.audio_analysis.nlp_processor import NLPProcessor
from src.integration.multimodal_integrator import MultimodalIntegrator
from src.recipe_extraction.recipe_extractor import RecipeExtractor
from src.output_formatting.formatter import OutputFormatter
from src.pipeline import Pipeline
from src.utils.model_downloader import download_models

# Test data paths
TEST_VIDEO_PATH = Path(__file__).parent / "data" / "test_cooking_video.mp4"
TEST_AUDIO_PATH = Path(__file__).parent / "data" / "test_cooking_audio.wav"
TEST_TRANSCRIPT_PATH = Path(__file__).parent / "data" / "test_transcript.json"
TEST_OUTPUT_PATH = Path(__file__).parent / "data" / "test_output.json"

class TestPhase2Integration(unittest.TestCase):
    """
    Integration tests for Phase 2 of the Cookify project.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test environment.
        """
        # Create test data directory if it doesn't exist
        os.makedirs(Path(__file__).parent / "data", exist_ok=True)
        
        # Download models if needed
        if not os.path.exists(Path(__file__).parent.parent / "models" / "yolov8" / "yolov8x.pt"):
            print("Downloading models...")
            download_models()
        
        # Initialize components
        cls.video_processor = VideoProcessor()
        cls.object_detector = ObjectDetector()
        cls.scene_detector = SceneDetector(detector_type=DetectorType.COMBINED)
        cls.text_recognizer = TextRecognizer()
        cls.action_recognizer = ActionRecognizer()
        cls.speech_recognizer = SpeechRecognizer()
        cls.nlp_processor = NLPProcessor()
        cls.multimodal_integrator = MultimodalIntegrator()
        cls.recipe_extractor = RecipeExtractor()
        cls.output_formatter = OutputFormatter()
        cls.pipeline = Pipeline()
    
    def setUp(self):
        """
        Set up before each test.
        """
        # Skip tests if test video doesn't exist
        if not os.path.exists(TEST_VIDEO_PATH):
            self.skipTest(f"Test video not found: {TEST_VIDEO_PATH}")
    
    def test_video_preprocessing(self):
        """
        Test video preprocessing.
        """
        # Process video
        frames, audio = self.video_processor.process_video(TEST_VIDEO_PATH)
        
        # Check if frames and audio are extracted
        self.assertIsNotNone(frames)
        self.assertIsNotNone(audio)
        self.assertGreater(len(frames), 0)
        self.assertGreater(len(audio), 0)
        
        # Check frame shape
        self.assertEqual(len(frames[0].shape), 3)  # Height, width, channels
        self.assertEqual(frames[0].shape[2], 3)    # RGB channels
    
    def test_object_detection(self):
        """
        Test object detection.
        """
        # Process video
        frames, _ = self.video_processor.process_video(TEST_VIDEO_PATH)
        
        # Sample frames for testing
        sample_frames = frames[::30]  # Every 30th frame
        
        # Detect objects
        detections = self.object_detector.detect(sample_frames)
        
        # Check if detections are returned
        self.assertIsNotNone(detections)
        self.assertEqual(len(detections), len(sample_frames))
        
        # Check if detections have the expected format
        for detection in detections:
            self.assertIn("frame_idx", detection)
            self.assertIn("detections", detection)
    
    def test_scene_detection(self):
        """
        Test scene detection.
        """
        # Detect scenes
        scenes = self.scene_detector.detect(str(TEST_VIDEO_PATH))
        
        # Check if scenes are returned
        self.assertIsNotNone(scenes)
        self.assertGreater(len(scenes), 0)
        
        # Check if scenes have the expected format
        for scene in scenes:
            self.assertIn("scene_idx", scene)
            self.assertIn("start_frame", scene)
            self.assertIn("end_frame", scene)
            self.assertIn("start_time", scene)
            self.assertIn("end_time", scene)
            self.assertIn("duration", scene)
    
    def test_text_recognition(self):
        """
        Test text recognition.
        """
        # Process video
        frames, _ = self.video_processor.process_video(TEST_VIDEO_PATH)
        
        # Sample frames for testing
        sample_frames = frames[::30]  # Every 30th frame
        
        # Recognize text
        text_results = self.text_recognizer.recognize(sample_frames)
        
        # Check if text results are returned
        self.assertIsNotNone(text_results)
        self.assertEqual(len(text_results), len(sample_frames))
        
        # Check if text results have the expected format
        for result in text_results:
            self.assertIn("frame_idx", result)
            self.assertIn("detections", result)
    
    def test_action_recognition(self):
        """
        Test action recognition.
        """
        # Process video
        frames, _ = self.video_processor.process_video(TEST_VIDEO_PATH)
        
        # Detect scenes
        scenes = self.scene_detector.detect(str(TEST_VIDEO_PATH))
        
        # Recognize actions
        actions = self.action_recognizer.recognize(frames, scenes)
        
        # Check if actions are returned
        self.assertIsNotNone(actions)
        
        # Check if actions have the expected format
        for action in actions:
            self.assertIn("start_frame", action)
            self.assertIn("end_frame", action)
            self.assertIn("start_time", action)
            self.assertIn("end_time", action)
            self.assertIn("action", action)
            self.assertIn("category", action)
            self.assertIn("confidence", action)
    
    def test_speech_recognition(self):
        """
        Test speech recognition.
        """
        # Skip test if audio file doesn't exist
        if not os.path.exists(TEST_AUDIO_PATH):
            self.skipTest(f"Test audio not found: {TEST_AUDIO_PATH}")
        
        # Transcribe audio
        transcription = self.speech_recognizer.transcribe(str(TEST_AUDIO_PATH))
        
        # Check if transcription is returned
        self.assertIsNotNone(transcription)
        self.assertIn("text", transcription)
        self.assertIn("segments", transcription)
        
        # Save transcription for other tests
        with open(TEST_TRANSCRIPT_PATH, "w") as f:
            json.dump(transcription, f)
    
    def test_nlp_processing(self):
        """
        Test NLP processing.
        """
        # Skip test if transcript file doesn't exist
        if not os.path.exists(TEST_TRANSCRIPT_PATH):
            self.skipTest(f"Test transcript not found: {TEST_TRANSCRIPT_PATH}")
        
        # Load transcription
        with open(TEST_TRANSCRIPT_PATH, "r") as f:
            transcription = json.load(f)
        
        # Process transcription
        recipe = self.nlp_processor.process(transcription)
        
        # Check if recipe is returned
        self.assertIsNotNone(recipe)
        self.assertIn("title", recipe)
        self.assertIn("ingredients", recipe)
        self.assertIn("steps", recipe)
    
    def test_multimodal_integration(self):
        """
        Test multimodal integration.
        """
        # Process video
        frames, audio = self.video_processor.process_video(TEST_VIDEO_PATH)
        
        # Sample frames for testing
        sample_frames = frames[::30]  # Every 30th frame
        
        # Detect objects
        object_detections = self.object_detector.detect(sample_frames)
        
        # Detect scenes
        scenes = self.scene_detector.detect(str(TEST_VIDEO_PATH))
        
        # Recognize text
        text_results = self.text_recognizer.recognize(sample_frames)
        
        # Recognize actions
        actions = self.action_recognizer.recognize(frames, scenes)
        
        # Skip test if transcript file doesn't exist
        if not os.path.exists(TEST_TRANSCRIPT_PATH):
            self.skipTest(f"Test transcript not found: {TEST_TRANSCRIPT_PATH}")
        
        # Load transcription
        with open(TEST_TRANSCRIPT_PATH, "r") as f:
            transcription = json.load(f)
        
        # Process transcription
        recipe = self.nlp_processor.process(transcription)
        
        # Integrate results
        integrated_data = self.multimodal_integrator.integrate(
            object_detections=object_detections,
            scenes=scenes,
            text_results=text_results,
            actions=actions,
            transcription=transcription,
            recipe=recipe
        )
        
        # Check if integrated data is returned
        self.assertIsNotNone(integrated_data)
        self.assertIn("objects", integrated_data)
        self.assertIn("scenes", integrated_data)
        self.assertIn("text", integrated_data)
        self.assertIn("actions", integrated_data)
        self.assertIn("transcription", integrated_data)
        self.assertIn("recipe", integrated_data)
    
    def test_recipe_extraction(self):
        """
        Test recipe extraction.
        """
        # Process video
        frames, audio = self.video_processor.process_video(TEST_VIDEO_PATH)
        
        # Sample frames for testing
        sample_frames = frames[::30]  # Every 30th frame
        
        # Detect objects
        object_detections = self.object_detector.detect(sample_frames)
        
        # Detect scenes
        scenes = self.scene_detector.detect(str(TEST_VIDEO_PATH))
        
        # Recognize text
        text_results = self.text_recognizer.recognize(sample_frames)
        
        # Recognize actions
        actions = self.action_recognizer.recognize(frames, scenes)
        
        # Skip test if transcript file doesn't exist
        if not os.path.exists(TEST_TRANSCRIPT_PATH):
            self.skipTest(f"Test transcript not found: {TEST_TRANSCRIPT_PATH}")
        
        # Load transcription
        with open(TEST_TRANSCRIPT_PATH, "r") as f:
            transcription = json.load(f)
        
        # Process transcription
        recipe = self.nlp_processor.process(transcription)
        
        # Integrate results
        integrated_data = self.multimodal_integrator.integrate(
            object_detections=object_detections,
            scenes=scenes,
            text_results=text_results,
            actions=actions,
            transcription=transcription,
            recipe=recipe
        )
        
        # Extract recipe
        extracted_recipe = self.recipe_extractor.extract(integrated_data)
        
        # Check if extracted recipe is returned
        self.assertIsNotNone(extracted_recipe)
        self.assertIn("title", extracted_recipe)
        self.assertIn("servings", extracted_recipe)
        self.assertIn("ingredients", extracted_recipe)
        self.assertIn("steps", extracted_recipe)
        self.assertIn("tools", extracted_recipe)
    
    def test_output_formatting(self):
        """
        Test output formatting.
        """
        # Process video
        frames, audio = self.video_processor.process_video(TEST_VIDEO_PATH)
        
        # Sample frames for testing
        sample_frames = frames[::30]  # Every 30th frame
        
        # Detect objects
        object_detections = self.object_detector.detect(sample_frames)
        
        # Detect scenes
        scenes = self.scene_detector.detect(str(TEST_VIDEO_PATH))
        
        # Recognize text
        text_results = self.text_recognizer.recognize(sample_frames)
        
        # Recognize actions
        actions = self.action_recognizer.recognize(frames, scenes)
        
        # Skip test if transcript file doesn't exist
        if not os.path.exists(TEST_TRANSCRIPT_PATH):
            self.skipTest(f"Test transcript not found: {TEST_TRANSCRIPT_PATH}")
        
        # Load transcription
        with open(TEST_TRANSCRIPT_PATH, "r") as f:
            transcription = json.load(f)
        
        # Process transcription
        recipe = self.nlp_processor.process(transcription)
        
        # Integrate results
        integrated_data = self.multimodal_integrator.integrate(
            object_detections=object_detections,
            scenes=scenes,
            text_results=text_results,
            actions=actions,
            transcription=transcription,
            recipe=recipe
        )
        
        # Extract recipe
        extracted_recipe = self.recipe_extractor.extract(integrated_data)
        
        # Format output
        formatted_output = self.output_formatter.format(extracted_recipe, format="json")
        
        # Check if formatted output is returned
        self.assertIsNotNone(formatted_output)
        
        # Save output for inspection
        with open(TEST_OUTPUT_PATH, "w") as f:
            f.write(formatted_output)
    
    def test_full_pipeline(self):
        """
        Test the full pipeline.
        """
        # Run the pipeline
        result = self.pipeline.process_video(str(TEST_VIDEO_PATH))
        
        # Check if result is returned
        self.assertIsNotNone(result)
        self.assertIn("title", result)
        self.assertIn("servings", result)
        self.assertIn("ingredients", result)
        self.assertIn("steps", result)
        self.assertIn("tools", result)

if __name__ == "__main__":
    unittest.main()
