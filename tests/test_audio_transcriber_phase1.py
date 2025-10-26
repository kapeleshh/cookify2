"""
Unit tests for Phase 1 improvements in AudioTranscriber
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_analysis.transcriber import AudioTranscriber, ModelLoadError, TranscriptionError


class TestAudioTranscriberPhase1(unittest.TestCase):
    """Test cases for Phase 1 AudioTranscriber improvements."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        
        # Create a test audio file
        with open(self.test_audio_path, 'w') as f:
            f.write("fake audio content")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_transcriber_initialization(self):
        """Test transcriber initialization."""
        transcriber = AudioTranscriber()
        self.assertEqual(transcriber.model_name, "base")
        self.assertIsNone(transcriber.language)
        self.assertTrue(transcriber.timestamps)
        self.assertFalse(transcriber.translate)
        self.assertIsNone(transcriber.model)
    
    def test_transcriber_initialization_with_params(self):
        """Test transcriber initialization with custom parameters."""
        transcriber = AudioTranscriber(
            model_name="small",
            language="en",
            timestamps=False,
            translate=True
        )
        self.assertEqual(transcriber.model_name, "small")
        self.assertEqual(transcriber.language, "en")
        self.assertFalse(transcriber.timestamps)
        self.assertTrue(transcriber.translate)
    
    def test_transcribe_with_string_path(self):
        """Test transcription with string path."""
        transcriber = AudioTranscriber()
        
        with patch.object(transcriber, '_load_model') as mock_load:
            with patch.object(transcriber, 'model') as mock_model:
                mock_model.transcribe.return_value = {
                    "text": "Test transcription",
                    "segments": [],
                    "language": "en"
                }
                
                result = transcriber.transcribe(self.test_audio_path)
                
                self.assertEqual(result["text"], "Test transcription")
                self.assertEqual(result["language"], "en")
                self.assertEqual(len(result["segments"]), 0)
    
    def test_transcribe_with_path_object(self):
        """Test transcription with Path object."""
        transcriber = AudioTranscriber()
        path_obj = Path(self.test_audio_path)
        
        with patch.object(transcriber, '_load_model') as mock_load:
            with patch.object(transcriber, 'model') as mock_model:
                mock_model.transcribe.return_value = {
                    "text": "Test transcription",
                    "segments": [],
                    "language": "en"
                }
                
                result = transcriber.transcribe(path_obj)
                
                self.assertEqual(result["text"], "Test transcription")
                # Verify that the Path object was converted to string
                mock_model.transcribe.assert_called_once()
                call_args = mock_model.transcribe.call_args[0]
                self.assertEqual(call_args[0], str(path_obj))
    
    def test_transcribe_empty_path(self):
        """Test transcription with empty path."""
        transcriber = AudioTranscriber()
        
        with self.assertRaises(TranscriptionError):
            transcriber.transcribe("")
    
    def test_transcribe_nonexistent_file(self):
        """Test transcription with non-existent file."""
        transcriber = AudioTranscriber()
        
        with self.assertRaises(TranscriptionError):
            transcriber.transcribe("nonexistent.wav")
    
    def test_transcribe_empty_file(self):
        """Test transcription with empty file."""
        empty_file = os.path.join(self.temp_dir, "empty.wav")
        with open(empty_file, 'w') as f:
            pass  # Create empty file
        
        transcriber = AudioTranscriber()
        
        with self.assertRaises(TranscriptionError):
            transcriber.transcribe(empty_file)
    
    def test_transcribe_with_model_load_error(self):
        """Test transcription when model loading fails."""
        transcriber = AudioTranscriber()
        
        with patch.object(transcriber, '_load_model', side_effect=ModelLoadError("Model load failed")):
            with self.assertRaises(TranscriptionError):
                transcriber.transcribe(self.test_audio_path)
    
    def test_transcribe_with_whisper_error(self):
        """Test transcription when Whisper fails."""
        transcriber = AudioTranscriber()
        
        with patch.object(transcriber, '_load_model'):
            with patch.object(transcriber, 'model') as mock_model:
                mock_model.transcribe.side_effect = Exception("Whisper error")
                
                with self.assertRaises(TranscriptionError):
                    transcriber.transcribe(self.test_audio_path)
    
    def test_transcribe_invalid_result(self):
        """Test transcription with invalid result."""
        transcriber = AudioTranscriber()
        
        with patch.object(transcriber, '_load_model'):
            with patch.object(transcriber, 'model') as mock_model:
                mock_model.transcribe.return_value = {}  # Missing 'text' key
                
                with self.assertRaises(TranscriptionError):
                    transcriber.transcribe(self.test_audio_path)
    
    def test_transcribe_with_timestamps(self):
        """Test transcription with timestamps."""
        transcriber = AudioTranscriber(timestamps=True)
        
        with patch.object(transcriber, '_load_model'):
            with patch.object(transcriber, 'model') as mock_model:
                mock_model.transcribe.return_value = {
                    "text": "Test transcription",
                    "segments": [
                        {
                            "id": 0,
                            "start": 0.0,
                            "end": 2.0,
                            "text": "Test transcription",
                            "words": []
                        }
                    ],
                    "language": "en"
                }
                
                result = transcriber.transcribe(self.test_audio_path)
                
                self.assertEqual(len(result["segments"]), 1)
                self.assertEqual(result["segments"][0]["start"], 0.0)
                self.assertEqual(result["segments"][0]["end"], 2.0)
    
    def test_transcribe_without_timestamps(self):
        """Test transcription without timestamps."""
        transcriber = AudioTranscriber(timestamps=False)
        
        with patch.object(transcriber, '_load_model'):
            with patch.object(transcriber, 'model') as mock_model:
                mock_model.transcribe.return_value = {
                    "text": "Test transcription",
                    "segments": [
                        {
                            "id": 0,
                            "start": 0.0,
                            "end": 2.0,
                            "text": "Test transcription",
                            "words": []
                        }
                    ],
                    "language": "en"
                }
                
                result = transcriber.transcribe(self.test_audio_path)
                
                self.assertEqual(len(result["segments"]), 0)
    
    def test_extract_cooking_instructions(self):
        """Test cooking instruction extraction."""
        transcriber = AudioTranscriber()
        
        transcription = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "First, chop the onions. Then add them to the pan."
                },
                {
                    "start": 5.0,
                    "end": 10.0,
                    "text": "Now stir the mixture gently."
                }
            ]
        }
        
        instructions = transcriber.extract_cooking_instructions(transcription)
        
        self.assertGreater(len(instructions), 0)
        # Should extract imperative sentences
        for instruction in instructions:
            self.assertIn("start", instruction)
            self.assertIn("end", instruction)
            self.assertIn("text", instruction)
    
    def test_is_likely_instruction(self):
        """Test instruction detection logic."""
        transcriber = AudioTranscriber()
        
        # Test cooking verbs
        self.assertTrue(transcriber._is_likely_instruction("Chop the onions"))
        self.assertTrue(transcriber._is_likely_instruction("Add salt to taste"))
        self.assertTrue(transcriber._is_likely_instruction("Stir the mixture"))
        
        # Test instruction patterns
        self.assertTrue(transcriber._is_likely_instruction("Now add the ingredients"))
        self.assertTrue(transcriber._is_likely_instruction("Next, heat the oil"))
        self.assertTrue(transcriber._is_likely_instruction("First, prepare the vegetables"))
        
        # Test non-instructions
        self.assertFalse(transcriber._is_likely_instruction("This is a beautiful day"))
        self.assertFalse(transcriber._is_likely_instruction("The weather is nice"))
        self.assertFalse(transcriber._is_likely_instruction("I like cooking"))


if __name__ == '__main__':
    unittest.main()
