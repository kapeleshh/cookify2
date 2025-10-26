"""
Audio Transcriber - Transcribe audio from cooking videos using Whisper
"""

import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """
    Transcribes audio from cooking videos using OpenAI Whisper.
    """
    
    def __init__(self, model_name: str = "base", 
                 language: str = "en",
                 timestamps: bool = True):
        """
        Initialize the AudioTranscriber.
        
        Args:
            model_name (str): Whisper model name ('tiny', 'base', 'small', 'medium', 'large').
            language (str): Language code for transcription.
            timestamps (bool): Whether to include word-level timestamps.
        """
        self.model_name = model_name
        self.language = language
        self.timestamps = timestamps
        self.model = None
        
        logger.info(f"AudioTranscriber initialized with model={model_name}, language={language}")
    
    def _load_model(self):
        """
        Lazy-load the Whisper model.
        """
        if self.model is None:
            try:
                import whisper
                logger.info(f"Loading Whisper model: {self.model_name}")
                self.model = whisper.load_model(self.model_name)
                logger.info(f"Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio from a file.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            dict: Transcription results with text, segments, and metadata.
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return {
                "text": "",
                "segments": [],
                "language": self.language
            }
        
        self._load_model()
        
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                word_timestamps=self.timestamps,
                verbose=False
            )
            
            # Process segments
            segments = []
            for segment in result.get("segments", []):
                processed_segment = {
                    "id": segment["id"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "confidence": segment.get("no_speech_prob", 0.0)
                }
                
                # Add word-level timestamps if available
                if "words" in segment:
                    processed_segment["words"] = [
                        {
                            "word": word["word"],
                            "start": word["start"],
                            "end": word["end"],
                            "confidence": word.get("probability", 1.0)
                        }
                        for word in segment["words"]
                    ]
                
                segments.append(processed_segment)
            
            transcription = {
                "text": result.get("text", ""),
                "segments": segments,
                "language": result.get("language", self.language)
            }
            
            logger.info(f"Transcription completed: {len(segments)} segments")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {
                "text": "",
                "segments": [],
                "language": self.language,
                "error": str(e)
            }
    
    def align_with_frames(self, transcription: Dict[str, Any], 
                         fps: float = 30.0) -> Dict[str, Any]:
        """
        Align transcription segments with video frames.
        
        Args:
            transcription (dict): Transcription results.
            fps (float): Video frame rate.
            
        Returns:
            dict: Transcription with frame alignment.
        """
        aligned_segments = []
        
        for segment in transcription.get("segments", []):
            aligned_segment = segment.copy()
            
            # Convert timestamps to frame numbers
            aligned_segment["start_frame"] = int(segment["start"] * fps)
            aligned_segment["end_frame"] = int(segment["end"] * fps)
            
            # Align words if available
            if "words" in segment:
                aligned_words = []
                for word in segment["words"]:
                    aligned_word = word.copy()
                    aligned_word["start_frame"] = int(word["start"] * fps)
                    aligned_word["end_frame"] = int(word["end"] * fps)
                    aligned_words.append(aligned_word)
                
                aligned_segment["words"] = aligned_words
            
            aligned_segments.append(aligned_segment)
        
        return {
            **transcription,
            "segments": aligned_segments,
            "fps": fps
        }

