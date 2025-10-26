"""
Audio Transcriber - Transcribes audio from cooking videos using Whisper
"""

import os
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AudioTranscriber:
    """
    Class for transcribing audio from cooking videos using Whisper.
    """
    
    def __init__(self, model_name="base", language=None, timestamps=True, translate=False):
        """
        Initialize the AudioTranscriber.
        
        Args:
            model_name (str, optional): Whisper model to use. Defaults to "base".
                Options: "tiny", "base", "small", "medium", "large"
            language (str, optional): Language code. Defaults to None (auto-detect).
            timestamps (bool, optional): Whether to include timestamps. Defaults to True.
            translate (bool, optional): Whether to translate non-English audio to English. Defaults to False.
        """
        self.model_name = model_name
        self.language = language
        self.timestamps = timestamps
        self.translate = translate
        self.model = None
        
        # Lazy load the model when needed
    
    def _load_model(self):
        """
        Load the Whisper model.
        """
        if self.model is not None:
            return
        
        try:
            import whisper
            
            logger.info(f"Loading Whisper {self.model_name} model...")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    def transcribe(self, audio_path):
        """
        Transcribe audio from a file.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            dict: Transcription result with text and segments.
        """
        self._load_model()
        
        logger.info(f"Transcribing audio: {audio_path}")
        
        try:
            # Transcribe audio
            transcribe_options = {
                "language": self.language,
                "task": "translate" if self.translate else "transcribe",
                "verbose": False
            }
            
            result = self.model.transcribe(audio_path, **transcribe_options)
            
            # Process segments if timestamps are requested
            if self.timestamps:
                segments = self._process_segments(result["segments"])
            else:
                segments = []
            
            # Create structured result
            transcription = {
                "text": result["text"],
                "segments": segments,
                "language": result.get("language", "unknown")
            }
            
            logger.info(f"Transcription completed: {len(transcription['text'])} characters, {len(segments)} segments")
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            # Return empty transcription as fallback
            return {
                "text": "",
                "segments": [],
                "language": "unknown"
            }
    
    def _process_segments(self, segments):
        """
        Process segments from Whisper result.
        
        Args:
            segments (list): List of segments from Whisper.
            
        Returns:
            list: Processed segments with start time, end time, and text.
        """
        processed_segments = []
        
        for segment in segments:
            processed_segment = {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "words": segment.get("words", [])
            }
            
            processed_segments.append(processed_segment)
        
        return processed_segments
    
    def extract_cooking_instructions(self, transcription):
        """
        Extract cooking instructions from transcription.
        
        Args:
            transcription (dict): Transcription result.
            
        Returns:
            list: List of cooking instructions with timestamps.
        """
        instructions = []
        
        # Simple heuristic: Split by sentences and filter for imperative sentences
        import re
        
        # Get segments with text
        segments = transcription.get("segments", [])
        
        for segment in segments:
            text = segment["text"]
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                
                # Skip empty sentences
                if not sentence:
                    continue
                
                # Check if sentence is likely an instruction
                if self._is_likely_instruction(sentence):
                    instruction = {
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": sentence
                    }
                    
                    instructions.append(instruction)
        
        return instructions
    
    def _is_likely_instruction(self, sentence):
        """
        Check if a sentence is likely a cooking instruction.
        
        Args:
            sentence (str): Sentence to check.
            
        Returns:
            bool: True if likely an instruction, False otherwise.
        """
        # Simple heuristic: Check if sentence starts with a verb
        import re
        
        # Common cooking verbs
        cooking_verbs = [
            "add", "bake", "beat", "blend", "boil", "break", "bring", "brown",
            "chop", "combine", "cook", "cool", "cover", "cut", "dice", "drain",
            "drizzle", "drop", "dry", "fill", "flip", "fold", "fry", "garnish",
            "grate", "grill", "heat", "knead", "layer", "marinate", "mash", "melt",
            "mix", "pour", "preheat", "prepare", "press", "reduce", "remove", "rinse",
            "roast", "roll", "rub", "season", "serve", "set", "simmer", "slice",
            "spread", "sprinkle", "stir", "strain", "stuff", "taste", "toss", "transfer",
            "turn", "whip", "whisk"
        ]
        
        # Check if sentence starts with a cooking verb
        first_word = re.split(r'\s+', sentence.lower())[0]
        
        if first_word in cooking_verbs:
            return True
        
        # Check for common instruction patterns
        instruction_patterns = [
            r"^now\s+\w+",  # "Now add..."
            r"^next\s+\w+",  # "Next, stir..."
            r"^then\s+\w+",  # "Then mix..."
            r"^first\s+\w+",  # "First, chop..."
            r"^finally\s+\w+",  # "Finally, serve..."
            r"let(?:'s)?\s+\w+",  # "Let's add..." or "Let it simmer..."
            r"you(?:'ll)?\s+want\s+to\s+\w+",  # "You'll want to add..."
            r"we(?:'re)?\s+going\s+to\s+\w+"  # "We're going to mix..."
        ]
        
        for pattern in instruction_patterns:
            if re.search(pattern, sentence.lower()):
                return True
        
        return False
