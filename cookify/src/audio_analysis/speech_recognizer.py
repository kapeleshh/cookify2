"""
Speech Recognizer - Transcribes speech from cooking videos using Whisper
"""

import os
import logging
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Cooking-specific vocabulary to improve transcription accuracy
COOKING_VOCABULARY = [
    # Ingredients
    "flour", "sugar", "salt", "pepper", "butter", "oil", "olive oil", "vegetable oil",
    "garlic", "onion", "tomato", "potato", "carrot", "celery", "bell pepper", 
    "chicken", "beef", "pork", "fish", "shrimp", "egg", "milk", "cream", "cheese",
    "yogurt", "vinegar", "soy sauce", "honey", "maple syrup", "vanilla extract",
    "cinnamon", "oregano", "basil", "thyme", "rosemary", "parsley", "cilantro",
    "cumin", "paprika", "cayenne", "turmeric", "ginger", "nutmeg", "cardamom",
    
    # Measurements
    "cup", "tablespoon", "teaspoon", "ounce", "pound", "gram", "kilogram",
    "milliliter", "liter", "pinch", "dash", "handful", "tbsp", "tsp", "oz", "lb",
    "g", "kg", "ml", "l", "qt", "quart", "gallon", "gal",
    
    # Cooking methods
    "bake", "boil", "broil", "fry", "grill", "poach", "roast", "sauté", "simmer",
    "steam", "stir-fry", "blanch", "braise", "caramelize", "deglaze", "flambé",
    "marinate", "reduce", "render", "sear", "smoke", "sous vide", "sweat", "temper",
    
    # Cooking tools
    "bowl", "pan", "pot", "skillet", "knife", "cutting board", "spoon", "spatula",
    "whisk", "blender", "food processor", "mixer", "grater", "peeler", "measuring cup",
    "measuring spoon", "thermometer", "timer", "oven", "stove", "microwave", "grill",
    "slow cooker", "pressure cooker", "air fryer", "colander", "strainer",
    
    # Cooking instructions
    "preheat", "mix", "stir", "whisk", "beat", "fold", "knead", "roll", "cut", "chop",
    "dice", "mince", "slice", "grate", "peel", "core", "seed", "zest", "juice",
    "season", "sprinkle", "drizzle", "garnish", "serve", "refrigerate", "freeze",
    
    # Common cooking phrases
    "bring to a boil", "reduce heat", "simmer until", "cook until", "set aside",
    "let it rest", "stir occasionally", "season to taste", "preheat the oven",
    "medium-high heat", "medium-low heat", "room temperature", "al dente",
    "mise en place", "fold in", "whip until", "bake until golden"
]

class SpeechRecognizer:
    """
    Class for transcribing speech from cooking videos using Whisper.
    """
    
    def __init__(self, model_size="base", language=None, device=None, 
                 use_cooking_vocabulary=True, filter_kitchen_noise=True,
                 compute_word_confidence=True, beam_size=5):
        """
        Initialize the SpeechRecognizer.
        
        Args:
            model_size (str, optional): Whisper model size ('tiny', 'base', 'small', 'medium', 'large'). 
                                        Defaults to "base".
            language (str, optional): Language code (e.g., 'en', 'fr'). Defaults to None (auto-detect).
            device (str, optional): Device to use ('cpu', 'cuda'). Defaults to None (auto-select).
            use_cooking_vocabulary (bool, optional): Whether to use cooking vocabulary. Defaults to True.
            filter_kitchen_noise (bool, optional): Whether to filter kitchen noise. Defaults to True.
            compute_word_confidence (bool, optional): Whether to compute word-level confidence. Defaults to True.
            beam_size (int, optional): Beam size for decoding. Defaults to 5.
        """
        self.model_size = model_size
        self.language = language
        self.device = device
        self.use_cooking_vocabulary = use_cooking_vocabulary
        self.filter_kitchen_noise = filter_kitchen_noise
        self.compute_word_confidence = compute_word_confidence
        self.beam_size = beam_size
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
            import torch
            
            # Determine device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Whisper {self.model_size} model on {self.device}")
            self.model = whisper.load_model(self.model_size, device=self.device)
            
            # Load cooking vocabulary if configured
            if self.use_cooking_vocabulary:
                self._load_cooking_vocabulary()
            
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    def _load_cooking_vocabulary(self):
        """
        Load cooking vocabulary to improve transcription accuracy.
        """
        # In a real implementation, we would add the vocabulary to the model
        # For now, we'll just log that we're using it
        logger.info(f"Using cooking vocabulary with {len(COOKING_VOCABULARY)} terms")
    
    def transcribe(self, audio_path, segment_length=30.0, overlap=5.0):
        """
        Transcribe speech from an audio file.
        
        Args:
            audio_path (str): Path to the audio file.
            segment_length (float, optional): Length of each segment in seconds. Defaults to 30.0.
            overlap (float, optional): Overlap between segments in seconds. Defaults to 5.0.
            
        Returns:
            dict: Transcription result with text and segments.
        """
        self._load_model()
        
        logger.info(f"Transcribing audio: {audio_path}")
        
        try:
            # Preprocess audio if needed
            if self.filter_kitchen_noise:
                audio = self._preprocess_audio(audio_path)
            else:
                # Load audio directly
                audio = self._load_audio(audio_path)
            
            # Transcribe with timestamps
            transcribe_options = {
                "language": self.language,
                "beam_size": self.beam_size,
                "word_timestamps": self.compute_word_confidence,
                "verbose": False
            }
            
            # For long audio, process in segments
            if segment_length > 0:
                result = self._transcribe_in_segments(audio, segment_length, overlap, transcribe_options)
            else:
                # Transcribe the entire audio at once
                result = self.model.transcribe(audio, **transcribe_options)
            
            # Post-process transcription
            processed_result = self._post_process_transcription(result)
            
            logger.info(f"Transcription completed: {len(processed_result['text'])} characters, {len(processed_result['segments'])} segments")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            # Return empty result as fallback
            return {
                "text": "",
                "segments": []
            }
    
    def _load_audio(self, audio_path):
        """
        Load audio from file.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            numpy.ndarray: Audio data.
        """
        try:
            import whisper
            
            # Use Whisper's audio loading function
            audio = whisper.load_audio(audio_path)
            return audio
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def _preprocess_audio(self, audio_path):
        """
        Preprocess audio to filter kitchen noise.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            numpy.ndarray: Preprocessed audio data.
        """
        try:
            import whisper
            import librosa
            import numpy as np
            from scipy import signal
            
            # Load audio
            audio = whisper.load_audio(audio_path)
            
            # Apply noise reduction
            # This is a simplified implementation
            # In a real implementation, we would use more sophisticated techniques
            
            # 1. Apply a high-pass filter to remove low-frequency noise
            b, a = signal.butter(4, 100 / (16000 / 2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio)
            
            # 2. Apply spectral gating noise reduction
            # (simplified version)
            stft = librosa.stft(filtered_audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Simple spectral gating
            threshold = np.mean(magnitude) * 0.5
            mask = (magnitude > threshold).astype(np.float32)
            
            # Apply mask
            magnitude = magnitude * mask
            
            # Reconstruct signal
            stft_processed = magnitude * np.exp(1j * phase)
            processed_audio = librosa.istft(stft_processed)
            
            # Normalize
            processed_audio = librosa.util.normalize(processed_audio)
            
            return processed_audio
        except Exception as e:
            logger.warning(f"Error preprocessing audio: {e}, falling back to standard loading")
            return self._load_audio(audio_path)
    
    def _transcribe_in_segments(self, audio, segment_length, overlap, options):
        """
        Transcribe audio in segments.
        
        Args:
            audio (numpy.ndarray): Audio data.
            segment_length (float): Length of each segment in seconds.
            overlap (float): Overlap between segments in seconds.
            options (dict): Transcription options.
            
        Returns:
            dict: Combined transcription result.
        """
        # Calculate segment size in samples
        sample_rate = 16000  # Whisper uses 16kHz
        segment_samples = int(segment_length * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        
        # Initialize result
        combined_result = {
            "text": "",
            "segments": []
        }
        
        # Process segments
        offset = 0
        segment_idx = 0
        
        with tqdm(total=len(audio) // sample_rate, desc="Transcribing segments") as pbar:
            while offset < len(audio):
                # Extract segment
                end = min(offset + segment_samples, len(audio))
                segment = audio[offset:end]
                
                # Transcribe segment
                segment_result = self.model.transcribe(segment, **options)
                
                # Adjust timestamps
                offset_seconds = offset / sample_rate
                for seg in segment_result["segments"]:
                    seg["start"] += offset_seconds
                    seg["end"] += offset_seconds
                    seg["segment_idx"] = segment_idx
                
                # Add to combined result
                if combined_result["text"]:
                    combined_result["text"] += " " + segment_result["text"]
                else:
                    combined_result["text"] = segment_result["text"]
                
                combined_result["segments"].extend(segment_result["segments"])
                
                # Move to next segment with overlap
                offset = end - overlap_samples
                segment_idx += 1
                
                # Update progress bar
                pbar.update(segment_length)
        
        return combined_result
    
    def _post_process_transcription(self, result):
        """
        Post-process transcription result.
        
        Args:
            result (dict): Transcription result.
            
        Returns:
            dict: Processed transcription result.
        """
        # This is a simplified implementation
        # In a real implementation, we would apply more sophisticated post-processing
        
        # 1. Merge adjacent segments with the same speaker
        merged_segments = []
        current_segment = None
        
        for segment in result["segments"]:
            if current_segment is None:
                current_segment = segment.copy()
            elif segment.get("speaker") == current_segment.get("speaker"):
                # Merge with current segment
                current_segment["end"] = segment["end"]
                current_segment["text"] += " " + segment["text"]
                
                # Merge words if available
                if "words" in current_segment and "words" in segment:
                    current_segment["words"].extend(segment["words"])
            else:
                # Add current segment to result and start a new one
                merged_segments.append(current_segment)
                current_segment = segment.copy()
        
        # Add the last segment
        if current_segment is not None:
            merged_segments.append(current_segment)
        
        # 2. Apply cooking-specific corrections
        if self.use_cooking_vocabulary:
            corrected_segments = self._apply_cooking_corrections(merged_segments)
        else:
            corrected_segments = merged_segments
        
        # 3. Create final result
        processed_result = {
            "text": result["text"],
            "segments": corrected_segments
        }
        
        return processed_result
    
    def _apply_cooking_corrections(self, segments):
        """
        Apply cooking-specific corrections to transcription segments.
        
        Args:
            segments (list): Transcription segments.
            
        Returns:
            list: Corrected segments.
        """
        corrected_segments = []
        
        # Common cooking terms that might be misrecognized
        corrections = {
            "olive oil": ["all of oil", "olive all", "all of all"],
            "tablespoon": ["table spoon", "table spoons"],
            "teaspoon": ["tea spoon", "tea spoons"],
            "sauté": ["saute", "sautee", "saw tay"],
            "al dente": ["all dente", "all dentay"],
            "mise en place": ["meez on plas", "mees on place"],
            "sous vide": ["sue veed", "sue vide", "soo veed"],
            "julienne": ["julie n", "julie en"],
            "chiffonade": ["shiffonade", "chiffon aid"],
            "roux": ["rue", "roo"],
            "bouquet garni": ["bouquet garny", "bouquet garnee"],
            "mirepoix": ["meer pwah", "meer pwa", "meer poix"]
        }
        
        for segment in segments:
            text = segment["text"]
            
            # Apply corrections
            for correct, variants in corrections.items():
                for variant in variants:
                    text = re.sub(r'\b' + re.escape(variant) + r'\b', correct, text, flags=re.IGNORECASE)
            
            # Update segment
            corrected_segment = segment.copy()
            corrected_segment["text"] = text
            corrected_segments.append(corrected_segment)
        
        return corrected_segments
    
    def transcribe_with_diarization(self, audio_path, num_speakers=2):
        """
        Transcribe speech with speaker diarization.
        
        Args:
            audio_path (str): Path to the audio file.
            num_speakers (int, optional): Number of speakers. Defaults to 2.
            
        Returns:
            dict: Transcription result with text, segments, and speaker information.
        """
        self._load_model()
        
        logger.info(f"Transcribing audio with diarization: {audio_path}")
        
        try:
            # First, transcribe the audio
            transcription = self.transcribe(audio_path)
            
            # Then, perform speaker diarization
            diarized_segments = self._perform_diarization(audio_path, transcription["segments"], num_speakers)
            
            # Create final result
            result = {
                "text": transcription["text"],
                "segments": diarized_segments
            }
            
            logger.info(f"Diarization completed: {len(diarized_segments)} segments")
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing with diarization: {e}")
            # Return the original transcription as fallback
            return self.transcribe(audio_path)
    
    def _perform_diarization(self, audio_path, segments, num_speakers):
        """
        Perform speaker diarization.
        
        Args:
            audio_path (str): Path to the audio file.
            segments (list): Transcription segments.
            num_speakers (int): Number of speakers.
            
        Returns:
            list: Segments with speaker information.
        """
        try:
            # This is a placeholder implementation
            # In a real implementation, we would use a speaker diarization model
            
            # For now, we'll just assign speakers based on alternating segments
            diarized_segments = []
            
            for i, segment in enumerate(segments):
                diarized_segment = segment.copy()
                diarized_segment["speaker"] = f"SPEAKER_{i % num_speakers + 1}"
                diarized_segments.append(diarized_segment)
            
            return diarized_segments
        except Exception as e:
            logger.error(f"Error performing diarization: {e}")
            return segments
    
    def extract_cooking_instructions(self, transcription):
        """
        Extract cooking instructions from transcription.
        
        Args:
            transcription (dict): Transcription result.
            
        Returns:
            list: List of cooking instructions with text, start time, and end time.
        """
        instructions = []
        
        # Common instruction patterns
        instruction_patterns = [
            r"(?:first|next|then|after that|finally),?\s+(.+)",
            r"(?:start by|begin by|you want to)\s+(.+)",
            r"(?:now|let's|we're going to)\s+(.+)",
            r"(?:you need to|you'll need to|you have to)\s+(.+)"
        ]
        
        # Process each segment
        for segment in transcription["segments"]:
            text = segment["text"]
            
            # Check for instruction patterns
            for pattern in instruction_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    instruction_text = match.group(1)
                    
                    # Calculate start and end positions within the segment
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Calculate approximate timestamps
                    segment_duration = segment["end"] - segment["start"]
                    segment_length = len(text)
                    
                    start_time = segment["start"] + (start_pos / segment_length) * segment_duration
                    end_time = segment["start"] + (end_pos / segment_length) * segment_duration
                    
                    instruction = {
                        "text": instruction_text,
                        "start_time": start_time,
                        "end_time": end_time,
                        "segment_idx": segment.get("segment_idx", 0)
                    }
                    
                    instructions.append(instruction)
        
        return instructions
    
    def extract_ingredients_from_speech(self, transcription):
        """
        Extract ingredients from transcription.
        
        Args:
            transcription (dict): Transcription result.
            
        Returns:
            list: List of ingredients with name, quantity, unit, and timestamp.
        """
        ingredients = []
        
        # Common ingredient patterns
        ingredient_patterns = [
            r"(\d+(?:\.\d+)?)\s+(cup|tablespoon|tbsp|teaspoon|tsp|ounce|oz|pound|lb|gram|g|kg|ml|l|liter|pinch|dash|handful)s?\s+(?:of\s+)?(.+?)(?:,|\.|\s+and|\s+or|\s+to|\s+for|\s+until|\s+then|\s+$)",
            r"(\d+(?:\.\d+)?)\s+(.+?)(?:,|\.|\s+and|\s+or|\s+to|\s+for|\s+until|\s+then|\s+$)"
        ]
        
        # Process each segment
        for segment in transcription["segments"]:
            text = segment["text"]
            
            # Check for ingredient patterns
            for pattern in ingredient_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    if len(match.groups()) >= 3:
                        # Pattern with quantity, unit, and ingredient
                        quantity = match.group(1)
                        unit = match.group(2)
                        name = match.group(3).strip()
                    else:
                        # Pattern with quantity and ingredient
                        quantity = match.group(1)
                        unit = ""
                        name = match.group(2).strip()
                    
                    # Skip if name is not a likely ingredient
                    if not self._is_likely_ingredient(name):
                        continue
                    
                    # Calculate approximate timestamp
                    start_pos = match.start()
                    segment_duration = segment["end"] - segment["start"]
                    segment_length = len(text)
                    
                    timestamp = segment["start"] + (start_pos / segment_length) * segment_duration
                    
                    ingredient = {
                        "name": name,
                        "quantity": quantity,
                        "unit": unit,
                        "timestamp": timestamp,
                        "segment_idx": segment.get("segment_idx", 0)
                    }
                    
                    ingredients.append(ingredient)
        
        return ingredients
    
    def _is_likely_ingredient(self, text):
        """
        Check if text is likely an ingredient.
        
        Args:
            text (str): Text to check.
            
        Returns:
            bool: True if likely an ingredient, False otherwise.
        """
        # Common ingredients
        common_ingredients = [
            "salt", "pepper", "sugar", "flour", "butter", "oil", "water", "milk",
            "egg", "garlic", "onion", "tomato", "potato", "carrot", "chicken",
            "beef", "pork", "fish", "rice", "pasta", "cheese", "cream", "yogurt",
            "vinegar", "lemon", "lime", "orange", "apple", "banana", "berry",
            "chocolate", "vanilla", "cinnamon", "oregano", "basil", "thyme",
            "rosemary", "parsley", "cilantro", "ginger", "soy sauce", "honey",
            "maple syrup", "mustard", "ketchup", "mayonnaise", "bread", "tortilla"
        ]
        
        text_lower = text.lower()
        
        # Check if text contains a common ingredient
        for ingredient in common_ingredients:
            if ingredient in text_lower:
                return True
        
        # Check if text is in cooking vocabulary
        for term in COOKING_VOCABULARY:
            if term.lower() in text_lower:
                return True
        
        return False
