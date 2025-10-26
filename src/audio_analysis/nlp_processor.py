"""
NLP Processor - Process transcribed text to extract cooking-related information
"""

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class NLPProcessor:
    """
    Processes transcribed text to extract cooking-related information.
    """
    
    # Common cooking verbs
    COOKING_VERBS = [
        'add', 'bake', 'beat', 'blend', 'boil', 'broil', 'brown',
        'chop', 'combine', 'cook', 'cool', 'cover', 'cut', 'dice',
        'drain', 'fry', 'grate', 'heat', 'knead', 'mix', 'pour',
        'preheat', 'roll', 'saute', 'season', 'simmer', 'slice',
        'spread', 'sprinkle', 'stir', 'whisk'
    ]
    
    # Measurement units
    MEASUREMENT_UNITS = [
        'cup', 'cups', 'tablespoon', 'tablespoons', 'tbsp',
        'teaspoon', 'teaspoons', 'tsp', 'ounce', 'ounces', 'oz',
        'pound', 'pounds', 'lb', 'lbs', 'gram', 'grams', 'g',
        'kilogram', 'kilograms', 'kg', 'milliliter', 'milliliters', 'ml',
        'liter', 'liters', 'l', 'pinch', 'dash', 'handful'
    ]
    
    # Temperature keywords
    TEMPERATURE_KEYWORDS = [
        'degrees', 'fahrenheit', 'celsius', '°f', '°c', 'temperature'
    ]
    
    # Time keywords
    TIME_KEYWORDS = [
        'minute', 'minutes', 'min', 'hour', 'hours', 'hr', 'hrs',
        'second', 'seconds', 'sec', 'secs'
    ]
    
    def __init__(self, language: str = "en"):
        """
        Initialize the NLPProcessor.
        
        Args:
            language (str): Language code for processing.
        """
        self.language = language
        self.nlp = None
        
        logger.info(f"NLPProcessor initialized with language={language}")
    
    def _load_model(self):
        """
        Lazy-load the spaCy NLP model.
        """
        if self.nlp is None:
            try:
                import spacy
                # Try to load the model
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy model 'en_core_web_sm' not found. Using basic processing.")
                    self.nlp = None
                
                if self.nlp:
                    logger.info("spaCy NLP model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load spaCy model: {e}")
                self.nlp = None
    
    def process(self, transcription: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process transcription to extract cooking-related information.
        
        Args:
            transcription (dict): Transcription results.
            
        Returns:
            dict: Processed information with extracted entities and actions.
        """
        self._load_model()
        
        text = transcription.get("text", "")
        segments = transcription.get("segments", [])
        
        # Extract information
        ingredients = self.extract_ingredients(text)
        measurements = self.extract_measurements(text)
        temperatures = self.extract_temperatures(text)
        times = self.extract_times(text)
        actions = self.extract_actions(segments)
        
        return {
            "text": text,
            "segments": segments,
            "extracted": {
                "ingredients": ingredients,
                "measurements": measurements,
                "temperatures": temperatures,
                "times": times,
                "actions": actions
            }
        }
    
    def extract_ingredients(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract ingredient mentions from text.
        
        Args:
            text (str): Input text.
            
        Returns:
            list: List of extracted ingredients.
        """
        ingredients = []
        
        # Simple keyword-based extraction
        # In a real implementation, this would use NER or a specialized ingredient database
        
        # Common ingredients (subset for demonstration)
        common_ingredients = [
            'flour', 'sugar', 'salt', 'pepper', 'butter', 'oil', 'egg', 'eggs',
            'milk', 'water', 'chicken', 'beef', 'pork', 'fish', 'rice', 'pasta',
            'tomato', 'onion', 'garlic', 'cheese', 'cream', 'vanilla'
        ]
        
        text_lower = text.lower()
        
        for ingredient in common_ingredients:
            if ingredient in text_lower:
                ingredients.append({
                    "name": ingredient,
                    "mentioned": True
                })
        
        return ingredients
    
    def extract_measurements(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract measurements from text.
        
        Args:
            text (str): Input text.
            
        Returns:
            list: List of extracted measurements.
        """
        measurements = []
        
        # Pattern: number + optional fraction + unit
        pattern = r'(\d+(?:\s*\d+/\d+)?|\d+\.\d+)\s*(' + '|'.join(self.MEASUREMENT_UNITS) + r')'
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            amount = match.group(1).strip()
            unit = match.group(2).strip()
            
            measurements.append({
                "amount": amount,
                "unit": unit,
                "text": match.group(0)
            })
        
        return measurements
    
    def extract_temperatures(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract temperatures from text.
        
        Args:
            text (str): Input text.
            
        Returns:
            list: List of extracted temperatures.
        """
        temperatures = []
        
        # Pattern: number + degrees/°/F/C
        pattern = r'(\d+)\s*(?:degrees?|°)?\s*(fahrenheit|celsius|f|c)?'
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            # Check if this is in a temperature context
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].lower()
            
            if any(kw in context for kw in self.TEMPERATURE_KEYWORDS):
                value = match.group(1)
                unit = match.group(2) if match.group(2) else "unknown"
                
                temperatures.append({
                    "value": int(value),
                    "unit": unit.lower(),
                    "text": match.group(0)
                })
        
        return temperatures
    
    def extract_times(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract cooking times from text.
        
        Args:
            text (str): Input text.
            
        Returns:
            list: List of extracted times.
        """
        times = []
        
        # Pattern: number + time unit
        pattern = r'(\d+)\s*(' + '|'.join(self.TIME_KEYWORDS) + r')'
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            value = match.group(1)
            unit = match.group(2).strip()
            
            times.append({
                "value": int(value),
                "unit": unit.lower(),
                "text": match.group(0)
            })
        
        return times
    
    def extract_actions(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract cooking actions from transcription segments.
        
        Args:
            segments (list): Transcription segments.
            
        Returns:
            list: List of extracted actions.
        """
        actions = []
        
        for segment in segments:
            text = segment.get("text", "").lower()
            
            # Find cooking verbs in the segment
            found_verbs = [verb for verb in self.COOKING_VERBS if verb in text]
            
            if found_verbs:
                actions.append({
                    "segment_id": segment.get("id"),
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": segment.get("text"),
                    "actions": found_verbs
                })
        
        return actions

