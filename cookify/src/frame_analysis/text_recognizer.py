"""
Text Recognizer - Recognizes text in video frames using OCR
"""

import os
import logging
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class TextCategory(Enum):
    """Enum for different categories of cooking-related text."""
    INGREDIENT = "ingredient"
    MEASUREMENT = "measurement"
    TEMPERATURE = "temperature"
    TIME = "time"
    INSTRUCTION = "instruction"
    TITLE = "title"
    UNKNOWN = "unknown"

# Comprehensive list of cooking-related keywords
COOKING_KEYWORDS = {
    # Measurements
    "cup", "tablespoon", "teaspoon", "tbsp", "tsp", "oz", "ounce", "pound", "lb",
    "gram", "g", "kg", "ml", "l", "liter", "quart", "qt", "gallon", "gal", "pinch",
    "dash", "handful", "bunch", "clove", "sprig", "stalk", "head", "slice", "piece",
    
    # Time
    "minute", "min", "hour", "hr", "second", "sec", "overnight", "instant",
    
    # Temperature
    "degrees", "°", "°c", "°f", "celsius", "fahrenheit", "temperature", "heat",
    "low", "medium", "high", "warm", "hot", "cool", "cold",
    
    # Cooking methods
    "bake", "boil", "simmer", "fry", "roast", "grill", "broil", "sauté", "steam",
    "poach", "blanch", "braise", "stew", "toast", "microwave", "pressure cook",
    "slow cook", "barbecue", "bbq", "smoke", "cure", "pickle", "ferment", "marinate",
    
    # Preparation methods
    "chop", "slice", "dice", "mince", "grate", "shred", "julienne", "peel", "core",
    "seed", "pit", "trim", "cut", "crush", "grind", "puree", "blend", "whisk", "beat",
    "stir", "mix", "fold", "knead", "roll", "shape", "stuff", "fill", "coat", "dredge",
    "batter", "bread", "season", "sprinkle", "drizzle", "glaze", "baste", "brush",
    
    # Common ingredients
    "salt", "pepper", "sugar", "flour", "butter", "oil", "water", "milk", "cream",
    "egg", "cheese", "meat", "chicken", "beef", "pork", "fish", "shrimp", "vegetable",
    "fruit", "herb", "spice", "garlic", "onion", "tomato", "potato", "carrot", "rice",
    "pasta", "bread", "sauce", "broth", "stock", "wine", "vinegar",
    
    # Recipe-specific terms
    "recipe", "ingredient", "serving", "yield", "portion", "preparation", "cook time",
    "prep time", "total time", "difficulty", "easy", "medium", "hard", "nutrition",
    "calorie", "protein", "fat", "carb", "fiber", "sodium", "step", "instruction"
}

# Regular expressions for identifying text categories
TEXT_CATEGORY_PATTERNS = {
    TextCategory.INGREDIENT: r"(?i)([\w\s]+)\s*(?:,|\(|$)",
    TextCategory.MEASUREMENT: r"(?i)(\d+(?:\.\d+)?)\s*(?:cup|tablespoon|tbsp|teaspoon|tsp|oz|ounce|pound|lb|gram|g|kg|ml|l|liter|quart|qt|gallon|gal)",
    TextCategory.TEMPERATURE: r"(?i)(\d+)\s*(?:degrees|°|°c|°f|celsius|fahrenheit)",
    TextCategory.TIME: r"(?i)(\d+)\s*(?:minute|min|hour|hr|second|sec)",
    TextCategory.INSTRUCTION: r"(?i)(stir|mix|blend|whisk|fold|knead|roll|shape|bake|boil|simmer|fry|roast|grill|broil|sauté|steam)",
    TextCategory.TITLE: r"(?i)^([\w\s]+recipe|how to [\w\s]+|[\w\s]+ preparation)$"
}

class TextRecognizer:
    """
    Class for recognizing text in video frames using OCR.
    """
    
    def __init__(self, languages=None, confidence_threshold=0.5, enhance_text=True,
                 cooking_specific=True, detect_text_regions=True, use_custom_dictionary=True):
        """
        Initialize the TextRecognizer.
        
        Args:
            languages (list, optional): List of languages to detect. Defaults to ['en'].
            confidence_threshold (float, optional): Confidence threshold for text detection. Defaults to 0.5.
            enhance_text (bool, optional): Whether to enhance text regions before OCR. Defaults to True.
            cooking_specific (bool, optional): Whether to optimize for cooking videos. Defaults to True.
            detect_text_regions (bool, optional): Whether to detect text regions before OCR. Defaults to True.
            use_custom_dictionary (bool, optional): Whether to use a custom cooking dictionary. Defaults to True.
        """
        self.languages = languages or ['en']
        self.confidence_threshold = confidence_threshold
        self.enhance_text = enhance_text
        self.cooking_specific = cooking_specific
        self.detect_text_regions = detect_text_regions
        self.use_custom_dictionary = use_custom_dictionary
        self.reader = None
        self.custom_dictionary = self._create_cooking_dictionary() if use_custom_dictionary else None
        
        # Lazy load the OCR model when needed
    
    def _create_cooking_dictionary(self):
        """
        Create a custom dictionary of cooking terms to improve OCR accuracy.
        
        Returns:
            dict: Dictionary of cooking terms with their categories.
        """
        dictionary = {}
        
        # Common ingredients
        ingredients = [
            "salt", "pepper", "sugar", "flour", "butter", "oil", "water", "milk", "cream",
            "egg", "cheese", "chicken", "beef", "pork", "fish", "shrimp", "vegetable",
            "garlic", "onion", "tomato", "potato", "carrot", "rice", "pasta", "bread",
            "sauce", "broth", "stock", "wine", "vinegar", "lemon", "lime", "orange",
            "apple", "banana", "berry", "chocolate", "vanilla", "cinnamon", "oregano",
            "basil", "thyme", "rosemary", "parsley", "cilantro", "cumin", "paprika",
            "cayenne", "mustard", "ketchup", "mayonnaise", "yogurt", "honey", "maple",
            "syrup", "soy sauce", "olive oil", "vegetable oil", "canola oil", "sesame oil",
            "baking powder", "baking soda", "yeast", "cornstarch", "gelatin", "cocoa",
            "almond", "walnut", "pecan", "cashew", "peanut", "pistachio", "avocado",
            "mushroom", "spinach", "kale", "lettuce", "cabbage", "broccoli", "cauliflower",
            "corn", "pea", "bean", "lentil", "chickpea", "tofu", "tempeh", "seitan"
        ]
        
        for ingredient in ingredients:
            dictionary[ingredient] = TextCategory.INGREDIENT.value
        
        # Common measurements
        measurements = [
            "cup", "tablespoon", "teaspoon", "tbsp", "tsp", "oz", "ounce", "pound", "lb",
            "gram", "g", "kg", "ml", "l", "liter", "quart", "qt", "gallon", "gal", "pinch",
            "dash", "handful", "bunch", "clove", "sprig", "stalk", "head", "slice", "piece"
        ]
        
        for measurement in measurements:
            dictionary[measurement] = TextCategory.MEASUREMENT.value
        
        # Common temperature terms
        temperatures = [
            "degrees", "celsius", "fahrenheit", "°c", "°f", "temperature", "heat",
            "low", "medium", "high", "warm", "hot", "cool", "cold"
        ]
        
        for temp in temperatures:
            dictionary[temp] = TextCategory.TEMPERATURE.value
        
        # Common time terms
        times = [
            "minute", "min", "hour", "hr", "second", "sec", "overnight", "instant"
        ]
        
        for time in times:
            dictionary[time] = TextCategory.TIME.value
        
        # Common cooking instructions
        instructions = [
            "bake", "boil", "simmer", "fry", "roast", "grill", "broil", "sauté", "steam",
            "poach", "blanch", "braise", "stew", "toast", "microwave", "pressure cook",
            "slow cook", "barbecue", "bbq", "smoke", "cure", "pickle", "ferment", "marinate",
            "chop", "slice", "dice", "mince", "grate", "shred", "julienne", "peel", "core",
            "seed", "pit", "trim", "cut", "crush", "grind", "puree", "blend", "whisk", "beat",
            "stir", "mix", "fold", "knead", "roll", "shape", "stuff", "fill", "coat", "dredge",
            "batter", "bread", "season", "sprinkle", "drizzle", "glaze", "baste", "brush"
        ]
        
        for instruction in instructions:
            dictionary[instruction] = TextCategory.INSTRUCTION.value
        
        return dictionary
    
    def _load_model(self):
        """
        Load the OCR model.
        """
        if self.reader is not None:
            return
        
        try:
            import easyocr
            
            logger.info(f"Loading EasyOCR model for languages: {self.languages}")
            self.reader = easyocr.Reader(self.languages)
            logger.info("EasyOCR model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading EasyOCR model: {e}")
            raise
    
    def recognize(self, frames):
        """
        Recognize text in a list of frames.
        
        Args:
            frames (list): List of frames as numpy arrays.
            
        Returns:
            list: List of text detection results for each frame.
                Each result is a dictionary with keys:
                - frame_idx: Index of the frame
                - detections: List of detected text with text, confidence, and bounding box
        """
        self._load_model()
        
        results = []
        
        for i, frame in enumerate(tqdm(frames, desc="Recognizing text")):
            # Enhance text regions if configured
            if self.enhance_text:
                frame = self._enhance_text_regions(frame)
            
            # Run OCR
            ocr_results = self.reader.readtext(frame)
            
            # Process results
            frame_results = self._process_results(ocr_results, i)
            results.append(frame_results)
        
        return results
    
    def _enhance_text_regions(self, frame):
        """
        Enhance text regions in a frame to improve OCR accuracy.
        
        Args:
            frame (numpy.ndarray): Input frame.
            
        Returns:
            numpy.ndarray: Enhanced frame.
        """
        try:
            import cv2
            
            # Make a copy of the original frame
            original = frame.copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply different enhancement techniques based on settings
            if self.cooking_specific:
                # Cooking videos often have text overlays with high contrast
                # Use a combination of techniques for better results
                
                # 1. Adaptive thresholding for variable lighting
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                
                # 2. Apply morphological operations to remove noise
                kernel = np.ones((1, 1), np.uint8)
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                # 3. Edge enhancement for text boundaries
                edges = cv2.Canny(gray, 100, 200)
                dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                
                # 4. Combine the results
                combined = cv2.bitwise_and(opening, opening, mask=dilated_edges)
                
                # 5. Apply contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_gray = clahe.apply(gray)
                
                # Convert back to BGR for OCR
                enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
                
                # If text region detection is enabled, only enhance text regions
                if self.detect_text_regions:
                    text_regions = self._detect_text_regions(original)
                    if text_regions:
                        # Create a mask for text regions
                        mask = np.zeros_like(gray)
                        for (x1, y1, x2, y2) in text_regions:
                            mask[y1:y2, x1:x2] = 255
                        
                        # Apply mask to enhanced image
                        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        enhanced = np.where(mask_3channel > 0, enhanced, original)
            else:
                # Simple enhancement for general cases
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                
                kernel = np.ones((1, 1), np.uint8)
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                # Convert back to BGR for OCR
                enhanced = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Error enhancing text regions: {e}")
            return frame
    
    def _detect_text_regions(self, frame):
        """
        Detect potential text regions in a frame.
        
        Args:
            frame (numpy.ndarray): Input frame.
            
        Returns:
            list: List of text region bounding boxes (x1, y1, x2, y2).
        """
        try:
            import cv2
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours based on size and aspect ratio
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size
                if w < 20 or h < 10 or w > frame.shape[1] * 0.9 or h > frame.shape[0] * 0.9:
                    continue
                
                # Filter by aspect ratio
                aspect_ratio = w / float(h)
                if aspect_ratio < 0.1 or aspect_ratio > 10:
                    continue
                
                # Add some padding
                padding = 10
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                text_regions.append((x1, y1, x2, y2))
            
            # Merge overlapping regions
            text_regions = self._merge_overlapping_regions(text_regions)
            
            return text_regions
        except Exception as e:
            logger.warning(f"Error detecting text regions: {e}")
            return []
    
    def _merge_overlapping_regions(self, regions):
        """
        Merge overlapping text regions.
        
        Args:
            regions (list): List of region bounding boxes (x1, y1, x2, y2).
            
        Returns:
            list: Merged region bounding boxes.
        """
        if not regions:
            return []
        
        # Sort regions by x-coordinate
        sorted_regions = sorted(regions, key=lambda r: r[0])
        
        merged_regions = [sorted_regions[0]]
        
        for current in sorted_regions[1:]:
            previous = merged_regions[-1]
            
            # Check if regions overlap
            if (current[0] <= previous[2] and current[1] <= previous[3] and
                current[2] >= previous[0] and current[3] >= previous[1]):
                # Merge regions
                merged_regions[-1] = (
                    min(previous[0], current[0]),
                    min(previous[1], current[1]),
                    max(previous[2], current[2]),
                    max(previous[3], current[3])
                )
            else:
                merged_regions.append(current)
        
        return merged_regions
    
    def _process_results(self, ocr_results, frame_idx):
        """
        Process OCR results into a structured format.
        
        Args:
            ocr_results: Results from EasyOCR.
            frame_idx (int): Index of the frame.
            
        Returns:
            dict: Structured text detection results.
        """
        detections = []
        
        # Process each detection
        for result in ocr_results:
            # Get bounding box, text, and confidence
            box = result[0]
            text = result[1]
            confidence = result[2]
            
            # Filter by confidence threshold
            if confidence < self.confidence_threshold:
                continue
            
            # Create detection object
            detection = {
                "text": text,
                "confidence": confidence,
                "box": {
                    "top_left": (float(box[0][0]), float(box[0][1])),
                    "top_right": (float(box[1][0]), float(box[1][1])),
                    "bottom_right": (float(box[2][0]), float(box[2][1])),
                    "bottom_left": (float(box[3][0]), float(box[3][1]))
                }
            }
            
            detections.append(detection)
        
        return {
            "frame_idx": frame_idx,
            "detections": detections
        }
    
    def filter_cooking_related(self, detections, cooking_keywords=None):
        """
        Filter text detections to only include cooking-related text.
        
        Args:
            detections (list): List of text detection results.
            cooking_keywords (set, optional): Set of cooking-related keywords.
                Defaults to COOKING_KEYWORDS.
            
        Returns:
            list: Filtered text detection results.
        """
        if cooking_keywords is None:
            cooking_keywords = COOKING_KEYWORDS
        
        filtered_results = []
        
        for frame_result in detections:
            filtered_detections = []
            
            for detection in frame_result["detections"]:
                text = detection["text"].lower()
                words = re.findall(r'\b\w+\b', text)
                
                # Check if any cooking keyword is in the text
                if any(keyword in text for keyword in cooking_keywords) or any(word in cooking_keywords for word in words):
                    filtered_detections.append(detection)
            
            filtered_results.append({
                "frame_idx": frame_result["frame_idx"],
                "detections": filtered_detections
            })
        
        return filtered_results
    
    def categorize_text(self, detections):
        """
        Categorize detected text into cooking-related categories.
        
        Args:
            detections (list): List of text detection results.
            
        Returns:
            list: Text detection results with added category field.
        """
        categorized_results = []
        
        for frame_result in detections:
            categorized_detections = []
            
            for detection in frame_result["detections"]:
                text = detection["text"]
                category = self._categorize_single_text(text)
                
                # Add category to detection
                categorized_detection = detection.copy()
                categorized_detection["category"] = category.value
                categorized_detections.append(categorized_detection)
            
            categorized_results.append({
                "frame_idx": frame_result["frame_idx"],
                "detections": categorized_detections
            })
        
        return categorized_results
    
    def _categorize_single_text(self, text):
        """
        Categorize a single text string.
        
        Args:
            text (str): Text to categorize.
            
        Returns:
            TextCategory: Category of the text.
        """
        text_lower = text.lower()
        
        # Check each category pattern
        for category, pattern in TEXT_CATEGORY_PATTERNS.items():
            if re.search(pattern, text_lower):
                return category
        
        # Check custom dictionary if available
        if self.custom_dictionary:
            words = re.findall(r'\b\w+\b', text_lower)
            for word in words:
                if word in self.custom_dictionary:
                    return TextCategory(self.custom_dictionary[word])
        
        return TextCategory.UNKNOWN
    
    def extract_structured_data(self, categorized_detections):
        """
        Extract structured data from categorized text detections.
        
        Args:
            categorized_detections (list): List of categorized text detection results.
            
        Returns:
            dict: Structured data with ingredients, measurements, etc.
        """
        structured_data = {
            "ingredients": [],
            "measurements": [],
            "temperatures": [],
            "times": [],
            "instructions": [],
            "title": None
        }
        
        # Process each frame's detections
        for frame_result in categorized_detections:
            for detection in frame_result["detections"]:
                text = detection["text"]
                category = detection["category"]
                
                if category == TextCategory.INGREDIENT.value:
                    # Extract ingredient name
                    ingredient_match = re.search(r"(?i)([\w\s]+)(?:\s*,|\s*\(|$)", text)
                    if ingredient_match:
                        ingredient = ingredient_match.group(1).strip()
                        if ingredient and ingredient not in structured_data["ingredients"]:
                            structured_data["ingredients"].append(ingredient)
                
                elif category == TextCategory.MEASUREMENT.value:
                    # Extract measurement value and unit
                    measurement_match = re.search(r"(?i)(\d+(?:\.\d+)?)\s*(cup|tablespoon|tbsp|teaspoon|tsp|oz|ounce|pound|lb|gram|g|kg|ml|l|liter|quart|qt|gallon|gal)", text)
                    if measurement_match:
                        value = measurement_match.group(1)
                        unit = measurement_match.group(2)
                        measurement = {"value": value, "unit": unit}
                        if measurement not in structured_data["measurements"]:
                            structured_data["measurements"].append(measurement)
                
                elif category == TextCategory.TEMPERATURE.value:
                    # Extract temperature value and unit
                    temp_match = re.search(r"(?i)(\d+)\s*(degrees|°|°c|°f|celsius|fahrenheit)", text)
                    if temp_match:
                        value = temp_match.group(1)
                        unit = temp_match.group(2)
                        temperature = {"value": value, "unit": unit}
                        if temperature not in structured_data["temperatures"]:
                            structured_data["temperatures"].append(temperature)
                
                elif category == TextCategory.TIME.value:
                    # Extract time value and unit
                    time_match = re.search(r"(?i)(\d+)\s*(minute|min|hour|hr|second|sec)", text)
                    if time_match:
                        value = time_match.group(1)
                        unit = time_match.group(2)
                        time = {"value": value, "unit": unit}
                        if time not in structured_data["times"]:
                            structured_data["times"].append(time)
                
                elif category == TextCategory.INSTRUCTION.value:
                    # Extract instruction
                    if text not in structured_data["instructions"]:
                        structured_data["instructions"].append(text)
                
                elif category == TextCategory.TITLE.value:
                    # Extract title (use the longest one if multiple are found)
                    if not structured_data["title"] or len(text) > len(structured_data["title"]):
                        structured_data["title"] = text
        
        return structured_data
    
    def recognize_with_postprocessing(self, frames):
        """
        Recognize text in frames with additional post-processing.
        
        Args:
            frames (list): List of frames as numpy arrays.
            
        Returns:
            dict: Structured text data extracted from frames.
        """
        # Get raw text detections
        detections = self.recognize(frames)
        
        # Filter for cooking-related text
        filtered_detections = self.filter_cooking_related(detections)
        
        # Categorize text
        categorized_detections = self.categorize_text(filtered_detections)
        
        # Extract structured data
        structured_data = self.extract_structured_data(categorized_detections)
        
        return {
            "raw_detections": detections,
            "filtered_detections": filtered_detections,
            "categorized_detections": categorized_detections,
            "structured_data": structured_data
        }
