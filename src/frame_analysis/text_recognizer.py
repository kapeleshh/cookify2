"""
Text Recognizer using EasyOCR for reading text from frames
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import re

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("easyocr not installed. Text recognition will be disabled.")

logger = logging.getLogger(__name__)

class TextRecognizer:
    """
    Recognizes text in video frames using EasyOCR.
    """
    
    # Common measurement patterns
    MEASUREMENT_PATTERNS = [
        r'\d+\s*(cup|cups|tbsp|tsp|oz|g|kg|ml|l|lb|lbs)',
        r'\d+/\d+\s*(cup|cups|tbsp|tsp)',
        r'\d+\.\d+\s*(cup|cups|tbsp|tsp|oz|g|kg|ml|l)'
    ]
    
    def __init__(self, languages=['en'], gpu=False):
        """
        Initialize the text recognizer.
        
        Args:
            languages: List of language codes
            gpu: Whether to use GPU
        """
        self.reader = None
        self.gpu = gpu
        
        if EASYOCR_AVAILABLE:
            try:
                logger.info(f"Loading EasyOCR for languages: {languages}")
                self.reader = easyocr.Reader(languages, gpu=gpu)
                logger.info("✓ EasyOCR loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load EasyOCR: {e}")
                self.reader = None
        else:
            logger.warning("EasyOCR not available")
    
    def recognize(self, frame_path: str, confidence_threshold=0.5) -> Dict[str, Any]:
        """
        Recognize text in a frame.
        
        Args:
            frame_path: Path to the frame image
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            Dictionary with recognized text
        """
        if not self.reader:
            return {'text_items': [], 'measurements': [], 'all_text': ''}
        
        try:
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"Could not read frame: {frame_path}")
                return {'text_items': [], 'measurements': [], 'all_text': ''}
            
            # Run OCR
            results = self.reader.readtext(frame)
            
            # Parse results
            text_items = []
            measurements = []
            all_text = []
            
            for (bbox, text, conf) in results:
                if conf >= confidence_threshold:
                    text_items.append({
                        'text': text,
                        'confidence': round(conf, 3),
                        'bbox': bbox
                    })
                    all_text.append(text)
                    
                    # Check for measurements
                    for pattern in self.MEASUREMENT_PATTERNS:
                        if re.search(pattern, text.lower()):
                            measurements.append(text)
            
            return {
                'text_items': text_items,
                'measurements': measurements,
                'all_text': ' '.join(all_text)
            }
            
        except Exception as e:
            logger.error(f"Text recognition failed: {e}")
            return {'text_items': [], 'measurements': [], 'all_text': ''}
    
    def recognize_batch(self, frame_paths: List[str]) -> List[Dict[str, Any]]:
        """Recognize text in multiple frames."""
        return [self.recognize(fp) for fp in frame_paths]

