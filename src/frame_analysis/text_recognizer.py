"""
Text Recognizer - Extract text from video frames using OCR
"""

import logging
import cv2
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TextRecognizer:
    """
    Extracts text from video frames using EasyOCR.
    """
    
    # Common cooking-related keywords
    COOKING_KEYWORDS = [
        'cup', 'cups', 'tablespoon', 'tbsp', 'teaspoon', 'tsp',
        'ounce', 'oz', 'pound', 'lb', 'gram', 'g', 'kg', 'ml', 'l',
        'fahrenheit', 'celsius', 'degrees', '°f', '°c',
        'minute', 'minutes', 'min', 'hour', 'hours', 'hr',
        'salt', 'pepper', 'sugar', 'flour', 'water', 'oil',
        'bake', 'cook', 'heat', 'mix', 'stir', 'blend'
    ]
    
    def __init__(self, languages: List[str] = ['en'], 
                 confidence_threshold: float = 0.5,
                 enhance_text: bool = True):
        """
        Initialize the TextRecognizer.
        
        Args:
            languages (list): List of language codes for OCR.
            confidence_threshold (float): Confidence threshold for text detection.
            enhance_text (bool): Whether to enhance text regions before OCR.
        """
        self.languages = languages
        self.confidence_threshold = confidence_threshold
        self.enhance_text = enhance_text
        self.reader = None
        
        logger.info(f"TextRecognizer initialized with languages={languages}, threshold={confidence_threshold}")
    
    def _load_model(self):
        """
        Lazy-load the EasyOCR model.
        """
        if self.reader is None:
            try:
                import easyocr
                self.reader = easyocr.Reader(self.languages, gpu=False)
                logger.info(f"EasyOCR model loaded for languages: {self.languages}")
            except Exception as e:
                logger.error(f"Failed to load EasyOCR model: {e}")
                raise
    
    def recognize(self, frames: List[Any]) -> List[Dict[str, Any]]:
        """
        Recognize text in a list of frames.
        
        Args:
            frames (list): List of frames as numpy arrays.
            
        Returns:
            list: List of text detection results for each frame.
        """
        self._load_model()
        
        results = []
        
        for i, frame in enumerate(tqdm(frames, desc="Recognizing text")):
            try:
                # Enhance text regions if configured
                if self.enhance_text:
                    frame = self._enhance_text_regions(frame)
                
                # Run OCR
                ocr_results = self.reader.readtext(frame)
                
                # Process results
                frame_results = self._process_results(ocr_results, i)
                results.append(frame_results)
            except Exception as e:
                logger.warning(f"Error recognizing text in frame {i}: {e}")
                results.append({
                    "frame_idx": i,
                    "detections": []
                })
        
        return results
    
    def _process_results(self, ocr_results, frame_idx: int) -> Dict[str, Any]:
        """
        Process OCR results into a structured format.
        
        Args:
            ocr_results: EasyOCR detection results.
            frame_idx (int): Frame index.
            
        Returns:
            dict: Processed text detection results.
        """
        detections = []
        
        for result in ocr_results:
            bbox, text, confidence = result
            
            # Filter by confidence
            if confidence >= self.confidence_threshold:
                detections.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": {
                        "points": [[float(x), float(y)] for x, y in bbox]
                    }
                })
        
        return {
            "frame_idx": frame_idx,
            "detections": detections
        }
    
    def _enhance_text_regions(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance text regions in a frame to improve OCR accuracy.
        
        Args:
            frame (np.ndarray): Input frame.
            
        Returns:
            np.ndarray: Enhanced frame.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            enhanced = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to BGR for EasyOCR
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Error enhancing text regions: {e}")
            return frame
    
    def filter_cooking_related(self, detections: List[Dict[str, Any]], 
                              cooking_keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        Filter text detections to focus on cooking-related text.
        
        Args:
            detections (list): List of text detection results.
            cooking_keywords (list, optional): List of cooking keywords. 
                                              Defaults to self.COOKING_KEYWORDS.
            
        Returns:
            list: Filtered text detection results.
        """
        if cooking_keywords is None:
            cooking_keywords = self.COOKING_KEYWORDS
        
        filtered = []
        
        for frame_result in detections:
            filtered_detections = []
            
            for det in frame_result["detections"]:
                text_lower = det["text"].lower()
                
                # Check if any cooking keyword is in the text
                if any(keyword in text_lower for keyword in cooking_keywords):
                    filtered_detections.append(det)
            
            filtered.append({
                "frame_idx": frame_result["frame_idx"],
                "detections": filtered_detections
            })
        
        return filtered

