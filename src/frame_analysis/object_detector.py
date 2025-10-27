"""
Object Detector using YOLOv8 for detecting objects in frames
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics not installed. Object detection will be limited.")

logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Detects objects in video frames using YOLOv8.
    """
    
    # Cooking-related objects from COCO dataset
    COOKING_OBJECTS = {
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
        'pizza', 'donut', 'cake', 'chair', 'dining table', 'oven', 'toaster',
        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors'
    }
    
    def __init__(self, model_name="yolov8n.pt", confidence_threshold=0.25):
        """
        Initialize the object detector.
        
        Args:
            model_name: YOLOv8 model to use (n/s/m/l/x)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if YOLO_AVAILABLE:
            try:
                logger.info(f"Loading YOLO model: {model_name}")
                self.model = YOLO(model_name)
                logger.info("✓ YOLO model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                self.model = None
        else:
            logger.warning("YOLO not available")
    
    def detect(self, frame_path: str) -> Dict[str, Any]:
        """
        Detect objects in a frame.
        
        Args:
            frame_path: Path to the frame image
            
        Returns:
            Dictionary with detection results
        """
        if not self.model:
            return {'objects': [], 'count': 0, 'cooking_related': []}
        
        try:
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"Could not read frame: {frame_path}")
                return {'objects': [], 'count': 0, 'cooking_related': []}
            
            # Run detection
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Parse results
            detections = []
            cooking_objects = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = result.names[cls_id]
                    
                    detection = {
                        'class': class_name,
                        'confidence': round(conf, 3),
                        'bbox': box.xyxy[0].tolist()
                    }
                    detections.append(detection)
                    
                    # Check if cooking-related
                    if class_name.lower() in self.COOKING_OBJECTS:
                        cooking_objects.append(class_name)
            
            return {
                'objects': detections,
                'count': len(detections),
                'cooking_related': list(set(cooking_objects))
            }
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {'objects': [], 'count': 0, 'cooking_related': []}
    
    def detect_batch(self, frame_paths: List[str]) -> List[Dict[str, Any]]:
        """Detect objects in multiple frames."""
        return [self.detect(fp) for fp in frame_paths]

