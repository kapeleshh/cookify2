"""
Object Detector - Detect objects in video frames using YOLOv8
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Detects objects in video frames using YOLOv8.
    """
    
    # Cooking-related COCO classes
    COOKING_CLASSES = [
        'person', 'bowl', 'cup', 'fork', 'knife', 'spoon', 'bottle',
        'wine glass', 'dining table', 'oven', 'sink', 'refrigerator',
        'microwave', 'toaster', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake'
    ]
    
    def __init__(self, model_path: str = "models/object_detection/yolov8n.pt",
                 confidence_threshold: float = 0.5):
        """
        Initialize the ObjectDetector.
        
        Args:
            model_path (str): Path to the YOLOv8 model.
            confidence_threshold (float): Confidence threshold for detections.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        logger.info(f"ObjectDetector initialized with threshold={confidence_threshold}")
    
    def _load_model(self):
        """
        Lazy-load the YOLOv8 model.
        """
        if self.model is None:
            try:
                from ultralytics import YOLO
                
                if not os.path.exists(self.model_path):
                    logger.warning(f"Model not found at {self.model_path}. Using default YOLOv8n.")
                    self.model = YOLO('yolov8n.pt')
                else:
                    self.model = YOLO(self.model_path)
                
                logger.info(f"YOLOv8 model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load YOLOv8 model: {e}")
                raise
    
    def detect(self, frames: List[Any]) -> List[Dict[str, Any]]:
        """
        Detect objects in a list of frames.
        
        Args:
            frames (list): List of frames as numpy arrays.
            
        Returns:
            list: List of detection results for each frame.
        """
        self._load_model()
        
        results = []
        
        for i, frame in enumerate(tqdm(frames, desc="Detecting objects")):
            try:
                # Run inference
                yolo_results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Process results
                frame_results = self._process_results(yolo_results, i)
                results.append(frame_results)
            except Exception as e:
                logger.warning(f"Error detecting objects in frame {i}: {e}")
                results.append({
                    "frame_idx": i,
                    "detections": []
                })
        
        return results
    
    def _process_results(self, yolo_results, frame_idx: int) -> Dict[str, Any]:
        """
        Process YOLOv8 results into a structured format.
        
        Args:
            yolo_results: YOLO detection results.
            frame_idx (int): Frame index.
            
        Returns:
            dict: Processed detection results.
        """
        detections = []
        
        for result in yolo_results:
            boxes = result.boxes
            
            for box in boxes:
                # Extract box information
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detections.append({
                    "class": class_name,
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })
        
        return {
            "frame_idx": frame_idx,
            "detections": detections
        }
    
    def filter_cooking_related(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter detections to focus on cooking-related objects.
        
        Args:
            detections (list): List of detection results.
            
        Returns:
            list: Filtered detection results.
        """
        filtered = []
        
        for frame_result in detections:
            filtered_detections = [
                det for det in frame_result["detections"]
                if det["class"] in self.COOKING_CLASSES
            ]
            
            filtered.append({
                "frame_idx": frame_result["frame_idx"],
                "detections": filtered_detections
            })
        
        return filtered

