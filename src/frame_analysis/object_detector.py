"""
Object Detector - Detects objects in video frames using YOLOv8
"""

import os
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass

# Cooking-related classes from COCO dataset that YOLOv8 can detect
COOKING_CLASSES = {
    # Food items
    'apple', 'orange', 'banana', 'broccoli', 'carrot', 'hot dog', 'pizza', 
    'donut', 'cake', 'sandwich', 'tomato', 'potato', 'onion', 'garlic',
    'lettuce', 'cucumber', 'pepper', 'cheese', 'egg', 'meat', 'chicken',
    'beef', 'pork', 'fish', 'shrimp', 'rice', 'pasta', 'bread', 'butter',
    'oil', 'salt', 'sugar', 'flour', 'herb', 'spice',
    
    # Kitchen tools and containers
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender',
    'mixer', 'cutting board', 'pan', 'pot', 'plate', 'whisk', 'grater',
    'colander', 'strainer', 'measuring cup', 'measuring spoon', 'spatula',
    'tongs', 'ladle', 'rolling pin', 'peeler', 'can opener', 'scale',
    'timer', 'thermometer',
    
    # People
    'person',
}

# Custom cooking classes for fine-tuned model
CUSTOM_COOKING_CLASSES = [
    # Common ingredients
    'tomato', 'potato', 'onion', 'garlic', 'ginger', 'carrot', 'bell_pepper',
    'chili_pepper', 'lettuce', 'spinach', 'kale', 'cabbage', 'broccoli',
    'cauliflower', 'mushroom', 'zucchini', 'eggplant', 'cucumber', 'corn',
    'peas', 'green_beans', 'asparagus', 'avocado', 'lemon', 'lime', 'orange',
    'apple', 'banana', 'berry', 'grape', 'watermelon', 'beef', 'chicken',
    'pork', 'fish', 'shrimp', 'egg', 'milk', 'cheese', 'yogurt', 'butter',
    'cream', 'flour', 'sugar', 'salt', 'pepper', 'oil', 'vinegar', 'soy_sauce',
    'ketchup', 'mustard', 'mayonnaise', 'honey', 'maple_syrup', 'rice', 'pasta',
    'bread', 'tortilla', 'noodle', 'bean', 'lentil', 'nut', 'seed',
    
    # Kitchen tools
    'knife', 'cutting_board', 'bowl', 'plate', 'pan', 'pot', 'skillet',
    'baking_sheet', 'baking_dish', 'measuring_cup', 'measuring_spoon', 'spoon',
    'fork', 'whisk', 'spatula', 'tongs', 'ladle', 'grater', 'peeler', 'can_opener',
    'bottle_opener', 'colander', 'strainer', 'mixer', 'blender', 'food_processor',
    'scale', 'timer', 'thermometer', 'oven_mitt', 'kitchen_towel', 'rolling_pin',
    'mortar_pestle', 'sieve', 'funnel', 'scissors', 'zester', 'masher',
    
    # Appliances
    'stove', 'oven', 'microwave', 'refrigerator', 'freezer', 'dishwasher',
    'toaster', 'coffee_maker', 'kettle', 'slow_cooker', 'pressure_cooker',
    'air_fryer', 'food_dehydrator', 'rice_cooker', 'waffle_maker', 'grill',
    
    # Actions
    'cutting', 'chopping', 'slicing', 'dicing', 'mincing', 'peeling', 'grating',
    'mixing', 'stirring', 'whisking', 'kneading', 'rolling', 'folding', 'pouring',
    'measuring', 'weighing', 'boiling', 'simmering', 'frying', 'sauteing',
    'baking', 'roasting', 'grilling', 'broiling', 'steaming', 'blanching',
    'marinating', 'seasoning', 'tasting', 'plating', 'serving'
]

class ObjectDetector:
    """
    Class for detecting objects in video frames using YOLOv8.
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.25):
        """
        Initialize the ObjectDetector.
        
        Args:
            model_path (str, optional): Path to the YOLOv8 model. Defaults to None (uses default model).
            confidence_threshold (float, optional): Confidence threshold for detections. Defaults to 0.25.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # Lazy load the model when needed
    
    def _load_model(self):
        """
        Load the YOLOv8 model with comprehensive error handling and fallbacks.
        """
        if self.model is not None:
            return
        
        try:
            from ultralytics import YOLO
            import os
            from pathlib import Path
            
            # Define models directory
            models_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / "models"
            
            # Try to load model with multiple fallback strategies
            model_loaded = False
            
            # Strategy 1: Use specified model path
            if self.model_path and os.path.exists(self.model_path):
                try:
                    logger.info(f"Loading YOLOv8 model from {self.model_path}")
                    self.model = YOLO(self.model_path)
                    model_loaded = True
                except Exception as e:
                    logger.warning(f"Failed to load specified model {self.model_path}: {e}")
            
            # Strategy 2: Try fine-tuned cooking model
            if not model_loaded:
                cooking_model_path = models_dir / "yolov8" / "yolov8_cooking.pt"
                if os.path.exists(cooking_model_path):
                    try:
                        logger.info(f"Loading fine-tuned cooking model from {cooking_model_path}")
                        self.model = YOLO(cooking_model_path)
                        model_loaded = True
                    except Exception as e:
                        logger.warning(f"Failed to load cooking model: {e}")
            
            # Strategy 3: Try local YOLOv8x model
            if not model_loaded:
                default_model_path = models_dir / "yolov8" / "yolov8x.pt"
                if os.path.exists(default_model_path):
                    try:
                        logger.info(f"Loading YOLOv8x model from {default_model_path}")
                        self.model = YOLO(default_model_path)
                        model_loaded = True
                    except Exception as e:
                        logger.warning(f"Failed to load local YOLOv8x model: {e}")
            
            # Strategy 4: Download and use default YOLOv8x model
            if not model_loaded:
                try:
                    logger.info("Downloading and loading default YOLOv8x model from ultralytics")
                    self.model = YOLO("yolov8x.pt")
                    model_loaded = True
                except Exception as e:
                    logger.warning(f"Failed to download YOLOv8x model: {e}")
            
            # Strategy 5: Use smaller model as last resort
            if not model_loaded:
                try:
                    logger.warning("Falling back to YOLOv8n (nano) model")
                    self.model = YOLO("yolov8n.pt")
                    model_loaded = True
                except Exception as e:
                    logger.error(f"Failed to load any YOLOv8 model: {e}")
                    raise ModelLoadError("Could not load any YOLOv8 model")
            
            if model_loaded:
                logger.info("YOLOv8 model loaded successfully")
                # Test the model with a dummy input to ensure it works
                try:
                    import numpy as np
                    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
                    _ = self.model(dummy_input, verbose=False)
                    logger.info("Model validation successful")
                except Exception as e:
                    logger.warning(f"Model validation failed: {e}")
                    
        except ImportError as e:
            logger.error(f"Ultralytics not installed: {e}")
            raise ModelLoadError("Ultralytics package not found. Please install with: pip install ultralytics")
        except Exception as e:
            logger.error(f"Unexpected error loading YOLOv8 model: {e}")
            raise ModelLoadError(f"Failed to load YOLOv8 model: {e}")
    
    def detect(self, frames):
        """
        Detect objects in a list of frames with comprehensive error handling.
        
        Args:
            frames (list): List of frames as numpy arrays.
            
        Returns:
            list: List of detection results for each frame.
                Each result is a dictionary with keys:
                - frame_idx: Index of the frame
                - detections: List of detected objects with class, confidence, and bounding box
        """
        if not frames:
            logger.warning("No frames provided for object detection")
            return []
        
        try:
            self._load_model()
        except ModelLoadError as e:
            logger.error(f"Model loading failed: {e}")
            # Return empty results for all frames
            return [{"frame_idx": i, "detections": []} for i in range(len(frames))]
        
        results = []
        
        try:
            for i, frame in enumerate(tqdm(frames, desc="Detecting objects")):
                try:
                    # Validate frame
                    if frame is None or frame.size == 0:
                        logger.warning(f"Invalid frame at index {i}, skipping")
                        results.append({"frame_idx": i, "detections": []})
                        continue
                    
                    # Ensure frame is in correct format
                    if len(frame.shape) != 3 or frame.shape[2] != 3:
                        logger.warning(f"Frame {i} has unexpected shape {frame.shape}, skipping")
                        results.append({"frame_idx": i, "detections": []})
                        continue
                    
                    # Run inference
                    yolo_results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                    
                    # Process results
                    frame_results = self._process_results(yolo_results, i)
                    results.append(frame_results)
                    
                except Exception as e:
                    logger.error(f"Error processing frame {i}: {e}")
                    # Add empty result for this frame
                    results.append({"frame_idx": i, "detections": []})
            
            logger.info(f"Object detection completed for {len(frames)} frames")
            return results
            
        except Exception as e:
            logger.error(f"Unexpected error during object detection: {e}")
            # Return empty results for all frames
            return [{"frame_idx": i, "detections": []} for i in range(len(frames))]
    
    def _process_results(self, yolo_results, frame_idx):
        """
        Process YOLOv8 results into a structured format.
        
        Args:
            yolo_results: Results from YOLOv8 model.
            frame_idx (int): Index of the frame.
            
        Returns:
            dict: Structured detection results.
        """
        detections = []
        
        # Process each detection
        for result in yolo_results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes.xyxy[i].cpu().numpy()
                
                # Get class and confidence
                cls_id = int(boxes.cls[i].item())
                cls_name = result.names[cls_id]
                confidence = boxes.conf[i].item()
                
                # Filter for cooking-related classes if needed
                if cls_name.lower() not in COOKING_CLASSES:
                    continue
                
                # Create detection object
                detection = {
                    "class": cls_name,
                    "confidence": confidence,
                    "box": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3])
                    }
                }
                
                detections.append(detection)
        
        return {
            "frame_idx": frame_idx,
            "detections": detections
        }
    
    def filter_cooking_related(self, detections):
        """
        Filter detections to only include cooking-related objects.
        
        Args:
            detections (list): List of detection results.
            
        Returns:
            list: Filtered detection results.
        """
        filtered_results = []
        
        for frame_result in detections:
            filtered_detections = [
                d for d in frame_result["detections"]
                if d["class"].lower() in COOKING_CLASSES
            ]
            
            filtered_results.append({
                "frame_idx": frame_result["frame_idx"],
                "detections": filtered_detections
            })
        
        return filtered_results
    
    def fine_tune(self, dataset_path, epochs=10, batch_size=16):
        """
        Fine-tune the YOLOv8 model on a custom cooking dataset.
        
        Args:
            dataset_path (str): Path to the dataset in YOLO format.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            batch_size (int, optional): Batch size for training. Defaults to 16.
            
        Returns:
            bool: True if fine-tuning was successful, False otherwise.
        """
        try:
            self._load_model()
            
            # Define output directory
            import os
            from pathlib import Path
            models_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / "models"
            output_dir = models_dir / "yolov8" / "fine_tuned"
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Fine-tuning YOLOv8 model on {dataset_path} for {epochs} epochs")
            
            # Train the model
            results = self.model.train(
                data=dataset_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=640,
                project=str(output_dir),
                name="cooking_model",
                save=True
            )
            
            # Save the fine-tuned model
            fine_tuned_path = models_dir / "yolov8" / "yolov8_cooking.pt"
            self.model.save(fine_tuned_path)
            
            logger.info(f"Fine-tuned model saved to {fine_tuned_path}")
            
            # Update the model path
            self.model_path = str(fine_tuned_path)
            
            return True
        except Exception as e:
            logger.error(f"Error fine-tuning YOLOv8 model: {e}")
            return False
    
    def detect_with_postprocessing(self, frames, apply_nms=True, iou_threshold=0.45):
        """
        Detect objects in frames with additional post-processing.
        
        Args:
            frames (list): List of frames as numpy arrays.
            apply_nms (bool, optional): Whether to apply non-maximum suppression. Defaults to True.
            iou_threshold (float, optional): IoU threshold for NMS. Defaults to 0.45.
            
        Returns:
            list: List of detection results for each frame.
        """
        # Get raw detections
        results = self.detect(frames)
        
        # Apply post-processing
        if apply_nms:
            results = self._apply_nms(results, iou_threshold)
        
        # Filter for cooking-related objects
        results = self.filter_cooking_related(results)
        
        # Track objects across frames
        results = self._track_objects(results)
        
        return results
    
    def _apply_nms(self, results, iou_threshold=0.45):
        """
        Apply non-maximum suppression to detection results.
        
        Args:
            results (list): Detection results.
            iou_threshold (float, optional): IoU threshold. Defaults to 0.45.
            
        Returns:
            list: Processed detection results.
        """
        import numpy as np
        from ultralytics.utils.ops import non_max_suppression
        
        processed_results = []
        
        for frame_result in results:
            detections = frame_result["detections"]
            
            if not detections:
                processed_results.append(frame_result)
                continue
            
            # Convert detections to format expected by NMS
            boxes = np.array([[d["box"]["x1"], d["box"]["y1"], d["box"]["x2"], d["box"]["y2"]] for d in detections])
            scores = np.array([d["confidence"] for d in detections])
            classes = np.array([d["class"] for d in detections])
            
            # Apply NMS
            indices = non_max_suppression(
                boxes, 
                scores, 
                iou_threshold=iou_threshold
            )
            
            # Create new detections list
            new_detections = [detections[i] for i in indices]
            
            processed_results.append({
                "frame_idx": frame_result["frame_idx"],
                "detections": new_detections
            })
        
        return processed_results
    
    def _track_objects(self, results):
        """
        Track objects across frames.
        
        Args:
            results (list): Detection results.
            
        Returns:
            list: Detection results with tracking information.
        """
        # Simple tracking based on IoU overlap
        tracked_results = []
        object_tracks = {}  # Dictionary to store object tracks
        next_track_id = 0
        
        for frame_idx, frame_result in enumerate(results):
            detections = frame_result["detections"]
            tracked_detections = []
            
            for detection in detections:
                box = [detection["box"]["x1"], detection["box"]["y1"], 
                       detection["box"]["x2"], detection["box"]["y2"]]
                cls = detection["class"]
                
                # Try to match with existing tracks
                matched_track_id = None
                max_iou = 0.5  # Minimum IoU threshold for matching
                
                for track_id, track in object_tracks.items():
                    if track["class"] != cls:
                        continue
                    
                    if frame_idx - track["last_seen"] > 5:  # Skip if track is too old
                        continue
                    
                    last_box = track["boxes"][-1]
                    iou = self._calculate_iou(box, last_box)
                    
                    if iou > max_iou:
                        max_iou = iou
                        matched_track_id = track_id
                
                # If matched, update track
                if matched_track_id is not None:
                    track = object_tracks[matched_track_id]
                    track["boxes"].append(box)
                    track["last_seen"] = frame_idx
                    detection["track_id"] = matched_track_id
                # Otherwise, create new track
                else:
                    track_id = next_track_id
                    next_track_id += 1
                    
                    object_tracks[track_id] = {
                        "class": cls,
                        "boxes": [box],
                        "first_seen": frame_idx,
                        "last_seen": frame_idx
                    }
                    
                    detection["track_id"] = track_id
                
                tracked_detections.append(detection)
            
            tracked_results.append({
                "frame_idx": frame_result["frame_idx"],
                "detections": tracked_detections
            })
        
        return tracked_results
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two boxes.
        
        Args:
            box1 (list): First box [x1, y1, x2, y2].
            box2 (list): Second box [x1, y1, x2, y2].
            
        Returns:
            float: IoU value.
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
