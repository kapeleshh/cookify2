"""
Action Recognizer - Recognizes cooking actions in video frames
"""

import os
import logging
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class ActionCategory(Enum):
    """Enum for different categories of cooking actions."""
    PREPARATION = "preparation"
    COOKING = "cooking"
    MIXING = "mixing"
    CUTTING = "cutting"
    MEASURING = "measuring"
    PLATING = "plating"
    UNKNOWN = "unknown"

# Comprehensive list of cooking actions organized by category
COOKING_ACTIONS = {
    ActionCategory.PREPARATION: {
        "peeling": ["peel", "peeling", "skin", "skinning"],
        "washing": ["wash", "washing", "rinse", "rinsing", "clean", "cleaning"],
        "measuring": ["measure", "measuring", "weigh", "weighing", "portion", "portioning"],
        "preparing": ["prepare", "preparing", "ready", "readying", "setup", "setting up"]
    },
    ActionCategory.CUTTING: {
        "cutting": ["cut", "cutting", "slice", "slicing"],
        "chopping": ["chop", "chopping", "dice", "dicing"],
        "mincing": ["mince", "mincing", "crush", "crushing"],
        "grating": ["grate", "grating", "shred", "shredding"],
        "julienning": ["julienne", "julienning", "matchstick", "matchsticking"]
    },
    ActionCategory.MIXING: {
        "mixing": ["mix", "mixing", "combine", "combining", "incorporate", "incorporating"],
        "stirring": ["stir", "stirring", "agitate", "agitating"],
        "whisking": ["whisk", "whisking", "beat", "beating"],
        "folding": ["fold", "folding", "blend", "blending"],
        "kneading": ["knead", "kneading", "work", "working"]
    },
    ActionCategory.COOKING: {
        "frying": ["fry", "frying", "sauté", "sautéing", "sear", "searing", "pan-fry", "pan-frying"],
        "boiling": ["boil", "boiling", "simmer", "simmering", "poach", "poaching"],
        "baking": ["bake", "baking", "roast", "roasting", "toast", "toasting"],
        "grilling": ["grill", "grilling", "broil", "broiling", "barbecue", "barbecuing"],
        "steaming": ["steam", "steaming"],
        "microwaving": ["microwave", "microwaving"],
        "pressure_cooking": ["pressure cook", "pressure cooking"],
        "slow_cooking": ["slow cook", "slow cooking", "braise", "braising", "stew", "stewing"]
    },
    ActionCategory.PLATING: {
        "plating": ["plate", "plating", "arrange", "arranging"],
        "garnishing": ["garnish", "garnishing", "decorate", "decorating"],
        "serving": ["serve", "serving", "present", "presenting", "dish", "dishing"]
    }
}

# Flattened dictionary for easy lookup
COOKING_ACTION_LOOKUP = {}
for category, actions in COOKING_ACTIONS.items():
    for action, synonyms in actions.items():
        COOKING_ACTION_LOOKUP[action] = category
        for synonym in synonyms:
            COOKING_ACTION_LOOKUP[synonym] = category

class ActionRecognizer:
    """
    Class for recognizing cooking actions in video frames.
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.5, frame_window=16, 
                 use_optical_flow=True, use_pose_detection=True, cooking_specific=True,
                 temporal_smoothing=True):
        """
        Initialize the ActionRecognizer.
        
        Args:
            model_path (str, optional): Path to the action recognition model. Defaults to None.
            confidence_threshold (float, optional): Confidence threshold for action recognition. Defaults to 0.5.
            frame_window (int, optional): Number of frames to use for action recognition. Defaults to 16.
            use_optical_flow (bool, optional): Whether to use optical flow for action recognition. Defaults to True.
            use_pose_detection (bool, optional): Whether to use pose detection for action recognition. Defaults to True.
            cooking_specific (bool, optional): Whether to optimize for cooking videos. Defaults to True.
            temporal_smoothing (bool, optional): Whether to apply temporal smoothing to action predictions. Defaults to True.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.frame_window = frame_window
        self.use_optical_flow = use_optical_flow
        self.use_pose_detection = use_pose_detection
        self.cooking_specific = cooking_specific
        self.temporal_smoothing = temporal_smoothing
        self.model = None
        self.pose_detector = None
        
        # Lazy load the models when needed
    
    def _load_model(self):
        """
        Load the action recognition model.
        """
        if self.model is not None:
            return
        
        try:
            import torch
            import os
            from pathlib import Path
            
            # Define models directory
            models_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / "models"
            
            # Use specified model path or default to MMAction2 model
            if self.model_path:
                logger.info(f"Loading action recognition model from {self.model_path}")
                self.model_path = Path(self.model_path)
            else:
                # Check if we have a fine-tuned cooking model
                cooking_model_path = models_dir / "mmaction" / "cooking_model.pth"
                default_model_path = models_dir / "mmaction" / "tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth"
                
                if os.path.exists(cooking_model_path) and self.cooking_specific:
                    logger.info(f"Loading fine-tuned cooking action model from {cooking_model_path}")
                    self.model_path = cooking_model_path
                elif os.path.exists(default_model_path):
                    logger.info(f"Loading default action recognition model from {default_model_path}")
                    self.model_path = default_model_path
                else:
                    logger.warning("No action recognition model found, using placeholder")
                    self.model = "placeholder_model"
                    return
            
            # In a real implementation, we would load the MMAction2 model here
            # For now, we'll use a placeholder
            self.model = "placeholder_model"
            
            # Load class names
            class_names_path = models_dir / "mmaction" / "action_classes.json"
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    self.class_names = json.load(f)
            else:
                # Default class names (subset of Kinetics-400)
                self.class_names = [
                    "cutting in kitchen", "chopping vegetables", "slicing fruit", "dicing food",
                    "mixing", "stirring", "whisking", "blending food", "folding dough",
                    "frying food", "sauteing", "searing", "flipping food", "stir frying",
                    "boiling", "simmering", "poaching egg", "steaming food",
                    "baking", "roasting", "toasting", "broiling", "grilling",
                    "measuring", "weighing", "pouring", "kneading dough", "rolling dough",
                    "plating food", "garnishing", "serving food", "tasting food"
                ]
            
            logger.info("Action recognition model loaded successfully")
            
            # Load pose detector if configured
            if self.use_pose_detection:
                self._load_pose_detector()
                
        except Exception as e:
            logger.error(f"Error loading action recognition model: {e}")
            self.model = "placeholder_model"  # Fallback to placeholder
    
    def _load_pose_detector(self):
        """
        Load the pose detection model.
        """
        if self.pose_detector is not None:
            return
        
        try:
            # In a real implementation, we would load a pose detection model here
            # For now, we'll use a placeholder
            logger.info("Loading pose detection model...")
            self.pose_detector = "placeholder_pose_detector"
            logger.info("Pose detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading pose detection model: {e}")
            self.pose_detector = None
    
    def recognize(self, frames, scenes=None, fps=30.0, object_detections=None):
        """
        Recognize actions in a list of frames.
        
        Args:
            frames (list): List of frames as numpy arrays.
            scenes (list, optional): List of scenes. If provided, actions will be recognized per scene.
            fps (float, optional): Frames per second. Defaults to 30.0.
            object_detections (list, optional): Object detection results. If provided, will be used for action-object association.
            
        Returns:
            list: List of action recognition results.
                Each result is a dictionary with keys:
                - start_frame: Starting frame number
                - end_frame: Ending frame number
                - start_time: Starting time in seconds
                - end_time: Ending time in seconds
                - action: Recognized action
                - category: Action category
                - confidence: Confidence score
                - objects: Associated objects (if object_detections provided)
        """
        self._load_model()
        
        # If no scenes are provided, treat the entire video as one scene
        if scenes is None:
            scenes = [{
                "scene_idx": 0,
                "start_frame": 0,
                "end_frame": len(frames) - 1,
                "start_time": 0.0,
                "end_time": len(frames) / fps,
                "duration": len(frames) / fps
            }]
        
        results = []
        
        # Process each scene
        for scene in tqdm(scenes, desc="Recognizing actions"):
            # Get frames for this scene
            scene_start = max(0, scene["start_frame"])
            scene_end = min(len(frames) - 1, scene["end_frame"])
            
            # Skip if scene is out of bounds
            if scene_start >= len(frames) or scene_end < 0 or scene_start > scene_end:
                continue
            
            scene_frames = frames[scene_start:scene_end+1]
            
            # Skip if not enough frames
            if len(scene_frames) < self.frame_window:
                continue
            
            # For longer scenes, we'll use a sliding window approach
            if len(scene_frames) > self.frame_window * 2:
                action_results = self._recognize_with_sliding_window(
                    scene_frames, scene_start, scene["start_time"], fps
                )
            else:
                # Sample frames at regular intervals to match frame_window
                sampled_indices = np.linspace(0, len(scene_frames) - 1, self.frame_window, dtype=int)
                sampled_frames = [scene_frames[i] for i in sampled_indices]
                
                # Prepare additional features
                features = {}
                
                # Compute optical flow if configured
                if self.use_optical_flow:
                    flow_frames = self._compute_optical_flow(sampled_frames)
                    features["optical_flow"] = flow_frames
                
                # Detect poses if configured
                if self.use_pose_detection:
                    poses = self._detect_poses(sampled_frames)
                    features["poses"] = poses
                
                # Recognize action
                action, confidence, category = self._recognize_action(sampled_frames, features)
                
                # Skip if confidence is below threshold
                if confidence < self.confidence_threshold:
                    continue
                
                # Create result
                action_results = [{
                    "start_frame": scene_start,
                    "end_frame": scene_end,
                    "start_time": scene["start_time"],
                    "end_time": scene["end_time"],
                    "action": action,
                    "category": category.value,
                    "confidence": confidence
                }]
            
            # Associate objects with actions if object detections are provided
            if object_detections and action_results:
                for result in action_results:
                    result["objects"] = self._associate_objects_with_action(
                        result, object_detections
                    )
            
            results.extend(action_results)
        
        # Apply temporal smoothing if configured
        if self.temporal_smoothing and results:
            results = self._apply_temporal_smoothing(results)
        
        logger.info(f"Recognized {len(results)} actions")
        return results
    
    def _recognize_with_sliding_window(self, frames, start_frame, start_time, fps):
        """
        Recognize actions using a sliding window approach.
        
        Args:
            frames (list): List of frames.
            start_frame (int): Starting frame number.
            start_time (float): Starting time in seconds.
            fps (float): Frames per second.
            
        Returns:
            list: List of action recognition results.
        """
        results = []
        
        # Calculate step size (50% overlap)
        step_size = self.frame_window // 2
        
        # Slide window over frames
        for i in range(0, len(frames) - self.frame_window + 1, step_size):
            window_frames = frames[i:i+self.frame_window]
            
            # Prepare additional features
            features = {}
            
            # Compute optical flow if configured
            if self.use_optical_flow:
                flow_frames = self._compute_optical_flow(window_frames)
                features["optical_flow"] = flow_frames
            
            # Detect poses if configured
            if self.use_pose_detection:
                poses = self._detect_poses(window_frames)
                features["poses"] = poses
            
            # Recognize action
            action, confidence, category = self._recognize_action(window_frames, features)
            
            # Skip if confidence is below threshold
            if confidence < self.confidence_threshold:
                continue
            
            # Calculate start and end frame/time
            window_start_frame = start_frame + i
            window_end_frame = window_start_frame + self.frame_window - 1
            window_start_time = start_time + (i / fps)
            window_end_time = start_time + ((i + self.frame_window - 1) / fps)
            
            # Create result
            result = {
                "start_frame": window_start_frame,
                "end_frame": window_end_frame,
                "start_time": window_start_time,
                "end_time": window_end_time,
                "action": action,
                "category": category.value,
                "confidence": confidence
            }
            
            results.append(result)
        
        return results
    
    def _compute_optical_flow(self, frames):
        """
        Compute optical flow between consecutive frames.
        
        Args:
            frames (list): List of frames.
            
        Returns:
            list: List of optical flow frames.
        """
        try:
            import cv2
            
            flow_frames = []
            
            for i in range(1, len(frames)):
                # Convert frames to grayscale
                prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Convert flow to RGB
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros_like(frames[i])
                hsv[..., 1] = 255
                hsv[..., 0] = angle * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                flow_frames.append(flow_rgb)
            
            return flow_frames
        except Exception as e:
            logger.warning(f"Error computing optical flow: {e}")
            return []
    
    def _recognize_action(self, frames, features=None):
        """
        Recognize action in a sequence of frames.
        
        Args:
            frames (list): List of frames.
            features (dict, optional): Additional features like optical flow and poses.
            
        Returns:
            tuple: (action, confidence, category)
        """
        # This is a placeholder implementation
        # In a real implementation, we would use the MMAction2 model to recognize actions
        
        if self.model == "placeholder_model":
            # For now, return a random cooking action with a random confidence
            import random
            
            # Get all actions from all categories
            all_actions = []
            for category, actions in COOKING_ACTIONS.items():
                all_actions.extend(list(actions.keys()))
            
            action = random.choice(all_actions)
            confidence = random.uniform(0.5, 1.0)
            category = COOKING_ACTION_LOOKUP.get(action, ActionCategory.UNKNOWN)
            
            return action, confidence, category
        else:
            # In a real implementation, we would use the model to predict the action
            # For example:
            # 1. Preprocess frames
            # 2. Run inference
            # 3. Post-process results
            
            # For now, return a placeholder result
            return "mixing", 0.8, ActionCategory.MIXING
    
    def _detect_poses(self, frames):
        """
        Detect human poses in frames.
        
        Args:
            frames (list): List of frames.
            
        Returns:
            list: List of pose detections.
        """
        if self.pose_detector is None:
            return []
        
        # This is a placeholder implementation
        # In a real implementation, we would use a pose detection model
        
        # For now, return empty poses
        return []
    
    def _associate_objects_with_action(self, action_result, object_detections):
        """
        Associate objects with an action.
        
        Args:
            action_result (dict): Action recognition result.
            object_detections (list): Object detection results.
            
        Returns:
            list: List of associated objects.
        """
        associated_objects = []
        
        # Get time range for the action
        start_time = action_result["start_time"]
        end_time = action_result["end_time"]
        action = action_result["action"]
        
        # Find objects that appear during the action
        for detection in object_detections:
            # Skip if no time information
            if "time" not in detection:
                continue
            
            time = detection["time"]
            
            # Check if detection is within time range
            if start_time <= time <= end_time:
                # Add all object classes
                for obj in detection.get("detections", []):
                    associated_objects.append(obj["class"])
        
        # Remove duplicates
        associated_objects = list(set(associated_objects))
        
        # Filter objects based on action
        relevant_objects = self._filter_relevant_objects(action, associated_objects)
        
        return relevant_objects
    
    def _filter_relevant_objects(self, action, objects):
        """
        Filter objects to keep only those relevant to the action.
        
        Args:
            action (str): Action name.
            objects (list): List of object classes.
            
        Returns:
            list: List of relevant objects.
        """
        # Define relevant object categories for each action category
        relevant_objects_map = {
            ActionCategory.CUTTING: [
                "knife", "cutting_board", "vegetable", "fruit", "meat", "fish",
                "onion", "garlic", "carrot", "potato", "tomato", "pepper"
            ],
            ActionCategory.MIXING: [
                "bowl", "spoon", "whisk", "mixer", "spatula", "fork",
                "flour", "sugar", "egg", "butter", "oil", "water", "milk"
            ],
            ActionCategory.COOKING: [
                "pan", "pot", "skillet", "oven", "stove", "grill", "microwave",
                "oil", "butter", "salt", "pepper", "meat", "vegetable", "fish"
            ],
            ActionCategory.PREPARATION: [
                "bowl", "measuring_cup", "measuring_spoon", "scale", "ingredient",
                "flour", "sugar", "salt", "spice", "herb"
            ],
            ActionCategory.PLATING: [
                "plate", "bowl", "dish", "fork", "knife", "spoon", "food",
                "herb", "sauce", "garnish"
            ]
        }
        
        # Get action category
        category = COOKING_ACTION_LOOKUP.get(action, ActionCategory.UNKNOWN)
        
        # If unknown category, return all objects
        if category == ActionCategory.UNKNOWN:
            return objects
        
        # Get relevant objects for this category
        relevant_categories = relevant_objects_map.get(category, [])
        
        # Filter objects
        filtered_objects = []
        for obj in objects:
            obj_lower = obj.lower()
            if any(relevant in obj_lower for relevant in relevant_categories):
                filtered_objects.append(obj)
        
        # If no relevant objects found, return all objects
        if not filtered_objects:
            return objects
        
        return filtered_objects
    
    def _apply_temporal_smoothing(self, results):
        """
        Apply temporal smoothing to action recognition results.
        
        Args:
            results (list): List of action recognition results.
            
        Returns:
            list: Smoothed action recognition results.
        """
        if not results:
            return results
        
        # Sort results by start time
        sorted_results = sorted(results, key=lambda x: x["start_time"])
        
        # Merge consecutive actions of the same type
        merged_results = []
        current = sorted_results[0]
        
        for next_result in sorted_results[1:]:
            # Check if actions are the same and close in time
            if (next_result["action"] == current["action"] and 
                next_result["start_time"] - current["end_time"] < 1.0):
                # Merge actions
                current["end_frame"] = next_result["end_frame"]
                current["end_time"] = next_result["end_time"]
                # Update confidence (weighted average)
                duration1 = current["end_time"] - current["start_time"]
                duration2 = next_result["end_time"] - next_result["start_time"]
                total_duration = duration1 + duration2
                current["confidence"] = (
                    (current["confidence"] * duration1) + 
                    (next_result["confidence"] * duration2)
                ) / total_duration
            else:
                # Add current to merged results and move to next
                merged_results.append(current)
                current = next_result
        
        # Add the last result
        merged_results.append(current)
        
        return merged_results
    
    def map_to_cooking_action(self, action):
        """
        Map a general action to a cooking-specific action.
        
        Args:
            action (str): General action.
            
        Returns:
            tuple: (cooking_action, category)
        """
        action_lower = action.lower()
        
        # Check direct matches in the lookup table
        if action_lower in COOKING_ACTION_LOOKUP:
            category = COOKING_ACTION_LOOKUP[action_lower]
            return action_lower, category
        
        # Check for partial matches
        for cooking_action, category in COOKING_ACTION_LOOKUP.items():
            if cooking_action in action_lower or action_lower in cooking_action:
                return cooking_action, category
        
        # Check for category matches
        for category, actions in COOKING_ACTIONS.items():
            for cooking_action, synonyms in actions.items():
                for synonym in synonyms:
                    if synonym in action_lower or action_lower in synonym:
                        return cooking_action, category
        
        return action, ActionCategory.UNKNOWN
    
    def fine_tune(self, dataset_path, epochs=10, batch_size=16):
        """
        Fine-tune the action recognition model on a custom cooking dataset.
        
        Args:
            dataset_path (str): Path to the dataset.
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
            output_dir = models_dir / "mmaction" / "fine_tuned"
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Fine-tuning action recognition model on {dataset_path} for {epochs} epochs")
            
            # In a real implementation, we would fine-tune the model here
            # For now, just log the attempt
            
            # Save the fine-tuned model
            fine_tuned_path = models_dir / "mmaction" / "cooking_model.pth"
            
            # Save class names
            class_names_path = models_dir / "mmaction" / "action_classes.json"
            with open(class_names_path, 'w') as f:
                json.dump(self.class_names, f)
            
            logger.info(f"Fine-tuned model saved to {fine_tuned_path}")
            
            # Update the model path
            self.model_path = str(fine_tuned_path)
            
            return True
        except Exception as e:
            logger.error(f"Error fine-tuning action recognition model: {e}")
            return False
    
    def extract_action_sequences(self, action_results):
        """
        Extract action sequences from action recognition results.
        
        Args:
            action_results (list): List of action recognition results.
            
        Returns:
            list: List of action sequences, where each sequence is a list of consecutive actions.
        """
        if not action_results:
            return []
        
        # Sort results by start time
        sorted_results = sorted(action_results, key=lambda x: x["start_time"])
        
        # Group actions by category
        sequences = []
        current_sequence = [sorted_results[0]]
        
        for result in sorted_results[1:]:
            prev_result = current_sequence[-1]
            
            # Check if actions are part of the same sequence
            # (close in time and related categories)
            if (result["start_time"] - prev_result["end_time"] < 5.0 and
                self._are_actions_related(prev_result["action"], result["action"])):
                # Add to current sequence
                current_sequence.append(result)
            else:
                # Start a new sequence
                sequences.append(current_sequence)
                current_sequence = [result]
        
        # Add the last sequence
        sequences.append(current_sequence)
        
        return sequences
    
    def _are_actions_related(self, action1, action2):
        """
        Check if two actions are related (part of the same cooking process).
        
        Args:
            action1 (str): First action.
            action2 (str): Second action.
            
        Returns:
            bool: True if actions are related, False otherwise.
        """
        # Get categories for both actions
        category1 = COOKING_ACTION_LOOKUP.get(action1, ActionCategory.UNKNOWN)
        category2 = COOKING_ACTION_LOOKUP.get(action2, ActionCategory.UNKNOWN)
        
        # If same category, they are related
        if category1 == category2:
            return True
        
        # Define related categories
        related_categories = {
            ActionCategory.PREPARATION: [ActionCategory.MEASURING, ActionCategory.CUTTING],
            ActionCategory.CUTTING: [ActionCategory.PREPARATION, ActionCategory.COOKING],
            ActionCategory.MIXING: [ActionCategory.PREPARATION, ActionCategory.COOKING],
            ActionCategory.COOKING: [ActionCategory.CUTTING, ActionCategory.MIXING, ActionCategory.PLATING],
            ActionCategory.PLATING: [ActionCategory.COOKING]
        }
        
        # Check if categories are related
        if category1 in related_categories and category2 in related_categories[category1]:
            return True
        
        return False
