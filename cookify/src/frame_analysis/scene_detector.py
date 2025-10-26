"""
Scene Detector - Detects scene changes in videos using SceneDetect
"""

import os
import logging
import numpy as np
from pathlib import Path
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class DetectorType(Enum):
    """Enum for different scene detection methods."""
    CONTENT = "content"
    THRESHOLD = "threshold"
    ADAPTIVE = "adaptive"
    COMBINED = "combined"

# Cooking scene types
class CookingSceneType(Enum):
    """Enum for different types of cooking scenes."""
    PREPARATION = "preparation"
    COOKING = "cooking"
    PLATING = "plating"
    INTRODUCTION = "introduction"
    INGREDIENTS = "ingredients"
    TOOLS = "tools"
    UNKNOWN = "unknown"

# Cooking scene characteristics
COOKING_SCENE_CHARACTERISTICS = {
    CookingSceneType.PREPARATION: {
        "description": "Preparing ingredients (chopping, mixing, etc.)",
        "typical_duration": (5, 60),  # seconds
        "typical_objects": ["knife", "cutting_board", "bowl", "ingredients"],
        "typical_actions": ["cutting", "chopping", "mixing", "preparing"]
    },
    CookingSceneType.COOKING: {
        "description": "Actual cooking process (frying, boiling, baking, etc.)",
        "typical_duration": (30, 300),  # seconds
        "typical_objects": ["pan", "pot", "stove", "oven"],
        "typical_actions": ["frying", "boiling", "baking", "cooking"]
    },
    CookingSceneType.PLATING: {
        "description": "Plating and presenting the final dish",
        "typical_duration": (5, 30),  # seconds
        "typical_objects": ["plate", "dish", "food"],
        "typical_actions": ["plating", "garnishing", "serving"]
    },
    CookingSceneType.INTRODUCTION: {
        "description": "Introduction to the recipe or chef",
        "typical_duration": (5, 30),  # seconds
        "typical_objects": ["person"],
        "typical_actions": ["talking", "introducing"]
    },
    CookingSceneType.INGREDIENTS: {
        "description": "Showing or listing ingredients",
        "typical_duration": (5, 30),  # seconds
        "typical_objects": ["ingredients"],
        "typical_actions": ["showing", "displaying"]
    },
    CookingSceneType.TOOLS: {
        "description": "Showing or listing tools",
        "typical_duration": (5, 20),  # seconds
        "typical_objects": ["tools", "utensils"],
        "typical_actions": ["showing", "displaying"]
    }
}

class SceneDetector:
    """
    Class for detecting scene changes in videos using SceneDetect.
    """
    
    def __init__(self, 
                 detector_type: DetectorType = DetectorType.COMBINED,
                 threshold: float = 30.0, 
                 min_scene_len: int = 15,
                 adaptive_threshold: float = 3.0,
                 window_size: int = 60,
                 luma_only: bool = True,
                 optimize_for_cooking: bool = True):
        """
        Initialize the SceneDetector.
        
        Args:
            detector_type (DetectorType, optional): Type of detector to use. Defaults to COMBINED.
            threshold (float, optional): Threshold for content detection. Defaults to 30.0.
            min_scene_len (int, optional): Minimum scene length in frames. Defaults to 15.
            adaptive_threshold (float, optional): Threshold for adaptive detection. Defaults to 3.0.
            window_size (int, optional): Window size for adaptive detection. Defaults to 60.
            luma_only (bool, optional): Whether to use only luma channel. Defaults to True.
            optimize_for_cooking (bool, optional): Whether to optimize parameters for cooking videos. Defaults to True.
        """
        self.detector_type = detector_type
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.adaptive_threshold = adaptive_threshold
        self.window_size = window_size
        self.luma_only = luma_only
        self.optimize_for_cooking = optimize_for_cooking
        
        # Optimize parameters for cooking videos if requested
        if optimize_for_cooking:
            self._optimize_for_cooking()
    
    def _optimize_for_cooking(self):
        """
        Optimize detector parameters for cooking videos.
        
        Cooking videos often have:
        - Gradual transitions between preparation steps
        - Similar backgrounds throughout
        - Focus on specific objects/actions
        """
        if self.detector_type == DetectorType.CONTENT:
            # Lower threshold for more sensitive detection of subtle changes
            self.threshold = 25.0
            # Increase min scene length to avoid detecting too many short scenes
            self.min_scene_len = 24  # About 1 second at 24fps
        elif self.detector_type == DetectorType.THRESHOLD:
            # Lower threshold for more sensitive detection
            self.threshold = 20.0
            self.min_scene_len = 24
        elif self.detector_type == DetectorType.ADAPTIVE:
            # Increase adaptive threshold for better handling of gradual changes
            self.adaptive_threshold = 2.5
            self.window_size = 90  # Larger window for context
            self.min_scene_len = 24
        elif self.detector_type == DetectorType.COMBINED:
            # Balanced parameters for combined detection
            self.threshold = 27.0
            self.adaptive_threshold = 2.8
            self.min_scene_len = 24
            self.window_size = 75
    
    def detect(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Detect scenes in a video.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            list: List of scenes, where each scene is a dictionary with:
                - scene_idx: Scene index
                - start_frame: Starting frame number
                - end_frame: Ending frame number
                - start_time: Starting time in seconds
                - end_time: Ending time in seconds
                - duration: Duration in seconds
                - scene_type: Type of cooking scene (if classify_scenes=True)
        """
        logger.info(f"Detecting scenes in video: {video_path}")
        
        try:
            from scenedetect import SceneManager, open_video
            from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector
            
            # Open video
            video = open_video(video_path)
            
            # Create scene manager
            scene_manager = SceneManager()
            
            # Add appropriate detector(s) based on detector_type
            if self.detector_type == DetectorType.CONTENT:
                scene_manager.add_detector(
                    ContentDetector(
                        threshold=self.threshold,
                        min_scene_len=self.min_scene_len,
                        luma_only=self.luma_only
                    )
                )
            elif self.detector_type == DetectorType.THRESHOLD:
                scene_manager.add_detector(
                    ThresholdDetector(
                        threshold=self.threshold,
                        min_scene_len=self.min_scene_len
                    )
                )
            elif self.detector_type == DetectorType.ADAPTIVE:
                try:
                    # Try with window_size parameter
                    scene_manager.add_detector(
                        AdaptiveDetector(
                            adaptive_threshold=self.adaptive_threshold,
                            min_scene_len=self.min_scene_len,
                            window_size=self.window_size,
                            luma_only=self.luma_only
                        )
                    )
                except TypeError:
                    # If window_size is not supported, try without it
                    logger.warning("AdaptiveDetector does not support window_size parameter, using default")
                    scene_manager.add_detector(
                        AdaptiveDetector(
                            adaptive_threshold=self.adaptive_threshold,
                            min_scene_len=self.min_scene_len,
                            luma_only=self.luma_only
                        )
                    )
            elif self.detector_type == DetectorType.COMBINED:
                # Add both content and adaptive detectors for better results
                scene_manager.add_detector(
                    ContentDetector(
                        threshold=self.threshold,
                        min_scene_len=self.min_scene_len,
                        luma_only=self.luma_only
                    )
                )
                
                try:
                    # Try with window_size parameter
                    scene_manager.add_detector(
                        AdaptiveDetector(
                            adaptive_threshold=self.adaptive_threshold,
                            min_scene_len=self.min_scene_len,
                            window_size=self.window_size,
                            luma_only=self.luma_only
                        )
                    )
                except TypeError:
                    # If window_size is not supported, try without it
                    logger.warning("AdaptiveDetector does not support window_size parameter, using default")
                    scene_manager.add_detector(
                        AdaptiveDetector(
                            adaptive_threshold=self.adaptive_threshold,
                            min_scene_len=self.min_scene_len,
                            luma_only=self.luma_only
                        )
                    )
            
            # Perform scene detection
            scene_manager.detect_scenes(video)
            
            # Get scene list and frame metrics
            scene_list = scene_manager.get_scene_list()
            
            # Get fps to calculate timestamps
            fps = video.frame_rate
            
            # Convert scene list to our format
            scenes = []
            for i, scene in enumerate(scene_list):
                start_frame = scene[0].frame_num
                end_frame = scene[1].frame_num - 1  # Inclusive end frame
                
                start_time = start_frame / fps
                end_time = end_frame / fps
                duration = end_time - start_time
                
                scenes.append({
                    "scene_idx": i,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "scene_type": CookingSceneType.UNKNOWN.value
                })
            
            # Merge similar consecutive scenes if optimizing for cooking
            if self.optimize_for_cooking:
                scenes = self._merge_similar_scenes(scenes)
            
            logger.info(f"Detected {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            logger.error(f"Error detecting scenes: {e}")
            # Return a single scene covering the entire video as fallback
            return [{
                "scene_idx": 0,
                "start_frame": 0,
                "end_frame": float('inf'),  # Will be replaced with actual frame count later
                "start_time": 0.0,
                "end_time": float('inf'),  # Will be replaced with actual duration later
                "duration": float('inf'),  # Will be replaced with actual duration later
                "scene_type": CookingSceneType.UNKNOWN.value
            }]
    
    def get_scene_for_frame(self, scenes, frame_idx):
        """
        Get the scene that contains a specific frame.
        
        Args:
            scenes (list): List of scenes.
            frame_idx (int): Frame index.
            
        Returns:
            dict: Scene containing the frame, or None if not found.
        """
        for scene in scenes:
            if scene["start_frame"] <= frame_idx <= scene["end_frame"]:
                return scene
        
        return None
    
    def get_scene_for_timestamp(self, scenes, timestamp):
        """
        Get the scene that contains a specific timestamp.
        
        Args:
            scenes (list): List of scenes.
            timestamp (float): Timestamp in seconds.
            
        Returns:
            dict: Scene containing the timestamp, or None if not found.
        """
        for scene in scenes:
            if scene["start_time"] <= timestamp <= scene["end_time"]:
                return scene
        
        return None
    
    def _merge_similar_scenes(self, scenes: List[Dict[str, Any]], 
                             max_gap: float = 1.0, 
                             similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Merge similar consecutive scenes that are likely part of the same cooking step.
        
        Args:
            scenes (list): List of scenes.
            max_gap (float, optional): Maximum time gap between scenes to consider merging. Defaults to 1.0.
            similarity_threshold (float, optional): Threshold for scene similarity. Defaults to 0.7.
            
        Returns:
            list: Merged scenes.
        """
        if len(scenes) <= 1:
            return scenes
        
        merged_scenes = [scenes[0]]
        
        for i in range(1, len(scenes)):
            current_scene = scenes[i]
            prev_scene = merged_scenes[-1]
            
            # Check if scenes are close in time
            time_gap = current_scene["start_time"] - prev_scene["end_time"]
            
            # If scenes are close and similar, merge them
            if time_gap <= max_gap and self._are_scenes_similar(prev_scene, current_scene, similarity_threshold):
                # Update end frame and time of previous scene
                prev_scene["end_frame"] = current_scene["end_frame"]
                prev_scene["end_time"] = current_scene["end_time"]
                prev_scene["duration"] = prev_scene["end_time"] - prev_scene["start_time"]
            else:
                merged_scenes.append(current_scene)
        
        # Renumber scene indices
        for i, scene in enumerate(merged_scenes):
            scene["scene_idx"] = i
        
        return merged_scenes
    
    def _are_scenes_similar(self, scene1: Dict[str, Any], scene2: Dict[str, Any], 
                           threshold: float = 0.7) -> bool:
        """
        Check if two scenes are similar based on duration and other characteristics.
        
        Args:
            scene1 (dict): First scene.
            scene2 (dict): Second scene.
            threshold (float, optional): Similarity threshold. Defaults to 0.7.
            
        Returns:
            bool: True if scenes are similar, False otherwise.
        """
        # For now, use a simple heuristic based on duration
        # In a real implementation, this would use visual features, object detections, etc.
        duration1 = scene1["duration"]
        duration2 = scene2["duration"]
        
        # If either duration is very short, they might be part of the same action
        if duration1 < 2.0 or duration2 < 2.0:
            return True
        
        # Calculate duration ratio (smaller / larger)
        ratio = min(duration1, duration2) / max(duration1, duration2)
        
        return ratio > threshold
    
    def classify_scenes(self, scenes: List[Dict[str, Any]], 
                       object_detections: Optional[List[Dict[str, Any]]] = None,
                       action_detections: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Classify scenes into cooking scene types based on duration, objects, and actions.
        
        Args:
            scenes (list): List of scenes.
            object_detections (list, optional): Object detection results. Defaults to None.
            action_detections (list, optional): Action detection results. Defaults to None.
            
        Returns:
            list: Scenes with added scene_type field.
        """
        classified_scenes = []
        
        for scene in scenes:
            scene_type = self._classify_single_scene(
                scene, object_detections, action_detections
            )
            
            # Create a copy of the scene with the scene_type added
            classified_scene = scene.copy()
            classified_scene["scene_type"] = scene_type.value
            classified_scenes.append(classified_scene)
        
        return classified_scenes
    
    def _classify_single_scene(self, scene: Dict[str, Any],
                              object_detections: Optional[List[Dict[str, Any]]] = None,
                              action_detections: Optional[List[Dict[str, Any]]] = None) -> CookingSceneType:
        """
        Classify a single scene based on its characteristics.
        
        Args:
            scene (dict): Scene to classify.
            object_detections (list, optional): Object detection results. Defaults to None.
            action_detections (list, optional): Action detection results. Defaults to None.
            
        Returns:
            CookingSceneType: Type of cooking scene.
        """
        # Extract scene information
        duration = scene["duration"]
        start_time = scene["start_time"]
        end_time = scene["end_time"]
        
        # Initialize scores for each scene type
        scores = {scene_type: 0.0 for scene_type in CookingSceneType}
        
        # Score based on duration
        for scene_type, characteristics in COOKING_SCENE_CHARACTERISTICS.items():
            min_duration, max_duration = characteristics["typical_duration"]
            
            # If duration is within typical range, increase score
            if min_duration <= duration <= max_duration:
                scores[scene_type] += 1.0
            # If close to range, add partial score
            elif duration < min_duration:
                scores[scene_type] += max(0, 1.0 - (min_duration - duration) / min_duration)
            else:  # duration > max_duration
                scores[scene_type] += max(0, 1.0 - (duration - max_duration) / max_duration)
        
        # Score based on objects if available
        if object_detections:
            scene_objects = self._get_objects_in_timerange(object_detections, start_time, end_time)
            
            for scene_type, characteristics in COOKING_SCENE_CHARACTERISTICS.items():
                typical_objects = characteristics["typical_objects"]
                
                # Count matching objects
                matching_objects = sum(1 for obj in scene_objects if any(typ in obj.lower() for typ in typical_objects))
                
                # Add score based on matching objects
                if scene_objects:
                    scores[scene_type] += matching_objects / len(scene_objects) * 2.0
        
        # Score based on actions if available
        if action_detections:
            scene_actions = self._get_actions_in_timerange(action_detections, start_time, end_time)
            
            for scene_type, characteristics in COOKING_SCENE_CHARACTERISTICS.items():
                typical_actions = characteristics["typical_actions"]
                
                # Count matching actions
                matching_actions = sum(1 for act in scene_actions if any(typ in act.lower() for typ in typical_actions))
                
                # Add score based on matching actions
                if scene_actions:
                    scores[scene_type] += matching_actions / len(scene_actions) * 3.0
        
        # Get scene type with highest score
        best_scene_type = max(scores.items(), key=lambda x: x[1])[0]
        
        # If score is too low, return UNKNOWN
        if scores[best_scene_type] < 1.0:
            return CookingSceneType.UNKNOWN
        
        return best_scene_type
    
    def _get_objects_in_timerange(self, object_detections: List[Dict[str, Any]], 
                                 start_time: float, end_time: float) -> List[str]:
        """
        Get objects detected within a time range.
        
        Args:
            object_detections (list): Object detection results.
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.
            
        Returns:
            list: List of object classes.
        """
        objects = []
        
        for detection in object_detections:
            # Skip if no time information
            if "time" not in detection:
                continue
            
            time = detection["time"]
            
            # Check if detection is within time range
            if start_time <= time <= end_time:
                # Add all object classes
                for obj in detection.get("detections", []):
                    objects.append(obj["class"])
        
        return objects
    
    def _get_actions_in_timerange(self, action_detections: List[Dict[str, Any]], 
                                 start_time: float, end_time: float) -> List[str]:
        """
        Get actions detected within a time range.
        
        Args:
            action_detections (list): Action detection results.
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.
            
        Returns:
            list: List of action classes.
        """
        actions = []
        
        for detection in action_detections:
            # Skip if no time information
            if "start_time" not in detection or "end_time" not in detection:
                continue
            
            action_start = detection["start_time"]
            action_end = detection["end_time"]
            
            # Check if action overlaps with time range
            if (start_time <= action_start <= end_time) or \
               (start_time <= action_end <= end_time) or \
               (action_start <= start_time and action_end >= end_time):
                actions.append(detection["action"])
        
        return actions
