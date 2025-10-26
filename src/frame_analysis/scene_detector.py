"""
Scene Detector - Detect scene changes in videos using PySceneDetect
"""

import logging
from typing import List, Dict, Any
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

logger = logging.getLogger(__name__)


class SceneDetector:
    """
    Detects scene changes in videos using PySceneDetect.
    """
    
    def __init__(self, threshold: float = 27.0, min_scene_len: int = 15):
        """
        Initialize the SceneDetector.
        
        Args:
            threshold (float): Threshold for scene detection (0-255).
            min_scene_len (int): Minimum scene length in frames.
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        
        logger.info(f"SceneDetector initialized with threshold={threshold}, min_scene_len={min_scene_len}")
    
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
        """
        try:
            # Create video manager and scene manager
            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()
            
            # Add content detector
            scene_manager.add_detector(
                ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len)
            )
            
            # Set downscale factor for faster processing
            video_manager.set_downscale_factor()
            
            # Start video manager
            video_manager.start()
            
            # Perform scene detection
            scene_manager.detect_scenes(frame_source=video_manager)
            
            # Get scene list
            scene_list = scene_manager.get_scene_list()
            
            # Get FPS
            fps = video_manager.get_framerate()
            
            # Release video manager
            video_manager.release()
            
            # Convert scene list to our format
            scenes = []
            for i, scene in enumerate(scene_list):
                start_frame = scene[0].get_frames()
                end_frame = scene[1].get_frames() - 1  # Inclusive end frame
                
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                
                scenes.append({
                    "scene_idx": i,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time
                })
            
            logger.info(f"Detected {len(scenes)} scenes in video")
            
            return scenes
            
        except Exception as e:
            logger.error(f"Error detecting scenes: {e}")
            # Return a single scene covering the entire video as fallback
            return [{
                "scene_idx": 0,
                "start_frame": 0,
                "end_frame": 0,
                "start_time": 0.0,
                "end_time": 0.0,
                "duration": 0.0
            }]
    
    def get_scene_for_frame(self, scenes: List[Dict[str, Any]], frame_idx: int) -> Dict[str, Any]:
        """
        Get the scene that contains a specific frame.
        
        Args:
            scenes (list): List of scenes.
            frame_idx (int): Frame index.
            
        Returns:
            dict: Scene dictionary, or None if not found.
        """
        for scene in scenes:
            if scene["start_frame"] <= frame_idx <= scene["end_frame"]:
                return scene
        return None
    
    def get_scene_for_timestamp(self, scenes: List[Dict[str, Any]], timestamp: float) -> Dict[str, Any]:
        """
        Get the scene that contains a specific timestamp.
        
        Args:
            scenes (list): List of scenes.
            timestamp (float): Timestamp in seconds.
            
        Returns:
            dict: Scene dictionary, or None if not found.
        """
        for scene in scenes:
            if scene["start_time"] <= timestamp <= scene["end_time"]:
                return scene
        return None

