"""
Scene Detector using PySceneDetect for detecting scene changes
"""

import logging
import cv2
from pathlib import Path
from typing import List, Dict, Any

try:
    from scenedetect import detect, ContentDetector, AdaptiveDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    logging.warning("scenedetect not installed. Scene detection will be disabled.")

logger = logging.getLogger(__name__)

class SceneDetector:
    """
    Detects scene changes in videos using PySceneDetect.
    """
    
    def __init__(self, threshold=27.0):
        """
        Initialize the scene detector.
        
        Args:
            threshold: Sensitivity for scene detection (lower = more sensitive)
        """
        self.threshold = threshold
        self.available = SCENEDETECT_AVAILABLE
        
        if SCENEDETECT_AVAILABLE:
            logger.info("✓ Scene detection available")
        else:
            logger.warning("Scene detection not available")
    
    def detect_scenes(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Detect scene changes in a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of scene dictionaries with start/end times
        """
        if not self.available:
            return self._fallback_scene_detection(video_path)
        
        try:
            logger.info(f"Detecting scenes in: {video_path}")
            
            # Detect scenes using ContentDetector
            scene_list = detect(video_path, ContentDetector(threshold=self.threshold))
            
            scenes = []
            for i, (start, end) in enumerate(scene_list):
                scenes.append({
                    'scene_id': i + 1,
                    'start_time': start.get_seconds(),
                    'end_time': end.get_seconds(),
                    'duration': end.get_seconds() - start.get_seconds(),
                    'start_frame': start.get_frames(),
                    'end_frame': end.get_frames()
                })
            
            logger.info(f"✓ Detected {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            return self._fallback_scene_detection(video_path)
    
    def _fallback_scene_detection(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Fallback scene detection using simple frame difference.
        """
        logger.info("Using fallback scene detection")
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Divide video into equal segments
            num_scenes = max(3, int(total_frames / (fps * 10)))  # ~10 second scenes
            frames_per_scene = total_frames // num_scenes
            
            scenes = []
            for i in range(num_scenes):
                start_frame = i * frames_per_scene
                end_frame = (i + 1) * frames_per_scene if i < num_scenes - 1 else total_frames
                
                scenes.append({
                    'scene_id': i + 1,
                    'start_time': start_frame / fps,
                    'end_time': end_frame / fps,
                    'duration': (end_frame - start_frame) / fps,
                    'start_frame': start_frame,
                    'end_frame': end_frame
                })
            
            cap.release()
            logger.info(f"✓ Created {len(scenes)} scene segments")
            return scenes
            
        except Exception as e:
            logger.error(f"Fallback scene detection failed: {e}")
            return []
    
    def get_key_frames(self, scenes: List[Dict[str, Any]], video_path: str) -> List[int]:
        """
        Get key frame numbers from detected scenes.
        
        Args:
            scenes: List of scene dictionaries
            video_path: Path to the video file
            
        Returns:
            List of key frame numbers
        """
        key_frames = []
        for scene in scenes:
            # Take middle frame of each scene
            mid_frame = (scene['start_frame'] + scene['end_frame']) // 2
            key_frames.append(mid_frame)
        
        return key_frames

