"""
Traditional Pipeline for Cookify (Using Classic CV: YOLO, OCR, SceneDetect)
Uses actual computer vision techniques without VLM
"""

import os
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import cv2
from collections import Counter

from src.preprocessing.video_processor import VideoProcessor
from src.frame_analysis.object_detector import ObjectDetector
from src.frame_analysis.text_recognizer import TextRecognizer
from src.frame_analysis.scene_detector import SceneDetector
from src.frame_analysis.action_recognizer import ActionRecognizer

logger = logging.getLogger(__name__)

class TraditionalPipeline:
    """
    Traditional pipeline using YOLO, OCR, and SceneDetect.
    """
    
    def __init__(self):
        """Initialize the traditional pipeline with CV components."""
        logger.info("Initializing Traditional Pipeline with CV components")
        
        # Initialize components
        self.video_processor = VideoProcessor()
        self.object_detector = ObjectDetector(model_name="yolov8n.pt", confidence_threshold=0.3)
        self.text_recognizer = TextRecognizer(languages=['en'], gpu=False)
        self.scene_detector = SceneDetector(threshold=27.0)
        self.action_recognizer = ActionRecognizer()
        
        logger.info("✓ Traditional pipeline initialized")
    
    def process_video(self, video_path: str, progress_callback=None) -> Dict[str, Any]:
        """
        Process video using traditional CV methods.
        
        Args:
            video_path: Path to the video file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing extracted recipe information
        """
        logger.info(f"Processing video (Traditional CV): {video_path}")
        start_time = time.time()
        
        result = {
            'video_path': video_path,
            'processing_time': 0,
            'frames_analyzed': 0,
            'recipe': {
                'title': 'Extracted Recipe (Traditional CV)',
                'ingredients': [],
                'steps': [],
                'tools': [],
                'cuisine': None,
                'estimated_time': None
            },
            'metadata': {
                'method': 'traditional_cv',
                'vlm_enabled': False,
                'techniques': ['yolo', 'easyocr', 'scenedetect', 'heuristics']
            }
        }
        
        try:
            # Step 1: Detect scenes
            if progress_callback:
                progress_callback('Detecting scenes...', 10)
            
            scenes = self.scene_detector.detect_scenes(video_path)
            logger.info(f"Detected {len(scenes)} scenes")
            
            # Step 2: Extract key frames
            if progress_callback:
                progress_callback('Extracting frames...', 20)
            
            frames = self._extract_key_frames(video_path, scenes)
            logger.info(f"Extracted {len(frames)} key frames")
            result['frames_analyzed'] = len(frames)
            
            # Step 3: Analyze frames with traditional CV
            if progress_callback:
                progress_callback('Analyzing frames with YOLO and OCR...', 30)
            
            frame_analysis = self._analyze_frames_cv(frames, progress_callback)
            
            # Step 4: Generate recipe from detections
            if progress_callback:
                progress_callback('Generating recipe from detections...', 85)
            
            recipe = self._generate_recipe_from_detections(frame_analysis, video_path)
            result['recipe'] = recipe
            
            result['processing_time'] = time.time() - start_time
            
            if progress_callback:
                progress_callback('Complete!', 100)
            
            logger.info(f"Video processed (Traditional CV) in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            result['error'] = str(e)
            return result
    
    def _extract_key_frames(self, video_path: str, scenes: List[Dict]) -> List[str]:
        """Extract key frames from detected scenes."""
        frames_dir = Path('data/temp/frames_traditional')
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean previous frames
        for f in frames_dir.glob('frame_*.jpg'):
            f.unlink()
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Extract middle frame from each scene
        for scene in scenes:
            mid_frame_num = (scene['start_frame'] + scene['end_frame']) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_num)
            ret, frame = cap.read()
            
            if ret:
                frame_path = frames_dir / f'frame_{len(frames):04d}.jpg'
                cv2.imwrite(str(frame_path), frame)
                frames.append(str(frame_path))
        
        cap.release()
        return frames[:15]  # Limit to 15 frames
    
    def _analyze_frames_cv(self, frames: List[str], progress_callback=None) -> Dict:
        """Analyze frames using YOLO, OCR, and action recognition."""
        analysis = {
            'objects_per_frame': [],
            'text_per_frame': [],
            'actions_per_frame': [],
            'all_objects': [],
            'all_measurements': [],
            'all_text': []
        }
        
        for i, frame_path in enumerate(frames):
            if progress_callback:
                progress = 30 + int((i / len(frames)) * 50)
                progress_callback(f'Analyzing frame {i+1}/{len(frames)}...', progress)
            
            # Object detection with YOLO
            obj_result = self.object_detector.detect(frame_path)
            analysis['objects_per_frame'].append(obj_result)
            analysis['all_objects'].extend(obj_result.get('cooking_related', []))
            
            # Text recognition with OCR
            text_result = self.text_recognizer.recognize(frame_path)
            analysis['text_per_frame'].append(text_result)
            analysis['all_measurements'].extend(text_result.get('measurements', []))
            if text_result.get('all_text'):
                analysis['all_text'].append(text_result['all_text'])
            
            # Action recognition
            action_result = self.action_recognizer.recognize(
                obj_result.get('cooking_related', []),
                text_result.get('all_text', '')
            )
            analysis['actions_per_frame'].append(action_result)
        
        return analysis
    
    def _generate_recipe_from_detections(self, analysis: Dict, video_path: str) -> Dict:
        """Generate recipe from CV detections."""
        recipe = {
            'title': Path(video_path).stem.replace('_', ' ').title(),
            'ingredients': [],
            'steps': [],
            'tools': [],
            'cuisine': None,
            'estimated_time': None
        }
        
        # Extract ingredients from detected objects and text
        object_counts = Counter(analysis['all_objects'])
        
        # Map detected objects to likely ingredients
        ingredient_mapping = {
            'bowl': 'various ingredients',
            'banana': 'banana',
            'apple': 'apple',
            'orange': 'orange',
            'broccoli': 'broccoli',
            'carrot': 'carrot',
            'bottle': 'liquid ingredient (oil/sauce)',
        }
        
        ingredients_set = {}
        for obj, count in object_counts.most_common(10):
            if obj in ingredient_mapping:
                ing_name = ingredient_mapping[obj]
                ingredients_set[ing_name] = {
                    'name': ing_name.title(),
                    'quantity': str(count) if count > 1 else '',
                    'unit': '',
                    'state': '',
                    'source': 'yolo_detection'
                }
        
        # Add ingredients from measurements in text
        for measurement in analysis['all_measurements']:
            ingredients_set[f'measured_{len(ingredients_set)}'] = {
                'name': measurement,
                'quantity': '',
                'unit': '',
                'state': '',
                'source': 'ocr_measurement'
            }
        
        recipe['ingredients'] = list(ingredients_set.values())
        
        # Extract cooking steps from detected actions
        all_actions = []
        for frame_actions in analysis['actions_per_frame']:
            for action in frame_actions.get('actions', []):
                all_actions.append(action['action'])
        
        # Deduplicate and order actions
        unique_actions = []
        seen = set()
        for action in all_actions:
            if action not in seen:
                unique_actions.append(action)
                seen.add(action)
        
        # Generate steps from actions
        for i, action in enumerate(unique_actions, 1):
            recipe['steps'].append({
                'step': i,
                'description': f'{action.title()} the ingredients',
                'timestamp': f'~{i * 10}s',
                'source': 'action_recognition'
            })
        
        # Extract tools from detected objects
        tool_objects = {'knife', 'fork', 'spoon', 'bowl', 'cup'}
        detected_tools = [obj for obj in set(analysis['all_objects']) if obj in tool_objects]
        recipe['tools'] = detected_tools
        
        # Estimate cuisine from detected ingredients (simple heuristic)
        if 'broccoli' in analysis['all_objects'] or 'carrot' in analysis['all_objects']:
            recipe['cuisine'] = 'Asian/Vegetarian'
        elif 'pizza' in analysis['all_objects'] or 'sandwich' in analysis['all_objects']:
            recipe['cuisine'] = 'Western'
        
        return recipe
