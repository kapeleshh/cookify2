"""
Simplified Video Pipeline for Cookify VLM Demo
Processes cooking videos and extracts recipes using VLM
"""

import os
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import cv2
from tqdm import tqdm

from src.preprocessing.video_processor import VideoProcessor
from src.vlm_analysis.ollama_engine import OllamaVLMEngine
from src.vlm_analysis.ollama_frame_analyzer import OllamaFrameAnalyzer
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

class SimpleVideoPipeline:
    """
    Simplified pipeline for video recipe extraction using VLM.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the simplified pipeline."""
        self.config = load_config(config_path)
        self.video_processor = VideoProcessor()
        
        # Initialize VLM components
        vlm_config = self.config.get('vlm', {})
        if vlm_config.get('enabled', False):
            try:
                self.vlm_engine = OllamaVLMEngine(
                    model=vlm_config.get('model', 'llava:7b'),
                    host=vlm_config.get('host', 'http://localhost:11434'),
                    use_cache=vlm_config.get('use_cache', True),
                    cache_path=vlm_config.get('cache_path', 'data/temp/ollama_vlm_cache.json'),
                    timeout=vlm_config.get('timeout', 120)
                )
                
                if not self.vlm_engine.test_connection():
                    raise ConnectionError("Cannot connect to Ollama VLM")
                
                self.vlm_analyzer = OllamaFrameAnalyzer(self.vlm_engine, self.config)
                logger.info("✓ VLM pipeline initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize VLM: {e}")
                self.vlm_engine = None
                self.vlm_analyzer = None
        else:
            self.vlm_engine = None
            self.vlm_analyzer = None
            logger.warning("VLM is disabled in config")
    
    def process_video(self, video_path: str, progress_callback=None) -> Dict[str, Any]:
        """
        Process a cooking video and extract recipe information.
        
        Args:
            video_path: Path to the video file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing extracted recipe information
        """
        logger.info(f"Processing video: {video_path}")
        start_time = time.time()
        
        result = {
            'video_path': video_path,
            'processing_time': 0,
            'frames_analyzed': 0,
            'recipe': {
                'title': 'Extracted Recipe',
                'ingredients': [],
                'steps': [],
                'tools': [],
                'cuisine': None,
                'estimated_time': None
            },
            'metadata': {
                'vlm_enabled': self.vlm_analyzer is not None,
                'model': self.config.get('vlm', {}).get('model', 'none')
            }
        }
        
        try:
            # Step 1: Extract frames from video
            if progress_callback:
                progress_callback('Extracting frames from video...', 10)
            
            frames = self._extract_video_frames(video_path)
            logger.info(f"Extracted {len(frames)} frames")
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # Step 2: Select key frames for VLM analysis
            if progress_callback:
                progress_callback('Selecting key frames for analysis...', 20)
            
            max_frames = self.config.get('vlm', {}).get('max_frames_per_video', 20)
            key_frames = self._select_key_frames(frames, max_frames)
            logger.info(f"Selected {len(key_frames)} key frames")
            
            # Step 3: Analyze frames with VLM
            if progress_callback:
                progress_callback('Analyzing frames with VLM...', 30)
            
            vlm_results = []
            if self.vlm_analyzer:
                for i, frame_path in enumerate(key_frames):
                    if progress_callback:
                        progress = 30 + int((i / len(key_frames)) * 50)
                        progress_callback(f'Analyzing frame {i+1}/{len(key_frames)}...', progress)
                    
                    frame_analysis = self.vlm_analyzer.analyze_frame(
                        frame_path,
                        analysis_types=['ingredients', 'actions', 'tools', 'process']
                    )
                    vlm_results.append(frame_analysis)
                
                result['frames_analyzed'] = len(key_frames)
            
            # Step 4: Aggregate results into recipe
            if progress_callback:
                progress_callback('Extracting recipe from analysis...', 85)
            
            recipe = self._aggregate_to_recipe(vlm_results, video_path)
            result['recipe'] = recipe
            
            # Step 5: Format final output
            if progress_callback:
                progress_callback('Finalizing recipe...', 95)
            
            result['processing_time'] = time.time() - start_time
            
            if progress_callback:
                progress_callback('Complete!', 100)
            
            logger.info(f"Video processed successfully in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            result['error'] = str(e)
            return result
    
    def _extract_video_frames(self, video_path: str) -> List[str]:
        """Extract frames from video at specified intervals."""
        frames_dir = Path('data/temp/frames')
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean previous frames
        for f in frames_dir.glob('frame_*.jpg'):
            f.unlink()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract 1 frame per second
        frame_interval = max(1, int(fps))
        
        frames = []
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = frames_dir / f'frame_{saved_count:04d}.jpg'
                cv2.imwrite(str(frame_path), frame)
                frames.append(str(frame_path))
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def _select_key_frames(self, frames: List[str], max_frames: int) -> List[str]:
        """Select key frames for VLM analysis."""
        if len(frames) <= max_frames:
            return frames
        
        # Uniform sampling
        step = len(frames) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        return [frames[i] for i in indices]
    
    def _aggregate_to_recipe(self, vlm_results: List[Dict], video_path: str) -> Dict:
        """Aggregate VLM frame analyses into a coherent recipe."""
        recipe = {
            'title': Path(video_path).stem.replace('_', ' ').title(),
            'ingredients': [],
            'steps': [],
            'tools': [],
            'cuisine': None,
            'estimated_time': None
        }
        
        # Aggregate ingredients
        ingredients_set = {}
        for result in vlm_results:
            if 'ingredients' in result and 'items' in result['ingredients']:
                for item in result['ingredients']['items']:
                    name = item.get('name', '').lower()
                    if name and name not in ingredients_set:
                        ingredients_set[name] = {
                            'name': item.get('name', ''),
                            'quantity': item.get('quantity', ''),
                            'unit': item.get('unit', ''),
                            'state': item.get('state', '')
                        }
        
        recipe['ingredients'] = list(ingredients_set.values())
        
        # Aggregate tools
        tools_set = set()
        for result in vlm_results:
            if 'tools' in result and 'items' in result['tools']:
                for tool in result['tools']['items']:
                    tools_set.add(tool)
        
        recipe['tools'] = sorted(list(tools_set))
        
        # Extract steps from actions and process descriptions
        steps = []
        for i, result in enumerate(vlm_results):
            step_desc = None
            
            # Try to get from process description
            if 'process' in result and 'description' in result['process']:
                step_desc = result['process']['description']
            
            # Or from actions
            elif 'actions' in result and 'items' in result['actions']:
                actions = result['actions']['items']
                if actions:
                    step_desc = ', '.join(actions)
            
            if step_desc and step_desc not in steps:
                steps.append({
                    'step': len(steps) + 1,
                    'description': step_desc,
                    'timestamp': f'~{i}s'
                })
        
        recipe['steps'] = steps
        
        # Try to detect cuisine from first frame that has it
        for result in vlm_results:
            if 'cuisine' in result and result['cuisine'].get('type'):
                recipe['cuisine'] = result['cuisine']['type']
                break
        
        return recipe

