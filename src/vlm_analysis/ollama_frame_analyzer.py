"""
Ollama VLM Frame Analyzer - Analyzes video frames using Ollama VLM

This module provides frame-level analysis using Vision-Language Models
for cooking video understanding.
"""
import logging
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from .ollama_engine import OllamaVLMEngine
from .vlm_prompts import CookingPrompts

logger = logging.getLogger(__name__)


class OllamaFrameAnalyzer:
    """
    Analyzes video frames using Ollama Vision-Language Model for cooking understanding.
    """
    
    def __init__(
        self,
        ollama_engine: Optional[OllamaVLMEngine] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize Ollama Frame Analyzer.
        
        Args:
            ollama_engine: OllamaVLMEngine instance (creates new if None)
            config: Configuration dict
        """
        self.config = config or {}
        
        if ollama_engine is None:
            logger.info("Creating new Ollama VLM Engine...")
            vlm_config = self.config.get('vlm', {})
            self.ollama_engine = OllamaVLMEngine(
                model=vlm_config.get('model', 'qwen2-vl:7b'),
                host=vlm_config.get('host', 'http://localhost:11434'),
                use_cache=vlm_config.get('use_cache', True),
                cache_path=vlm_config.get('cache_path', 'data/temp/ollama_vlm_cache.json'),
                timeout=vlm_config.get('timeout', 120)
            )
        else:
            self.ollama_engine = ollama_engine
        
        self.prompts = CookingPrompts()
        logger.info("Ollama Frame Analyzer initialized")
    
    def analyze_frame(
        self,
        frame_path: str,
        analysis_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a single frame.
        
        Args:
            frame_path: Path to frame image
            analysis_types: List of analysis types to perform
                          ['ingredients', 'actions', 'tools', 'measurements']
                          If None, performs all analyses
                          
        Returns:
            Dict containing all analysis results
        """
        if analysis_types is None:
            analysis_types = ['ingredients', 'actions', 'tools', 'measurements']
        
        logger.info(f"Analyzing frame: {frame_path}")
        logger.debug(f"Analysis types: {analysis_types}")
        
        results = {
            'frame_path': frame_path,
            'analyses': {}
        }
        
        try:
            # Perform each requested analysis
            for analysis_type in analysis_types:
                logger.debug(f"Running {analysis_type} analysis...")
                
                if analysis_type == 'ingredients':
                    results['analyses']['ingredients'] = self._analyze_ingredients(frame_path)
                elif analysis_type == 'actions':
                    results['analyses']['actions'] = self._analyze_actions(frame_path)
                elif analysis_type == 'tools':
                    results['analyses']['tools'] = self._analyze_tools(frame_path)
                elif analysis_type == 'measurements':
                    results['analyses']['measurements'] = self._analyze_measurements(frame_path)
                elif analysis_type == 'process':
                    results['analyses']['process'] = self._analyze_process(frame_path)
                elif analysis_type == 'cuisine':
                    results['analyses']['cuisine'] = self._analyze_cuisine(frame_path)
                elif analysis_type == 'comprehensive':
                    results['analyses']['comprehensive'] = self._analyze_comprehensive(frame_path)
                else:
                    logger.warning(f"Unknown analysis type: {analysis_type}")
            
            logger.info(f"✓ Frame analysis complete: {len(results['analyses'])} analyses")
            return results
            
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            results['error'] = str(e)
            return results
    
    def _analyze_ingredients(self, frame_path: str) -> Dict:
        """Analyze ingredients in frame."""
        prompt = self.prompts.identify_ingredients()
        response = self.ollama_engine.query(frame_path, prompt, temperature=0.1)
        
        # Parse JSON response
        try:
            ingredients_data = self._parse_json_response(response['response'])
            return {
                'ingredients': ingredients_data,
                'raw_response': response['response'],
                'confidence': 'high' if ingredients_data else 'low',
                'inference_time': response.get('inference_time', 0)
            }
        except Exception as e:
            logger.warning(f"Failed to parse ingredients response: {e}")
            return {
                'ingredients': [],
                'raw_response': response['response'],
                'error': str(e)
            }
    
    def _analyze_actions(self, frame_path: str) -> Dict:
        """Analyze cooking actions in frame."""
        prompt = self.prompts.identify_actions()
        response = self.ollama_engine.query(frame_path, prompt, temperature=0.1)
        
        try:
            action_data = self._parse_json_response(response['response'])
            return {
                'action': action_data.get('action', 'unknown'),
                'confidence': action_data.get('confidence', 'low'),
                'description': action_data.get('description', ''),
                'technique': action_data.get('technique', ''),
                'raw_response': response['response'],
                'inference_time': response.get('inference_time', 0)
            }
        except Exception as e:
            logger.warning(f"Failed to parse action response: {e}")
            return {
                'action': 'unknown',
                'confidence': 'low',
                'raw_response': response['response'],
                'error': str(e)
            }
    
    def _analyze_tools(self, frame_path: str) -> Dict:
        """Analyze cooking tools in frame."""
        prompt = self.prompts.identify_tools()
        response = self.ollama_engine.query(frame_path, prompt, temperature=0.1)
        
        try:
            tools_data = self._parse_json_response(response['response'])
            if isinstance(tools_data, list):
                tools_list = tools_data
            elif isinstance(tools_data, dict) and 'tools' in tools_data:
                tools_list = tools_data['tools']
            else:
                tools_list = []
            
            return {
                'tools': tools_list,
                'count': len(tools_list),
                'raw_response': response['response'],
                'inference_time': response.get('inference_time', 0)
            }
        except Exception as e:
            logger.warning(f"Failed to parse tools response: {e}")
            return {
                'tools': [],
                'count': 0,
                'raw_response': response['response'],
                'error': str(e)
            }
    
    def _analyze_measurements(self, frame_path: str) -> Dict:
        """Analyze measurements and text in frame."""
        prompt = self.prompts.read_measurements()
        response = self.ollama_engine.query(frame_path, prompt, temperature=0.0)
        
        try:
            measurements_data = self._parse_json_response(response['response'])
            return {
                'measurements': measurements_data.get('measurements', []),
                'temperatures': measurements_data.get('temperatures', []),
                'durations': measurements_data.get('durations', []),
                'instructions': measurements_data.get('instructions', []),
                'other_text': measurements_data.get('other_text', []),
                'raw_response': response['response'],
                'inference_time': response.get('inference_time', 0)
            }
        except Exception as e:
            logger.warning(f"Failed to parse measurements response: {e}")
            return {
                'measurements': [],
                'temperatures': [],
                'durations': [],
                'instructions': [],
                'other_text': [],
                'raw_response': response['response'],
                'error': str(e)
            }
    
    def _analyze_process(self, frame_path: str) -> Dict:
        """Analyze cooking process in frame."""
        prompt = self.prompts.analyze_cooking_process()
        response = self.ollama_engine.query(frame_path, prompt, temperature=0.2)
        
        return {
            'process_description': response['response'],
            'raw_response': response['response'],
            'inference_time': response.get('inference_time', 0)
        }
    
    def _analyze_cuisine(self, frame_path: str) -> Dict:
        """Analyze cuisine type and style."""
        prompt = self.prompts.identify_cuisine_style()
        response = self.ollama_engine.query(frame_path, prompt, temperature=0.1)
        
        try:
            cuisine_data = self._parse_json_response(response['response'])
            cuisine_data['inference_time'] = response.get('inference_time', 0)
            return cuisine_data
        except Exception as e:
            logger.warning(f"Failed to parse cuisine response: {e}")
            return {
                'cuisine_type': 'unknown',
                'raw_response': response['response'],
                'error': str(e)
            }
    
    def _analyze_comprehensive(self, frame_path: str) -> Dict:
        """Perform comprehensive analysis in one query."""
        prompt = self.prompts.comprehensive_frame_analysis()
        response = self.ollama_engine.query(frame_path, prompt, temperature=0.1)
        
        try:
            comprehensive_data = self._parse_json_response(response['response'])
            comprehensive_data['inference_time'] = response.get('inference_time', 0)
            return comprehensive_data
        except Exception as e:
            logger.warning(f"Failed to parse comprehensive response: {e}")
            return {
                'raw_response': response['response'],
                'error': str(e)
            }
    
    def analyze_frames_batch(
        self,
        frame_paths: List[str],
        analysis_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple frames in batch.
        
        Args:
            frame_paths: List of frame paths
            analysis_types: Types of analysis to perform
            
        Returns:
            List of analysis results
        """
        results = []
        total = len(frame_paths)
        
        for idx, frame_path in enumerate(frame_paths, 1):
            logger.info(f"Analyzing frame {idx}/{total}")
            result = self.analyze_frame(frame_path, analysis_types)
            results.append(result)
        
        return results
    
    def analyze_key_frames(
        self,
        frames: List[str],
        max_frames: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Analyze key frames (evenly sampled if too many).
        
        Args:
            frames: List of all frame paths
            max_frames: Maximum number of frames to analyze
            
        Returns:
            List of analysis results for key frames
        """
        if len(frames) <= max_frames:
            key_frames = frames
        else:
            # Sample evenly
            step = len(frames) / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            key_frames = [frames[i] for i in indices]
        
        logger.info(f"Analyzing {len(key_frames)} key frames out of {len(frames)} total")
        return self.analyze_frames_batch(key_frames)
    
    def quick_ingredient_scan(self, frame_path: str) -> List[str]:
        """
        Quick ingredient identification without detailed analysis.
        
        Args:
            frame_path: Path to frame
            
        Returns:
            List of ingredient names
        """
        prompt = self.prompts.quick_ingredient_scan()
        response = self.ollama_engine.query(frame_path, prompt, temperature=0.0)
        
        # Parse response (expecting simple list)
        text = response['response'].strip()
        # Extract ingredients from text
        ingredients = [line.strip('- ').strip() for line in text.split('\n') if line.strip()]
        return ingredients
    
    def quick_action_scan(self, frame_path: str) -> str:
        """
        Quick action identification without detailed analysis.
        
        Args:
            frame_path: Path to frame
            
        Returns:
            Action description string
        """
        prompt = self.prompts.quick_action_scan()
        response = self.ollama_engine.query(frame_path, prompt, temperature=0.0)
        return response['response'].strip()
    
    def _parse_json_response(self, response: str) -> Any:
        """
        Parse JSON from VLM response.
        
        Args:
            response: VLM response text
            
        Returns:
            Parsed JSON data
        """
        response = response.strip()
        
        # Find JSON in response
        json_start = response.find('{')
        json_start_list = response.find('[')
        
        if json_start == -1 and json_start_list == -1:
            raise ValueError("No JSON found in response")
        
        if json_start_list != -1 and (json_start == -1 or json_start_list < json_start):
            json_start = json_start_list
            json_end = response.rfind(']') + 1
        else:
            json_end = response.rfind('}') + 1
        
        json_str = response[json_start:json_end]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            logger.debug(f"Attempted to parse: {json_str[:200]}")
            raise
    
    def get_comprehensive_frame_understanding(
        self,
        frame_path: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive understanding of a frame (all analyses).
        
        Args:
            frame_path: Path to frame
            
        Returns:
            Complete analysis dict
        """
        return self.analyze_frame(
            frame_path,
            analysis_types=['ingredients', 'actions', 'tools', 'measurements', 'process']
        )

