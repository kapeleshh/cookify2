"""
Cookify Pipeline - Main pipeline for recipe extraction from cooking videos
"""

import os
import logging
import json
import yaml
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Any

from .preprocessing.video_processor import VideoProcessor
from .frame_analysis.object_detector import ObjectDetector
from .frame_analysis.scene_detector import SceneDetector
from .frame_analysis.text_recognizer import TextRecognizer
from .frame_analysis.action_recognizer import ActionRecognizer
from .audio_analysis.transcriber import AudioTranscriber
from .audio_analysis.nlp_processor import NLPProcessor
from .integration.multimodal_integrator import MultimodalIntegrator
from .recipe_extraction.recipe_extractor import RecipeExtractor
from .output_formatting.formatter import OutputFormatter
from .utils.config_loader import load_config
from .utils.logger import setup_pipeline_logging, CookifyLogger
import psutil
import gc

logger = logging.getLogger(__name__)

class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass

class VideoProcessingError(PipelineError):
    """Exception raised when video processing fails."""
    pass

class AudioProcessingError(PipelineError):
    """Exception raised when audio processing fails."""
    pass

class RecipeExtractionError(PipelineError):
    """Exception raised when recipe extraction fails."""
    pass

class Pipeline:
    """
    Main pipeline for recipe extraction from cooking videos.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the pipeline.
        
        Args:
            config_path (str, optional): Path to the configuration file. Defaults to None.
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup enhanced logging
        self.cookify_logger = setup_pipeline_logging(self.config)
        self.logger = self.cookify_logger.get_logger()
        
        # Initialize components
        self._init_components()
        
        # Log initialization completion
        self.cookify_logger.log_processing_step("Pipeline initialization", {
            "config_loaded": True,
            "components_initialized": True
        })
        
        # Initialize performance monitoring
        self.performance_data = {}
        self.start_times = {}
    
    def _init_components(self):
        """
        Initialize pipeline components.
        """
        # Video preprocessing
        self.video_processor = VideoProcessor(
            output_dir=self.config["general"]["output_dir"],
            frame_rate=self.config["preprocessing"]["frame_rate"],
            temp_dir=self.config["general"]["temp_dir"]
        )
        
        # Frame analysis
        self.object_detector = ObjectDetector(
            confidence_threshold=self.config["object_detection"]["confidence_threshold"]
        )
        
        self.scene_detector = SceneDetector(
            threshold=self.config["scene_detection"]["threshold"],
            min_scene_len=self.config["scene_detection"]["min_scene_len"]
        )
        
        # Initialize other components as needed
        # These will be implemented in their respective modules
        self.text_recognizer = None  # TextRecognizer()
        self.action_recognizer = None  # ActionRecognizer()
        
        # Initialize audio transcriber if available
        try:
            from .audio_analysis.transcriber import AudioTranscriber
            self.audio_transcriber = AudioTranscriber(
                model_name=self.config["transcription"]["model"],
                language=self.config["transcription"]["language"],
                timestamps=self.config["transcription"]["timestamps"]
            )
            logger.info("Audio transcriber initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize audio transcriber: {e}")
            self.audio_transcriber = None
        
        self.nlp_processor = None  # NLPProcessor()
        self.multimodal_integrator = None  # MultimodalIntegrator()
        self.recipe_extractor = None  # RecipeExtractor()
        self.output_formatter = None  # OutputFormatter()
    
    def _validate_input(self, video_path: str) -> None:
        """
        Validate input video file.
        
        Args:
            video_path (str): Path to the video file.
            
        Raises:
            VideoProcessingError: If validation fails.
        """
        if not video_path:
            raise VideoProcessingError("Video path cannot be empty")
        
        if not os.path.exists(video_path):
            raise VideoProcessingError(f"Video file not found: {video_path}")
        
        if not os.path.isfile(video_path):
            raise VideoProcessingError(f"Path is not a file: {video_path}")
        
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            raise VideoProcessingError(f"Video file is empty: {video_path}")
        
        # Check file extension
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        file_ext = Path(video_path).suffix.lower()
        if file_ext not in valid_extensions:
            logger.warning(f"Unusual video file extension: {file_ext}")
        
        logger.info(f"Input validation passed: {video_path} ({file_size} bytes)")
    
    def _start_performance_monitoring(self, operation: str) -> None:
        """Start monitoring performance for an operation."""
        self.start_times[operation] = time.time()
        self.logger.debug(f"Started performance monitoring: {operation}")
    
    def _end_performance_monitoring(self, operation: str) -> float:
        """End monitoring performance for an operation and return duration."""
        if operation not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        del self.start_times[operation]
        
        # Store performance data
        if operation not in self.performance_data:
            self.performance_data[operation] = []
        self.performance_data[operation].append(duration)
        
        self.logger.info(f"Operation '{operation}' completed in {duration:.2f}s")
        return duration
    
    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage by cleaning up and garbage collecting."""
        gc.collect()
        memory_percent = psutil.virtual_memory().percent
        self.logger.debug(f"Memory usage: {memory_percent:.1f}%")
        
        if memory_percent > 90:
            self.logger.warning("High memory usage detected, forcing garbage collection")
            gc.collect()
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        memory = psutil.virtual_memory()
        return {
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent_used": memory.percent
            },
            "cpu_percent": psutil.cpu_percent(interval=1)
        }
    
    def _log_performance_summary(self) -> None:
        """Log performance summary for all operations."""
        if not self.performance_data:
            return
        
        self.logger.info("Performance Summary:")
        for operation, times in self.performance_data.items():
            if times:
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                self.logger.info(f"  {operation}: {len(times)} calls, avg: {avg_time:.2f}s, total: {total_time:.2f}s")
        
        # Log system status
        system_status = self._get_system_status()
        self.logger.info(f"System Status - Memory: {system_status['memory']['percent_used']:.1f}%, CPU: {system_status['cpu_percent']:.1f}%")

    def process(self, video_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a video and extract the recipe with comprehensive error handling.
        
        Args:
            video_path (str): Path to the video file.
            output_path (str, optional): Path to save the output. Defaults to None.
            
        Returns:
            dict: Extracted recipe.
            
        Raises:
            VideoProcessingError: If video processing fails.
            AudioProcessingError: If audio processing fails.
            RecipeExtractionError: If recipe extraction fails.
        """
        self.logger.info(f"Processing video: {video_path}")
        self.cookify_logger.start_timer("video_processing")
        
        # Validate input
        try:
            self._validate_input(video_path)
            self.cookify_logger.log_processing_step("Input validation", {"status": "passed"})
        except VideoProcessingError as e:
            self.cookify_logger.log_error_with_context(e, {"step": "input_validation", "video_path": video_path})
            raise
        
        # Determine output path
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = os.path.join(self.config["general"]["output_dir"], f"{video_name}_recipe.json")
        
        # Create output directory if it doesn't exist
        try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        except OSError as e:
            raise VideoProcessingError(f"Failed to create output directory: {e}")
        
        # Initialize result tracking
        processing_results = {
            "video_path": video_path,
            "metadata": {},
            "scenes": [],
            "object_detections": [],
            "text_detections": [],
            "action_detections": [],
            "transcription": {},
            "nlp_results": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Step 1: Preprocess video
            self._start_performance_monitoring("video_preprocessing")
            self.cookify_logger.start_timer("video_preprocessing")
            self.logger.info("Preprocessing video...")
            try:
            frames, audio_path, metadata = self.video_processor.process(video_path)
                processing_results["metadata"] = metadata
                duration = self.cookify_logger.end_timer("video_preprocessing")
                self._end_performance_monitoring("video_preprocessing")
                self.cookify_logger.log_processing_step("Video preprocessing", {
                    "frames_extracted": len(frames),
                    "duration": duration,
                    "metadata": metadata
                })
                # Optimize memory after video preprocessing
                self._optimize_memory_usage()
            except Exception as e:
                self.cookify_logger.end_timer("video_preprocessing")
                self._end_performance_monitoring("video_preprocessing")
                error_msg = f"Video preprocessing failed: {e}"
                self.cookify_logger.log_error_with_context(e, {"step": "video_preprocessing", "video_path": video_path})
                processing_results["errors"].append(error_msg)
                raise VideoProcessingError(error_msg) from e
            
            # Step 2: Detect scenes
            logger.info("Detecting scenes...")
            try:
            scenes = self.scene_detector.detect(video_path)
                processing_results["scenes"] = scenes
                logger.info(f"Scene detection completed: {len(scenes)} scenes detected")
            except Exception as e:
                error_msg = f"Scene detection failed: {e}"
                logger.warning(error_msg)
                processing_results["warnings"].append(error_msg)
                scenes = []
            
            # Step 3: Detect objects in frames
            self._start_performance_monitoring("object_detection")
            logger.info("Detecting objects in frames...")
            try:
            object_detections = self.object_detector.detect(frames)
            
            # Filter for cooking-related objects if configured
            if self.config["object_detection"]["filter_cooking_objects"]:
                object_detections = self.object_detector.filter_cooking_related(object_detections)
                
                processing_results["object_detections"] = object_detections
                duration = self._end_performance_monitoring("object_detection")
                logger.info(f"Object detection completed: {len(object_detections)} detections in {duration:.2f}s")
                # Optimize memory after object detection
                self._optimize_memory_usage()
            except Exception as e:
                self._end_performance_monitoring("object_detection")
                error_msg = f"Object detection failed: {e}"
                logger.warning(error_msg)
                processing_results["warnings"].append(error_msg)
                object_detections = []
            
            # Step 4: Recognize text in frames (placeholder)
            logger.info("Recognizing text in frames...")
            text_detections = []  # self.text_recognizer.recognize(frames)
            processing_results["text_detections"] = text_detections
            
            # Step 5: Recognize actions (placeholder)
            logger.info("Recognizing actions...")
            action_detections = []  # self.action_recognizer.recognize(frames, scenes)
            processing_results["action_detections"] = action_detections
            
            # Step 6: Transcribe audio
            logger.info("Transcribing audio...")
            transcription = {}
            if self.audio_transcriber and audio_path and os.path.exists(audio_path):
                try:
                    transcription = self.audio_transcriber.transcribe(audio_path)
                    processing_results["transcription"] = transcription
                    logger.info(f"Audio transcribed: {len(transcription.get('text', ''))} characters")
                except Exception as e:
                    error_msg = f"Audio transcription failed: {e}"
                    logger.warning(error_msg)
                    processing_results["warnings"].append(error_msg)
                    transcription = {}
            else:
                warning_msg = "Audio transcriber not available or audio file not found"
                logger.warning(warning_msg)
                processing_results["warnings"].append(warning_msg)
            
            # Step 7: Process natural language (placeholder)
            logger.info("Processing natural language...")
            nlp_results = {}  # self.nlp_processor.process(transcription)
            processing_results["nlp_results"] = nlp_results
            
            # Step 8: Integrate multimodal data
            logger.info("Integrating multimodal data...")
            integrated_data = {
                "video_path": video_path,
                "metadata": processing_results["metadata"],
                "scenes": processing_results["scenes"],
                "object_detections": processing_results["object_detections"],
                "text_detections": processing_results["text_detections"],
                "action_detections": processing_results["action_detections"],
                "transcription": processing_results["transcription"],
                "nlp_results": processing_results["nlp_results"]
            }
            
            # Step 9: Extract recipe structure from actual data
            logger.info("Extracting recipe structure...")
            try:
                recipe = self._extract_recipe_from_data(integrated_data, processing_results["metadata"])
                logger.info(f"Recipe extraction completed: {len(recipe.get('ingredients', []))} ingredients, {len(recipe.get('steps', []))} steps")
            except Exception as e:
                error_msg = f"Recipe extraction failed: {e}"
                logger.error(error_msg)
                processing_results["errors"].append(error_msg)
                raise RecipeExtractionError(error_msg) from e
            
            # Step 10: Format output
            logger.info("Formatting output...")
            formatted_output = recipe
            
            # Add processing metadata
            formatted_output["_processing_metadata"] = {
                "processing_errors": processing_results["errors"],
                "processing_warnings": processing_results["warnings"],
                "components_used": {
                    "video_processor": True,
                    "scene_detector": True,
                    "object_detector": True,
                    "text_recognizer": False,
                    "action_recognizer": False,
                    "audio_transcriber": self.audio_transcriber is not None,
                    "nlp_processor": False
                }
            }
            
            # Save output
            try:
            with open(output_path, 'w') as f:
                json.dump(formatted_output, f, indent=2)
                self.logger.info(f"Recipe extracted and saved to {output_path}")
                
                # Log recipe extraction summary
                self.cookify_logger.log_recipe_extraction(formatted_output)
                
            except Exception as e:
                error_msg = f"Failed to save output: {e}"
                self.cookify_logger.log_error_with_context(e, {"step": "save_output", "output_path": output_path})
                raise VideoProcessingError(error_msg) from e
            
            # Log final performance summary
            total_duration = self.cookify_logger.end_timer("video_processing")
            self.cookify_logger.log_processing_step("Video processing completed", {
                "total_duration": total_duration,
                "output_path": output_path
            })
            
            # Log performance summary
            self._log_performance_summary()
            
            return formatted_output
            
        except (VideoProcessingError, AudioProcessingError, RecipeExtractionError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            error_msg = f"Unexpected error processing video: {e}"
            logger.error(error_msg, exc_info=True)
            raise PipelineError(error_msg) from e
    
    def _extract_recipe_from_data(self, integrated_data, metadata):
        """
        Extract recipe from actual processed data with comprehensive error handling.
        
        Args:
            integrated_data (dict): Integrated multimodal data.
            metadata (dict): Video metadata.
            
        Returns:
            dict: Extracted recipe.
            
        Raises:
            RecipeExtractionError: If recipe extraction fails.
        """
        try:
            # Extract ingredients from object detections and audio
            try:
                ingredients = self._extract_ingredients_from_objects(integrated_data.get("object_detections", []))
                logger.info(f"Ingredients extracted: {len(ingredients)} items")
            except Exception as e:
                logger.warning(f"Ingredient extraction failed: {e}")
                ingredients = []
            
            # Enhance ingredients with audio transcription if available
            transcription = integrated_data.get("transcription", {})
            if transcription.get("text"):
                try:
                    ingredients = self._enhance_ingredients_with_audio(ingredients, transcription)
                    logger.info(f"Ingredients enhanced with audio: {len(ingredients)} items")
                except Exception as e:
                    logger.warning(f"Ingredient enhancement failed: {e}")
            
            # Extract tools from object detections
            try:
                tools = self._extract_tools_from_objects(integrated_data.get("object_detections", []))
                logger.info(f"Tools extracted: {len(tools)} items")
            except Exception as e:
                logger.warning(f"Tool extraction failed: {e}")
                tools = []
            
            # Extract steps from scenes, objects, and audio
            try:
                steps = self._extract_steps_from_scenes(integrated_data.get("scenes", []), 
                                                       integrated_data.get("object_detections", []),
                                                       metadata)
                logger.info(f"Steps extracted: {len(steps)} steps")
            except Exception as e:
                logger.warning(f"Step extraction failed: {e}")
                steps = []
            
            # Enhance steps with audio transcription if available
            if transcription.get("text"):
                try:
                    steps = self._enhance_steps_with_audio(steps, transcription)
                    logger.info(f"Steps enhanced with audio: {len(steps)} steps")
                except Exception as e:
                    logger.warning(f"Step enhancement failed: {e}")
            
            # Generate title based on detected ingredients and audio
            try:
                title = self._generate_recipe_title(ingredients, transcription)
            except Exception as e:
                logger.warning(f"Title generation failed: {e}")
                title = "Cooking Recipe"
            
            # Estimate servings based on ingredient quantities and audio
            try:
                servings = self._estimate_servings(ingredients, transcription)
            except Exception as e:
                logger.warning(f"Serving estimation failed: {e}")
                servings = 4
            
            # Validate minimum recipe requirements
            if not ingredients and not steps:
                raise RecipeExtractionError("No ingredients or steps could be extracted from the video")
            
            recipe = {
                "title": title,
                "servings": servings,
                "ingredients": ingredients,
                "tools": tools,
                "steps": steps
            }
            
            logger.info(f"Recipe extraction completed: {len(ingredients)} ingredients, {len(tools)} tools, {len(steps)} steps")
            return recipe
            
        except RecipeExtractionError:
            # Re-raise recipe extraction errors
            raise
        except Exception as e:
            error_msg = f"Unexpected error during recipe extraction: {e}"
            logger.error(error_msg)
            raise RecipeExtractionError(error_msg) from e
    
    def _extract_ingredients_from_objects(self, object_detections):
        """Extract ingredients from object detection results."""
        ingredients = []
        ingredient_counts = {}
        
        # Cooking-related food items
        food_items = {
            'apple', 'orange', 'banana', 'broccoli', 'carrot', 'hot dog', 'pizza', 
            'donut', 'cake', 'sandwich', 'tomato', 'potato', 'onion', 'garlic',
            'lettuce', 'cucumber', 'pepper', 'cheese', 'egg', 'meat', 'chicken',
            'beef', 'pork', 'fish', 'shrimp', 'rice', 'pasta', 'bread', 'butter',
            'oil', 'salt', 'sugar', 'flour', 'herb', 'spice'
        }
        
        for frame_result in object_detections:
            for detection in frame_result.get("detections", []):
                class_name = detection.get("class", "").lower()
                confidence = detection.get("confidence", 0.0)
                
                # Only include food items with reasonable confidence
                if class_name in food_items and confidence > 0.3:
                    if class_name not in ingredient_counts:
                        ingredient_counts[class_name] = 0
                    ingredient_counts[class_name] += 1
        
        # Convert to ingredient list
        for ingredient, count in ingredient_counts.items():
            # Estimate quantity based on frequency
            if count >= 3:
                qty = "2-3"
                unit = "pieces" if ingredient in ['apple', 'orange', 'banana', 'potato', 'onion'] else "cups"
            elif count >= 2:
                qty = "1-2"
                unit = "pieces" if ingredient in ['apple', 'orange', 'banana', 'potato', 'onion'] else "cups"
            else:
                qty = "1"
                unit = "piece" if ingredient in ['apple', 'orange', 'banana', 'potato', 'onion'] else "cup"
            
            ingredients.append({
                "name": ingredient.title(),
                "qty": qty,
                "unit": unit
            })
        
        return ingredients
    
    def _extract_tools_from_objects(self, object_detections):
        """Extract cooking tools from object detection results."""
        tools = set()
        
        # Kitchen tools and containers
        tool_items = {
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender',
            'mixer', 'cutting board', 'pan', 'pot', 'plate', 'whisk', 'grater',
            'colander', 'strainer', 'measuring cup', 'measuring spoon', 'spatula',
            'tongs', 'ladle', 'rolling pin', 'peeler', 'can opener', 'scale',
            'timer', 'thermometer'
        }
        
        for frame_result in object_detections:
            for detection in frame_result.get("detections", []):
                class_name = detection.get("class", "").lower()
                confidence = detection.get("confidence", 0.0)
                
                # Only include tools with reasonable confidence
                if class_name in tool_items and confidence > 0.3:
                    tools.add(class_name.title())
        
        return list(tools)
    
    def _extract_steps_from_scenes(self, scenes, object_detections, metadata):
        """Extract cooking steps from scene data."""
        steps = []
        
        if not scenes:
            # Create a basic step if no scenes detected
            steps.append({
                "idx": 1,
                "start": 0.0,
                "end": metadata.get("duration", 20.0),
                "action": "prepare",
                "objects": [],
                "details": "Prepare ingredients and cook according to video",
                "temp": None,
                "duration": None
            })
            return steps
        
        # Create steps based on scenes
        for i, scene in enumerate(scenes):
            step_idx = i + 1
            start_time = scene.get("start", 0.0)
            end_time = scene.get("end", metadata.get("duration", 20.0))
            
            # Get objects detected in this scene
            scene_objects = self._get_objects_in_timeframe(object_detections, start_time, end_time, metadata)
            
            # Determine action based on objects and scene
            action = self._determine_cooking_action(scene_objects, i, len(scenes))
            
            # Create step details
            details = self._generate_step_details(action, scene_objects, step_idx)
            
            steps.append({
                "idx": step_idx,
                "start": start_time,
                "end": end_time,
                "action": action,
                "objects": scene_objects,
                "details": details,
                "temp": None,
                "duration": end_time - start_time
            })
        
        return steps
    
    def _get_objects_in_timeframe(self, object_detections, start_time, end_time, metadata):
        """Get objects detected within a specific timeframe."""
        objects = set()
        fps = metadata.get("fps", 25.0)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        for frame_result in object_detections:
            frame_idx = frame_result.get("frame_idx", 0)
            if start_frame <= frame_idx <= end_frame:
                for detection in frame_result.get("detections", []):
                    class_name = detection.get("class", "").title()
                    confidence = detection.get("confidence", 0.0)
                    if confidence > 0.3:
                        objects.add(class_name)
        
        return list(objects)
    
    def _determine_cooking_action(self, objects, scene_idx, total_scenes):
        """Determine cooking action based on objects and scene position."""
        if scene_idx == 0:
            return "prepare"
        elif scene_idx == total_scenes - 1:
            return "serve"
        elif any(obj.lower() in ['knife', 'cutting board'] for obj in objects):
            return "chop"
        elif any(obj.lower() in ['pan', 'pot', 'stove'] for obj in objects):
            return "cook"
        elif any(obj.lower() in ['bowl', 'mixer', 'whisk'] for obj in objects):
            return "mix"
        else:
            return "prepare"
    
    def _generate_step_details(self, action, objects, step_idx):
        """Generate detailed step description."""
        if not objects:
            return f"Step {step_idx}: {action.title()} ingredients"
        
        object_list = ", ".join(objects[:3])  # Limit to first 3 objects
        if len(objects) > 3:
            object_list += f" and {len(objects) - 3} more items"
        
        return f"Step {step_idx}: {action.title()} {object_list}"
    
    def _enhance_ingredients_with_audio(self, ingredients, transcription):
        """Enhance ingredients list with information from audio transcription."""
        text = transcription.get("text", "").lower()
        
        # Look for quantity mentions in audio
        import re
        quantity_patterns = [
            r'(\d+(?:\.\d+)?)\s*(cup|cups|tablespoon|tablespoons|tbsp|teaspoon|teaspoons|tsp|ounce|ounces|oz|pound|pounds|lb|gram|grams|g|kilogram|kilograms|kg|ml|milliliter|milliliters|l|liter|liters)',
            r'(half|quarter|third|pinch|dash|handful|bunch|clove|cloves)',
            r'(\d+(?:/\d+)?)\s*(cup|cups|tablespoon|tablespoons|tbsp|teaspoon|teaspoons|tsp)'
        ]
        
        # Try to match quantities with ingredients
        for ingredient in ingredients:
            name = ingredient["name"].lower()
            if name in text:
                # Look for quantity near the ingredient name
                for pattern in quantity_patterns:
                    matches = re.findall(pattern, text)
                    for match in matches:
                        if isinstance(match, tuple):
                            qty, unit = match
                        else:
                            qty, unit = match, "piece"
                        
                        # If we find a quantity near this ingredient, update it
                        ingredient["qty"] = qty
                        ingredient["unit"] = unit
                        break
        
        return ingredients
    
    def _enhance_steps_with_audio(self, steps, transcription):
        """Enhance steps with information from audio transcription."""
        text = transcription.get("text", "")
        segments = transcription.get("segments", [])
        
        if not segments:
            return steps
        
        # Try to match audio segments to steps
        for i, step in enumerate(steps):
            step_start = step["start"]
            step_end = step["end"]
            
            # Find audio segments that overlap with this step
            overlapping_segments = []
            for segment in segments:
                seg_start = segment.get("start", 0)
                seg_end = segment.get("end", 0)
                
                # Check for overlap
                if not (seg_end < step_start or seg_start > step_end):
                    overlapping_segments.append(segment)
            
            # Combine overlapping segments
            if overlapping_segments:
                combined_text = " ".join([seg.get("text", "") for seg in overlapping_segments])
                if combined_text.strip():
                    # Enhance step details with audio
                    step["details"] = f"{step['details']} ({combined_text.strip()})"
        
        return steps
    
    def _generate_recipe_title(self, ingredients, transcription=None):
        """Generate recipe title based on main ingredients and audio."""
        # First try to extract title from audio
        if transcription and transcription.get("text"):
            text = transcription["text"].lower()
            # Look for common recipe title patterns
            title_patterns = [
                r'how to make (.+?)(?:recipe|dish|food)',
                r'(.+?) recipe',
                r'let\'s make (.+?)(?:recipe|dish|food)',
                r'today we\'re making (.+?)(?:recipe|dish|food)'
            ]
            
            import re
            for pattern in title_patterns:
                match = re.search(pattern, text)
                if match:
                    title = match.group(1).strip().title()
                    if len(title) > 3:  # Avoid very short titles
                        return f"{title} Recipe"
        
        # Fallback to ingredient-based title
        if not ingredients:
            return "Cooking Video Recipe"
        
        # Get the most frequently mentioned ingredients
        main_ingredients = [ing["name"] for ing in ingredients[:3]]
        if len(main_ingredients) == 1:
            return f"{main_ingredients[0]} Recipe"
        elif len(main_ingredients) == 2:
            return f"{main_ingredients[0]} and {main_ingredients[1]} Recipe"
        else:
            return f"{main_ingredients[0]}, {main_ingredients[1]} and {main_ingredients[2]} Recipe"
    
    def _estimate_servings(self, ingredients, transcription=None):
        """Estimate servings based on ingredient quantities and audio."""
        # First try to extract servings from audio
        if transcription and transcription.get("text"):
            text = transcription["text"].lower()
            import re
            serving_patterns = [
                r'serves?\s+(\d+)',
                r'(\d+)\s+servings?',
                r'feeds?\s+(\d+)',
                r'for\s+(\d+)\s+people'
            ]
            
            for pattern in serving_patterns:
                match = re.search(pattern, text)
                if match:
                    servings = match.group(1)
                    if servings.isdigit() and 1 <= int(servings) <= 20:
                        return servings
        
        # Fallback to ingredient-based estimation
        if not ingredients:
            return "4"
        
        # Simple heuristic based on number of ingredients
        ingredient_count = len(ingredients)
        if ingredient_count <= 3:
            return "2"
        elif ingredient_count <= 6:
            return "4"
        elif ingredient_count <= 10:
            return "6"
        else:
            return "8"
    
    def print_summary(self, recipe):
        """
        Print a summary of the extracted recipe.
        
        Args:
            recipe (dict): Extracted recipe.
        """
        print("\nRecipe Extraction Summary:")
        print(f"Title: {recipe.get('title', 'Unknown')}")
        print(f"Servings: {recipe.get('servings', 'Unknown')}")
        print(f"Ingredients: {len(recipe.get('ingredients', []))} items")
        print(f"Steps: {len(recipe.get('steps', []))} steps")
    
    def process_video(self, video_path, output_path=None):
        """
        Process a video and extract the recipe. This is an alias for the process method.
        
        Args:
            video_path (str): Path to the video file.
            output_path (str, optional): Path to save the output. Defaults to None.
            
        Returns:
            dict: Extracted recipe.
        """
        return self.process(video_path, output_path)
