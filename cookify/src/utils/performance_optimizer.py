"""
Performance Optimizer - Optimizes the performance of the Cookify pipeline
"""

import os
import time
import logging
import numpy as np
import psutil
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """
    Class for optimizing the performance of the Cookify pipeline.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the PerformanceOptimizer.
        
        Args:
            config_path (str, optional): Path to the configuration file. Defaults to None.
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.profiling_results = {}
        self.optimization_results = {}
    
    def _load_config(self):
        """
        Load configuration from file.
        
        Returns:
            dict: Configuration dictionary.
        """
        default_config = {
            "batch_size": 16,
            "frame_sampling_rate": 5,
            "use_gpu": True,
            "parallel_processing": True,
            "num_workers": os.cpu_count() or 4,
            "cache_enabled": True,
            "cache_dir": "cache",
            "memory_limit_mb": 4096,  # 4GB
            "optimizations": {
                "video_processor": True,
                "object_detector": True,
                "scene_detector": True,
                "text_recognizer": True,
                "action_recognizer": True,
                "speech_recognizer": True,
                "nlp_processor": True,
                "multimodal_integrator": True,
                "recipe_extractor": True
            }
        }
        
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Merge with default config
                for key, value in config.items():
                    if key == "optimizations" and isinstance(value, dict):
                        default_config["optimizations"].update(value)
                    else:
                        default_config[key] = value
                
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        return default_config
    
    def save_config(self, path=None):
        """
        Save configuration to file.
        
        Args:
            path (str, optional): Path to save the configuration. Defaults to None.
        """
        save_path = path or self.config_path
        
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                with open(save_path, 'w') as f:
                    json.dump(self.config, f, indent=4)
                
                logger.info(f"Saved configuration to {save_path}")
            except Exception as e:
                logger.error(f"Error saving configuration: {e}")
    
    def profile_component(self, component_name, component, method_name, *args, **kwargs):
        """
        Profile a component's method.
        
        Args:
            component_name (str): Name of the component.
            component (object): Component instance.
            method_name (str): Name of the method to profile.
            *args: Arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.
            
        Returns:
            tuple: (result, profiling_data)
        """
        logger.info(f"Profiling {component_name}.{method_name}")
        
        # Get method
        method = getattr(component, method_name)
        
        # Measure memory usage before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure time
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        
        # Measure memory usage after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_usage = memory_after - memory_before
        
        # Create profiling data
        profiling_data = {
            "execution_time": execution_time,
            "memory_usage": memory_usage,
            "timestamp": time.time()
        }
        
        # Store profiling results
        if component_name not in self.profiling_results:
            self.profiling_results[component_name] = {}
        
        if method_name not in self.profiling_results[component_name]:
            self.profiling_results[component_name][method_name] = []
        
        self.profiling_results[component_name][method_name].append(profiling_data)
        
        logger.info(f"Profiled {component_name}.{method_name}: {execution_time:.2f}s, {memory_usage:.2f}MB")
        
        return result, profiling_data
    
    def optimize_video_processor(self, video_processor):
        """
        Optimize the VideoProcessor component.
        
        Args:
            video_processor: VideoProcessor instance.
            
        Returns:
            VideoProcessor: Optimized VideoProcessor instance.
        """
        if not self.config["optimizations"]["video_processor"]:
            return video_processor
        
        logger.info("Optimizing VideoProcessor")
        
        # Set batch processing
        video_processor.batch_size = self.config["batch_size"]
        
        # Set frame sampling rate
        video_processor.frame_sampling_rate = self.config["frame_sampling_rate"]
        
        # Enable parallel processing if configured
        if self.config["parallel_processing"]:
            video_processor.parallel_processing = True
            video_processor.num_workers = self.config["num_workers"]
        
        # Enable caching if configured
        if self.config["cache_enabled"]:
            video_processor.cache_enabled = True
            video_processor.cache_dir = self.config["cache_dir"]
        
        # Store optimization results
        self.optimization_results["video_processor"] = {
            "batch_size": video_processor.batch_size,
            "frame_sampling_rate": video_processor.frame_sampling_rate,
            "parallel_processing": getattr(video_processor, "parallel_processing", False),
            "num_workers": getattr(video_processor, "num_workers", 1),
            "cache_enabled": getattr(video_processor, "cache_enabled", False),
            "cache_dir": getattr(video_processor, "cache_dir", None)
        }
        
        return video_processor
    
    def optimize_object_detector(self, object_detector):
        """
        Optimize the ObjectDetector component.
        
        Args:
            object_detector: ObjectDetector instance.
            
        Returns:
            ObjectDetector: Optimized ObjectDetector instance.
        """
        if not self.config["optimizations"]["object_detector"]:
            return object_detector
        
        logger.info("Optimizing ObjectDetector")
        
        # Set device based on configuration
        if self.config["use_gpu"]:
            object_detector.device = "cuda"
        else:
            object_detector.device = "cpu"
        
        # Set batch processing
        object_detector.batch_size = self.config["batch_size"]
        
        # Store optimization results
        self.optimization_results["object_detector"] = {
            "device": getattr(object_detector, "device", "cpu"),
            "batch_size": getattr(object_detector, "batch_size", 1)
        }
        
        return object_detector
    
    def optimize_scene_detector(self, scene_detector):
        """
        Optimize the SceneDetector component.
        
        Args:
            scene_detector: SceneDetector instance.
            
        Returns:
            SceneDetector: Optimized SceneDetector instance.
        """
        if not self.config["optimizations"]["scene_detector"]:
            return scene_detector
        
        logger.info("Optimizing SceneDetector")
        
        # Enable parallel processing if configured
        if self.config["parallel_processing"]:
            scene_detector.parallel_processing = True
            scene_detector.num_workers = self.config["num_workers"]
        
        # Enable caching if configured
        if self.config["cache_enabled"]:
            scene_detector.cache_enabled = True
            scene_detector.cache_dir = self.config["cache_dir"]
        
        # Store optimization results
        self.optimization_results["scene_detector"] = {
            "parallel_processing": getattr(scene_detector, "parallel_processing", False),
            "num_workers": getattr(scene_detector, "num_workers", 1),
            "cache_enabled": getattr(scene_detector, "cache_enabled", False),
            "cache_dir": getattr(scene_detector, "cache_dir", None)
        }
        
        return scene_detector
    
    def optimize_text_recognizer(self, text_recognizer):
        """
        Optimize the TextRecognizer component.
        
        Args:
            text_recognizer: TextRecognizer instance.
            
        Returns:
            TextRecognizer: Optimized TextRecognizer instance.
        """
        if not self.config["optimizations"]["text_recognizer"] or text_recognizer is None:
            return text_recognizer
        
        logger.info("Optimizing TextRecognizer")
        
        # Set device based on configuration
        if self.config["use_gpu"]:
            text_recognizer.device = "cuda"
        else:
            text_recognizer.device = "cpu"
        
        # Set batch processing
        text_recognizer.batch_size = self.config["batch_size"]
        
        # Enable parallel processing if configured
        if self.config["parallel_processing"]:
            text_recognizer.parallel_processing = True
            text_recognizer.num_workers = self.config["num_workers"]
        
        # Store optimization results
        self.optimization_results["text_recognizer"] = {
            "device": getattr(text_recognizer, "device", "cpu"),
            "batch_size": getattr(text_recognizer, "batch_size", 1),
            "parallel_processing": getattr(text_recognizer, "parallel_processing", False),
            "num_workers": getattr(text_recognizer, "num_workers", 1)
        }
        
        return text_recognizer
    
    def optimize_action_recognizer(self, action_recognizer):
        """
        Optimize the ActionRecognizer component.
        
        Args:
            action_recognizer: ActionRecognizer instance.
            
        Returns:
            ActionRecognizer: Optimized ActionRecognizer instance.
        """
        if not self.config["optimizations"]["action_recognizer"] or action_recognizer is None:
            return action_recognizer
        
        logger.info("Optimizing ActionRecognizer")
        
        # Set device based on configuration
        if self.config["use_gpu"]:
            action_recognizer.device = "cuda"
        else:
            action_recognizer.device = "cpu"
        
        # Set batch processing
        action_recognizer.batch_size = self.config["batch_size"]
        
        # Store optimization results
        self.optimization_results["action_recognizer"] = {
            "device": getattr(action_recognizer, "device", "cpu"),
            "batch_size": getattr(action_recognizer, "batch_size", 1)
        }
        
        return action_recognizer
    
    def optimize_speech_recognizer(self, speech_recognizer):
        """
        Optimize the SpeechRecognizer component.
        
        Args:
            speech_recognizer: SpeechRecognizer instance.
            
        Returns:
            SpeechRecognizer: Optimized SpeechRecognizer instance.
        """
        if not self.config["optimizations"]["speech_recognizer"] or speech_recognizer is None:
            return speech_recognizer
        
        logger.info("Optimizing SpeechRecognizer")
        
        # Set device based on configuration
        if self.config["use_gpu"]:
            speech_recognizer.device = "cuda"
        else:
            speech_recognizer.device = "cpu"
        
        # Enable caching if configured
        if self.config["cache_enabled"]:
            speech_recognizer.cache_enabled = True
            speech_recognizer.cache_dir = self.config["cache_dir"]
        
        # Store optimization results
        self.optimization_results["speech_recognizer"] = {
            "device": getattr(speech_recognizer, "device", "cpu"),
            "cache_enabled": getattr(speech_recognizer, "cache_enabled", False),
            "cache_dir": getattr(speech_recognizer, "cache_dir", None)
        }
        
        return speech_recognizer
    
    def optimize_nlp_processor(self, nlp_processor):
        """
        Optimize the NLPProcessor component.
        
        Args:
            nlp_processor: NLPProcessor instance.
            
        Returns:
            NLPProcessor: Optimized NLPProcessor instance.
        """
        if not self.config["optimizations"]["nlp_processor"] or nlp_processor is None:
            return nlp_processor
        
        logger.info("Optimizing NLPProcessor")
        
        # Enable parallel processing if configured
        if self.config["parallel_processing"]:
            nlp_processor.parallel_processing = True
            nlp_processor.num_workers = self.config["num_workers"]
        
        # Store optimization results
        self.optimization_results["nlp_processor"] = {
            "parallel_processing": getattr(nlp_processor, "parallel_processing", False),
            "num_workers": getattr(nlp_processor, "num_workers", 1)
        }
        
        return nlp_processor
    
    def optimize_multimodal_integrator(self, multimodal_integrator):
        """
        Optimize the MultimodalIntegrator component.
        
        Args:
            multimodal_integrator: MultimodalIntegrator instance.
            
        Returns:
            MultimodalIntegrator: Optimized MultimodalIntegrator instance.
        """
        if not self.config["optimizations"]["multimodal_integrator"] or multimodal_integrator is None:
            return multimodal_integrator
        
        logger.info("Optimizing MultimodalIntegrator")
        
        # Enable parallel processing if configured
        if self.config["parallel_processing"]:
            multimodal_integrator.parallel_processing = True
            multimodal_integrator.num_workers = self.config["num_workers"]
        
        # Store optimization results
        self.optimization_results["multimodal_integrator"] = {
            "parallel_processing": getattr(multimodal_integrator, "parallel_processing", False),
            "num_workers": getattr(multimodal_integrator, "num_workers", 1)
        }
        
        return multimodal_integrator
    
    def optimize_recipe_extractor(self, recipe_extractor):
        """
        Optimize the RecipeExtractor component.
        
        Args:
            recipe_extractor: RecipeExtractor instance.
            
        Returns:
            RecipeExtractor: Optimized RecipeExtractor instance.
        """
        if not self.config["optimizations"]["recipe_extractor"] or recipe_extractor is None:
            return recipe_extractor
        
        logger.info("Optimizing RecipeExtractor")
        
        # No specific optimizations for RecipeExtractor yet
        
        # Store optimization results
        self.optimization_results["recipe_extractor"] = {}
        
        return recipe_extractor
    
    def optimize_pipeline(self, pipeline):
        """
        Optimize the entire pipeline.
        
        Args:
            pipeline: Pipeline instance.
            
        Returns:
            Pipeline: Optimized Pipeline instance.
        """
        logger.info("Optimizing Pipeline")
        
        # Optimize each component
        if hasattr(pipeline, "video_processor"):
            pipeline.video_processor = self.optimize_video_processor(pipeline.video_processor)
        
        if hasattr(pipeline, "object_detector"):
            pipeline.object_detector = self.optimize_object_detector(pipeline.object_detector)
        
        if hasattr(pipeline, "scene_detector"):
            pipeline.scene_detector = self.optimize_scene_detector(pipeline.scene_detector)
        
        if hasattr(pipeline, "text_recognizer"):
            pipeline.text_recognizer = self.optimize_text_recognizer(pipeline.text_recognizer)
        
        if hasattr(pipeline, "action_recognizer"):
            pipeline.action_recognizer = self.optimize_action_recognizer(pipeline.action_recognizer)
        
        if hasattr(pipeline, "speech_recognizer"):
            pipeline.speech_recognizer = self.optimize_speech_recognizer(pipeline.speech_recognizer)
        
        if hasattr(pipeline, "nlp_processor"):
            pipeline.nlp_processor = self.optimize_nlp_processor(pipeline.nlp_processor)
        
        if hasattr(pipeline, "multimodal_integrator"):
            pipeline.multimodal_integrator = self.optimize_multimodal_integrator(pipeline.multimodal_integrator)
        
        if hasattr(pipeline, "recipe_extractor"):
            pipeline.recipe_extractor = self.optimize_recipe_extractor(pipeline.recipe_extractor)
        
        # Set batch size for pipeline
        pipeline.batch_size = self.config["batch_size"]
        
        # Set frame sampling rate for pipeline
        pipeline.frame_sampling_rate = self.config["frame_sampling_rate"]
        
        # Enable caching if configured
        if self.config["cache_enabled"]:
            pipeline.cache_enabled = True
            pipeline.cache_dir = self.config["cache_dir"]
        
        # Store optimization results
        self.optimization_results["pipeline"] = {
            "batch_size": getattr(pipeline, "batch_size", 1),
            "frame_sampling_rate": getattr(pipeline, "frame_sampling_rate", 1),
            "cache_enabled": getattr(pipeline, "cache_enabled", False),
            "cache_dir": getattr(pipeline, "cache_dir", None)
        }
        
        return pipeline
    
    def benchmark(self, pipeline, video_path, num_runs=3):
        """
        Benchmark the pipeline.
        
        Args:
            pipeline: Pipeline instance.
            video_path (str): Path to the video file.
            num_runs (int, optional): Number of benchmark runs. Defaults to 3.
            
        Returns:
            dict: Benchmark results.
        """
        logger.info(f"Benchmarking pipeline with {video_path}")
        
        benchmark_results = {
            "video_path": video_path,
            "num_runs": num_runs,
            "runs": [],
            "average_time": 0.0,
            "average_memory": 0.0
        }
        
        for i in range(num_runs):
            logger.info(f"Benchmark run {i+1}/{num_runs}")
            
            # Measure memory usage before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Measure time
            start_time = time.time()
            result = pipeline.process_video(video_path)
            end_time = time.time()
            
            # Measure memory usage after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_usage = memory_after - memory_before
            
            # Store run results
            run_result = {
                "run_idx": i,
                "execution_time": execution_time,
                "memory_usage": memory_usage
            }
            
            benchmark_results["runs"].append(run_result)
            
            logger.info(f"Run {i+1}: {execution_time:.2f}s, {memory_usage:.2f}MB")
        
        # Calculate averages
        benchmark_results["average_time"] = sum(run["execution_time"] for run in benchmark_results["runs"]) / num_runs
        benchmark_results["average_memory"] = sum(run["memory_usage"] for run in benchmark_results["runs"]) / num_runs
        
        logger.info(f"Benchmark results: {benchmark_results['average_time']:.2f}s, {benchmark_results['average_memory']:.2f}MB")
        
        return benchmark_results
    
    def save_benchmark_results(self, benchmark_results, path):
        """
        Save benchmark results to file.
        
        Args:
            benchmark_results (dict): Benchmark results.
            path (str): Path to save the results.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(benchmark_results, f, indent=4)
            
            logger.info(f"Saved benchmark results to {path}")
        except Exception as e:
            logger.error(f"Error saving benchmark results: {e}")
    
    def save_profiling_results(self, path):
        """
        Save profiling results to file.
        
        Args:
            path (str): Path to save the results.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(self.profiling_results, f, indent=4)
            
            logger.info(f"Saved profiling results to {path}")
        except Exception as e:
            logger.error(f"Error saving profiling results: {e}")
    
    def save_optimization_results(self, path):
        """
        Save optimization results to file.
        
        Args:
            path (str): Path to save the results.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(self.optimization_results, f, indent=4)
            
            logger.info(f"Saved optimization results to {path}")
        except Exception as e:
            logger.error(f"Error saving optimization results: {e}")
    
    def generate_optimization_report(self, path):
        """
        Generate an optimization report.
        
        Args:
            path (str): Path to save the report.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            report = {
                "config": self.config,
                "profiling_results": self.profiling_results,
                "optimization_results": self.optimization_results,
                "recommendations": self._generate_recommendations()
            }
            
            with open(path, 'w') as f:
                json.dump(report, f, indent=4)
            
            logger.info(f"Generated optimization report at {path}")
        except Exception as e:
            logger.error(f"Error generating optimization report: {e}")
    
    def _generate_recommendations(self):
        """
        Generate optimization recommendations.
        
        Returns:
            dict: Optimization recommendations.
        """
        recommendations = {}
        
        # Check if GPU is available but not used
        try:
            import torch
            if torch.cuda.is_available() and not self.config["use_gpu"]:
                recommendations["use_gpu"] = "GPU is available but not used. Enable use_gpu for better performance."
        except ImportError:
            pass
        
        # Check if parallel processing is disabled
        if not self.config["parallel_processing"] and os.cpu_count() > 2:
            recommendations["parallel_processing"] = "Parallel processing is disabled. Enable it for better performance on multi-core systems."
        
        # Check if batch size is too small
        if self.config["batch_size"] < 8:
            recommendations["batch_size"] = "Batch size is small. Increase it for better throughput."
        
        # Check if caching is disabled
        if not self.config["cache_enabled"]:
            recommendations["cache_enabled"] = "Caching is disabled. Enable it to avoid redundant processing."
        
        return recommendations

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    optimizer = PerformanceOptimizer()
    
    # Save default configuration
    optimizer.save_config("config/performance_config.json")
    
    logger.info("Performance optimizer initialized")
