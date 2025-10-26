"""
Enhanced logging utilities for Cookify pipeline
"""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime

class CookifyLogger:
    """
    Enhanced logger for Cookify with structured logging and debugging capabilities.
    """
    
    def __init__(self, name: str, log_dir: str = "logs", level: str = "INFO"):
        """
        Initialize the Cookify logger.
        
        Args:
            name (str): Logger name.
            log_dir (str): Directory for log files.
            level (str): Logging level.
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = getattr(logging, level.upper())
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers()
        
        # Performance tracking
        self.performance_data = {}
        self.start_times = {}
    
    def _setup_handlers(self):
        """Setup logging handlers."""
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler for general logs
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "cookify.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(self.level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Error handler for errors only
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        self.logger.addHandler(error_handler)
        
        # Debug handler for detailed debugging
        debug_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "debug.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=2
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        debug_handler.setFormatter(debug_format)
        self.logger.addHandler(debug_handler)
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
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
    
    def log_performance_summary(self) -> Dict[str, Any]:
        """Log and return performance summary."""
        summary = {}
        for operation, times in self.performance_data.items():
            if times:
                summary[operation] = {
                    "count": len(times),
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }
        
        self.logger.info("Performance Summary:")
        for operation, stats in summary.items():
            self.logger.info(
                f"  {operation}: {stats['count']} calls, "
                f"avg: {stats['avg_time']:.2f}s, "
                f"total: {stats['total_time']:.2f}s"
            )
        
        return summary
    
    def log_processing_step(self, step: str, details: Dict[str, Any] = None) -> None:
        """Log a processing step with details."""
        message = f"Processing step: {step}"
        if details:
            message += f" - {json.dumps(details, indent=2)}"
        self.logger.info(message)
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log an error with additional context."""
        error_msg = f"Error: {str(error)}"
        if context:
            error_msg += f" - Context: {json.dumps(context, indent=2)}"
        self.logger.error(error_msg, exc_info=True)
    
    def log_data_quality(self, data_type: str, quality_metrics: Dict[str, Any]) -> None:
        """Log data quality metrics."""
        self.logger.info(f"Data quality for {data_type}: {json.dumps(quality_metrics, indent=2)}")
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, Any]) -> None:
        """Log model performance metrics."""
        self.logger.info(f"Model performance for {model_name}: {json.dumps(metrics, indent=2)}")
    
    def log_recipe_extraction(self, recipe_data: Dict[str, Any]) -> None:
        """Log recipe extraction results."""
        summary = {
            "title": recipe_data.get("title", "Unknown"),
            "ingredients_count": len(recipe_data.get("ingredients", [])),
            "tools_count": len(recipe_data.get("tools", [])),
            "steps_count": len(recipe_data.get("steps", [])),
            "servings": recipe_data.get("servings", "Unknown")
        }
        self.logger.info(f"Recipe extracted: {json.dumps(summary, indent=2)}")
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self.logger

def setup_pipeline_logging(config: Dict[str, Any]) -> CookifyLogger:
    """
    Setup logging for the pipeline based on configuration.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        CookifyLogger: Configured logger instance.
    """
    log_config = config.get("logging", {})
    log_dir = log_config.get("log_dir", "logs")
    log_level = log_config.get("level", "INFO")
    
    return CookifyLogger("cookify_pipeline", log_dir, log_level)

def create_component_logger(component_name: str, config: Dict[str, Any]) -> CookifyLogger:
    """
    Create a logger for a specific component.
    
    Args:
        component_name (str): Name of the component.
        config (dict): Configuration dictionary.
        
    Returns:
        CookifyLogger: Configured logger instance.
    """
    log_config = config.get("logging", {})
    log_dir = log_config.get("log_dir", "logs")
    log_level = log_config.get("level", "INFO")
    
    return CookifyLogger(f"cookify_{component_name}", log_dir, log_level)
