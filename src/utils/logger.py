"""
Logger - Enhanced logging for the cookify application
"""

import os
import logging
import time
from pathlib import Path

class CookifyLogger:
    """Enhanced logger for cookify with timers and structured logging."""
    
    def __init__(self, config=None):
        """
        Initialize the logger.
        
        Args:
            config (dict, optional): Logger configuration. Defaults to None.
        """
        self.config = config or {"logging": {"level": "INFO", "log_dir": "logs"}}
        self.logger = self._setup_logger()
        self.timers = {}
        
    def _setup_logger(self):
        """Set up the logger."""
        logger = logging.getLogger("cookify")
        level_name = self.config["logging"].get("level", "INFO")
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)
        
        # Add console handler if no handlers
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # Add file handler
            log_dir = self.config["logging"].get("log_dir", "logs")
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(log_dir, "cookify.log")
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def get_logger(self):
        """Get the logger instance."""
        return self.logger
    
    def start_timer(self, name):
        """
        Start a timer with the given name.
        
        Args:
            name (str): Timer name.
        """
        self.timers[name] = time.time()
        self.logger.debug(f"Timer started: {name}")
    
    def end_timer(self, name):
        """
        End a timer and return the duration.
        
        Args:
            name (str): Timer name.
            
        Returns:
            float: Duration in seconds.
        """
        if name not in self.timers:
            self.logger.warning(f"Timer not found: {name}")
            return 0
        
        duration = time.time() - self.timers[name]
        self.logger.info(f"Timer {name} ended: {duration:.2f} seconds")
        return duration
    
    def log_processing_step(self, step, details=None):
        """
        Log a processing step with details.
        
        Args:
            step (str): Step name.
            details (dict, optional): Step details. Defaults to None.
        """
        details_str = f" - {details}" if details else ""
        self.logger.info(f"Processing step: {step}{details_str}")
    
    def log_error_with_context(self, error, context=None):
        """
        Log an error with context.
        
        Args:
            error (Exception): The error to log.
            context (dict, optional): Error context. Defaults to None.
        """
        context_str = f" - Context: {context}" if context else ""
        self.logger.error(f"Error: {error}{context_str}", exc_info=True)
    
    def log_recipe_extraction(self, recipe):
        """
        Log recipe extraction summary.
        
        Args:
            recipe (dict): Extracted recipe.
        """
        self.logger.info(
            f"Recipe extracted: {recipe.get('title', 'Unknown')} "
            f"({len(recipe.get('ingredients', []))} ingredients, "
            f"{len(recipe.get('steps', []))} steps)"
        )

def setup_pipeline_logging(config=None):
    """
    Set up logging for the pipeline.
    
    Args:
        config (dict, optional): Logger configuration. Defaults to None.
        
    Returns:
        CookifyLogger: Logger instance.
    """
    return CookifyLogger(config)
