"""
VLM Analysis Module - Vision-Language Model integration for Cookify

This module provides Vision-Language Model capabilities using Ollama
for enhanced recipe extraction accuracy.

Components:
- ollama_engine: Core VLM engine wrapper
- vlm_prompts: Cooking-specific prompt templates
- ollama_frame_analyzer: Frame analysis using VLM
"""

__version__ = "0.1.0"
__author__ = "Cookify Team"

from .ollama_engine import OllamaVLMEngine
from .ollama_frame_analyzer import OllamaFrameAnalyzer

__all__ = [
    'OllamaVLMEngine',
    'OllamaFrameAnalyzer',
]

