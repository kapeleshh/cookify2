#!/usr/bin/env python3
"""
Cookify - Recipe Extraction from Cooking Videos
Setup script for installing the package and its dependencies.
"""

from setuptools import setup, find_packages

setup(
    name="cookify",
    version="0.1.0",
    description="Extract structured recipes from cooking videos",
    author="Cookify Team",
    packages=find_packages(),
    install_requires=[
        # Core dependencies - Python 3.12 compatible
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        
        # Video processing
        "opencv-python>=4.8.0",
        "ffmpeg-python>=0.2.0",
        "scenedetect==0.6.1",
        
        # Machine learning and computer vision
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        
        # OCR and text recognition
        "easyocr>=1.7.0",
        "pytesseract>=0.3.10",
        "Pillow>=8.3.1",
        
        # Audio processing and speech recognition
        "openai-whisper>=20231117",
        "librosa>=0.10.0",
        
        # NLP
        "spacy>=3.6.0",
        "nltk>=3.6.2",
        "transformers>=4.30.0",
        
        # Utilities
        "jsonschema>=3.2.0",
        "PyYAML>=6.0",
        "requests>=2.26.0",
        "psutil>=5.9.0",
        
        # Web interface
        "Flask>=2.3.0",
        "Werkzeug>=2.3.0",
        "Jinja2>=3.0.1",
        "itsdangerous>=2.0.1",
        "click>=8.0.1",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "cookify=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
