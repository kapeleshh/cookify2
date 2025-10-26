"""
Model Downloader - Utility to download pre-trained models for Cookify
"""

import os
import sys
import logging
import requests
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

# Define model directory
MODELS_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / "models"

# Define model configurations
MODEL_CONFIGS = {
    "yolov8": {
        "name": "YOLOv8x",
        "source": "ultralytics",
        "version": "yolov8x.pt",
        "url": None,  # Will be downloaded via the ultralytics package
        "description": "Object detection model for ingredients and tools"
    },
    "whisper": {
        "name": "Whisper Base",
        "source": "openai",
        "version": "base",
        "url": None,  # Will be downloaded via the whisper package
        "description": "Speech-to-text model for transcription"
    },
    "spacy": {
        "name": "spaCy English Model",
        "source": "spacy",
        "version": "en_core_web_lg",
        "url": None,  # Will be downloaded via spacy
        "description": "NLP model for language processing"
    },
    "mmaction": {
        "name": "MMAction2 TSN",
        "source": "mmaction2",
        "version": "tsn_r50_1x1x3_100e_kinetics400_rgb",
        "url": "https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth",
        "config_url": "https://raw.githubusercontent.com/open-mmlab/mmaction2/master/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py",
        "description": "Action recognition model"
    },
    "easyocr": {
        "name": "EasyOCR English",
        "source": "easyocr",
        "version": "english",
        "url": None,  # Will be downloaded via easyocr
        "description": "OCR model for text recognition"
    }
}

def download_file(url, destination):
    """Download a file from URL to destination with progress bar."""
    if os.path.exists(destination):
        logger.info(f"File already exists: {destination}")
        return
    
    logger.info(f"Downloading {url} to {destination}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def download_yolov8():
    """Download YOLOv8 model."""
    try:
        from ultralytics import YOLO
        
        model_path = MODELS_DIR / "yolov8" / "yolov8x.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        logger.info("Downloading YOLOv8x model...")
        model = YOLO("yolov8x.pt")
        
        # Save model info
        with open(MODELS_DIR / "yolov8" / "info.yaml", 'w') as f:
            yaml.dump({
                "name": MODEL_CONFIGS["yolov8"]["name"],
                "source": MODEL_CONFIGS["yolov8"]["source"],
                "version": MODEL_CONFIGS["yolov8"]["version"],
                "description": MODEL_CONFIGS["yolov8"]["description"],
                "path": str(model_path),
                "downloaded": True
            }, f)
        
        logger.info("YOLOv8x model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading YOLOv8 model: {e}")
        return False

def download_whisper():
    """Download Whisper model."""
    try:
        import whisper
        
        model_dir = MODELS_DIR / "whisper"
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info("Downloading Whisper base model...")
        model = whisper.load_model("base")
        
        # Save model info
        with open(model_dir / "info.yaml", 'w') as f:
            yaml.dump({
                "name": MODEL_CONFIGS["whisper"]["name"],
                "source": MODEL_CONFIGS["whisper"]["source"],
                "version": MODEL_CONFIGS["whisper"]["version"],
                "description": MODEL_CONFIGS["whisper"]["description"],
                "path": str(model_dir),
                "downloaded": True
            }, f)
        
        logger.info("Whisper model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading Whisper model: {e}")
        return False

def download_spacy():
    """Download spaCy model."""
    try:
        import spacy
        from spacy.cli import download
        
        model_name = "en_core_web_lg"
        model_dir = MODELS_DIR / "spacy"
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Downloading spaCy model: {model_name}")
        download(model_name)
        
        # Save model info
        with open(model_dir / "info.yaml", 'w') as f:
            yaml.dump({
                "name": MODEL_CONFIGS["spacy"]["name"],
                "source": MODEL_CONFIGS["spacy"]["source"],
                "version": MODEL_CONFIGS["spacy"]["version"],
                "description": MODEL_CONFIGS["spacy"]["description"],
                "path": str(model_dir),
                "downloaded": True
            }, f)
        
        logger.info("spaCy model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading spaCy model: {e}")
        return False

def download_mmaction():
    """Download MMAction2 model."""
    try:
        model_dir = MODELS_DIR / "mmaction"
        os.makedirs(model_dir, exist_ok=True)
        
        # Download model weights
        model_url = MODEL_CONFIGS["mmaction"]["url"]
        model_path = model_dir / "tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth"
        download_file(model_url, model_path)
        
        # Download model config
        config_url = MODEL_CONFIGS["mmaction"]["config_url"]
        config_path = model_dir / "tsn_r50_1x1x3_100e_kinetics400_rgb.py"
        download_file(config_url, config_path)
        
        # Save model info
        with open(model_dir / "info.yaml", 'w') as f:
            yaml.dump({
                "name": MODEL_CONFIGS["mmaction"]["name"],
                "source": MODEL_CONFIGS["mmaction"]["source"],
                "version": MODEL_CONFIGS["mmaction"]["version"],
                "description": MODEL_CONFIGS["mmaction"]["description"],
                "path": str(model_dir),
                "weights": str(model_path),
                "config": str(config_path),
                "downloaded": True
            }, f)
        
        logger.info("MMAction2 model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading MMAction2 model: {e}")
        return False

def download_easyocr():
    """Download EasyOCR model."""
    try:
        import easyocr
        
        model_dir = MODELS_DIR / "easyocr"
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info("Downloading EasyOCR English model...")
        reader = easyocr.Reader(['en'])
        
        # Save model info
        with open(model_dir / "info.yaml", 'w') as f:
            yaml.dump({
                "name": MODEL_CONFIGS["easyocr"]["name"],
                "source": MODEL_CONFIGS["easyocr"]["source"],
                "version": MODEL_CONFIGS["easyocr"]["version"],
                "description": MODEL_CONFIGS["easyocr"]["description"],
                "path": str(model_dir),
                "downloaded": True
            }, f)
        
        logger.info("EasyOCR model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading EasyOCR model: {e}")
        return False

def download_models():
    """Download all required models."""
    logger.info("Starting model downloads...")
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Download each model
    results = {
        "yolov8": download_yolov8(),
        "whisper": download_whisper(),
        "spacy": download_spacy(),
        "mmaction": download_mmaction(),
        "easyocr": download_easyocr()
    }
    
    # Log results
    logger.info("Model download summary:")
    for model, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {model}: {status}")
    
    # Check if all models were downloaded successfully
    if all(results.values()):
        logger.info("All models downloaded successfully")
    else:
        failed_models = [model for model, success in results.items() if not success]
        logger.warning(f"Failed to download some models: {', '.join(failed_models)}")
    
    return results

if __name__ == "__main__":
    # Configure logging when run as script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Download models
    download_models()
