"""
Model Downloader - Download and manage pre-trained models
"""

import os
import logging
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Dictionary of model URLs and their MD5 checksums
MODEL_URLS = {
    "object_detection": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "md5": "37b40b33987a193a84fe679917df75f3",
        "local_path": "models/object_detection/yolov8n.pt"
    },
    "scene_detection": {
        "url": "https://example.com/models/scene_detection_model.pt",
        "md5": "placeholder_md5_checksum",
        "local_path": "models/scene_detection/model.pt"
    },
    "text_recognition": {
        "url": "https://example.com/models/text_recognition_model.pt",
        "md5": "placeholder_md5_checksum",
        "local_path": "models/text_recognition/model.pt"
    }
}

def download_file(url, destination_path, chunk_size=8192):
    """
    Download a file from URL to the specified path with progress bar.
    
    Args:
        url (str): URL of the file to download.
        destination_path (str): Local path to save the file.
        chunk_size (int, optional): Size of chunks for downloading. Defaults to 8192.
    
    Returns:
        bool: True if download succeeded, False otherwise.
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size from headers
        file_size = int(response.headers.get('content-length', 0))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        desc = os.path.basename(destination_path)
        with open(destination_path, 'wb') as f, tqdm(
            desc=f"Downloading {desc}",
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        logger.info(f"Downloaded {url} to {destination_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def check_md5(file_path, expected_md5):
    """
    Check if a file's MD5 hash matches the expected value.
    
    Args:
        file_path (str): Path to the file to check.
        expected_md5 (str): Expected MD5 hash.
        
    Returns:
        bool: True if MD5 matches, False otherwise.
    """
    if not os.path.exists(file_path):
        return False
    
    # Calculate MD5 hash
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest() == expected_md5

def download_models():
    """
    Download all required models.
    
    Returns:
        bool: True if all downloads succeeded, False otherwise.
    """
    logger.info("Checking and downloading models...")
    
    all_succeeded = True
    for model_name, model_info in MODEL_URLS.items():
        local_path = model_info["local_path"]
        url = model_info["url"]
        md5 = model_info["md5"]
        
        # Check if model exists and has correct MD5
        if not check_md5(local_path, md5):
            logger.info(f"Model {model_name} not found or invalid, downloading...")
            success = download_file(url, local_path)
            if not success:
                all_succeeded = False
                logger.warning(f"Failed to download {model_name} model")
        else:
            logger.info(f"Model {model_name} already exists and is valid")
    
    return all_succeeded

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Download models
    download_models()
