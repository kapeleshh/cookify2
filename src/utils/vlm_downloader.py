"""
VLM Model Downloader - Download and manage Ollama Vision-Language Models
"""
import os
import sys
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Available VLM models for Ollama
AVAILABLE_VLM_MODELS = {
    "qwen2-vl:7b": {
        "description": "Qwen2-VL 7B - Recommended for cooking videos (best balance)",
        "size": "~4.7GB",
        "accuracy": "High",
        "speed": "Medium",
        "recommended": True
    },
    "qwen2-vl:2b": {
        "description": "Qwen2-VL 2B - Lightweight, faster inference",
        "size": "~1.6GB",
        "accuracy": "Medium",
        "speed": "Fast",
        "recommended": False
    },
    "llava:13b": {
        "description": "LLaVA 13B - Alternative VLM with strong reasoning",
        "size": "~7.4GB",
        "accuracy": "High",
        "speed": "Slow",
        "recommended": False
    },
    "llava:7b": {
        "description": "LLaVA 7B - Balanced alternative",
        "size": "~4.7GB",
        "accuracy": "Medium-High",
        "speed": "Medium",
        "recommended": False
    }
}


def check_ollama_installed() -> bool:
    """
    Check if Ollama is installed on the system.
    
    Returns:
        bool: True if Ollama is installed, False otherwise
    """
    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info(f"Ollama is installed: {version}")
            return True
        else:
            logger.warning("Ollama command failed")
            return False
            
    except FileNotFoundError:
        logger.warning("Ollama is not installed")
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama installation: {e}")
        return False


def check_ollama_running() -> bool:
    """
    Check if Ollama service is running.
    
    Returns:
        bool: True if Ollama is running, False otherwise
    """
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        
        if response.status_code == 200:
            logger.info("Ollama service is running")
            return True
        else:
            logger.warning("Ollama service returned unexpected status")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.warning("Cannot connect to Ollama service")
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama service: {e}")
        return False


def list_installed_models() -> List[str]:
    """
    List all models currently installed in Ollama.
    
    Returns:
        List[str]: List of installed model names
    """
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # Parse output (skip header line)
            lines = result.stdout.strip().split('\n')[1:]
            models = [line.split()[0] for line in lines if line.strip()]
            logger.info(f"Found {len(models)} installed models")
            return models
        else:
            logger.error("Failed to list Ollama models")
            return []
            
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return []


def pull_model(model_name: str) -> bool:
    """
    Pull a VLM model using Ollama.
    
    Args:
        model_name: Name of the model to pull (e.g., 'qwen2-vl:7b')
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Pulling model: {model_name}")
    logger.info("This may take several minutes depending on your internet connection...")
    
    try:
        # Run ollama pull with real-time output
        process = subprocess.Popen(
            ['ollama', 'pull', model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
        
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"✓ Successfully pulled {model_name}")
            return True
        else:
            logger.error(f"✗ Failed to pull {model_name}")
            return False
            
    except FileNotFoundError:
        logger.error("Ollama command not found. Please install Ollama first.")
        logger.info("Visit: https://ollama.com/download")
        return False
    except Exception as e:
        logger.error(f"Error pulling model: {e}")
        return False


def verify_model(model_name: str) -> bool:
    """
    Verify that a model is properly installed and working.
    
    Args:
        model_name: Name of the model to verify
        
    Returns:
        bool: True if model is working, False otherwise
    """
    logger.info(f"Verifying model: {model_name}")
    
    try:
        # Try to run a simple test query
        result = subprocess.run(
            ['ollama', 'run', model_name, 'Hello'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and result.stdout.strip():
            logger.info(f"✓ Model {model_name} is working correctly")
            return True
        else:
            logger.warning(f"Model {model_name} may not be working properly")
            return False
            
    except Exception as e:
        logger.error(f"Error verifying model: {e}")
        return False


def download_vlm_model(
    model_name: str = "qwen2-vl:7b",
    verify: bool = True
) -> bool:
    """
    Download and setup a VLM model for Cookify.
    
    Args:
        model_name: Name of the model to download
        verify: Whether to verify the model after download
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("=" * 70)
    logger.info("Cookify VLM Model Downloader")
    logger.info("=" * 70)
    
    # Step 1: Check if Ollama is installed
    if not check_ollama_installed():
        logger.error("✗ Ollama is not installed!")
        logger.info("\nTo install Ollama:")
        logger.info("  Linux/WSL: curl -fsSL https://ollama.com/install.sh | sh")
        logger.info("  Windows: Visit https://ollama.com/download/windows")
        logger.info("  macOS: brew install ollama")
        return False
    
    logger.info("✓ Ollama is installed")
    
    # Step 2: Check if Ollama service is running
    if not check_ollama_running():
        logger.warning("⚠ Ollama service is not running")
        logger.info("\nStarting Ollama service...")
        logger.info("Please run in a separate terminal: ollama serve")
        logger.info("Then run this script again.")
        return False
    
    logger.info("✓ Ollama service is running")
    
    # Step 3: Check if model is already installed
    installed_models = list_installed_models()
    
    if model_name in installed_models:
        logger.info(f"✓ Model {model_name} is already installed")
        
        if verify:
            return verify_model(model_name)
        return True
    
    # Step 4: Show model info
    if model_name in AVAILABLE_VLM_MODELS:
        info = AVAILABLE_VLM_MODELS[model_name]
        logger.info(f"\nModel: {model_name}")
        logger.info(f"Description: {info['description']}")
        logger.info(f"Size: {info['size']}")
        logger.info(f"Accuracy: {info['accuracy']}")
        logger.info(f"Speed: {info['speed']}")
        logger.info("")
    
    # Step 5: Pull the model
    logger.info(f"Downloading {model_name}...")
    logger.info("This may take several minutes...\n")
    
    success = pull_model(model_name)
    
    if not success:
        return False
    
    # Step 6: Verify the model
    if verify:
        time.sleep(2)  # Give Ollama a moment
        return verify_model(model_name)
    
    return True


def setup_vlm_environment():
    """
    Complete VLM environment setup for Cookify.
    """
    print("\n" + "=" * 70)
    print("Cookify VLM Environment Setup")
    print("=" * 70)
    
    # Check prerequisites
    print("\n1. Checking prerequisites...")
    
    if not check_ollama_installed():
        print("   ✗ Ollama not installed")
        print("\n   Please install Ollama:")
        print("   - Linux/WSL: curl -fsSL https://ollama.com/install.sh | sh")
        print("   - Windows: https://ollama.com/download/windows")
        print("   - macOS: brew install ollama")
        return False
    
    print("   ✓ Ollama installed")
    
    if not check_ollama_running():
        print("   ⚠ Ollama service not running")
        print("\n   Please start Ollama in another terminal:")
        print("   $ ollama serve")
        print("\n   Then run this script again.")
        return False
    
    print("   ✓ Ollama service running")
    
    # Show available models
    print("\n2. Available VLM Models:")
    print()
    
    for i, (model_name, info) in enumerate(AVAILABLE_VLM_MODELS.items(), 1):
        marker = "⭐" if info['recommended'] else "  "
        print(f"   {marker} {i}. {model_name}")
        print(f"      {info['description']}")
        print(f"      Size: {info['size']} | Accuracy: {info['accuracy']} | Speed: {info['speed']}")
        print()
    
    # Download recommended model
    print("3. Downloading recommended model (qwen2-vl:7b)...")
    print()
    
    success = download_vlm_model("qwen2-vl:7b", verify=True)
    
    if success:
        print("\n" + "=" * 70)
        print("✓ VLM Environment Setup Complete!")
        print("=" * 70)
        print("\nYou can now use Cookify with VLM support.")
        print("\nTo test the integration, run:")
        print("  $ python test_ollama_integration.py")
        return True
    else:
        print("\n" + "=" * 70)
        print("✗ VLM Environment Setup Failed")
        print("=" * 70)
        return False


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download and setup VLM models for Cookify'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='qwen2-vl:7b',
        help='Model to download (default: qwen2-vl:7b)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List installed models'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check Ollama installation and status'
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Run complete environment setup'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Installed models:")
        models = list_installed_models()
        for model in models:
            print(f"  - {model}")
        return 0
    
    if args.check:
        print("Checking Ollama installation...")
        installed = check_ollama_installed()
        running = check_ollama_running()
        
        print(f"  Installed: {'✓' if installed else '✗'}")
        print(f"  Running: {'✓' if running else '✗'}")
        
        if installed and running:
            models = list_installed_models()
            print(f"  Models: {len(models)}")
        
        return 0 if (installed and running) else 1
    
    if args.setup:
        success = setup_vlm_environment()
        return 0 if success else 1
    
    # Default: download specified model
    success = download_vlm_model(args.model, verify=True)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

