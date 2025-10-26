#!/bin/bash
# Cookify Installation Script

# Exit on error
set -e

echo "=== Cookify Installation Script ==="
echo "This script will set up the Cookify environment and install dependencies."

# Check if Python 3.8+ is installed
echo "Checking Python version..."
if command -v python3 &>/dev/null; then
    python_version=$(python3 --version | cut -d' ' -f2)
    echo "Found Python $python_version"
    
    # Check Python version
    major=$(echo $python_version | cut -d. -f1)
    minor=$(echo $python_version | cut -d. -f2)
    
    if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 8 ]); then
        echo "Error: Cookify requires Python 3.8 or higher."
        echo "Please install Python 3.8+ and try again."
        exit 1
    fi
else
    echo "Error: Python 3 not found."
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

# Check if FFmpeg is installed
echo "Checking FFmpeg installation..."
if command -v ffmpeg &>/dev/null; then
    ffmpeg_version=$(ffmpeg -version | head -n1)
    echo "Found $ffmpeg_version"
else
    echo "Warning: FFmpeg not found."
    echo "FFmpeg is required for video processing."
    echo "Please install FFmpeg before using Cookify."
    
    # Suggest installation commands based on OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "For Ubuntu/Debian: sudo apt-get install ffmpeg"
        echo "For Fedora: sudo dnf install ffmpeg"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "For macOS with Homebrew: brew install ffmpeg"
    elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]]; then
        echo "For Windows: Download from https://ffmpeg.org/download.html"
    fi
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "Installing Cookify and dependencies..."
pip install -e .

# Download pre-trained models
echo "Downloading pre-trained models..."
python -m cookify.src.utils.model_downloader

echo ""
echo "=== Installation Complete ==="
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To use Cookify, run:"
echo "  cookify path/to/video.mp4"
echo ""
