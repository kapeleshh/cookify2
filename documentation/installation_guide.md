# Cookify Installation Guide

This guide provides detailed instructions for installing Cookify and its dependencies on different operating systems.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Windows Installation](#windows-installation)
- [macOS Installation](#macos-installation)
- [Linux Installation](#linux-installation)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Cookify Installation](#cookify-installation)
- [Model Download](#model-download)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before installing Cookify, ensure your system meets the following requirements:

- **Python**: 3.8 or higher (Python 3.12 is fully supported)
- **Storage**: At least 2GB of free disk space
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **GPU**: Optional but recommended for faster processing (CUDA-compatible)
- **FFmpeg**: Required for video processing
- **Internet connection**: Required for downloading models

## Windows Installation

### 1. Install Python

1. Download Python from [python.org](https://www.python.org/downloads/windows/)
2. Run the installer and check "Add Python to PATH"
3. Verify installation by opening Command Prompt and typing:
   ```
   python --version
   ```

### 2. Install FFmpeg

1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows) or use a package manager like [Chocolatey](https://chocolatey.org/):
   ```
   choco install ffmpeg
   ```
2. Add FFmpeg to your PATH environment variable
3. Verify installation:
   ```
   ffmpeg -version
   ```

### 3. Install Git (if not already installed)

1. Download Git from [git-scm.com](https://git-scm.com/download/win)
2. Run the installer with default options
3. Verify installation:
   ```
   git --version
   ```

## macOS Installation

### 1. Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python, FFmpeg, and Git

```bash
brew install python ffmpeg git
```

### 3. Verify installations

```bash
python3 --version
ffmpeg -version
git --version
```

## Linux Installation

### For Ubuntu/Debian-based distributions:

```bash
# Update package list
sudo apt update

# Install Python and development tools
sudo apt install python3 python3-pip python3-dev python3-venv

# Install FFmpeg
sudo apt install ffmpeg

# Install Git (if not already installed)
sudo apt install git
```

### For Fedora/RHEL-based distributions:

```bash
# Install Python and development tools
sudo dnf install python3 python3-pip python3-devel

# Install FFmpeg
sudo dnf install ffmpeg

# Install Git (if not already installed)
sudo dnf install git
```

### Verify installations

```bash
python3 --version
ffmpeg -version
git --version
```

## Virtual Environment Setup

Creating a virtual environment is recommended to avoid dependency conflicts.

### Windows

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

### macOS/Linux

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

## Cookify Installation

Once you have the prerequisites and a virtual environment set up, you can install Cookify.

### 1. Clone the repository

```bash
git clone https://github.com/kapeleshh/cookify2.git
cd cookify2
```

### 2. Install Cookify and its dependencies

With your virtual environment activated:

```bash
pip install -e .
```

This will install all required dependencies including:
- OpenCV and FFmpeg for video processing
- PyTorch and YOLOv8 for object detection
- EasyOCR for text recognition
- Whisper for speech-to-text
- spaCy for NLP

## Model Download

Before using Cookify, download the necessary machine learning models:

```bash
python -m src.utils.model_downloader
```

This will download:
- YOLOv8 object detection model
- Scene detection models
- Text recognition models (as needed)

## Verification

Verify your installation by running:

```bash
# Check if the CLI works
cookify --help

# Try running a simple example (if you have a sample video)
cookify path/to/sample/video.mp4
```

## Troubleshooting

If you encounter any issues during installation, please refer to the [Troubleshooting Guide](troubleshooting.md) for common solutions.

### Common Installation Issues

1. **"Command not found" errors:**
   - Ensure your virtual environment is activated
   - Check that Python and pip are in your PATH

2. **Dependency installation fails:**
   - Try installing problematic dependencies separately:
     ```bash
     pip install torch torchvision
     pip install -e .
     ```

3. **FFmpeg not found:**
   - Verify FFmpeg is installed: `ffmpeg -version`
   - Make sure it's in your PATH environment variable

4. **GPU not being utilized:**
   - Check if CUDA is available:
     ```bash
     python -c "import torch; print(torch.cuda.is_available())"
     ```
   - Install appropriate NVIDIA drivers for your GPU

For additional assistance, please [open an issue](https://github.com/kapeleshh/cookify2/issues) on GitHub.

---

After completing installation, see the [User Manual](user_manual.md) (planned) for instructions on using Cookify.
