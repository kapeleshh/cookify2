# Cookify Troubleshooting Guide

This guide addresses common issues that may arise when using Cookify and provides solutions and workarounds.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Web Interface Issues](#web-interface-issues)
- [Model Loading Issues](#model-loading-issues)
- [Video Processing Issues](#video-processing-issues)
- [Performance Issues](#performance-issues)
- [Output and Results Issues](#output-and-results-issues)
- [Dependency Issues](#dependency-issues)

## Installation Issues

### Package Installation Fails

**Problem:** Error messages when running `pip install -e .`

**Solutions:**

1. Ensure you have the latest pip version:
   ```bash
   python -m pip install --upgrade pip
   ```

2. Install specific problematic dependencies separately:
   ```bash
   pip install torch torchvision
   pip install -e .
   ```

3. Check for OS-specific dependencies:
   - On Windows: Ensure you have Visual C++ Build Tools installed
   - On Linux: Ensure you have python-dev packages installed:
     ```bash
     # Ubuntu/Debian
     sudo apt-get install python3-dev
     # Fedora
     sudo dnf install python3-devel
     ```

### FFmpeg Not Found

**Problem:** System cannot find FFmpeg, which is required for video processing

**Solutions:**

1. Install FFmpeg:
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`

2. Verify installation:
   ```bash
   ffmpeg -version
   ```

3. If installed but not found, add to PATH environment variable.

## Web Interface Issues

### Server Won't Start

**Problem:** Error when starting the web server with `python src/ui/app.py`

**Solutions:**

1. Check if Flask is installed:
   ```bash
   pip install flask
   ```

2. Check if port 5000 is already in use:
   ```bash
   # Linux/macOS
   lsof -i :5000
   # Windows
   netstat -ano | findstr :5000
   ```
   
   Change the port in `src/ui/app.py`:
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)  # Change port number
   ```

3. Ensure you're running from the project root directory (cookify2/).

### Upload Fails

**Problem:** Videos can't be uploaded through the interface

**Solutions:**

1. Check file size - default limit is 50MB
   
2. Verify upload directories exist and are writable:
   ```bash
   mkdir -p src/ui/uploads src/ui/results
   chmod 755 src/ui/uploads src/ui/results
   ```

3. Check browser console for JavaScript errors.

4. Try a different browser.

### Processing Stalls

**Problem:** Upload succeeds but processing gets stuck

**Solutions:**

1. Check server logs in the terminal.

2. Ensure you have sufficient disk space:
   ```bash
   # Linux/macOS
   df -h
   # Windows
   dir
   ```

3. Restart the server and try with a smaller video file.

4. Check that the required ML models are downloaded correctly.

## Model Loading Issues

### Models Not Downloading

**Problem:** `model_downloader.py` fails to download models

**Solutions:**

1. Check internet connection.

2. Verify you have write permission to the models directory:
   ```bash
   mkdir -p models
   chmod 755 models
   ```

3. Try downloading models manually:
   ```bash
   mkdir -p models/object_detection
   curl -L "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt" -o models/object_detection/yolov8n.pt
   ```

### Models Load Slowly

**Problem:** System takes a long time to initialize models

**Solutions:**

1. Switch to a smaller model in `config.yaml`:
   ```yaml
   object_detection:
     model: "yolov8n.pt"  # Use smallest model
   ```

2. Pre-load models before running the server:
   ```bash
   python -m src.utils.model_downloader
   ```

3. If you have a GPU, ensure CUDA is properly set up:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

## Video Processing Issues

### Video Conversion Fails

**Problem:** Error during video processing stage

**Solutions:**

1. Check video format compatibility:
   ```bash
   ffmpeg -i your_video.mp4 -f null -
   ```

2. Try converting your video to a different format:
   ```bash
   ffmpeg -i your_video.mp4 -c:v libx264 -c:a aac converted_video.mp4
   ```

3. Ensure video is not corrupted:
   ```bash
   ffmpeg -v error -i your_video.mp4 -f null - 2>error.log
   ```

### Processing Takes Too Long

**Problem:** Video processing is extremely slow

**Solutions:**

1. Use a shorter video for testing.

2. Try reducing the video resolution before processing:
   ```bash
   ffmpeg -i input.mp4 -vf "scale=640:360" -c:a copy reduced_input.mp4
   ```

3. Adjust frame extraction rate in `config.yaml`:
   ```yaml
   preprocessing:
     frame_rate: 0.5  # Extract one frame every 2 seconds
   ```

## Performance Issues

### High Memory Usage

**Problem:** System uses too much RAM during processing

**Solutions:**

1. Process smaller videos or reduce frame extraction rate.

2. Adjust batch size in `config.yaml`:
   ```yaml
   object_detection:
     batch_size: 4  # Reduce from default
   ```

3. Close other memory-intensive applications.

4. If on Linux, increase swap space.

### GPU Not Being Used

**Problem:** Processing is slow because GPU acceleration isn't working

**Solutions:**

1. Verify CUDA is available and set up correctly:
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
   ```

2. Ensure you have compatible NVIDIA drivers installed.

3. Verify GPU compatibility with PyTorch/CUDA version:
   ```bash
   python -c "import torch; print('PyTorch:', torch.__version__, ', CUDA:', torch.version.cuda)"
   ```

4. Force CUDA usage in `config.yaml`:
   ```yaml
   general:
     force_gpu: true
   ```

## Output and Results Issues

### No Ingredients Detected

**Problem:** The system doesn't identify ingredients in the video

**Solutions:**

1. Use a clearer video with good lighting and close-ups of ingredients.

2. Lower detection confidence threshold in `config.yaml`:
   ```yaml
   object_detection:
     confidence_threshold: 0.2  # Lower than default
   ```

3. Check if the model is loaded correctly by examining logs.

### Incorrect Recipe Structure

**Problem:** The output JSON has missing fields or incorrect structure

**Solutions:**

1. Ensure the video shows clear steps in chronological order.

2. Check if transcription is working by examining logs.

3. Update model weights if available.

4. Try with a professional cooking video that has clear narration.

## Dependency Issues

### Incompatible Python Version

**Problem:** Error messages related to Python version compatibility

**Solutions:**

1. Verify your Python version is 3.8 or higher:
   ```bash
   python --version
   ```

2. Create a new environment with a compatible Python version:
   ```bash
   conda create -n cookify python=3.8
   conda activate cookify
   ```

### CUDA/cuDNN Issues

**Problem:** Errors related to CUDA or cuDNN

**Solutions:**

1. Install compatible versions:
   ```bash
   pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. Fall back to CPU mode by modifying `config.yaml`:
   ```yaml
   general:
     force_cpu: true
   ```

### OpenCV Issues

**Problem:** Errors related to OpenCV

**Solutions:**

1. Reinstall OpenCV:
   ```bash
   pip uninstall opencv-python
   pip install opencv-python
   ```

2. For GUI-related errors on Linux, install dependencies:
   ```bash
   sudo apt-get install libsm6 libxext6 libxrender-dev
   ```

---

If you continue to experience issues not covered in this guide, please:

1. Check the terminal/console output for specific error messages
2. Search for the error message in the [GitHub issues](https://github.com/kapeleshh/cookify2/issues)
3. Submit a new issue with detailed information about the problem, including:
   - Error message
   - Operating system
   - Python version
   - Steps to reproduce the issue
   - Log files (if available)

We're continuously improving Cookify and appreciate your feedback!
