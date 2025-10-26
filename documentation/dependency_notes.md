# Dependency Notes

This document provides important information about the dependencies used in the Cookify project and how to handle potential installation issues.

## Key Dependencies

The Cookify project relies on several key dependencies:

1. **numpy**: Used for numerical operations and array handling
2. **opencv-python**: Used for video processing and computer vision tasks
3. **scenedetect**: Used for detecting scene changes in videos
4. **torch** and **torchvision**: Used for deep learning models
5. **easyocr**: Used for text recognition in video frames
6. **openai-whisper**: Used for speech-to-text transcription
7. **spacy**: Used for natural language processing

## Installation Notes

### Python Version

The project is designed to work with Python 3.7 or higher. Some dependencies may have specific Python version requirements:

- **numpy**: Version 1.22.4 is specified to avoid compatibility issues with different Python versions.
- **torch** and **torchvision**: May have specific Python version requirements depending on the platform.

### Package Names vs. Import Names

Some packages have different names on PyPI (for installation) versus their import names in Python code:

- **openai-whisper**: Install with `pip install openai-whisper`, but import with `import whisper`
- **scenedetect**: Install with `pip install scenedetect`, but import with specific modules like `from scenedetect import SceneManager, open_video`

### Optional Dependencies

Some dependencies are optional and can be commented out if they cause installation issues:

- **mmaction2**: This is used for action recognition but can be difficult to install. It's commented out in the requirements.txt file.

## Handling Installation Issues

If you encounter installation issues, try the following:

1. **Create a virtual environment**: Always use a virtual environment to avoid conflicts with system packages.
2. **Update pip**: Make sure you have the latest version of pip with `pip install --upgrade pip`.
3. **Install dependencies one by one**: If installing all dependencies at once fails, try installing them one by one.
4. **Check Python version**: Make sure you're using a compatible Python version.
5. **Platform-specific issues**: Some packages may have platform-specific installation requirements.

### Common Issues and Solutions

#### numpy version conflicts

If you see errors about numpy version conflicts, try:

```bash
pip install numpy==1.22.4
```

#### scenedetect installation issues

If you have issues installing scenedetect, make sure you're using the correct package name:

```bash
pip install scenedetect
```

#### openai-whisper installation issues

If you have issues installing openai-whisper, try:

```bash
pip install git+https://github.com/openai/whisper.git
```

#### torch installation issues

For torch and torchvision, it's often best to follow the official installation instructions from the PyTorch website, which will provide commands tailored to your specific platform and CUDA version.

## Updating Dependencies

When updating dependencies, be careful to maintain compatibility between packages. Always test the system after updating dependencies to ensure everything still works correctly.

## Future Dependency Management

In future phases of the project, we may consider using tools like Poetry or Conda for more robust dependency management, especially as the project grows in complexity.
