# Cookify: Recipe Extraction from Cooking Videos

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-beta-orange)

Cookify is a Python-based tool that extracts structured recipe information from cooking videos using computer vision, speech recognition, and natural language processing to identify ingredients, tools, cooking steps, and other recipe components.

<div align="center">
  <!-- Replace with an actual logo image once available -->
  <img src="https://via.placeholder.com/200x200?text=Cookify" alt="Cookify Logo" width="200"/>
</div>

## 📋 Table of Contents

- [Features](#-features)
- [Recent Improvements](#-recent-improvements)
- [System Requirements](#-system-requirements)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Web Interface](#-web-interface)
- [Output Format](#-output-format)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)
- [FAQ](#-faq)

## ✨ Features

- 🎥 **Extract recipe information** from cooking videos with ingredient quantities and units
- 🔧 **Identify cooking tools** used in the video
- 📝 **Extract step-by-step instructions** with timestamps and cooking actions
- 🍳 **Recognize cooking techniques** and temperatures
- 📤 **Generate structured JSON output** of complete recipes
- 🌐 **Modern web interface** for easy video upload and result visualization
- 🛡️ **Robust error handling** and graceful degradation
- 🔄 **Multiple model fallback strategies** for better reliability
- 📱 **Responsive design** that works on desktop and mobile devices
- ⬇️ **Drag & drop upload** for easy video processing

## 🚀 Recent Improvements

- **Fixed directory structure**: Eliminated nested `cookify` directory
- **Enhanced error handling**: Comprehensive error handling throughout the pipeline
- **Improved model loading**: Robust model loading with fallback strategies
- **Better dependency management**: Updated dependencies for Python 3.12 compatibility
- **Working examples**: Added functional example scripts
- **Graceful degradation**: System continues working even if optional dependencies are missing

## 💻 System Requirements

- Python 3.8 or higher (Python 3.12 supported)
- FFmpeg (for video and audio processing)
- CUDA-compatible GPU recommended for faster processing (but not required)

## 🚀 Quick Start

Cookify offers two ways to process videos:

### Command Line Interface

```bash
# Clone and navigate to the project
git clone https://github.com/kapeleshh/cookify2.git
cd cookify2

# Install dependencies
pip install -e .

# Download models
python -m src.utils.model_downloader

# Process a video
cookify path/to/cooking/video.mp4
```

### Web Interface (Recommended)

```bash
# Clone and navigate to the project
git clone https://github.com/kapeleshh/cookify2.git
cd cookify2

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Download models
python -m src.utils.model_downloader

# Start the web server
python src/ui/app.py
```

Then open your browser to [http://localhost:5000](http://localhost:5000)

## 📥 Installation

### 1. Clone the repository

```bash
git clone https://github.com/kapeleshh/cookify2.git
cd cookify2
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the package and dependencies

```bash
pip install -e .
```

This will install all required dependencies including:
- OpenCV and FFmpeg for video processing
- PyTorch and YOLOv8 for object detection
- EasyOCR for text recognition
- Whisper for speech-to-text
- spaCy for NLP

### 4. Download pre-trained models

```bash
python -m src.utils.model_downloader
```

## 🛠️ Usage

### Basic Usage

```bash
cookify path/to/cooking/video.mp4
```

This will process the video and save the extracted recipe as `recipe.json` in the current directory.

### Advanced Options

```bash
cookify path/to/cooking/video.mp4 --output custom_output.json --verbose
```

For more options:

```bash
cookify --help
```

## 🌐 Web Interface

Cookify includes a modern web interface for easy video upload and recipe extraction.

<!-- Insert screenshot of web interface here -->
<div align="center">
  <img src="https://via.placeholder.com/800x450?text=Cookify+Web+Interface" alt="Cookify Web Interface" width="800"/>
</div>

### Starting the Web Server

1. **Ensure you've completed the installation steps above** (virtual environment, dependencies)

2. **Activate the virtual environment** (if not already active):
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Start the Flask web server** from the project root:
   ```bash
   python src/ui/app.py
   ```

   You should see output like:
   ```
   Initializing pipeline
   Pipeline initialized and optimized
   * Running on http://127.0.0.1:5000
   * Running on http://0.0.0.0:5000
   Press CTRL+C to quit
   ```

4. **Open your web browser** and navigate to:
   - **http://127.0.0.1:5000** (localhost)
   - **http://0.0.0.0:5000** (network access)

### Using the Web Interface

#### Step 1: Upload a Video
- Drag and drop a cooking video onto the upload area, or
- Click "Browse Files" to select a video file
- Supported formats: MP4, AVI, MOV, MKV, WEBM (max 50MB)
- Wait for the upload to complete

<!-- Insert screenshot of upload interface here -->
<div align="center">
  <img src="https://via.placeholder.com/800x450?text=Video+Upload+Interface" alt="Upload Interface" width="800"/>
</div>

#### Step 2: Processing
- The system will automatically process your video after upload
- A progress indicator shows the processing status
- Processing time depends on video length (typically 1-5 minutes)
- You'll see updates in real-time

#### Step 3: View Results
- Once complete, you'll be redirected to the results page
- The results page displays:
  - Recipe title and metadata (servings, total time)
  - Ingredients list with quantities and units
  - Step-by-step cooking instructions with timestamps
  - Cooking tools used
  - Interactive video player with clickable timestamps for each step

<!-- Insert screenshot of results page here -->
<div align="center">
  <img src="https://via.placeholder.com/800x450?text=Recipe+Results+Page" alt="Recipe Results" width="800"/>
</div>

#### Step 4: Download or Share
- Click "Download Recipe (JSON)" to save the structured recipe data
- Use "Upload Another Video" to process additional videos
- Share the results page URL to share with others (while the server is running)

### Web Interface Features

- **Modern UI**: Clean, responsive design built with Bootstrap 5
- **Drag & Drop Upload**: Easy video file upload with visual feedback
- **Real-time Progress**: Live progress updates during video processing
- **Interactive Results**: Click on cooking steps to jump to specific video timestamps
- **Mobile Friendly**: Responsive design works on desktop and mobile devices
- **Error Handling**: Clear error messages and graceful failure handling
- **Upload Management**: Automatic file organization with unique IDs
- **JSON Export**: Download recipe data in structured JSON format

### Troubleshooting Web Interface

If you encounter issues starting the web server:

1. **Check dependencies**: Ensure Flask is installed:
   ```bash
   pip install flask
   ```

2. **Verify Python path**: Make sure you're running from the project root directory

3. **Ensure models are downloaded**: The pipeline requires model files. Run:
   ```bash
   python -m src.utils.model_downloader
   ```

4. **Check port availability**: If port 5000 is in use, modify the port in `src/ui/app.py`:
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)  # Change port number
   ```

5. **View logs**: The terminal will show detailed logs of any errors during startup or processing

For more troubleshooting information, see the [Troubleshooting Guide](documentation/troubleshooting.md).

## 📊 Output Format

The extracted recipe is saved as a JSON file with the following structure:

```json
{
  "title": "Recipe Title",
  "servings": "Number of servings",
  "ingredients": [
    {"name": "ingredient name", "qty": "quantity", "unit": "measurement unit"}
  ],
  "tools": ["tool1", "tool2"],
  "steps": [
    {
      "idx": "step number",
      "start": "timestamp start",
      "end": "timestamp end",
      "action": "cooking action",
      "objects": ["ingredients/tools involved"],
      "details": "additional instructions",
      "temp": "temperature (optional)",
      "duration": "cooking duration (optional)"
    }
  ]
}
```

## 📁 Project Structure

```
cookify2/
├── data/               # Data directory for input/output files
│   ├── input/          # Input videos
│   └── output/         # Output recipes and processed data
├── models/             # Pre-trained models
├── src/                # Source code
│   ├── preprocessing/  # Video preprocessing
│   ├── frame_analysis/ # Frame analysis (object detection, OCR)
│   ├── audio_analysis/ # Audio transcription and NLP
│   ├── integration/    # Multimodal integration
│   ├── recipe_extraction/ # Recipe structure extraction
│   ├── output_formatting/ # Output formatting
│   ├── utils/          # Utility functions
│   │   ├── config_loader.py  # Configuration management
│   │   ├── logger.py         # Enhanced logging system
│   │   ├── model_downloader.py # Model management
│   │   └── performance_optimizer.py # Performance optimization
│   ├── ui/             # Web interface
│   │   ├── app.py      # Flask web application
│   │   ├── templates/  # HTML templates
│   │   ├── uploads/    # Uploaded video files
│   │   └── results/    # Processing results
│   └── pipeline.py     # Main processing pipeline
├── tests/              # Unit tests
├── documentation/      # Project documentation
├── examples/           # Example scripts
├── main.py             # Main entry point (CLI)
├── requirements.txt    # Dependencies
├── requirements-dev.txt # Development dependencies
├── config.yaml         # Configuration file
└── setup.py            # Setup script
```

### Key Files

- **`main.py`** - Command line interface entry point
- **`src/pipeline.py`** - Core processing pipeline for recipe extraction
- **`src/ui/app.py`** - Flask web application
- **`src/utils/config_loader.py`** - Configuration management
- **`src/utils/model_downloader.py`** - Model downloading and management

For a more detailed technical overview, see the [Architecture Documentation](documentation/01_architecture_overview.md).

## 👨‍💻 Development

### Setting Up a Development Environment

```bash
# Clone the repository
git clone https://github.com/kapeleshh/cookify2.git
cd cookify2

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e .
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run tests with coverage report
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_phase1.py
```

### Project Documentation

The `documentation/` directory contains detailed information about the project:

- [Documentation Index](documentation/00_documentation_index.md)
- [Architecture Overview](documentation/01_architecture_overview.md)
- [Preprocessing Phase](documentation/02_preprocessing_phase.md)
- [Frame Analysis Phase](documentation/03_frame_analysis_phase.md)
- [Audio Analysis Phase](documentation/04_audio_analysis_phase.md)
- [Multimodal Integration](documentation/05_multimodal_integration_phase.md)
- [Recipe Extraction](documentation/06_recipe_extraction_phase.md)
- [Output Formatting](documentation/07_output_formatting_phase.md)
- [Future Directions](documentation/08_future_directions.md)

### Machine Learning Models

Cookify uses several pre-trained machine learning models:

- **Object Detection**: YOLOv8 for identifying food items and cooking tools
- **Text Recognition**: EasyOCR for recognizing text in video frames
- **Speech Recognition**: Whisper for transcribing audio to text
- **Action Recognition**: Custom model for identifying cooking actions

For more information on how the models work together, see the [Multimodal Integration documentation](documentation/05_multimodal_integration_phase.md).

## 🤝 Contributing

We welcome contributions to Cookify! Here's how to get started:

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add some feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request

### Coding Standards

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, and modules
- Include type hints
- Ensure test coverage for new features

## 📄 License

[MIT License](LICENSE)

## ❓ FAQ

### Does Cookify work with all cooking videos?

Cookify works best with clear, well-lit cooking videos that feature close-ups of ingredients and cooking actions. While it can handle a variety of formats, professional cooking videos typically yield the best results.

### Do I need a GPU to run Cookify?

No, but a CUDA-compatible GPU will significantly speed up processing, especially for longer videos.

### How accurate is the recipe extraction?

Accuracy depends on several factors including video quality, clarity of actions, and audio quality. In optimal conditions, Cookify can achieve over 90% accuracy for ingredients and 80% for cooking steps.

### Can Cookify handle videos in languages other than English?

Currently, Cookify works best with English-language videos, but it can recognize ingredients and tools in any language. Audio transcription is optimized for English but can work with other languages with reduced accuracy.

---

<div align="center">
  <p>Made with ❤️ by the Cookify team</p>
  <p>
    <a href="https://github.com/kapeleshh/cookify2/issues">Report Bug</a> •
    <a href="https://github.com/kapeleshh/cookify2/issues">Request Feature</a>
  </p>
</div>
