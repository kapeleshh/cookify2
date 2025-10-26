# Cookify: Recipe Extraction from Cooking Videos

Cookify is a Python-based tool that extracts structured recipe information from cooking videos. It uses computer vision, speech recognition, and natural language processing to identify ingredients, tools, cooking steps, and other recipe components.

## Features

- Extract recipe title, servings, and ingredients with quantities and units
- Identify cooking tools used in the video
- Extract step-by-step cooking instructions with timestamps
- Recognize cooking actions and techniques
- Generate structured JSON output of the complete recipe
- 🌐 **Modern web interface** for easy video upload and result visualization
- 🛡️ **Robust error handling** and graceful degradation
- 📱 **Responsive design** that works on desktop and mobile devices

## System Requirements

- Python 3.8 or higher
- FFmpeg (for video and audio processing)
- CUDA-compatible GPU recommended for faster processing (but not required)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/cookify.git
cd cookify
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
python -m cookify.src.utils.model_downloader
```

## Usage

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

## Web Interface

Cookify includes a modern web interface for easy video upload and recipe extraction.

### Starting the Web Server

1. **Ensure you've completed the installation steps above** (virtual environment, dependencies)

2. **Activate the virtual environment** (if not already active):
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Start the Flask web server** (choose one method):

   **Option A - Using the launch script (recommended):**
   ```bash
   ./cookify/launch_webapp.sh
   ```
   
   **Option B - Run directly:**
   ```bash
   python cookify/src/ui/app.py
   ```
   
   **Option C - From the cookify directory:**
   ```bash
   cd cookify
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

   You should see the Cookify homepage where you can upload cooking videos.

   **To stop the server**, press `CTRL+C` in the terminal.

### Using the Web Interface

#### Step 1: Upload a Video
- Drag and drop a cooking video onto the upload area, or
- Click "Browse Files" to select a video file
- Supported formats: MP4, AVI, MOV, MKV, WEBM (max 50MB)
- Wait for the upload to complete

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

### Troubleshooting Web Interface

If you encounter issues starting the web server:

1. **Check dependencies**: Ensure Flask is installed:
   ```bash
   pip install flask
   ```

2. **Verify Python path**: Make sure you're running from the project root directory

3. **Ensure models are downloaded**: The pipeline requires model files. Run:
   ```bash
   python -m cookify.src.utils.model_downloader
   ```

4. **Check port availability**: If port 5000 is in use, the server will show an error. You can modify the port in `src/ui/app.py` (last line):
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)  # Change port number
   ```

5. **Directory structure**: Ensure the project structure is correct. You should have:
   - `src/ui/app.py` (the web app)
   - `src/ui/templates/` (HTML templates)
   - `src/pipeline.py` (the processing pipeline)

6. **View logs**: The terminal will show detailed logs of any errors during startup or processing

7. **Common issues**:
   - If you see "ModuleNotFoundError", ensure you're in the venv and have run `pip install -e .`
   - If file upload fails, check that the `src/ui/uploads` and `src/ui/results` directories exist (they're created automatically on first run)
   - If processing fails, ensure FFmpeg is installed and in your PATH
   - On first startup, the pipeline initialization may take a few seconds to load models

## Output Format

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

## Project Structure

```
cookify/
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
│   ├── ui/             # Web interface
│   │   ├── app.py      # Flask web application
│   │   ├── templates/  # HTML templates
│   │   ├── uploads/    # Uploaded video files
│   │   └── results/    # Processing results
│   ├── utils/          # Utility functions
│   └── pipeline.py     # Main processing pipeline
├── tests/              # Unit tests
├── main.py             # Main entry point (CLI)
├── requirements.txt    # Dependencies
└── setup.py            # Setup script
```

### Key Files for Web Interface

- **`src/ui/app.py`** - Flask web application that handles video upload, processing, and result display
- **`src/ui/templates/`** - HTML templates for the web interface (index, results, about, error pages)
- **`src/pipeline.py`** - Core processing pipeline that extracts recipes from videos
```

## Development

### Running Tests

```bash
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

[MIT License](LICENSE)
