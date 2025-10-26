# Cookify Frequently Asked Questions (FAQ)

This document answers common questions about the Cookify project, its capabilities, and usage.

## General Questions

### What is Cookify?

Cookify is a Python-based tool that extracts structured recipe information from cooking videos. It uses computer vision, speech recognition, and natural language processing to identify ingredients, tools, cooking steps, and other recipe components.

### How does Cookify work?

Cookify processes cooking videos through a multi-stage pipeline:
1. **Preprocessing**: Extracts frames and audio from the video
2. **Frame Analysis**: Detects objects, scenes, text, and actions in video frames
3. **Audio Analysis**: Transcribes speech and extracts information from text
4. **Multimodal Integration**: Combines visual and audio information into a timeline
5. **Recipe Extraction**: Structures information into recipe components
6. **Output Formatting**: Generates structured output in various formats

### Is Cookify free to use?

Yes, Cookify is open-source software released under the MIT License, which allows free use, modification, and distribution.

## Technical Questions

### Does Cookify work with all cooking videos?

Cookify works best with clear, well-lit cooking videos that feature close-ups of ingredients and cooking actions. While it can handle a variety of formats, professional cooking videos typically yield the best results.

### Do I need a GPU to run Cookify?

No, but a CUDA-compatible GPU will significantly speed up processing, especially for longer videos. The object detection and video processing components benefit most from GPU acceleration.

### How accurate is the recipe extraction?

Accuracy depends on several factors including video quality, clarity of actions, and audio quality. In optimal conditions, Cookify can achieve over 90% accuracy for ingredients and 80% for cooking steps.

### Can Cookify handle videos in languages other than English?

Currently, Cookify works best with English-language videos, but it can recognize ingredients and tools in any language. Audio transcription is optimized for English but can work with other languages with reduced accuracy.

### What video formats are supported?

Cookify supports common video formats including:
- MP4 (recommended)
- AVI
- MOV
- MKV
- WEBM

### What is the maximum video size/length Cookify can process?

The web interface has a default limit of 50MB for uploaded videos. For the command-line interface, there is no hard limit, but processing very long videos (over 30 minutes) may require significant system resources and time.

## Installation & Setup Questions

### What are the minimum system requirements?

- Python 3.8 or higher
- 4GB RAM (8GB+ recommended)
- 2GB free disk space
- FFmpeg installed on the system

### Why do I get "No module named" errors after installation?

This usually happens if you're not using the virtual environment where Cookify was installed or if some dependencies failed to install. Make sure to activate your virtual environment and try reinstalling with:

```bash
pip install -e .
```

### How do I verify that my installation is working correctly?

Run the following command to test your installation:

```bash
python -m src.utils.model_downloader
cookify --help
```

If these commands complete without errors, your installation is working correctly.

## Usage Questions

### How do I extract a recipe from a video?

Using the command-line interface:
```bash
cookify path/to/video.mp4
```

Using the web interface:
1. Start the web server with `python src/ui/app.py`
2. Open your browser to http://localhost:5000
3. Upload your video and follow the on-screen instructions

### Where is the extracted recipe saved?

By default, the recipe is saved as `[video_name]_recipe.json` in the `data/output` directory. You can specify a custom output location:

```bash
cookify path/to/video.mp4 --output custom/path/output.json
```

### Can I get the output in formats other than JSON?

Currently, the primary output format is JSON. Future versions will include Markdown, YAML, and PDF output options.

### How can I improve the quality of recipe extraction?

For best results:
1. Use well-lit videos with clear narration
2. Choose videos with explicit ingredient measurements
3. Process videos where ingredients are clearly shown to the camera
4. Use videos with step-by-step cooking demonstrations
5. Adjust the confidence threshold in `config.yaml` if needed

### Can I use Cookify in my own application?

Yes! You can integrate Cookify into your application by:
1. Using it as a Python library (`from src.pipeline import Pipeline`)
2. Using the command-line interface through subprocess calls
3. Creating a REST API wrapper around the core functionality (not included but possible)

## Troubleshooting Questions

### Why is model downloading so slow?

Model downloads depend on your internet connection and the model size. The YOLOv8 models can be several hundred MB. If downloads are consistently failing, try downloading them manually as described in the [Troubleshooting Guide](troubleshooting.md).

### Why does Cookify crash with "out of memory" errors?

Processing videos, especially with object detection, can be memory-intensive. Try:
1. Processing shorter videos
2. Reducing the frame extraction rate in `config.yaml`
3. Using a smaller object detection model
4. Closing other memory-intensive applications

### The web interface is stuck at "Processing..." - what should I do?

Check the terminal where the server is running for error messages. Common issues include:
1. Video format incompatibility
2. Insufficient memory
3. Missing dependencies

For detailed troubleshooting steps, see the [Troubleshooting Guide](troubleshooting.md).

## Development Questions

### How can I contribute to Cookify?

We welcome contributions! See the [Contributing Guidelines](contributing.md) (planned) for details on:
- Setting up a development environment
- Coding standards
- Pull request process
- Issue reporting

### Can I extend Cookify with new features?

Yes! The modular architecture makes it easy to extend with new features:
- Add new object detection models in `src/frame_analysis/object_detector.py`
- Implement new output formats in `src/output_formatting/formatter.py`
- Improve recipe extraction in `src/recipe_extraction/recipe_extractor.py`

### Are there plans for future development?

Yes, see [Future Directions](08_future_directions.md) for planned improvements including:
- Better ingredient recognition
- Enhanced step extraction
- Multiple language support
- Improved user interface
- Mobile applications

---

If your question isn't answered here, check the [Troubleshooting Guide](troubleshooting.md) or [open an issue](https://github.com/kapeleshh/cookify2/issues) on the GitHub repository.
