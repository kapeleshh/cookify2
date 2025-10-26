# Cookify Architecture Overview

## Project Purpose

Cookify is a Python-based tool designed to extract structured recipe information from cooking videos. It uses computer vision, speech recognition, and natural language processing to identify ingredients, tools, cooking steps, and other recipe components, transforming unstructured video content into a structured recipe format.

## System Architecture

The system follows a pipeline architecture with modular components that process different aspects of the video. This design allows for:

1. **Separation of concerns**: Each module handles a specific task, making the code more maintainable and testable.
2. **Flexibility**: Components can be replaced or upgraded independently.
3. **Extensibility**: New features can be added by extending existing modules or adding new ones.

### High-Level Architecture Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Preprocessing  │────▶│ Frame Analysis  │────▶│ Audio Analysis  │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         │                      │                       │
         ▼                      ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    Multimodal Integration                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                     Recipe Extraction                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                     Output Formatting                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Preprocessing Module

**Purpose**: Prepare the video for analysis by extracting frames and audio.

**Key Components**:
- `VideoProcessor`: Handles video decoding, frame extraction, and audio separation.

**Design Decisions**:
- Frame extraction at configurable intervals to balance processing speed and accuracy.
- Audio extraction as a separate WAV file for optimal speech recognition.
- Metadata extraction to provide context for later processing stages.

### 2. Frame Analysis Module

**Purpose**: Analyze video frames to detect ingredients, tools, actions, and visual cues.

**Key Components**:
- `ObjectDetector`: Identifies ingredients and tools in frames using YOLOv8.
- `SceneDetector`: Identifies scene changes to segment the video into logical parts.
- `TextRecognizer`: Extracts text overlays using OCR for measurements, temperatures, etc.
- `ActionRecognizer`: Identifies cooking actions from sequences of frames.

**Design Decisions**:
- Use of pre-trained models to avoid the need for custom training.
- Lazy loading of models to optimize memory usage.
- Filtering mechanisms to focus on cooking-relevant detections.

### 3. Audio Analysis Module

**Purpose**: Convert speech to text and extract verbal instructions.

**Key Components**:
- `AudioTranscriber`: Transcribes speech to text using Whisper.
- `NLPProcessor`: Processes transcribed text to extract cooking-related information.

**Design Decisions**:
- Use of Whisper for state-of-the-art speech recognition.
- Specialized NLP processing for cooking terminology and instructions.
- Timestamp alignment for mapping transcription to video segments.

### 4. Multimodal Integration Module

**Purpose**: Combine visual and audio analysis to create a coherent understanding of the recipe.

**Key Components**:
- `MultimodalIntegrator`: Aligns and integrates information from different modalities.

**Design Decisions**:
- Timeline-based approach to align events from different modalities.
- Conflict resolution strategies when visual and audio information disagree.
- Structured intermediate representation for further processing.

### 5. Recipe Extraction Module

**Purpose**: Convert the multimodal analysis into a structured recipe format.

**Key Components**:
- `RecipeExtractor`: Extracts structured recipe information from integrated data.

**Design Decisions**:
- Inference mechanisms to fill in missing information.
- Normalization of quantities and units for consistency.
- Grouping of similar steps for clarity.

### 6. Output Formatting Module

**Purpose**: Format the extracted information into the required output format.

**Key Components**:
- `OutputFormatter`: Formats and saves the recipe in various formats (JSON, YAML, Markdown).

**Design Decisions**:
- Multiple output formats to support different use cases.
- Configurable inclusion of metadata like timestamps and confidence scores.
- Human-readable Markdown output for easy consumption.

## Configuration System

The system uses a YAML-based configuration system that allows users to customize various aspects of the pipeline:

- Frame extraction rate
- Model selection and parameters
- Processing options for each module
- Output formatting options

This configuration system makes the tool flexible and adaptable to different types of cooking videos and user preferences.

## Utility Components

- **Model Downloader**: Handles downloading and management of pre-trained models.
- **Configuration Loader**: Loads and validates configuration from YAML files.
- **Logging System**: Provides detailed logging for debugging and monitoring.

## Design Principles

1. **Modularity**: Each component has a single responsibility and clear interfaces.
2. **Configurability**: Users can customize the behavior without changing code.
3. **Robustness**: The system handles errors gracefully and provides fallbacks.
4. **Extensibility**: New models and techniques can be easily integrated.
5. **Usability**: Simple command-line interface and clear documentation.

## Technology Choices

- **Python**: Chosen for its rich ecosystem of ML and data processing libraries.
- **OpenCV & FFmpeg**: Industry-standard tools for video processing.
- **PyTorch & YOLOv8**: State-of-the-art object detection.
- **Whisper**: Leading speech-to-text model from OpenAI.
- **spaCy**: Powerful NLP library with customizable pipelines.

These technology choices balance performance, ease of use, and community support, making the system both powerful and maintainable.
