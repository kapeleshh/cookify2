# Preprocessing Phase

The preprocessing phase is the first step in the Cookify pipeline and is crucial for preparing the video data for subsequent analysis. This phase handles the extraction of frames and audio from the input video, as well as gathering metadata that will be useful for later processing stages.

## Purpose and Goals

The primary goals of the preprocessing phase are:

1. **Extract frames** from the video at appropriate intervals for visual analysis
2. **Extract audio** from the video for speech recognition and transcription
3. **Gather metadata** about the video (resolution, duration, frame rate, etc.)
4. **Prepare data structures** for subsequent processing phases

## Components

### VideoProcessor Class

The `VideoProcessor` class is the main component of the preprocessing phase. It handles all video processing tasks and provides a clean interface for the rest of the pipeline.

#### Key Methods

- `process(video_path)`: Main entry point that processes a video and returns frames, audio path, and metadata
- `_extract_metadata(video_path)`: Extracts metadata from the video file
- `_extract_frames(video_path, output_dir)`: Extracts frames from the video at specified intervals
- `_extract_audio(video_path, output_dir)`: Extracts audio from the video as a WAV file

## Implementation Details

### Frame Extraction Strategy

The frame extraction strategy is a critical design decision that balances processing speed, memory usage, and analysis accuracy:

1. **Configurable frame rate**: The system extracts frames at a configurable rate (default: 1 fps) rather than processing every frame. This significantly reduces processing time and memory usage while still capturing enough visual information for recipe extraction.

2. **Adaptive sampling**: For videos with varying content density, an adaptive sampling approach could be implemented in future versions, extracting more frames during high-activity segments and fewer during static scenes.

3. **Scene-based extraction**: When scene detection is enabled, the system can extract frames based on scene boundaries, ensuring that important visual transitions are captured.

#### Code Example:
```python
def _extract_frames(self, video_path, output_dir):
    """
    Extract frames from a video file.
    
    Args:
        video_path (str): Path to the video file.
        output_dir (Path): Directory to save extracted frames.
        
    Returns:
        list: List of extracted frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame extraction interval
    interval = int(fps / self.frame_rate)
    interval = max(1, interval)  # Ensure interval is at least 1
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract frame at specified interval
        if frame_idx % interval == 0:
            # Save frame to disk
            frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            # Add frame to list
            frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    
    return frames
```

### Audio Extraction

Audio extraction is performed using FFmpeg, which provides high-quality audio extraction with various configuration options:

1. **WAV format**: Audio is extracted in WAV format for maximum compatibility with speech recognition systems.

2. **Quality preservation**: The extraction process preserves audio quality to ensure optimal speech recognition results.

3. **Error handling**: Robust error handling ensures that the pipeline can continue even if audio extraction fails.

#### Code Example:
```python
def _extract_audio(self, video_path, output_dir):
    """
    Extract audio from a video file.
    
    Args:
        video_path (str): Path to the video file.
        output_dir (Path): Directory to save extracted audio.
        
    Returns:
        Path: Path to the extracted audio file
    """
    audio_path = output_dir / "audio.wav"
    
    # Use ffmpeg to extract audio
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        "-y",  # Overwrite output file if it exists
        str(audio_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e}")
        logger.error(f"ffmpeg stderr: {e.stderr.decode()}")
        raise
    
    return audio_path
```

### Metadata Extraction

Metadata extraction provides valuable context for later processing stages:

1. **Basic properties**: Width, height, frame rate, frame count, and duration.

2. **FFprobe integration**: For more detailed metadata, the system uses FFprobe to extract comprehensive information about the video.

3. **Structured format**: Metadata is returned in a structured format for easy access by other components.

#### Code Example:
```python
def _extract_metadata(self, video_path):
    """
    Extract metadata from a video file.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        dict: Dictionary containing video metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    # Extract basic metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    # Use ffprobe for more detailed metadata
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        ffprobe_data = result.stdout
        
        # Parse ffprobe output if available
        if ffprobe_data:
            import json
            ffprobe_json = json.loads(ffprobe_data)
        
    except Exception as e:
        logger.warning(f"Could not extract detailed metadata with ffprobe: {e}")
        ffprobe_json = {}
    
    metadata = {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "path": video_path,
        "ffprobe": ffprobe_json
    }
    
    return metadata
```

## Design Decisions and Rationale

### 1. Separate Frame and Audio Processing

**Decision**: Process frames and audio separately rather than in a single pass.

**Rationale**:
- **Specialization**: Different tools are optimal for different media types (OpenCV for frames, FFmpeg for audio).
- **Parallelization**: In future versions, frame and audio processing could be parallelized for performance gains.
- **Error isolation**: Issues in one processing stream don't affect the other.

### 2. Frame Sampling Strategy

**Decision**: Extract frames at regular intervals rather than processing every frame.

**Rationale**:
- **Performance**: Processing every frame would be prohibitively expensive for longer videos.
- **Diminishing returns**: For recipe extraction, additional frames often provide redundant information.
- **Configurability**: The frame rate is configurable to balance performance and accuracy for different use cases.

### 3. Disk Storage of Intermediate Results

**Decision**: Save extracted frames and audio to disk rather than keeping everything in memory.

**Rationale**:
- **Memory efficiency**: Videos can be large, and keeping all frames in memory could cause out-of-memory errors.
- **Checkpointing**: Saved files allow for resuming processing if an error occurs.
- **Debugging**: Saved files facilitate debugging and manual inspection of intermediate results.

### 4. Use of FFmpeg for Audio Extraction

**Decision**: Use FFmpeg for audio extraction rather than implementing custom extraction.

**Rationale**:
- **Robustness**: FFmpeg handles a wide variety of video formats and codecs.
- **Quality**: FFmpeg provides high-quality audio extraction with configurable parameters.
- **Community support**: FFmpeg is widely used and well-documented.

## Performance Considerations

1. **Frame extraction rate**: The default frame rate of 1 fps balances processing speed and accuracy, but can be adjusted based on video content and available computational resources.

2. **Disk I/O**: Saving frames to disk introduces I/O overhead, which could be optimized in future versions by using memory mapping or streaming processing.

3. **Video codec handling**: Different video codecs have different decoding performance characteristics. The system uses OpenCV and FFmpeg, which handle most common codecs efficiently.

## Error Handling and Edge Cases

1. **Corrupted videos**: The system includes error handling for corrupted or unreadable videos, providing meaningful error messages.

2. **Videos without audio**: For videos without audio tracks, the system gracefully handles the absence of audio and continues with visual processing only.

3. **Very long videos**: For very long videos, the system uses frame sampling to keep memory usage manageable.

## Future Improvements

1. **Adaptive frame sampling**: Implement intelligent frame sampling that adapts to video content, extracting more frames during high-activity segments.

2. **Parallel processing**: Process frames and audio in parallel to improve performance.

3. **Streaming processing**: Implement streaming processing to handle very large videos without loading everything into memory.

4. **GPU acceleration**: Utilize GPU acceleration for video decoding and frame processing to improve performance.

## Conclusion

The preprocessing phase lays the foundation for the entire recipe extraction pipeline by providing high-quality frames and audio for subsequent analysis. The design decisions in this phase focus on balancing performance, accuracy, and resource usage, while providing flexibility through configuration options.
