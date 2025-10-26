# Frame Analysis Phase

The Frame Analysis phase is a critical component of the Cookify pipeline, responsible for extracting visual information from video frames. This phase analyzes the extracted frames to detect objects, identify scene changes, recognize text overlays, and classify cooking actions.

## Purpose and Goals

The primary goals of the frame analysis phase are:

1. **Object Detection**: Identify ingredients, tools, and other cooking-related objects in video frames
2. **Scene Detection**: Segment the video into logical scenes based on visual changes
3. **Text Recognition**: Extract text overlays that may contain measurements, temperatures, or other recipe information
4. **Action Recognition**: Identify cooking actions being performed in sequences of frames

## Components

### 1. ObjectDetector

The `ObjectDetector` class identifies objects in video frames using YOLOv8, a state-of-the-art object detection model.

#### Key Methods

- `detect(frames)`: Detects objects in a list of frames
- `filter_cooking_related(detections)`: Filters detections to focus on cooking-related objects
- `_load_model()`: Lazy-loads the YOLOv8 model
- `_process_results(yolo_results, frame_idx)`: Processes YOLOv8 results into a structured format

### 2. SceneDetector

The `SceneDetector` class identifies scene changes in the video, which helps segment the cooking process into logical steps.

#### Key Methods

- `detect(video_path)`: Detects scenes in a video
- `get_scene_for_frame(scenes, frame_idx)`: Gets the scene that contains a specific frame
- `get_scene_for_timestamp(scenes, timestamp)`: Gets the scene that contains a specific timestamp

### 3. TextRecognizer

The `TextRecognizer` class extracts text from video frames using OCR (Optical Character Recognition).

#### Key Methods

- `recognize(frames)`: Recognizes text in a list of frames
- `filter_cooking_related(detections, cooking_keywords)`: Filters text detections to focus on cooking-related text
- `_enhance_text_regions(frame)`: Enhances text regions in a frame to improve OCR accuracy
- `_load_model()`: Lazy-loads the OCR model

### 4. ActionRecognizer

The `ActionRecognizer` class identifies cooking actions in sequences of frames.

#### Key Methods

- `recognize(frames, scenes)`: Recognizes actions in a list of frames, optionally using scene information
- `map_to_cooking_action(action)`: Maps a general action to a cooking-specific action
- `_compute_optical_flow(frames)`: Computes optical flow between consecutive frames
- `_load_model()`: Lazy-loads the action recognition model

## Implementation Details

### Object Detection with YOLOv8

YOLOv8 is used for object detection due to its excellent balance of speed and accuracy. The implementation includes:

1. **Pre-trained model**: Using a pre-trained YOLOv8 model eliminates the need for custom training.

2. **Cooking-related filtering**: A list of cooking-related classes from the COCO dataset is used to filter detections.

3. **Confidence thresholding**: A configurable confidence threshold filters out low-confidence detections.

#### Code Example:
```python
def detect(self, frames):
    """
    Detect objects in a list of frames.
    
    Args:
        frames (list): List of frames as numpy arrays.
        
    Returns:
        list: List of detection results for each frame.
    """
    self._load_model()
    
    results = []
    
    for i, frame in enumerate(tqdm(frames, desc="Detecting objects")):
        # Run inference
        yolo_results = self.model(frame, conf=self.confidence_threshold)
        
        # Process results
        frame_results = self._process_results(yolo_results, i)
        results.append(frame_results)
    
    return results
```

### Scene Detection with PySceneDetect

PySceneDetect is used for scene detection, which helps segment the video into logical parts:

1. **Content-based detection**: The ContentDetector algorithm identifies scene changes based on visual content.

2. **Configurable parameters**: Threshold and minimum scene length are configurable to adapt to different video styles.

3. **Structured output**: Scene information includes start/end frames, timestamps, and duration.

#### Code Example:
```python
def detect(self, video_path):
    """
    Detect scenes in a video.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
            list: List of scenes, where each scene is a dictionary with:
                - start_frame: Starting frame number
                - end_frame: Ending frame number
                - start_time: Starting time in seconds
                - end_time: Ending time in seconds
    """
    # Create video manager and scene manager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    
    # Add content detector
    scene_manager.add_detector(
        ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len)
    )
    
    # Improve processing speed by downscaling the video
    video_manager.set_downscale_factor()
    
    # Start video manager
    video_manager.start()
    
    # Perform scene detection
    scene_manager.detect_scenes(frame_source=video_manager)
    
    # Get scene list and frame metrics
    scene_list = scene_manager.get_scene_list()
    
    # Convert scene list to our format
    scenes = []
    for i, scene in enumerate(scene_list):
        start_frame = scene[0].frame_num
        end_frame = scene[1].frame_num - 1  # Inclusive end frame
        
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        scenes.append({
            "scene_idx": i,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time
        })
    
    return scenes
```

### Text Recognition with EasyOCR

EasyOCR is used for text recognition to extract measurements, temperatures, and other textual information:

1. **Pre-processing**: Text regions are enhanced before OCR to improve accuracy.

2. **Multi-language support**: The system supports multiple languages, defaulting to English.

3. **Confidence filtering**: Low-confidence text detections are filtered out.

#### Code Example:
```python
def recognize(self, frames):
    """
    Recognize text in a list of frames.
    
    Args:
        frames (list): List of frames as numpy arrays.
        
    Returns:
        list: List of text detection results for each frame.
    """
    self._load_model()
    
    results = []
    
    for i, frame in enumerate(tqdm(frames, desc="Recognizing text")):
        # Enhance text regions if configured
        if self.enhance_text:
            frame = self._enhance_text_regions(frame)
        
        # Run OCR
        ocr_results = self.reader.readtext(frame)
        
        # Process results
        frame_results = self._process_results(ocr_results, i)
        results.append(frame_results)
    
    return results
```

### Action Recognition

Action recognition identifies cooking actions in sequences of frames:

1. **Frame windowing**: Actions are recognized in windows of frames to capture temporal information.

2. **Scene-based processing**: When scene information is available, actions are recognized within scene boundaries.

3. **Optical flow (optional)**: Optical flow can be computed to improve action recognition accuracy.

#### Code Example:
```python
def recognize(self, frames, scenes=None):
    """
    Recognize actions in a list of frames.
    
    Args:
        frames (list): List of frames as numpy arrays.
        scenes (list, optional): List of scenes. If provided, actions will be recognized per scene.
        
    Returns:
        list: List of action recognition results.
    """
    self._load_model()
    
    # If no scenes are provided, treat the entire video as one scene
    if scenes is None:
        scenes = [{
            "scene_idx": 0,
            "start_frame": 0,
            "end_frame": len(frames) - 1,
            "start_time": 0.0,
            "end_time": len(frames) / 30.0,  # Assuming 30 fps
            "duration": len(frames) / 30.0
        }]
    
    results = []
    
    # Process each scene
    for scene in tqdm(scenes, desc="Recognizing actions"):
        # Get frames for this scene
        scene_start = max(0, scene["start_frame"])
        scene_end = min(len(frames) - 1, scene["end_frame"])
        
        scene_frames = frames[scene_start:scene_end+1]
        
        # Skip if not enough frames
        if len(scene_frames) < self.frame_window:
            continue
        
        # Sample frames at regular intervals to match frame_window
        sampled_indices = np.linspace(0, len(scene_frames) - 1, self.frame_window, dtype=int)
        sampled_frames = [scene_frames[i] for i in sampled_indices]
        
        # Recognize action
        action, confidence = self._recognize_action(sampled_frames)
        
        # Create result
        result = {
            "start_frame": scene_start,
            "end_frame": scene_end,
            "start_time": scene["start_time"],
            "end_time": scene["end_time"],
            "action": action,
            "confidence": confidence
        }
        
        results.append(result)
    
    return results
```

## Design Decisions and Rationale

### 1. Use of Pre-trained Models

**Decision**: Use pre-trained models rather than training custom models.

**Rationale**:
- **Efficiency**: Pre-trained models eliminate the need for collecting and annotating training data.
- **Performance**: State-of-the-art models like YOLOv8 and EasyOCR provide excellent performance out of the box.
- **Maintainability**: Pre-trained models are easier to maintain and update.

### 2. Lazy Loading of Models

**Decision**: Load models only when needed rather than at initialization.

**Rationale**:
- **Memory efficiency**: Models are only loaded into memory when they are actually used.
- **Startup performance**: The system starts up faster since models are not loaded immediately.
- **Resource optimization**: If a component is not used in a particular run, its model is never loaded.

### 3. Scene-Based Processing

**Decision**: Use scene detection to segment the video and process frames within scene boundaries.

**Rationale**:
- **Contextual coherence**: Scenes provide natural boundaries for cooking steps.
- **Performance optimization**: Some analyses can be performed at the scene level rather than the frame level.
- **Temporal structure**: Scenes help establish the temporal structure of the recipe.

### 4. Filtering Mechanisms

**Decision**: Implement filtering mechanisms for object and text detections.

**Rationale**:
- **Focus on relevant information**: Cooking videos contain many objects and text elements that are not relevant to the recipe.
- **Reduce noise**: Filtering reduces noise in the data, improving downstream processing.
- **Configurability**: Filtering can be adjusted based on the specific requirements of the recipe extraction.

## Performance Considerations

1. **Batch processing**: When possible, frames are processed in batches to leverage GPU parallelism.

2. **Resolution scaling**: For some analyses, frames can be downscaled to improve performance without significantly affecting accuracy.

3. **Model selection**: The choice of models balances accuracy and performance, with options for lighter models in resource-constrained environments.

4. **Caching**: Results are cached to avoid redundant processing, especially for expensive operations like OCR.

## Error Handling and Edge Cases

1. **Missing or corrupted frames**: The system handles missing or corrupted frames gracefully, continuing with available frames.

2. **Model loading failures**: If a model fails to load, the system provides a meaningful error message and can fall back to simpler analyses.

3. **Low-confidence detections**: Configurable confidence thresholds allow filtering out uncertain detections.

4. **Empty scenes**: Scenes with too few frames for meaningful analysis are skipped.

## Future Improvements

1. **Custom fine-tuning**: Fine-tune models on cooking-specific datasets to improve accuracy.

2. **Multi-modal fusion**: Integrate visual and textual information at the frame analysis level.

3. **Temporal modeling**: Improve action recognition with more sophisticated temporal modeling.

4. **Attention mechanisms**: Implement attention mechanisms to focus on relevant parts of frames.

## Conclusion

The Frame Analysis phase is a critical component of the recipe extraction pipeline, providing rich visual information that complements the audio analysis. The design decisions in this phase focus on leveraging state-of-the-art models while maintaining flexibility and performance.
