# Multimodal Integration Phase

The Multimodal Integration phase is a critical component of the Cookify pipeline, responsible for combining information from different modalities (visual and audio) into a coherent representation. This phase aligns and integrates the outputs from the Frame Analysis and Audio Analysis phases to create a unified understanding of the cooking process.

## Purpose and Goals

The primary goals of the multimodal integration phase are:

1. **Temporal Alignment**: Align visual and audio events based on timestamps
2. **Cross-Modal Correlation**: Identify relationships between objects, actions, and spoken instructions
3. **Information Fusion**: Combine complementary information from different modalities
4. **Conflict Resolution**: Resolve conflicts when different modalities provide contradictory information
5. **Timeline Creation**: Create a coherent timeline of the cooking process

## Components

### MultimodalIntegrator

The `MultimodalIntegrator` class is responsible for integrating information from different modalities.

#### Key Methods

- `integrate(object_detections, text_detections, action_detections, transcription, nlp_results, scenes, metadata)`: Main integration method
- `_create_timeline(scenes, action_detections, transcription)`: Creates a timeline of events from scenes, actions, and transcription
- `_align_objects_with_timeline(object_detections, timeline)`: Aligns object detections with the timeline
- `_align_text_with_timeline(text_detections, timeline)`: Aligns text detections with the timeline
- `_extract_ingredients(nlp_results, aligned_objects, aligned_text)`: Extracts ingredients from NLP results and aligned visual data
- `_extract_tools(nlp_results, aligned_objects)`: Extracts tools from NLP results and aligned objects
- `_extract_steps(timeline, aligned_objects, aligned_text, nlp_results)`: Extracts cooking steps from the timeline and aligned data

## Implementation Details

### Timeline Creation

The timeline is the backbone of the integration process, providing a chronological structure for aligning events from different modalities:

1. **Event types**: The timeline includes different types of events (scenes, actions, speech) with start and end times.

2. **Chronological ordering**: Events are sorted by start time to create a sequential representation of the cooking process.

3. **Hierarchical structure**: Events can have parent-child relationships, with scenes containing actions and speech segments.

#### Code Example:
```python
def _create_timeline(self, scenes, action_detections, transcription):
    """
    Create a timeline of events from scenes, actions, and transcription.
    
    Args:
        scenes (list): Scene detection results.
        action_detections (list): Action detection results.
        transcription (dict): Audio transcription results.
        
    Returns:
        list: Timeline of events.
    """
    timeline = []
    
    # Add scenes to timeline
    for scene in scenes:
        event = {
            "type": "scene",
            "start_time": scene["start_time"],
            "end_time": scene["end_time"],
            "data": scene
        }
        
        timeline.append(event)
    
    # Add actions to timeline
    for action in action_detections:
        event = {
            "type": "action",
            "start_time": action["start_time"],
            "end_time": action["end_time"],
            "data": action
        }
        
        timeline.append(event)
    
    # Add transcription segments to timeline
    segments = transcription.get("segments", [])
    for segment in segments:
        event = {
            "type": "speech",
            "start_time": segment["start"],
            "end_time": segment["end"],
            "data": segment
        }
        
        timeline.append(event)
    
    # Sort timeline by start time
    timeline.sort(key=lambda x: x["start_time"])
    
    return timeline
```

### Object and Text Alignment

Object and text detections from frames need to be aligned with the timeline to establish when they appear in the cooking process:

1. **Frame-to-time conversion**: Frame indices are converted to timestamps based on the video's frame rate.

2. **Event association**: Each detection is associated with the timeline event that contains its timestamp.

3. **Structured representation**: Aligned detections include both the detection data and the associated event information.

#### Code Example:
```python
def _align_objects_with_timeline(self, object_detections, timeline):
    """
    Align object detections with timeline.
    
    Args:
        object_detections (list): Object detection results.
        timeline (list): Timeline of events.
        
    Returns:
        list: Aligned object detections.
    """
    aligned_objects = []
    
    # Assume 1 frame per second for simplicity
    # In a real implementation, we would use the video's FPS from metadata
    fps = 1.0
    
    for detection in object_detections:
        frame_idx = detection["frame_idx"]
        time = frame_idx / fps
        
        # Find the timeline event that contains this time
        event = None
        for e in timeline:
            if e["start_time"] <= time <= e["end_time"]:
                event = e
                break
        
        # If no event found, use the closest one
        if event is None:
            # Find closest event by start time
            timeline_sorted = sorted(timeline, key=lambda x: abs(x["start_time"] - time))
            if timeline_sorted:
                event = timeline_sorted[0]
        
        if event:
            aligned_object = {
                "time": time,
                "event_type": event["type"],
                "event_start": event["start_time"],
                "event_end": event["end_time"],
                "detections": detection["detections"]
            }
            
            aligned_objects.append(aligned_object)
    
    return aligned_objects
```

### Information Extraction

After alignment, the system extracts structured information by combining data from different modalities:

1. **Ingredient extraction**: Ingredients are extracted from NLP results and aligned visual detections.

2. **Tool extraction**: Tools are extracted from NLP results and aligned object detections.

3. **Step extraction**: Cooking steps are extracted from the timeline, combining actions, objects, and speech.

#### Code Example:
```python
def _extract_steps(self, timeline, aligned_objects, aligned_text, nlp_results):
    """
    Extract steps from timeline, aligned objects, aligned text, and NLP results.
    
    Args:
        timeline (list): Timeline of events.
        aligned_objects (list): Aligned object detections.
        aligned_text (list): Aligned text detections.
        nlp_results (dict): NLP processing results.
        
    Returns:
        list: Extracted steps.
    """
    steps = []
    
    # Extract actions from timeline
    action_events = [event for event in timeline if event["type"] == "action"]
    
    # Create a step for each action
    for i, event in enumerate(action_events):
        action = event["data"]["action"]
        
        # Find objects involved in this action
        objects = self._find_objects_for_event(event, aligned_objects)
        
        # Find text associated with this action
        text = self._find_text_for_event(event, aligned_text)
        
        # Find speech associated with this action
        speech_events = [e for e in timeline if e["type"] == "speech" and 
                       e["start_time"] <= event["end_time"] and 
                       e["end_time"] >= event["start_time"]]
        
        details = ""
        if speech_events:
            details = " ".join([e["data"]["text"] for e in speech_events])
        
        # Extract temperature and duration if available
        temp = None
        duration = None
        
        # Check for temperature in text
        for text_item in text:
            if "degree" in text_item.lower() or "°" in text_item:
                temp = text_item
                break
        
        # Check for duration in NLP results
        times = nlp_results.get("times", [])
        for time_item in times:
            if time_item["start"] >= event["start_time"] and time_item["end"] <= event["end_time"]:
                duration = f"{time_item['value']} {time_item['unit']}"
                break
        
        step = {
            "idx": i + 1,
            "start": event["start_time"],
            "end": event["end_time"],
            "action": action,
            "objects": objects,
            "details": details,
            "temp": temp,
            "duration": duration
        }
        
        steps.append(step)
    
    return steps
```

## Design Decisions and Rationale

### 1. Timeline-Based Integration

**Decision**: Use a timeline as the backbone for multimodal integration.

**Rationale**:
- **Temporal coherence**: Cooking is inherently a temporal process, and a timeline provides a natural way to represent this.
- **Alignment framework**: The timeline provides a common framework for aligning events from different modalities.
- **Step extraction**: The timeline makes it easier to extract sequential cooking steps.

### 2. Event-Centric Approach

**Decision**: Represent the cooking process as a sequence of events rather than a continuous stream.

**Rationale**:
- **Discrete actions**: Cooking involves discrete actions (e.g., chopping, mixing, baking) that are better represented as events.
- **Hierarchical structure**: Events can be organized hierarchically (scenes > actions > objects), which matches the structure of cooking processes.
- **Simplicity**: An event-based representation is simpler to work with than continuous signals.

### 3. Late Fusion Strategy

**Decision**: Perform modality fusion at a late stage, after each modality has been processed separately.

**Rationale**:
- **Modularity**: Each modality can be processed with specialized techniques before integration.
- **Robustness**: If one modality fails or provides poor results, the system can still use information from other modalities.
- **Flexibility**: Different fusion strategies can be applied based on the quality and availability of each modality.

### 4. Object-Action-Speech Association

**Decision**: Associate objects with actions and speech segments based on temporal overlap.

**Rationale**:
- **Contextual understanding**: Understanding which objects are involved in which actions is crucial for recipe extraction.
- **Instruction alignment**: Associating speech with actions helps understand the verbal instructions for each step.
- **Complementary information**: Different modalities provide complementary information (e.g., visual shows the action, audio explains the details).

## Challenges and Solutions

### 1. Temporal Misalignment

**Challenge**: Visual and audio events may not be perfectly synchronized.

**Solution**:
- **Flexible matching**: Allow for some temporal flexibility when matching events from different modalities.
- **Overlapping windows**: Use overlapping time windows rather than exact matches.
- **Confidence-based alignment**: Consider the confidence of detections when aligning events.

### 2. Conflicting Information

**Challenge**: Different modalities may provide conflicting information.

**Solution**:
- **Confidence weighting**: Weight information based on the confidence of the detection.
- **Modality prioritization**: Prioritize certain modalities for certain types of information (e.g., visual for objects, audio for instructions).
- **Consensus approach**: When possible, look for consensus across modalities.

### 3. Missing Information

**Challenge**: Information may be missing from one or more modalities.

**Solution**:
- **Cross-modal inference**: Infer missing information from other modalities.
- **Default values**: Use reasonable defaults when information is missing.
- **Explicit marking**: Clearly mark information that is inferred or uncertain.

### 4. Noise and Irrelevant Information

**Challenge**: Both visual and audio modalities may contain noise or irrelevant information.

**Solution**:
- **Filtering**: Filter out detections with low confidence or irrelevant to cooking.
- **Context-aware processing**: Use context to determine the relevance of information.
- **Domain knowledge**: Apply cooking domain knowledge to identify relevant information.

## Performance Considerations

1. **Memory usage**: The integration process can be memory-intensive, especially for long videos with many detections.

2. **Computational complexity**: The alignment and association algorithms need to be efficient to handle large numbers of events.

3. **Incremental processing**: For very long videos, incremental processing can be used to avoid loading everything into memory.

4. **Parallelization**: Some integration tasks can be parallelized to improve performance.

## Future Improvements

1. **Graph-based representation**: Represent the cooking process as a graph of events, objects, and actions to capture more complex relationships.

2. **Attention mechanisms**: Use attention mechanisms to focus on the most relevant parts of each modality.

3. **Temporal reasoning**: Improve the system's ability to reason about temporal relationships between events.

4. **Multi-modal transformers**: Use transformer-based models specifically designed for multi-modal integration.

5. **Active learning**: Implement an active learning approach to continuously improve the integration process based on user feedback.

## Conclusion

The Multimodal Integration phase is the heart of the recipe extraction pipeline, bringing together information from different modalities to create a coherent understanding of the cooking process. By aligning and integrating visual and audio information, the system can extract a structured recipe that captures the ingredients, tools, and steps involved in the cooking process.
