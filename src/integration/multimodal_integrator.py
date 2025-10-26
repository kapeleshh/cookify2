"""
Multimodal Integrator - Integrates information from video, audio, and text analysis
"""

import os
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MultimodalIntegrator:
    """
    Class for integrating information from video, audio, and text analysis.
    """
    
    def __init__(self):
        """
        Initialize the MultimodalIntegrator.
        """
        pass
    
    def integrate(self, object_detections, text_detections, action_detections, 
                 transcription, nlp_results, scenes, metadata):
        """
        Integrate information from various sources.
        
        Args:
            object_detections (list): Object detection results.
            text_detections (list): Text detection results.
            action_detections (list): Action detection results.
            transcription (dict): Audio transcription results.
            nlp_results (dict): NLP processing results.
            scenes (list): Scene detection results.
            metadata (dict): Video metadata.
            
        Returns:
            dict: Integrated information.
        """
        logger.info("Integrating multimodal information...")
        
        # Create timeline of events
        timeline = self._create_timeline(scenes, action_detections, transcription)
        
        # Align objects with timeline
        aligned_objects = self._align_objects_with_timeline(object_detections, timeline)
        
        # Align text with timeline
        aligned_text = self._align_text_with_timeline(text_detections, timeline)
        
        # Extract ingredients
        ingredients = self._extract_ingredients(nlp_results, aligned_objects, aligned_text)
        
        # Extract tools
        tools = self._extract_tools(nlp_results, aligned_objects)
        
        # Extract steps
        steps = self._extract_steps(timeline, aligned_objects, aligned_text, nlp_results)
        
        # Extract title and servings
        title = nlp_results.get("title", "")
        servings = nlp_results.get("servings", "")
        
        # Create integrated result
        integrated_data = {
            "title": title,
            "servings": servings,
            "ingredients": ingredients,
            "tools": tools,
            "steps": steps,
            "timeline": timeline,
            "metadata": metadata
        }
        
        logger.info(f"Integration complete: {len(ingredients)} ingredients, {len(tools)} tools, {len(steps)} steps")
        return integrated_data
    
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
        
        # Assuming object_detections has frame_idx and we need to convert to time
        # This is a simplified implementation
        
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
    
    def _align_text_with_timeline(self, text_detections, timeline):
        """
        Align text detections with timeline.
        
        Args:
            text_detections (list): Text detection results.
            timeline (list): Timeline of events.
            
        Returns:
            list: Aligned text detections.
        """
        aligned_text = []
        
        # Similar to _align_objects_with_timeline
        # This is a simplified implementation
        
        # Assume 1 frame per second for simplicity
        fps = 1.0
        
        for detection in text_detections:
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
                aligned_text_item = {
                    "time": time,
                    "event_type": event["type"],
                    "event_start": event["start_time"],
                    "event_end": event["end_time"],
                    "detections": detection["detections"]
                }
                
                aligned_text.append(aligned_text_item)
        
        return aligned_text
    
    def _extract_ingredients(self, nlp_results, aligned_objects, aligned_text):
        """
        Extract ingredients from NLP results, aligned objects, and aligned text.
        
        Args:
            nlp_results (dict): NLP processing results.
            aligned_objects (list): Aligned object detections.
            aligned_text (list): Aligned text detections.
            
        Returns:
            list: Extracted ingredients.
        """
        ingredients = []
        
        # Extract ingredients from NLP results
        nlp_ingredients = nlp_results.get("ingredients", [])
        for ingredient in nlp_ingredients:
            # Handle both "quantity" and "qty" field names
            qty = ingredient.get("qty", ingredient.get("quantity", ""))
            ingredients.append({
                "name": ingredient.get("name", ""),
                "qty": qty,
                "unit": ingredient.get("unit", "")
            })
        
        # Extract ingredients from aligned objects
        # This is a simplified implementation
        # In a real implementation, we would use more sophisticated techniques
        
        # Extract ingredients from aligned text
        # This is a simplified implementation
        
        # Deduplicate ingredients
        unique_ingredients = []
        seen_names = set()
        
        for ingredient in ingredients:
            name = ingredient["name"].lower()
            if name and name not in seen_names:
                seen_names.add(name)
                unique_ingredients.append(ingredient)
        
        return unique_ingredients
    
    def _extract_tools(self, nlp_results, aligned_objects):
        """
        Extract tools from NLP results and aligned objects.
        
        Args:
            nlp_results (dict): NLP processing results.
            aligned_objects (list): Aligned object detections.
            
        Returns:
            list: Extracted tools.
        """
        tools = []
        
        # Extract tools from NLP results
        nlp_tools = nlp_results.get("tools", [])
        for tool in nlp_tools:
            tools.append(tool.get("tool", ""))
        
        # Extract tools from aligned objects
        # This is a simplified implementation
        
        # Deduplicate tools
        unique_tools = list(set(tools))
        
        return unique_tools
    
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
        
        # Check if NLP results already have steps
        nlp_steps = nlp_results.get("steps", [])
        if nlp_steps:
            # Format NLP steps into the expected format
            for step in nlp_steps:
                # Handle different step formats
                if isinstance(step, dict):
                    # Already a dict
                    formatted_step = {
                        "idx": step.get("idx", step.get("index", len(steps) + 1)),
                        "start": step.get("start", step.get("start_time", 0.0)),
                        "end": step.get("end", step.get("end_time", 0.0)),
                        "action": step.get("action", step.get("text", "")).split(":")[0].strip(),
                        "objects": step.get("objects", step.get("entities", [])),
                        "details": step.get("details", step.get("text", "")),
                        "temp": step.get("temp", None),
                        "duration": step.get("duration", None)
                    }
                    steps.append(formatted_step)
                else:
                    # Simple string
                    steps.append({
                        "idx": len(steps) + 1,
                        "start": 0.0,
                        "end": 0.0,
                        "action": "",
                        "objects": [],
                        "details": str(step),
                        "temp": None,
                        "duration": None
                    })
        
        # If no NLP steps, try to extract from timeline
        if not steps:
            # Extract actions from timeline
            action_events = [event for event in timeline if event["type"] == "action"]
            
            # Create a step for each action
            for i, event in enumerate(action_events):
                action = event["data"].get("action", "")
                
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
    
    def _find_objects_for_event(self, event, aligned_objects):
        """
        Find objects involved in an event.
        
        Args:
            event (dict): Event.
            aligned_objects (list): Aligned object detections.
            
        Returns:
            list: Objects involved in the event.
        """
        objects = []
        
        # Find objects that appear during this event
        for aligned_object in aligned_objects:
            if aligned_object["time"] >= event["start_time"] and aligned_object["time"] <= event["end_time"]:
                for detection in aligned_object["detections"]:
                    objects.append(detection["class"])
        
        # Deduplicate objects
        unique_objects = list(set(objects))
        
        return unique_objects
    
    def _find_text_for_event(self, event, aligned_text):
        """
        Find text associated with an event.
        
        Args:
            event (dict): Event.
            aligned_text (list): Aligned text detections.
            
        Returns:
            list: Text associated with the event.
        """
        text = []
        
        # Find text that appears during this event
        for aligned_text_item in aligned_text:
            if aligned_text_item["time"] >= event["start_time"] and aligned_text_item["time"] <= event["end_time"]:
                for detection in aligned_text_item["detections"]:
                    text.append(detection["text"])
        
        return text
