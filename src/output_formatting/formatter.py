"""
Output Formatter - Formats extracted recipe information for output
"""

import os
import logging
import json
import yaml
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class OutputFormatter:
    """
    Class for formatting extracted recipe information for output.
    """
    
    def __init__(self, format="json", include_confidence=False, include_timestamps=True, include_frame_refs=False):
        """
        Initialize the OutputFormatter.
        
        Args:
            format (str, optional): Output format. Defaults to "json".
                Options: "json", "yaml", "markdown"
            include_confidence (bool, optional): Whether to include confidence scores. Defaults to False.
            include_timestamps (bool, optional): Whether to include timestamps. Defaults to True.
            include_frame_refs (bool, optional): Whether to include frame references. Defaults to False.
        """
        self.format = format
        self.include_confidence = include_confidence
        self.include_timestamps = include_timestamps
        self.include_frame_refs = include_frame_refs
    
    def format(self, recipe):
        """
        Format extracted recipe information.
        
        Args:
            recipe (dict): Extracted recipe information.
            
        Returns:
            dict: Formatted recipe information.
        """
        logger.info(f"Formatting recipe in {self.format} format...")
        
        # Create a copy of the recipe to avoid modifying the original
        formatted_recipe = recipe.copy()
        
        # Format recipe based on selected format
        if self.format == "json":
            return self._format_json(formatted_recipe)
        elif self.format == "yaml":
            return self._format_yaml(formatted_recipe)
        elif self.format == "markdown":
            return self._format_markdown(formatted_recipe)
        else:
            logger.warning(f"Unknown format: {self.format}. Using JSON.")
            return self._format_json(formatted_recipe)
    
    def _format_json(self, recipe):
        """
        Format recipe as JSON.
        
        Args:
            recipe (dict): Recipe information.
            
        Returns:
            dict: Formatted recipe.
        """
        # Remove confidence scores if not included
        if not self.include_confidence:
            self._remove_confidence_scores(recipe)
        
        # Remove timestamps if not included
        if not self.include_timestamps:
            self._remove_timestamps(recipe)
        
        # Remove frame references if not included
        if not self.include_frame_refs:
            self._remove_frame_refs(recipe)
        
        return recipe
    
    def _format_yaml(self, recipe):
        """
        Format recipe as YAML.
        
        Args:
            recipe (dict): Recipe information.
            
        Returns:
            dict: Formatted recipe.
        """
        # Same as JSON format, but will be serialized as YAML
        return self._format_json(recipe)
    
    def _format_markdown(self, recipe):
        """
        Format recipe as Markdown.
        
        Args:
            recipe (dict): Recipe information.
            
        Returns:
            dict: Formatted recipe with additional markdown field.
        """
        # Format recipe as JSON
        json_recipe = self._format_json(recipe)
        
        # Generate Markdown representation
        markdown = self._generate_markdown(json_recipe)
        
        # Add Markdown representation to recipe
        json_recipe["markdown"] = markdown
        
        return json_recipe
    
    def _generate_markdown(self, recipe):
        """
        Generate Markdown representation of recipe.
        
        Args:
            recipe (dict): Recipe information.
            
        Returns:
            str: Markdown representation.
        """
        markdown = []
        
        # Title
        title = recipe.get("title", "Untitled Recipe")
        markdown.append(f"# {title}")
        markdown.append("")
        
        # Servings
        servings = recipe.get("servings", "")
        if servings:
            markdown.append(f"**Servings:** {servings}")
            markdown.append("")
        
        # Ingredients
        markdown.append("## Ingredients")
        markdown.append("")
        
        ingredients = recipe.get("ingredients", [])
        for ingredient in ingredients:
            name = ingredient.get("name", "")
            qty = ingredient.get("qty", "")
            unit = ingredient.get("unit", "")
            
            if qty and unit:
                markdown.append(f"- {qty} {unit} {name}")
            elif qty:
                markdown.append(f"- {qty} {name}")
            else:
                markdown.append(f"- {name}")
        
        markdown.append("")
        
        # Tools
        tools = recipe.get("tools", [])
        if tools:
            markdown.append("## Tools")
            markdown.append("")
            
            for tool in tools:
                markdown.append(f"- {tool}")
            
            markdown.append("")
        
        # Steps
        markdown.append("## Instructions")
        markdown.append("")
        
        steps = recipe.get("steps", [])
        for step in steps:
            idx = step.get("idx", 0)
            action = step.get("action", "")
            objects = step.get("objects", [])
            details = step.get("details", "")
            temp = step.get("temp", None)
            duration = step.get("duration", None)
            
            # Format step
            step_text = f"{idx}. {action.capitalize()}"
            
            if objects:
                step_text += f" {', '.join(objects)}"
            
            if details:
                step_text += f". {details}"
            
            if temp:
                step_text += f" at {temp}"
            
            if duration:
                step_text += f" for {duration}"
            
            markdown.append(step_text)
            
            # Add timestamps if included
            if self.include_timestamps:
                start = step.get("start", 0)
                end = step.get("end", 0)
                
                if start and end:
                    markdown.append(f"   *(Time: {self._format_timestamp(start)} - {self._format_timestamp(end)})*")
            
            markdown.append("")
        
        return "\n".join(markdown)
    
    def _format_timestamp(self, seconds):
        """
        Format timestamp in MM:SS format.
        
        Args:
            seconds (float): Timestamp in seconds.
            
        Returns:
            str: Formatted timestamp.
        """
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        
        return f"{minutes:02d}:{seconds:02d}"
    
    def _remove_confidence_scores(self, recipe):
        """
        Remove confidence scores from recipe.
        
        Args:
            recipe (dict): Recipe information.
        """
        # Remove confidence scores from steps
        steps = recipe.get("steps", [])
        for step in steps:
            if "confidence" in step:
                del step["confidence"]
    
    def _remove_timestamps(self, recipe):
        """
        Remove timestamps from recipe.
        
        Args:
            recipe (dict): Recipe information.
        """
        # Remove timestamps from steps
        steps = recipe.get("steps", [])
        for step in steps:
            if "start" in step:
                del step["start"]
            
            if "end" in step:
                del step["end"]
    
    def _remove_frame_refs(self, recipe):
        """
        Remove frame references from recipe.
        
        Args:
            recipe (dict): Recipe information.
        """
        # Remove frame references from steps
        steps = recipe.get("steps", [])
        for step in steps:
            if "frame_refs" in step:
                del step["frame_refs"]
    
    def save(self, recipe, output_path):
        """
        Save formatted recipe to file.
        
        Args:
            recipe (dict): Formatted recipe information.
            output_path (str): Output file path.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save recipe based on format
            if self.format == "json":
                with open(output_path, "w") as f:
                    json.dump(recipe, f, indent=2)
            elif self.format == "yaml":
                with open(output_path, "w") as f:
                    yaml.dump(recipe, f, default_flow_style=False)
            elif self.format == "markdown":
                # Save Markdown representation
                markdown_path = output_path
                if not markdown_path.endswith(".md"):
                    markdown_path = os.path.splitext(output_path)[0] + ".md"
                
                with open(markdown_path, "w") as f:
                    f.write(recipe.get("markdown", ""))
                
                # Also save JSON representation
                json_path = os.path.splitext(output_path)[0] + ".json"
                with open(json_path, "w") as f:
                    json.dump(recipe, f, indent=2)
            
            logger.info(f"Recipe saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving recipe: {e}")
            return False
