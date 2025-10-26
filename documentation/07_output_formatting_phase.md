# Output Formatting Phase

The Output Formatting phase is the final step in the Cookify pipeline, responsible for transforming the structured recipe data into user-friendly formats. This phase ensures that the extracted recipe is presented in a way that is easy to read, share, and use.

## Purpose and Goals

The primary goals of the output formatting phase are:

1. **Format Conversion**: Convert the structured recipe data into various output formats (JSON, YAML, Markdown)
2. **Presentation Enhancement**: Improve the readability and usability of the recipe
3. **Metadata Management**: Control the inclusion of metadata like timestamps and confidence scores
4. **File Output**: Save the formatted recipe to disk in the appropriate format
5. **User Customization**: Allow users to customize the output format according to their preferences

## Components

### OutputFormatter

The `OutputFormatter` class is responsible for formatting and saving the extracted recipe in various formats.

#### Key Methods

- `format(recipe)`: Format the recipe according to the specified format
- `_format_json(recipe)`: Format the recipe as JSON
- `_format_yaml(recipe)`: Format the recipe as YAML
- `_format_markdown(recipe)`: Format the recipe as Markdown
- `_generate_markdown(recipe)`: Generate a Markdown representation of the recipe
- `_format_timestamp(seconds)`: Format a timestamp in MM:SS format
- `_remove_confidence_scores(recipe)`: Remove confidence scores from the recipe
- `_remove_timestamps(recipe)`: Remove timestamps from the recipe
- `_remove_frame_refs(recipe)`: Remove frame references from the recipe
- `save(recipe, output_path)`: Save the formatted recipe to a file

## Implementation Details

### Format Conversion

The system supports multiple output formats to accommodate different use cases:

1. **JSON**: A machine-readable format that preserves the full structure of the recipe.

2. **YAML**: A human-readable format that is also machine-parseable.

3. **Markdown**: A text-based format that is optimized for human readability and can be rendered as formatted text.

#### Code Example:
```python
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
```

### Markdown Generation

Markdown generation creates a human-readable representation of the recipe:

1. **Structured sections**: The recipe is divided into clear sections (title, ingredients, tools, steps).

2. **Formatting**: Markdown formatting is applied to enhance readability.

3. **Timestamps**: Optional timestamps are included to link steps back to the video.

#### Code Example:
```python
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
```

### Metadata Management

The system allows users to control the inclusion of metadata in the output:

1. **Confidence scores**: Scores indicating the system's confidence in the extracted information.

2. **Timestamps**: Timestamps linking steps back to the original video.

3. **Frame references**: References to specific frames in the video.

#### Code Example:
```python
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
```

### File Output

The system saves the formatted recipe to disk in the appropriate format:

1. **JSON output**: The recipe is saved as a JSON file.

2. **YAML output**: The recipe is saved as a YAML file.

3. **Markdown output**: The recipe is saved as a Markdown file, with an optional JSON file for machine readability.

#### Code Example:
```python
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
```

## Design Decisions and Rationale

### 1. Multiple Output Formats

**Decision**: Support multiple output formats (JSON, YAML, Markdown).

**Rationale**:
- **Flexibility**: Different formats serve different purposes and user preferences.
- **Machine readability**: JSON and YAML are easily parseable by machines for further processing.
- **Human readability**: Markdown is optimized for human consumption and can be rendered as formatted text.

### 2. Configurable Metadata Inclusion

**Decision**: Make the inclusion of metadata (confidence scores, timestamps, frame references) configurable.

**Rationale**:
- **User preferences**: Different users have different needs regarding metadata.
- **Clarity**: Removing unnecessary metadata can make the recipe clearer and more concise.
- **Debugging**: Including metadata can be useful for debugging and improving the system.

### 3. Markdown as a Human-Readable Format

**Decision**: Use Markdown as the primary human-readable format.

**Rationale**:
- **Widespread support**: Markdown is supported by many platforms and can be easily converted to other formats.
- **Simple syntax**: Markdown's simple syntax makes it easy to generate and read.
- **Formatting capabilities**: Markdown provides enough formatting capabilities for recipe presentation.

### 4. Dual Output for Markdown

**Decision**: When outputting Markdown, also save a JSON version.

**Rationale**:
- **Completeness**: The JSON version preserves all information, including metadata that might be omitted in the Markdown.
- **Machine readability**: The JSON version can be easily parsed by machines for further processing.
- **Round-trip conversion**: The dual output allows for round-trip conversion between formats.

## Challenges and Solutions

### 1. Balancing Human and Machine Readability

**Challenge**: Different formats have different trade-offs between human and machine readability.

**Solution**:
- **Multiple formats**: Support multiple formats to accommodate different needs.
- **Dual output**: For human-readable formats, also provide a machine-readable version.
- **Configurable detail level**: Allow users to control the level of detail in the output.

### 2. Handling Complex Recipe Structures

**Challenge**: Some recipes have complex structures that are difficult to represent in simple formats.

**Solution**:
- **Hierarchical representation**: Use hierarchical structures to represent complex recipes.
- **Format-specific optimizations**: Optimize the representation for each format.
- **Simplification options**: Provide options to simplify complex recipes for better readability.

### 3. Internationalization and Localization

**Challenge**: Recipes may need to be presented in different languages and formats.

**Solution**:
- **Unicode support**: Ensure proper handling of Unicode characters for international recipes.
- **Localization hooks**: Provide hooks for localizing the output format.
- **Cultural adaptations**: Consider cultural differences in recipe presentation.

### 4. Accessibility

**Challenge**: The output should be accessible to users with disabilities.

**Solution**:
- **Semantic structure**: Use semantic structure in the output to improve accessibility.
- **Alternative text**: Include alternative text for visual elements.
- **Screen reader compatibility**: Ensure the output is compatible with screen readers.

## Performance Considerations

1. **Memory usage**: The formatting process should be memory-efficient, especially for large recipes.

2. **File I/O**: File operations should be optimized to minimize I/O overhead.

3. **Format conversion**: Format conversion should be efficient, especially for large recipes.

## Future Improvements

1. **Additional formats**: Support additional output formats, such as HTML, PDF, or specialized recipe formats.

2. **Interactive output**: Create interactive versions of the recipe, such as web applications or mobile apps.

3. **Customizable templates**: Allow users to customize the output format using templates.

4. **Rich media integration**: Integrate images, videos, or other rich media into the output.

5. **Accessibility improvements**: Enhance accessibility features for users with disabilities.

## Conclusion

The Output Formatting phase ensures that the extracted recipe is presented in a way that is useful and accessible to users. By supporting multiple formats and allowing users to customize the output, the system can accommodate a wide range of use cases and preferences. The careful design of the output formats ensures that the recipes are both human-readable and machine-parseable, enabling further processing and integration with other systems.
