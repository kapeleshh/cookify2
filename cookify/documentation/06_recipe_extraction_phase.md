# Recipe Extraction Phase

The Recipe Extraction phase is responsible for transforming the integrated multimodal data into a structured recipe format. This phase takes the aligned and integrated information from the Multimodal Integration phase and extracts a coherent recipe with title, servings, ingredients, tools, and steps.

## Purpose and Goals

The primary goals of the recipe extraction phase are:

1. **Structure Creation**: Transform unstructured data into a structured recipe format
2. **Information Consolidation**: Consolidate information from different sources into a coherent whole
3. **Missing Information Inference**: Infer missing information based on context and domain knowledge
4. **Normalization**: Standardize quantities, units, and terminology
5. **Quality Assurance**: Ensure the extracted recipe is complete, consistent, and usable

## Components

### RecipeExtractor

The `RecipeExtractor` class is responsible for extracting structured recipe information from integrated data.

#### Key Methods

- `extract(integrated_data)`: Main extraction method that transforms integrated data into a structured recipe
- `_extract_title(integrated_data)`: Extracts the recipe title
- `_extract_servings(integrated_data)`: Extracts servings information
- `_extract_ingredients(integrated_data)`: Extracts ingredients with quantities and units
- `_extract_tools(integrated_data)`: Extracts cooking tools
- `_extract_steps(integrated_data)`: Extracts cooking steps with actions, objects, and details
- `_infer_missing_ingredients(ingredients, steps)`: Infers missing ingredients from steps
- `_normalize_quantities(ingredients)`: Normalizes ingredient quantities and units
- `_group_similar_steps(steps)`: Groups similar steps for clarity

## Implementation Details

### Recipe Structure Extraction

The recipe extraction process transforms the integrated data into a structured recipe format:

1. **Title extraction**: The recipe title is extracted from the integrated data, typically from the beginning of the video.

2. **Servings extraction**: Servings information is extracted from explicit mentions or inferred from the quantities.

3. **Ingredient extraction**: Ingredients are extracted with their quantities and units, consolidating information from visual and audio sources.

4. **Tool extraction**: Cooking tools are extracted from visual detections and audio mentions.

5. **Step extraction**: Cooking steps are extracted from the timeline, combining actions, objects, and instructions.

#### Code Example:
```python
def extract(self, integrated_data):
    """
    Extract structured recipe information from integrated data.
    
    Args:
        integrated_data (dict): Integrated data from multimodal integration.
        
    Returns:
        dict: Structured recipe information.
    """
    logger.info("Extracting structured recipe information...")
    
    # Extract title
    title = self._extract_title(integrated_data)
    
    # Extract servings
    servings = self._extract_servings(integrated_data)
    
    # Extract ingredients
    ingredients = self._extract_ingredients(integrated_data)
    
    # Extract tools
    tools = self._extract_tools(integrated_data)
    
    # Extract steps
    steps = self._extract_steps(integrated_data)
    
    # Infer missing ingredients if configured
    if self.infer_missing_ingredients:
        ingredients = self._infer_missing_ingredients(ingredients, steps)
    
    # Normalize quantities if configured
    if self.normalize_quantities:
        ingredients = self._normalize_quantities(ingredients)
    
    # Group similar steps if configured
    if self.group_similar_steps:
        steps = self._group_similar_steps(steps)
    
    # Create structured recipe
    recipe = {
        "title": title,
        "servings": servings,
        "ingredients": ingredients,
        "tools": tools,
        "steps": steps
    }
    
    logger.info(f"Recipe extraction complete: {len(ingredients)} ingredients, {len(tools)} tools, {len(steps)} steps")
    return recipe
```

### Ingredient Extraction and Normalization

Ingredient extraction consolidates information from different sources and normalizes quantities and units:

1. **Consolidation**: Ingredients are extracted from NLP results and visual detections, with duplicates removed.

2. **Quantity normalization**: Quantities are normalized to standard formats (e.g., converting fractions to decimals).

3. **Unit normalization**: Units are normalized to standard abbreviations (e.g., "tablespoon" to "tbsp").

#### Code Example:
```python
def _normalize_quantities(self, ingredients):
    """
    Normalize ingredient quantities.
    
    Args:
        ingredients (list): Extracted ingredients.
        
    Returns:
            list: Updated ingredients.
    """
    for ingredient in ingredients:
        qty = ingredient["qty"]
        unit = ingredient["unit"]
        
        # Skip ingredients with no quantity or unit
        if not qty or not unit:
            continue
        
        # Convert fractions to decimals
        if "/" in qty:
            try:
                numerator, denominator = qty.split("/")
                qty = str(float(numerator) / float(denominator))
                ingredient["qty"] = qty
            except:
                pass
        
        # Normalize units
        unit_lower = unit.lower()
        
        # Tablespoon
        if unit_lower in ["tablespoon", "tablespoons", "tbsp", "tbsps", "tbs", "tb"]:
            ingredient["unit"] = "tbsp"
        
        # Teaspoon
        elif unit_lower in ["teaspoon", "teaspoons", "tsp", "tsps", "ts", "t"]:
            ingredient["unit"] = "tsp"
        
        # Cup
        elif unit_lower in ["cup", "cups", "c"]:
            ingredient["unit"] = "cup"
        
        # Ounce
        elif unit_lower in ["ounce", "ounces", "oz"]:
            ingredient["unit"] = "oz"
        
        # Pound
        elif unit_lower in ["pound", "pounds", "lb", "lbs"]:
            ingredient["unit"] = "lb"
        
        # Gram
        elif unit_lower in ["gram", "grams", "g"]:
            ingredient["unit"] = "g"
        
        # Kilogram
        elif unit_lower in ["kilogram", "kilograms", "kg", "kgs"]:
            ingredient["unit"] = "kg"
        
        # Milliliter
        elif unit_lower in ["milliliter", "milliliters", "ml"]:
            ingredient["unit"] = "ml"
        
        # Liter
        elif unit_lower in ["liter", "liters", "l"]:
            ingredient["unit"] = "l"
    
    return ingredients
```

### Step Extraction and Grouping

Step extraction creates a sequence of cooking steps from the timeline and can group similar steps for clarity:

1. **Action-based steps**: Each cooking action becomes a step, with associated objects and details.

2. **Temporal ordering**: Steps are ordered based on their timestamps in the video.

3. **Step grouping**: Similar consecutive steps can be grouped to reduce redundancy.

#### Code Example:
```python
def _group_similar_steps(self, steps):
    """
    Group similar steps.
    
    Args:
        steps (list): Extracted steps.
        
    Returns:
            list: Updated steps.
    """
    # Group steps with the same action
    grouped_steps = []
    current_group = None
    
    for step in steps:
        if current_group is None:
            # Start a new group
            current_group = step
        elif step["action"] == current_group["action"]:
            # Merge with current group
            current_group["end"] = step["end"]
            current_group["objects"].extend(step["objects"])
            current_group["details"] += " " + step["details"]
            
            # Use the latest temperature and duration
            if step["temp"] is not None:
                current_group["temp"] = step["temp"]
            
            if step["duration"] is not None:
                current_group["duration"] = step["duration"]
        else:
            # Add current group to grouped steps
            grouped_steps.append(current_group)
            
            # Start a new group
            current_group = step
    
    # Add the last group
    if current_group is not None:
        grouped_steps.append(current_group)
    
    # Deduplicate objects in each step
    for step in grouped_steps:
        step["objects"] = list(set(step["objects"]))
    
    # Renumber steps
    for i, step in enumerate(grouped_steps):
        step["idx"] = i + 1
    
    return grouped_steps
```

### Missing Information Inference

The system can infer missing information based on context and domain knowledge:

1. **Ingredient inference**: Missing ingredients can be inferred from cooking steps.

2. **Quantity inference**: Missing quantities can be inferred from typical usage patterns.

3. **Step details inference**: Missing details can be inferred from similar recipes or cooking knowledge.

#### Code Example:
```python
def _infer_missing_ingredients(self, ingredients, steps):
    """
    Infer missing ingredients from steps.
    
    Args:
        ingredients (list): Extracted ingredients.
        steps (list): Extracted steps.
        
    Returns:
            list: Updated ingredients.
    """
    # Get all ingredient names
    ingredient_names = [ingredient["name"].lower() for ingredient in ingredients]
    
    # Look for potential ingredients in steps
    for step in steps:
        # Check objects
        for obj in step["objects"]:
            obj_lower = obj.lower()
            
            # Check if object is a potential ingredient
            if obj_lower not in ingredient_names and self._is_likely_ingredient(obj_lower):
                # Add as a new ingredient
                ingredients.append({
                    "name": obj,
                    "qty": "",
                    "unit": ""
                })
                
                # Add to ingredient names
                ingredient_names.append(obj_lower)
        
        # Check details
        if step["details"]:
            # Look for potential ingredients in details
            words = re.findall(r'\b\w+\b', step["details"].lower())
            
            for word in words:
                # Check if word is a potential ingredient
                if word not in ingredient_names and self._is_likely_ingredient(word):
                    # Add as a new ingredient
                    ingredients.append({
                        "name": word.capitalize(),
                        "qty": "",
                        "unit": ""
                    })
                    
                    # Add to ingredient names
                    ingredient_names.append(word)
    
    return ingredients
```

## Design Decisions and Rationale

### 1. Structured Recipe Format

**Decision**: Use a structured JSON format for the recipe with specific fields for title, servings, ingredients, tools, and steps.

**Rationale**:
- **Standardization**: A structured format ensures consistency across different recipes.
- **Machine readability**: The structured format is easy to process programmatically.
- **Flexibility**: The format can accommodate various recipe styles and complexities.

### 2. Ingredient Normalization

**Decision**: Normalize ingredient quantities and units to standard formats.

**Rationale**:
- **Consistency**: Normalized quantities and units are more consistent and easier to understand.
- **Scaling**: Normalized quantities make it easier to scale recipes up or down.
- **Comparison**: Normalized units make it easier to compare ingredients across recipes.

### 3. Step Grouping

**Decision**: Provide an option to group similar consecutive steps.

**Rationale**:
- **Clarity**: Grouped steps can be clearer and more concise.
- **Redundancy reduction**: Grouping reduces redundancy in the recipe.
- **Flexibility**: Making this optional allows users to choose the level of detail they prefer.

### 4. Missing Information Inference

**Decision**: Infer missing information based on context and domain knowledge.

**Rationale**:
- **Completeness**: Inference helps create more complete recipes.
- **Robustness**: The system can handle incomplete or noisy input data.
- **User experience**: Complete recipes are more useful to users.

## Challenges and Solutions

### 1. Incomplete Information

**Challenge**: The integrated data may not contain all the information needed for a complete recipe.

**Solution**:
- **Inference mechanisms**: Use context and domain knowledge to infer missing information.
- **Confidence scoring**: Assign confidence scores to inferred information to indicate uncertainty.
- **Default values**: Use reasonable defaults for missing information when inference is not possible.

### 2. Inconsistent Information

**Challenge**: The integrated data may contain inconsistent information from different sources.

**Solution**:
- **Conflict resolution**: Implement strategies to resolve conflicts, such as preferring higher-confidence sources.
- **Consistency checking**: Check for inconsistencies in the extracted recipe and resolve them.
- **User feedback**: Allow users to correct inconsistencies in the extracted recipe.

### 3. Ambiguous Terminology

**Challenge**: Cooking terminology can be ambiguous and vary across cultures and contexts.

**Solution**:
- **Terminology normalization**: Normalize terminology to standard forms.
- **Context-aware interpretation**: Use context to disambiguate terms.
- **Cultural adaptation**: Consider cultural variations in cooking terminology.

### 4. Complex Recipes

**Challenge**: Some recipes are complex, with multiple parallel processes, optional steps, or variations.

**Solution**:
- **Hierarchical representation**: Use a hierarchical representation to capture complex structures.
- **Optional elements**: Mark elements as optional when appropriate.
- **Variations**: Support recipe variations for different preferences or constraints.

## Performance Considerations

1. **Memory usage**: The extraction process should be memory-efficient, especially for complex recipes.

2. **Computational complexity**: The inference and normalization algorithms should be efficient.

3. **Scalability**: The system should scale well to handle recipes of varying complexity.

## Future Improvements

1. **Advanced inference**: Use more sophisticated inference techniques, such as machine learning models trained on recipe data.

2. **Personalization**: Adapt the extraction process to user preferences and dietary restrictions.

3. **Recipe validation**: Implement more comprehensive validation to ensure the extracted recipe is complete and consistent.

4. **Semantic understanding**: Improve the system's understanding of cooking semantics to better interpret ambiguous instructions.

5. **Cross-recipe learning**: Learn from multiple recipes to improve extraction for new recipes.

## Conclusion

The Recipe Extraction phase transforms the integrated multimodal data into a structured recipe that is useful for users. By consolidating information, normalizing quantities and units, and inferring missing information, the system creates recipes that are complete, consistent, and easy to follow. The structured format ensures that the recipes can be easily stored, searched, and used in various applications.
