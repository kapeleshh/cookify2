"""
VLM Prompts - Specialized prompts for cooking video analysis

This module contains optimized prompts for extracting cooking-related
information from video frames using Vision-Language Models.
"""
from typing import Dict, List, Optional


class CookingPrompts:
    """Collection of optimized prompts for cooking video analysis."""
    
    @staticmethod
    def identify_ingredients() -> str:
        """Prompt for identifying ingredients in a frame."""
        return """List all visible ingredients in this cooking image. For each ingredient, provide the name, quantity if visible, and state (raw/chopped/etc). Keep it simple and clear."""
    
    @staticmethod
    def identify_actions() -> str:
        """Prompt for identifying cooking actions."""
        return """What cooking action or technique is being performed in this frame?

Provide:
1. Primary action (e.g., chopping, stirring, sautéing, mixing)
2. Confidence level (high/medium/low)
3. Brief description of what's happening
4. Any relevant cooking techniques being demonstrated

Format as JSON:
{
  "action": "primary_action",
  "confidence": "high/medium/low",
  "description": "brief description",
  "technique": "specific technique if applicable"
}"""
    
    @staticmethod
    def identify_tools() -> str:
        """Prompt for identifying cooking tools."""
        return """List all cooking tools and equipment visible in this frame.

Include:
- Utensils (knives, spoons, spatulas, etc.)
- Cookware (pans, pots, bowls, etc.)
- Appliances (stove, oven, mixer, etc.)
- Measuring tools

Format as JSON list:
["tool1", "tool2", ...]

Only list items that are clearly visible."""
    
    @staticmethod
    def read_measurements() -> str:
        """Prompt for reading on-screen measurements and text."""
        return """Read and extract ALL text visible in this frame, particularly:

1. Measurement quantities (cups, tablespoons, grams, etc.)
2. Temperatures (degrees F/C)
3. Cooking times/durations
4. Recipe steps or instructions
5. Ingredient labels

Format as JSON:
{
  "measurements": [{"value": "amount", "unit": "unit", "item": "what it measures"}],
  "temperatures": [{"value": "temp", "unit": "F/C"}],
  "durations": [{"value": "time", "unit": "minutes/hours"}],
  "instructions": ["step1", "step2"],
  "other_text": ["any other visible text"]
}

If no text is visible, return empty lists."""
    
    @staticmethod
    def analyze_cooking_process() -> str:
        """Prompt for understanding the cooking process."""
        return """Analyze the cooking process shown in this frame.

Describe:
1. What stage of cooking is this? (prep, cooking, finishing, plating)
2. What is the cook trying to achieve?
3. What should happen next in the process?
4. Are there any visible temperature or time indicators?
5. What is the approximate state of doneness/completion?

Provide a detailed but concise analysis."""
    
    @staticmethod
    def temporal_analysis(previous_description: str) -> str:
        """Prompt for analyzing changes between frames."""
        return f"""Compare this frame to the previous state of the cooking process.

Previous state: {previous_description}

Describe:
1. What has changed?
2. What cooking transformation occurred?
3. How much time likely passed?
4. What was the cooking action performed?
5. Is the dish progressing correctly?

Focus on meaningful changes in the cooking process."""
    
    @staticmethod
    def validate_recipe_step(
        step_description: str,
        ingredients: List[str],
        tools: List[str]
    ) -> str:
        """Prompt for validating a recipe step."""
        return f"""Validate this recipe step based on the frame:

Proposed step: "{step_description}"
Claimed ingredients: {', '.join(ingredients)}
Claimed tools: {', '.join(tools)}

Questions:
1. Does the visual evidence support this step description?
2. Are the listed ingredients actually visible?
3. Are the listed tools present and being used correctly?
4. Is anything missing from the description?
5. Rate the accuracy of this step (0-100%)

Provide detailed validation."""
    
    @staticmethod
    def estimate_quantities() -> str:
        """Prompt for estimating ingredient quantities from visual cues."""
        return """Estimate the quantities of ingredients based on visual cues.

Look for:
1. Measuring cups/spoons in use
2. Size of bowls/containers and how full they are
3. Number of discrete items (eggs, vegetables, etc.)
4. Approximate volume based on container size
5. Visual portion sizes

For each visible ingredient, estimate:
{
  "ingredient": "name",
  "estimated_quantity": "amount",
  "unit": "cups/tbsp/pieces/etc",
  "confidence": "high/medium/low",
  "reasoning": "how you estimated this"
}

Be realistic and err on the side of common recipe proportions."""
    
    @staticmethod
    def identify_cuisine_style() -> str:
        """Prompt for identifying cuisine type and cooking style."""
        return """Based on all visible elements in this frame, identify:

1. Cuisine type (Italian, Chinese, French, etc.)
2. Cooking style (traditional, modern, fusion, etc.)
3. Dish category (appetizer, main course, dessert, etc.)
4. Confidence level for each identification

Provide evidence for your classifications based on:
- Ingredients used
- Cooking techniques
- Equipment and tools
- Visual presentation style
- Any cultural indicators

Format as JSON."""
    
    @staticmethod
    def generate_step_description(
        action: str,
        ingredients: List[str],
        duration: Optional[str] = None,
        temperature: Optional[str] = None
    ) -> str:
        """Prompt for generating natural language step description."""
        context = f"Action: {action}\nIngredients involved: {', '.join(ingredients)}"
        if duration:
            context += f"\nDuration: {duration}"
        if temperature:
            context += f"\nTemperature: {temperature}"
        
        return f"""{context}

Generate a clear, concise recipe step instruction that a home cook could follow.

Requirements:
- Start with an action verb
- Be specific but not overly verbose
- Include relevant details (temperature, time, technique)
- Use common cooking terminology
- Make it easy to understand

Provide just the step instruction, no extra explanation."""
    
    @staticmethod
    def comprehensive_frame_analysis() -> str:
        """Prompt for comprehensive frame understanding."""
        return """Provide a comprehensive analysis of this cooking video frame.

Include ALL of the following in your response:

1. INGREDIENTS:
   - List all visible ingredients
   - Estimate quantities where possible
   - Note preparation state (chopped, whole, etc.)

2. COOKING TOOLS & EQUIPMENT:
   - All visible utensils
   - Cookware being used
   - Appliances in frame

3. COOKING ACTION:
   - What is being done right now
   - Technique being used
   - Stage of cooking process

4. TEXT & MEASUREMENTS:
   - Any visible text, numbers, or labels
   - Measurements or quantities shown
   - Timers, temperatures, etc.

5. COOKING CONTEXT:
   - What dish is likely being made
   - Cuisine style
   - Current cooking stage

Format as structured JSON with these sections."""
    
    @staticmethod
    def quick_ingredient_scan() -> str:
        """Quick prompt for fast ingredient identification."""
        return """List all ingredients you can see in this image. 
        
Be brief and specific. Just provide a simple list of ingredient names."""
    
    @staticmethod
    def quick_action_scan() -> str:
        """Quick prompt for fast action identification."""
        return """In one sentence, what cooking action is being performed in this image?"""
    
    @staticmethod
    def custom_query(question: str, context: Optional[str] = None) -> str:
        """Create a custom prompt with optional context."""
        prompt = f"{question}\n"
        if context:
            prompt += f"\nContext: {context}\n"
        prompt += "\nProvide a detailed, accurate response based on what you see in the image."
        return prompt


# Convenience function for quick access
def get_prompt(prompt_type: str, **kwargs) -> str:
    """
    Get a prompt by type.
    
    Args:
        prompt_type: Type of prompt (ingredients, actions, tools, etc.)
        **kwargs: Additional arguments for the prompt
        
    Returns:
        Formatted prompt string
    """
    prompts = CookingPrompts()
    
    prompt_map = {
        'ingredients': prompts.identify_ingredients,
        'actions': prompts.identify_actions,
        'tools': prompts.identify_tools,
        'measurements': prompts.read_measurements,
        'process': prompts.analyze_cooking_process,
        'quantities': prompts.estimate_quantities,
        'cuisine': prompts.identify_cuisine_style,
        'comprehensive': prompts.comprehensive_frame_analysis,
        'quick_ingredients': prompts.quick_ingredient_scan,
        'quick_action': prompts.quick_action_scan,
    }
    
    if prompt_type in prompt_map:
        prompt_func = prompt_map[prompt_type]
        if kwargs:
            return prompt_func(**kwargs)
        return prompt_func()
    
    raise ValueError(f"Unknown prompt type: {prompt_type}")

