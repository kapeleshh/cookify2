"""
Recipe Extractor - Extracts structured recipe information from integrated data
"""

import os
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class RecipeExtractor:
    """
    Class for extracting structured recipe information from integrated data.
    """
    
    def __init__(self, infer_missing_ingredients=True, normalize_quantities=True, 
                group_similar_steps=False, min_confidence=0.6):
        """
        Initialize the RecipeExtractor.
        
        Args:
            infer_missing_ingredients (bool, optional): Whether to infer missing ingredients from steps. Defaults to True.
            normalize_quantities (bool, optional): Whether to normalize ingredient quantities. Defaults to True.
            group_similar_steps (bool, optional): Whether to group similar steps. Defaults to False.
            min_confidence (float, optional): Minimum confidence for recipe elements. Defaults to 0.6.
        """
        self.infer_missing_ingredients = infer_missing_ingredients
        self.normalize_quantities = normalize_quantities
        self.group_similar_steps = group_similar_steps
        self.min_confidence = min_confidence
    
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
    
    def _extract_title(self, integrated_data):
        """
        Extract recipe title from integrated data.
        
        Args:
            integrated_data (dict): Integrated data.
            
        Returns:
            str: Recipe title.
        """
        title = integrated_data.get("title", "")
        
        # If no title found, use a generic title
        if not title:
            title = "Untitled Recipe"
        
        return title
    
    def _extract_servings(self, integrated_data):
        """
        Extract servings information from integrated data.
        
        Args:
            integrated_data (dict): Integrated data.
            
        Returns:
            str: Servings information.
        """
        servings = integrated_data.get("servings", "")
        
        # If no servings found, use a default value
        if not servings:
            servings = "4"
        
        return servings
    
    def _extract_ingredients(self, integrated_data):
        """
        Extract ingredients from integrated data.
        
        Args:
            integrated_data (dict): Integrated data.
            
        Returns:
            list: Extracted ingredients.
        """
        ingredients = []
        
        # Get ingredients from integrated data
        integrated_ingredients = integrated_data.get("ingredients", [])
        
        for ingredient in integrated_ingredients:
            # Extract name, quantity, and unit
            name = ingredient.get("name", "")
            qty = ingredient.get("qty", "")
            unit = ingredient.get("unit", "")
            
            # Skip ingredients with no name
            if not name:
                continue
            
            # Clean up name
            name = self._clean_ingredient_name(name)
            
            # Add ingredient to list
            ingredients.append({
                "name": name,
                "qty": qty,
                "unit": unit
            })
        
        return ingredients
    
    def _clean_ingredient_name(self, name):
        """
        Clean up ingredient name.
        
        Args:
            name (str): Ingredient name.
            
        Returns:
            str: Cleaned ingredient name.
        """
        # Remove leading/trailing whitespace
        name = name.strip()
        
        # Remove quantity and unit patterns
        name = re.sub(r'^\d+\s+', '', name)
        name = re.sub(r'^\d+\.\d+\s+', '', name)
        name = re.sub(r'^\d+/\d+\s+', '', name)
        
        # Remove common units
        units = ["cup", "tablespoon", "teaspoon", "tbsp", "tsp", "oz", "ounce", "pound", "lb",
                "gram", "g", "kg", "ml", "l", "liter", "pinch", "dash", "handful"]
        
        for unit in units:
            name = re.sub(rf'^\s*{unit}s?\s+of\s+', '', name, flags=re.IGNORECASE)
            name = re.sub(rf'^\s*{unit}s?\s+', '', name, flags=re.IGNORECASE)
        
        # Remove "of" at the beginning
        name = re.sub(r'^\s*of\s+', '', name, flags=re.IGNORECASE)
        
        # Capitalize first letter
        if name:
            name = name[0].upper() + name[1:]
        
        return name
    
    def _extract_tools(self, integrated_data):
        """
        Extract tools from integrated data.
        
        Args:
            integrated_data (dict): Integrated data.
            
        Returns:
            list: Extracted tools.
        """
        tools = []
        
        # Get tools from integrated data
        integrated_tools = integrated_data.get("tools", [])
        
        for tool in integrated_tools:
            # Skip empty tools
            if not tool:
                continue
            
            # Clean up tool name
            tool = tool.strip()
            
            # Capitalize first letter
            if tool:
                tool = tool[0].upper() + tool[1:]
            
            # Add tool to list
            tools.append(tool)
        
        return tools
    
    def _extract_steps(self, integrated_data):
        """
        Extract steps from integrated data.
        
        Args:
            integrated_data (dict): Integrated data.
            
        Returns:
            list: Extracted steps.
        """
        steps = []
        
        # Get steps from integrated data
        integrated_steps = integrated_data.get("steps", [])
        
        for step in integrated_steps:
            # Extract step information
            idx = step.get("idx", 0)
            start = step.get("start", 0.0)
            end = step.get("end", 0.0)
            action = step.get("action", "")
            objects = step.get("objects", [])
            details = step.get("details", "")
            temp = step.get("temp", None)
            duration = step.get("duration", None)
            
            # Skip steps with no action
            if not action:
                continue
            
            # Clean up action
            action = action.strip()
            
            # Clean up details
            details = details.strip()
            
            # Add step to list
            steps.append({
                "idx": idx,
                "start": start,
                "end": end,
                "action": action,
                "objects": objects,
                "details": details,
                "temp": temp,
                "duration": duration
            })
        
        # Sort steps by index
        steps.sort(key=lambda x: x["idx"])
        
        # Renumber steps
        for i, step in enumerate(steps):
            step["idx"] = i + 1
        
        return steps
    
    def _infer_missing_ingredients(self, ingredients, steps):
        """
        Infer missing ingredients from steps.
        
        Args:
            ingredients (list): Extracted ingredients.
            steps (list): Extracted steps.
            
        Returns:
            list: Updated ingredients.
        """
        # This is a simplified implementation
        # In a real implementation, we would use more sophisticated techniques
        
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
    
    def _is_likely_ingredient(self, text):
        """
        Check if text is likely an ingredient.
        
        Args:
            text (str): Text to check.
            
        Returns:
            bool: True if likely an ingredient, False otherwise.
        """
        # Common ingredients
        common_ingredients = [
            "salt", "pepper", "sugar", "flour", "butter", "oil", "water", "milk",
            "egg", "garlic", "onion", "tomato", "potato", "carrot", "chicken",
            "beef", "pork", "fish", "rice", "pasta", "cheese", "cream", "yogurt",
            "vinegar", "lemon", "lime", "orange", "apple", "banana", "berry",
            "chocolate", "vanilla", "cinnamon", "oregano", "basil", "thyme",
            "rosemary", "parsley", "cilantro", "ginger", "soy sauce", "honey",
            "maple syrup", "mustard", "ketchup", "mayonnaise", "bread", "tortilla"
        ]
        
        # Check if text is a common ingredient
        if text in common_ingredients:
            return True
        
        return False
    
    def _normalize_quantities(self, ingredients):
        """
        Normalize ingredient quantities.
        
        Args:
            ingredients (list): Extracted ingredients.
            
        Returns:
            list: Updated ingredients.
        """
        # This is a simplified implementation
        # In a real implementation, we would use more sophisticated techniques
        
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
    
    def _group_similar_steps(self, steps):
        """
        Group similar steps.
        
        Args:
            steps (list): Extracted steps.
            
        Returns:
            list: Updated steps.
        """
        # This is a simplified implementation
        # In a real implementation, we would use more sophisticated techniques
        
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
