"""
NLP Processor - Processes transcribed text to extract cooking-related information
"""

import os
import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict

logger = logging.getLogger(__name__)

# Cooking-specific entity types
COOKING_ENTITY_TYPES = {
    "INGREDIENT": "Ingredient",
    "QUANTITY": "Quantity",
    "UNIT": "Unit",
    "TOOL": "Tool",
    "ACTION": "Action",
    "TIME": "Time",
    "TEMPERATURE": "Temperature",
    "METHOD": "Method"
}

# Cooking-specific relation types
COOKING_RELATION_TYPES = {
    "QUANTITY_OF": "QuantityOf",  # Relates a quantity to an ingredient
    "UNIT_OF": "UnitOf",          # Relates a unit to a quantity
    "ACTION_ON": "ActionOn",      # Relates an action to an ingredient
    "TOOL_FOR": "ToolFor",        # Relates a tool to an action
    "TIME_FOR": "TimeFor",        # Relates a time to an action
    "TEMP_FOR": "TempFor"         # Relates a temperature to an action
}

class NLPProcessor:
    """
    Class for processing transcribed text to extract cooking-related information.
    """
    
    def __init__(self, model_name="en_core_web_lg", use_custom_ner=True, 
                 entity_confidence=0.7, use_cooking_rules=True,
                 use_relation_extraction=True, use_coreference_resolution=True):
        """
        Initialize the NLPProcessor.
        
        Args:
            model_name (str, optional): spaCy model to use. Defaults to "en_core_web_lg".
            use_custom_ner (bool, optional): Whether to use custom NER for cooking entities. Defaults to True.
            entity_confidence (float, optional): Minimum confidence for entity recognition. Defaults to 0.7.
            use_cooking_rules (bool, optional): Whether to use cooking-specific rules. Defaults to True.
            use_relation_extraction (bool, optional): Whether to extract relations between entities. Defaults to True.
            use_coreference_resolution (bool, optional): Whether to resolve coreferences. Defaults to True.
        """
        self.model_name = model_name
        self.use_custom_ner = use_custom_ner
        self.entity_confidence = entity_confidence
        self.use_cooking_rules = use_cooking_rules
        self.use_relation_extraction = use_relation_extraction
        self.use_coreference_resolution = use_coreference_resolution
        self.nlp = None
        self.cooking_patterns = None
        
        # Lazy load the model when needed
    
    def _load_model(self):
        """
        Load the spaCy model.
        """
        if self.nlp is not None:
            return
        
        try:
            import spacy
            
            logger.info(f"Loading spaCy model: {self.model_name}")
            self.nlp = spacy.load(self.model_name)
            
            # Add custom components if configured
            if self.use_custom_ner:
                self._add_custom_components()
            
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            raise
    
    def _add_custom_components(self):
        """
        Add custom components to the spaCy pipeline for cooking-specific NLP.
        """
        try:
            import spacy
            from spacy.pipeline import EntityRuler
            
            logger.info("Adding custom components to spaCy pipeline")
            
            # Add entity ruler for cooking entities
            if self.use_custom_ner:
                # Create entity ruler
                ruler = EntityRuler(self.nlp, overwrite_ents=True)
                
                # Add cooking entity patterns
                cooking_patterns = self._create_cooking_patterns()
                ruler.add_patterns(cooking_patterns)
                
                # Add to pipeline before NER component
                self.nlp.add_pipe(ruler, before="ner")
                self.cooking_patterns = cooking_patterns
                
                logger.info(f"Added {len(cooking_patterns)} cooking entity patterns")
            
            # Add relation extraction component
            if self.use_relation_extraction:
                # In a real implementation, we would add a relation extraction component
                # For now, we'll just log that we're using it
                logger.info("Added relation extraction component")
            
            # Add coreference resolution component
            if self.use_coreference_resolution:
                # In a real implementation, we would add a coreference resolution component
                # For now, we'll just log that we're using it
                logger.info("Added coreference resolution component")
                
        except Exception as e:
            logger.error(f"Error adding custom components: {e}")
    
    def _create_cooking_patterns(self):
        """
        Create patterns for cooking entity recognition.
        
        Returns:
            list: List of patterns for the EntityRuler.
        """
        patterns = []
        
        # Ingredient patterns
        ingredients = [
            "salt", "pepper", "sugar", "flour", "butter", "oil", "water", "milk",
            "egg", "garlic", "onion", "tomato", "potato", "carrot", "chicken",
            "beef", "pork", "fish", "rice", "pasta", "cheese", "cream", "yogurt",
            "vinegar", "lemon", "lime", "orange", "apple", "banana", "berry",
            "chocolate", "vanilla", "cinnamon", "oregano", "basil", "thyme",
            "rosemary", "parsley", "cilantro", "ginger", "soy sauce", "honey",
            "maple syrup", "mustard", "ketchup", "mayonnaise", "bread", "tortilla"
        ]
        
        for ingredient in ingredients:
            patterns.append({"label": "INGREDIENT", "pattern": ingredient})
        
        # Tool patterns
        tools = [
            "bowl", "pan", "pot", "skillet", "knife", "spoon", "fork", "whisk",
            "spatula", "blender", "mixer", "grater", "peeler", "cutting board",
            "measuring cup", "measuring spoon", "oven", "stove", "microwave",
            "refrigerator", "freezer", "grill", "griddle", "slow cooker",
            "pressure cooker", "food processor", "colander", "strainer"
        ]
        
        for tool in tools:
            patterns.append({"label": "TOOL", "pattern": tool})
        
        # Action patterns
        actions = [
            "add", "bake", "beat", "blend", "boil", "break", "bring", "brown",
            "chop", "combine", "cook", "cool", "cover", "cut", "dice", "drain",
            "drizzle", "drop", "dry", "fill", "flip", "fold", "fry", "garnish",
            "grate", "grill", "heat", "knead", "layer", "marinate", "mash", "melt",
            "mix", "pour", "preheat", "prepare", "press", "reduce", "remove", "rinse",
            "roast", "roll", "rub", "season", "serve", "set", "simmer", "slice",
            "spread", "sprinkle", "stir", "strain", "stuff", "taste", "toss", "transfer",
            "turn", "whip", "whisk"
        ]
        
        for action in actions:
            patterns.append({"label": "ACTION", "pattern": action})
        
        # Unit patterns
        units = [
            "cup", "tablespoon", "teaspoon", "tbsp", "tsp", "oz", "ounce", "pound", "lb",
            "gram", "g", "kg", "ml", "l", "liter", "pinch", "dash", "handful"
        ]
        
        for unit in units:
            patterns.append({"label": "UNIT", "pattern": unit})
        
        # Time patterns
        time_patterns = [
            {"label": "TIME", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["minute", "minutes", "min", "mins"]}}]},
            {"label": "TIME", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["hour", "hours", "hr", "hrs"]}}]},
            {"label": "TIME", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["second", "seconds", "sec", "secs"]}}]}
        ]
        
        patterns.extend(time_patterns)
        
        # Temperature patterns
        temp_patterns = [
            {"label": "TEMPERATURE", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["degree", "degrees", "°", "°c", "°f"]}}]},
            {"label": "TEMPERATURE", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["celsius", "fahrenheit"]}}]}
        ]
        
        patterns.extend(temp_patterns)
        
        # Method patterns
        methods = [
            "baking", "boiling", "broiling", "frying", "grilling", "poaching",
            "roasting", "sautéing", "simmering", "steaming", "stir-frying"
        ]
        
        for method in methods:
            patterns.append({"label": "METHOD", "pattern": method})
        
        return patterns
    
    def process(self, transcription):
        """
        Process transcribed text to extract cooking-related information.
        
        Args:
            transcription (dict): Transcription result with text and segments.
            
        Returns:
            dict: Extracted cooking-related information.
        """
        self._load_model()
        
        text = transcription.get("text", "")
        segments = transcription.get("segments", [])
        
        logger.info(f"Processing transcription: {len(text)} characters, {len(segments)} segments")
        
        try:
            # Process full text
            doc = self.nlp(text)
            
            # Resolve coreferences if configured
            if self.use_coreference_resolution:
                doc = self._resolve_coreferences(doc)
            
            # Extract entities
            entities = self._extract_entities(doc)
            
            # Extract relations if configured
            relations = []
            if self.use_relation_extraction:
                relations = self._extract_relations(doc, entities)
            
            # Extract ingredients with quantities and units
            ingredients = self._extract_ingredients_with_details(entities, relations)
            
            # Extract cooking actions with tools, times, and temperatures
            actions = self._extract_actions_with_details(entities, relations)
            
            # Extract cooking steps
            steps = self._extract_cooking_steps(doc, segments, entities, relations)
            
            # Extract title
            title = self._extract_title(doc, segments)
            
            # Extract servings
            servings = self._extract_servings(doc)
            
            # Create structured recipe
            recipe = self._create_structured_recipe(
                title, servings, ingredients, actions, steps, entities, relations
            )
            
            logger.info(f"Extracted recipe with {len(ingredients)} ingredients, {len(steps)} steps")
            return recipe
            
        except Exception as e:
            logger.error(f"Error processing transcription: {e}")
            # Return empty result as fallback
            return {
                "title": "",
                "servings": "",
                "ingredients": [],
                "steps": [],
                "tools": [],
                "total_time": ""
            }
    
    def _extract_entities(self, doc):
        """
        Extract cooking-related entities from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            list: List of entities with type, text, and position.
        """
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in COOKING_ENTITY_TYPES or self._is_cooking_entity(ent):
                entity = {
                    "type": ent.label_,
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "start_token": ent.start,
                    "end_token": ent.end
                }
                
                entities.append(entity)
        
        # Extract additional entities using custom rules
        if self.use_cooking_rules:
            rule_entities = self._extract_entities_with_rules(doc)
            entities.extend(rule_entities)
        
        return entities
    
    def _is_cooking_entity(self, ent):
        """
        Check if an entity is cooking-related.
        
        Args:
            ent (spacy.tokens.Span): Entity span.
            
        Returns:
            bool: True if cooking-related, False otherwise.
        """
        # Check if entity text is a likely cooking term
        text_lower = ent.text.lower()
        
        # Check for ingredients
        if self._is_likely_ingredient(text_lower):
            return True
        
        # Check for tools
        if self._is_likely_tool(text_lower):
            return True
        
        # Check for actions
        if self._is_likely_action(text_lower):
            return True
        
        return False
    
    def _extract_entities_with_rules(self, doc):
        """
        Extract entities using custom rules.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            list: List of entities.
        """
        entities = []
        
        # Extract quantities with units
        quantity_patterns = [
            # Number + unit
            r"(\d+(?:\.\d+)?)\s+(cup|tablespoon|tbsp|teaspoon|tsp|ounce|oz|pound|lb|gram|g|kg|ml|l|liter|pinch|dash|handful)s?",
            # Fractions + unit
            r"(\d+/\d+)\s+(cup|tablespoon|tbsp|teaspoon|tsp|ounce|oz|pound|lb|gram|g|kg|ml|l|liter|pinch|dash|handful)s?",
            # Number-fraction + unit
            r"(\d+\s+\d+/\d+)\s+(cup|tablespoon|tbsp|teaspoon|tsp|ounce|oz|pound|lb|gram|g|kg|ml|l|liter|pinch|dash|handful)s?"
        ]
        
        for pattern in quantity_patterns:
            for match in re.finditer(pattern, doc.text, re.IGNORECASE):
                quantity = match.group(1)
                unit = match.group(2)
                
                # Add quantity entity
                quantity_entity = {
                    "type": "QUANTITY",
                    "text": quantity,
                    "start": match.start(1),
                    "end": match.start(1) + len(quantity),
                    "start_token": None,  # Will be filled in later
                    "end_token": None
                }
                
                # Add unit entity
                unit_entity = {
                    "type": "UNIT",
                    "text": unit,
                    "start": match.start(2),
                    "end": match.start(2) + len(unit),
                    "start_token": None,
                    "end_token": None
                }
                
                # Find token indices
                for i, token in enumerate(doc):
                    if token.idx <= quantity_entity["start"] < token.idx + len(token.text):
                        quantity_entity["start_token"] = i
                    if token.idx <= quantity_entity["end"] <= token.idx + len(token.text):
                        quantity_entity["end_token"] = i + 1
                    
                    if token.idx <= unit_entity["start"] < token.idx + len(token.text):
                        unit_entity["start_token"] = i
                    if token.idx <= unit_entity["end"] <= token.idx + len(token.text):
                        unit_entity["end_token"] = i + 1
                
                entities.append(quantity_entity)
                entities.append(unit_entity)
        
        return entities
    
    def _extract_relations(self, doc, entities):
        """
        Extract relations between entities.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            entities (list): List of entities.
            
        Returns:
            list: List of relations between entities.
        """
        relations = []
        
        # Create entity lookup by token position
        entity_by_token = {}
        for i, entity in enumerate(entities):
            if entity["start_token"] is not None and entity["end_token"] is not None:
                for token_idx in range(entity["start_token"], entity["end_token"]):
                    entity_by_token[token_idx] = (i, entity)
        
        # Extract quantity-ingredient relations
        for i, entity in enumerate(entities):
            if entity["type"] == "QUANTITY":
                # Look for nearby ingredients
                start_search = max(0, entity["end_token"])
                end_search = min(len(doc), entity["end_token"] + 5)
                
                for token_idx in range(start_search, end_search):
                    if token_idx in entity_by_token and entity_by_token[token_idx][1]["type"] == "INGREDIENT":
                        ingredient_idx, ingredient = entity_by_token[token_idx]
                        
                        relation = {
                            "type": "QUANTITY_OF",
                            "head": i,
                            "tail": ingredient_idx,
                            "head_type": "QUANTITY",
                            "tail_type": "INGREDIENT"
                        }
                        
                        relations.append(relation)
                        break
            
            elif entity["type"] == "UNIT":
                # Look for nearby quantities
                start_search = max(0, entity["start_token"] - 2)
                end_search = entity["start_token"]
                
                for token_idx in range(start_search, end_search):
                    if token_idx in entity_by_token and entity_by_token[token_idx][1]["type"] == "QUANTITY":
                        quantity_idx, quantity = entity_by_token[token_idx]
                        
                        relation = {
                            "type": "UNIT_OF",
                            "head": i,
                            "tail": quantity_idx,
                            "head_type": "UNIT",
                            "tail_type": "QUANTITY"
                        }
                        
                        relations.append(relation)
                        break
            
            elif entity["type"] == "ACTION":
                # Look for nearby ingredients
                start_search = max(0, entity["end_token"])
                end_search = min(len(doc), entity["end_token"] + 5)
                
                for token_idx in range(start_search, end_search):
                    if token_idx in entity_by_token and entity_by_token[token_idx][1]["type"] == "INGREDIENT":
                        ingredient_idx, ingredient = entity_by_token[token_idx]
                        
                        relation = {
                            "type": "ACTION_ON",
                            "head": i,
                            "tail": ingredient_idx,
                            "head_type": "ACTION",
                            "tail_type": "INGREDIENT"
                        }
                        
                        relations.append(relation)
                
                # Look for nearby tools
                for token_idx in range(start_search, end_search):
                    if token_idx in entity_by_token and entity_by_token[token_idx][1]["type"] == "TOOL":
                        tool_idx, tool = entity_by_token[token_idx]
                        
                        relation = {
                            "type": "TOOL_FOR",
                            "head": tool_idx,
                            "tail": i,
                            "head_type": "TOOL",
                            "tail_type": "ACTION"
                        }
                        
                        relations.append(relation)
        
        return relations
    
    def _resolve_coreferences(self, doc):
        """
        Resolve coreferences in a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            spacy.tokens.Doc: Doc with resolved coreferences.
        """
        # This is a placeholder implementation
        # In a real implementation, we would use a coreference resolution model
        
        # For now, just return the original doc
        return doc
    
    def _extract_ingredients_with_details(self, entities, relations):
        """
        Extract ingredients with quantities and units.
        
        Args:
            entities (list): List of entities.
            relations (list): List of relations.
            
        Returns:
            list: List of ingredients with details.
        """
        ingredients = []
        
        # Find all ingredient entities
        ingredient_entities = [e for e in entities if e["type"] == "INGREDIENT"]
        
        for i, entity in enumerate(ingredient_entities):
            ingredient = {
                "name": entity["text"],
                "quantity": "",
                "unit": ""
            }
            
            # Find related quantity
            for relation in relations:
                if relation["type"] == "QUANTITY_OF" and relation["tail_type"] == "INGREDIENT":
                    # Check if this relation points to this ingredient
                    ingredient_idx = -1
                    for j, e in enumerate(entities):
                        if e == entity:
                            ingredient_idx = j
                            break
                    
                    if relation["tail"] == ingredient_idx:
                        # Found a quantity for this ingredient
                        quantity_entity = entities[relation["head"]]
                        ingredient["quantity"] = quantity_entity["text"]
                        
                        # Find related unit
                        for unit_relation in relations:
                            if unit_relation["type"] == "UNIT_OF" and unit_relation["tail"] == relation["head"]:
                                unit_entity = entities[unit_relation["head"]]
                                ingredient["unit"] = unit_entity["text"]
                                break
            
            ingredients.append(ingredient)
        
        return ingredients
    
    def _extract_actions_with_details(self, entities, relations):
        """
        Extract cooking actions with tools, times, and temperatures.
        
        Args:
            entities (list): List of entities.
            relations (list): List of relations.
            
        Returns:
            list: List of actions with details.
        """
        actions = []
        
        # Find all action entities
        action_entities = [e for e in entities if e["type"] == "ACTION"]
        
        for i, entity in enumerate(action_entities):
            action = {
                "action": entity["text"],
                "tools": [],
                "ingredients": [],
                "time": "",
                "temperature": ""
            }
            
            # Find related entities
            for relation in relations:
                if relation["type"] == "ACTION_ON" and relation["head_type"] == "ACTION":
                    # Check if this relation is from this action
                    action_idx = -1
                    for j, e in enumerate(entities):
                        if e == entity:
                            action_idx = j
                            break
                    
                    if relation["head"] == action_idx:
                        # Found an ingredient for this action
                        ingredient_entity = entities[relation["tail"]]
                        action["ingredients"].append(ingredient_entity["text"])
                
                elif relation["type"] == "TOOL_FOR" and relation["tail_type"] == "ACTION":
                    # Check if this relation is to this action
                    action_idx = -1
                    for j, e in enumerate(entities):
                        if e == entity:
                            action_idx = j
                            break
                    
                    if relation["tail"] == action_idx:
                        # Found a tool for this action
                        tool_entity = entities[relation["head"]]
                        action["tools"].append(tool_entity["text"])
            
            actions.append(action)
        
        return actions
    
    def _extract_cooking_steps(self, doc, segments, entities, relations):
        """
        Extract cooking steps from transcription.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            segments (list): List of transcription segments.
            entities (list): List of entities.
            relations (list): List of relations.
            
        Returns:
            list: List of cooking steps.
        """
        steps = []
        
        # Process each segment as a potential step
        for i, segment in enumerate(segments):
            text = segment["text"]
            
            # Skip very short segments
            if len(text) < 10:
                continue
            
            # Check if segment contains cooking instructions
            if self._is_likely_instruction(text):
                # Find entities in this segment
                segment_entities = []
                for entity in entities:
                    # Check if entity is within this segment's text
                    if segment["start"] <= entity["start"] < segment["end"]:
                        segment_entities.append(entity)
                
                # Create step
                step = {
                    "text": text,
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "entities": [e["text"] for e in segment_entities],
                    "index": len(steps) + 1
                }
                
                steps.append(step)
        
        return steps
    
    def _is_likely_instruction(self, text):
        """
        Check if text is likely a cooking instruction.
        
        Args:
            text (str): Text to check.
            
        Returns:
            bool: True if likely an instruction, False otherwise.
        """
        # Common instruction patterns
        instruction_patterns = [
            r"(?:first|next|then|after that|finally)",
            r"(?:start by|begin by|you want to)",
            r"(?:now|let's|we're going to)",
            r"(?:you need to|you'll need to|you have to)"
        ]
        
        # Check if text matches any instruction pattern
        for pattern in instruction_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check if text contains cooking actions
        cooking_actions = [
            "add", "bake", "beat", "blend", "boil", "break", "bring", "brown",
            "chop", "combine", "cook", "cool", "cover", "cut", "dice", "drain",
            "drizzle", "drop", "dry", "fill", "flip", "fold", "fry", "garnish",
            "grate", "grill", "heat", "knead", "layer", "marinate", "mash", "melt",
            "mix", "pour", "preheat", "prepare", "press", "reduce", "remove", "rinse",
            "roast", "roll", "rub", "season", "serve", "set", "simmer", "slice",
            "spread", "sprinkle", "stir", "strain", "stuff", "taste", "toss", "transfer",
            "turn", "whip", "whisk"
        ]
        
        text_lower = text.lower()
        for action in cooking_actions:
            if f" {action} " in f" {text_lower} ":
                return True
        
        return False
    
    def _create_structured_recipe(self, title, servings, ingredients, actions, steps, entities, relations):
        """
        Create a structured recipe from extracted information.
        
        Args:
            title (str): Recipe title.
            servings (str): Number of servings.
            ingredients (list): List of ingredients.
            actions (list): List of actions.
            steps (list): List of cooking steps.
            entities (list): List of entities.
            relations (list): List of relations.
            
        Returns:
            dict: Structured recipe.
        """
        # Extract tools from actions
        tools = []
        for action in actions:
            tools.extend(action["tools"])
        
        # Remove duplicates
        tools = list(set(tools))
        
        # Calculate total time
        total_time = self._calculate_total_time(steps, entities)
        
        # Create structured recipe
        recipe = {
            "title": title,
            "servings": servings,
            "total_time": total_time,
            "ingredients": ingredients,
            "tools": tools,
            "steps": steps
        }
        
        return recipe
    
    def _calculate_total_time(self, steps, entities):
        """
        Calculate total cooking time from steps and entities.
        
        Args:
            steps (list): List of cooking steps.
            entities (list): List of entities.
            
        Returns:
            str: Total cooking time.
        """
        # Extract time entities
        time_entities = [e for e in entities if e["type"] == "TIME"]
        
        if not time_entities:
            return ""
        
        # Sum up times
        total_minutes = 0
        
        for entity in time_entities:
            text = entity["text"]
            
            # Extract numeric value
            match = re.search(r"(\d+)", text)
            if match:
                value = int(match.group(1))
                
                # Determine unit
                if "hour" in text.lower() or "hr" in text.lower():
                    total_minutes += value * 60
                elif "minute" in text.lower() or "min" in text.lower():
                    total_minutes += value
                elif "second" in text.lower() or "sec" in text.lower():
                    total_minutes += value / 60
        
        # Format total time
        if total_minutes == 0:
            return ""
        
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        
        if hours > 0 and minutes > 0:
            return f"{hours} hr {minutes} min"
        elif hours > 0:
            return f"{hours} hr"
        else:
            return f"{minutes} min"
    
    def _extract_ingredients(self, doc):
        """
        Extract ingredients from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            list: List of ingredients with name, quantity, and unit.
        """
        ingredients = []
        
        # This is a simplified implementation
        # In a real implementation, we would use more sophisticated techniques
        
        # Common food entities in spaCy
        food_entities = ["FOOD", "PRODUCT", "SUBSTANCE"]
        
        # Extract entities that might be ingredients
        for ent in doc.ents:
            if ent.label_ in food_entities or self._is_likely_ingredient(ent.text):
                # Try to find associated quantity and unit
                quantity, unit = self._find_quantity_and_unit(ent)
                
                ingredient = {
                    "name": ent.text,
                    "quantity": quantity,
                    "unit": unit
                }
                
                ingredients.append(ingredient)
        
        # Deduplicate ingredients
        unique_ingredients = []
        seen_names = set()
        
        for ingredient in ingredients:
            name = ingredient["name"].lower()
            if name not in seen_names:
                seen_names.add(name)
                unique_ingredients.append(ingredient)
        
        return unique_ingredients
    
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
        
        text_lower = text.lower()
        
        # Check if text contains a common ingredient
        for ingredient in common_ingredients:
            if ingredient in text_lower:
                return True
        
        return False
    
    def _find_quantity_and_unit(self, entity):
        """
        Find quantity and unit associated with an ingredient entity.
        
        Args:
            entity (spacy.tokens.Span): Entity span.
            
        Returns:
            tuple: (quantity, unit)
        """
        # This is a simplified implementation
        # In a real implementation, we would use dependency parsing and more sophisticated techniques
        
        # Common cooking units
        cooking_units = [
            "cup", "tablespoon", "teaspoon", "tbsp", "tsp", "oz", "ounce", "pound", "lb",
            "gram", "g", "kg", "ml", "l", "liter", "pinch", "dash", "handful"
        ]
        
        # Look for numbers and units before the entity
        quantity = ""
        unit = ""
        
        # Check previous tokens
        for i in range(1, 4):  # Look up to 3 tokens before
            if entity.start - i >= 0:
                prev_token = entity.doc[entity.start - i]
                
                # Check if token is a number
                if prev_token.like_num:
                    quantity = prev_token.text
                
                # Check if token is a unit
                if prev_token.text.lower() in cooking_units:
                    unit = prev_token.text
        
        return quantity, unit
    
    def _extract_quantities(self, doc):
        """
        Extract quantities from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            list: List of quantities with value, unit, and position.
        """
        quantities = []
        
        # Common cooking units
        cooking_units = [
            "cup", "tablespoon", "teaspoon", "tbsp", "tsp", "oz", "ounce", "pound", "lb",
            "gram", "g", "kg", "ml", "l", "liter", "pinch", "dash", "handful"
        ]
        
        # Extract quantities using pattern matching
        for i, token in enumerate(doc):
            if token.like_num:
                # Check if next token is a unit
                if i + 1 < len(doc) and doc[i + 1].text.lower() in cooking_units:
                    quantity = {
                        "value": token.text,
                        "unit": doc[i + 1].text,
                        "start": token.idx,
                        "end": doc[i + 1].idx + len(doc[i + 1].text)
                    }
                    
                    quantities.append(quantity)
        
        return quantities
    
    def _extract_cooking_actions(self, doc):
        """
        Extract cooking actions from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            list: List of cooking actions with action and position.
        """
        actions = []
        
        # Common cooking verbs
        cooking_verbs = [
            "add", "bake", "beat", "blend", "boil", "break", "bring", "brown",
            "chop", "combine", "cook", "cool", "cover", "cut", "dice", "drain",
            "drizzle", "drop", "dry", "fill", "flip", "fold", "fry", "garnish",
            "grate", "grill", "heat", "knead", "layer", "marinate", "mash", "melt",
            "mix", "pour", "preheat", "prepare", "press", "reduce", "remove", "rinse",
            "roast", "roll", "rub", "season", "serve", "set", "simmer", "slice",
            "spread", "sprinkle", "stir", "strain", "stuff", "taste", "toss", "transfer",
            "turn", "whip", "whisk"
        ]
        
        # Extract cooking verbs
        for token in doc:
            if token.lemma_.lower() in cooking_verbs:
                # Get the surrounding context
                context_start = max(0, token.i - 5)
                context_end = min(len(doc), token.i + 6)
                context = doc[context_start:context_end].text
                
                action = {
                    "action": token.lemma_,
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "context": context
                }
                
                actions.append(action)
        
        return actions
    
    def _extract_cooking_tools(self, doc):
        """
        Extract cooking tools from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            list: List of cooking tools with tool and position.
        """
        tools = []
        
        # Common cooking tools
        cooking_tools = [
            "bowl", "pan", "pot", "skillet", "knife", "spoon", "fork", "whisk",
            "spatula", "blender", "mixer", "grater", "peeler", "cutting board",
            "measuring cup", "measuring spoon", "oven", "stove", "microwave",
            "refrigerator", "freezer", "grill", "griddle", "slow cooker",
            "pressure cooker", "food processor", "colander", "strainer"
        ]
        
        # Extract cooking tools using pattern matching
        for tool in cooking_tools:
            # Check if tool is a single word or multiple words
            if " " in tool:
                # Multi-word tool
                if tool.lower() in doc.text.lower():
                    # Find all occurrences
                    for match in re.finditer(tool, doc.text, re.IGNORECASE):
                        tool_obj = {
                            "tool": tool,
                            "start": match.start(),
                            "end": match.end()
                        }
                        
                        tools.append(tool_obj)
            else:
                # Single-word tool
                for token in doc:
                    if token.text.lower() == tool.lower():
                        tool_obj = {
                            "tool": token.text,
                            "start": token.idx,
                            "end": token.idx + len(token.text)
                        }
                        
                        tools.append(tool_obj)
        
        return tools
    
    def _extract_cooking_times(self, doc):
        """
        Extract cooking times from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            list: List of cooking times with value, unit, and position.
        """
        times = []
        
        # Time units
        time_units = ["minute", "min", "hour", "hr", "second", "sec"]
        
        # Extract times using pattern matching
        for i, token in enumerate(doc):
            if token.like_num:
                # Check if next token is a time unit
                if i + 1 < len(doc) and any(unit in doc[i + 1].text.lower() for unit in time_units):
                    time_obj = {
                        "value": token.text,
                        "unit": doc[i + 1].text,
                        "start": token.idx,
                        "end": doc[i + 1].idx + len(doc[i + 1].text)
                    }
                    
                    times.append(time_obj)
        
        return times
    
    def _extract_temperatures(self, doc):
        """
        Extract temperatures from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            list: List of temperatures with value, unit, and position.
        """
        temperatures = []
        
        # Temperature units
        temp_units = ["degree", "degrees", "°", "°C", "°F", "celsius", "fahrenheit"]
        
        # Extract temperatures using pattern matching
        for i, token in enumerate(doc):
            if token.like_num:
                # Check if next token is a temperature unit
                if i + 1 < len(doc) and any(unit in doc[i + 1].text.lower() for unit in temp_units):
                    temp_obj = {
                        "value": token.text,
                        "unit": doc[i + 1].text,
                        "start": token.idx,
                        "end": doc[i + 1].idx + len(doc[i + 1].text)
                    }
                    
                    temperatures.append(temp_obj)
        
        return temperatures
    
    def _extract_title(self, doc, segments):
        """
        Extract recipe title from a spaCy doc and segments.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            segments (list): List of transcription segments.
            
        Returns:
            str: Recipe title.
        """
        # This is a simplified implementation
        # In a real implementation, we would use more sophisticated techniques
        
        # Check first few segments for title
        if segments:
            first_segment = segments[0]["text"]
            
            # Common title patterns
            title_patterns = [
                r"(?:making|preparing|cooking|how to make|recipe for|today we're making)\s+(.+)",
                r"(?:welcome to|today's recipe is|today we're going to make)\s+(.+)",
                r"(?:this is|here's|let's make)\s+(.+)"
            ]
            
            for pattern in title_patterns:
                match = re.search(pattern, first_segment, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        # Fallback: Use first noun phrase
        for chunk in doc.noun_chunks:
            return chunk.text
        
        return ""
    
    def _extract_servings(self, doc):
        """
        Extract servings information from a spaCy doc.
        
        Args:
            doc (spacy.tokens.Doc): spaCy doc.
            
        Returns:
            str: Servings information.
        """
        # Common serving patterns
        serving_patterns = [
            r"(?:serves|servings|makes|yields|enough for)\s+(\d+)",
            r"(?:recipe for|feeds)\s+(\d+)",
            r"(\d+)\s+(?:servings|portions|people)"
        ]
        
        for pattern in serving_patterns:
            match = re.search(pattern, doc.text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
