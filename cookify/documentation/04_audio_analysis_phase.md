# Audio Analysis Phase

The Audio Analysis phase is responsible for extracting information from the audio track of cooking videos. This phase involves transcribing speech to text and then processing the transcribed text to extract cooking-related information such as ingredients, quantities, cooking actions, and instructions.

## Purpose and Goals

The primary goals of the audio analysis phase are:

1. **Speech-to-Text Transcription**: Convert spoken instructions into text
2. **Ingredient Identification**: Extract mentions of ingredients from the transcription
3. **Quantity Extraction**: Identify measurements and quantities associated with ingredients
4. **Action Recognition**: Identify cooking actions and techniques mentioned in the audio
5. **Temporal Alignment**: Associate transcribed segments with specific timestamps in the video

## Components

### 1. AudioTranscriber

The `AudioTranscriber` class converts speech to text using OpenAI's Whisper model, a state-of-the-art speech recognition system.

#### Key Methods

- `transcribe(audio_path)`: Transcribes audio from a file
- `extract_cooking_instructions(transcription)`: Extracts cooking instructions from transcription
- `_process_segments(segments)`: Processes segments from Whisper result
- `_is_likely_instruction(sentence)`: Checks if a sentence is likely a cooking instruction

### 2. NLPProcessor

The `NLPProcessor` class processes transcribed text to extract cooking-related information using spaCy, a powerful natural language processing library.

#### Key Methods

- `process(transcription)`: Processes transcribed text to extract cooking-related information
- `_extract_ingredients(doc)`: Extracts ingredients from a spaCy doc
- `_extract_quantities(doc)`: Extracts quantities from a spaCy doc
- `_extract_cooking_actions(doc)`: Extracts cooking actions from a spaCy doc
- `_extract_cooking_tools(doc)`: Extracts cooking tools from a spaCy doc
- `_extract_cooking_times(doc)`: Extracts cooking times from a spaCy doc
- `_extract_temperatures(doc)`: Extracts temperatures from a spaCy doc
- `_extract_title(doc, segments)`: Extracts recipe title from a spaCy doc and segments
- `_extract_servings(doc)`: Extracts servings information from a spaCy doc

## Implementation Details

### Speech-to-Text Transcription with Whisper

Whisper is used for speech-to-text transcription due to its excellent performance across various accents and background noise conditions:

1. **Model selection**: The system supports different Whisper model sizes (tiny, base, small, medium, large) to balance accuracy and performance.

2. **Timestamp generation**: Whisper provides timestamps for each transcribed segment, which are crucial for aligning speech with video frames.

3. **Multi-language support**: Whisper can automatically detect and transcribe multiple languages, with an option to translate non-English speech to English.

#### Code Example:
```python
def transcribe(self, audio_path):
    """
    Transcribe audio from a file.
    
    Args:
        audio_path (str): Path to the audio file.
        
    Returns:
        dict: Transcription result with text and segments.
    """
    self._load_model()
    
    logger.info(f"Transcribing audio: {audio_path}")
    
    try:
        # Transcribe audio
        transcribe_options = {
            "language": self.language,
            "task": "translate" if self.translate else "transcribe",
            "verbose": False
        }
        
        result = self.model.transcribe(audio_path, **transcribe_options)
        
        # Process segments if timestamps are requested
        if self.timestamps:
            segments = self._process_segments(result["segments"])
        else:
            segments = []
        
        # Create structured result
        transcription = {
            "text": result["text"],
            "segments": segments,
            "language": result.get("language", "unknown")
        }
        
        logger.info(f"Transcription completed: {len(transcription['text'])} characters, {len(segments)} segments")
        return transcription
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        # Return empty transcription as fallback
        return {
            "text": "",
            "segments": [],
            "language": "unknown"
        }
```

### Natural Language Processing with spaCy

spaCy is used for natural language processing to extract cooking-related information from the transcription:

1. **Entity recognition**: spaCy's named entity recognition is used to identify ingredients, quantities, and other entities.

2. **Dependency parsing**: Dependency parsing helps connect quantities with their associated ingredients.

3. **Pattern matching**: Regular expressions and pattern matching are used to extract specific information like cooking times and temperatures.

#### Code Example:
```python
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
        
        # Extract ingredients
        ingredients = self._extract_ingredients(doc)
        
        # Extract quantities
        quantities = self._extract_quantities(doc)
        
        # Extract cooking actions
        actions = self._extract_cooking_actions(doc)
        
        # Extract cooking tools
        tools = self._extract_cooking_tools(doc)
        
        # Extract cooking times
        times = self._extract_cooking_times(doc)
        
        # Extract temperatures
        temperatures = self._extract_temperatures(doc)
        
        # Extract title
        title = self._extract_title(doc, segments)
        
        # Extract servings
        servings = self._extract_servings(doc)
        
        # Create structured result
        result = {
            "title": title,
            "servings": servings,
            "ingredients": ingredients,
            "quantities": quantities,
            "actions": actions,
            "tools": tools,
            "times": times,
            "temperatures": temperatures
        }
        
        logger.info(f"Extracted {len(ingredients)} ingredients, {len(actions)} actions, {len(tools)} tools")
        return result
        
    except Exception as e:
        logger.error(f"Error processing transcription: {e}")
        # Return empty result as fallback
        return {
            "title": "",
            "servings": "",
            "ingredients": [],
            "quantities": [],
            "actions": [],
            "tools": [],
            "times": [],
            "temperatures": []
        }
```

### Ingredient Extraction

Ingredient extraction identifies food items mentioned in the transcription:

1. **Entity recognition**: Food entities are identified using spaCy's entity recognition.

2. **Custom heuristics**: Additional heuristics are used to identify ingredients that may not be recognized as entities.

3. **Quantity association**: Quantities are associated with ingredients based on proximity and syntactic relationships.

#### Code Example:
```python
def _extract_ingredients(self, doc):
    """
    Extract ingredients from a spaCy doc.
    
    Args:
        doc (spacy.tokens.Doc): spaCy doc.
        
    Returns:
        list: List of ingredients with name, quantity, and unit.
    """
    ingredients = []
    
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
```

### Cooking Action Extraction

Cooking action extraction identifies verbs that represent cooking techniques:

1. **Verb identification**: Cooking verbs are identified based on a predefined list.

2. **Context extraction**: The surrounding context is extracted to provide more information about the action.

3. **Lemmatization**: Verbs are lemmatized to normalize different forms (e.g., "chopping" -> "chop").

#### Code Example:
```python
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
```

### Instruction Extraction

Instruction extraction identifies sentences that contain cooking instructions:

1. **Sentence segmentation**: The transcription is split into sentences.

2. **Imperative detection**: Sentences that start with verbs or contain instruction patterns are identified as instructions.

3. **Timestamp association**: Instructions are associated with timestamps from the transcription segments.

#### Code Example:
```python
def extract_cooking_instructions(self, transcription):
    """
    Extract cooking instructions from transcription.
    
    Args:
        transcription (dict): Transcription result.
        
    Returns:
        list: List of cooking instructions with timestamps.
    """
    instructions = []
    
    # Simple heuristic: Split by sentences and filter for imperative sentences
    import re
    
    # Get segments with text
    segments = transcription.get("segments", [])
    
    for segment in segments:
        text = segment["text"]
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip empty sentences
            if not sentence:
                continue
            
            # Check if sentence is likely an instruction
            if self._is_likely_instruction(sentence):
                instruction = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": sentence
                }
                
                instructions.append(instruction)
    
    return instructions
```

## Design Decisions and Rationale

### 1. Use of Whisper for Speech Recognition

**Decision**: Use OpenAI's Whisper model for speech-to-text transcription.

**Rationale**:
- **Accuracy**: Whisper provides state-of-the-art accuracy, especially for diverse accents and noisy environments common in cooking videos.
- **Timestamps**: Whisper provides accurate timestamps for each transcribed segment, which are crucial for aligning speech with video frames.
- **Multi-language support**: Whisper can handle multiple languages, making the system more versatile.

### 2. Use of spaCy for NLP

**Decision**: Use spaCy for natural language processing rather than other NLP libraries.

**Rationale**:
- **Performance**: spaCy is designed for production use and provides good performance.
- **Entity recognition**: spaCy's named entity recognition is well-suited for identifying ingredients and quantities.
- **Extensibility**: spaCy's pipeline architecture allows for easy extension with custom components.

### 3. Separate Transcription and NLP Processing

**Decision**: Separate the transcription and NLP processing into distinct components.

**Rationale**:
- **Modularity**: Each component has a clear responsibility, making the code more maintainable.
- **Reusability**: The transcription component can be used independently of the NLP processing.
- **Flexibility**: Different NLP processing approaches can be swapped in without affecting the transcription.

### 4. Rule-Based Extraction for Specific Entities

**Decision**: Use rule-based extraction for specific entities like cooking times and temperatures.

**Rationale**:
- **Precision**: Rule-based extraction can be more precise for well-defined patterns.
- **Simplicity**: For certain entities, rule-based extraction is simpler and more efficient than machine learning approaches.
- **Interpretability**: Rule-based systems are more interpretable and easier to debug.

## Performance Considerations

1. **Model size selection**: Whisper offers models of different sizes, allowing users to balance accuracy and performance based on their resources.

2. **Batch processing**: When processing multiple segments, batching can improve performance.

3. **Caching**: Transcription results can be cached to avoid redundant processing when re-analyzing the same video.

4. **Lazy loading**: Models are loaded only when needed to reduce memory usage when certain components are not used.

## Error Handling and Edge Cases

1. **Noisy audio**: Whisper is relatively robust to background noise, but extremely noisy audio may result in poor transcription. The system includes fallbacks for such cases.

2. **Accented speech**: Whisper handles various accents well, but very strong accents may affect transcription quality.

3. **Technical terminology**: Cooking videos often contain technical terminology that may not be recognized correctly. The system includes domain-specific knowledge to handle such cases.

4. **Overlapping speech**: When multiple people speak simultaneously, transcription quality may suffer. The system focuses on the dominant speaker.

## Future Improvements

1. **Custom fine-tuning**: Fine-tune Whisper on cooking-specific audio to improve transcription accuracy for cooking terminology.

2. **Speaker diarization**: Identify different speakers in the video to better understand the structure of the content.

3. **Custom NER model**: Train a custom named entity recognition model specifically for cooking entities.

4. **Contextual understanding**: Improve the system's ability to understand context and resolve ambiguities in the transcription.

## Conclusion

The Audio Analysis phase provides crucial information from the spoken content of cooking videos, complementing the visual information extracted in the Frame Analysis phase. The combination of state-of-the-art speech recognition with specialized NLP processing enables the system to extract detailed cooking instructions, ingredients, and other recipe components from the audio track.
