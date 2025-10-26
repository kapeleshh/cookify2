"""
Ollama VLM Engine - Core wrapper for Vision-Language Model using Ollama
"""
import logging
import base64
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from PIL import Image
import io

# Try to use ollama client, fall back to requests
try:
    import ollama
    USE_OLLAMA_CLIENT = True
except ImportError:
    import requests
    USE_OLLAMA_CLIENT = False
    logging.warning("ollama package not found, using requests")

logger = logging.getLogger(__name__)


class OllamaVLMEngine:
    """
    VLM Engine using Ollama for vision-language understanding.
    """
    
    def __init__(
        self,
        model: str = "qwen2-vl:7b",
        host: str = "http://localhost:11434",
        use_cache: bool = True,
        cache_path: str = "data/temp/ollama_vlm_cache.json",
        timeout: int = 120
    ):
        """
        Initialize Ollama VLM Engine.
        
        Args:
            model: Ollama model name (e.g., 'qwen2-vl:7b')
            host: Ollama server URL
            use_cache: Whether to cache responses
            cache_path: Path to cache file
            timeout: Request timeout in seconds
        """
        self.model = model
        self.host = host
        self.timeout = timeout
        self.use_cache = use_cache
        self.cache_path = Path(cache_path)
        self.response_cache = self._load_cache()
        
        logger.info(f"Initializing Ollama VLM Engine")
        logger.info(f"Model: {model}")
        logger.info(f"Host: {host}")
        logger.info(f"Using client: {'ollama' if USE_OLLAMA_CLIENT else 'requests'}")
        
        # Verify Ollama is running and model is available
        self._verify_setup()
    
    def _verify_setup(self):
        """Verify Ollama is running and model is available."""
        try:
            if USE_OLLAMA_CLIENT:
                # Check if model is available
                models = ollama.list()
                model_names = [m['name'] for m in models.get('models', [])]
                
                if self.model not in model_names:
                    logger.warning(f"Model {self.model} not found locally")
                    logger.info(f"Attempting to pull {self.model}...")
                    ollama.pull(self.model)
                    logger.info(f"✓ Model {self.model} pulled successfully")
                else:
                    logger.info(f"✓ Model {self.model} is available")
            else:
                # Use requests to check
                response = requests.get(f"{self.host}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    model_names = [m['name'] for m in models]
                    
                    if self.model not in model_names:
                        logger.warning(f"Model {self.model} not found")
                        logger.info(f"Please run: ollama pull {self.model}")
                    else:
                        logger.info(f"✓ Ollama is running with {self.model}")
                else:
                    raise ConnectionError("Cannot connect to Ollama")
                    
        except Exception as e:
            logger.error(f"Ollama verification failed: {e}")
            logger.error("Make sure Ollama is running: 'ollama serve'")
            raise
    
    def _load_cache(self) -> Dict:
        """Load response cache from disk."""
        if not self.use_cache:
            return {}
        
        try:
            if self.cache_path.exists():
                with open(self.cache_path, 'r') as f:
                    cache = json.load(f)
                logger.info(f"Loaded {len(cache)} cached responses")
                return cache
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
        
        return {}
    
    def _save_cache(self):
        """Save response cache to disk."""
        if not self.use_cache:
            return
        
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'w') as f:
                json.dump(self.response_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, image_path: str, prompt: str) -> str:
        """Generate cache key from image and prompt."""
        key_str = f"{image_path}:{prompt}:{self.model}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _encode_image(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            Base64 encoded image string
        """
        if isinstance(image, (str, Path)):
            # Read from file
            with open(image, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        else:
            # PIL Image
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def query(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        temperature: float = 0.1,
        use_cache: Optional[bool] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Query the VLM with an image and prompt using Ollama.
        
        Args:
            image: Image path or PIL Image
            prompt: Text prompt/question
            temperature: Sampling temperature (0.0 - 1.0)
            use_cache: Override instance cache setting
            stream: Whether to stream the response
            
        Returns:
            Dict with response text and metadata
        """
        # Handle caching
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        if use_cache:
            image_path = str(image) if not isinstance(image, Image.Image) else "pil_image"
            cache_key = self._get_cache_key(image_path, prompt)
            
            if cache_key in self.response_cache:
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return self.response_cache[cache_key]
        
        try:
            # Encode image
            image_base64 = self._encode_image(image)
            
            start_time = time.time()
            
            # Query using appropriate method
            if USE_OLLAMA_CLIENT:
                response_text = self._query_with_client(
                    image_base64, prompt, temperature, stream
                )
            else:
                response_text = self._query_with_requests(
                    image_base64, prompt, temperature, stream
                )
            
            inference_time = time.time() - start_time
            
            result = {
                "response": response_text,
                "prompt": prompt,
                "model": self.model,
                "temperature": temperature,
                "inference_time": inference_time
            }
            
            # Cache result
            if use_cache and not stream:
                image_path = str(image) if not isinstance(image, Image.Image) else "pil_image"
                cache_key = self._get_cache_key(image_path, prompt)
                self.response_cache[cache_key] = result
                self._save_cache()
            
            logger.info(f"VLM query completed in {inference_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"VLM query failed: {e}")
            return {
                "response": "",
                "error": str(e),
                "prompt": prompt
            }
    
    def _query_with_client(
        self,
        image_base64: str,
        prompt: str,
        temperature: float,
        stream: bool
    ) -> str:
        """Query using ollama Python client."""
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            images=[image_base64],
            options={
                'temperature': temperature,
            },
            stream=stream
        )
        
        if stream:
            # Handle streaming response
            full_response = ""
            for chunk in response:
                if 'response' in chunk:
                    full_response += chunk['response']
            return full_response
        else:
            return response['response']
    
    def _query_with_requests(
        self,
        image_base64: str,
        prompt: str,
        temperature: float,
        stream: bool
    ) -> str:
        """Query using requests library."""
        url = f"{self.host}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }
        
        response = requests.post(url, json=payload, timeout=self.timeout)
        
        if response.status_code == 200:
            if stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            full_response += chunk['response']
                return full_response
            else:
                return response.json()['response']
        else:
            raise Exception(f"Ollama request failed: {response.status_code} - {response.text}")
    
    def query_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        prompts: List[str],
        temperature: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Query VLM with multiple images and prompts.
        
        Args:
            images: List of image paths or PIL Images
            prompts: List of prompts (one per image)
            temperature: Sampling temperature
            
        Returns:
            List of response dicts
        """
        if len(images) != len(prompts):
            raise ValueError("Number of images must match number of prompts")
        
        results = []
        total = len(images)
        
        for idx, (image, prompt) in enumerate(zip(images, prompts), 1):
            logger.info(f"Processing {idx}/{total}...")
            result = self.query(image, prompt, temperature)
            results.append(result)
        
        return results
    
    def test_connection(self) -> bool:
        """
        Test connection to Ollama server.
        
        Returns:
            True if connection successful
        """
        try:
            if USE_OLLAMA_CLIENT:
                ollama.list()
                return True
            else:
                response = requests.get(f"{self.host}/api/tags", timeout=5)
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def clear_cache(self):
        """Clear response cache."""
        self.response_cache = {}
        if self.cache_path.exists():
            self.cache_path.unlink()
        logger.info("Cache cleared")
    
    def list_available_models(self) -> List[str]:
        """
        List available models in Ollama.
        
        Returns:
            List of model names
        """
        try:
            if USE_OLLAMA_CLIENT:
                models = ollama.list()
                return [m['name'] for m in models.get('models', [])]
            else:
                response = requests.get(f"{self.host}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    return [m['name'] for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        
        return []
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'use_cache') and self.use_cache:
            self._save_cache()

