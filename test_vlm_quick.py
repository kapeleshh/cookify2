#!/usr/bin/env python
"""Quick test of VLM with simplified prompt"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.vlm_analysis.ollama_engine import OllamaVLMEngine

# Create test image if needed
test_image = "data/input/test_cooking.jpg"
if not Path(test_image).exists():
    print(f"No test image at {test_image}")
    print("Please provide a test cooking image")
    sys.exit(1)

# Initialize VLM
print("Initializing VLM...")
vlm = OllamaVLMEngine(
    model="llava:7b",
    host="http://localhost:11434",
    timeout=300,
    use_cache=False
)

# Simple test
print(f"Testing with: {test_image}")
print("Sending simple prompt...")

result = vlm.query(
    test_image,
    "What ingredients do you see in this cooking image? List them simply.",
    temperature=0.1
)

print("\n=== RESULT ===")
print(result.get('response', 'No response'))
print("\n=== SUCCESS ===")

