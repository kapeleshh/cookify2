"""
Test Ollama Vision capabilities for Cookify
Simple script to verify Ollama can process images
"""
import sys
import requests
import json
import base64
from pathlib import Path


def encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def test_ollama_connection() -> bool:
    """Test if Ollama is running."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✓ Connected to Ollama successfully")
            
            models = response.json().get('models', [])
            print(f"\n📦 Available models:")
            for model in models:
                print(f"   - {model['name']}")
            
            return True
        else:
            print("✗ Ollama returned unexpected status")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama")
        print("\nMake sure Ollama is running:")
        print("  $ ollama serve")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_ollama_vision(image_path: str, prompt: str, model: str = "qwen2-vl:7b"):
    """Test Ollama with vision."""
    
    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        print("\nPlease provide a test cooking image at this location.")
        print("You can use any cooking image for testing.")
        return False
    
    try:
        # Encode image
        print(f"\n📸 Image: {image_path}")
        print(f"🤖 Model: {model}")
        print(f"❓ Prompt: {prompt}")
        print("\n⏳ Querying Ollama VLM...")
        
        image_base64 = encode_image(image_path)
        
        # Prepare request
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }
        
        # Send request
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n" + "=" * 70)
            print("✓ Response:")
            print("=" * 70)
            print(result['response'])
            print("=" * 70)
            
            return True
        else:
            print(f"\n✗ Error: {response.status_code}")
            print(response.text)
            
            if response.status_code == 404:
                print(f"\nModel '{model}' not found.")
                print(f"Please download it first:")
                print(f"  $ ollama pull {model}")
            
            return False
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def create_sample_image_if_needed():
    """Create a sample placeholder if no test image exists."""
    test_path = Path("data/input/test_cooking.jpg")
    
    if not test_path.exists():
        print(f"\n⚠️  No test image found at: {test_path}")
        print("\nPlease provide a cooking image for testing.")
        print("You can:")
        print("  1. Place any cooking image at: data/input/test_cooking.jpg")
        print("  2. Or specify a different path when running the test")
        
        # Create directory if it doesn't exist
        test_path.parent.mkdir(parents=True, exist_ok=True)
        
        return False
    
    return True


def main():
    """Main test function."""
    print("\n" + "=" * 70)
    print("Ollama Vision Test for Cookify")
    print("=" * 70)
    
    # Test 1: Check Ollama connection
    print("\n1. Testing Ollama Connection...")
    if not test_ollama_connection():
        return 1
    
    # Test 2: Check for test image
    print("\n2. Checking for test image...")
    test_image = "data/input/test_cooking.jpg"
    
    if not create_sample_image_if_needed():
        # Try alternative path
        alt_paths = [
            "data/input/sample.jpg",
            "examples/sample_cooking.jpg",
            "test_image.jpg"
        ]
        
        for alt_path in alt_paths:
            if Path(alt_path).exists():
                test_image = alt_path
                print(f"✓ Found test image at: {test_image}")
                break
        else:
            print("\n✗ No test image available")
            print("\nSkipping vision test.")
            print("Once you have a test image, run:")
            print("  $ python test_ollama_vision.py")
            return 0  # Not a failure, just skipped
    else:
        print(f"✓ Test image available: {test_image}")
    
    # Test 3: Test vision with simple prompt
    print("\n3. Testing Vision Capabilities...")
    
    prompts = [
        "What do you see in this image? Describe it briefly.",
        "What ingredients and cooking tools are visible in this image?"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n   Test {i}/{len(prompts)}:")
        success = test_ollama_vision(test_image, prompt)
        
        if not success:
            return 1
    
    # Success
    print("\n" + "=" * 70)
    print("✓ All Tests Passed!")
    print("=" * 70)
    print("\nOllama VLM is working correctly for Cookify.")
    print("\nNext steps:")
    print("  1. Run full integration tests: python test_ollama_integration.py")
    print("  2. Process a cooking video: python main.py video.mp4")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

