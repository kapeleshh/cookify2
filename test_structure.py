#!/usr/bin/env python3
"""
Simple structure test - Test directory structure without requiring dependencies
"""

import os
import sys
from pathlib import Path

def test_directory_structure():
    """Test that directory structure is correct."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "src",
        "src/preprocessing",
        "src/frame_analysis", 
        "src/audio_analysis",
        "src/integration",
        "src/recipe_extraction",
        "src/output_formatting",
        "src/ui",
        "src/utils",
        "tests",
        "data",
        "documentation"
    ]
    
    base_dir = Path(__file__).parent
    missing_dirs = []
    
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"✗ Missing directories: {missing_dirs}")
        return False
    
    print("✓ Directory structure is correct")
    return True

def test_basic_imports():
    """Test basic imports without requiring all dependencies."""
    print("\nTesting basic imports...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.utils.config_loader import load_config
        print("✓ Config loader import successful")
    except ImportError as e:
        print(f"✗ Config loader import failed: {e}")
        return False
    
    try:
        from src.preprocessing.video_processor import VideoProcessor
        print("✓ Video processor import successful")
    except ImportError as e:
        print(f"✗ Video processor import failed: {e}")
        return False
    
    return True

def main():
    """Run structure tests."""
    print("=" * 60)
    print("COOKIFY STRUCTURE TEST")
    print("=" * 60)
    
    tests = [
        test_directory_structure,
        test_basic_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed! Flattened directory structure is working.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
