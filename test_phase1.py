#!/usr/bin/env python3
"""
Phase 1 Test Script - Test basic functionality after structure cleanup
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils.config_loader import load_config
        print("✓ Config loader import successful")
    except ImportError as e:
        print(f"✗ Config loader import failed: {e}")
        return False
    
    try:
        from utils.logger import CookifyLogger
        print("✓ Logger import successful")
    except ImportError as e:
        print(f"✗ Logger import failed: {e}")
        return False
    
    try:
        from pipeline import Pipeline
        print("✓ Pipeline import successful")
    except ImportError as e:
        print(f"✗ Pipeline import failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from utils.config_loader import load_config
        config = load_config()
        
        # Check required sections
        required_sections = ["general", "preprocessing", "object_detection", "logging"]
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing config section: {section}")
                return False
        
        print("✓ Configuration loading successful")
        print(f"  - Log level: {config['general']['log_level']}")
        print(f"  - Output dir: {config['general']['output_dir']}")
        print(f"  - Frame rate: {config['preprocessing']['frame_rate']}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline initialization."""
    print("\nTesting pipeline initialization...")
    
    try:
        from pipeline import Pipeline
        pipeline = Pipeline()
        
        # Check that components are initialized
        if not hasattr(pipeline, 'video_processor'):
            print("✗ Video processor not initialized")
            return False
        
        if not hasattr(pipeline, 'object_detector'):
            print("✗ Object detector not initialized")
            return False
        
        if not hasattr(pipeline, 'scene_detector'):
            print("✗ Scene detector not initialized")
            return False
        
        print("✓ Pipeline initialization successful")
        print(f"  - Config loaded: {pipeline.config is not None}")
        print(f"  - Logger initialized: {pipeline.logger is not None}")
        
        return True
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        return False

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

def main():
    """Run all tests."""
    print("=" * 60)
    print("COOKIFY PHASE 1 STRUCTURE CLEANUP TEST")
    print("=" * 60)
    
    tests = [
        test_directory_structure,
        test_imports,
        test_config_loading,
        test_pipeline_initialization
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
        print("🎉 All tests passed! Phase 1 cleanup successful.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
