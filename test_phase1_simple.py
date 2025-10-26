#!/usr/bin/env python3
"""
Phase 1 Simple Test Script - Test basic functionality after structure cleanup
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

def test_logger_functionality():
    """Test logger functionality."""
    print("\nTesting logger functionality...")
    
    try:
        from utils.logger import CookifyLogger
        
        # Create a test logger
        logger = CookifyLogger("test_logger", "test_logs", "INFO")
        
        # Test basic logging
        logger.get_logger().info("Test log message")
        
        # Test performance timing
        logger.start_timer("test_operation")
        import time
        time.sleep(0.1)  # Simulate some work
        duration = logger.end_timer("test_operation")
        
        if duration > 0:
            print("✓ Logger functionality successful")
            print(f"  - Test operation duration: {duration:.3f}s")
            return True
        else:
            print("✗ Logger timing failed")
            return False
            
    except Exception as e:
        print(f"✗ Logger functionality failed: {e}")
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

def test_file_cleanup():
    """Test that duplicate files have been removed."""
    print("\nTesting file cleanup...")
    
    base_dir = Path(__file__).parent.parent  # Go up to cookify2 directory
    
    # Check that duplicate directories are gone
    duplicate_paths = [
        "src/pipeline.py",
        "src/utils/config_loader.py", 
        "src/utils/logger.py",
        "config.yaml",
        "tests/test_phase1.py",
        "tests/test_phase2.py"
    ]
    
    existing_duplicates = []
    for path in duplicate_paths:
        full_path = base_dir / path
        if full_path.exists():
            existing_duplicates.append(path)
    
    if existing_duplicates:
        print(f"✗ Duplicate files still exist: {existing_duplicates}")
        return False
    
    print("✓ File cleanup successful - no duplicate files found")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("COOKIFY PHASE 1 STRUCTURE CLEANUP TEST")
    print("=" * 60)
    
    tests = [
        test_directory_structure,
        test_file_cleanup,
        test_imports,
        test_config_loading,
        test_logger_functionality
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
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download models: python -m src.utils.model_downloader")
        print("3. Test with sample video: python main.py sample_video.mp4")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
