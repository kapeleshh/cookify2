"""
Demo Web UI for Cookify VLM - Testing VLM Integration
"""

import os
import sys
import logging
import json
import base64
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_from_directory
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# Initialize VLM components
vlm_engine = None
vlm_analyzer = None
config = None

def init_vlm():
    """Initialize VLM components."""
    global vlm_engine, vlm_analyzer, config
    
    if vlm_engine is None:
        try:
            logger.info("Initializing VLM components")
            config = load_config()
            
            vlm_config = config.get('vlm', {})
            if not vlm_config.get('enabled', False):
                logger.warning("VLM is disabled in config")
                return False
            
            from src.vlm_analysis.ollama_engine import OllamaVLMEngine
            from src.vlm_analysis.ollama_frame_analyzer import OllamaFrameAnalyzer
            
            vlm_engine = OllamaVLMEngine(
                model=vlm_config.get('model', 'qwen2-vl:7b'),
                host=vlm_config.get('host', 'http://localhost:11434'),
                use_cache=vlm_config.get('use_cache', True),
                cache_path=vlm_config.get('cache_path', 'data/temp/ollama_vlm_cache.json'),
                timeout=vlm_config.get('timeout', 120)
            )
            
            vlm_analyzer = OllamaFrameAnalyzer(vlm_engine, config)
            
            logger.info("VLM components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize VLM: {e}")
            return False
    
    return vlm_engine is not None

@app.route('/')
def index():
    """Render the demo index page."""
    return render_template('demo.html')

@app.route('/api/vlm/status')
def vlm_status():
    """Check VLM status and return as JSON."""
    try:
        if not init_vlm():
            config_vlm = load_config().get('vlm', {}) if config is None else config.get('vlm', {})
            return jsonify({
                'enabled': config_vlm.get('enabled', False),
                'connected': False,
                'message': 'VLM not initialized or disabled',
                'status': 'disabled'
            })
        
        is_connected = vlm_engine.test_connection()
        models = vlm_engine.list_available_models() if is_connected else []
        model_name = config.get('vlm', {}).get('model', 'unknown')
        
        return jsonify({
            'enabled': True,
            'connected': is_connected,
            'model': model_name,
            'available_models': models,
            'status': 'active' if is_connected else 'disconnected'
        })
    except Exception as e:
        logger.error(f"VLM status check failed: {e}")
        return jsonify({
            'enabled': False,
            'connected': False,
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze an uploaded image with VLM."""
    if not init_vlm():
        return jsonify({'error': 'VLM not available'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    analysis_types = request.form.get('analysis_types', 'ingredients,actions,tools').split(',')
    
    try:
        # Save uploaded image temporarily
        image_path = UPLOAD_FOLDER / file.filename
        file.save(str(image_path))
        
        # Analyze with VLM
        logger.info(f"Analyzing image with VLM: {analysis_types}")
        result = vlm_analyzer.analyze_frame(str(image_path), analysis_types)
        
        # Encode image for display
        with open(image_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'analysis': result,
            'image_data': f'data:image/jpeg;base64,{img_data}',
            'analysis_types': analysis_types
        })
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def custom_query():
    """Perform a custom VLM query on an image."""
    if not init_vlm():
        return jsonify({'error': 'VLM not available'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    prompt = request.form.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        # Save uploaded image temporarily
        image_path = UPLOAD_FOLDER / file.filename
        file.save(str(image_path))
        
        # Query VLM
        logger.info(f"Custom VLM query: {prompt}")
        result = vlm_engine.query(str(image_path), prompt)
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'prompt': prompt
        })
    except Exception as e:
        logger.error(f"Custom query failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize VLM on startup
    init_vlm()
    
    # Run app
    logger.info("Starting Cookify VLM Demo Server")
    app.run(debug=True, host='0.0.0.0', port=5000)

