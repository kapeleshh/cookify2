"""
Video Processing Web UI for Cookify - Full video recipe extraction
"""

import os
import sys
import logging
import json
import uuid
import time
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_from_directory

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simple_video_pipeline import SimpleVideoPipeline
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
RESULTS_FOLDER = Path(__file__).parent / "results"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['RESULTS_FOLDER'] = str(RESULTS_FOLDER)

# Initialize pipeline
pipeline = None
processing_status = {}

def init_pipeline():
    """Initialize the video processing pipeline."""
    global pipeline
    
    if pipeline is None:
        try:
            logger.info("Initializing video processing pipeline")
            pipeline = SimpleVideoPipeline()
            logger.info("Pipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
    return True

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the video upload page."""
    return render_template('video.html')

@app.route('/api/vlm/status')
def vlm_status():
    """Check VLM and pipeline status."""
    try:
        if not init_pipeline():
            return jsonify({
                'enabled': False,
                'connected': False,
                'status': 'error',
                'message': 'Failed to initialize pipeline'
            }), 500
        
        if hasattr(pipeline, 'vlm_engine') and pipeline.vlm_engine:
            is_connected = pipeline.vlm_engine.test_connection()
            models = pipeline.vlm_engine.list_available_models() if is_connected else []
            model_name = pipeline.config.get('vlm', {}).get('model', 'unknown')
            
            return jsonify({
                'enabled': True,
                'connected': is_connected,
                'model': model_name,
                'available_models': models,
                'status': 'active' if is_connected else 'disconnected'
            })
        else:
            return jsonify({
                'enabled': False,
                'connected': False,
                'message': 'VLM not initialized',
                'status': 'disabled'
            })
    except Exception as e:
        logger.error(f"VLM status check failed: {e}")
        return jsonify({
            'enabled': False,
            'connected': False,
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Generate unique ID and save file
        unique_id = str(uuid.uuid4())
        extension = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{unique_id}.{extension}"
        file_path = UPLOAD_FOLDER / unique_filename
        
        file.save(str(file_path))
        logger.info(f"Video uploaded: {file_path}")
        
        # Initialize processing status
        processing_status[unique_id] = {
            'status': 'uploaded',
            'progress': 0,
            'message': 'Video uploaded successfully',
            'filename': file.filename
        }
        
        return jsonify({
            'success': True,
            'id': unique_id,
            'filename': file.filename,
            'message': 'Video uploaded successfully'
        })
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process/<unique_id>', methods=['POST'])
def process_video(unique_id):
    """Process uploaded video and extract recipe."""
    if not init_pipeline():
        return jsonify({'error': 'Pipeline not initialized'}), 500
    
    # Find uploaded video
    video_path = None
    for ext in ALLOWED_EXTENSIONS:
        path = UPLOAD_FOLDER / f"{unique_id}.{ext}"
        if path.exists():
            video_path = path
            break
    
    if not video_path:
        return jsonify({'error': 'Video not found'}), 404
    
    try:
        # Update status
        processing_status[unique_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting video processing...'
        }
        
        # Progress callback
        def progress_callback(message, progress):
            processing_status[unique_id] = {
                'status': 'processing',
                'progress': progress,
                'message': message
            }
        
        # Process video
        logger.info(f"Processing video: {video_path}")
        result = pipeline.process_video(str(video_path), progress_callback)
        
        # Save result
        result_path = RESULTS_FOLDER / f"{unique_id}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Update status
        processing_status[unique_id] = {
            'status': 'complete',
            'progress': 100,
            'message': 'Processing complete!'
        }
        
        logger.info(f"Video processed successfully: {result_path}")
        
        return jsonify({
            'success': True,
            'id': unique_id,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        processing_status[unique_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<unique_id>')
def get_status(unique_id):
    """Get processing status for a video."""
    if unique_id in processing_status:
        return jsonify(processing_status[unique_id])
    else:
        return jsonify({'error': 'ID not found'}), 404

@app.route('/api/result/<unique_id>')
def get_result(unique_id):
    """Get processing result."""
    result_path = RESULTS_FOLDER / f"{unique_id}.json"
    
    if not result_path.exists():
        return jsonify({'error': 'Result not found'}), 404
    
    try:
        with open(result_path, 'r') as f:
            result = json.load(f)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        logger.error(f"Failed to load result: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/view/<unique_id>')
def view_result(unique_id):
    """View the recipe result."""
    result_path = RESULTS_FOLDER / f"{unique_id}.json"
    
    if not result_path.exists():
        return render_template('error.html', error='Result not found')
    
    try:
        with open(result_path, 'r') as f:
            result = json.load(f)
        
        return render_template('recipe_result.html', result=result, unique_id=unique_id)
    except Exception as e:
        logger.error(f"Failed to load result: {e}")
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    init_pipeline()
    logger.info("Starting Cookify Video Processing Server")
    app.run(debug=True, host='0.0.0.0', port=5001)

