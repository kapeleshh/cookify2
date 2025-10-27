"""
Web UI for Cookify - A Flask-based web interface for the Cookify recipe extraction system
"""

import os
import sys
import logging
import json
import time
import uuid
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_from_directory, redirect, url_for

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline import Pipeline
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

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Set maximum file size (50MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['RESULTS_FOLDER'] = str(RESULTS_FOLDER)

# Initialize pipeline
pipeline = None

def allowed_file(filename):
    """
    Check if file has an allowed extension.
    
    Args:
        filename (str): Filename to check.
        
    Returns:
        bool: True if file has an allowed extension, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_pipeline():
    """
    Initialize the pipeline.
    
    Returns:
        Pipeline: Initialized pipeline.
    """
    global pipeline
    
    if pipeline is None:
        logger.info("Initializing pipeline")
        
        # Load config
        config = load_config()
        
        # Create pipeline
        pipeline = Pipeline(config_path='config.yaml')
        
        logger.info("Pipeline initialized successfully")
    
    return pipeline

@app.route('/')
def index():
    """
    Render the index page.
    
    Returns:
        str: Rendered HTML.
    """
    return render_template('index.html')

@app.route('/api/vlm/status')
def vlm_status():
    """Check VLM status and return as JSON."""
    global pipeline
    
    try:
        # Initialize pipeline if needed
        if pipeline is None:
            init_pipeline()
        
        if hasattr(pipeline, 'vlm_engine') and pipeline.vlm_engine:
            is_connected = pipeline.vlm_engine.test_connection()
            models = pipeline.vlm_engine.list_available_models()
            model_name = pipeline.config.get('vlm', {}).get('model', 'unknown')
            
            return jsonify({
                'enabled': True,
                'connected': is_connected,
                'model': model_name,
                'available_models': models,
                'status': 'active' if is_connected else 'disconnected'
            })
        else:
            vlm_enabled = False
            if hasattr(pipeline, 'config'):
                vlm_enabled = pipeline.config.get('vlm', {}).get('enabled', False)
            
            return jsonify({
                'enabled': vlm_enabled,
                'connected': False,
                'message': 'VLM not initialized' if vlm_enabled else 'VLM disabled in config',
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

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload.
    
    Returns:
        Response: JSON response with upload status.
    """
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Generate unique filename
    unique_id = str(uuid.uuid4())
    extension = file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{unique_id}.{extension}"
    
    # Save file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    
    logger.info(f"File uploaded: {file_path}")
    
    # Return success response with unique ID
    return jsonify({
        'success': True,
        'message': 'File uploaded successfully',
        'id': unique_id,
        'filename': file.filename,
        'path': file_path
    })

@app.route('/process/<unique_id>', methods=['POST'])
def process_video(unique_id):
    """
    Process uploaded video.
    
    Args:
        unique_id (str): Unique ID of the uploaded video.
        
    Returns:
        Response: JSON response with processing status.
    """
    # Find the uploaded file
    for extension in ALLOWED_EXTENSIONS:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}.{extension}")
        if os.path.exists(file_path):
            break
    else:
        return jsonify({'error': 'File not found'}), 404
    
    # Initialize pipeline if needed
    init_pipeline()
    
    try:
        # Process video
        logger.info(f"Processing video: {file_path}")
        start_time = time.time()
        
        result = pipeline.process_video(file_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Save result
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{unique_id}.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        logger.info(f"Video processed successfully in {processing_time:.2f}s: {result_path}")
        
        # Return success response
        return jsonify({
            'success': True,
            'message': 'Video processed successfully',
            'id': unique_id,
            'processing_time': processing_time,
            'result_path': result_path,
            'result': result
        })
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

@app.route('/results/<unique_id>', methods=['GET'])
def get_results(unique_id):
    """
    Get processing results.
    
    Args:
        unique_id (str): Unique ID of the processed video.
        
    Returns:
        Response: JSON response with processing results.
    """
    # Find the result file
    result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{unique_id}.json")
    
    if not os.path.exists(result_path):
        return jsonify({'error': 'Results not found'}), 404
    
    try:
        # Load result
        with open(result_path, 'r') as f:
            result = json.load(f)
        
        # Return result
        return jsonify({
            'success': True,
            'id': unique_id,
            'result': result
        })
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return jsonify({'error': f'Error loading results: {str(e)}'}), 500

@app.route('/view/<unique_id>', methods=['GET'])
def view_results(unique_id):
    """
    View processing results in a formatted page.
    
    Args:
        unique_id (str): Unique ID of the processed video.
        
    Returns:
        str: Rendered HTML.
    """
    # Find the result file
    result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{unique_id}.json")
    
    if not os.path.exists(result_path):
        return render_template('error.html', error='Results not found')
    
    try:
        # Load result
        with open(result_path, 'r') as f:
            result = json.load(f)
        
        # Find the original video file
        video_path = None
        for extension in ALLOWED_EXTENSIONS:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}.{extension}")
            if os.path.exists(file_path):
                video_path = f"/uploads/{unique_id}.{extension}"
                break
        
        # Render result page
        return render_template('result.html', result=result, video_path=video_path, unique_id=unique_id)
    except Exception as e:
        logger.error(f"Error viewing results: {e}")
        return render_template('error.html', error=f'Error viewing results: {str(e)}')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files.
    
    Args:
        filename (str): Filename to serve.
        
    Returns:
        Response: File response.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/download/<unique_id>', methods=['GET'])
def download_results(unique_id):
    """
    Download processing results as JSON.
    
    Args:
        unique_id (str): Unique ID of the processed video.
        
    Returns:
        Response: File response.
    """
    # Find the result file
    result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{unique_id}.json")
    
    if not os.path.exists(result_path):
        return jsonify({'error': 'Results not found'}), 404
    
    return send_from_directory(app.config['RESULTS_FOLDER'], f"{unique_id}.json", as_attachment=True)

@app.route('/about')
def about():
    """
    Render the about page.
    
    Returns:
        str: Rendered HTML.
    """
    return render_template('about.html')

if __name__ == '__main__':
    # Initialize pipeline
    init_pipeline()
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)
