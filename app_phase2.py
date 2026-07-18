"""
Flask Web Application - Phase 2
Text-Guided Artistic Photo Generation with Web Interface
"""

from flask import Flask, render_template, request, jsonify, Response
import os
import uuid
import subprocess
from pathlib import Path
import glob
import time
import json
import torch
from threading import Thread
from queue import Queue

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Global state for generation sessions
GENERATION_QUEUES = {}  # session_id -> Queue
GENERATION_STATE = {}   # session_id -> {'status': 'running'|'completed'|'error', 'progress': 0-100}

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Sample prompts for suggestions
SAMPLE_PROMPTS = [
    "Van Gogh painting style with swirling brushstrokes",
    "Watercolor painting aesthetic with soft colors",
    "Cinematic photography with dramatic lighting and moody atmosphere",
    "Oil painting style with impressionist brushwork",
    "Abstract art style with vibrant colors",
    "Pencil sketch style with fine details",
    "Surrealism artistic style with dreamlike elements",
    "Comic book art style with bold outlines",
    "Renaissance oil painting style",
    "Modern digital art style with neon colors"
]


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_device_info():
    """Get GPU/CPU device information"""
    if torch.cuda.is_available():
        return {
            'device': 'cuda',
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1e9
        }
    else:
        return {'device': 'cpu'}


@app.route('/')
def index():
    """Main page"""
    return render_template(
        'index_phase2.html',
        sample_prompts=SAMPLE_PROMPTS,
        device_info=get_device_info()
    )


@app.route('/api/device-info', methods=['GET'])
def device_info():
    """Get device information"""
    info = get_device_info()
    return jsonify(info)


@app.route('/api/sample-prompts', methods=['GET'])
def sample_prompts():
    """Get sample prompts"""
    return jsonify({'prompts': SAMPLE_PROMPTS})


@app.route('/api/output-stream/<session_id>', methods=['GET'])
def output_stream(session_id):
    """Stream generation output using Server-Sent Events (SSE)"""
    def generate_output():
        while True:
            # Check if generation is complete
            if session_id not in GENERATION_STATE:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
                break
            
            state = GENERATION_STATE[session_id]
            queue = GENERATION_QUEUES.get(session_id)
            
            if queue and not queue.empty():
                # Send queued output
                line = queue.get()
                if line == '[GENERATION_COMPLETE]':
                    yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'type': 'output', 'text': line})}\n\n"
            else:
                # Keep connection alive if generation is still running
                if state['status'] == 'running':
                    time.sleep(0.1)
                else:
                    yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                    break
    
    return Response(generate_output(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    })


@app.route('/api/generate', methods=['POST'])
def generate():
    """
    Start generation and return immediately with session_id
    Client will connect to /api/output-stream/<session_id> for real-time output
    """
    try:
        # Check if image file is present
        if 'content_image' not in request.files:
            return jsonify({'error': 'No content image provided'}), 400
        
        content_file = request.files['content_image']
        if content_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(content_file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Get text prompt
        text_prompt = request.form.get('text_prompt', '').strip()
        if not text_prompt:
            return jsonify({'error': 'Text prompt cannot be empty'}), 400
        
        # Get generation parameters
        try:
            num_variations = int(request.form.get('num_variations', 2))
            num_steps = int(request.form.get('num_steps', 30))
            strength = float(request.form.get('strength', 0.75))
            guidance_scale = float(request.form.get('guidance_scale', 7.5))
        except ValueError:
            return jsonify({'error': 'Invalid parameter values'}), 400
        
        # Validate parameters - Optimized for GTX 1650
        num_variations = max(1, min(num_variations, 2))  # Limit to 1-2 for GPU memory
        num_steps = max(10, min(num_steps, 50))  # Limit to 10-50 for faster generation
        strength = max(0.3, min(strength, 0.95))  # Limit to 0.3-0.95
        guidance_scale = max(1.0, min(guidance_scale, 20.0))  # Limit to 1.0-20.0
        
        # Save uploaded image
        unique_id = str(uuid.uuid4())
        content_filename = f"{unique_id}_content_{content_file.filename}"
        content_path = os.path.join(UPLOAD_FOLDER, content_filename)
        content_file.save(content_path)
        
        # Create output directory
        output_dir = os.path.join(RESULT_FOLDER, unique_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize session state
        GENERATION_QUEUES[unique_id] = Queue()
        GENERATION_STATE[unique_id] = {'status': 'running', 'progress': 0}
        
        # Start generation in a background thread
        gen_thread = Thread(
            target=run_generation,
            args=(unique_id, content_path, text_prompt, output_dir, 
                  num_variations, num_steps, strength, guidance_scale)
        )
        gen_thread.daemon = True
        gen_thread.start()
        
        # Return immediately with session_id
        return jsonify({
            'success': True,
            'session_id': unique_id,
            'content_image': Path(content_path).as_posix().replace('\\', '/'),
            'text_prompt': text_prompt
        })
    
    except Exception as e:
        print(f"[ERROR] Error in /api/generate: {e}")
        return jsonify({'error': str(e)}), 500


def run_generation(session_id, content_path, text_prompt, output_dir, 
                   num_variations, num_steps, strength, guidance_scale):
    """Run generation in background thread and queue output"""
    queue = GENERATION_QUEUES.get(session_id)
    if not queue:
        return
    
    try:
        queue.put(f"{'='*70}\n[*] Starting Phase 2 Generation\n[*] Session ID: {session_id}\n[*] Text Prompt: '{text_prompt}'\n[*] Parameters: Steps={num_steps}, Strength={strength}, Guidance={guidance_scale}\n{'='*70}\n")
        
        # Call Phase 2 generator script
        process = subprocess.Popen(
            [
                'python', 'phase2_multimodal_generator.py',
                content_path,
                text_prompt,
                output_dir,
                '--num_variations', str(num_variations),
                '--num_steps', str(num_steps),
                '--strength', str(strength),
                '--guidance_scale', str(guidance_scale),
                '--device', 'cuda' if torch.cuda.is_available() else 'cpu'
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output line by line
        for line in process.stdout:
            line = line.rstrip('\n')
            queue.put(line)
            print(line)  # Also print to server console
        
        process.wait()
        
        if process.returncode == 0:
            # Get generated images
            pattern = os.path.join(output_dir, "variation_*.png")
            result_images = sorted(
                glob.glob(pattern),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            
            result_count = len(result_images)
            queue.put(f"\n[✓] Generation completed successfully! Generated {result_count} image(s)")
            GENERATION_STATE[session_id] = {
                'status': 'completed',
                'progress': 100,
                'result_images': [Path(p).as_posix().replace('\\', '/') for p in result_images],
                'num_generated': result_count
            }
        else:
            queue.put(f"\n[✗] Generation failed with return code {process.returncode}")
            GENERATION_STATE[session_id] = {'status': 'error', 'progress': 0}
        
        queue.put('[GENERATION_COMPLETE]')
    
    except Exception as e:
        error_msg = f"\n[ERROR] Generation error: {str(e)}"
        queue.put(error_msg)
        print(error_msg)
        GENERATION_STATE[session_id] = {'status': 'error', 'progress': 0, 'error': str(e)}
        queue.put('[GENERATION_COMPLETE]')



@app.route('/api/download/<session_id>/<image_name>', methods=['GET'])
def download_image(session_id, image_name):
    """Download generated image"""
    try:
        filepath = os.path.join(RESULT_FOLDER, session_id, image_name)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        from flask import send_file
        return send_file(filepath, as_attachment=True)
    
    except Exception as e:
        print(f"[ERROR] Download error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generation-result/<session_id>', methods=['GET'])
def generation_result(session_id):
    """Get final generation result after completion"""
    try:
        if session_id not in GENERATION_STATE:
            return jsonify({'error': 'Session not found'}), 404
        
        state = GENERATION_STATE[session_id]
        
        if state['status'] == 'running':
            return jsonify({'error': 'Generation still in progress'}), 202
        
        if state['status'] == 'error':
            return jsonify({'error': state.get('error', 'Generation failed')}), 500
        
        result = {
            'success': True,
            'session_id': session_id,
            'result_images': state.get('result_images', []),
            'num_generated': state.get('num_generated', 0)
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"[ERROR] Error in /api/generation-result: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })


if __name__ == '__main__':
    print("="*70)
    print("Phase 2: Text-Guided Artistic Photo Generation Web Application")
    print("="*70)
    print(f"[*] Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print(f"[*] Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"[*] Result folder: {os.path.abspath(RESULT_FOLDER)}")
    print("="*70)
    
    app.run(debug=True, port=5000)
