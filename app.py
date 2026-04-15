# ================================================================
# FakeShield v2 — Live Web Evidence Fake News Detector
# Team: Pratap Bambadi (2BU23CS100) | Preetam J Hiremath (2BU23CS104)
# Dept. CSE, SGBIT College
# ================================================================

import os
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, abort, make_response
from dotenv import load_dotenv

import config
from utils.web_fetcher import WebFetcher

# Load master predictor and models
try:
    from utils.predictor import FakeNewsPredictor
    print("Initializing FakeNewsPredictor Engine...")
    master_predictor = FakeNewsPredictor()
    model_loaded = master_predictor.ml_model is not None
except Exception as e:
    print(f"Error loading master predictor: {e}")
    master_predictor = None
    model_loaded = False

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super_secret_dev_key")

# --- Rate Limiting Architecture ---
rate_limits = {}
MAX_REQS_PER_MIN = 10

def check_rate_limit(ip_address):
    """ Enforces 10 requests per minute securely via sliding dictionary map. """
    now = time.time()
    if ip_address not in rate_limits:
        rate_limits[ip_address] = []
        
    # Filter array retaining only elements hitting within the last 60 seconds
    rate_limits[ip_address] = [t for t in rate_limits[ip_address] if now - t < 60]
    
    if len(rate_limits[ip_address]) >= MAX_REQS_PER_MIN:
        return True
        
    rate_limits[ip_address].append(now)
    return False

@app.before_request
def apply_rate_limit():
    # Only enforce strictly on analysis endpoints to preserve simple browsing operations
    if request.path in ['/analyze', '/api/v1/analyze'] and request.method == 'POST':
        client_ip = request.remote_addr or "127.0.0.1"
        if check_rate_limit(client_ip):
            app.logger.warning(f"Rate limit breached by {client_ip}")
            abort(429)

def log_request_data(route, input_type, processing_time):
    """ Centralized custom server logger mapping required structural outputs """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    app.logger.info(f"[{timestamp}] Route: {route} | InputType: {input_type} | ProcessingTime: {processing_time}ms")

# --- Endpoints ---

@app.route('/')
def index():
    return render_template(
        'index.html',
        app_name=config.APP_NAME,
        tagline=config.TAGLINE,
        team=config.TEAM,
        college=config.COLLEGE,
        version=config.VERSION
    )

@app.route('/about')
def about():
    return render_template(
        'about.html',
        team=config.TEAM,
        college=config.COLLEGE,
        app_name=config.APP_NAME
    )

@app.route('/analyze', methods=['POST'])
def analyze():
    """ 
    Primary UI Orchestration API processing JSON or Form requests natively.
    """
    if not master_predictor:
        return jsonify({"error": "Prediction Engine failed to dynamically initialize"}), 500

    # Accommodate both direct Frontend JSON and standard multi-part Form payloads organically
    if request.is_json:
        data = request.get_json()
        raw_text = data.get('text', '').strip()
        is_url = data.get('is_url', False)
        
        input_text = None
        input_url = None
        if is_url:
            input_url = raw_text
        else:
            input_text = raw_text
    else:
        input_text = request.form.get('news_text', '').strip() or None
        input_url = request.form.get('news_url', '').strip() or None

    # Length constraints
    if not input_text and not input_url:
        return jsonify({"error": "At least one of text or URL must be globally provided."}), 400
        
    if input_text and len(input_text) < 20 and not input_url: # Bounds checking natively mapped
        return jsonify({"error": "Text segment must safely remain above 20 characters length."}), 400
        
    if input_url:
        # Auto-inject https:// to prevent generic Regex crashes on standard URL blocks
        if not input_url.startswith('http://') and not input_url.startswith('https://'):
            input_url = 'https://' + input_url
            
        web = WebFetcher()
        if not web.is_valid_url(input_url):
            return jsonify({"error": "Provide structurally valid standard URL pathways."}), 400

    # Analytics execution
    try:
        results = master_predictor.predict(input_text=input_text, input_url=input_url)
        process_time = results.get("processing_time_ms", 0)
        
        # Log securely
        log_request_data('/analyze', results.get("input_type", "unknown"), process_time)
        
        # AJAX Check logic determining pure downstream transmission
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json
        
        if is_ajax:
            return jsonify(results)
        else:
            return render_template('result.html', result=results, app_name=config.APP_NAME)
            
    except Exception as e:
        app.logger.error(f"Execution sequence failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/analyze', methods=['POST'])
def api_analyze():
    """ Structured programmatic API endpoint containing CORS. """
    if not master_predictor:
        abort(500)
        
    data = request.get_json(silent=True)
    if not data:
        return make_response(jsonify({"error": "Malformed or entirely empty JSON block passed."}), 400)
        
    input_text = data.get('text')
    input_url = data.get('url')
    
    if not input_text and not input_url:
        return make_response(jsonify({"error": "Payload must enforce at least 'text' or 'url' inclusion."}), 400)
        
    results = master_predictor.predict(input_text=input_text, input_url=input_url)
    
    log_request_data('/api/v1/analyze', results.get("input_type", "unknown"), results.get("processing_time_ms", 0))
    
    # Simple explicit API CORS mappings
    response = jsonify(results)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/api/v1/status', methods=['GET'])
def api_status():
    """ Outputs global internal backend mapping constraints securely. """
    newsapi_active = master_predictor.news_api.is_api_available() if master_predictor else False
    
    return jsonify({
        "status": "ok" if master_predictor else "error",
        "newsapi": newsapi_active,
        "model_loaded": model_loaded,
        "version": config.VERSION
    })

# --- Error Bindings ---

@app.errorhandler(404)
def not_found_error(error):
    # Differentiate between API mapping or HTML
    if request.path.startswith('/api/'):
        return jsonify({"error": "API route definitively unrecognized."}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    if request.path.startswith('/api/'):
        return jsonify({"error": "Central API processing crash encountered internally."}), 500
    return render_template('error.html'), 500

@app.errorhandler(429)
def ratelimit_error(error):
    msg = "Usage throttle triggered: Exhausted the 10 request per minute maximal limit safely."
    if request.path.startswith('/api/') or request.is_json:
        return jsonify({"error": msg, "retry_after": 60}), 429
    return str(msg), 429

if __name__ == '__main__':
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ('true', '1', 't')
    app.run(debug=debug_mode, port=5000)
