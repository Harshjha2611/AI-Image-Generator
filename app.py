import os
import base64
import logging
import requests
import json
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Stability AI API configuration
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")
STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

# User session tracking - limit non-logged in users to 1 generation
app.config['GUEST_LIMIT'] = 1  # Allow 1 image for non-logged-in users

# Import models
from models import User, Image

# Setup user loader callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create tables
with app.app_context():
    db.create_all()

@app.route('/generate', methods=['POST'])
def generate_image():
    """
    Generate an image using Stability AI API based on the text prompt.
    
    Returns:
        JSON response with the generated image data or error message.
    """
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # Check if guest user has exceeded limit
        if not current_user.is_authenticated:
            if session.get('guest_generations', 0) >= app.config['GUEST_LIMIT']:
                return jsonify({
                    'error': 'Guest Limit Reached',
                    'details': 'You have reached the limit for free image generations. Please sign up or log in to continue.',
                    'require_auth': True
                }), 403
            session['guest_generations'] = session.get('guest_generations', 0) + 1
        
        if not STABILITY_API_KEY:
            logger.error("No API key found for Stability AI")
            return jsonify({
                'error': 'API Key Missing', 
                'details': 'Please provide a valid Stability AI API key to use this service.'
            }), 401
        
        logger.debug(f"Sending request to Stability AI API with prompt: {prompt}")
        
        # Parameters for Stability AI
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "text_prompts": [
                {"text": prompt, "weight": 1.0},
                {"text": "blurry, bad quality, distorted, disfigured", "weight": -1.0}
            ],
            "cfg_scale": 7.0,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30
        }
        
        response = requests.post(STABILITY_API_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code != 200:
            logger.error(f"Stability AI API error: {response.status_code}, {response.text}")
            error_detail = response.json().get('message', f"API returned status code {response.status_code}")
            return jsonify({
                'error': 'Failed to generate image', 
                'details': error_detail
            }), 500
        
        response_data = response.json()
        
        if not response_data.get('artifacts'):
            return jsonify({'error': 'No image was generated'}), 500
        
        image_base64 = response_data['artifacts'][0]['base64']
        
        if current_user.is_authenticated:
            new_image = Image(prompt=prompt, image_data=image_base64, user_id=current_user.id)
        else:
            new_image = Image(prompt=prompt, image_data=image_base64)
        
        db.session.add(new_image)
        db.session.commit()
        
        return jsonify({
            'image': image_base64,
            'is_authenticated': current_user.is_authenticated,
            'guest_generations': session.get('guest_generations', 0) if not current_user.is_authenticated else None,
            'guest_limit': app.config['GUEST_LIMIT']
        })
    
    except requests.exceptions.ConnectionError:
        logger.error("Connection error: Could not connect to Stability AI API")
        return jsonify({'error': 'Connection Error', 'details': 'Could not connect to Stability AI API. Please check your internet connection.'}), 503
    
    except requests.exceptions.Timeout:
        logger.error("Request timeout: Stability AI API took too long to respond")
        return jsonify({'error': 'Request Timeout', 'details': 'Stability AI API took too long to respond. Try a simpler prompt.'}), 504
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'Unexpected Error', 'details': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
