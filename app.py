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

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "dreampixel2611@gmail.com"
SMTP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD')
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

@app.route('/')
def index():
    """Render the main page of the application."""
    # Initialize guest session generation count if not exists
    if 'guest_generations' not in session and not current_user.is_authenticated:
        session['guest_generations'] = 0
    
    return render_template('index.html', 
                          is_authenticated=current_user.is_authenticated,
                          guest_generations=session.get('guest_generations', 0) if not current_user.is_authenticated else None)

@app.route('/gallery')
def gallery():
    """Render the gallery page."""
    return render_template('gallery.html')

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = 'remember' in request.form
        
        if not email or not password:
            flash('Please enter both email and password', 'error')
            return render_template('login.html')
            
        # Find the user
        user = User.query.filter_by(email=email).first()
        
        # Check if user exists and password is correct
        if user and user.check_password(password):
            login_user(user, remember=remember)
            
            # Reset guest generations counter
            if 'guest_generations' in session:
                session.pop('guest_generations')
                
            # Redirect to the requested page or home
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'error')
            
    return render_template('login.html')
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Basic validation
        errors = []
        if not username or len(username) < 3:
            errors.append('Username must be at least 3 characters long')
        if not email or '@' not in email:
            errors.append('Please enter a valid email address')
        if not password or len(password) < 6:
            errors.append('Password must be at least 6 characters long')
        if password != confirm_password:
            errors.append('Passwords do not match')
            
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            errors.append('Username already in use')
        if User.query.filter_by(email=email).first():
            errors.append('Email already registered')
            
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('register.html')
            
        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        # Save to database
        db.session.add(new_user)
        db.session.commit()
        
        # Log in the new user
        login_user(new_user)
        
        # Reset guest generations counter
        if 'guest_generations' in session:
            session.pop('guest_generations')
            
        # Send welcome email
        try:
            send_welcome_email(new_user)
        except Exception as e:
            logger.error(f"Failed to send welcome email: {e}")
            
        flash('Account created successfully! Welcome to DreamPixel!', 'success')
        return redirect(url_for('index'))
        
    return render_template('register.html')
    
@app.route('/logout')
@login_required
def logout():
    """Log out the current user."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))
    
@app.route('/profile')
@login_required
def profile():
    """Show user profile and generated images."""
    user_images = current_user.images.order_by(Image.created_at.desc()).all()
    return render_template('profile.html', user=current_user, images=user_images)

def send_welcome_email(user):
    """Send welcome email to a new user using SMTP."""
    if not SMTP_PASSWORD:
        logger.warning("SMTP password not set. Skipping welcome email.")
        return

    try:
        # Create SMTP connection
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)

        # Send welcome email to user
        welcome_msg = MIMEMultipart()
        welcome_msg['From'] = SMTP_USERNAME
        welcome_msg['To'] = user.email
        welcome_msg['Subject'] = "Welcome to DreamPixel!"
        
        welcome_body = f"""Hello {user.username},

Thank you for joining DreamPixel! We're excited to have you as part of our community.

With your account, you now have unlimited access to our AI image generation tools.

Feel free to explore the gallery and start creating amazing images from your imagination!

Best regards,
The DreamPixel Team"""
        
        welcome_msg.attach(MIMEText(welcome_body, 'plain'))
        server.send_message(welcome_msg)

        # Send notification to admin
        admin_msg = MIMEMultipart()
        admin_msg['From'] = SMTP_USERNAME
        admin_msg['To'] = 'dreampixel2611@gmail.com'
        admin_msg['Subject'] = "New User Registration"
        
        admin_body = f"""New User Registration:

Username: {user.username}
Email: {user.email}
Registration Time: {user.created_at}"""
        
        admin_msg.attach(MIMEText(admin_body, 'plain'))
        server.send_message(admin_msg)

        server.quit()
        logger.info(f"Successfully sent welcome emails for user {user.username}")
    except Exception as e:
        logger.error(f"Failed to send welcome email: {str(e)}")

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
            # Check if guest has reached the limit
            if session.get('guest_generations', 0) >= app.config['GUEST_LIMIT']:
                return jsonify({
                    'error': 'Guest Limit Reached',
                    'details': 'You have reached the limit for free image generations. Please sign up or log in to continue.',
                    'require_auth': True
                }), 403
            
            # Increment the guest generation count
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
                {
                    "text": prompt,
                    "weight": 1.0
                },
                {
                    "text": "blurry, bad quality, distorted, disfigured",
                    "weight": -1.0
                }
            ],
            "cfg_scale": 7.0,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30
        }
        
        # Send request to Stability AI API
        response = requests.post(STABILITY_API_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code != 200:
            logger.error(f"Stability AI API error: {response.status_code}, {response.text}")
            error_detail = response.json().get('message', f"API returned status code {response.status_code}")
            return jsonify({
                'error': 'Failed to generate image', 
                'details': error_detail
            }), 500
        
        # Extract image data from response
        response_data = response.json()
        
        if not response_data.get('artifacts'):
            return jsonify({'error': 'No image was generated'}), 500
        
        # Get the base64 image from the first artifact
        image_base64 = response_data['artifacts'][0]['base64']
        
        # Save to database with user_id if logged in
        if current_user.is_authenticated:
            new_image = Image(prompt=prompt, image_data=image_base64, user_id=current_user.id)
        else:
            new_image = Image(prompt=prompt, image_data=image_base64)
            
        db.session.add(new_image)
        db.session.commit()
        
        # Return the image (base64 encoded) with auth status
        return jsonify({
            'image': image_base64,
            'is_authenticated': current_user.is_authenticated,
            'guest_generations': session.get('guest_generations', 0) if not current_user.is_authenticated else None,
            'guest_limit': app.config['GUEST_LIMIT']
        })
    
    except requests.exceptions.ConnectionError:
        logger.error("Connection error: Could not connect to Stability AI API")
        return jsonify({
            'error': 'Connection Error', 
            'details': 'Could not connect to Stability AI API. Please check your internet connection.'
        }), 503
    
    except requests.exceptions.Timeout:
        logger.error("Request timeout: Stability AI API took too long to respond")
        return jsonify({
            'error': 'Request Timeout', 
            'details': 'Stability AI API took too long to respond. Try a simpler prompt.'
        }), 504
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': 'Unexpected Error', 
            'details': str(e)
        }), 500

@app.route('/recent-images', methods=['GET'])
def recent_images():
    """Get recent images generated by users for the home page."""
    try:
        # Fetch 10 most recent images
        images = Image.query.order_by(Image.created_at.desc()).limit(10).all()
        
        image_list = [{
            'id': img.id,
            'prompt': img.prompt,
            'image_data': img.image_data,
            'created_at': img.created_at.isoformat()
        } for img in images]
        
        return jsonify({'images': image_list})
        
    except Exception as e:
        logger.error(f"Error fetching recent images: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/gallery-images', methods=['GET'])
def gallery_images():
    """Get images for the gallery page with optional filtering."""
    try:
        # Get query parameters
        search = request.args.get('search', '').strip()
        sort_order = request.args.get('sort', 'newest')
        
        # Base query
        query = Image.query
        
        # Add search filter if provided
        if search:
            query = query.filter(Image.prompt.ilike(f'%{search}%'))
        
        # Add sorting
        if sort_order == 'oldest':
            query = query.order_by(Image.created_at)
        else:  # default to newest
            query = query.order_by(Image.created_at.desc())
        
        # Execute query
        images = query.all()
        
        image_list = [{
            'id': img.id,
            'prompt': img.prompt,
            'image_data': img.image_data,
            'created_at': img.created_at.isoformat()
        } for img in images]
        
        return jsonify({'images': image_list})
        
    except Exception as e:
        logger.error(f"Error fetching gallery images: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/image/<int:image_id>', methods=['GET', 'DELETE'])
def image_operations(image_id):
    """Handle operations on a specific image."""
    try:
        # Find the image
        image = Image.query.get_or_404(image_id)
        
        if request.method == 'GET':
            # Return image data
            return jsonify({
                'id': image.id,
                'prompt': image.prompt,
                'image_data': image.image_data,
                'created_at': image.created_at.isoformat()
            })
        
        elif request.method == 'DELETE':
            # Delete the image
            db.session.delete(image)
            db.session.commit()
            
            return jsonify({'success': True, 'message': 'Image deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error in image operations: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
