document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const promptInput = document.getElementById('prompt-input');
    const generateBtn = document.getElementById('generate-btn');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessage = document.getElementById('error-message');
    const imageContainer = document.getElementById('image-container');
    const generatedImage = document.getElementById('generated-image');
    const emptyState = document.getElementById('empty-state');
    const downloadBtn = document.getElementById('download-btn');
    const shareBtn = document.getElementById('share-btn');
    const recentImagesContainer = document.getElementById('recent-images-container');
    
    // Authentication state
    let isAuthenticated = false;
    let guestGenerations = 0;
    let guestLimit = 1;
    
    // Event listeners
    generateBtn.addEventListener('click', generateImage);
    promptInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            generateImage();
        }
    });
    
    downloadBtn.addEventListener('click', downloadImage);
    shareBtn.addEventListener('click', shareImage);
    
    // Initialize scroll reveal animation
    initScrollReveal();
    
    // Load recent images
    loadRecentImages();
    
    /**
     * Generate image based on prompt
     */
    function generateImage() {
        const prompt = promptInput.value.trim();
        
        if (!prompt) {
            showError('Please enter a prompt to generate an image');
            return;
        }
        
        // Show loading state
        showLoading();
        
        // Send request to the backend
        fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt: prompt }),
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.details || 'Failed to generate image');
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.details || data.error);
            }
            
            // Update authentication state
            isAuthenticated = data.is_authenticated || false;
            
            // Update guest generation count if applicable
            if (!isAuthenticated && data.guest_generations !== null && data.guest_generations !== undefined) {
                guestGenerations = data.guest_generations;
                
                // Update guest limit if provided
                if (data.guest_limit) {
                    guestLimit = data.guest_limit;
                }
                
                // Display guest limit notification if approaching limit
                if (guestGenerations >= guestLimit) {
                    showGuestLimitWarning();
                }
            }
            
            // Display the image
            displayImage(data.image);
            
            // Reload recent images to include the new one
            loadRecentImages();
        })
        .catch(error => {
            console.error('Error generating image:', error);
            
            if (error.message.includes('API Key Missing')) {
                showError('Stability AI API key is missing. Please add a valid API key to use this service.');
            } else if (error.message.includes('Connection Error')) {
                showError('Could not connect to the Stability AI API. Please check your internet connection.');
            } else if (error.message.includes('Request Timeout')) {
                showError('The image generation timed out. Try a simpler prompt.');
            } else if (error.message.includes('Guest Limit Reached')) {
                showGuestLimitReached();
            } else {
                showError(error.message || 'An unexpected error occurred while generating the image');
            }
        });
    }
    
    /**
     * Display the generated image
     */
    function displayImage(base64Image) {
        // Hide loading indicator and empty state
        hideLoading();
        emptyState.classList.add('d-none');
        
        // Set image source
        generatedImage.src = `data:image/png;base64,${base64Image}`;
        
        // Show image container with a smooth animation
        imageContainer.classList.remove('d-none');
        imageContainer.classList.add('animate-scale-in');
        
        // Store the current image data
        imageContainer.dataset.imageData = base64Image;
    }
    
    /**
     * Show loading indicator
     */
    function showLoading() {
        loadingIndicator.classList.remove('d-none');
        imageContainer.classList.add('d-none');
        errorMessage.classList.add('d-none');
        emptyState.classList.add('d-none');
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
    }
    
    /**
     * Hide loading indicator
     */
    function hideLoading() {
        loadingIndicator.classList.add('d-none');
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<i class="fas fa-wand-magic-sparkles"></i> Generate';
    }
    
    /**
     * Show error message
     */
    function showError(message) {
        hideLoading();
        errorMessage.textContent = message;
        errorMessage.classList.remove('d-none');
        imageContainer.classList.add('d-none');
        
        // Show empty state if no image is currently displayed
        if (imageContainer.classList.contains('d-none')) {
            emptyState.classList.remove('d-none');
        }
        
        // Auto-hide error after 8 seconds
        setTimeout(() => {
            errorMessage.classList.add('d-none');
        }, 8000);
    }
    
    /**
     * Download the generated image
     */
    function downloadImage() {
        const imageData = imageContainer.dataset.imageData;
        
        if (!imageData) {
            showError('No image available to download');
            return;
        }
        
        // Create a temporary link for download
        const downloadLink = document.createElement('a');
        downloadLink.href = `data:image/png;base64,${imageData}`;
        downloadLink.download = `ai-generated-image-${Date.now()}.png`;
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
    }
    
    /**
     * Share the generated image
     */
    function shareImage() {
        const imageData = imageContainer.dataset.imageData;
        
        if (!imageData) {
            showError('No image available to share');
            return;
        }
        
        // Check if Web Share API is supported
        if (navigator.share) {
            // Create a blob from the base64 image
            fetch(`data:image/png;base64,${imageData}`)
                .then(res => res.blob())
                .then(blob => {
                    const file = new File([blob], 'ai-generated-image.png', { type: 'image/png' });
                    
                    navigator.share({
                        title: 'My AI Generated Image',
                        text: 'Check out this image I created with DreamPixel AI!',
                        files: [file]
                    }).catch(error => {
                        console.error('Error sharing:', error);
                        showSocialShareFallback();
                    });
                });
        } else {
            // Fallback for browsers that don't support Web Share API
            showSocialShareFallback();
        }
    }
    
    /**
     * Show social sharing fallback options
     */
    function showSocialShareFallback() {
        alert('Direct sharing is not supported on this browser. You can download the image and share it manually.');
    }
    
    /**
     * Load recent images from the server
     */
    function loadRecentImages() {
        fetch('/recent-images')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load recent images');
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                displayRecentImages(data.images);
            })
            .catch(error => {
                console.error('Error loading recent images:', error);
                recentImagesContainer.innerHTML = `
                    <div class="col-12 text-center">
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle"></i> No recent images available yet. Be the first to create one!
                        </div>
                    </div>
                `;
            });
    }
    
    /**
     * Display the recent images in the container
     */
    function displayRecentImages(images) {
        if (!images || images.length === 0) {
            recentImagesContainer.innerHTML = `
                <div class="col-12 text-center">
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle"></i> No images have been generated yet. Be the first to create one!
                    </div>
                </div>
            `;
            return;
        }
        
        // Clear the container
        recentImagesContainer.innerHTML = '';
        
        // Add each image
        images.forEach(image => {
            const createdDate = new Date(image.created_at);
            const formattedDate = createdDate.toLocaleDateString() + ' ' + createdDate.toLocaleTimeString();
            
            const imageCard = document.createElement('div');
            imageCard.className = 'col-md-4 col-sm-6';
            imageCard.innerHTML = `
                <div class="recent-image-card">
                    <img src="data:image/png;base64,${image.image_data}" alt="AI Generated Image" loading="lazy">
                    <div class="recent-image-info">
                        <div class="recent-image-prompt">"${image.prompt}"</div>
                        <div class="recent-image-date"><i class="far fa-clock"></i> ${formattedDate}</div>
                    </div>
                </div>
            `;
            
            recentImagesContainer.appendChild(imageCard);
        });
    }
    
    /**
     * Show warning when guest is approaching their generation limit
     */
    function showGuestLimitWarning() {
        // Create a floating notification about the guest limit
        const notification = document.createElement('div');
        notification.className = 'guest-limit-notification';
        notification.innerHTML = `
            <div class="notification-content">
                <h4><i class="fas fa-exclamation-circle"></i> Generation Limit</h4>
                <p>You have used ${guestGenerations} of your ${guestLimit} free generations.</p>
                <p>Create an account to unlock unlimited generations!</p>
                <div class="notification-actions">
                    <a href="/login" class="btn btn-sm btn-outline-light">Login</a>
                    <a href="/register" class="btn btn-sm btn-light">Sign Up</a>
                    <button class="btn-close" onclick="this.parentElement.parentElement.parentElement.remove()"></button>
                </div>
            </div>
        `;
        
        // Add notification to the page
        document.body.appendChild(notification);
        
        // Auto-remove after 12 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 12000);
    }
    
    /**
     * Show error when guest has reached their generation limit
     */
    function showGuestLimitReached() {
        hideLoading();
        
        // Create a full modal overlay
        const modal = document.createElement('div');
        modal.className = 'auth-modal';
        modal.innerHTML = `
            <div class="auth-modal-content">
                <h3><i class="fas fa-lock"></i> Free Limit Reached</h3>
                <p>You've used all your free image generations.</p>
                <p>Sign up for a free account to unlock unlimited AI image generation!</p>
                <div class="auth-modal-buttons">
                    <a href="/register" class="btn btn-primary">Create Account</a>
                    <a href="/login" class="btn btn-outline-primary">Login</a>
                </div>
                <button class="auth-modal-close" onclick="this.parentElement.parentElement.remove()">&times;</button>
            </div>
        `;
        
        // Add modal to the page
        document.body.appendChild(modal);
    }
    
    /**
     * Initialize scroll reveal animations
     */
    function initScrollReveal() {
        const revealElements = document.querySelectorAll('.feature-item, .timeline-item');
        
        function checkReveal() {
            const windowHeight = window.innerHeight;
            const revealPoint = 150;
            
            revealElements.forEach(element => {
                const revealTop = element.getBoundingClientRect().top;
                
                if (revealTop < windowHeight - revealPoint) {
                    element.classList.add('reveal', 'active');
                }
            });
        }
        
        // Add initial classes
        revealElements.forEach(element => {
            element.classList.add('reveal');
        });
        
        // Check on scroll
        window.addEventListener('scroll', checkReveal);
        
        // Check on load
        checkReveal();
    }
});
