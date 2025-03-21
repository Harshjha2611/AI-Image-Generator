document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const galleryContainer = document.getElementById('gallery-images-container');
    const searchInput = document.getElementById('gallery-search');
    const searchBtn = document.getElementById('search-btn');
    const sortSelect = document.getElementById('gallery-sort');
    
    // Event listeners
    searchBtn.addEventListener('click', filterGallery);
    searchInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            filterGallery();
        }
    });
    
    sortSelect.addEventListener('change', loadGalleryImages);
    
    // Initial load
    loadGalleryImages();
    
    /**
     * Load gallery images from the server
     */
    function loadGalleryImages() {
        // Show loading
        galleryContainer.innerHTML = `
            <div class="col-12 text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-3">Loading gallery...</p>
            </div>
        `;
        
        // Get filter values
        const searchTerm = searchInput.value.trim();
        const sortOrder = sortSelect.value;
        
        // Fetch data from the API
        fetch(`/gallery-images?search=${encodeURIComponent(searchTerm)}&sort=${sortOrder}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load gallery images');
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                displayGalleryImages(data.images);
            })
            .catch(error => {
                console.error('Error loading gallery images:', error);
                galleryContainer.innerHTML = `
                    <div class="col-12 text-center">
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-circle"></i> 
                            Error loading images: ${error.message || 'Unknown error'}
                        </div>
                    </div>
                `;
            });
    }
    
    /**
     * Display the gallery images in the container
     */
    function displayGalleryImages(images) {
        if (!images || images.length === 0) {
            galleryContainer.innerHTML = `
                <div class="col-12 text-center">
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle"></i> No images found. 
                        Try a different search term or be the first to create some!
                    </div>
                </div>
            `;
            return;
        }
        
        // Clear the container
        galleryContainer.innerHTML = '';
        
        // Add each image
        images.forEach(image => {
            const createdDate = new Date(image.created_at);
            const formattedDate = createdDate.toLocaleDateString() + ' ' + createdDate.toLocaleTimeString();
            
            const imageCard = document.createElement('div');
            imageCard.className = 'col-lg-4 col-md-6 col-sm-12 gallery-item mb-4';
            imageCard.innerHTML = `
                <div class="gallery-card" data-id="${image.id}">
                    <div class="gallery-image">
                        <img src="data:image/png;base64,${image.image_data}" alt="${escapeHtml(image.prompt)}" loading="lazy">
                    </div>
                    <div class="gallery-details">
                        <div class="gallery-prompt">"${escapeHtml(image.prompt)}"</div>
                        <div class="gallery-date"><i class="far fa-calendar-alt"></i> ${formattedDate}</div>
                        <div class="gallery-actions">
                            <button class="btn btn-sm btn-download" onclick="downloadGalleryImage(${image.id})">
                                <i class="fas fa-download"></i>
                            </button>
                            <button class="btn btn-sm btn-delete" onclick="deleteGalleryImage(${image.id})">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `;
            
            galleryContainer.appendChild(imageCard);
        });
    }
    
    /**
     * Filter gallery images based on search criteria
     */
    function filterGallery() {
        loadGalleryImages();
    }
});

/**
 * Download a gallery image by ID
 */
function downloadGalleryImage(imageId) {
    fetch(`/image/${imageId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to retrieve image');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Create a temporary link for download
            const downloadLink = document.createElement('a');
            downloadLink.href = `data:image/png;base64,${data.image_data}`;
            downloadLink.download = `ai-generated-image-${imageId}.png`;
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
        })
        .catch(error => {
            console.error('Error downloading image:', error);
            alert('Failed to download image: ' + error.message);
        });
}

/**
 * Delete a gallery image by ID
 */
function deleteGalleryImage(imageId) {
    if (!confirm('Are you sure you want to delete this image? This action cannot be undone.')) {
        return;
    }
    
    fetch(`/image/${imageId}`, {
        method: 'DELETE',
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to delete image');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Remove the image from the DOM
            const galleryItem = document.querySelector(`.gallery-card[data-id="${imageId}"]`).parentNode;
            galleryItem.remove();
            
            // Show success message
            alert('Image deleted successfully');
        })
        .catch(error => {
            console.error('Error deleting image:', error);
            alert('Failed to delete image: ' + error.message);
        });
}

/**
 * Helper to escape HTML in user-generated content
 */
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}