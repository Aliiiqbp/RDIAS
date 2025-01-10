// Function to add a fixed-position mark on the top right of the webpage
function addFixedMark(isAuthentic) {
    // Check if the mark already exists
    let mark = document.getElementById('my-authenticity-mark');
    if (mark) {
        // Update existing mark
        mark.style.backgroundColor = isAuthentic ? 'green' : 'red';
        mark.textContent = isAuthentic ? '✔' : '✖';
        return;
    }

    // Create the mark element
    mark = document.createElement('div');
    mark.id = 'my-authenticity-mark';

    // Style the mark
    mark.style.position = 'fixed'; // Fixed position relative to the viewport
    mark.style.top = '70px';       // 30 pixels from the top
    mark.style.left = '15px';     // 30 pixels from the right
    mark.style.width = '100px';     // Adjust size as needed
    mark.style.height = '100px';
    mark.style.borderRadius = '50%';
    mark.style.backgroundColor = isAuthentic ? 'green' : 'red';
    mark.style.color = 'white';
    mark.style.fontSize = '64px';
    mark.style.fontWeight = 'bold';
    mark.style.textAlign = 'center';
    mark.style.lineHeight = '100px'; // Line height matches height for vertical centering
    mark.style.zIndex = '9999';     // Ensure it appears above other elements
    mark.textContent = isAuthentic ? '✔' : '✖';

    // Optional: Add a tooltip for accessibility
    mark.title = isAuthentic ? 'All images are authentic' : 'Some images may not be authentic';
    mark.setAttribute('aria-label', mark.title);

    // Append the mark to the body
    document.body.appendChild(mark);
}

// Function to classify an image and return its authenticity
async function classifyImage(img) {
    const imageSrc = img.src;
    if (!imageSrc) return true; // Assume true if no source

    try {
        const response = await fetch('http://localhost:5000/check_image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_url: imageSrc })
        });
        const result = await response.json();
        return result.flag === "authentic";
    } catch (error) {
        console.error('Error fetching image classification:', error);
        return true; // Assume authentic in case of error
    }
}

// Function to process all images and update the mark
async function processImages() {
    const images = Array.from(document.querySelectorAll('img'));
    const imagePromises = images.map(async (img) => {
        if (img.naturalWidth >= 256 && img.naturalHeight >= 256) {
            return await classifyImage(img);
        } else {
            return true; // Assume authentic for small images
        }
    });

    const results = await Promise.all(imagePromises);
    const allImagesAuthentic = results.every((isAuthentic) => isAuthentic);
    addFixedMark(allImagesAuthentic);
}

// Observe changes to the DOM to detect newly added images
const observer = new MutationObserver(() => {
    processImages();
});
observer.observe(document.body, { childList: true, subtree: true });

// Initial processing of images
processImages();
