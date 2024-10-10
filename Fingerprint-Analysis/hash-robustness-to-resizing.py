import os
import imagehash
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def compute_hash(image):
    """Compute the perceptual hash of the image."""
    return imagehash.dhash(image, hash_size=16)

def resize_image_in_memory(image, scale):
    """Resize the image in memory by a given scale factor without changing aspect ratio."""
    width, height = image.size
    new_size = (int(width * scale), int(height * scale))
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    return resized_image

def hamming_distance(hash1, hash2):
    """Compute the Hamming distance between two perceptual hashes in hexadecimal format."""
    # Convert the hexadecimal hash to binary
    bin_hash1 = ''.join(f'{int(c, 16):04b}' for c in str(hash1))
    bin_hash2 = ''.join(f'{int(c, 16):04b}' for c in str(hash2))

    # Compute the Hamming distance by comparing the binary strings
    return sum(b1 != b2 for b1, b2 in zip(bin_hash1, bin_hash2))

def process_images_with_resize(directory, scale=0.5):
    """Compute hash, resize images in memory, and analyze the robustness to resizing."""
    original_hashes = {}
    hamming_distances = []

    # Compute the hash for all original images
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            original_path = os.path.join(directory, filename)

            with Image.open(original_path) as original_image:
                original_hash = compute_hash(original_image)
                original_hashes[filename] = original_hash

                # Resize the image in memory
                resized_image = resize_image_in_memory(original_image, scale=scale)

                # Compute the hash for the resized image
                resized_hash = compute_hash(resized_image)

                # Calculate the Hamming distance between original and resized image
                distance = hamming_distance(original_hash, resized_hash)
                hamming_distances.append(distance)

    # Calculate mean and standard deviation of Hamming distances
    mean_distance = np.mean(hamming_distances)
    std_distance = np.std(hamming_distances)

    # Get the histogram data (frequency counts for each bin)
    counts, bin_edges = np.histogram(hamming_distances, bins=range(min(hamming_distances), max(hamming_distances) + 2))

    # Create a bar plot of the distribution
    plt.bar(bin_edges[:-1], counts, width=1, color='blue', alpha=0.7)

    # Add vertical lines for mean and standard deviation
    plt.axvline(mean_distance, color='red', linestyle='--', label=f'Mean: {mean_distance:.2f}')
    plt.axvline(mean_distance - std_distance, color='green', linestyle='--', label=f'STD: {std_distance:.2f}')
    plt.axvline(mean_distance + std_distance, color='green', linestyle='--')

    # Set plot labels and title
    plt.title(f'Hamming Distance Distribution after Resizing (Scale={scale})')
    plt.xlabel('Hamming Distance')
    plt.ylabel('Frequency')
    plt.legend()

    plt.xticks(range(min(hamming_distances), max(hamming_distances) + 1))

    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()


# Example usage
directory = 'div2k-801-900-jpeg'  # Replace with your image directory path
scale = 0.5  # Resizing scale factor
process_images_with_resize(directory, scale)

