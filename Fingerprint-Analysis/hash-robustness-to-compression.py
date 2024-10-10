import os
import imagehash
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import pdqhash
import cv2


def pdq_string_hash(image):
    tmp_image = cv2.imread(image)
    tmp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hash_vector, _ = pdqhash.compute(tmp_image)
    string_representation = ''.join(map(str, hash_vector.tolist()))
    return string_representation


def compute_hash(image):
    """Compute the perceptual hash of the image."""
    return imagehash.phash(image, hash_size=16)


def compress_image_in_memory(image, quality):
    """Compress the image in memory using JPEG compression with the given quality."""
    compressed_image_io = BytesIO()
    image.save(compressed_image_io, format='JPEG', quality=quality)
    compressed_image_io.seek(0)
    return Image.open(compressed_image_io)


def hamming_distance(hash1, hash2):
    """Compute the Hamming distance between two perceptual hashes in hexadecimal format."""
    # Convert the hexadecimal hash to binary
    bin_hash1 = ''.join(f'{int(c, 16):04b}' for c in str(hash1))
    bin_hash2 = ''.join(f'{int(c, 16):04b}' for c in str(hash2))

    for i in range(len(bin_hash1)):
        if bin_hash1[i] != bin_hash2[i]:
            print(i)
    print('##########')

    # Compute the Hamming distance by comparing the binary strings
    return sum(b1 != b2 for b1, b2 in zip(bin_hash1, bin_hash2))


def process_images(directory, quality=90):
    """Compute hash, compress images in memory, and analyze the robustness to compression."""
    original_hashes = {}
    hamming_distances = []

    # Compute the hash for all original images
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            original_path = os.path.join(directory, filename)

            with Image.open(original_path) as original_image:
                # phash, whash, dhash, ahash:
                # original_hash = compute_hash(original_image)
                # pdq:
                original_hash = pdq_string_hash(original_image)

                original_hashes[filename] = original_hash

                # Compress the image in memory
                compressed_image = compress_image_in_memory(original_image, quality=quality)

                # Compute the hash for the compressed image
                # phash, whash, dhash, ahash:
                # compressed_hash = compute_hash(compressed_image)
                # pdq:
                compressed_hash = pdq_string_hash(compressed_image)

                # Compute the Hamming distance between original and compressed image
                distance = hamming_distance(original_hash, compressed_hash)
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

    # Annotate the mean and std on the plot
    # plt.text(mean_distance, max(counts) * 0.9, f'Mean: {mean_distance:.2f}', color='red', ha='center')
    # plt.text(mean_distance + std_distance, max(counts) * 0.8, f'+1 STD: {mean_distance + std_distance:.2f}',
    #          color='green', ha='center')
    # plt.text(mean_distance - std_distance, max(counts) * 0.8, f'-1 STD: {mean_distance - std_distance:.2f}',
    #          color='green', ha='center')

    # Set plot labels and title
    plt.title(f'Hamming Distance Distribution after JPEG Compression (Quality={quality}%)')
    plt.xlabel('Hamming Distance')
    plt.ylabel('Frequency')
    plt.legend()

    plt.xticks(range(min(hamming_distances), max(hamming_distances) + 1))

    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()


# Example usage
directory = 'div2k-png-1-900'  # Replace with your image directory path
quality = 50  # JPEG compression quality factor
process_images(directory, quality)
