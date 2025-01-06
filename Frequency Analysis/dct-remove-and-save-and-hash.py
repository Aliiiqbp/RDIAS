import cv2
import numpy as np
import matplotlib.pyplot as plt
import imagehash
from PIL import Image
import pandas as pd
import os

# Load image
def load_image(image_path, grayscale=True):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) if grayscale else cv2.imread(image_path)

# Compute DCT
def compute_dct(image):
    return cv2.dct(image.astype(np.float32))

# Inverse DCT
def inverse_dct(dct_image):
    return cv2.idct(dct_image)

# Remove high frequency components from DCT
def remove_high_freq_dct(dct_image, percent):
    percent = 100 - percent
    dct_copy = np.copy(dct_image)
    height, width = dct_copy.shape
    max_freq = np.sqrt(height ** 2 + width ** 2)

    # Calculate the cutoff radius based on the percentage
    radius = int(max_freq * (percent / 100.0))

    for y in range(height):
        for x in range(width):
            distance = np.sqrt(x ** 2 + y ** 2)
            if distance > radius:
                dct_copy[y, x] = 0

    return dct_copy


# Compute perceptual hash
def compute_phash(image):
    return imagehash.phash(Image.fromarray(image), hash_size=10)


# Save images and compute hashes
def process_and_save_images(image, dct_image, results, rm_range, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    original_hash = compute_phash(image)
    data = []

    # Save the original image and its hash
    original_image_path = os.path.join(output_dir, "original_image.png")
    cv2.imwrite(original_image_path, image)
    data.append({"Filename": "original_image.png", "Hash": str(original_hash), "Hamming Distance": 0})

    # Process each modified image
    for percent in rm_range:
        _, spatial_image = results[percent]
        output_image_path = os.path.join(output_dir, f"image_{percent}_percent_removed.png")
        cv2.imwrite(output_image_path, spatial_image)

        modified_hash = compute_phash(np.uint8(spatial_image))
        hamming_distance = original_hash - modified_hash

        data.append({"Filename": f"image_{percent}_percent_removed.png",
                     "Hash": str(modified_hash),
                     "Hamming Distance": hamming_distance})

    # Save results to CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "hash_results.csv")
    df.to_csv(csv_path, index=False)


# Plot images
def plot_images(original, dct_original, results, rm_range):
    num_levels = len(results)
    plt.figure(figsize=(15, 5 * (num_levels + 1)))

    plt.subplot(num_levels + 1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')

    plt.subplot(num_levels + 1, 2, 2)
    plt.title('DCT of Original Image')
    plt.imshow(np.log(np.abs(dct_original) + 1), cmap='gray')

    for i, percent in enumerate(rm_range):
        dct_image, spatial_image = results[percent]

        plt.subplot(num_levels + 1, 2, 2 * i + 3)
        plt.title(f'{percent}% High Frequencies Removed (DCT)')
        plt.imshow(np.log(np.abs(dct_image) + 1), cmap='gray')

        plt.subplot(num_levels + 1, 2, 2 * i + 4)
        plt.title(f'{percent}% High Frequencies Removed (Spatial)')
        plt.imshow(spatial_image, cmap='gray')

    plt.tight_layout()
    plt.show()


# Paths to the images
image_path = '0801.jpeg'
output_dir = 'output_images-dct-removed'

# Load image
image = load_image(image_path)

# Compute DCT
dct_image = compute_dct(image)

# Define the removal ranges
rm_range = [50, 70, 80, 90, 95, 99]

# Compute the images with high frequencies removed
results = {}
for percent in rm_range:
    dct_modified = remove_high_freq_dct(dct_image, percent)
    spatial_image = inverse_dct(dct_modified)
    results[percent] = (dct_modified, np.clip(spatial_image, 0, 255))

# Process, save images, and compute hashes
process_and_save_images(image, dct_image, results, rm_range, output_dir)

# Plot the results
plot_images(image, dct_image, results, rm_range)
