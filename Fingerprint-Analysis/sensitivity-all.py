import os
import random
import pdqhash
import numpy as np
import cv2
from PIL import Image
import imagehash
import csv
import json

# Define the hash functions to be used
hash_functions = {
    # 'average_hash': imagehash.average_hash,
    # 'dhash': imagehash.dhash,
    # 'phash': imagehash.phash,
    # 'whash': imagehash.whash,
    'pdqhash': pdqhash.compute
}

# Define the transformations and parameters
transformations = {
    'Cropout': [0.2, 0.3],
    'Cropping': [0.9, 0.7],
    'Copy-Move': [0.1, 0.2],
    'Inpainting': [0.1, 0.2]
}

def pdq_string_hash(image):
    tmp_image = np.array(image)
    hash_vector, _ = pdqhash.compute(tmp_image)
    string_representation = ''.join(map(str, hash_vector.tolist()))
    return string_representation


# Hamming distance between two hash values
def hamming_distance(hash1, hash2):
    """Compute the Hamming distance between two perceptual hashes in hexadecimal format."""
    # Convert the hexadecimal hash to binary
    bin_hash1 = ''.join(f'{int(c, 16):04b}' for c in str(hash1))
    bin_hash2 = ''.join(f'{int(c, 16):04b}' for c in str(hash2))
    # Compute the Hamming distance by comparing the binary strings
    return sum(b1 != b2 for b1, b2 in zip(bin_hash1, bin_hash2))


# Apply transformations in memory without saving the file
def apply_transformation(image, transformation, parameter):
    if transformation == 'Cropout':
        np_image = np.array(image)
        h, w, _ = np_image.shape
        ch, cw = int(h * parameter), int(w * parameter)
        x = random.randint(0, w - cw)
        y = random.randint(0, h - ch)
        np_image[y:y + ch, x:x + cw] = 0
        image = Image.fromarray(np_image)
    elif transformation == 'Cropping':
        width, height = image.size
        crop_width, crop_height = int(width * parameter), int(height * parameter)
        image = image.crop((0, 0, crop_width, crop_height))
    elif transformation == 'Copy-Move':
        np_image = np.array(image)
        height, width, _ = np_image.shape

        # Randomly select the patch size as a portion of the image size
        patch_size_ratio = np.random.uniform(parameter, parameter + 0.1)
        patch_size = int(min(height, width) * patch_size_ratio)

        # Randomly select the top-left corner of the patch to copy
        src_y = np.random.randint(0, height - patch_size)
        src_x = np.random.randint(0, width - patch_size)

        # Randomly select the top-left corner of the location to paste the patch
        dst_y = np.random.randint(0, height - patch_size)
        dst_x = np.random.randint(0, width - patch_size)

        # Copy the patch
        patch = np_image[src_y:src_y + patch_size, src_x:src_x + patch_size]

        # Paste the patch at the new location
        np_image[dst_y:dst_y + patch_size, dst_x:dst_x + patch_size] = patch

        image = Image.fromarray(np_image)
    elif transformation == 'Inpainting':
        image = np.array(image)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[int(image.shape[0] * parameter):, int(image.shape[1] * parameter):] = 1
        image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        image = Image.fromarray(image)
    return image


# Directory with images
image_directory = "div2k-png-1-900"  # Set the correct path

# Loop over each hash function
for hash_name, hash_func in hash_functions.items():
    results = {}

    # Process each transformation
    for transformation, params in transformations.items():
        results[transformation] = {}
        for param in params:
            print(transformation, param)
            hamming_counts = [0] * 257  # List to count occurrences of each Hamming distance (0-256)
            hash_distances = []
            for image_file in os.listdir(image_directory):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    original_image = Image.open(os.path.join(image_directory, image_file))
                    # phash, whahs, ahash, dhash:
                    # original_hash = hash_func(original_image, hash_size=16)
                    # PDQ:
                    original_hash = pdq_string_hash(original_image)

                    transformed_image = apply_transformation(original_image, transformation, param)

                    # phash, whahs, ahash, dhash:
                    # transformed_hash = hash_func(transformed_image, hash_size=16)
                    # PDQ:
                    transformed_hash = pdq_string_hash(transformed_image)

                    distance = hamming_distance(original_hash, transformed_hash)
                    hash_distances.append(distance)
                    hamming_counts[distance] += 1

            min_distance = min(hash_distances)
            max_distance = max(hash_distances)
            avg_distance = np.mean(hash_distances)

            # Save the Hamming distance counts to a CSV file
            csv_file = f'sensitivity-output/{hash_name}_{transformation}_param_{param}.csv'
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Hamming Distance', 'Number of Images'])
                for i in range(257):
                    writer.writerow([i, hamming_counts[i]])

            # Save the min, max, avg values to the results dictionary
            results[transformation][param] = {
                'min': min_distance,
                'max': max_distance,
                'avg': avg_distance
            }

            print(f'Results saved to sensitivity-output/{csv_file} with min={min_distance}, max={max_distance}, avg={avg_distance:.3f}')

    # Save the summary results to a JSON file
    output_file = f'sensitivity-output/{hash_name}_sensitivity_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f'Summary results saved to {output_file}')
