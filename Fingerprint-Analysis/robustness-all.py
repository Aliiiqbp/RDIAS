import os
from PIL import Image, ImageFilter
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise
import imagehash
import json
import pdqhash
import cv2
import onnxruntime
from PIL import Image


def neuralhash(image):
    # Load ONNX model
    # session = onnxruntime.InferenceSession(sys.argv[1])
    session = onnxruntime.InferenceSession('neuralhash-files/model.onnx')

    # Load output hash matrix
    seed1 = open('neuralhash-files/neuralhash_128x96_seed1.dat', 'rb').read()[128:]
    seed1 = np.frombuffer(seed1, dtype=np.float32)
    seed1 = seed1.reshape([96, 128])

    # Preprocess image
    img = image.convert('RGB')
    img = img.resize([360, 360])
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    arr = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])

    # Run model
    inputs = {session.get_inputs()[0].name: arr}
    outs = session.run(None, inputs)

    # Convert model output to hex hash
    hash_output = seed1.dot(outs[0].flatten())
    hash_bits = ''.join(['1' if it >= 0 else '0' for it in hash_output])

    return hash_bits


# Define the transformations and parameters
transformations = {
    'JPEG Compression': [70],  # 50, 70, 90
    'Resizing': [0.5],  # 0.25, 0.5, 0.75
    # 'Gaussian Noise': [0.2, 0.4],
    # 'Salt and Pepper Noise': [0.005]
}


# Hash function (using perceptual hash based on DCT)
def compute_hash(image):
    return imagehash.phash(image, hash_size=16)


def pdq_string_hash(image):
    tmp_image = np.array(image)
    hash_vector, _ = pdqhash.compute(tmp_image)
    string_representation = ''.join(map(str, hash_vector.tolist()))
    return string_representation


def hamming_distance(hash1, hash2):
    """Compute the Hamming distance between two perceptual hashes in hexadecimal format."""
    # Convert the hexadecimal hash to binary
    bin_hash1 = ''.join(f'{int(c, 16):04b}' for c in str(hash1))
    bin_hash2 = ''.join(f'{int(c, 16):04b}' for c in str(hash2))

    # Compute the Hamming distance by comparing the binary strings
    return sum(b1 != b2 for b1, b2 in zip(bin_hash1, bin_hash2))


# Apply transformations in memory without saving the file
def apply_transformation(image, transformation, parameter):
    if transformation == 'JPEG Compression':
        # Simulate JPEG compression
        buffer = image.copy()
        buffer.save('/tmp/compressed_image.jpg', 'JPEG', quality=parameter)
        image = Image.open('/tmp/compressed_image.jpg')
    elif transformation == 'Resizing':
        # Resize the image
        width, height = image.size
        image = image.resize((int(width * parameter), int(height * parameter)), Image.Resampling.LANCZOS)
    elif transformation == 'Gaussian Noise':
        # Add Gaussian noise
        image = np.array(image)
        image = gaussian_filter(image, sigma=parameter)
        image = Image.fromarray(np.uint8(image))
    elif transformation == 'Salt and Pepper Noise':
        # Add salt and pepper noise
        image = np.array(image)
        image = random_noise(image, mode='s&p', amount=parameter)
        image = np.array(255 * image, dtype=np.uint8)
        image = Image.fromarray(image)
    return image


# Directory with images
image_directory = "div2k-png-1-900"  # Set the correct path
results = {}


# Process each transformation
for transformation, params in transformations.items():
    results[transformation] = {}
    for param in params:
        print(transformation, param)
        hash_distances = []
        for image_file in os.listdir(image_directory):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):

                # phash, dhash, ahash, whash:
                # original_image = Image.open(os.path.join(image_directory, image_file))
                # original_hash = compute_hash(original_image)

                # PDQ:
                # original_image = Image.open(os.path.join(image_directory, image_file))
                # original_hash = pdq_string_hash(original_image)

                # Apple NeuralHash
                original_image = Image.open(os.path.join(image_directory, image_file))
                original_hash = neuralhash(original_image)

                transformed_image = apply_transformation(original_image, transformation, param)

                # phash, dhash, ahash, whash:
                # transformed_hash = compute_hash(transformed_image)
                # PDQ:
                # transformed_hash = pdq_string_hash(transformed_image)
                # Apple:
                transformed_hash = neuralhash(transformed_image)

                distance = hamming_distance(original_hash, transformed_hash)
                hash_distances.append(distance)

        min_distance = min(hash_distances)
        max_distance = max(hash_distances)
        avg_distance = np.mean(hash_distances)

        results[transformation][param] = {
            'min': min_distance,
            'max': max_distance,
            'avg': avg_distance
        }


# Save results to a file
output_file = 'neuralhash_96_robustness_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)


print(f'Results saved to {output_file}')


'''
I have 4 JSON files, each of them related to a specific hash algorithm and its robustness against some common transformations. the transformations and their parameters are available in each JSON file alongside with min, max, and average values of the hamming distance between the original image and the transformed image using that transformation with that parameter.

Combine all these files and give me the table that includes all the information in LATEX format with the following structures:

Structure 1:
columns: algorithm name, transformation1 (sub columns: parameter1, parametere2, ...), transformation2 (sub columns: parameter1, parametere2, ...), .....
rows: algorithm1, algorithm2, algorithm3, algorithm4.

Structure 2:
columns: transformations, algorithm1, algorithm2, algorithm3, algorithm4
rows: transformation1 (sub rows: parameter1, parameter2, ...), transformation2 (sub rows: parameter1, parameter2, ...), .....
'''
