import os
import imagehash
from PIL import Image
import pandas as pd
import itertools
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
import numpy as np
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
    img = Image.open(image).convert('RGB')
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


def pdq_string_hash(image):
    tmp_image = Image.open(image)
    tmp_image = np.array(tmp_image)
    hash_vector, _ = pdqhash.compute(tmp_image)
    string_representation = ''.join(map(str, hash_vector.tolist()))
    return string_representation


# Function to compute perceptual hash
def compute_phash(image_path):
    image = Image.open(image_path)
    return str(imagehash.whash(image, hash_size=16))


# Directory containing images
image_dir = "div2k-png-1-900"

# Collecting all image file paths
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Compute the perceptual hash for each image
hashes = {}
for image_file in sorted(image_files):
    image_path = os.path.join(image_dir, image_file)

    # phash, whahs, ahash, dhash:
    # hash_value = compute_phash(image_path)

    # pdq:
    # hash_value = pdq_string_hash(image_path)

    # neuralhash
    hash_value = neuralhash(image_path)
    print(hash_value)

    hashes[image_file] = hash_value

# Save the hash values into a CSV file
df_hashes = pd.DataFrame(list(hashes.items()), columns=['Image', 'pHash'])
df_hashes.to_csv('image_hashes.csv', index=False)

# Compute the Hamming distances between each pair of images
hamming_distances = []
for (img1, hash1), (img2, hash2) in itertools.combinations(hashes.items(), 2):
    # Convert the hexadecimal hash to binary
    bin_hash1 = ''.join(f'{int(c, 16):04b}' for c in hash1)
    bin_hash2 = ''.join(f'{int(c, 16):04b}' for c in hash2)

    # Calculate the Hamming distance by comparing the bits
    distance = sum(c1 != c2 for c1, c2 in zip(bin_hash1, bin_hash2))
    hamming_distances.append(distance)


mean_distance = np.mean(hamming_distances)
std_distance = np.std(hamming_distances)

# Get the histogram data (frequency counts for each bin)
counts, bin_edges = np.histogram(hamming_distances, bins=range(min(hamming_distances), max(hamming_distances) + 1))
# histogram
# Create a line plot of the distribution
plt.plot(bin_edges[:-1], counts, color='blue', alpha=0.7, linestyle='-', marker='o')

# Add vertical lines for mean and standard deviation
plt.axvline(mean_distance, color='red', linestyle='--', label=f'Mean: {mean_distance:.2f}')
plt.axvline(mean_distance - std_distance, color='green', linestyle='--', label=f'STD: {std_distance:.2f}')
plt.axvline(mean_distance + std_distance, color='green', linestyle='--')

# Annotate the mean and std on the plot
plt.text(mean_distance, max(counts) * 0.9, '', color='red', ha='center')

plt.title('Distribution of Hamming Distances with Mean and STD')
plt.xlabel('Hamming Distance')
plt.ylabel('Frequency')
plt.legend()

plt.show()
