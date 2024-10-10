import os
import imagehash
from PIL import Image
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm




# Functions to compute different hash types
def compute_phash(image_path):
    image = Image.open(image_path)
    return str(imagehash.phash(image, hash_size=16))

def compute_dhash(image_path):
    image = Image.open(image_path)
    return str(imagehash.dhash(image, hash_size=16))

def compute_ahash(image_path):
    image = Image.open(image_path)
    return str(imagehash.average_hash(image, hash_size=16))

def compute_whash(image_path):
    image = Image.open(image_path)
    return str(imagehash.whash(image, hash_size=16))

# Directory containing images
image_dir = "../div2k-png-1-900"  # Replace with your actual directory

# Collecting all image file paths
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# List of hash functions to use
hash_functions = {
    'pHash': compute_phash,
    'dHash': compute_dhash,
    'aHash': compute_ahash,
    'wHash': compute_whash
}

# Colors for different hash functions
colors = {
    'pHash': 'blue',
    'dHash': 'orange',
    'aHash': 'green',
    'wHash': 'red'
}

# To store the mean and std for each hash function
hash_statistics = []

plt.figure(figsize=(10, 6))

# Loop over hash functions
for hash_name, hash_func in hash_functions.items():
    hashes = {}
    # Compute the hash for each image
    for image_file in sorted(image_files):
        image_path = os.path.join(image_dir, image_file)
        hash_value = hash_func(image_path)
        hashes[image_file] = hash_value

    # Compute normalized Hamming distances
    hash_size = 16 * 16  # Hash size for 16x16 bits
    normalized_hamming_distances = []
    for (img1, hash1), (img2, hash2) in itertools.combinations(hashes.items(), 2):
        bin_hash1 = ''.join(f'{int(c, 16):04b}' for c in hash1)
        bin_hash2 = ''.join(f'{int(c, 16):04b}' for c in hash2)
        distance = sum(c1 != c2 for c1, c2 in zip(bin_hash1, bin_hash2)) / hash_size
        normalized_hamming_distances.append(distance)

    # Calculate mean and std
    mean_distance = np.mean(normalized_hamming_distances)
    std_distance = np.std(normalized_hamming_distances)
    hash_statistics.append([hash_name, mean_distance, std_distance])

    # Plot normal distribution curve
    x = np.linspace(0, 1, 100)  # Set x-axis range between 0 and 1
    p = norm.pdf(x, mean_distance, std_distance)
    plt.plot(x, p, color=colors[hash_name], linewidth=1.5, label=f'{hash_name}')

# Add legend and labels
plt.title('Hash Length: 256 bits')
plt.xlabel('Normalized Hamming Distance')
plt.ylabel('Density')
plt.legend()

# Set the x-axis range between 0 and 1
plt.xlim(0, 1)

# Save hash statistics (mean and std) to CSV
df_statistics = pd.DataFrame(hash_statistics, columns=['Hash Function', 'Mean', 'STD'])
df_statistics.to_csv('hash_statistics.csv', index=False)

plt.show()
