import os
import imagehash
from PIL import Image
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Function to compute perceptual hash
def compute_phash(image_path):
    image = Image.open(image_path)
    return str(imagehash.phash(image, hash_size=16))

# Directory containing images
image_dir = "div2k-png-1-900"  # Replace with your actual directory

# Collecting all image file paths
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Compute the perceptual hash for each image
hashes = {}
for image_file in sorted(image_files):
    image_path = os.path.join(image_dir, image_file)
    hash_value = compute_phash(image_path)
    hashes[image_file] = hash_value

# Save the hash values into a CSV file
df_hashes = pd.DataFrame(list(hashes.items()), columns=['Image', 'pHash'])
df_hashes.to_csv('image_hashes.csv', index=False)

# Compute the normalized Hamming distances between each pair of images
hash_size = 16 * 16  # Since phash with hash_size=16 means 256 bits
normalized_hamming_distances = []
for (img1, hash1), (img2, hash2) in itertools.combinations(hashes.items(), 2):
    # Convert the hexadecimal hash to binary
    bin_hash1 = ''.join(f'{int(c, 16):04b}' for c in hash1)
    bin_hash2 = ''.join(f'{int(c, 16):04b}' for c in hash2)

    # Calculate the normalized Hamming distance by comparing the bits
    distance = sum(c1 != c2 for c1, c2 in zip(bin_hash1, bin_hash2)) / hash_size
    normalized_hamming_distances.append(distance)

# Plot the distribution of the normalized Hamming distances
plt.figure(figsize=(10, 6))
plt.hist(normalized_hamming_distances, bins=30, color='blue', alpha=0.7, edgecolor='black', density=True)

# Calculate mean and std for normalized distances
mean_distance = np.mean(normalized_hamming_distances)
std_distance = np.std(normalized_hamming_distances)

# Plot the normal distribution curve with the computed mean and std
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_distance, std_distance)
plt.plot(x, p, 'k', linewidth=2)

# Add vertical lines for mean and standard deviation
plt.axvline(mean_distance, color='red', linestyle='--', label=f'Mean: {mean_distance:.4f}')
plt.axvline(mean_distance - std_distance, color='green', linestyle='--', label=f'STD: {std_distance:.4f}')
plt.axvline(mean_distance + std_distance, color='green', linestyle='--')

plt.title('Algorithm: pHash\nSize: 256 bits')
plt.xlabel('Normalized Hamming Distance')
plt.ylabel('Density')
plt.legend()

plt.show()
