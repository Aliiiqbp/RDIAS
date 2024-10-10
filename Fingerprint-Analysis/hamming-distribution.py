import os
import imagehash
from PIL import Image
import pandas as pd
import itertools
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
import numpy as np


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
    hash_value = compute_phash(image_path)
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

# hash_size = 64
# normalized_distances = [distance / hash_size for distance in hamming_distances]
# bins = [i / 10.0 for i in range(11)]

# min_distance = min(hamming_distances)
# max_distance = max(hamming_distances)

# Set the bins to be a range that includes every integer from min_distance to max_distance
# bins = range(min_distance, max_distance + 1)

# Plot the distribution of the Hamming distances
# plt.hist(hamming_distances, bins=bins, color='blue', alpha=0.7, edgecolor='black')
# plt.xticks(bins, rotation=90)
# plt.title('Distribution of Hamming Distances')
# plt.xlabel('Hamming Distance')
# plt.ylabel('Frequency')
# plt.show()
# #

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
