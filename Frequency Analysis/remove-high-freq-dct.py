import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path, grayscale=True):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) if grayscale else cv2.imread(image_path)


def compute_dct(image):
    return cv2.dct(image.astype(np.float32))


def inverse_dct(dct_image):
    return cv2.idct(dct_image)


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

# Load image
image = load_image(image_path)

# Compute DCT
dct_image = compute_dct(image)

# Define the removal ranges
rm_range = [50, 75, 90, 95, 99]

# Compute the images with high frequencies removed
results = {}
for percent in rm_range:
    dct_modified = remove_high_freq_dct(dct_image, percent)
    spatial_image = inverse_dct(dct_modified)
    results[percent] = (dct_modified, np.clip(spatial_image, 0, 255))

# Plot the results
plot_images(image, dct_image, results, rm_range)
