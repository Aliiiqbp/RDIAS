import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path, grayscale=True):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) if grayscale else cv2.imread(image_path)


def compute_fft(image):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    return f_shift, magnitude_spectrum


def inverse_fft(f_transform):
    f_ishift = np.fft.ifftshift(f_transform)
    image_back = np.fft.ifft2(f_ishift)
    return np.abs(image_back)


def remove_high_freq(f_transform, percent):
    f_copy = np.copy(f_transform)
    height, width = f_copy.shape
    center_x, center_y = width // 2, height // 2

    # Calculate the cutoff radius
    radius = int(np.sqrt(center_x ** 2 + center_y ** 2) * (percent / 100.0))

    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if distance > radius:
                f_copy[y, x] = 0

    return f_copy


def plot_images(original, fft_original, results, rm_range):
    num_levels = len(results)
    plt.figure(figsize=(15, 5 * (num_levels + 1)))

    plt.subplot(num_levels + 1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')

    plt.subplot(num_levels + 1, 2, 2)
    plt.title('FFT of Original Image')
    plt.imshow(fft_original, cmap='gray')

    for i, percent in enumerate(rm_range):
        fft_image, spatial_image = results[percent]

        plt.subplot(num_levels + 1, 2, 2 * i + 3)
        plt.title(f'{percent}% High Frequencies Removed (FFT)')
        plt.imshow(np.log(np.abs(fft_image) + 1), cmap='gray')

        plt.subplot(num_levels + 1, 2, 2 * i + 4)
        plt.title(f'{percent}% High Frequencies Removed (Spatial)')
        plt.imshow(spatial_image, cmap='gray')

    plt.tight_layout()
    plt.show()


# Paths to the images
image_path = '0801.jpeg'

# Load image
image = load_image(image_path)

# Compute FFT
fft_image, magnitude_spectrum = compute_fft(image)

# Define the removal ranges
rm_range = [1, 5, 15, 90]

# Compute the images with high frequencies removed
results = {}
for percent in rm_range:
    fft_modified = remove_high_freq(fft_image, percent)
    spatial_image = inverse_fft(fft_modified)
    results[percent] = (fft_modified, np.clip(spatial_image, 0, 255))

# Plot the results
plot_images(image, magnitude_spectrum, results, rm_range)
