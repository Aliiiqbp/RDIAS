import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path, grayscale=True):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) if grayscale else cv2.imread(image_path)


def spatial_domain_difference(image1, image2, alpha):
    difference = image1.astype(np.float32) - image2.astype(np.float32)
    magnified_difference = alpha * difference
    return np.clip(magnified_difference, 0, 255).astype(np.uint8)


def frequency_domain_difference(image1, image2, alpha):
    f_image1 = np.fft.fft2(image1)
    f_image2 = np.fft.fft2(image2)
    f_difference = f_image1 - f_image2
    f_magnified_difference = alpha * f_difference
    spatial_difference = np.fft.ifft2(f_magnified_difference)
    return np.abs(spatial_difference)


def dct_difference(image1, image2, alpha):
    d_image1 = cv2.dct(image1.astype(np.float32))
    d_image2 = cv2.dct(image2.astype(np.float32))
    d_difference = d_image1 - d_image2
    d_magnified_difference = alpha * d_difference
    spatial_difference = cv2.idct(d_magnified_difference)
    return np.abs(spatial_difference)


def compute_fft(image):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    return magnitude_spectrum


def compute_dct(image):
    d_transform = cv2.dct(image.astype(np.float32))
    magnitude_spectrum = 20 * np.log(np.abs(d_transform))
    return magnitude_spectrum


def plot_images(original1, original2, spatial_diff, frequency_diff, dct_diff, f_image1, f_image2, d_image1, d_image2):
    plt.figure(figsize=(15, 12))

    plt.subplot(3, 3, 1)
    plt.title('Original Image 1')
    plt.imshow(original1, cmap='gray')

    plt.subplot(3, 3, 2)
    plt.title('Original Image 2')
    plt.imshow(original2, cmap='gray')

    plt.subplot(3, 3, 3)
    plt.title('Spatial Domain Difference')
    plt.imshow(spatial_diff, cmap='gray')

    plt.subplot(3, 3, 4)
    plt.title('Frequency Domain of Image 1 (FFT)')
    plt.imshow(f_image1, cmap='gray')

    plt.subplot(3, 3, 5)
    plt.title('Frequency Domain of Image 2 (FFT)')
    plt.imshow(f_image2, cmap='gray')

    plt.subplot(3, 3, 6)
    plt.title('Frequency Domain Difference (FFT)')
    plt.imshow(frequency_diff, cmap='gray')

    plt.subplot(3, 3, 7)
    plt.title('Frequency Domain of Image 1 (DCT)')
    plt.imshow(d_image1, cmap='gray')

    plt.subplot(3, 3, 8)
    plt.title('Frequency Domain of Image 2 (DCT)')
    plt.imshow(d_image2, cmap='gray')

    plt.subplot(3, 3, 9)
    plt.title('Frequency Domain Difference (DCT)')
    plt.imshow(dct_diff, cmap='gray')

    plt.tight_layout()
    plt.show()


# Paths to the images
image_path1 = 'div2k-801-900-jpeg/0830.jpeg'
image_path2 = 'div2k-801-900-jpeg-immune-images-phash-Q/wm_0830.jpeg'

# Load images
image1 = load_image(image_path1)
image2 = load_image(image_path2)

# Define the magnification factor
alpha = 10.0

# Compute differences
spatial_diff = spatial_domain_difference(image1, image2, alpha)
frequency_diff = frequency_domain_difference(image1, image2, alpha)
dct_diff = dct_difference(image1, image2, alpha)

# Compute FFTs for original images
f_image1 = compute_fft(image1)
f_image2 = compute_fft(image2)

# Compute DCTs for original images
d_image1 = compute_dct(image1)
d_image2 = compute_dct(image2)

# Plot the results
plot_images(image1, image2, spatial_diff, frequency_diff, dct_diff, f_image1, f_image2, d_image1, d_image2)
