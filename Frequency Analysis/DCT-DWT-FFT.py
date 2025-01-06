import os
import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct, fft2, ifft2, fftshift


def apply_dct(image):
    """
    Apply Discrete Cosine Transform to the image and return the transformed image.
    """
    dct_transformed = dct(dct(image.T, norm='ortho').T, norm='ortho')
    dct_transformed_log = np.log(np.abs(dct_transformed) + 1)  # Log transformation - more visually distinct DCT-img
    return dct_transformed_log


def apply_dwt(image):
    """
    Apply Discrete Wavelet Transform to the image and return the transformed image.
    """
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    # Reconstruct the image from the DWT coefficients for visualization
    dwt_transformed = np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))
    return dwt_transformed


def apply_fft(image):
    """
    Apply Fast Fourier Transform to the image and return the transformed image.
    """
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))
    return magnitude_spectrum


def normalize_image(image):
    """
    Normalize the image to 0-255 range for saving.
    """
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(image_normalized)


def save_transformed_image(image, output_path):
    """
    Save the transformed image to the specified output path.
    """
    cv2.imwrite(output_path, image)


def process_images(input_dir, output_dirs):
    """
    Process images in the input directory and save DCT, DWT, and FFT transformed images
    to the respective output directories.

    :param input_dir: Directory containing image files.
    :param output_dirs: Dictionary with output directories for DCT, DWT, and FFT.
    """
    for output_dir in output_dirs.values():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Process each image file in the directory
    for image_file in os.listdir(input_dir):
        if image_file.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            # Apply DCT
            dct_image = apply_dct(image)
            dct_image_normalized = normalize_image(dct_image)
            dct_output_path = os.path.join(output_dirs['dct'], f"dct_{image_file}")
            save_transformed_image(dct_image_normalized, dct_output_path)

            # Apply DWT
            dwt_image = apply_dwt(image)
            dwt_image_normalized = normalize_image(dwt_image)
            dwt_output_path = os.path.join(output_dirs['dwt'], f"dwt_{image_file}")
            save_transformed_image(dwt_image_normalized, dwt_output_path)

            # Apply FFT
            fft_image = apply_fft(image)
            fft_image_normalized = normalize_image(fft_image)
            fft_output_path = os.path.join(output_dirs['fft'], f"fft_{image_file}")
            save_transformed_image(fft_image_normalized, fft_output_path)


if __name__ == "__main__":
    input_directory = "div2k-801-900-jpeg"  # Replace with your input directory
    output_directories = {
        'dct': os.path.join(input_directory, "dct_transformed"),
        'dwt': os.path.join(input_directory, "dwt_transformed"),
        'fft': os.path.join(input_directory, "fft_transformed")
    }

    process_images(input_directory, output_directories)

    print(f"Transformed images saved to {output_directories}")
