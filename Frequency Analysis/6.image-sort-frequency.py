import os
import cv2
import numpy as np
import pandas as pd
from scipy.fftpack import fft2, fftshift

def calculate_frequency_energies(image_path):
    """
    Calculate the high-frequency and low-frequency energies of an image.

    :param image_path: Path to the image file.
    :return: High-frequency and low-frequency energies of the image.
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Apply Fourier Transform
    f_transform = fft2(img)
    f_transform_shifted = fftshift(f_transform)

    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)

    # Define the high-frequency region (e.g., outer 75% of the spectrum)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    high_freq_region = magnitude_spectrum.copy()
    low_freq_region = magnitude_spectrum.copy()

    # Zero out the low-frequency region in high_freq_region
    high_freq_region[crow-rows//4:crow+rows//4, ccol-cols//4:ccol+cols//4] = 0
    # Zero out the high-frequency region in low_freq_region
    low_freq_region[:crow-rows//4, :] = 0
    low_freq_region[crow+rows//4:, :] = 0
    low_freq_region[:, :ccol-cols//4] = 0
    low_freq_region[:, ccol+cols//4:] = 0

    # Calculate the high-frequency and low-frequency energies
    high_frequency_energy = np.sum(high_freq_region)
    low_frequency_energy = np.sum(low_freq_region)

    return high_frequency_energy, low_frequency_energy

def process_images(input_dir, output_dir):
    """
    Process images in the input directory to sort them based on frequency energies
    and store the sorted results in separate CSV files.

    :param input_dir: Directory containing image files.
    :param output_dir: Directory to store the output CSV files.
    """
    image_data = []

    # Process each image file in the directory
    for image_file in os.listdir(input_dir):
        if image_file.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_dir, image_file)
            try:
                high_frequency_energy, low_frequency_energy = calculate_frequency_energies(image_path)
                energy_difference = high_frequency_energy - low_frequency_energy
                image_data.append({
                    "Image Name": image_file,
                    "High Frequency Energy": high_frequency_energy,
                    "Low Frequency Energy": low_frequency_energy,
                    "Energy Difference": energy_difference
                })
            except FileNotFoundError:
                continue

    # Create a DataFrame from the image data
    image_df = pd.DataFrame(image_data)

    # Sort by High Frequency Energy and save
    sorted_by_high_freq_df = image_df.sort_values(by="High Frequency Energy", ascending=False).reset_index(drop=True)
    sorted_by_high_freq_df.to_csv(os.path.join(output_dir, "sorted_by_high_frequency.csv"), index=False)

    # Sort by Low Frequency Energy and save
    sorted_by_low_freq_df = image_df.sort_values(by="Low Frequency Energy", ascending=False).reset_index(drop=True)
    sorted_by_low_freq_df.to_csv(os.path.join(output_dir, "sorted_by_low_frequency.csv"), index=False)

    # Sort by Energy Difference and save
    sorted_by_energy_diff_df = image_df.sort_values(by="Energy Difference", ascending=False).reset_index(drop=True)
    sorted_by_energy_diff_df.to_csv(os.path.join(output_dir, "sorted_by_energy_difference.csv"), index=False)

if __name__ == "__main__":
    input_directory = "div2k-801-900-jpeg"  # Replace with your input directory
    output_directory = "div2k-801-900-jpeg"  # Replace with your output directory

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    process_images(input_directory, output_directory)

    print(f"Images sorted by frequency energies and saved to {output_directory}")
