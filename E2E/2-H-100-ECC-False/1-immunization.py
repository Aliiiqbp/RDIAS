from PIL import Image
import imagehash
import random
from PIL import ImageFilter
from trustmark import TrustMark
import os
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from scipy.stats import wasserstein_distance
import numpy as np
import cv2


def compute_phash(image_path, hash_size):
    image = Image.open(image_path)
    phash_string = str(imagehash.phash(image, hash_size=hash_size))
    binary_string = ''.join(format(int(nibble, 16), '04b') for nibble in phash_string)
    return binary_string


def embed_watermark(image_path, hash_value):
    tm = TrustMark(verbose=False, model_type='Q', use_ECC=False)
    cover = Image.open(image_path).convert('RGB')
    watermarked_image = tm.encode(cover_image=cover, string_secret=hash_value, MODE='binary')
    return watermarked_image


def calculate_metrics(cover_image, watermarked_image):
    cover_array = np.array(cover_image)
    watermarked_array = np.array(watermarked_image)

    if len(cover_array.shape) == 3:  # Multichannel image
        channel_axis = -1
    else:  # Grayscale image
        channel_axis = None

    psnr_value = psnr(cover_array, watermarked_array)
    ssim_value = ssim(cover_array, watermarked_array, channel_axis=channel_axis)
    # wasserstein_value = wasserstein_distance(cover_array.flatten(), watermarked_array.flatten())

    return psnr_value, ssim_value  # , wasserstein_value


def hamming_distance(str1, str2):
    """Calculate the Hamming distance between two binary strings."""
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


def main(input_directory, output_directory, csv_output):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    data = []

    cnt = 0
    for filename in os.listdir(input_directory):
        cnt += 1
        print(str(cnt), filename)
        if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
            original_image_path = os.path.join(input_directory, filename)

            # Step 1: Immunization
            phash = compute_phash(image_path=original_image_path, hash_size=10)
            watermarked_image = embed_watermark(original_image_path, phash)

            # Save the watermarked image
            watermarked_image_name = 'wm_' + filename
            watermarked_image_path = os.path.join(output_directory, watermarked_image_name)
            watermarked_image.save(watermarked_image_path)

            # Calculate metrics
            cover_image = Image.open(original_image_path)
            psnr_value, ssim_value = calculate_metrics(cover_image, watermarked_image)

            # Compute the hash from the watermarked image
            watermarked_image_hash = compute_phash(image_path=watermarked_image_path, hash_size=10)

            # Compare hashes
            # hash_match = phash == watermarked_image_hash
            min_distance = hamming_distance(phash, watermarked_image_hash)

            # Add data to list
            data.append([
                filename,
                watermarked_image_name,
                "pHash-100",
                "TrustMark-Q",
                phash,
                watermarked_image_hash,  # hash computed from the watermarked image
                min_distance,
                # hash_match,
                psnr_value,
                ssim_value,
                # wasserstein_value
            ])

    # Save data to CSV
    columns = ['image_name', 'watermarked_image_name', 'hash_function_name', 'watermarking_method_name', 'hash_value',
               'watermarked_image_hash', 'min_distance', 'PSNR', 'SSIM']
    df = pd.DataFrame(data, columns=columns)
    df = df.astype({
        'hash_value': str,
        'watermarked_image_hash': str
    })

    df.sort_values(by='image_name', inplace=True)
    df.to_csv(csv_output, index=False)


if __name__ == "__main__":
    # "CLIC"
    Datasets = ["CLIC", "DIV2K"]  # "MetFace"

    for data in Datasets:
        input_directory = '../Data/' + data
        output_directory = data + '-Immune'
        csv_output = output_directory + '/immunization.csv'
        main(input_directory, output_directory, csv_output)
