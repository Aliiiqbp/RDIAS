from PIL import Image
import imagehash
import random
import os
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
import cv2
from trustmark import TrustMark


def hamming_distance(str1, str2):
    """Calculate the Hamming distance between two binary strings."""
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def compute_phash(image, hash_size):
    phash_string = str(imagehash.phash(image, hash_size=hash_size))
    binary_string = ''.join(format(int(nibble, 16), '04b') for nibble in phash_string)
    return binary_string

def embed_watermark(image, hash_value):
    tm = TrustMark(verbose=False, model_type='Q', use_ECC=False)
    watermarked_image = tm.encode(cover_image=image, string_secret=hash_value, MODE='binary')
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
    return psnr_value, ssim_value

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
            cover_image = Image.open(original_image_path)

            # Compute the hash of the entire image (400 bits)
            phash_full = compute_phash(cover_image, hash_size=20)

            # Split the image into four equal parts
            width, height = cover_image.size
            parts = {
                'top_left': cover_image.crop((0, 0, width // 2, height // 2)),
                'top_right': cover_image.crop((width // 2, 0, width, height // 2)),
                'bottom_left': cover_image.crop((0, height // 2, width // 2, height)),
                'bottom_right': cover_image.crop((width // 2, height // 2, width, height))
            }

            extracted_hash_parts = {}
            watermarked_parts = {}

            # Embed 100 bits of the full hash into each part
            for idx, (part_name, part_image) in enumerate(parts.items()):
                hash_part = phash_full[idx * 100:(idx + 1) * 100]  # Extract 100 bits
                watermarked_part = embed_watermark(part_image, hash_part)
                watermarked_parts[part_name] = watermarked_part

            # Reassemble the watermarked image
            new_image = Image.new('RGB', (width, height))
            new_image.paste(watermarked_parts['top_left'], (0, 0))
            new_image.paste(watermarked_parts['top_right'], (width // 2, 0))
            new_image.paste(watermarked_parts['bottom_left'], (0, height // 2))
            new_image.paste(watermarked_parts['bottom_right'], (width // 2, height // 2))

            # Compute the hash of the final watermarked image
            phash_watermarked = compute_phash(new_image, hash_size=20)

            # Save the final watermarked image
            watermarked_image_name = 'wm_' + filename
            watermarked_image_path = os.path.join(output_directory, watermarked_image_name)
            new_image.save(watermarked_image_path)

            psnr_value, ssim_value = calculate_metrics(cover_image, new_image)

            # Calculate the Hamming distance between the original and watermarked image hashes
            hash_distance = hamming_distance(phash_full, phash_watermarked)

            # Add data for CSV (total computed hash, total extracted hash, and hash distance)
            data.append([
                filename,                              # image_name
                watermarked_image_name,                # watermarked_image_name
                "pHash-400",                           # hash_function_name
                "TrustMark-Q",                         # watermarking_method_name
                phash_full,                            # computed full hash
                phash_watermarked,                     # hash of watermarked image
                hash_distance,                         # Hamming distance between computed and extracted hashes
                psnr_value,
                ssim_value
            ])

    # Save data to CSV
    columns = [
        'image_name', 'watermarked_image_name', 'hash_function_name', 'watermarking_method_name',
        'computed_hash', 'watermarked_hash', 'Hamming_distance', 'PSNR', 'SSIM'
    ]

    df = pd.DataFrame(data, columns=columns)
    df.sort_values(by='image_name', inplace=True)
    df.to_csv(csv_output, index=False)


if __name__ == "__main__":
    # Specify datasets
    Datasets = ["CLIC", "DIV2K"]  # Add your datasets here --- "CLIC"

    for data in Datasets:
        input_directory = '../Data/' + data
        output_directory = data + '-Immune'
        csv_output = output_directory + '/immunization.csv'
        main(input_directory, output_directory, csv_output)
