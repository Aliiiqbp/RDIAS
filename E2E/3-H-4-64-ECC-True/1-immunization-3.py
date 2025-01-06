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
    tm = TrustMark(verbose=False, model_type='Q', use_ECC=True, encoding_type=TrustMark.Encoding.BCH_4)
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

            # Split the image into four equal parts
            width, height = cover_image.size
            parts = {
                'top_left': cover_image.crop((0, 0, width // 2, height // 2)),
                'top_right': cover_image.crop((width // 2, 0, width, height // 2)),
                'bottom_left': cover_image.crop((0, height // 2, width // 2, height)),
                'bottom_right': cover_image.crop((width // 2, height // 2, width, height))
            }

            part_hashes_before = {}
            part_hashes_after = {}
            distances = {}
            watermarked_parts = {}

            # Process each part separately
            for part_name, part_image in parts.items():
                # Compute hash before watermarking
                phash_before = compute_phash(part_image, hash_size=8)
                part_hashes_before[part_name] = phash_before

                # Embed watermark in part and compute hash after
                watermarked_part = embed_watermark(part_image, phash_before)
                watermarked_parts[part_name] = watermarked_part
                phash_after = compute_phash(watermarked_part, hash_size=8)
                part_hashes_after[part_name] = phash_after

                # Calculate Hamming distance between the original and watermarked part
                distances[part_name] = hamming_distance(phash_before, phash_after)

            # Reassemble the watermarked image
            new_image = Image.new('RGB', (width, height))
            new_image.paste(watermarked_parts['top_left'], (0, 0))
            new_image.paste(watermarked_parts['top_right'], (width // 2, 0))
            new_image.paste(watermarked_parts['bottom_left'], (0, height // 2))
            new_image.paste(watermarked_parts['bottom_right'], (width // 2, height // 2))

            # Save the final watermarked image
            watermarked_image_name = 'wm_' + filename
            watermarked_image_path = os.path.join(output_directory, watermarked_image_name)
            new_image.save(watermarked_image_path)

            psnr_value, ssim_value = calculate_metrics(cover_image, new_image)

            # Add data for CSV (each part's before/after hashes and Hamming distances)
            data.append([
                filename,                              # image_name
                watermarked_image_name,                # watermarked_image_name
                "pHash-4*64",                            # hash_function_name
                "TrustMark-Q",                         # watermarking_method_name

                part_hashes_before['top_left'],        # pHash top-left before watermarking
                part_hashes_before['top_right'],       # pHash top-right before watermarking
                part_hashes_before['bottom_left'],     # pHash bottom-left before watermarking
                part_hashes_before['bottom_right'],    # pHash bottom-right before watermarking

                part_hashes_after['top_left'],         # pHash top-left after watermarking
                part_hashes_after['top_right'],        # pHash top-right after watermarking
                part_hashes_after['bottom_left'],      # pHash bottom-left after watermarking
                part_hashes_after['bottom_right'],     # pHash bottom-right after watermarking

                distances['top_left'],                 # Hamming distance for top-left
                distances['top_right'],                # Hamming distance for top-right
                distances['bottom_left'],              # Hamming distance for bottom-left
                distances['bottom_right'],             # Hamming distance for bottom-right
                psnr_value,
                ssim_value
            ])

    # Save data to CSV
    columns = [
        'image_name', 'watermarked_image_name', 'hash_function_name', 'watermarking_method_name',
        'pHash_top_left_before', 'pHash_top_right_before', 'pHash_bottom_left_before', 'pHash_bottom_right_before',
        'pHash_top_left_after', 'pHash_top_right_after', 'pHash_bottom_left_after', 'pHash_bottom_right_after',
        'Hamming_distance_top_left', 'Hamming_distance_top_right', 'Hamming_distance_bottom_left', 'Hamming_distance_bottom_right',
        'PSNR', 'SSIM'
    ]

    df = pd.DataFrame(data, columns=columns)
    df.sort_values(by='image_name', inplace=True)
    df.to_csv(csv_output, index=False)


if __name__ == "__main__":
    # Specify datasets
    Datasets = ["DIV2K"]  # Add your datasets here --- "CLIC"

    for data in Datasets:
        input_directory = '../Data/' + data
        output_directory = data + '-Immune'
        csv_output = output_directory + '/immunization.csv'
        main(input_directory, output_directory, csv_output)
