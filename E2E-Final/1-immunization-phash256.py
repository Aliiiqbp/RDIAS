from PIL import Image
import imagehash
import random
import os
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
import cv2
from trustmark import TrustMark
import torch
import lpips
import time

from lpips import LPIPS
lpips_net = LPIPS(net='alex')
tm = TrustMark(verbose=False, model_type='Q', use_ECC=True, encoding_type=TrustMark.Encoding.BCH_4)


def hamming_distance(str1, str2):
    """Calculate the Hamming distance between two binary strings."""
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def compute_phash(image, hash_size):
    phash_string = str(imagehash.phash(image, hash_size=hash_size))
    binary_string = ''.join(format(int(nibble, 16), '04b') for nibble in phash_string)
    return binary_string

def embed_watermark(image, hash_value):

    # tm = TrustMark(verbose=False, model_type='Q', use_ECC=True, encoding_type=TrustMark.Encoding.BCH_4)
    watermarked_image, ecc_time, embed_time = tm.encode(cover_image=image, string_secret=hash_value, MODE='binary')
    return watermarked_image, ecc_time, embed_time

def calculate_metrics(cover_image, watermarked_image):
    cover_array = np.array(cover_image)
    watermarked_array = np.array(watermarked_image)

    if len(cover_array.shape) == 3:  # Multichannel image
        channel_axis = -1
    else:  # Grayscale image
        channel_axis = None

    psnr_value = psnr(cover_array, watermarked_array)
    ssim_value = ssim(cover_array, watermarked_array, channel_axis=channel_axis)

    original_tensor = torch.tensor(cover_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    watermarked_tensor = torch.tensor(watermarked_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # Calculate LPIPS
    lpips_value = lpips_net(original_tensor, watermarked_tensor).item()


    return psnr_value, ssim_value, lpips_value

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
            cover_file_size = os.path.getsize(original_image_path)

            # TODO: time immunization start
            start_time = time.time()

            cover_image = Image.open(original_image_path)

            # Compute the hash of the entire image (256 bits)
            phash_full = compute_phash(cover_image, hash_size=16)

            hash_time = time.time()
            wm_time_start = time.time()

            # Split the image into four equal parts
            width, height = cover_image.size
            parts = {
                'top_left': cover_image.crop((0, 0, width // 2, height // 2)),
                'top_right': cover_image.crop((width // 2, 0, width, height // 2)),
                'bottom_left': cover_image.crop((0, height // 2, width // 2, height)),
                'bottom_right': cover_image.crop((width // 2, height // 2, width, height))
            }

            watermarked_parts = {}

            wm_times = []
            ecc_times = []
            # Embed 64 bits of the full hash into each part
            for idx, (part_name, part_image) in enumerate(parts.items()):
                hash_part = phash_full[idx * 64:(idx + 1) * 64]  # Extract 64 bits

                watermarked_part, ecc_time, embed_time = embed_watermark(part_image, hash_part)

                watermarked_parts[part_name] = watermarked_part
                ecc_times.append(ecc_time)
                wm_times.append(embed_time)

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


            wm_time_end = time.time()
            end_time = time.time()
            # TODO: time immunization end
            immuned_file_size = os.path.getsize(watermarked_image_path)


            file_size_change = immuned_file_size - cover_file_size
            ECC_sum = sum(ecc_times)
            wm_sum = (wm_time_end - wm_time_start) - ECC_sum
            E2E_time = end_time - start_time
            Fingerptinting_time = hash_time - start_time
            phash_watermarked = compute_phash(new_image, hash_size=16)
            psnr_value, ssim_value, lpips_value = calculate_metrics(cover_image, new_image)

            # Calculate the Hamming distance between the original and watermarked image hashes
            hash_distance = hamming_distance(phash_full, phash_watermarked)

            # Add data for CSV (total computed hash, total extracted hash, and hash distance)
            data.append([
                filename,                              # image_name
                watermarked_image_name,                # watermarked_image_name
                "pHash-256",                           # hash_function_name
                "TrustMark-Q",                         # watermarking_method_name
                phash_full,                            # computed full hash
                phash_watermarked,                     # hash of watermarked image
                hash_distance,                         # Hamming distance between computed and extracted hashes
                psnr_value,
                ssim_value,
                lpips_value,
                E2E_time,
                Fingerptinting_time,
                ECC_sum,
                wm_sum,
                cover_file_size,
                file_size_change
            ])

    # Save data to CSV
    columns = [
        'image_name', 'watermarked_image_name', 'hash_function_name', 'watermarking_method_name',
        'computed_hash', 'watermarked_hash', 'Hamming_distance', 'PSNR', 'SSIM', 'LPIPS', "E2E_time",
        "Fingerptinting_time", "ECC_time", "WM_time", "Cover File Size", "Size Change"
    ]

    df = pd.DataFrame(data, columns=columns)
    df.sort_values(by='image_name', inplace=True)
    df.to_csv(csv_output, index=False)


if __name__ == "__main__":
    # Specify datasets
    Datasets = ["DIV2K+CLIC"]  # Add your datasets here --- "CLIC" "CLIC", "DIV2K", "FFHQ1000"

    for data in Datasets:
        input_directory = '../Data/' + data
        output_directory = data + '-Immune'
        csv_output = output_directory + '/immunization.csv'
        main(input_directory, output_directory, csv_output)
