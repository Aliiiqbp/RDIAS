import numpy as np
import cv2
from PIL.ImageOps import cover
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import random
import os
from trustmark import TrustMark
import csv
import torch
import imagehash

# LPIPS setup
from lpips import LPIPS
lpips_net = LPIPS(net='alex')


def generate_random_watermark(length):
    return ''.join(str(random.choice([0, 1])) for _ in range(length))

def hamming_distance(hash1, hash2):
    """Compute the Hamming distance between two perceptual hashes in hexadecimal format."""
    # Convert the hexadecimal hash to binary
    bin_hash1 = ''.join(f'{int(c, 16):04b}' for c in str(hash1))
    bin_hash2 = ''.join(f'{int(c, 16):04b}' for c in str(hash2))

    # Compute the Hamming distance by comparing the binary strings
    return sum(b1 != b2 for b1, b2 in zip(bin_hash1, bin_hash2))


def image_hashes(image, wm_image):
    h1 = imagehash.average_hash(image, hash_size=16)
    h2 = imagehash.average_hash(wm_image, hash_size=16)
    H1 = hamming_distance(h1, h2)

    h1 = imagehash.dhash(image, hash_size=16)
    h2 = imagehash.dhash(wm_image, hash_size=16)
    H2 = hamming_distance(h1, h2)

    h1 = imagehash.whash(image, hash_size=16)
    h2 = imagehash.whash(wm_image, hash_size=16)
    H3 = hamming_distance(h1, h2)

    h1 = imagehash.phash(image, hash_size=16)
    h2 = imagehash.phash(wm_image, hash_size=16)
    H4 = hamming_distance(h1, h2)

    return [H1, H2, H3, H4]

# TODO 1 START: watermark encoder ############################################## 100 bits
def encode_watermark(image, watermark):
    cover = image.convert('RGB')
    wmimg = tm.encode(cover,
              watermark,
              MODE='binary')
    return wmimg
# TODO 1 END: watermark encoder ################################################

# TODO 1 START: watermark encoder ############################################## 400 bits
# def encode_watermark(image, watermark):
#     # Ensure the image is in RGB mode
#     cover = image.convert('RGB')
#
#     # Get dimensions of the cover image
#     width, height = cover.size
#     mid_width = width // 2
#     mid_height = height // 2
#
#     # Split the cover image into four quadrants
#     cover_tl = cover.crop((0, 0, mid_width, mid_height))            # Top-left
#     cover_tr = cover.crop((mid_width, 0, width, mid_height))        # Top-right
#     cover_bl = cover.crop((0, mid_height, mid_width, height))       # Bottom-left
#     cover_br = cover.crop((mid_width, mid_height, width, height))   # Bottom-right
#
#     # Split the watermark string into four equal parts
#     total_length = len(watermark)
#     part_length = total_length // 4
#
#     wm_part1 = watermark[0:part_length]
#     wm_part2 = watermark[part_length:2*part_length]
#     wm_part3 = watermark[2*part_length:3*part_length]
#     wm_part4 = watermark[3*part_length:]
#
#     # Encode each watermark part into each cover quadrant
#     wm_cover_tl = tm.encode(cover_tl, wm_part1, MODE='binary')
#     wm_cover_tr = tm.encode(cover_tr, wm_part2, MODE='binary')
#     wm_cover_bl = tm.encode(cover_bl, wm_part3, MODE='binary')
#     wm_cover_br = tm.encode(cover_br, wm_part4, MODE='binary')
#
#     # Create a new image to hold the combined quadrants
#     watermarked_image = Image.new('RGB', (width, height))
#
#     # Paste the watermarked quadrants back into the new image
#     watermarked_image.paste(wm_cover_tl, (0, 0))
#     watermarked_image.paste(wm_cover_tr, (mid_width, 0))
#     watermarked_image.paste(wm_cover_bl, (0, mid_height))
#     watermarked_image.paste(wm_cover_br, (mid_width, mid_height))
#
#     return watermarked_image
# TODO 1 END: watermark encoder ################################################


# TODO 2 START: watermark decoder ############################################## 100 bit
def decode_watermark(image):
    cover = image.convert('RGB')
    wm_secret, wm_present, wm_schema, _ = tm.decode(cover)
    if wm_present:
        return wm_secret
    else:
        return 'watermark extraction failed!'
# TODO 2 END: watermark encoder ################################################

# TODO 2 START: watermark decoder ############################################## 400 bit
# def decode_watermark(image):
#     # Ensure the image is in RGB mode
#     cover = image.convert('RGB')
#
#     # Get dimensions of the cover image
#     width, height = cover.size
#     mid_width = width // 2
#     mid_height = height // 2
#
#     # Split the cover image into four quadrants
#     cover_tl = cover.crop((0, 0, mid_width, mid_height))            # Top-left
#     cover_tr = cover.crop((mid_width, 0, width, mid_height))        # Top-right
#     cover_bl = cover.crop((0, mid_height, mid_width, height))       # Bottom-left
#     cover_br = cover.crop((mid_width, mid_height, width, height))   # Bottom-right
#
#     # Decode the watermark from each quadrant
#     wm_part1, wm_present1, wm_schema1, _ = tm.decode(cover_tl)
#     wm_part2, wm_present2, wm_schema2, _ = tm.decode(cover_tr)
#     wm_part3, wm_present3, wm_schema3, _ = tm.decode(cover_bl)
#     wm_part4, wm_present4, wm_schema4, _ = tm.decode(cover_br)
#
#     # Check if watermark is present in all quadrants
#     if wm_present1 and wm_present2 and wm_present3 and wm_present4:
#         # Concatenate the watermark parts
#         wm_secret = wm_part1 + wm_part2 + wm_part3 + wm_part4
#         return wm_secret
#     else:
#         return 'watermark extraction failed!'
# TODO 2 END: watermark encoder ################################################

def apply_jpeg_compression(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return Image.open(buffer)

def apply_webp_compression(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=quality)
    return Image.open(buffer)

def apply_jpeg2000_compression(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG2000", quality=quality)
    return Image.open(buffer)

def apply_resize(image, scale):
    return image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)

def apply_gaussian_noise(image, sigma):
    noisy_image = np.array(image)
    noise = np.random.normal(0, sigma, noisy_image.shape)
    noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def apply_saturation(image, level):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(level)

def apply_brightness(image, level):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(level)

def apply_contrast(image, level):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(level)

def apply_sharpness(image, level):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(level)

def apply_gaussian_blur(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius))

def apply_median_blur(image, size):
    return image.filter(ImageFilter.MedianFilter(size))

def apply_average_filter(image, size):
    return image.filter(ImageFilter.BoxBlur(size))


def calculate_metrics(original_image, watermarked_image):
    original = np.array(original_image)
    watermarked = np.array(watermarked_image)
    psnr_value = psnr(original, watermarked)
    ssim_value = ssim(original, watermarked, multichannel=True, win_size=3)

    # Convert numpy arrays to PyTorch tensors for LPIPS
    original_tensor = torch.tensor(original).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    watermarked_tensor = torch.tensor(watermarked).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # Calculate LPIPS
    lpips_value = lpips_net(original_tensor, watermarked_tensor).item()

    return psnr_value, ssim_value, lpips_value

transformations = {
    "JPEG": [50, 70, 90],
    # "WebP": [50, 70, 90],

    # "Resize": [0.5, 0.75, 1.5],

    "GaussianNoise": [0.02, 0.04, 0.08],

    # "Saturation": [0.5, 1.5, 2.0],
    # "Brightness": [0.5, 1.5, 2.0],
    # "Contrast": [0.5, 1.5, 2.0],
    # "Sharpness": [0.5, 1.5, 2.0],

    "GaussianBlur": [1, 3, 5],
    # "MedianBlur": [1, 3, 5],
    # "AverageFilter": [1, 3, 5]

    ########## JUST TO TEST IF ALL TRANSFORMATIONS WORKS PROPERLY ###########
    # "JPEG": [50],
    # "Resize": [0.25],
    # "GaussianNoise": [10],
    # "WebP": [50],
    # "JPEG2000": [50],
    # "Saturation": [0.5],
    # "Brightness": [0.5],
    # "Contrast": [0.5],
    # "Sharpness": [0.5]
}

def apply_transformation(image, transformation_name, param):
    if transformation_name == "JPEG":
        return apply_jpeg_compression(image, param)
    elif transformation_name == "Resize":
        return apply_resize(image, param)
    elif transformation_name == "GaussianNoise":
        return apply_gaussian_noise(image, param)
    elif transformation_name == "WebP":
        return apply_webp_compression(image, param)
    elif transformation_name == "JPEG2000":
        return apply_jpeg2000_compression(image, param)
    elif transformation_name == "Saturation":
        return apply_saturation(image, param)
    elif transformation_name == "Brightness":
        return apply_brightness(image, param)
    elif transformation_name == "Contrast":
        return apply_contrast(image, param)
    elif transformation_name == "GaussianBlur":
        return apply_gaussian_blur(image, param)
    elif transformation_name == "MedianBlur":
        return apply_median_blur(image, param)
    elif transformation_name == "AverageFilter":
        return apply_average_filter(image, param)
    elif transformation_name == "Sharpness":
        return apply_sharpness(image, param)
    return image

def process_images_in_directory(directory_path, wm_size):
    results = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image = Image.open(image_path)

            watermark64 = generate_random_watermark(wm_size)
            watermarked_image = encode_watermark(image, watermark64)

            hashes = image_hashes(image, watermarked_image)
            metrics = calculate_metrics(image, watermarked_image)
            print("###########################")
            print(filename, "- (psnr, ssim, lpips):", metrics)

            for trans_name, params in transformations.items():
                for param in params:
                    transformed_image = apply_transformation(watermarked_image, trans_name, param)
                    decoded_watermark = decode_watermark(transformed_image)

                    flip_0_to_1, flip_1_to_0, error_burst = 0, 0, 0
                    if decoded_watermark != 'watermark extraction failed!':
                        bit_accuracy = sum(a == b for a, b in zip(watermark64, decoded_watermark)) / len(watermark64)
                        error_indexes = [i for i, (a, b) in enumerate(zip(watermark64, decoded_watermark)) if a != b]
                        if len(error_indexes) != 0:
                            prev_index = None
                            for idx in error_indexes:
                                # Start a new burst if this is the first index or not consecutive
                                if prev_index is None or idx != prev_index + 1:
                                    error_burst += 1
                                prev_index = idx

                        # New code to count bit flips
                        flip_0_to_1 = sum(1 for a, b in zip(watermark64, decoded_watermark) if a == '0' and b == '1')
                        flip_1_to_0 = sum(1 for a, b in zip(watermark64, decoded_watermark) if a == '1' and b == '0')

                    else:
                        bit_accuracy = 50
                        error_indexes = []

                    print("Trans:",trans_name, "- Param:",param, "- bit acc:", bit_accuracy)
                    results.append({
                        'Image': filename,
                        'Transformation': trans_name,
                        'Parameter': param,
                        'Bit Accuracy': bit_accuracy,
                        'Error Indexes': ', '.join(map(str, error_indexes)),
                        'PSNR': metrics[0],
                        'SSIM': metrics[1],
                        'LPIPS': metrics[2],
                        'aHash': hashes[0],
                        'dHash': hashes[1],
                        'wHash': hashes[2],
                        'pHash': hashes[3],
                        '0-to-1': flip_0_to_1,
                        '1-to-0': flip_1_to_0,
                        'burst_error_count': error_burst,
                        'bit_error_count': len(error_indexes),
                        'burst_index': error_burst / len(error_indexes) if error_burst != 0 else None
                    })

    with open(directory_path + '_test_watermarking_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Image', 'Transformation', 'Parameter', 'Bit Accuracy', 'Error Indexes', 'PSNR', 'SSIM', 'LPIPS',
                      'aHash', 'dHash', 'wHash', 'pHash', '0-to-1', '1-to-0', 'burst_error_count', 'bit_error_count', 'burst_index']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


if __name__ == "__main__":

    # TODO 3 START: watermark Algorithm + Image Directory + wm size ############################################
    tm = TrustMark(verbose=False, use_ECC=False, secret_len=100, model_type='Q')
    for directory_path in ["Amin"]:  # , "CLIC-686", "DIV2K-900", "MetFace-1336"
        process_images_in_directory(directory_path, wm_size=100)
    # TODO 3 END: watermark Algorithm + Image Directory + wm size ##############################################
