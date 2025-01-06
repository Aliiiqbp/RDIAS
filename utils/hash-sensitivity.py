import os
import imagehash
import pandas as pd
from PIL import Image
from transformations import apply_transformations  # Ensure this matches the name of your transformations script
import numpy as np
import os
import random
import csv
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
import re
import cv2
# import pdqhash


def jpeg_compression(image, quality):
    """Apply JPEG compression."""
    image.save("temp.jpg", "JPEG", quality=quality)
    return Image.open("temp.jpg")


def resize_keep_ratio(image, scale):
    """Resize image keeping the aspect ratio."""
    new_size = (int(image.width * scale), int(image.height * scale))
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    return resized_image


def resize_no_ratio(image, size):
    """Resize image without keeping the aspect ratio."""
    return image.resize(size)


def add_gaussian_noise(image, mean, std):
    """Add Gaussian noise to the image."""
    np_image = np.array(image)
    noisy_img = np_image + np.random.normal(mean, std, np_image.shape)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def apply_gaussian_blur(image, radius):
    """Apply Gaussian blur to the image."""
    return image.filter(ImageFilter.GaussianBlur(radius))


def dropout(image, drop_ratio):
    """Randomly drop out pixels in the image."""
    np_image = np.array(image)
    drop_mask = np.random.binomial(1, drop_ratio, np_image.shape)
    np_image = np_image * drop_mask
    np_image = np_image.astype(np.uint8)  # Ensure the data type is uint8
    return Image.fromarray(np_image)


def adjust_saturation(image, factor):
    """Adjust saturation."""
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def adjust_brightness(image, factor):
    """Adjust brightness."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image, factor):
    """Adjust contrast."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def jpeg2000_compression(image, quality_layer):
    """Apply JPEG2000 compression."""
    image.save("temp.jp2", "JPEG2000", quality_mode='rates', quality_layers=[quality_layer])
    return Image.open("temp.jp2")


def webp_compression(image, quality):
    """Apply WebP compression."""
    image.save("temp.webp", "WEBP", quality=quality)
    return Image.open("temp.webp")


def apply_median_filter(image, size):
    """Apply Median filter to the image."""
    return image.filter(ImageFilter.MedianFilter(size=size))


def apply_average_filter(image):
    """Apply Average filter to the image."""
    return image.filter(ImageFilter.BLUR)


def pixel_elimination(image, elimination_ratio):
    """Eliminate random pixels in the image."""
    np_image = np.array(image)
    mask = np.random.rand(*np_image.shape[:2]) < elimination_ratio
    np_image[mask] = 0
    return Image.fromarray(np_image)


def cropout(image, crop_ratio):
    """Crop out a random portion of the image."""
    np_image = np.array(image)
    h, w, _ = np_image.shape
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)
    np_image[y:y + ch, x:x + cw] = 0
    return Image.fromarray(np_image)


def copy_move_attack(image, min_patch_size, max_patch_size):
    """
    Apply a copy-move attack to the image by copying a random patch
    and pasting it onto another random location.

    The size of the patch is chosen randomly between the given min and max values.
    """
    np_image = np.array(image)
    height, width, _ = np_image.shape

    # Randomly select the patch size as a portion of the image size
    patch_size_ratio = np.random.uniform(min_patch_size, max_patch_size)
    patch_size = int(min(height, width) * patch_size_ratio)

    # Randomly select the top-left corner of the patch to copy
    src_y = np.random.randint(0, height - patch_size)
    src_x = np.random.randint(0, width - patch_size)

    # Randomly select the top-left corner of the location to paste the patch
    dst_y = np.random.randint(0, height - patch_size)
    dst_x = np.random.randint(0, width - patch_size)

    # Copy the patch
    patch = np_image[src_y:src_y + patch_size, src_x:src_x + patch_size]

    # Paste the patch at the new location
    np_image[dst_y:dst_y + patch_size, dst_x:dst_x + patch_size] = patch

    return Image.fromarray(np_image)


def inpainting_attack(image, min_patch_size, max_patch_size):
    """
    Apply an inpainting attack to the image by removing a random patch
    and filling it with surrounding pixels.

    The size of the patch is chosen randomly between the given min and max values.
    """
    np_image = np.array(image)
    height, width, _ = np_image.shape

    # Randomly select the patch size as a portion of the image size
    patch_size_ratio = np.random.uniform(min_patch_size, max_patch_size)
    patch_size = int(min(height, width) * patch_size_ratio)

    # Randomly select the top-left corner of the patch to remove
    src_y = np.random.randint(0, height - patch_size)
    src_x = np.random.randint(0, width - patch_size)

    # Create a mask for inpainting
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[src_y:src_y + patch_size, src_x:src_x + patch_size] = 1

    # Inpaint the image
    inpainted_image = cv2.inpaint(np_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return Image.fromarray(inpainted_image)


def compute_phash(image):
    """Compute the perceptual hash (pHash) of an image."""
    return imagehash.whash(image, hash_size=16)


def compute_hamming_distance(hash1, hash2):
    """Compute the Hamming distance between two pHash values."""
    return hash1 - hash2


def process_images_with_phash(input_dir, output_dir, transformations, params, combine=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    results = {transformation: [] for transformation in transformations}

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)

        if os.path.isfile(image_path) and os.path.splitext(image_name)[1].lower() in allowed_extensions:
            # original_image = cv2.imread(image_path)
            # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            # original_phash, _ = pdqhash.compute(original_image)

            original_image = Image.open(image_path).convert('RGB')
            original_phash = compute_phash(original_image)

            for transformation in transformations:
                # Apply the transformation
                transformed_image, applied_transforms = apply_transformations(image_path, [transformation], params, combine)
                transformed_phash = compute_phash(transformed_image)

                # Compute the Hamming distance
                hamming_distance = compute_hamming_distance(original_phash, transformed_phash)

                # Save the transformed image
                transformed_image_name = f"{os.path.splitext(image_name)[0]}_{transformation}.jpeg"
                transformed_image_path = os.path.join(output_dir, transformed_image_name)
                # transformed_image.save(transformed_image_path)
                print(str(transformation))

                # Append the results for this transformation
                results[transformation].append({
                    "img_name": image_name,
                    "transformation_name": transformation,
                    "original_hash": str(original_phash),
                    "transformed_hash": str(transformed_phash),
                    "hamming_distance": hamming_distance
                })

    # Save each transformation's results to a separate CSV file
    for transformation, data in results.items():
        df = pd.DataFrame(data)
        csv_file = os.path.join(output_dir, f"{transformation}_phash_results.csv")
        df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    input_dir = "div2k-801-900-jpeg"
    output_dir = "hash-sensitivity"

    # Define the transformations and params
    transformations = [
        # Add the transformations you want to apply
        jpeg_compression,
        resize_keep_ratio,
        resize_no_ratio,
        add_gaussian_noise,
        apply_gaussian_blur,
        dropout,
        adjust_saturation,
        adjust_brightness,
        adjust_contrast,
        jpeg2000_compression,
        webp_compression,
        apply_median_filter,
        apply_average_filter,
        pixel_elimination,
        cropout,
        # image_splicing,
        inpainting_attack,
        copy_move_attack
    ]
    params = {
        # Add the corresponding parameters for each transformation
        "jpeg_compression": {"quality": 70},  # CP
        "add_gaussian_noise": {"mean": 0, "std": 0.04},  # CP
        "apply_gaussian_blur": {"radius": 3},  # CP
        "adjust_saturation": {"factor": 1.0},  # CP
        "adjust_brightness": {"factor": 1.0},  # CP
        "adjust_contrast": {"factor": 1.0},  # CP
        "webp_compression": {"quality": 50},  # CP
        "apply_median_filter": {"size": 3},  # CP
        "apply_average_filter": {},  # CP
        "pixel_elimination": {"elimination_ratio": 0.05},  # CP
        "resize_keep_ratio": {"scale": 0.5},  # CP
        "resize_no_ratio": {"size": (256, 256)},  # CP
        "dropout": {"drop_ratio": 0.9},  # CP - drop ratio is actually the keeping ratio
        "jpeg2000_compression": {"quality_layer": 0.1},  # CP

        "cropout": {"crop_ratio": 0.2},  # CC
        "copy_move_attack": {"min_patch_size": 0.2, "max_patch_size": 0.4},  # CC
        "inpainting_attack": {"min_patch_size": 0.2, "max_patch_size": 0.4},  # CC
        "image_splicing": {"min_patch_size": 0.1, "max_patch_size": 0.3}  # CC
    }

    process_images_with_phash(input_dir, output_dir, transformations, params, combine=False)
