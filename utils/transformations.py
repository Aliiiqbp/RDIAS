import os
import random
import csv
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
import re
import cv2


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


# def get_image_path(image):
#     """
#     Find the path of the given image.
#     """
#     return image.filename
#
#
# def get_random_image_from_directory(directory, exclude_image):
#     """
#     Get a random image from the directory, excluding the given image.
#     """
#     images = [file for file in os.listdir(directory) if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
#     images.remove(os.path.basename(exclude_image))
#     random_image = random.choice(images)
#     return os.path.join(directory, random_image)
#
#
# def image_splicing(image, min_patch_size, max_patch_size):
#     """
#     Apply an image splicing attack by copying a patch from a randomly chosen image
#     in the same directory as the input image and pasting it onto a random location
#     in the input image.
#
#     The size of the patch is chosen randomly between the given min and max values.
#     """
#     np_image1 = np.array(image)
#     height1, width1, _ = np_image1.shape
#
#     # Find the directory of the current image
#     image_path = get_image_path(image)
#     directory = os.path.dirname(image_path)
#
#     # Get a random image from the same directory
#     image2_path = get_random_image_from_directory(directory, image_path)
#     image2 = Image.open(image2_path)
#     np_image2 = np.array(image2)
#     height2, width2, _ = np_image2.shape
#
#     # Randomly select the patch size as a portion of the second image size
#     patch_size_ratio = np.random.uniform(min_patch_size, max_patch_size)
#     patch_size = int(min(height2, width2) * patch_size_ratio)
#
#     # Randomly select the top-left corner of the patch to copy from image2
#     src_y = np.random.randint(0, height2 - patch_size)
#     src_x = np.random.randint(0, width2 - patch_size)
#
#     # Randomly select the top-left corner of the location to paste the patch in image1
#     dst_y = np.random.randint(0, height1 - patch_size)
#     dst_x = np.random.randint(0, width1 - patch_size)
#
#     # Copy the patch from image2
#     patch = np_image2[src_y:src_y + patch_size, src_x:src_x + patch_size]
#
#     # Paste the patch into image1
#     np_image1[dst_y:dst_y + patch_size, dst_x:dst_x + patch_size] = patch
#
#     return Image.fromarray(np_image1)


def calculate_metrics(original, transformed, transformation_name):
    """Calculate PSNR, SSIM, and Wasserstein Distance between original and transformed images."""
    if re.match(r"^(resize_keep_ratio|resize_no_ratio)", transformation_name[0]):
        return np.nan, np.nan, np.nan

    original = np.array(original).astype(np.float32)
    transformed = np.array(transformed).astype(np.float32)

    data_range = original.max() - original.min()

    psnr_value = psnr(original, transformed, data_range=data_range)
    ssim_value = ssim(original, transformed, data_range=data_range, channel_axis=-1)
    wasserstein_value = wasserstein_distance(original.ravel(), transformed.ravel())

    return psnr_value, ssim_value, wasserstein_value


def apply_transformations(image_path, transformations, params, combine=False):
    """
    Apply transformations to an image.

    :param image_path: Path to the image file.
    :param transformations: List of transformations to apply.
    :param params: Dictionary of transformation parameters.
    :param combine: Whether to combine transformations together or choose randomly from the list.
    :return: Transformed image, List of transformations applied.
    """
    image = Image.open(image_path).convert('RGB')
    applied_transforms = []

    if combine:
        for transform in transformations:
            param = params[transform.__name__]
            image = transform(image, **param)
            applied_transforms.append(f"{transform.__name__}({param})")
    else:
        transform = random.choice(transformations)
        param = params[transform.__name__]
        image = transform(image, **param)
        applied_transforms.append(f"{transform.__name__}({param})")

    return image, applied_transforms


def process_images(input_dir, output_dir, transformations, params, combine=False):
    """
    Process all images in a directory with the specified transformations.

    :param input_dir: Directory containing input images.
    :param output_dir: Directory to save transformed images.
    :param transformations: List of transformations to apply.
    :param params: Dictionary of transformation parameters.
    :param combine: Whether to combine transformations together or choose randomly from the list.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

    with open(os.path.join(output_dir, 'report.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Image Name", "Transformed Image Name", "Transformations Applied", "PSNR", "SSIM", "Wasserstein Distance"])

        cnt = 0
        for image_name in os.listdir(input_dir):
            cnt += 1
            print(cnt, image_name)

            image_path = os.path.join(input_dir, image_name)

            if os.path.isfile(image_path) and os.path.splitext(image_name)[1].lower() in allowed_extensions:
                transformed_image, applied_transforms = apply_transformations(image_path, transformations, params,
                                                                              combine)
                transformed_image_name = f"{os.path.splitext(image_name)[0]}_transformed.jpeg"
                transformed_image_path = os.path.join(output_dir, transformed_image_name)
                transformed_image.save(transformed_image_path)

                original_image = Image.open(image_path).convert('RGB')
                psnr_value, ssim_value, wasserstein_value = calculate_metrics(original_image, transformed_image,
                                                                              applied_transforms)

                transformations_applied = ", ".join(applied_transforms)
                writer.writerow([image_name, transformed_image_name, transformations_applied, psnr_value, ssim_value,
                                 wasserstein_value])


if __name__ == "__main__":
    # Define the list of transformations
    transformations = [
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

    # transformations = [
        # image_splicing,
        # inpainting_attack,
        # copy_move_attack
    # ]

    # Define transformation parameters
    params = {
        "jpeg_compression": {"quality": 50},  # CP
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

    for t in transformations:
        # Directory paths
        input_dir = "div2k-801-900-jpeg-immune-images-dhash-Q"
        output_dir = input_dir + "-transformed/" + str(t)

        # Apply transformations (set combine=True to apply all transformations together)
        process_images(input_dir, output_dir, [t], params, combine=True)
