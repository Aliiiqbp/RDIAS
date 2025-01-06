import os
import random
import csv
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from numpy.core.defchararray import endswith
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


def image_splicing_attack(image, min_patch_size=0.2, max_patch_size=0.4, directory="CLIC-Immune"):
    """
    Apply an image splicing attack by copying a random patch from another image in the given directory
    and pasting it into the current image.

    The size of the patch is chosen randomly between the given min and max values.

    :param image: The original PIL image to be attacked.
    :param min_patch_size: Minimum size ratio of the patch to be spliced.
    :param max_patch_size: Maximum size ratio of the patch to be spliced.
    :param directory: Directory where the other images are stored. If not provided, raises an error.
    :return: PIL Image object after the splicing attack.
    """
    # Convert the PIL Image to a NumPy array
    np_image = np.array(image)
    height, width, _ = np_image.shape

    # Determine the directory of the image
    if directory is None:
        if hasattr(image, 'filename'):
            directory = os.path.dirname(image.filename)
        else:
            raise ValueError("Directory must be provided if the image does not have a filename attribute.")

    # Get all image files in the specified directory
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        raise ValueError("No other images found in the directory for splicing.")

    # Randomly select a different image from the directory
    random_image_path = os.path.join(directory, np.random.choice(image_files))
    random_image = Image.open(random_image_path)
    np_random_image = np.array(random_image)
    rand_height, rand_width, _ = np_random_image.shape

    # Randomly select the patch size as a portion of the image size
    patch_size_ratio = np.random.uniform(min_patch_size, max_patch_size)
    patch_size = int(min(rand_height, rand_width) * patch_size_ratio)

    # Randomly select the top-left corner of the patch to copy from the random image
    src_y = np.random.randint(0, rand_height - patch_size)
    src_x = np.random.randint(0, rand_width - patch_size)

    # Randomly select the top-left corner of the location to paste the patch in the original image
    dst_y = np.random.randint(0, height - patch_size)
    dst_x = np.random.randint(0, width - patch_size)

    # Copy the patch from the random image
    patch = np_random_image[src_y:src_y + patch_size, src_x:src_x + patch_size]

    # Paste the patch into the original image
    np_image[dst_y:dst_y + patch_size, dst_x:dst_x + patch_size] = patch

    return Image.fromarray(np_image)


def cropping(image, min_crop_ratio=0.5, max_crop_ratio=0.9):
    """
    Crop a random portion of the image.

    The size of the cropped region is chosen randomly between the given min and max crop ratios.

    :param image: The original PIL image to be cropped.
    :param min_crop_ratio: Minimum size ratio of the crop relative to the image dimensions.
    :param max_crop_ratio: Maximum size ratio of the crop relative to the image dimensions.
    :return: PIL Image object after cropping.
    """
    # Convert the PIL Image to a NumPy array
    np_image = np.array(image)
    height, width, _ = np_image.shape

    # Randomly determine the crop size as a portion of the image size
    crop_ratio = np.random.uniform(min_crop_ratio, max_crop_ratio)
    crop_height = int(height * crop_ratio)
    crop_width = int(width * crop_ratio)

    # Randomly select the top-left corner of the crop region
    start_y = np.random.randint(0, height - crop_height)
    start_x = np.random.randint(0, width - crop_width)

    # Crop the image
    cropped_image = np_image[start_y:start_y + crop_height, start_x:start_x + crop_width]

    return Image.fromarray(cropped_image)



def calculate_metrics(original, transformed, transformation_name):
    """Calculate PSNR, SSIM, and Wasserstein Distance between original and transformed images."""
    if re.match(r"^(resize_keep_ratio|resize_no_ratio)", transformation_name[0]) or re.match(r"^cropping", transformation_name[0]):
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
        print(transform)
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
            print(cnt)

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
        inpainting_attack,
        copy_move_attack,
        cropping,
        cropout,
        image_splicing_attack,

        jpeg_compression,
        resize_keep_ratio,
        resize_no_ratio,
        add_gaussian_noise,
        apply_gaussian_blur,

        # dropout,
        # adjust_saturation,
        # adjust_brightness,
        # adjust_contrast,
        # jpeg2000_compression,
        # webp_compression,
        # apply_median_filter,
        # apply_average_filter,
        # pixel_elimination
    ]

    Datasets = ["CLIC", "DIV2K"]  # "MetFace"
    for data in Datasets:
        for t in transformations:
            input_dir = data + "-Immune"
            output_dir = data + "-transformed/" + str(t)

            # Define transformation parameters
            params = {
                "jpeg_compression": {"quality": 90},  # CP
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
                "image_splicing_attack": {"min_patch_size": 0.2, "max_patch_size": 0.4, "directory": input_dir},
                # CC
                "cropping": {"min_crop_ratio": 0.5, "max_crop_ratio": 0.75}  # CC
            }

            # Apply transformations (set combine=True to apply all transformations together)
            process_images(input_dir, output_dir, [t], params, combine=True)
