import time

import cv2
from PIL import Image
import imagehash
import random
from PIL import ImageFilter
from trustmark import TrustMark
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
import re
import bchlib



def apply_transformation(image, transformation, parameter):
    if transformation == 'JPEG Compression':
        # Simulate JPEG compression
        buffer = image.copy()
        buffer.save('/tmp/compressed_image.jpg', 'JPEG', quality=parameter)
        image = Image.open('/tmp/compressed_image.jpg')
    elif transformation == 'Resizing':
        # Resize the image
        width, height = image.size
        image = image.resize((int(width * parameter), int(height * parameter)), Image.Resampling.LANCZOS)
    elif transformation == 'Gaussian Noise':
        # Add Gaussian noise
        image = np.array(image)
        image = image + np.random.normal(0, parameter, image.shape)
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = Image.fromarray(np.uint8(image))
    elif transformation == 'Gaussian Blur':
        # Add salt and pepper noise
        image = image.filter(ImageFilter.GaussianBlur(radius=parameter))
    return image


def add_gaussian_noise(image, mean, std):
    """Add Gaussian noise to the image."""
    np_image = np.array(image)
    noisy_img = np_image + np.random.normal(mean, std, np_image.shape)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def apply_gaussian_blur(image, radius):
    """Apply Gaussian blur to the image."""
    return image.filter(ImageFilter.GaussianBlur(radius))

def jpeg2000_compression(image, quality_layer):
    """Apply JPEG2000 compression."""
    image.save("temp.jp2", "JPEG2000", quality_mode='dB', quality_layers=[quality_layer])
    return Image.open("temp.jp2")


def webp_compression(image, quality):
    """Apply WebP compression."""
    image.save("temp.webp", "WEBP", quality=quality)
    return Image.open("temp.webp")


def calculate_metrics(original, transformed):

    original = np.array(original).astype(np.float32)
    transformed = np.array(transformed).astype(np.float32)

    data_range = original.max() - original.min()

    psnr_value = psnr(original, transformed, data_range=data_range)
    ssim_value = ssim(original, transformed, data_range=data_range, channel_axis=-1)
    wasserstein_value = wasserstein_distance(original.ravel(), transformed.ravel())

    return psnr_value, ssim_value, wasserstein_value

tic = time.time()
img1 = Image.open('wm_7.png').convert('RGB')
print(imagehash.phash(img1, hash_size=16))
toc = time.time()
print(toc - tic)


tic = time.time()
img1 = Image.open('wm_7_transformed.jpeg').convert('RGB')
print(imagehash.phash(img1, hash_size=16))
toc = time.time()
print(toc - tic)

# img2 = apply_transformation(img1, 'Gaussian Blur', parameter=0.2)
# img3 = apply_gaussian_blur(img1, 0.5)
# # img1.show()
# img2.show()
# img3.show()
# print(calculate_metrics(img1, img2))
# print(calculate_metrics(img1, img3))

# print(random.randint(70, 71))

# print(np.random.uniform(70, 90))










# tm=TrustMark(verbose=True, model_type='Q')
# cover = Image.open('0801.jpeg').convert('RGB')
# tm.encode(cover, 'mysecret').save('0801-wm.jpeg')
#
# # Image.open('0801-wm.jpeg').r.save('0801-wm-rotate.jpeg')
#
# cover = Image.open('0801-wm-rotate.jpeg').convert('RGB')
# wm_secret, wm_present, wm_schema = tm.decode(cover)
# if wm_present:
#     print(f'Extracted secret: {wm_secret}')
# else:
#     print('No watermark detected')
#     print(wm_secret)