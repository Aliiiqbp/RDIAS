import os
import PIL
from PIL import Image

from trustmark import TrustMark
from pathlib import Path
import math
import random
import numpy as np


class Watermarking:
    def embed_watermark(self, image, watermark, image_path, image_format):
        raise NotImplementedError

    def detect_watermark(self, image):
        raise NotImplementedError


class WatermarkingAlgorithm1(Watermarking):
    def __init__(self, params):
        self.params = params[0]
        self.tm = TrustMark(verbose=False, model_type=self.params, encoding_type=TrustMark.Encoding.BCH_4)

    def embed_watermark(self, image, watermark, image_path, image_format):
        # Implement watermark embedding logic for Algorithm 1
        # Ensure the watermark is embedded and the image is saved with "wm_" prefix
        image_dir, image_name = os.path.split(image_path)
        wm_image_name = f"wm_{image_name}"
        wm_image_path = os.path.join(image_dir, wm_image_name)

        # Example logic to embed the watermark (to be implemented)
        # watermarked_image = some_watermarking_function(image, watermark)
        cover = Image.open(image_path).convert('RGB')
        self.tm.encode(cover, string_secret=watermark).save(wm_image_path, format=image_format)

    def detect_watermark(self, image):
        # Implement watermark detection logic for Algorithm 1
        cover = Image.open(image).convert('RGB')
        wm_secret, wm_present, wm_schema = self.tm.decode(cover)
        if wm_present:
            print(f'Extracted secret: {wm_secret}')
        else:
            print('No watermark detected')


class WatermarkingAlgorithm2(Watermarking):
    def __init__(self, params):
        self.params = params

    def embed_watermark(self, image, watermark, image_path, image_format):
        # Implement watermark embedding logic for Algorithm 2
        # Ensure the watermark is embedded and the image is saved with "wm_" prefix
        image_dir, image_name = os.path.split(image_path)
        wm_image_name = f"wm_{image_name}"
        wm_image_path = os.path.join(image_dir, wm_image_name)

        # Example logic to embed the watermark (to be implemented)
        # watermarked_image = some_watermarking_function(image, watermark)

        # Save the watermarked image
        image.save(wm_image_path, format=image_format)

    def detect_watermark(self, image):
        # Implement watermark detection logic for Algorithm 2
        pass


# Add more algorithms as needed

class WatermarkingFactory:
    @staticmethod
    def create(algorithm, params):
        if algorithm == "trustmark":
            return WatermarkingAlgorithm1(params)
        elif algorithm == "algorithm2":
            return WatermarkingAlgorithm2(params)
        # Add more algorithms as needed
        else:
            raise ValueError(f"Unknown watermarking algorithm: {algorithm}")
