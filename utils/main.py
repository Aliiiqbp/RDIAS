import argparse
import os
from PIL import Image


# Import your modules
from error_correction import ErrorCorrectionFactory
from watermarking import WatermarkingFactory
from perceptual_hash import PerceptualHashFactory


def main(args):
    # Initialize components based on input arguments
    error_correction = ErrorCorrectionFactory.create(args.error_correction_alg, args.n, args.k, args.d)
    perceptual_hash = PerceptualHashFactory.create(args.hash_alg, args.hash_size)
    watermarking = WatermarkingFactory.create(args.watermarking_alg, args.watermarking_params)

    # Process images in the directory
    for image_name in os.listdir(args.image_dir):
        if image_name.endswith(args.image_format):
            image_path = os.path.join(args.image_dir, image_name)
            process_image(image_path, perceptual_hash, watermarking, args.image_format)


def process_image(image_path, perceptual_hash, watermarking, image_format):
    # Open the image
    image = Image.open(image_path)

    # Compute the perceptual hash
    hash_value = perceptual_hash.compute_hash(image)
    print("Perceptual Hash for " + image_path + ': ' + hash_value)

    # Use the hash value as the watermark
    watermark = hash_value

    # Embed the watermark into the image and save it with "wm_" prefix
    watermarking.embed_watermark(image, watermark, image_path, image_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image processing with error correction, watermarking, and perceptual hashing")

    parser.add_argument('--image_dir', type=str, required=True, help="Directory of images to process")
    parser.add_argument('--image_format', type=str, default="jpg", help="Format of images (e.g., jpg, png)")

    parser.add_argument('--error_correction_alg', type=str, required=True, help="Error correction algorithm")
    parser.add_argument('--n', type=int, required=False, help="Parameter n for error correction algorithm")
    parser.add_argument('--k', type=int, required=False, help="Parameter k for error correction algorithm")
    parser.add_argument('--d', type=int, required=False, help="Parameter d for error correction algorithm")

    parser.add_argument('--hash_alg', type=str, required=True, help="Perceptual hash algorithm")
    parser.add_argument('--hash_size', type=int, required=True, help="Size of the perceptual hash")

    parser.add_argument('--watermarking_alg', type=str, required=True, help="Watermarking algorithm")
    parser.add_argument('--watermarking_params', type=str, nargs='+', required=False,
                        help="Parameters for watermarking algorithm")

    args = parser.parse_args()
    main(args)
