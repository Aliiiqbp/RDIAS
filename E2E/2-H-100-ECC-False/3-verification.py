import os
import csv
from PIL import Image
import imagehash
import random
from PIL import ImageFilter
from trustmark import TrustMark


def compute_phash(image_path, hash_size):
    image = Image.open(image_path)
    phash_string = str(imagehash.phash(image, hash_size=hash_size))
    binary_string = ''.join(format(int(nibble, 16), '04b') for nibble in phash_string)
    return binary_string


def extract_watermark(image_path):
    image = Image.open(image_path)
    tm = TrustMark(verbose=False, model_type='Q', use_ECC=False)
    extracted_hash_str, wm_present, _ = tm.decode(stego_image=image, MODE='binary')
    if wm_present:
        return extracted_hash_str, True
    else:
        return extracted_hash_str, False


def hamming_distance(hash1, hash2):
    """
    Compute the Hamming distance between two binary strings.
    """
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def verify_image(image_path):
    """
    Verify the image by comparing the computed perceptual hash with the extracted watermark.
    """
    computed_hash = compute_phash(image_path=image_path, hash_size=10)
    extracted_hash, wm_successful = extract_watermark(image_path)
    ham_distance = hamming_distance(computed_hash, extracted_hash)

    return extracted_hash, computed_hash, ham_distance, wm_successful


def main(input_dir, output_csv):
    results = []
    # List of allowed image file extensions
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

    # Iterate through all images in the input directory
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)

        if os.path.isfile(image_path) and os.path.splitext(image_name)[1].lower() in allowed_extensions:
            extracted_hash, computed_hash, ham_distance, wm_successful = verify_image(image_path)
            results.append([image_name, extracted_hash, computed_hash, ham_distance, wm_successful])

    # Write results to the output CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Extracted Hash", "Computed Hash", "Hamming Distance", "WM_successful"])
        writer.writerows(results)


if __name__ == "__main__":

    Datasets = ["CLIC", "DIV2K"]  # "MetFace"

    for data in Datasets:
        base_input_dir = data + '-transformed'
        output_dir = data + '-verification'


        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir + '/plots', exist_ok=True)

        # Iterate over each subdirectory in the base input directory
        for transformation_dir in os.listdir(base_input_dir):
            transformation_path = os.path.join(base_input_dir, transformation_dir)
            print(transformation_dir)
            if os.path.isdir(transformation_path):
                output_csv = os.path.join(output_dir, f"{transformation_dir}.csv")
                main(transformation_path, output_csv)
