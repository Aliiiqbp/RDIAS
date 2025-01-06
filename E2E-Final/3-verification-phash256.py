import os
import csv
from PIL import Image
import imagehash
from trustmark import TrustMark
import time


tm = TrustMark(verbose=False, model_type='Q', use_ECC=True, encoding_type=TrustMark.Encoding.BCH_4)


# Function to compute pHash for an image
def compute_phash(image, hash_size):
    phash_string = str(imagehash.phash(image, hash_size=hash_size))
    binary_string = ''.join(format(int(nibble, 16), '04b') for nibble in phash_string)
    return binary_string


# Function to extract watermark from an image part
def extract_watermark(image):
    extracted_hash_str, wm_present, _, ecc_time = tm.decode(stego_image=image, MODE='binary')
    return extracted_hash_str[:-4], wm_present, ecc_time


# Function to calculate Hamming distance between two binary strings
def hamming_distance(hash1, hash2):
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


# Verification function for each image
def verify_image(image_path):

    hash_time_start = time.time()
    cover_image = Image.open(image_path)

    # Compute the 256-bit hash of the entire image
    computed_full_hash = compute_phash(cover_image, hash_size=16)
    hash_time_end = time.time()
    width, height = cover_image.size

    wm_time_start = time.time()
    # Split the image into four parts
    parts = {
        'top_left': cover_image.crop((0, 0, width // 2, height // 2)),
        'top_right': cover_image.crop((width // 2, 0, width, height // 2)),
        'bottom_left': cover_image.crop((0, height // 2, width // 2, height)),
        'bottom_right': cover_image.crop((width // 2, height // 2, width, height))
    }

    extracted_hash_parts = []
    wm_present_flags = {}
    ecc_times = []

    # Extract watermark from each part and combine them to form the full extracted 256-bit hash
    for part_name, part_image in parts.items():
        extracted_hash, wm_present, ecc_time = extract_watermark(part_image)
        extracted_hash_parts.append(extracted_hash)
        wm_present_flags[part_name] = wm_present
        ecc_times.append(ecc_time)

    extracted_full_hash = ''.join(extracted_hash_parts)  # Combine the 64-bit hashes into 256 bits
    wm_time_end = time.time()

    # Calculate Hamming distance between the full extracted hash and the computed hash of the entire image
    total_hamming_distance = hamming_distance(computed_full_hash, extracted_full_hash)

    return computed_full_hash, extracted_full_hash, wm_present_flags, total_hamming_distance, hash_time_end - hash_time_start, (wm_time_end - wm_time_start) - sum(ecc_times), sum(ecc_times)


# Main function to verify all images in a directory and output results to a CSV
def main(input_dir, output_csv):
    results = []
    # List of allowed image file extensions
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

    # Iterate through all images in the input directory
    for image_name in os.listdir(input_dir):
        # print(image_name)
        image_path = os.path.join(input_dir, image_name)

        if os.path.isfile(image_path) and os.path.splitext(image_name)[1].lower() in allowed_extensions:
            # TODO: time verification start
            start_time = time.time()
            computed_full_hash, extracted_full_hash, wm_present_flags, total_hamming_distance, hash_time, wm_time, ecc_time = verify_image(image_path)
            end_time = time.time()
            # TODO: time verification end

            # Add result for this image to the list
            results.append([
                image_name,
                computed_full_hash,
                extracted_full_hash,
                wm_present_flags['top_left'], wm_present_flags['top_right'], wm_present_flags['bottom_left'], wm_present_flags['bottom_right'],
                total_hamming_distance, end_time - start_time, hash_time, wm_time, ecc_time
            ])

    # Write results to the output CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Image Name",
            "Computed Full Hash", "Extracted Full Hash",
            "WM Present (Top Left)", "WM Present (Top Right)", "WM Present (Bottom Left)", "WM Present (Bottom Right)",
            "Total Hamming Distance", "E2E_time", "Fingerprinting_time", "wm_time", "ECC_time"
        ])
        writer.writerows(results)


if __name__ == "__main__":
    # Specify datasets
    Datasets = ["FFHQ"]  # Add your datasets here "CLIC", "DIV2K", 'FFHQ1000'

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


