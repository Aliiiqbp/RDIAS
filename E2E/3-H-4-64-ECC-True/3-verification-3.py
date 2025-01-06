import os
import csv
from PIL import Image
import imagehash
from trustmark import TrustMark


# Function to compute pHash for an image part
def compute_phash(image, hash_size):
    phash_string = str(imagehash.phash(image, hash_size=hash_size))
    binary_string = ''.join(format(int(nibble, 16), '04b') for nibble in phash_string)
    return binary_string


# Function to extract watermark from an image part
def extract_watermark(image):
    tm = TrustMark(verbose=False, model_type='Q', use_ECC=True, encoding_type=TrustMark.Encoding.BCH_4)
    extracted_hash_str, wm_present, _ = tm.decode(stego_image=image, MODE='binary')
    return extracted_hash_str[:-4], wm_present


# Function to calculate Hamming distance between two binary strings
def hamming_distance(hash1, hash2):
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


# Verification function for each image
def verify_image(image_path):
    cover_image = Image.open(image_path)
    width, height = cover_image.size

    # Split the image into four parts
    parts = {
        'top_left': cover_image.crop((0, 0, width // 2, height // 2)),
        'top_right': cover_image.crop((width // 2, 0, width, height // 2)),
        'bottom_left': cover_image.crop((0, height // 2, width // 2, height)),
        'bottom_right': cover_image.crop((width // 2, height // 2, width, height))
    }

    computed_hashes = {}
    extracted_hashes = {}
    wm_present_flags = {}
    hamming_distances = {}

    total_hamming_distance = 0

    # Process each part
    for part_name, part_image in parts.items():
        # Compute the perceptual hash (pHash) for the part
        computed_hash = compute_phash(part_image, hash_size=8)
        computed_hashes[part_name] = computed_hash

        # Extract the watermark from the part
        extracted_hash, wm_present = extract_watermark(part_image)
        extracted_hashes[part_name] = extracted_hash
        wm_present_flags[part_name] = wm_present

        # Calculate Hamming distance between computed and extracted hashes
        hamming_dist = hamming_distance(computed_hash, extracted_hash)
        hamming_distances[part_name] = hamming_dist

        # Add to the total Hamming distance
        total_hamming_distance += hamming_dist

    return computed_hashes, extracted_hashes, wm_present_flags, hamming_distances, total_hamming_distance

# Main function to verify all images in a directory and output results to a CSV
def main(input_dir, output_csv):
    results = []
    # List of allowed image file extensions
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

    # Iterate through all images in the input directory
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)

        if os.path.isfile(image_path) and os.path.splitext(image_name)[1].lower() in allowed_extensions:
            computed_hashes, extracted_hashes, wm_present_flags, hamming_distances, total_hamming_distance = verify_image(image_path)

            # Add result for this image to the list
            results.append([
                image_name,
                computed_hashes['top_left'], computed_hashes['top_right'], computed_hashes['bottom_left'], computed_hashes['bottom_right'],
                extracted_hashes['top_left'], extracted_hashes['top_right'], extracted_hashes['bottom_left'], extracted_hashes['bottom_right'],
                wm_present_flags['top_left'], wm_present_flags['top_right'], wm_present_flags['bottom_left'], wm_present_flags['bottom_right'],
                hamming_distances['top_left'], hamming_distances['top_right'], hamming_distances['bottom_left'], hamming_distances['bottom_right'],
                total_hamming_distance
            ])

    # Write results to the output CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Image Name",
            "Computed Hash (Top Left)", "Computed Hash (Top Right)", "Computed Hash (Bottom Left)", "Computed Hash (Bottom Right)",
            "Extracted Hash (Top Left)", "Extracted Hash (Top Right)", "Extracted Hash (Bottom Left)", "Extracted Hash (Bottom Right)",
            "WM Present (Top Left)", "WM Present (Top Right)", "WM Present (Bottom Left)", "WM Present (Bottom Right)",
            "Hamming Distance (Top Left)", "Hamming Distance (Top Right)", "Hamming Distance (Bottom Left)", "Hamming Distance (Bottom Right)",
            "Total Hamming Distance"
        ])
        writer.writerows(results)


if __name__ == "__main__":
    # Specify datasets
    Datasets = ["CLIC", "DIV2K"]  # Add your datasets here

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
