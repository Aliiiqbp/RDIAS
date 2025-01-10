from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
from io import BytesIO
import random
import os
import imagehash
import time
from trustmark import TrustMark


tm = TrustMark(verbose=False, model_type='Q', use_ECC=True, encoding_type=TrustMark.Encoding.BCH_4)


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def classify_image(image):
    """
    Dummy function to classify the image.
    Replace this with your actual authentication logic.
    """
    result = verify_image(image)
    return result

@app.route('/check_image', methods=['POST'])
def check_image():
    data = request.json
    image_url = data.get('image_url')

    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # Perform your authentication check here
        flag = classify_image(image)
        return jsonify({"flag": flag})

    except Exception as e:
        return jsonify({"flag": "error", "message": str(e)})


def compute_phash(image, hash_size):
    phash_string = str(imagehash.phash(image, hash_size=hash_size))
    binary_string = ''.join(format(int(nibble, 16), '04b') for nibble in phash_string)
    return binary_string

# Function to extract watermark from an image part
def extract_watermark(image):
    extracted_hash_str, wm_present, _ = tm.decode(stego_image=image, MODE='binary')
    return extracted_hash_str[:-4], wm_present

# Function to calculate Hamming distance between two binary strings
def hamming_distance(hash1, hash2):
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

# Verification function for a single image
def verify_image(image):
    cover_image = image

    # Compute the 256-bit hash of the entire image
    computed_full_hash = compute_phash(cover_image, hash_size=16)
    width, height = cover_image.size

    # Split the image into four parts
    parts = {
        'top_left': cover_image.crop((0, 0, width // 2, height // 2)),
        'top_right': cover_image.crop((width // 2, 0, width, height // 2)),
        'bottom_left': cover_image.crop((0, height // 2, width // 2, height)),
        'bottom_right': cover_image.crop((width // 2, height // 2, width, height))
    }

    extracted_hash_parts = []
    wm_present_flags = []

    # Extract watermark from each part and combine them to form the full extracted 256-bit hash
    for part_name, part_image in parts.items():
        extracted_hash, wm_present = extract_watermark(part_image)
        extracted_hash_parts.append(extracted_hash)
        wm_present_flags.append(wm_present)

    extracted_full_hash = ''.join(extracted_hash_parts)  # Combine the 64-bit hashes into 256 bits

    # Calculate Hamming distance between the full extracted hash and the computed hash of the entire image
    total_hamming_distance = hamming_distance(computed_full_hash, extracted_full_hash)

    if total_hamming_distance <= 4 and (False not in wm_present_flags):
        return "authentic"
    else:
        return "non-authentic"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
