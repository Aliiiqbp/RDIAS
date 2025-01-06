import lmdb
from PIL import Image
import io
import os


def extract_images(mdb_path, output_dir):
    env = lmdb.open(mdb_path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            image = Image.open(io.BytesIO(value))
            image.save(os.path.join(output_dir, f"{key.decode('utf-8')}.png"))


def remove_small_images(directory):
    """
    Removes images from the specified directory if their width or height is less than 1024 pixels.

    Args:
        directory (str): The path to the directory containing images.
    """
    # Supported image file extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Skip if it's not a file
        if not os.path.isfile(filepath):
            continue

        # Check if the file has an image extension
        if not filename.lower().endswith(image_extensions):
            continue

        try:
            # Open the image and get its dimensions
            with Image.open(filepath) as img:
                width, height = img.size

            # Remove the image if either dimension is less than 1024 pixels
            if width < 1024 or height < 1024:
                os.remove(filepath)
                print(f"Removed {filepath} (size {width}x{height})")
        except Exception as e:
            # Handle exceptions (e.g., file is not an image or is corrupted)
            print(f"Could not process {filepath}: {e}")

# Usage
# extract_images('baroque_lmdb', '7.baroque_lmdb')
remove_small_images('7.baroque_lmdb')
