import os
import sys
from PIL import Image

def resize_images(input_folder, output_folder, size):
    """
    Resizes all images in the input_folder to the specified size and saves them in the output_folder.

    Args:
        input_folder (str): Path to the folder containing original images.
        output_folder (str): Path to the folder where resized images will be saved.
        size (tuple): The desired image size as a tuple (width, height).
    """
    # Supported image file extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        # Skip if it's not a file or not an image
        if not os.path.isfile(filepath) or not filename.lower().endswith(image_extensions):
            continue

        try:
            with Image.open(filepath) as img:
                # Resize the image
                resized_img = img.resize(size, Image.Resampling.LANCZOS)

                # Save the resized image in the output folder
                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)

                print(f"Resized and saved {output_path}")
        except Exception as e:
            print(f"Could not process {filepath}: {e}")


if __name__ == "__main__":
    # Example usage
    input_folder = 'Mixed'  # Replace with your input folder path
    image_size = (1024, 1024)  # Desired image size (width, height)

    # Get the base name of the input folder
    input_folder_name = os.path.basename(os.path.normpath(input_folder))

    # Create the output folder name
    output_folder_name = f"{input_folder_name}-{image_size[0]}"
    output_folder = os.path.join(os.path.dirname(input_folder), output_folder_name)

    resize_images(input_folder, output_folder, image_size)
