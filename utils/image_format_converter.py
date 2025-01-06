import os
import argparse
from PIL import Image


def convert_image_format(input_dir, output_dir, target_format):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(input_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            image_path = os.path.join(input_dir, image_name)
            with Image.open(image_path) as img:
                # Construct the new filename with the target format
                base_name = os.path.splitext(image_name)[0]
                new_image_name = f"{base_name}.{target_format.lower()}"
                new_image_path = os.path.join(output_dir, new_image_name)

                # Save the image in the target format
                img.save(new_image_path, format=target_format.upper())
                print(f"Converted {image_path} to {new_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert image formats")
    parser.add_argument('--input_dir', type=str, default='image_dir', required=False, help="Directory of input images")
    parser.add_argument('--output_dir', type=str, default='div2k-test-jpeg', required=False, help="Directory to save converted images")
    parser.add_argument('--target_format', type=str, default='JPEG', required=False, choices=['JPEG', 'PNG', 'BMP', 'GIF', 'TIFF'],
                        help="Target image format")

    args = parser.parse_args()
    convert_image_format(args.input_dir, args.output_dir, args.target_format)
