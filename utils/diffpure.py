import torch
import torchvision.transforms as transforms
from PIL import Image
from diffusers import DDPMPipeline
import numpy as np

# Load the pre-trained DDPM model
model_name = "google/ddpm-cifar10-32"  # You can replace this with other models from Hugging Face
pipeline = DDPMPipeline.from_pretrained(model_name)

# Function to add noise (forward SDE)
def add_noise(image, num_steps, device):
    noisy_image = image.clone().to(device)
    for _ in range(num_steps):
        noise = torch.randn_like(noisy_image)
        noisy_image = noisy_image + noise
    return noisy_image

# Function to denoise (reverse SDE)
def denoise(noisy_image, num_steps, device):
    denoised_image = noisy_image.clone().to(device)
    for _ in range(num_steps):
        denoised_image = pipeline(denoised_image)["sample"]
    return denoised_image

# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize the image to match the model's input size
        transforms.ToTensor(),        # Convert the image to a PyTorch tensor
    ])
    return preprocess(image).unsqueeze(0)

# Function to save the purified image
def save_image(tensor, output_path):
    tensor = tensor.squeeze().cpu().clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(output_path)

# Main function to perform image purification
def purify_image(image_path, output_path, noise_steps=10, denoise_steps=10, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load and preprocess the image
    image = load_image(image_path).to(device)

    # Add noise (Forward SDE)
    noisy_image = add_noise(image, noise_steps, device)

    # Denoise (Reverse SDE)
    purified_image = denoise(noisy_image, denoise_steps, device)

    # Save the purified image
    save_image(purified_image, output_path)

# Example usage
if __name__ == "__main__":
    image_path = "image.png"  # Path to the input image
    output_path = "purified_image.png"  # Path to save the purified image
    noise_steps = 10  # Number of times to add noise
    denoise_steps = 10  # Number of times to remove noise

    purify_image(image_path, output_path, noise_steps, denoise_steps)
