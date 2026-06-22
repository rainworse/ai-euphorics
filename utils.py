import os
import random

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid


def load_and_resize_images(folder, size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    images = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            images.append(transform(img))

    return images

def display_images(images):
    batch = torch.stack(images)
    grid = make_grid(batch, nrow=4)

    plt.close('all')
    # plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0), vmin=0, vmax=1)
    plt.axis("off")
    plt.show()


def pixels_to_pil(img_pixels):
    arr = np.asarray(img_pixels.clone().detach())
    if arr.shape[0] == 3:  # CHW -> HWC
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        arr = (arr * 255 if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def shuffle_image_dict(images):
    shuffled_image_dict = []
    for idx, img in enumerate(images):
        shuffled_image_dict.append({'original_idx': idx, 'img_pixels': img})

    random.shuffle(shuffled_image_dict)
    return shuffled_image_dict

def create_noise_image(device, shape=(3, 256, 256)):
    return torch.rand(shape, requires_grad=True).to(device)

def get_random_comparison_prompt():
    from numpy import random

    candidate_prompts = [
        'Which image makes you feel the best?',
        'Which of these images do you prefer?',
        'Which image inspires you the most?',
        'Which of these images do you like most?',
        'Pick of the given images that make you feel the best.',
        'Which image makes you more happy/less sad?'
    ]
    return random.choice(candidate_prompts)