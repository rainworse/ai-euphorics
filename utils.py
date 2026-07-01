import os
import random

import numpy as np
import torch
from PIL import Image

def create_comparison_set(batch_images, buffer_images, candidate_image, k):
    total = len(batch_images) + len(buffer_images)
    n_draw = k - 1

    if n_draw > total:
        raise ValueError(f"Cannot draw {n_draw} images from pool of {total}")

    pool = [("batch", i, img) for i, img in enumerate(batch_images)] + \
           [("buffer", i, img) for i, img in enumerate(buffer_images)]
    weights = [1 / total] * total

    drawn = []
    remaining = list(zip(weights, pool))
    for _ in range(n_draw):
        total_w = sum(w for w, _ in remaining)
        r = random.uniform(0, total_w)
        cumulative = 0
        for idx, (w, item) in enumerate(remaining):
            cumulative += w
            if r <= cumulative:
                drawn.append(item)
                remaining.pop(idx)
                break

    comparison = [("candidate", None, candidate_image)] + drawn
    random.shuffle(comparison)

    comparison_set  = [img for _, _, img in comparison]
    buffer_indices  = {i: src_idx for i, (tag, src_idx, _) in enumerate(comparison) if tag == "buffer"}
    candidate_index = next(i for i, (tag, _, _) in enumerate(comparison) if tag == "candidate")

    return comparison_set, buffer_indices, candidate_index


def save_image(tensor, folder = "outputs", filename = "image.png"):
    os.makedirs(folder, exist_ok=True)

    t = tensor.detach().cpu()

    if t.max() > 1.0:
        t = t / 255.0

    t = t.clamp(0.0, 1.0)
    t = (t * 255).byte()
    t = t.permute(1, 2, 0).numpy()

    image = Image.fromarray(t, mode="RGB")
    path = os.path.join(folder, filename)
    image.save(path)
    print(f"Saved: {path}")

def load_image(path, device = "cpu"):
    image = Image.open(path).convert("RGB")
    t = torch.from_numpy(np.array(image))
    t = t.permute(2, 0, 1)
    return t.float().to(device)