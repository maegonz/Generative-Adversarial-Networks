import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output  # Pour effacer et mettre à jour le graphique
import glob
from PIL import Image
import os
import random

def set_seed(seed):
    """seeds every package that might introduce randomness."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_gif(image_dir: str, output_path: str, duration_ms: int=200, image_pattern: str='decision_boundary'):
    """
    Create a GIF from images in a directory.
    
    Parameters
    ----------
    image_dir : str
        Directory containing the images to be included in the GIF.
    output_path : str
        Path where the GIF will be saved.
    duration_ms : int, optional
        Duration of each frame in milliseconds. Default is 200 ms.
    """
    image_paths = [
        f'{image_dir}/{image_pattern}_{i}.png' for i in range(1, 33) if glob.glob(f'{image_dir}/{image_pattern}_{i}.png')
        if glob.glob(f'{image_dir}/{image_pattern}_{i}.png')
    ]
    # print(image_paths)
    print(f"Creating GIF from {len(image_paths)} images in {image_dir}...")
    print(f"Saving GIF to {output_path} with frame duration {duration_ms} ms.")
    if not image_paths:
        raise ValueError(f"No images found in directory: {image_dir}")
    
    images = [Image.open(img) for img in image_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0
    )