import torch
import numpy as np
import os
import random as rd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from glob import glob
from typing import Union
from pathlib import Path
from PIL import Image


class CelebaDataset(Dataset):
    def __init__(self,
                 path: str,
                 img_size: int = 64,
                 augment: bool = False,):
        """
        Params
        -------
        path : str
            Path to the dataset directory.
        """

        self.path = path
        # get sorted list of all image files
        self.path_imgs = sorted(glob(os.path.join(f"{self.path}/celeba/img_align_celeba", '*.jpg')))

        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),  # No operation if not augmenting
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.path_imgs)
    
    def __getitem__(self, idx):
        img = Image.open(self.path_imgs[idx])
        img = self.transform(img)

        return {
            "image": img,
        }