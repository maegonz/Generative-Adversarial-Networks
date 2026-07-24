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


class MapsDataset(Dataset):
    def __init__(self, root_satellite, root_maps, transform=None):
        self.root_satellite = root_satellite  ## Path to the folder containing satellite images
        self.root_maps = root_maps            ## Path to the folder containing map images

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)), # resize images to w=256 & h=256
                transforms.ToTensor(), # transform numpy array to torch tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # normalize images values to (-1, 1) interval
            ])
        else:
            self.transform = transform

        ## Get lists of all files in the satellite and map directories
        self.satellite_images = sorted(os.listdir(root_satellite))
        self.maps_images = sorted(os.listdir(root_maps))

        ## The length of the dataset is the maximum of the two image sets to handle unequal datasets
        self.length_dataset = max(len(self.maps_images), len(self.satellite_images))

        self.satellite_len = len(self.satellite_images)
        self.maps_len = len(self.maps_images)

    ## Return the length of the dataset
    def __len__(self):
        return self.length_dataset

    ## Get a pair of images (satellite and map) for a given index
    def __getitem__(self, index):

        satellite_img = self.satellite_images[index % self.satellite_len]
        maps_img = self.maps_images[index % self.maps_len]

        ## Construct full paths to the satellite and map images
        satellite_path = os.path.join(self.root_satellite, satellite_img)
        maps_path = os.path.join(self.root_maps, maps_img)

        ## Open the images and convert them to RGB format, then convert to numpy arrays
        satellite_img = Image.open(satellite_path).convert("RGB")
        maps_img = Image.open(maps_path).convert("RGB")

        ## Apply transformations to the images, if provided
        if self.transform:
            satellite_img = self.transform(satellite_img)
            maps_img = self.transform(maps_img)

        return {'satellite_imgs': satellite_img, 'maps_imgs': maps_img}