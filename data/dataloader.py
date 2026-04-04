import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from typing import Optional


def dataloader(name: str,
               batch_size: int,
               shuffle: bool=True,
               val_split: Optional[float] = None,
               augment: bool=False,
               img_size: Optional[int] = None):
  """
  Dataloader for the specified dataset.

  Params
  -------
  name: str
    Name of the dataset between 'CIFAR10', 'MNIST' and 'CIFAR100'.
  batch_size: int
    Batch size for the dataloader.
  shuffle: bool
    Whether to shuffle the dataset, defaults to True.
  val_split: float
    Fraction of the trainig set use as validation. Defaults to None.
  augment: bool
    Whether to apply data augmentation techniques. Defaults to False.

  Returns
  -------
  train_loader: DataLoader
    Dataloader for the training set.
  test_loader: DataLoader
    Dataloader for the test set.
  valid_loader: DataLoader
    DataLoader for the validation set.
  """

  # Image transformation
  transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4) if name in ['CIFAR10', 'CIFAR100'] and augment else
      transforms.Resize(28) if name == 'MNIST' else transforms.Resize(img_size) if name == 'CelebA' else transforms.Resize(32),
      transforms.RandomHorizontalFlip() if name in ['CIFAR10', 'CIFAR100'] and augment else
      transforms.CenterCrop(img_size) if name == 'CelebA' else transforms.Lambda(lambda x: x),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,)) if name == 'MNIST' else 
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  # Load dataset
  if name == 'CIFAR10':
    train_set = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
  elif name == 'MNIST':
    train_set = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
  elif name == 'CIFAR100':
    train_set = tv.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_set = tv.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
  elif name == 'CelebA':
    train_set = tv.datasets.CelebA(root='./data', split='train', download=True, transform=transform)
    test_set = tv.datasets.CelebA(root='./data', split='test', download=True, transform=transform)
  else:
    raise ValueError("Invalid dataset name. Choose between 'MNIST', 'CIFAR10', 'CelebA' or 'CIFAR100'.")

  # Create DataLoaders
  if val_split is not None:
    if name == 'CelebA':
      valid_set = tv.datasets.CelebA(root='./data', split='valid', download=True, transform=transform)
      valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
      return train_loader, test_loader, valid_loader
    else:
      # Split trainig data into train and validation
      total_size = len(train_set)
      valid_size = int(val_split * total_size)
      train_size = total_size - valid_size

      train_set, valid_set = random_split(train_set, [train_size, valid_size])
      
      train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
      valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
      test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
      
      return train_loader, test_loader, valid_loader
  
  else:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



def get_tensors(dataloader: DataLoader,
                array: bool=False):
  """
  Get tensors from dataloader.

  Params
  -------
  dataloader: DataLoader
    Dataloader for the dataset.
  array: bool
    Whether to return tensors as numpy arrays or torch tensor, defaults to False.

  Returns
  -------
  x: torch.Tensor
    Images tensor from the dataloader.
  y: torch.Tensor
    Label tensor of the images from the dataloader.
  """

  x, y = [], []

  for images, labels in dataloader:
    x.append(images)
    y.append(labels)

  # Concatenate all batches into single tensors
  if array:
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
  else:
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

  return x, y