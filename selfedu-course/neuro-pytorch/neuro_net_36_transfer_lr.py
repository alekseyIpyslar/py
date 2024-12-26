import os
import json
form PIL import Image

import torch
import torch.utils.data as data
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Custom dataset for loading dog images
class DogDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        # Determine the directory for training or testing data
        self.path = os.path.join(path, "train" if train else "test")
        self.transform = transform # Transformations to apply to the images

        # Load the format.json file, which maps folder names to class indices
        with open(os.path.join(self.path, "format.json"), "r") as fp:
            self.format = json.load(fp)

        self.length = 0 # Total number of images in the dataset
        self.files = [] # List to store file paths and corresponding targets
        self.targets = torch.eye(10) # One-hot encoding for 10 classes (10x10 identity matrix)

        # Populate the files list with image paths and corresponding class indices
        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir) # Directory for the current class
            list_files = os.listdir(path) # list all files in the directory
            self.length += len(list_files) # Update total image count
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

    def __getitem__(self, item):
        # Fetch the file path and target for the given index
        path_file, target = self.files[item]
        t = self.targets[target] # Convert class index to one-hot encoding
        img = Image.open(path_file) # Load the image

        # Appy transformations if specified
        if self.transform:
            img = self.transform(img)

        return img, t # Return the transformed image and one-hot target