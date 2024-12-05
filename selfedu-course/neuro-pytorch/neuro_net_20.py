import os
import json
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Custom dataset class for loading digit images
class DigitDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, 'train' if train else 'test') # Choose train or test directory
        self.transform = transform # Optional image transformation (e.g., convert to tensor)

        with open(os.path.join(path, 'format.json'), 'r') as fp:  #Load label mapping
            self.format = json.load(fp)

        self.length = 0  # Total number of images
        self.files = []  # List of (image_path, target_label)
        self.targets = torch.eye(10) # One-hot encoding for digit labels (0-9)

        for _dir, _target in self.format.items(): # Iterate over directories in 'format.json'
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path) # List all image files
            self.length += len(list_files) # Increment total dataset size
            # Add file paths and corresponding targets
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

    def __getitem__(self, item):
        path_file, target = self.files[item] # Retrieve file path and target
        t = self.targets[target] # Convert target to one-hot encoding
        img = Image.open(path_file) # Open the image

        if self.transform: # Apply transformations (e.g., convert to tensor, normalize)
            img = self.transform(img).ravel().float() / 255.0

        return img, t # Return the processed image and label

    def __len__(self):
        return self.length # Return total number of images in the dataset

    class DigitNN(nn.Module):
        def __init__(self, input_dim, num_hidden, output_dim):
            super().__init__()
            self.layer1 = nn.Linear(input_dim, num_hidden) # First Layer: Input to hidden
            self.layer2 = nn.Linear(num_hidden, output_dim) # Second Layer: Hidden to output