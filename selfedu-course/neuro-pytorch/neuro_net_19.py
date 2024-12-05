import os
import json

from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as tfs


# Custom Dataset for loading digit images
class DigitDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, "train" if train else "test") # Define path for training or test set
        self.transform = transform

        # Load format.json to map directory names to target labels
        with open(os.path.join(path, "format.json"), "r") as fp:
            self.format = json.load(fp)

        self.length = 0 # Initialize length of dataset
        self.files = [] # List to store file paths and their corresponding targets
        self.targets = torch.eye(10) # One-hot encoding for digits (0-9)

        # Iterate over directories and build the dataset
        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path) # List all files in the directory
            self.length += len(list_files) # Increment total dataset length
            #Add full file paths and targets to the files list
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

    # Method to retrieve a single data item
    def __getitem__(self, item):
        path_file, target = self.files[item] # get the file path and its target
        t = self.targets[target] # Convert target to one-hot encoding
        img = Image.open(path_file) # Open the image file

        # Apply transformations if defined
        if self.transform:
            img = self.transform(img).ravel().float() / 255.0 # Flatten and normalize image to [0, 1]

        return img, t # Return the image and its one-hot encoded label

    # Method to return the total length of the dataset
    def __len__(self):
        return self.length

d_train = DigitDataset(r"C:\Users\aleks\PycharmProjects\py\selfedu-course\dataset")
# # Transformation to convert image to tensor
# to_tensor = tfs.ToImage() # Equivalent to PILToTensor
# d_train = DigitDataset(train=True, transform=to_tensor) # Create training dataset instance
# train_data = data.DataLoader(d_train, batch_size=32, shuffle=True) # Dataloader with batch size of 32 and shuffling
#
# # Retrieve a batch of data
# it = iter(train_data)
# x, y = next(it)
# print(len(d_train)) # Print the total number of samples in the dataset