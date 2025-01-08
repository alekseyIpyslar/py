import os
import json
from PIL import Image

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

    def __len__(self):
        return self.length # Return the total number of images

# Load pre-trained ResNet50 model weights
resnet_weights = models.ResNet50_Weights.DEFAULT
# Define image transformations consistent with ResNet50's requirements
transforms = resnet_weights.transforms()

# Initialize the pre-trained ResNet50 model
model = models.resnet50(weights=resnet_weights)
model.requires_grad_(False) # Freeze all layers
# Replace the fully connected layer to match the dataset (10 classes)
model.fc = nn.Linear(512 * 4, 10) # ResNet50 outputs 2048-dimensional features
model.fc.requires_grad_(True) # Only train the new fully connected layer

# Create the training dataset and data loader
d_train = DogDataset(r"C:/Users/aleks/PycharmProjects/py/selfedu-course/dataset/dogs", transform=transforms)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

# Set up the optimizer for the trainable parameters (only `fc`)
optimizer = optim.Adam(params=model.fc.parameters(), lr=0.001, weight_decay=0.001)
# Cross-entropy loss for multi-class classification
loss_function = nn.CrossEntropyLoss()
epochs = 3 # Number of training epochs

# Set the model to training mode
model.train()

# Training loop
for _e in range(epochs):
    loss_mean = 0 # Track the average loss
    lm_count = 0 # Count batches processed

    train_tqdm = tqdm(train_data, leave=True) # Progress bar for training
    for x_train, y_train in train_tqdm:
        predict = model(x_train) # Forward pass
        loss = loss_function(predict, y_train) # Compute loss

        optimizer.zero_grad() # Clear gradients
        loss.backward() # Backpropagate
        optimizer.step() # Update weights

        # Update average loss
        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

# Save the model's state dictionary
torch.save(model.state_dict(), "model_transfer_resnet.tar")

# Load the testing dataset and data loader
d_test = DogDataset(r"C:/Users/aleks/PycharmProjects/py/selfedu-course/dataset/dogs", train=False, transform=transforms)
test_data = data.DataLoader(d_test, batch_size=50, shuffle=False)

# Initialize metrix for testing
Q = 0 # Average loss
P = 0 # Accuracy
count = 0 # Batch count

# Set the model to evaluation mode
model.eval()

# Testing loop
test_tqdm = tqdm(test_data, leave=True) # Progress bar for testing
for x_test, y_test in test_tqdm:
    with torch.no_grad(): # No gradient computation during testing
        p = model(x_test) # Forward pass
        p2 = torch.argmax(p, dim=1) # Predicted class indices
        y = torch.argmax(y_test, dim=1) # True class indices
        P += torch.sum(p2 == y).item() # Increment correct predictions
        Q += loss_function(p, y_test).item() # Accumulate loss
        count += 1

# Compute average loss and accuracy
Q /= count
P /= len(d_test)

# Print final metrics
print(f"Average Loss: {Q:.4f}")
print(f"Accuracy: {P:.4%}")