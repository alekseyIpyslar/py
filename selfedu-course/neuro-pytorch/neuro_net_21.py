import os
import json
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.datasets import ImageFolder

class RavelTransform(nn.Module):
    def forward(self, item):
        return item.ravel()

    class DigitNN(nn.Module):
        def __init__(self, input_dim, num_hidden, output_dim):
            super().__init__()
            self.layer1 = nn.Linear(input_dim, num_hidden) # Input to hidden layer
            self.layer2 = nn.Linear(num_hidden, output_dim) # Hidden to output layer

    def forward(self, x):
        x = self.layer1(x) # Apply first layer
        x = nn.functional.relu(x) # ReLU activation
        x = self.layer2(x) # Apply second layer
        return x # Return logits for 10 classes

    model = DigitNN(28 * 28, 32, 10) # Neural network with 784 inputs, 32 hidden units, and 10 outputs

    transforms = tfs.Compose([
        tfs.ToImage(), # Convert data to a PIL image
        tfs.Grayscale(), # Ensure images are grayscale
        tfs.ToDtype(torch.float32, scale=True), # Convert to float and normalize to [0, 1]
        RavelTransfrom(), # Flatten the image into a 1D tensor
    ])

    d_train = ImageFolder("dataser/train", transform=transforms) # Load training dataset with transformations
    train_data = data.DataLoader(d_train, batch_size=32, shuffle=True) # Batch and shuffle data

    optimizer = optim.Adam(params=model.parameters(), lr=0.01) # Adam optimizer with learning rate 0.01
    loss_function = nn.CrossEntropyLoss() # Cross-entropy loss for classification
    epochs = 2 # Number of epochs
    model.train() # Set model to training mode

    for _e in range(epochs): # Loop over epochs
        loss_mean = 0 # Track running mean of loss
        lm_count = 0 # Batch count

        train_tqdm = tqdm(train_data, leave=True) # Progress bar for taining
        for x_train, y_train in train_tqdm: # Iterate over training batches
            predict = model(x_train) # Forward pass: model predictions
            loss = loss_function(predict, y_train) # Compute loss

            optimizer.zero_grad() # Clear previous gradients
            loss.backward() # Backpropagation
            optimizer.step() # Update model parameters

            # Update running mean loss
            lm_count += 1
            loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
            train_tqdm.set_desription(f"Epoch [{_e + 1}], loss_mean={loss_mean:.3f}") # Update progress bar

    d_test = ImageFolder("dataset/test", transform=transforms) # Load test dataset with transformations
    test_data = data.DataLoader(d_test, batch_size=500, shuffle=False) # Batch test data

    Q = 0 # Counter for correct predictions

    model.eval() # Set model to evaluation mode

    for x_test, y_test in test_data: # Iterate over test batches
        with torch.no_grad(): # Disable gradient calculations
            p = model(x_test) # Forward pass: model predictions
            p = torch.argmax(p, dim=1) # Get predicted class for each image
            Q += torch.sum(p == y_test).item() # Count correct predictions

    Q /= len(d_test) # Calculate accuracy
    print(Q) # Print accuracy