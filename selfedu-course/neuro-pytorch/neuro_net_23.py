import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torchvision
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
        x = self.layer1(x) # First layer
        x = nn.functional.relu(x) # ReLU activation
        x = self.layer2(x) # Second layer
        return x # Outputs logits for 10 classes

model = DigitNN(28 * 28, 32, 10) # A network with 784 inputs, 32 hidden units, and 10 outputs

transforms = tfs.compose([
    tfs.ToImage(), # Convert input to PIL images
    tfs.Grayscale(), # Ensure images are grayscale
    tfs.ToDtype(torch.float32, scale=True), # Normalize to [0, 1]
    RavelTransform(), # Flatten the image into 1D
])

dataset_mnist = torchvision.datasets.MNIST(r'C:\datasets\mnist', download=True, train=True, transform=transforms)
d_train, d_val = data.random_split(dataset_mnist, [0.7, 0.3]) # Split dataset into 70% train, 30% validation
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True) # DataLoader for training
train_data_val = data.DataLoader(d_val, batch_size=32, shuffle=False) # DataLoader for validation

optimizer = optim.Adam(params=model.parameters(), lr=0.01) # Adam optimizer with learning rate 0.01
loss_function = nn.CrossEntropyLoss() # Cross-entropy loss for classification
epochs = 20 # Number of training epochs

loss_lst_val = [] # List to store validation loss over epochs
loss_lst = [] # List to store training loss over epochs

for _e in range(epochs): # Loop through epochs
    model.train() # Set model to training mode
    loss_mean = 0 # Initialize running mean for training loss
    lm_count = 0 # Count number of training batches

    train_tqdm = tqdm(train_data, leave=False) # Progress bar for training
    for x_train, y_train in train_tqdm: # Iterate over training batches
        predict = model(x_train) # Forward pass
        loss = loss_function(pre)