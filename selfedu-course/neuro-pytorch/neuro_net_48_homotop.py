import os
import numpy as np
import matplotlib.pyplot as plt
import re

from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim

# Define Autoencoder Model for MNIST Dataset
class AutoEncoderMNIST(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Initialize the Autoencoder model.

        :param input_dim (int): Dimensionality of the input (e.g., flattened 28x28 image size).
        :param output_dim (int): Dimensionality of the output (same as input_dim for reconstruction).
        :param hidden_dim (int): Dimensionality of the latent (compressed) representation.
        """
        super().__init__()
        self.hidden_dim = hidden_dim # Latent representation size

        # Encoder: Maps input data to a latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),     # Compress input to 128 dimensions
            nn.ELU(inplace=True),                      # Activation function
            nn.Linear(128, 64), # Compress further to 64 dimensions
            nn.ELU(inplace=True),                      # Activation function
            nn.Linear(64, self.hidden_dim)  # Final bottleneck layer to hidden_dim
        )

        # Decoder: Reconstructs data from the latent representation
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),     # Expand latent to 64 dimensions
            nn.ELU(inplace=True),                           # Activation function
            nn.Linear(64, 128),      # Expand further to 128 dimensions
            nn.ELU(inplace=True),                           # Activation function
            nn.Linear(128, output_dim),          # Reconstruct output to match input size
            nn.Sigmoid()                                    # Constrain output values to [0, 1]
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        :param x (torch.Tensor): Input tensor.

        :return:
            tuple: Reconstructed input and latent representation.
        """
        h = self.encoder(x) # Pass input through encoder
        x = self.decoder(h) # Pass latent representation through decoder
        return x, h

# Initialize the autoencoder model
model = AutoEncoderMNIST(784, 784, 28) # Input/output size = 28x28 flattened, latent size = 28

# Define data transformations
transforms = tfs_v2.Compose([
    tfs_v2.ToImage(),                       # Convert data to image format
    tfs_v2.ToDtype(dtype=torch.float32, scale=True), # Normalize to [0, 1]
    tfs_v2.Lambda(lambda _img: _img.ravel())       # Flatten the image to 1D
])

# Load MNIST dataset with the specified transformations
d_train = torchvision.datasets.MNIST(
    r'C:\datasets\mnist', download=True, train=True, transform=transforms
)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True) # DataLoader for batched data

# Define optimizer and loss function
optimizer = optim.Adam(params=model.parameters(), lr=0.001) # Adam optimizer
loss_func = nn.MSELoss() # Mean Squared Error (MSE) loss for reconstruction

# Training loop
epochs = 5 # Number of epochs for training
model.train() # Set model to training mode

for _e in range(epochs):
    loss_mean = 0 # Track average loss
    lm_count = 0 # Count processed batches

    train_tqdm = tqdm(train_data, leave=True) # Progress bar for training
    for x_train, y_train in train_tqdm:
        predict, _ = model(x_train) # Forward pass through the model
        loss = loss_func(predict, x_train) # Compute reconstruction loss

        optimizer.zero_grad() # Clear gradients
        loss.backward() # Backpropagate loss
        optimizer.step() # Update model weights

        # Update running average of the loss
        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean: {loss_mean:.4f}")

# Save model parameters to a file
# torch.save(model.state_dict(), 'model_autoencoder.tar')

# Load pre-trained model parameters from a file
st = torch.load('model_autoencoder.tar', weights_only=True)
model.load_state_dict(st)

# Visualization of Homotopy Transformation
n = 10 # Number of interpolated images to generate
model.eval() # Set model to evaluation mode

plt.figure(figsize=(2 * n, 2 * 2)) # Set plot size for visualization

# Homotopy between two images from the dataset
frm, to = d_train.data[d_train.targets == 5][10:12] # Select two images of digit '5'
frm = transforms(frm) # Apply transformations to the first image
to = transforms(to)   # Apply transformations to the second image

# Generate interpolated images between frm and to
for i, t in enumerate(np.linspace(0., 1., n)):
    img = frm * (1 - t) + to * t # Linear interpolation
    predict, _ = model(img.unsqueeze(0)) # Reconstruct interpolated image using the model
    predict = predict.squeeze(0).view(28, 28) # Reshape output for visualization
    dec_img = predict.detach().numpy()
    img = img.view(28, 28).numpy()

    # Plot original interpolated image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(img, cmap='gray')
    ax.get_xaxis().set_visible(False) # Hide x-axis
    ax.get_yaxis().set_visible(False) # Hide y-axis

    # Plot reconstructed interpolated image
    ax2 = plt.subplot(2, n, i + n + 1)
    plt.imshow(dec_img, cmap='gray')
    ax2.get_xaxis().set_visible(False) # Hide x-axis
    ax2.get_yaxis().set_visible(False) # Hide y-axis

plt.show() # Display the plots