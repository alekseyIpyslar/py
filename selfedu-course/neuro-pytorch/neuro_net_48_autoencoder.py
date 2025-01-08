import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim

# Autoencoder for MNIST Data
class AutoEncoderMNIST(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Initializes the AutoEncoder model.

        :param input_dim (int): Dimensionality of the input data (e.g., flattened image size)
        :param output_dim (int): Dimensionality of the output data (same as input_dim for reconstruction).
        :param hidden_dim (int): Dimensionality of the latent (bottleneck) representation.
        """
        super().__init__()
        self.hidden_dim = hidden_dim # Latent representation size

        # Encoder: Compresses the input to the latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),     # First linear layer
            nn.ELU(inplace=True),                      # Activation function
            nn.Linear(128, 64), # Second linear layer
            nn.ELU(inplace=True),                      # Activation function
            nn.Linear(64, self.hidden_dim)  # Bottleneck layer
        )

        # Decoder: Reconstructs the input from the latent representation
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),  # First linear layer
            nn.ELU(inplace=True),                        # Activation function
            nn.Linear(64, 128),   # Second linear layer
            nn.ELU(inplace=True),                        # Activation function
            nn.Linear(128, output_dim),       # Output layer
            nn.Sigmoid()                                 # Output activation to constrain values between 0 and 1
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        :param x (torch.Tensor): Input tensor.
        :return:
            tuple: Reconstructed input and latent representation
        """
        h = self.encoder(x) # Encode input to latent representation
        x = self.decoder(h) # Decode latent representation to reconstructed input
        return x, h

# Initialize the model
model = AutoEncoderMNIST(784, 784, 28) # Input/output size = 28x28 (flattened), latent size = 28

# Define data transformations
transforms = tfs_v2.Compose([
    tfs_v2.ToImage(),                                # Convert to image
    tfs_v2.ToDtype(dtype=torch.float32, scale=True), # Normalize to [0, 1]
    tfs_v2.Lambda(lambda _img: _img.ravel())         # Flatten the image
])

# Load MNIST dataset with the defined transformations
d_train = torchvision.datasets.MNIST(
    r'C:\datasets\mnist', download=True, train=True, transform=transforms
)
train_data = data.DataLoader(d_train, batch_size=100, shuffle=True) # Data loader for training

# Define optimizer and loss function
optimizer = optim.Adam(params=model.parameters(), lr=0.001) # Adam optimizer with learning rate 0.001
loss_func = nn.MSELoss() # Mean Squared Error (MSE) loss for reconstruction

# Training loop
epochs = 5 # Number of training epochs
model.train() # Set model to training mode

for _e in range(epochs):
    loss_mean = 0 # Track average loss
    lm_count = 0 # Count processed batches

    train_tqdm = tqdm(train_data ,leave=True) # Progress bar for training
    for x_train, y_train in train_tqdm:
        predict, _ = model(x_train) # Forward pass
        loss = loss_func(predict, x_train) # Compute reconstruction loss

        optimizer.zero_grad() # Clear gradients
        loss.backward() # Backpropagate loss
        optimizer.step() # Update model parameters

        # Update running average of loss
        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")

# Save the trained model
st = model.state_dict()
torch.save(st, 'model_autoencoder.tar') # Save model state dictionary

# Visualize reconstructed images
n = 10 # Number of images to visualize
model.eval() # Set model to evaluation mode

plt.figure(figsize=(2 * n, 2 * 2)) # Set figure size
for i in range(n):
    img, _ = d_train[i] # Get original image from the dataset
    predict, _ = model(img.unsqueeze(0)) # Reconstruct image with the autoencoder

    # Reshape tensors for visualization
    predict = predict.squeeze(0).view(28, 28) # Reshape reconstructed image
    img = img.view(28, 28) # Reshape original image

    # Convert tensors to numpy arrays for plotting
    dec_img = predict.detach().numpy()
    img = img.detach().numpy()

    # Plot original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(img, cmap='gray') # Display original image in grayscale
    ax.get_xaxis().set_visible(False) # Hide x-axis
    ax.get_yaxis().set_visible(False) # HIde y-axis

    # Plot reconstructed image
    ax2 = plt.subplot(2, n, i + n + 1)
    plt.imshow(dec_img, cmap='gray')
    ax2.get_xaxis().set_visible(False) # Hide x-axis
    ax2.get_yaxis().set_visible(False) # Hide y-axis

plt.show() # Display the plot
