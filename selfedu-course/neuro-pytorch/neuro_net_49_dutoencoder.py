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

# Define Autoencoder Model for MNIST Dataset
class AutoEncoderMNIST(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Initializes the Autoencoder model.

        :param input_dim (int): Dimensionality of the input (e.g., flattened 28x28 image size).
        :param output_dim (int): Dimensionality of the output (same as input_dim for reconstruction).
        :param hidden_data (int): Dimensionality of the latent (compressed) representation.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Encoder: Compress input to latent representation
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), # Expand latent to 64 units
            nn.ELU(inplace=True),                       # Activation function
            nn.Linear(128, 64),  # Compress further to 64 units
            nn.ELU(inplace=True),                       # Activation function
            nn.Linear(64, self.hidden_dim)   # Final bottleneck to hidden_dim
        )

        # Decoder: Reconstruct data from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), # Expand latent to 64 units
            nn.ELU(inplace=True),                       # Activation function
            nn.Linear(64, 128),  # Expand to 128 units
            nn.ELU(inplace=True),                       # Activation function
            nn.Linear(128, output_dim),      # Reconstruct to input_dim
            nn.Sigmoid()                                # Constrain output to [0, 1]
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        :param x (torch.Tensor): Input tensor.
        :return:
            tuple: reconstructed input and latent representation.
        """
        h = self.encoder(x) # Pass through encoder
        x = self.decoder(h) # Pass through decoder
        return x, h

# Initialize the autoencoder model
model = AutoEncoderMNIST(784, 784, 2) # Input/output size = 28x28 flattened, latent size = 2

# Define data transformations
transforms = tfs_v2.Compose([
    tfs_v2.ToImage(),                                # Convert data to image format
    tfs_v2.ToDtype(dtype=torch.float32, scale=True), # Normalize to [0, 1]
    tfs_v2.Lambda(lambda _img: _img.ravel())         # Flatten the image to 1D
])

# Load MNIST dataset with the specified transformations
d_train = torchvision.datasets.MNIST(
    r'C:\datasets\mnist', download=True, train=True, transform=transforms
)
train_data = data.DataLoader(d_train, batch_size=100, shuffle=True) # DataLoader for batched data

# Define optimizer and loss function
optimizer = optim.Adam(params=model.parameters(), lr=0.001) # Adam optimizer
loss_func = nn.MSELoss() # Mean Squared Error (MSE) loss for reconstruction

# Training loop
epochs = 0 # Number of epochs for training (set to 0 to skip training)
model.train() # Set model to training mode

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True) # Progress bar for training
    for x_train, y_train in train_tqdm:
        predict, _ = model(x_train) # Forward pass
        loss = loss_func(predict, x_train) # Compute reconstruction loss

        optimizer.zero_grad() # Clear gradients
        loss.backward() # Backpropagate loss
        optimizer.step() # Update weights

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")

# Save and load the model's state
# torch.save (model.state_dict(), 'model_var.tar') # Save the model
st = torch.load('model_vae.tar', weights_only=True) # Load the saved model
model.load_state_dict(st)

model.eval() # Set model to evaluation mode

# Generate a scatter plot of the latent space (commented-out code)
# Uncommented for visualization of the latent space representation:
# d_test = torchvision.datasets.MNIST(r'C:\datasets\mnist', download=True, train=False, transform=transforms)
# x_data = transforms(d_trst.data).view(len(d_test), -1) # Flatten test data
# h = model.encoder(x_data) # Encode test data to latent space
# h = h.detach().numpy() # Convert to numpy array for plotting
# plt.scatter(h[:, 0], h[:, 1]) # Scatter plot of latent space
# plt.grid()

# Generate a decoded image from a custom latent space vector
h = torch.tensor([-40, -20], dtype=torch.float32) # Define a latent space vector
predict = model.decoder(h.unsqueeze(0)) # Decode the vector into an image
predict = predict.detach().unsqueeze(0).view(28, 28) # Reshape output to 28x28
dec_img = predict.numpy() # Convert to numpy array for visualization

# Display the generated image
plt.imshow(dec_img, cmap='gray')
plt.show()