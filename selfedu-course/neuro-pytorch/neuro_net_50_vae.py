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

# Define Variational Autoencoder (VAE) for MNIST
class AutoEncoderMNIST(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Initializes the Variational Autoencoder (VAE) model.

        :param input_dim (int): Dimensionality of the input (28x28=784 for MNIST).
        :param output_dim (int): Dimensionality of the output (same as input_dim for reconstruction).
        :param hidden_dim (int): Dimensionality of the latent representation.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Encoder: Compress input into a latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64)
        )

        # Latent representation (mean and log variance)
        self.h_mean = nn.Linear(64, self.hidden_dim)
        self.h_log_var = nn.Linear(64, self.hidden_dim)

        # Decoder: Reconstruct input from the latent representation
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the VAE.

        :param x (torch.Tensor): Input tensor.
        :return:
            tuple: Reconstructed output, latent vector, mean, and log variance.
        """
        enc = self.encoder(x) # Encode input

        # Compute mean and log variance
        h_mean = self.h_mean(enc)
        h_log_var = self.h_log_var(enc)

        # Re-parameterization trick: sample latent vector
        noise = torch.normal(mean=torch.zeros_like(h_mean), std=torch.ones_like(h_log_var))
        h = noise * torch.exp(h_log_var / 2) + h_mean

        # Decode latent vector to reconstruct input
        x = self.decoder(h)
        return x, h, h_mean, h_log_var

# Custom loss function for VAE
class VAELoss(nn.Module):
    def forward(self, x, y, h_mean, h_log_var):
        """
        Compute VAE loss (reconstruction + KL divergence).

        :param x (torch.Tensor): Reconstructed input.
        :param y (torch.Tensor): Original input.
        :param h_mean (torch.Tensor): Mean of latent distribution.
        :param h_log_var (torch.Tensor): Log variance of latent distribution.
        :return:
            torch.Tensor: Combined loss.
        """
        img_loss = torch.sum(torch.square(x - y), dim=-1) # Reconstruction loss
        kl_loss = -0.5 * torch.sum(1 + h_log_var - torch.square(h_mean) - torch.exp(h_log_var), dim=-1) # KL divergence
        return torch.mean(img_loss + kl_loss)

# Initialize the VAE model
model = AutoEncoderMNIST(784, 784, 2) # Input/output size = 28x28 flattened, latent size = 2

# Define data transformations
transforms = tfs_v2.Compose([
    tfs_v2.ToImage(), # Convert data to image format
    tfs_v2.ToDtype(dtype=torch.float32, scale=True), # Normalize to [0, 1]
    tfs_v2.Lambda(lambda _img: _img.ravel()) # Flatten the image to 1D
])

# Load MNIST dataset with transformations
d_train = torchvision.datasets.MNIST(
    r'C:\datasets\mnist', download=True, train=True, transform=transforms
)
train_data = data.DataLoader(d_train, batch_size=100, shuffle=True)

# Define optimizer and loss function
optimizer = optim.Adam(params=model.parameters(), lr=0.001) # Adam optimizer
loss_func = VAELoss()

# Training loop
epochs = 5 # Number of epochs
model.train() # Set model to training mode

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True) # Progress bar
    for x_train, y_train in train_tqdm:
        predict, _, h_mean, h_log_var = model(x_train) # Forward pass
        loss = loss_func(predict, x_train, h_mean, h_log_var) # Compute loss

        optimizer.zero_grad() # Clear gradients
        loss.backward() # Backpropagate
        optimizer.step() # Update weights

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e + 1}] Loss: {loss_mean:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'model_vae_3.tar')

# Uncomment to load a pre-trained model:
# model.load_state_dict(torch.load('model_vae_3.tar))

model.eval() # Set model to evaluation mode

# Visualize latent space representation
d_test = torchvision.datasets.MNIST(
    r'C:\datasets\mnist', download=True, train=False, transform=transforms
)

x_data = transforms(d_test.data).view(len(d_test), -1) # Flatten test data

_, h, _, _ = model(x_data) # Encode test data
h = h.detach().numpy() # Convert latent representation to numpy

# Scatter plot of latent space
plt.scatter(h[:, 0], h[:, 1], alpha=0.5, s=10, c=d_test.targets.numpy(), cmap='tab10')
plt.colorbar(label='Digit Label')
plt.title("Latent Space Representation")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.grid()

# Generate and visualize a grid of reconstructed images from latent space
n = 5
total = 2 * n + 1
plt.figure(figsize=(total, total))

num = 1
for i in range(-n, n + 1):
    for j in range(-n, n + 1):
        ax = plt.subplot(total, total, num)
        num += 1
        h = torch.tensor([3 * i / n, 3 * j / n], dtype=torch.float32)
        predict = model.decoder(h.unsqueeze(0))
        predict = predict.detach().squeeze(0).view(28, 28)
        dec_img = predict.numpy()

        plt.imshow(dec_img, cmap='gray')
        ax.axis('off')

plt.show()