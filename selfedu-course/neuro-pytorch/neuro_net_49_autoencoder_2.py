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
        Initializes the Autoencoder model with Batch Normalization.

        :param input_dim (int): Dimensionality of the input (e.g., 28x28 = 784 for MNIST).
        :param output_dim (int): Dimensionality of the output (same as input_dim for reconstruction).
        :param hidden_dim (int): Dimensionality of the latent (compressed) representation.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Encoder: Compress input to latent representation with Batch Normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128, bias=False),     # Input to 128 units
            nn.ELU(inplace=True),                                  # Activation function
            nn.BatchNorm1d(128),                                   # Batch Normalization
            nn.Linear(128, 64, bias=False), # Compress to 64 units
            nn.ELU(inplace=True),                                  # Activation function
            nn.BatchNorm1d(64),                                    # Batch Normalization
            nn.Linear(64, self.hidden_dim)              # Final bottleneck to hidden_dim
        )

        # Decoder: Reconstruct data from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),  # Expand latent to 64 units
            nn.ELU(inplace=True),                        # Activation function
            nn.Linear(64, 128),   # Expand to 128 units
            nn.ELU(inplace=True),                        # Activation function
            nn.Linear(128, output_dim),       # Reconstruct to input_dim
            nn.Sigmoid()                                 # Constrain output to [0, 1]
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        :param x (torch.Tensor): Input tensor
        :return:
            tuple: Reconstructed input and latent representation.
        """
        h = self.encoder(x) # Pass through encoder
        x = self.decoder(h) # Pass through decoder
        return x, h

# Initialize the autoencoder model
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
train_data = data.DataLoader(d_train, batch_size=100, shuffle=True) # DataLoader for training data

# Define optimizer and loss function
optimizer = optim.Adam(params=model.parameters(), lr=0.001) # Adam optimizer
loss_func = nn.MSELoss() # Mean Squared Error (MSE) loss for reconstruction

# Training loop
epochs = 5 # Number of training epochs
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
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.4f}")

# Save the trained model
# st = model.state_dict()
# torch.save(st, 'model_vae_2.tar')

# Uncomment the following lines to load a pre-trained model:
st = torch.load('model_vae_2.tar', weights_only=True)
model.load_state_dict(st)

model.eval() # Set model to evaluation mode

# Load MNIST test dataset
d_test = torchvision.datasets.MNIST(
    r'C:\datasets\mnist', download=True, train=False, transform=transforms
)
x_data = transforms(d_test.data).view(len(d_test), -1) # Flatten test data

# Encode test data to latent space
h = model.encoder(x_data)
h = h.detach().numpy() # Convert to numpy array for visualization

# Visualize latent space as a scatter plot
# plt.scatter(h[:, 0], h[:, 1], alpha=0.5, s=10, c=d_test.targets.numpy(), cmap='tab10')
# plt.colorbar(label='Digit Label')
# plt.title("Latent Space Representation")
# plt.xlabel("Latent Dimension 1")
# plt.ylabel("Latent Dimension 2")
# plt.grid()

# Uncomment to decode a custom latent vector and visualize the output image:
h_custom = torch.tensor([-40, -10], dtype=torch.float32)
predict = model.decoder(h_custom.unsqueeze(0))
predict = predict.detach().squeeze(0).view(28, 28)
dec_img = predict.numpy()
plt.figure()
plt.imshow(dec_img, cmap='gray')
plt.title("Decoded Image from Latent Vector")

plt.show()
