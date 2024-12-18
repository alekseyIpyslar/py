import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Custom Transform: Flattens images
class RavelTransform(nn.Module):
    def forward(self, item):
        return item.ravel()

# Neural Network with Batch Normalization
class DigitNN(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden, bias=False) # No bias since BN handles it
        self.bm_1 = nn.BatchNorm1d(num_hidden) # Batch Normalization
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x) # Fully connected layer
        x = nn.functional.relu(x) # ReLU activation
        x = self.bm_1(x) # Apply Batch Normalization
        x = self.layer2(x) # Final output layer
        return x

# Initialize model
model = DigitNN(28 * 28, 128, 10)

# Transformations
transforms = tfs.Compose([
    tfs.ToImage(),
    tfs.Grayscale(),
    tfs.ToDtype(torch.float32, scale=True),
    RavelTransform(), # Flatten images to 1D
])

# Load MNIST dataset
dataset_mnist = torchvision.datasets.MNIST(
    r'C:\datasets\mnist', download=True, train=True, transform=transforms
)

d_train, d_val = data.random_split(dataset_mnist, [0.7, 0.3])
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)
train_data_val = data.DataLoader(d_val, batch_size=32, shuffle=False)

# Optimizer and Loss
optimizer = optim.Adam(params=model.parameters(), lr=0.01) # Adam Optimizer
loss_function = nn.CrossEntropyLoss() # Loss Function
epochs = 20 # Number of epochs

# Track training and validation losses
loss_lst_val = []
loss_lst = []

# Training and Validation Loop
for _e in range(epochs):
    model.train()
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=False)
    for x_train, y_train in train_tqdm:
        predict = model(x_train) # Forward pass
        loss = loss_function(predict, y_train) # Compute loss

        optimizer.zero_grad() # Clear gradients
        loss.backward() # Backpropagation
        optimizer.step() # Update weights

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")

    # Validation
    model.eval()
    Q_val = 0
    count_val = 0

    for x_val, y_val in train_data_val:
        with torch.no_grad():
            p = model(x_val) # Forward pass
            loss = loss_function(p, y_val) # Compute validation loss
            Q_val += loss.item()
            count_val += 1

    Q_val /= count_val
    loss_lst.append(loss_mean) # Store training loss
    loss_lst_val.append(Q_val) # Store validation loss

    print(f" | loss_mean={loss_mean:.3f}, Q_val={Q_val:.3f}")

# Test Set Evaluation
d_test = torchvision.datasets.MNIST(
    r'C:\datasets\mnist', download=True, train=False, transform=transforms
)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)

Q = 0
model.eval()

for x_test, y_test in test_data:
    with torch.no_grad():
        p = model(x_test) # Forward pass
        p = torch.argmax(p, dim=1) # Get predicted classes
        Q += torch.sum(p == y_test).item() # Count correct predictions

Q /= len(d_test)
print(f"Test Accuracy: {Q:.3f}")

# Loss Visualization
plt.plot(loss_lst, label="Training Loss")
plt.plot(loss_lst_val, label="Validation Loss")
plt.grid()
plt.legend()
plt.show()

