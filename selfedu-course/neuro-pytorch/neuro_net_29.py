import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.transforms import v2 as tfs
from tqdm import tqdm

# SunDataset class definition
class SunDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, "train" if train else "test")
        self.transform = transform

        # Load format.json for file structure
        with open(os.path.join(self.path, ), "r") as fp:
            self.format = json.load(fp)

        self.length = len(self.format)
        self.files = tuple(self.format.keys())
        self.targets = tuple(self.format.values())

    def __getitem__(self, item):
        # Load image and apply transformations
        path_file = os.path.join(self.path, self.files[item])
        img = Image.open(path_file).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Return image and corresponding target
        return img, torch.tensor(self.targets[item], dtype=torch.float32)

    def __len__(self):
        return self.length

# Define the model architecture
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding="same"), # Convolutional layer with 32 filters
    nn.ReLU(),                                                              # ReLU activation
    nn.MaxPool2d(2),                                                        # Max pooling
    nn.Conv2d(32, 8, 3, padding="same"), # Convolutional layer with 8 filters
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 4, 3, padding="same"),  # Convolutional layer with 4 filters
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),                                                           # Flatten for dense layers
    nn.Linear(4096, 128),                            # Fully connected layer
    nn.ReLU(),
    nn.Linear(128, 2)                                # Output layer with 2 neurons
)

# Define data transformations
transforms = tfs.Compose([
    tfs.ToImage(), # Convert to PIL Image
    tfs.ToDtype(torch.float32, scale=True) # Scale pixel values to [0, 1]
])

# Load training data
d_train = SunDataset("dataset_reg_gen", transform=transforms)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

# Optimizer and loss function
optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
loss_function = nn.MSELoss()

# Training loop
epochs = 5
loss_lst = []

model.train()
for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        # Forward pass
        predict = model(x_train)
        loss = loss_function(predict, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update loss
        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")

    loss_lst.append(loss_mean)
    print(f"Epoch [{_e+1}/{epochs}] - Training Loss: {loss_mean:.4f}")

# Save model state
torch.save(model.state_dict(), "model_sun_2.tar")

# Load testing data
d_test = SunDataset("dataset_seg", train=False, transform=transforms)
test_data = data.DataLoader(d_test, batch_size=50, shuffle=False)

# Testing loop
Q = 0
count = 0
model.eval()

test_tqdm = tqdm(test_data, leave=True)
for x_test, y_test in test_tqdm:
    with torch.no_grad():
        # Forward pass
        p = model(x_test)
        Q += loss_function(p, y_test).item()
        count += 1

# Calculate mean test loss
Q /= count
print(f"Test Loss: {Q:.4f}")

# Plot training loss
plt.plot(loss_lst, label="Training Loss")
plt.grid()
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.show()