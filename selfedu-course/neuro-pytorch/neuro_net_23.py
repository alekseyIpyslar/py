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

transforms = tfs.Compose([
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
        loss = loss_function(predict, y_train) # Calculate training loss

        optimizer.zero_grad() # Clear gradients
        loss.backward() # Back propagate loss
        optimizer.step() # Update weights

        # Update running mean loss
        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:3f}")

model.eval() # Set model to evaluation mode
Q_val = 0 # Track total validation loss
count_val = 0 # Count number of validation batches

for x_val, y_val in train_data_val: # Iterate over validation batches
    with torch.no_grad(): # Disable gradient computation
        p = model(x_val) # Forward pass
        loss = loss_function(p, y_val) # Calculate validation loss
        Q_val += loss.item() # Accumulate validation loss
        count_val += 1 # Increment batch count

Q_val /= count_val # Average validation loss

loss_lst.append(loss_mean) # Append training loss
loss_lst_val.append(Q_val) # Append validation loss

print(f" | loss_mean={loss_mean:.3f}, Q_val={Q_val:.3f}") # Print losses

d_test = ImageFolder(r"C:\Users\aleks\PycharmProjects\py\selfedu-course\dataset\test", transform=transforms) # Test dataset
test_data = data.DataLoader(d_test, batch_size=32, shuffle=False) # DataLoader for test set

Q = 0 # Count correct predictions
model.eval() # Set model to evaluation mode

for x_test, y_test in test_data: # Iterate over test batches
    with torch.no_grad(): # Disable gradient computation
        p = model(x_test) # Forward pass
        p = torch.argmax(p, dim=1) # Get predicted classes
        Q += torch.sum(p == y_test).item() # Count correct predictions

Q /= len(test_data) # Calculate accuracy
print(Q) # Print test accuracy

plt.plot(loss_lst, label="Training loss") # Plot training loss
plt.plot(loss_lst_val, label="Validation loss") # Plot validation loss
plt.grid() # Add grid to the plot
plt.legend() # Add legend
plt.show() # Display plot