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
        self.layer1 = nn.Linear(input_dim, num_hidden) # Fully connected layer (input to hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim) # Fully connected layer (hidden to output)
        self.dropout_1 = nn.Dropout(0.3) # Dropout with a probability of 0.3

    def forward(self, x):
        x = self.layer1(x) # Forward pass through first layer
        x = nn.functional.relu(x) # Apply ReLU activation
        x = self.dropout_1(x) # Apply dropout
        x = self.layer2(x) # Forward pass through second layer
        return x # Return logits for classification

model = DigitNN(28 * 28, 128, 10)

transforms = tfs.Compose([
    tfs.ToImage(), # Convert to PIL Image
    tfs.Grayscale(), # Ensure grayscale input
    tfs.ToDtype(torch.float32, scale=True), # Normalize to [0, 1]
    RavelTransform(), # Flatten the image
])

dataset_mnist = torchvision.datasets.MNIST(r'C:\datasets\mnist', train=True, transform=transforms, download=True)
d_train, d_val = data.random_split(dataset_mnist, [0.7, 0.3]) # 70% training, 30% validation
train_data = data.DataLoader(d_train, batch_size=32, shuffle=False)
train_data_val = data.DataLoader(d_val, batch_size=32, shuffle=False)

optimizer = optim.Adam(params=model.parameters(), lr=0.01) # , weight_decay=0.001) # Adam optimizer
loss_function = nn.CrossEntropyLoss() # Cross-entropy loss
epochs = 20 # Number of training epochs

loss_lst_val = [] # Track validation loss
loss_lst = [] # Track training loss

for _e in range(epochs):
    model.train() # Set model to training mode
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=False)
    for x_train, y_train in train_tqdm:
        predict = model(x_train) # Forward pass
        loss = loss_function(predict, y_train) # Compute training loss

        optimizer.zero_grad() # Clear gradients
        loss.backward() # Back propagate
        optimizer.step() # Update weights

        # Update running loss mean
        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e + 1}] loss_mean={loss_mean:.3f}")

    model.eval() # Set model to evaluation mode
    Q_val = 0
    count_val = 0

    for x_val, y_val in train_data_val:
        with torch.no_grad(): # Disable gradient calculation
            p = model(x_val) # Forward pass
            loss = loss_function(p, y_val) # Compute validation loss
            Q_val += loss.item() # Accumulate validation loss
            count_val += 1

    Q_val /= count_val # Average validation loss

    loss_lst.append(loss_mean) # Append training loss
    loss_lst_val.append(Q_val) # Append validation loss

    print(f" | loss_mean={loss_mean:.3f}, Q_val={Q_val:.3f}")

d_test =  torchvision.datasets.MNIST(r'C:\datasets\mnist', download=True, train=False, transform=transforms)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)

Q = 0 # Correct predictions count

model.eval() # Set model to evaluation mode

for x_test, y_test in test_data:
    with torch.no_grad(): # Disable gradient computation
        p = model(x_test) # Forward pass
        p = torch.argmax(p, dim=1) # Get predicted classes
        Q += torch.sum(p == y_test).item() # Count correct predictions

Q /= len(d_test) # Calculate test accuracy
print(Q) # Print test accuracy

plt.plot(loss_lst, label="Training Loss") # Training loss
plt.plot(loss_lst_val, label="Validation Loss") # Validation loss
plt.grid() # Add grid
plt.legend() # Add legend
plt.show() # Display plot
