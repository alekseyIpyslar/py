import os
import json
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Custom dataset class for loading digit images
class DigitDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, 'train' if train else 'test')  # Choose train or test directory
        self.transform = transform  # Optional image transformation (e.g., convert to tensor)

        with open(os.path.join(path, 'format.json'), 'r') as fp:  # Load label mapping
            self.format = json.load(fp)

        self.length = 0  # Total number of images
        self.files = []  # List of (image_path, target_label)
        self.targets = torch.eye(10)  # One-hot encoding for digit labels (0-9)

        for _dir, _target in self.format.items():  # Iterate over directories in 'format.json'
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)  # List all image files
            self.length += len(list_files)  # Increment total dataset size
            # Add file paths and corresponding targets
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

    def __getitem__(self, item):
        path_file, target = self.files[item]  # Retrieve file path and target
        t = self.targets[target]  # Convert target to one-hot encoding
        img = Image.open(path_file)  # Open the image

        if self.transform:  # Apply transformations (e.g., convert to tensor, normalize)
            img = self.transform(img).ravel().float() / 255.0

        return img, t  # Return the processed image and label

    def __len__(self):
        return self.length  # Return total number of images in the dataset

    class DigitNN(nn.Module):
        def __init__(self, input_dim, num_hidden, output_dim):
            super().__init__()
            self.layer1 = nn.Linear(input_dim, num_hidden)  # First Layer: Input to hidden
            self.layer2 = nn.Linear(num_hidden, output_dim)  # Second Layer: Hidden to output

    def forward(self, x):
        x = self.layer1(x)  # Apply first linear transformation
        x = nn.functional.relu(x)  # Apply ReLU activation
        x = self.layer2(x)  # Apply second linear transformation
        return x  # Output logits for 10 classes

    model = DigitNN(28 * 28, 32, 10)  # Neural network with 784 inputs, 32 hidden units, and 10 outputs
    to_tensor = tfs.ToImage()  # Convert images to tensors
    d_train = DigitDataset("dataset", transform=to_tensor)  # Create training dataset
    train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)  # DataLoader for batch and shuffling

    optimizer = optim.Adam(params=model.parameters(), lr=0.01)  # Adam optimizer with learning rate 0.01
    loss_function = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    epochs = 2  # Number of training epochs
    model.train()  # Set model to training mode

    for _e in range(epochs):  # Loop over epochs
        loss_mean = 0  # Running mean loss
        lm_count = 0  # Count batches

        train_tqdm = tqdm(train_data, leave=True)  # Create progress bar
        for x_train, y_train in train_tqdm:  # Loop over batches
            predict = model(x_train)  # Forward pass: predict outputs
            loss = loss_function(predict, y_train)  # Compute loss

            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights

            # Update running mean loss
            lm_count += 1
            loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
            train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

    d_test = DigitDataset("dataset", train=False, transform=to_tensor)  # Create test dataset
    test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)  # DataLoader for test set

    Q = 0  # Accuracy counter

    model.eval()  # Set model to evaluation mode

    for x_test, y_test in test_data:  # Loop over test batches
        with torch.no_grad():  # Disable gradient calculations
            p = model(x_test)  # Predict outputs
            p = torch.argmax(p, dim=1)  # Get predicted classes
            y = torch.argmax(y_test, dim=1)  # Get true classes
            Q += torch.sum(p == y).item()  # Count correct predictions

    Q /= len(d_test)  # Compute accuracy
    print(Q)  # Print accuracy
