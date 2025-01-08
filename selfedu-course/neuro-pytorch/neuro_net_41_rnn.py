import os
import numpy as np
import re

from PIL import Image
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
from torch.utils.data import BatchSampler, SequentialSampler
import torch.utils.data as data
import torchvision
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim

# Custom Dataset for Character-Based Text Prediction
class CharsDataset(data.Dataset):
    def __init__(self, path, prev_chars=3):
        """
        Initializes the dataset by loading text, preprocessing it, and preparing one-hot encodings.

        :param path (str): Path to the text file containing training data.
        :param prev_chars (int): Number of previous characters to consider for predicting the next one.
        """
        self.prev_chars = prev_chars

        # Load and preprocess the text
        with open(path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            self.text = self.text.replace('\ufeff', '') # Remove invisible characters
            self.text = re.sub(r'[^A-Za-zA-я0-9.,?;: ]', '', self.text) # Keep only valid characters

        self.text = self.text.lower()
        self.alphabet = set(self.text) # Unique characters in the text

        # Map characters to indices and vice versa
        self.int_to_alpha = dict(enumerate(sorted(self.alphabet)))
        self.alpha_to_int = {b: a for a, b in self.int_to_alpha.items()}

        self.num_characters = len(self.alphabet) # Size of the character set
        self.onehots = torch.eye(self.num_characters) # One-hot encodings for all characters

    def __getitem__(self, item):
        """
        Retrieves a sequence of previous characters and the target character for a given index.

        :param item (int): Index of the target character.
        :return:
            torch.Tensor: One-hot encoded representation of previous characters.
            int: Index of the target character.
        """
        # Create a sequence of one-hot encodings for previous characters
        _data = torch.vstack([self.onehots[self.alpha_to_int[self.text[x]]] for x in range(item, item+self.prev_chars)])
        # Target character index
        t = self.alpha_to_int[self.text[item+self.prev_chars]]
        return _data, t

    def __len__(self):
        """
        :return: Returns the number of valid samples in the dataset.
        """
        return len(self.text) - 1 - self.prev_chars

# Recurrent Neural Network Model for Text Prediction
class TextRNN(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Initializes the RNN model with input and output layers.

        :param in_features (int): Number of input features (size of one-hot encoding).
        :param out_features (int): Number of output features (size of character set).
        """
        super().__init__()
        self.hidden_size = 64 # Size of the RNN hidden state
        self.in_features = in_features
        self.out_features = out_features

        # Define RNN and output layers
        self.rnn = nn.RNN(in_features, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, out_features)

    def forward(self, x):
        """
        Performs a forward pass through the model.

        :param x (torch.Tensor): Input tensor with shape (batch_size, seq_len, in_features).
        :return: torch.Tensor: Output logits for the mext character prediction.
        """
        x, h = self.rnn(x) # Process input through the RNN
        y = self.out(h) # Compute output logits from the hidden state
        return y

# Load the dataset
train_path = "train_data_true"
d_train = CharsDataset(train_path, prev_chars=10)
train_data = data.DataLoader(d_train, batch_size=8, shuffle=False)

# Initialize the model
model = TextRNN(d_train.num_characters, d_train.num_characters)

# Define optimizer and loss function
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

# Training loop
epochs = 100
model.train()

for _e in range(epochs):
    loss_mean = 0 # Track average loss
    lm_count = 0 # Count processed batches

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train).squeeze(0) # Forward pass
        loss = loss_func(predict, y_train.long()) # Compute loss

        optimizer.zero_grad() # Clear gradients
        loss.backward() # Backpropagate
        optimizer.step() # Update model parameters

        # Update running average of loss
        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")

# Save the trained model
# model.eval()
predict = "Мой дядя самых".lower() # Initial text
num_predictions = 40 # Number of characters to generate

for _ in range(num_predictions):
    # Prepare input sequence
    _data = torch.vstack([d_train.onehots[d_train.alpha_to_int[predict[-x]]] for x in range(d_train.prev_chars, 0, -1)])
    # Predict the next character
    p = model(_data.unsqueeze(0)).squeeze(0)
    indx = torch.argmax(p, dim=1)
    predict += d_train.int_to_alpha[indx.item()]

# Output generated text
print(predict)




