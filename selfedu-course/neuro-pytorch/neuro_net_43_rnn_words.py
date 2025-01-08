import os
import numpy as np
import re

from navec import Navec
from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

# Custom Dataset for Word-Based Text Prediction
class WordsDataset(data.Dataset):
    def __init__(self, path, navec_emb, prev_words=3):
        """
        Initializes the dataset by loading text, preprocessing it, and preparing word embeddings.

        :param path (str): Path to the text file containing training data.
        :param navec_emb (Navec): Pretrained Navec embeddings.
        :param prev_words (int): Number of previous words to consider for predicting the next one.
        """
        self.prev_words = prev_words
        self.navec_emb = navec_emb

        # Load and preprocess the text
        with open(path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            self.text = self.text.replace('\ufeff', '') # Remove invisible characters
            self.text = self.text.replace('\n', ' ') # Replace newlines with spaces
            self.text = re.sub(r'[^А-яA-z- ]', '', self.text) # Remove invalid characters

        # Split text into words and filter by vocabulary
        self.words = self.text.lower().split()
        self.words = [word for word in self.words if word in self.navec_emb] # Keep only known words

        # Map words to indices and vice versa
        vocab = set(self.words)
        self.int_to_word = dict(enumerate(vocab))
        self.word_to_int = {b: a for a, b in self.int_to_word.items()}
        self.vocab_size = len(vocab) # Size of the vocabulary

    def __getitem__(self, item):
        """
        Retrieves a sequence of previous words and the target word for a given index.
        :param item (int): Index of the target word.
        :return:
            torch.Tensor: Embedding representation of previous words.
            int: Index of the target word.
        """

        # Create a sequence of embeddings for previous words
        _data = torch.vstack([torch.tensor(self.navec_emb[self.words[x]]) for x in range(item, item + self.prev_words)])
        # Target word index
        word = self.words[item + self.prev_words]
        t = self.word_to_int[word]
        return _data, t

    def __len__(self):
        """
        :return: Returns the number of valid samples in the dataset.
        """
        return len(self.words) - 1 - self.prev_words

# Recurrent Neural Network Model for Word Prediction
class WordsRNN(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Initializes the RNN model with input and output layers.

        :param in_features (int): Number of input features (size of word embeddings).
        :param out_features (int): Number of output features (size of vocabulary).
        """
        super().__init__()
        self.hidden_size = 256 # Size of the RNN hidden state
        self.in_features = in_features
        self.out_features = out_features

        # Define RNN and output layers
        self.rnn = nn.RNN(in_features, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, out_features)

    def forward(self, x):
        """
        Performs a forward pass through the model.

        :param x (torch.Tensor): Input tensor with shape (batch_size, seq_len, in_features).
        :return: torch.Tensor: Output logits for the next word prediction.
        """
        x, h = self.rnn(x) # Process input through the RNN
        y = self.out(h) # Compute output logits from the hidden state
        return y

# Load pretrained Navec embeddings
navec_path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(navec_path)

# Load the dataset
train_path = "text_2.txt"
d_train = WordsDataset(train_path, navec, prev_words=3)
train_data = data.DataLoader(d_train, batch_size=8, shuffle=False)

# Initialize the model
model = WordsRNN(300, d_train.vocab_size)

# Define optimizer and loss function
optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0001)
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
model_path = 'model_rnn_words.tar'
st = model.state_dict()
torch.save(st, model_path)

# Predict continuation of text
model.eval()
initial_words = "подумал встал и снова лег".lower().split() # Initial words
num_predictions = 10 # Number of words to generate

for _ in range(num_predictions):
    # Prepare input sequence
    _data = torch.vstack([torch.tensor(d_train.navec_emb[initial_words[-x]]) for x in range(d_train.prev_words, 0, -1)])
    # Predict the next word
    p = model(_data.unsqueeze(0)).squeeze(0)
    indx = torch.argmax(p, dim=1)
    initial_words.append(d_train.int_to_word[indx.item()])

# Output generated text
print(" ".join(initial_words))
