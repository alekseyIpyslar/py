import os
import numpy
import re

from navec import Navec
from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim

# Dataset for Predicting the Next Word in a Sequence
class WordsDataset(data.Dataset):
    def __init__(self, path, navec_emb, prev_words=3):
        """
        Initializes the dataset for predicting the next word in a sequence.

        :param path (str): Path to the text file containing the input text.
        :param navec_emb (Navec): Preloaded Navec word embeddings.
        :param prev_words (int): Number of previous words to use as input context
        """
        self.prev_words = prev_words # Number of previous words to consider as context
        self.navec_emb = navec_emb # Word embedding model

        # Load and preprocess the text
        with open(path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            self.text = self.text.replace('\ufeff', '') # Remove invisible BOM character
            self.text = self.text.replace('\n', ' ') # Replace newlines with spaces
            self.text = re.sub(r'[^А-яA-z- ]', '', self.text) # Remove non-alphanumeric characters

        # Tokenize text and retain only words in the embedding vocabulary
        self.words = self.text.lower().split()
        self.words = [word for word in self.words if word in self.navec_emb]

        # Create mappings for vocabulary
        vocab = set(self.words)
        self.int_to_word = dict(enumerate(vocab)) # Map index to word
        self.word_to_int = {b: a for a, b in self.int_to_word.items()} # Map word to index
        self.vocab_size = len(vocab) # Size of the vocabulary

    def __getitem__(self, item):
        """
        Retrieves a sample of previous words and the target word.

        :param item (int): Index of the starting word.
        :return:
            torch.Tensor: Tensor of word embeddings for the previous words.
            int: Index of the target word in the vocabulary.
        """
        # Create tensor of word embeddings for the previous words
        _data = torch.vstack([torch.tensor(self.navec_emb[self.words[x]]) for x in range(item, item + self.prev_words)])

        # Retrieve the target word and convert it to its index
        word = self.words[item + self.prev_words]
        t = self.word_to_int[word]
        return _data, t

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        :return:
            int: Total number of samples.
        """
        return len(self.words) - 1 - self.prev_words # Account for context and target words

# Recurrent Neural Network Model for Word Prediction
class WordsRNN(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Initializes the RNN model for word prediction.
        :param in_features (int): Dimensionality of input word embeddings.
        :param out_features (int): Number of output classes (vocabulary size).
        """
        super().__init__()
        self.hidden_size = 64 # Number of hidden units in the GRU
        self.in_features = in_features # Input dimensionality
        self.out_features = out_features # Output dimensionality (vocabulary size)

        # Define GRU and output layer
        self.rnn = nn.GRU(in_features, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, out_features)

    def forward(self, x):
        """
        Performs a forward pass through the RNN.

        :param x (torch.Tensor): Input tensor with shape (batch_size, seq_len, in_features).
        :return:
            torch.Tensor: Output logits for each word in the vocabulary.
        """
        x, h = self.rnn(x) # Process input through the GRU
        y = self.out(h) # Pass hidden state through the output layer
        return y

# Load Navec word embeddings
path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)

# Initialize dataset and data loader
d_train = WordsDataset("text_2.txt", navec, prev_words=3)
train_data = data.DataLoader(d_train, batch_size=8, shuffle=False)

# Initialize the model
model = WordsRNN(300, d_train.vocab_size)

# Define optimizer and loss function
optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0001)
loss_func = nn.CrossEntropyLoss()

# Training loop
epochs = 20
model.train()

for _e in range(epochs):
    loss_mean = 0 # Track average loss
    lm_count = 0 # Count processed batches

    train_tqdm = tqdm(train_data, leave=True) # Create progress bar
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
st = model.state_dict()
torch.save(st, 'model_rnn_words.tar')

# Inference: Predict next words for a given sequence
model.eval()
predict = "подумал встал и снова лег".lower().split() # Initial seed phrase
total = 10 # Number of additional words to generate

for _ in range(total):
    # Prepare input data from the last few words of the current prediction
    _data = torch.vstack([torch.tensor(d_train.navec_emb[predict[-x]]) for x in range(d_train.prev_words, 0, -1)])
    p = model(_data.unsqueeze(0)).squeeze(0) # Forward pass
    indx = torch.argmax(p, dim=1) # Find the most probable next word
    predict.append(d_train.int_to_word[indx.item()]) # Add predicted word to the sequence

print(" ".join(predict)) # Print the final generated sequence