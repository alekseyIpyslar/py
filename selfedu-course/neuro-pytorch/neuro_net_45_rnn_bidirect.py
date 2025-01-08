import os
import numpy as np
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

# Dataset for Phrase Classification using Word Embeddings
class PhraseDataset(data.Dataset):
    def __init__(self, path_true, path_false, navec_emb, batch_size=8):
        """
        Initializes the dataset by loading and preprocessing positive and negative phrases.

        :param path_true (str): Path to the file containing positive phrases.
        :param path_false (str): Path to the file containing negative phrases.
        :param navec_emb (Navec): Preloaded Navec word embeddings.
        :param batch_size (int): Number of samples per batch.
        """
        self.navec_emb = navec_emb
        self.batch_size = batch_size

        # Load and preprocess positive phrases
        with open(path_true, 'r', encoding='utf-8') as f:
            phrase_true = f.readlines()
            self._clear_phrase(phrase_true)

        # Load and preprocess negative phrases
        with open(path_false, 'r', encoding='utf-8') as f:
            phrase_false = f.readlines()
            self._clear_phrase(phrase_false)

        # Combine and sort phrases by length
        self.phrase_lst = [(_x, 0) for _x in phrase_true] + [(_x, 1) for _x in phrase_false]
        self.phrase_lst.sort(key=lambda _x: len(_x[0]))
        self.dataset_len = len(self.phrase_lst)

    def _clear_phrase(self, p_lst):
        """
        Cleans and tokenizes the phrases, retaining only words present in the Navec vocabulary.

        :param p_lst (list): List of phrases to be cleaned.
        :return:
        """
        for _i, _p in enumerate(p_lst):
            _p  = _p.lower().replace('\ufeff', '').strip()
            _p = re.sub(r'[^А-яA-z- ]', '', _p)
            _words = _p.split()
            _words = [w for w in _words if w in self.navec_emb]
            p_lst[_i] = _words

    def __getitem__(self, item):
        """
        Retrieves a batch of phrases and their corresponding labels.

        :param item (int): Index of the batch.
        :return:
            torch.Tensor: Tensor containing word embeddings for phrases.
            torch.Tensor: Tensor containing labels for phrases.
        """
        item *= self.batch_size
        item_last = item + self.batch_size
        if item_last > self.dataset_len:
            item_last = self.dataset_len

        _data = []
        _target = []
        max_length = len(self.phrase_lst[item_last-1][0])

        for i in range(item, item_last):
            words_emb = []
            phrase = self.phrase_lst[i]
            length = len(phrase[0])

            for k in range(max_length):
                t = torch.tensor(self.navec_emb[phrase[0][k]], dtype=torch.float32) if k < length else torch.zeros(300)
                words_emb.append(t)

            _data.append(torch.vstack(words_emb))
            _target.append(torch.tensor(phrase[1], dtype=torch.float32))

        _data_batch = torch.stack(_data)
        _target = torch.vstack(_target)
        return _data_batch, _target

    def __len__(self):
        """
        :return: Returns the number of batches in the dataset.
        """
        last = 0 if self.dataset_len % self.batch_size == 0 else 1
        return self.dataset_len // self.batch_size + last

# Recurrent Neural Network Model for Phrase Classification
class WordsRNN(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Initializes the bidirectional RNN model.

        :param in_features (int): Number of input features (size of word embeddings).
        :param out_features (int): Number of output features (binary classification).
        """
        super().__init__()
        self.hidden_size = 16
        self.in_features = in_features
        self.out_features = out_features

        # Define bidirectional RNN and output layer
        self.rnn = nn.RNN(in_features, self.hidden_size, batch_first=True, bidirectional=True)
        self.out = nn.Linear(self.hidden_size * 2, out_features)

    def forward(self, x):
        """
        Performs a forward pass through the model.

        :param x (torch.Tensor): Input tensor with shape (batch_size, seq_len, in_features).

        :return: torch.Tensor: Output logits for classification.
        """
        x, h = self.rnn(x)
        hh = torch.cat((h[-2, :, :], h[-1, :, :]), dim=-1) # Concatenate hidden states from both directions
        y = self.out(hh)
        return y

# Load Navec embeddings
path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)

# Initialize dataset and data loader
train_path_true = "train_data_true"
train_path_false = "train_data_false"
d_train = PhraseDataset(train_path_true, train_path_false, navec)
train_data = data.DataLoader(d_train, batch_size=1, shuffle=True)

# Initialize the model
model = WordsRNN(300, 1)

# Define optimizer and loss function
optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
loss_func = nn.BCEWithLogitsLoss()

# Training loop
epochs = 20
model.train()

for _e in range(epochs):
    loss_mean = 0 # Track average loss
    lm_count = 0 # Count processed batches

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train.squeeze(0)).squeeze(0) # Forward pass
        loss = loss_func(predict, y_train.squeeze(0)) # Compute loss

        optimizer.zero_grad() # Clear gradients
        loss.backward() # Backpropagate
        optimizer.step() # Update model parameters

        # Update running average of loss
        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

# Save the trained model
model_path = 'model_rnn_bidir.tar'
st = model.state_dict()
torch.save(st, model_path)

# Predict sentiment for a new phrase
model.eval()

phrase = "сегодня солнечная погода"
phrase_lst = phrase.lower().split()
phrase_lst = [torch.tensor(navec[w]) for w in phrase_lst if w in navec]
_data_batch = torch.stack(phrase_lst)
predict = model(_data_batch.unsqueeze(0)).squeeze(0)
p = torch.sigmoid(predict).item()
print(p)
print(phrase, ":", "положительное" if p < 0.5 else "отрицательное")