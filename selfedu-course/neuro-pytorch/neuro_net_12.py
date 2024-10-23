import torch
from random import randint

# Activation function
def act(z):
    return torch.tanh(z)

# Derivative of the activation function
def df(z):
    s = act(z)
    return 1 - s * s

# Forward pass function
def go_forward(x_inp, w1, w2):
    z1 = torch.mv(w1[:, :3], x_inp) + w1[:, 3]
    s = act(z1)

    z2 = torch.dot(w2[:2], s) + w2[2]
    y = act(z2)
    return y, z1, z2

torch.manual_seed(1)

W1 = torch.rand(8).view(2, 4) - 0.5 # Weights for the first layer
W2 = torch.rand(3) - 0.5 # Weights for the second layer

# Training dataset (input-output pairs)
x_train = torch.FloatTensor([(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)])
y_train = torch.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])

lmd = 0.05 # Learning rate
N = 1000 # Number of training iterations
total = len(y_train) # Size of the training dataset

# Training Loop
for _ in range(N):
    k = randint(0, total - 1)
    x = x_train[k] # Randomly choose an input sample from the training set
    y, z1, out = go_forward(x, W1, W2) # Forward pass through the network to compute outputs
    e = y - y_train[k] # Derivative of the quadratic loss function
    delta = e * df(out) # Compute local gradient for the output layer
    delta2 = W2[:2] * delta * df(z1) # Vector of local gradients for the hidden layer

    # Update weights of the second (output) layer
    W2[:2] = W2[:2] - lmd * delta * z1
    W2[2] = W2[2] - lmd * delta # Update bias

    # Update weights of the first (hidden) layer
    W1[0, :3] = W1[0, :3] - lmd * delta2[0] * x
    W1[1, :3] = W1[1, :3] - lmd * delta2[1] * x

    # Update biases for the first layer
    W1[0, 3] = W1[0, 3] - lmd * delta2[0]
    W1[1, 3] = W1[1, 3] - lmd * delta2[1]

# Test the trained network
for x, d in zip(x_train, y_train):
    y, z1, out = go_forward(x, W1, W2)
    print(f"Network output: {y} => Expected: {d}")

# Print the resulting weight coefficients
print(W1)
print(W2)