import torch
import torch.optim as optim

from random import randint
import matplotlib.pyplot as plt

# Model function: computes the linear combination of input X with weights w
def model(X, w):
    return X @ w # Matrix multiplication between input and weights

# Define the number of features (polynomial degree)
N = 2
# Initialize weights with small random values
w = torch.FloatTensor(N).uniform_(-1e-5, 1e-5)
w.requires_grad_(True)

# Generate the input data (x values  from 0 to 3 with a step of 0.1)
x = torch.arange(0, 3, 0.1)

# Define the target function y_train (0.5x + 0.2*sin(2x) - 3)
y_train = 0.5 * x + 0.2 * torch.sin(2 * x) - 3.0

# Create the training set with polynomial features (x^0, x^1, ... x^(N-1))
x_train = torch.tensor([[_x ** _n for _n in range(N)] for _x in x])

total = len(x) # Total number of data points
lr = torch.tensor([0.1, 0.01]) # Learning rate (not used directly here)
loss_func = torch.nn.L1Loss() # L1 loss function (Mean Absolute Error)
optimizer = optim.Adam(params=[w], lr=0.01) # Adam optimizer with learning rate of 0.01

# Training loop for 1000 iterations
for _ in range(1000):
    k = randint(0, total - 1) # Randomly select a training sample
    y = model(x_train[k], w) # Compute model output for the sample
    loss = loss_func(y, y_train[k]) # Compute loss for the sample

    loss.backward() # Back-propagate the gradient
    optimizer.step() # Update weights using the optimizer
    optimizer.zero_grad() # Reset gradients for the next iteration

# Print the learned weights
print(w)

# Make predictions on the entire dataset
predict = model(x_train, w)

# Plot the results
plt.plot(x, y_train.numpy(), label="True function") # Plot true function
plt.plot(x, predict.data.numpy(), label="Predicted function") # Plot predictions
plt.grid()
plt.legend()
plt.show()