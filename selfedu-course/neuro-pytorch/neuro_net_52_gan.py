import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim

# Custom dataset class for filtering a specific digit
class DigitDataset(data.Dataset):
    def __init__(self, path, train=True, target=5, transform=None):
        """
        Custom dataset for MNIST to extract only one target digit.
        :param path (str): Path to save or load the MNIST dataset.
        :param train (bool): Whether to load training or test set.
        :param target (int): Digit to filter and use in the dataset.
        :param transform (callable): Transformations to apply to the dataset.
        """
        _dataset = torchvision.datasets.MNIST(path, download=True, train=train)
        self.dataset = _dataset.data[_dataset.targets == target]
        self.length = self.dataset.size(0)
        self.target = torch.tensor([target], dtype=torch.float32)

        if transform:
            self.dataset = transform(self.dataset).view(-1, 1, 28, 28)

    def __getitem__(self, item):
        return self.dataset[item], self.target

    def __len__(self):
        return self.length

# Generator model
model_gen = nn.Sequential(
    nn.Linear(2, 512 * 7 * 7, bias=False),
    nn.ELU(inplace=True),
    nn.BatchNorm1d(512 * 7 * 7),
    nn.Unflatten(1, (512, 7, 7)),
    nn.Conv2d(512, 256, 5, 1, padding="same", bias=False),
    nn.ELU(inplace=True),
    nn.BatchNorm2d(256),
    nn.Conv2d(256, 128, 5, 1, padding="same", bias=False),
    nn.ELU(inplace=True),
    nn.BatchNorm2d(128),
    nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False),
    nn.ELU(inplace=True),
    nn.BatchNorm2d(64),
    nn.ConvTranspose2d(64, 32, 4, 2, padding=1, bias=False),
    nn.ELU(inplace=True),
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 1, 1, 1),
    nn.Sigmoid(),
)

# Discriminator model
model_dis = nn.Sequential(
    nn.Conv2d(1, 64, 5, 2, padding=2, bias=False),
    nn.ELU(inplace=True),
    nn.BatchNorm2d(64),
    nn.Conv2d(64, 128, 5, 2, padding=2, bias=False),
    nn.ELU(inplace=True),
    nn.BatchNorm2d(128),
    nn.Flatten(),
    nn.Linear(128 * 7 * 7, 1),
)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move models to device
model_gen.to(device)
model_dis.to(device)

# Hyperparameters
epochs = 20
hidden_dim = 2
batch_size = 16

# Transformations for MNIST dataset
transforms = tfs_v2.Compose(
    [tfs_v2.ToImage(), tfs_v2.ToDtype(dtype=torch.float32, scale=True)]
)

# Load the training dataset
d_train = DigitDataset(
    r"C:\datasets\mnist", train=True, transform=transforms
)
train_data = data.DataLoader(d_train, batch_size=batch_size, shuffle=True, drop_last=True)

# Optimizer and loss function
optimizer_gen = optim.Adam(params=model_gen.parameters(), lr=0.001)
optimizer_dis = optim.Adam(params=model_dis.parameters(), lr=0.001)
loss_func = nn.BCEWithLogitsLoss()

# Targets for real and fake data
targets_0 = torch.zeros(batch_size, 1).to(device)
targets_1 = torch.ones(batch_size, 1).to(device)

# Training loop
loss_gen_lst = []
loss_dis_lst = []

model_gen.train()
model_dis.train()

for _e in range(epochs):
    loss_mean_gen = 0
    loss_mean_dis = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        x_train = x_train.to(device)

        # Generator step
        h = torch.normal(
            mean=torch.zeros((batch_size, hidden_dim)),
            std=torch.ones((batch_size, hidden_dim)),
        ).to(device)

        img_gen = model_gen(h)
        fake_out = model_dis(img_gen)

        loss_gen = loss_func(fake_out, targets_1)

        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        # Discriminator step
        img_gen = model_gen(h)
        fake_out = model_dis(img_gen)
        real_out = model_dis(x_train)

        outputs = torch.cat([real_out, fake_out], dim=0).to(device)
        targets = torch.cat([targets_1, targets_0], dim=0).to(device)

        loss_dis = loss_func(outputs, targets)

        optimizer_dis.zero_grad()
        loss_dis.backward()
        optimizer_dis.step()

        lm_count += 1
        loss_mean_gen = 1 / lm_count * loss_gen.item() + (1 - 1 / lm_count) * loss_mean_gen
        loss_mean_dis = 1 / lm_count * loss_dis.item() + (1 - 1 / lm_count) * loss_mean_dis

        train_tqdm.set_description(
            f"Epoch [{_e+1}/{epochs}], loss_gen={loss_mean_gen:.4f}, loss_dis={loss_mean_dis:.4f}"
        )

    loss_gen_lst.append(loss_mean_gen)
    loss_dis_lst.append(loss_mean_dis)

# Save the models and losses
torch.save(model_gen.state_dict(), "model_gen.tar")
torch.save(model_dis.state_dict(), "model_dis.tar")
torch.save({"loss_gen": loss_gen_lst, "loss_dis": loss_dis_lst}, "model_gan_losses.tar")

# Visualization: Reconstruct digits from latent space
model_gen.eval()
n = 2
total = 2 * n + 1

plt.figure(figsize=(total, total))

num = 1
for i in range(-n, n + 1):
    for j in range(-n, n + 1):
        ax = plt.subplot(total, total, num)
        num += 1
        h = torch.tensor([[1 * j / n, 1 * j / n]], dtype=torch.float32).to(device)
        predict = model_gen(h).detach().squeeze()
        dec_img = predict.cpu().numpy()

        plt.imshow(dec_img, cmap="gray")
        ax.axis("off")

plt.show()
