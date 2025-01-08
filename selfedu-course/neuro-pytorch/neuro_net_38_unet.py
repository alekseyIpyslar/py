import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim

# Dataset class for segmentation tasks
class SegmentDataset(data.Dataset):
    def __init__(self, path, transform_img=None, transform_mask=None):
        self.path = path
        self.transform_img = transform_img # Transformations for images
        self.transform_mask = transform_mask # Transformations for masks

        # Load image file paths
        path = os.path.join(self.path, 'images')
        list_files = os.listdir(path)
        self.length = len(list_files)
        self.images = list(map(lambda _x: os.path.join(path, _x), list_files))

        # Load mask file paths
        path = os.path.join(self.path, 'masks')
        list_files = os.listdir(path)
        self.masks = list(map(lambda _x: os.path.join(path, _x), list_files))

    def __getitem__(self, item):
        # Load the image and corresponding mask
        path_img, path_mask = self.images[item], self.masks[item]
        img = Image.open(path_img).convert('RGB') # Convert image to RGB
        mask = Image.open(path_mask).convert('L') # Convert mask to grayscale

        # Apply transformations if specified
        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_mask:
            mask = self.transform_mask(mask)
            # Convert pixel values to binary (segmentation: 1 for foreground, 0 for background)
            mask[mask < 250] = 1
            mask[mask >= 250] = 0

        return img, mask # Return transformed image and mask

    def __len__(self):
        return self.length # Total number of images in the dataset

# U-Net model implementation for segmentation tasks
class UNetModel(nn.Module):
    class _TwoConvLayers(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            return self.model(x)

    class _EncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)
            self.max_pool = nn.MaxPool2d(2) # Downsampling by a factor of 2

        def forward(self, x):
            x = self.block(x) # Apply two convolutional layers
            y = self.max_pool(x) # Downsample
            return y, x # Return downsampled output and skip connection

    class _DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2) # Upsample
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)

        def forward(self, x, y):
            x = self.transpose(x) # Upsample the input
            u = torch.cat([x, y], dim=1) # Concatenate with skip connection
            u = self.block(u) # Apply two convolutional layers
            return u

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # Encoder blocks
        self.enc_block1 = self._EncoderBlock(in_channels, 64)
        self.enc_block2 = self._EncoderBlock(64, 128)
        self.enc_block3 = self._EncoderBlock(128, 256)
        self.enc_block4 = self._EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = self._TwoConvLayers(512, 1024)

        # Decoder blocks
        self.dec_block1 = self._DecoderBlock(1024, 512)
        self.dec_block2 = self._DecoderBlock(512, 256)
        self.dec_block3 = self._DecoderBlock(256, 128)
        self.dec_block4 = self._DecoderBlock(128, 64)

        # Final output layer
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder forward pass
        x, y1 = self.enc_block1(x)
        x, y2 = self.enc_block2(x)
        x, y3 = self.enc_block3(x)
        x, y4 = self.enc_block4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder forward pass
        x = self.dec_block1(x, y4)
        x = self.dec_block2(x, y3)
        x = self.dec_block3(x, y2)
        x = self.dec_block4(x, y1)

        return self.out(x)

# Custom loss function: Soft Dice Loss
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth # Smoothness parameter to avoid division by zero

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = nn.functional.sigmoid(logits) # Convert logits to probabilities
        m1 = probs.view(num, -1) # Flatten predictions
        m2 = targets.view(num, -1) # Flatten targets
        intersection = (m1 * m2) # Element-wise multiplication for intersection

        # Dice score calculation
        score = 2 * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num # Dice loss
        return score

# Define image and mask transformations
tr_img = tfs_v2.Compose([tfs_v2.ToImage(), tfs_v2.ToDtype(torch.float32, scale=True)])
tr_mask = tfs_v2.Compose([tfs_v2.ToImage(), tfs_v2.ToDtype(torch.float32)])

# Initialize dataset and data loader
d_train = SegmentDataset(r"C:/Users/aleks/PycharmProjects/py/selfedu-course/dataset_seg", transform_img=tr_img, transform_mask=tr_mask)
train_data = data.DataLoader(d_train, batch_size=2, shuffle=True)

# Initialize U-Net model
model = UNetModel()

# Optimizer and loss functions
optimizer = optim.RMSprop(params=model.parameters(), lr=0.001)
loss_1 = nn.BCEWithLogitsLoss() # Binary cross-entropy with logits
loss_2 = SoftDiceLoss() # Dice loss

# Training loop
epochs = 10
model.train()

for _e in range(epochs):
    loss_mean = 0 # Track average loss
    lm_count = 0 # Track batch count

    train_tqdm = tqdm(train_data, leave=True) # Progress bar for training
    for x_train, y_train in train_tqdm:
        predict = model(x_train) # Forward pass
        loss = loss_1(predict, y_train) + loss_2(predict, y_train) # Combined loss

        optimizer.zero_grad() # Clear gradients
        loss.backward() # Backpropagation
        optimizer.step() # Update weights

        # Update running mean of loss
        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

# Save model state
torch.save(model.state_dict(), 'model_unet_seg.tar')

# Predict and visualize a segmentation mask
img = Image.open(r"car_1.jpg").convert('RGB') # Load and preprocess an image
img = tr_img(img).unsqueeze(0)

p = model(img).squeeze(0) # Predict mask
x = nn.functional.sigmoid(p.permute(1, 2, 0)) # Convert logits to probabilities
x = x.detach().numpy() * 255 # Scale to 0-255
x = np.clip(x, 0, 255).astype('uint8') # Convert to uint8 for visualization
plt.imshow(x, cmap='gray') # Display mask
plt.show()