from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim
import os

# Define the neutral network to extract features from an image
class ModelStyle(nn.Module):
    def __init__(self):
        super().__init__()
        # Load a pre-trained VGG19 model
        _model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.mf = _model.features # Extract the feature layers only
        self.mf.requires_grad_(False) # Freeze the feature layers
        self.requires_grad_(False) # Ensure the model is entirely non-trainable
        self.mf.eval() # Set the model to evaluation mode (no dropout, batchnorm frozen)

        # Indices of layers used for style/content extraction
        self.idx_out = (0, 5, 10, 19, 28, 34)
        self.num_style_layers = len(self.idx_out) - 1 # The last layer is used for content

    def forward(self, x):
        outputs = [] # Store the outputs of selected layers
        for indx, layer in enumerate(self.mf):
            x = layer(x)
            if indx in self.idx_out: # If the current layer is selected, save its outputs
                outputs.append(x.squeeze(0)) # Remove batch dimension for convenience
        return outputs

# Compute the content loss between generated and target content representations
def get_content_loss(base_content, target):
    return torch.mean(torch.square(base_content - target))

# Compute the Gram matrix for a given feature map
def gram_matrix(x):
    channels = x.size(dim=0) # Number of channels in the feature map
    g = x.view(channels, -1) # Flatten the spatial dimensions
    gram = torch.mm(g, g.mT) / g.size(dim=1) # Compute Gram matrix and normalize by size
    return gram

# Compute the style loss between generated and target style representations
def get_style_loss(base_style, gram_target):
    style_weights = [1.0, 0.8, 0.5, 0.3, 0.1] # Assign weights to different style layers
    _loss = 0
    for i, (base, target) in enumerate(zip(base_style, gram_target)):
        gram_style = gram_matrix(base) # Compute Gram matrix for the generated image
        _loss += style_weights[i] * torch.mean(torch.square(gram_style - target)) # Weighted loss
    return _loss

# Load and transform an image
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB") # Ensure the image is in RGB format
    return img

def transform_image(img):
    # Transform image into a PyTorch tensor with normalization
    transform = tfs_v2.Compose([tfs_v2.ToImage(), tfs_v2.ToDtype(torch.float32, scale=True)])
    return transform(img).unsqueeze(0) # Add batch dimension

# Define paths for content and style images
img_path = 'C://Users//aleks//PycharmProjects//py//selfedu-course//images//cats//img.jpg'
img_style_path = 'C://Users//aleks//PycharmProjects//py//selfedu-course//images//cats//img_style.jpg'

try:
    # Load and preprocess the images
    img = transform_image(load_image(img_path))
    img_style = transform_image(load_image(img_style_path))
except FileNotFoundError as e:
    print(e)
    exit()

# Create an initial generated image by cloning the content image
img_create = img.clone()
img_create.requires_grad_(True) # Enable gradients for optimization

# Initialize the style model and compute feature maps for the content and images
model = ModelStyle()
outputs_img = model(img) # Content image features
outputs_img_style = model(img_style) # Style image features

# Compute the Gram matrices for the style features
gram_matrix_style = [gram_matrix(x) for x in outputs_img_style[:model.num_style_layers]]

# Style transfer parameters
content_weight = 1 # Weight for content loss
style_weight = 1000 # Weight for style loss
best_loss = float("inf") # Track the best loss value
epochs = 100 # Number of optimization iterations

# Set up an optimizer to modify the generated image
optimizer = optim.Adam(params=[img_create], lr=0.01)
best_img = img_create.clone() # Clone the initial generated image

# Training loop for style transfer
for _e in range(epochs):
    outputs_img_create = model(img_create) # Compute features for the generated image

    # Compute the losses
    loss_content = get_content_loss(outputs_img_create[-1], outputs_img[-1]) # Content loss
    loss_style = get_style_loss(outputs_img_create, gram_matrix_style) # Style loss
    loss = content_weight * loss_content + style_weight * loss_style # Total loss

    optimizer.zero_grad() # Clear previous gradients
    loss.backward() # Compute gradients
    optimizer.step() # Update the generated image

    img_create.data.clamp_(0, 1) # Clamp pixel values to valid range [0, 1]

    # Update the best image if the current loss is lower
    if loss < best_loss:
        best_loss = loss
        best_img = img_create.clone()

    print(
        f"Iteration: {_e + 1}, Content Loss: {loss_content.item():.4f}, Style Loss: {loss_style.item():.4f}, Total Loss: {loss.item():.4f}")

# Post-process the best image to save and display it
x = best_img.detach().squeeze() # Remove batch dimension
x = (x - torch.amin(x)) / (torch.amax(x) - torch.amin(x)) * 255.0 # Normalize to [0, 255]
x = x.permute(1, 2, 0).numpy() # Convert to NumPy array
x = np.clip(x, 0, 255).astype("uint8") # Ensure valid pixel range

# Save and display the result
result_image = Image.fromarray(x, "RGB")
result_image.save("result.jpg") # Save the generated image

print(f"Best Loss: {best_loss.item():.4f}")
plt.imshow(x)
plt.axis("off") # Remove axis for better visualization
plt.show()
