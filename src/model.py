"""
Filename: model.py
------------------
This module contains a simple CNN model for predicting velocity from images.

By: Marcus Forst
"""

import torch.nn as nn
import torch.nn.functional as F

class VelocityNet(nn.Module):
    def __init__(self):
        super(VelocityNet, self).__init__()
        # First convolutional layer: input channels = 1 (grayscale), output channels = 32, kernel size = 3x3, padding = 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # Max pooling layer with a 2x2 window
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional layer: input channels = 32, output channels = 64, kernel size = 3x3, padding = 1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # Third convolutional layer: input channels = 64, output channels = 128, kernel size = 3x3, padding = 1
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # Fully connected layer: input size = 128 * 16 * 16, output size = 512
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        # Output layer: input size = 512, output size = 1 (regression output)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Apply the first convolutional layer followed by ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply the second convolutional layer followed by ReLU activation and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Apply the third convolutional layer followed by ReLU activation and max pooling
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the tensor from 4D to 2D for the fully connected layers
        x = x.view(-1, 128 * 16 * 16)
        # Apply the first fully connected layer followed by ReLU activation
        x = F.relu(self.fc1(x))
        # Apply the output layer
        x = self.fc2(x)
        return x

