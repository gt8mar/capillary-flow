"""
Filename: train.py
-------------------
This module contains a script for training a velocity prediction model.

By: Marcus Forst
"""

import torch
import torch.optim as optim
from data_loader import VelocityDataset, transform
from model import VelocityNet
from torch.utils.data import DataLoader

import torch
import torch.optim as optim
from data_loader import VelocityDataset, transform
from model import VelocityNet
from torch.utils.data import DataLoader

csv_file = '/hpc/projects/capillary-flow/results/ML/240521_filename_df.csv'
root_dir = '/hpc/projects/capillary-flow/results/ML/kymographs'

# Load dataset with the new transform
dataset = VelocityDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = VelocityNet()

# Define loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        try:
            inputs, velocities = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, velocities.view(-1, 1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        except ValueError as e:
            # Handle the exception for images smaller than 256x256
            print(e)
            continue

print('Finished Training')