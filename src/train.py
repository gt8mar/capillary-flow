"""
Filename: train.py
-------------------
This module contains a script for training a velocity prediction model.

By: Marcus Forst
"""

import torch
import torch.optim as optim
from data_loader import create_datasets
from model import VelocityNet
from torch.utils.data import DataLoader
from accelerate import Accelerator
import numpy as np

# Initialize the accelerator for distributed training
accelerator = Accelerator()

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Update these paths to your actual files and directories
csv_file = '/hpc/projects/capillary-flow/results/ML/big_240521_filename_df.csv'
root_dir = '/hpc/projects/capillary-flow/results/ML/big_kymographs'

# Create train and test datasets
train_dataset, test_dataset = create_datasets(csv_file, root_dir)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model
model = VelocityNet()

# Define loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Prepare everything with the accelerator
model, optimizer, train_loader, test_loader = accelerator.prepare(
    model, optimizer, train_loader, test_loader
)

# Training loop
for epoch in range(50):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        try:
            inputs, velocities = data
            inputs, velocities = inputs.to(torch.float32), velocities.to(torch.float32)
            outputs = model(inputs)
            loss = criterion(outputs, velocities.view(-1, 1))

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        except ValueError as e:
            # Handle the exception for images smaller than 256x256
            print(f"Skipped batch due to error: {e}")
            continue

    # Evaluate on the test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, velocities = data
            inputs, velocities = inputs.to(torch.float32), velocities.to(torch.float32)
            outputs = model(inputs)
            loss = criterion(outputs, velocities.view(-1, 1))
            test_loss += loss.item()

    print(f'Epoch {epoch + 1}, Test Loss: {test_loss / len(test_loader):.3f}')

print('Finished Training')
