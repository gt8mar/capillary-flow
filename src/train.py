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

csv_file = '/hpc/projects/capillary-flow/results/ML/240521_filename_df.csv'
root_dir = '/hpc/projects/capillary-flow/results/ML/kymographs'

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update these paths to your actual files and directories
csv_file_path = 'path_to_labels.csv'
images_directory_path = 'path_to_images/'

# Create train and test datasets
train_dataset, test_dataset = create_datasets(csv_file_path, images_directory_path)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model and move it to the GPU if available
model = VelocityNet().to(device)

# Define loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        try:
            inputs, velocities = data
            inputs, velocities = inputs.to(device), velocities.to(device)  # Move data to GPU

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

    # Evaluate on the test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, velocities = data
            inputs, velocities = inputs.to(device), velocities.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, velocities.view(-1, 1))
            test_loss += loss.item()

    print(f'Epoch {epoch + 1}, Test Loss: {test_loss / len(test_loader):.3f}')

print('Finished Training')
