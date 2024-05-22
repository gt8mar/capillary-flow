"""
Filename: data_loader.py
------------------------
This module contains a custom dataset class for loading velocity data.

By: Marcus Forst
"""
import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split

class CenterCrop256:
    def __call__(self, image):
        width, height = image.size
        new_width, new_height = 256, 256
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = (width + new_width) // 2
        bottom = (height + new_height) // 2

        image = image.crop((left, top, right, bottom))
        return image

class VelocityDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # Convert image to grayscale
        
        # Check if the image is smaller than 256x256
        if image.size[0] < 256 or image.size[1] < 256:
            raise ValueError(f"Image {img_name} is smaller than 256x256 and will be excluded.")
        
        velocity = self.labels_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, velocity

# Define transforms including the custom CenterCrop256
transform = transforms.Compose([
    CenterCrop256(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std for single channel
])

# Function to create train and test datasets
def create_datasets(csv_file, root_dir, test_size=0.2, random_state=42):
    labels_frame = pd.read_csv(csv_file)
    train_frame, test_frame = train_test_split(labels_frame, test_size=test_size, random_state=random_state)
    train_dataset = VelocityDataset(csv_file=train_frame, root_dir=root_dir, transform=transform)
    test_dataset = VelocityDataset(csv_file=test_frame, root_dir=root_dir, transform=transform)
    return train_dataset, test_dataset