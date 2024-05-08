import os
import torch
from torchvision.datasets import Caltech256
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

def setup_caltech256(data_root='Data/Caltech256', batch_size=128, num_workers=4):
    # Ensure the data root directory exists
    os.makedirs(data_root, exist_ok=True)

    # Transformations applied on each image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 for typical CNNs
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Download and load the full Caltech-256 dataset
    dataset = Caltech256(root=data_root, download=True, transform=transform)

    # Splitting the dataset into train and test
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

    # Create Subset for train and test from indices
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    # Create DataLoaders for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

# Usage
train_loader, test_loader = setup_caltech256()