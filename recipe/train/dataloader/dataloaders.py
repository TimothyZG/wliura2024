import torchvision
from torchvision.datasets import DTD, EuroSAT
import argparse
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
random_state = 42 # For reproducibility

transform_routine = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for resnets
    transforms.Lambda(lambda x: x.convert('RGB')), # Convert to three channels
    transforms.ToTensor(), # Transform to tensor for torch
    # Standardize so that dataset is more similar to ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloader(ds,root,bs,nworkers):
    if(ds=="DTD"):
        train = DTD(root=root,split="train",transform=transform_routine,download=True)
        val = DTD(root=root,split="val",transform=transform_routine,download=True)
        test = DTD(root=root,split="test",transform=transform_routine,download=True)
        
        train_loader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=nworkers)
        val_loader = DataLoader(val, batch_size=bs, shuffle=False, num_workers=nworkers)
        test_loader = DataLoader(test, batch_size=bs, shuffle=False, num_workers=nworkers)
    elif(ds=="EuroSAT"):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        dataset = EuroSAT(root=root,transform=transform_routine,download=True)
        # Splitting the dataset into train val test since EuroSAT doesn't have 
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.4, random_state=random_state)
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=random_state)
        train = Subset(dataset, train_idx)
        val = Subset(dataset, val_idx)
        test = Subset(dataset, test_idx)
        train_loader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=nworkers)
        val_loader = DataLoader(val, batch_size=bs, shuffle=False, num_workers=nworkers)
        test_loader = DataLoader(test, batch_size=bs, shuffle=False, num_workers=nworkers)
    else:
        raise Exception(f"Unrecognized dataset provided to get_dataloader: {ds}, check spelling or implement get_numclass if working with new dataset")
    return train_loader,val_loader,test_loader

def get_numclass(ds):
    if(ds=="DTD"):
        return 47
    elif(ds=="EuroSAT"):
        return 10
    else:
        raise Exception(f"Unrecognized dataset provided to get_numclass :{ds}, check spelling or implement get_numclass if working with new dataset")