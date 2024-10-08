import torchvision
from torchvision.datasets import DTD, EuroSAT, GTSRB, SUN397, SVHN, MNIST, Caltech256, CIFAR10
import argparse
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from torchvision.transforms import v2
from pytorch_cinic.dataset import CINIC10
import torchvision.transforms.functional as F



random_state = 42 # For reproducibility

def standardize(x):
    mean = x.mean(dim=(1, 2), keepdim=True)
    std = x.std(dim=(1, 2), keepdim=True)
    std[std == 0.] = 1.0  # Prevent division by zero
    return (x - mean) / std

def get_dataloader(ds, root, bs, nworkers, resize=448, no_regularizer=False, no_resize=False):
    if no_regularizer:
        train_transform_routine = v2.Compose([
            transforms.Resize((resize, resize)),
            transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to three channels
            transforms.ToTensor(),  # Transform to tensor for torch
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standardize using ImageNet stats
        ])
    else:
        train_transform_routine = v2.Compose([
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.Resize((resize, resize)),
            v2.RandomHorizontalFlip(p=0.5),
            transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to three channels
            transforms.ToTensor(),  # Transform to tensor for torch
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standardize
        ])

    train_transform_routine_rxrx1 = v2.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to three channels
        transforms.RandomChoice([
            transforms.Lambda(lambda x: F.rotate(x, 0)),
            transforms.Lambda(lambda x: F.rotate(x, 90)),
            transforms.Lambda(lambda x: F.rotate(x, 180)),
            transforms.Lambda(lambda x: F.rotate(x, 270))
        ]),  # Rotate the image randomly by 0, 90, 180, or 270 degrees
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Transform to tensor for torch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standardize
    ])
    
    test_transform_routine = transforms.Compose([
        transforms.Resize((resize, resize)),  # Resize images to 224x224 for resnets
        transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to three channels
        transforms.ToTensor(),  # Transform to tensor for torch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standardize
    ])
    if(ds=="DTD"):
        train = DTD(root=root,split="train",transform=train_transform_routine,download=True)
        val = DTD(root=root,split="val",transform=test_transform_routine,download=True)
        test = DTD(root=root,split="test",transform=test_transform_routine,download=True)
    elif(ds=="CINIC10"):
        train = CINIC10(root=root,partition="train", transform=train_transform_routine,download=True)
        val = CINIC10(root=root,partition="valid",transform=test_transform_routine,download=True)
        test = CINIC10(root=root,partition="test",transform=test_transform_routine,download=True)
    elif (ds == "EuroSAT"):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        dataset = EuroSAT(root=root, transform=test_transform_routine, download=True)
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.4, random_state=random_state)
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=random_state)
        train = Subset(dataset, train_idx)
        train.dataset.transform = train_transform_routine  # Apply train transforms to the training subset
        val = Subset(dataset, val_idx)
        test = Subset(dataset, test_idx)
    elif(ds=="GTSRB" or ds=="SVHN" or ds=="MNIST" or ds=="CIFAR10"):
        if(ds=="GTSRB"):
            train = GTSRB(root=root,split = "train",transform=train_transform_routine,download=True)
            dataset = GTSRB(root=root,split = "test",transform=test_transform_routine,download=True)
        elif(ds=="SVHN"):
            train = SVHN(root=root,split = "train",transform=train_transform_routine,download=True)
            dataset = SVHN(root=root,split = "test",transform=test_transform_routine,download=True)
        elif(ds=="MNIST"):
            train = MNIST(root=root,train=True, transform=train_transform_routine,download=True)
            dataset = MNIST(root=root,train=False, transform=test_transform_routine,download=True)
        elif(ds=="CIFAR10"):
            train = CIFAR10(root=root,train=True, transform=train_transform_routine,download=True)
            dataset = CIFAR10(root=root,train=False, transform=test_transform_routine,download=True)
        # Splitting the dataset into train val test since these datasets don't have default val set
        val_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.5, random_state=random_state)
        val = Subset(dataset, val_idx)
        test = Subset(dataset, test_idx)
    elif(ds=="iWildCamOOD"):
        dataset = get_dataset(dataset="iwildcam", root_dir=root, download=True)
        train_data = dataset.get_subset("train", transform=train_transform_routine)
        train_loader = get_train_loader("standard", train_data, batch_size=bs, num_workers=nworkers)
        test_data = dataset.get_subset("test", transform=test_transform_routine)
        test_loader = get_eval_loader("standard", test_data, batch_size=bs, num_workers=nworkers)
        id_val_data = dataset.get_subset("id_val", transform=test_transform_routine)
        id_val_loader = get_eval_loader("standard", id_val_data, batch_size=bs, num_workers=nworkers)
        return train_loader,id_val_loader,test_loader
    elif(ds=="iWildCam"):
        dataset = get_dataset(dataset="iwildcam", root_dir=root, download=True)
        train_data = dataset.get_subset("train", transform=train_transform_routine)
        train_loader = get_train_loader("standard", train_data, batch_size=bs, num_workers=nworkers)
        id_val_data = dataset.get_subset("id_val", transform=test_transform_routine)
        id_val_loader = get_eval_loader("standard", id_val_data, batch_size=bs, num_workers=nworkers)
        id_test_data = dataset.get_subset("id_test", transform=test_transform_routine)
        id_test_loader = get_eval_loader("standard", id_test_data, batch_size=bs, num_workers=nworkers)
        ood_test_data = dataset.get_subset("test", transform=test_transform_routine)
        ood_test_loader = get_eval_loader("standard", ood_test_data, batch_size=bs, num_workers=nworkers)
        return train_loader,id_val_loader, id_test_loader, ood_test_loader
    elif(ds=="SUN397"):
        dataset = SUN397(root=root, transform=test_transform_routine, download=True)
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.4, random_state=random_state)
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=random_state)
        train = Subset(dataset, train_idx)
        train.dataset.transform = train_transform_routine  # Apply train transforms to the training subset
        val = Subset(dataset, val_idx)
        test = Subset(dataset, test_idx)
    elif(ds=="Caltech256"):
        dataset = Caltech256(root=root, transform=test_transform_routine, download=True)
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.4, random_state=random_state)
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=random_state)
        train = Subset(dataset, train_idx)
        train.dataset.transform = train_transform_routine  # Apply train transforms to the training subset
        val = Subset(dataset, val_idx)
        test = Subset(dataset, test_idx)
    elif(ds=="rxrx1"):
        dataset = get_dataset(dataset="rxrx1", root_dir=root, download=True)
        train_data = dataset.get_subset("train", transform=train_transform_routine_rxrx1)
        train_loader = get_train_loader("standard", train_data, batch_size=bs, num_workers=nworkers)
        id_val_data = dataset.get_subset("val", transform=test_transform_routine)
        id_val_loader = get_eval_loader("standard", id_val_data, batch_size=bs, num_workers=nworkers)
        id_test_data = dataset.get_subset("id_test", transform=test_transform_routine)
        id_test_loader = get_eval_loader("standard", id_test_data, batch_size=bs, num_workers=nworkers)
        ood_test_data = dataset.get_subset("test", transform=test_transform_routine)
        ood_test_loader = get_eval_loader("standard", ood_test_data, batch_size=bs, num_workers=nworkers)
        return train_loader,id_val_loader,id_test_loader,ood_test_loader
    elif(ds=="camelyon17"):
        dataset = get_dataset(dataset="camelyon17", root_dir=root, download=True)
        train_data = dataset.get_subset("train", transform=train_transform_routine)
        train_loader = get_train_loader("standard", train_data, batch_size=bs, num_workers=nworkers)
        id_val_data = dataset.get_subset("val", transform=test_transform_routine)
        id_val_loader = get_eval_loader("standard", id_val_data, batch_size=bs, num_workers=nworkers)
        id_test_data = dataset.get_subset("id_test", transform=test_transform_routine)
        id_test_loader = get_eval_loader("standard", id_test_data, batch_size=bs, num_workers=nworkers)
        ood_test_data = dataset.get_subset("test", transform=test_transform_routine)
        ood_test_loader = get_eval_loader("standard", ood_test_data, batch_size=bs, num_workers=nworkers)
        return train_loader,id_val_loader,id_test_loader,ood_test_loader
    else:
        raise Exception(f"Unrecognized dataset provided to get_dataloader: {ds}, check spelling or implement get_numclass if working with new dataset")
    train_loader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=nworkers)
    val_loader = DataLoader(val, batch_size=bs, shuffle=False, num_workers=nworkers)
    test_loader = DataLoader(test, batch_size=bs, shuffle=False, num_workers=nworkers)
    return train_loader,val_loader,test_loader

def get_numclass(ds):
    if(ds=="DTD"):
        return 47
    elif(ds=="EuroSAT" or ds=="CIFAR10" or ds=="CINIC10" or ds=="CIFAR10.1"):
        return 10
    elif(ds=="iWildCam" or ds=="iWildCamOOD"):
        return 182
    elif(ds=="GTSRB"):
        return 43
    elif(ds=="SUN397"):
        return 397
    elif(ds=="SVHN"):
        return 10
    elif(ds=="MNIST"):
        return 10
    elif(ds=="Caltech256"):
        return 257
    elif(ds=="rxrx1"):
        return 1139
    else:
        raise Exception(f"Unrecognized dataset provided to get_numclass :{ds}, check spelling or implement get_numclass if working with new dataset")
    

import torch
from torch.utils.data import TensorDataset, DataLoader

def load_into_memory(dataset, transform):
    data = []
    targets = []
    metadata = []
    for x, y, meta in dataset:
        if transform:
            x = transform(x)
        data.append(x)
        targets.append(y)
        metadata.append(meta)
    return TensorDataset(torch.stack(data), torch.tensor(targets), torch.tensor(metadata))

def load_iwildcam(root, bs, nworkers, resize):
    train_transform_routine = v2.Compose([
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.Resize((resize, resize)),
        v2.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to three channels
        transforms.ToTensor(),  # Transform to tensor for torch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standardize
    ])

    test_transform_routine = transforms.Compose([
        transforms.Resize((resize, resize)),  # Resize images to 224x224 for resnets
        transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to three channels
        transforms.ToTensor(),  # Transform to tensor for torch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standardize
    ])
    dataset = get_dataset(dataset="iwildcam", root_dir=root, download=True)
    
    train_data = load_into_memory(dataset.get_subset("train"), train_transform_routine)
    train_loader = DataLoader(train_data, batch_size=bs, num_workers=nworkers, shuffle=True)

    id_val_data = load_into_memory(dataset.get_subset("id_val"), test_transform_routine)
    id_val_loader = DataLoader(id_val_data, batch_size=bs, num_workers=nworkers, shuffle=False)

    id_test_data = load_into_memory(dataset.get_subset("id_test"), test_transform_routine)
    id_test_loader = DataLoader(id_test_data, batch_size=bs, num_workers=nworkers, shuffle=False)

    ood_test_data = load_into_memory(dataset.get_subset("test"), test_transform_routine)
    ood_test_loader = DataLoader(ood_test_data, batch_size=bs, num_workers=nworkers, shuffle=False)

    return train_loader, id_val_loader, id_test_loader, ood_test_loader
