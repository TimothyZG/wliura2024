import torch
import torch.nn as nn
import os
import sys
import argparse
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from torch.utils.data import Dataset, DataLoader
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from torchvision import transforms
import utils


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning script")
    parser.add_argument('--config-path', type=str, required=True, help='Path to the config file')
    return parser.parse_args()

class iWildCamDataset(Dataset):
    def __init__(self, subset, root_dir, transform=None):
        metadata_path = os.path.join(root_dir, 'iwildcam_v2.0', 'metadata.csv')
        print(f"Checking if metadata exists at {metadata_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file does not exist at {metadata_path}")
        
        self.dataset = get_dataset(dataset="iwildcam", root_dir=root_dir, download=True)
        self.data = self.dataset.get_subset(subset, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# Define the transforms
train_transform_routine = transforms.Compose([
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to three channels
    transforms.ToTensor(),  # Transform to tensor for torch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standardize
])

test_transform_routine = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for resnets
    transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to three channels
    transforms.ToTensor(),  # Transform to tensor for torch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standardize
])

def train_model(config):
    num_classes = 182
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device is {device}")

    # Use the custom DataLoader directly
    # train_dataset = iWildCamDataset(subset="train", root_dir=config['dataset_root'], transform=train_transform_routine)
    # val_dataset = iWildCamDataset(subset="id_val", root_dir=config['dataset_root'], transform=test_transform_routine)
    # id_test_dataset = iWildCamDataset(subset="id_test", root_dir=config['dataset_root'], transform=test_transform_routine)
    # ood_test_dataset = iWildCamDataset(subset="test", root_dir=config['dataset_root'], transform=test_transform_routine)

    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['workers'])
    # val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['workers'])
    # id_test_loader = DataLoader(id_test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['workers'])
    # ood_test_loader = DataLoader(ood_test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['workers'])
    
    dataset = get_dataset(dataset="iwildcam", root_dir=config['dataset_root'], download=True)
    
    train_data = dataset.get_subset("train", transform=train_transform_routine)
    val_data = dataset.get_subset("id_val", transform=test_transform_routine)
    
    train_loader = get_train_loader("standard", train_data, batch_size=config['batch_size'], num_workers=config['workers'])
    val_loader = get_eval_loader("standard", val_data, batch_size=config['batch_size'], num_workers=config['workers'])
    
    model = utils.get_model(config['model_name'])
    if config['num_epochs_linear'] > 0:
        model = utils.freeze_model(model)
    model = utils.swap_head(model, config['model_name'], num_classes)

    optimizer, scheduler = utils.init_optimizer_scheduler(model, config, config['num_epochs'])
    loss_function = nn.CrossEntropyLoss()

    model.train()
    model = nn.DataParallel(model)
    model = model.to(device)
    print(f"Model is moved to {device}")

    for epoch in range(config['num_epochs']):
        if config['num_epochs_linear'] > 0 and epoch == config['num_epochs_linear']:
            for param in model.parameters():
                param.requires_grad = True
            optimizer, scheduler = utils.init_optimizer_scheduler(model, config, config['num_epochs'] - epoch)
        
        total = 0
        correct = 0
        for batch_idx, labeled_batch in enumerate(train_loader):
            data, targets, metadata = labeled_batch
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        val_accuracy, val_loss, val_f1 = utils.evaluate(model, val_loader, loss_function, device)

        # Use session.report instead of tune.report
        session.report({"val_accuracy": val_accuracy, "val_loss": val_loss, "val_f1": val_f1})

        curr_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch+1}/{config['num_epochs']} Completed | Loss: {loss.item():.4f} | Accuracy: {correct / total:.2f}% | Val F1: {val_f1:.2f} | Val Loss: {val_loss:.2f} | LR: {curr_lr}")
        model.train()
        scheduler.step()


if __name__ == '__main__':
    args = parse_arguments()
    config_path = args.config_path

    config = utils.load_config(config_path)

    ray.init()

    tune_config = {
        "learning_rate": tune.loguniform(config['learning_rate']['min'], config['learning_rate']['max']),
        "weight_decay": tune.uniform(config['weight_decay']['min'], config['weight_decay']['max']),
        "lr_scheduler": tune.choice(config['lr_scheduler']['choices']),
        **config  # Include all other config parameters as they are
    }

    scheduler = ASHAScheduler(
        metric="val_f1",
        mode="max",
        max_t=config["num_epochs"],
        grace_period=1,
        reduction_factor=2
    )

    analysis = tune.run(
        train_model,
        resources_per_trial={"cpu": config["workers"], "gpu": 1},
        config=tune_config,
        num_samples=10,
        scheduler=scheduler,
        storage_path="./tune/"
    )

    print("Best hyperparameters found were: ", analysis.best_config)
