import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import wandb
import os
import argparse

from dataloader.dataloaders import get_dataloader, get_numclass

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default = "./Data", help="Directory to which dataset should be downloaded", type=str)
parser.add_argument("--model_root", default = "./Models", help="Directory to which models should be saved", type=str)
parser.add_argument("-d","--dataset", required = True, help="desired dataset", type=str, 
                    choices=["Cars","DTD","MNIST","iWildCam","GTSRB","EuroSAT","Resisc45","SUN397"])
parser.add_argument("-n","--num_workers", default=1, help="number of workers needed", type=int)
parser.add_argument("--batch_size", default=32, help="Specift batch size for dataloaders", type=int)
parser.add_argument("-e","--num_epochs", default=80, type=int)
parser.add_argument("--log_interval", default=100, type=int)
parser.add_argument("--lr", default=0.0005, type=float)
parser.add_argument("-wd","--weight_decay", default=0.0, type=float)
parser.add_argument("-o","--optimizer", default="ADAM", type=str, choices = ["ADAM", "SGD"])
parser.add_argument("-mom","--momentum", default=0.9, type=float, help="momentum is only used when we use SGD as optimizer")
parser.add_argument("-m","--model_name", required = True, choices = ["Resnet18", "Resnet50", "Resnet101"],type=str)
parser.add_argument("-pn","--wandb_project_name", required = True, type=str)

args = parser.parse_args()

dataset_root = args.data_root
dataset = args.dataset
num_workers = args.num_workers
num_epochs = args.num_epochs
log_interval = args.log_interval
lr=args.lr
batch_size = args.batch_size
model_name = args.model_name
model_root = args.model_root
weight_decay = args.weight_decay
opt = args.optimizer
momentum = args.momentum
model_path = model_root+"/"+model_name+"-"+dataset+"-"+opt+".pth"

os.makedirs(dataset_root, exist_ok=True)
os.makedirs(model_root, exist_ok=True)
num_classes = get_numclass(dataset)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project=args.wandb_project_name,
    name=model_name+"-"+opt,
    config={
    "learning_rate": lr,
    "architecture": model_name,
    "dataset": dataset,
    "epochs": num_epochs,
    "batch_size": batch_size,
    "optimizer": opt,
    "weight_decay": weight_decay
    }
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"current device is {device}")

# ========== Load Data =========== 
train_dataloader, val_dataloader, test_dataloader = get_dataloader(ds=dataset,root=dataset_root,bs=batch_size,nworkers=num_workers)

# ========== Get Pretrained Model =========== 
if(model_name=="Resnet18"):
    model = models.resnet18(weights='IMAGENET1K_V1')
elif(model_name=="Resnet50"):
    model = models.resnet50(weights='IMAGENET1K_V2')
elif(model_name=="Resnet101"):
    model = models.resnet101(weights='IMAGENET1K_V2')
# Swap fc layer for finetune dataset
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# =========== Define Optimizer =============
if(opt=="ADAM"):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
elif(opt=="SGD"):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

# =========== Define Optimizer =============
loss_function = nn.CrossEntropyLoss()

# ========== Fully Fine-Tune Model =========== 
model.train()
model = model.to(device)
print(f"model is moved to {device}")

# ========== Training Routine ============
for epoch in range(num_epochs):
    total = 0
    correct = 0
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        data, targets = data.to(device), targets.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(data)
        loss = loss_function(outputs, targets)
        # Perform backward pass
        loss.backward()
        # Update model parameters
        optimizer.step()
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        # Log to wandb every $log_interval batches
        if batch_idx % log_interval == 0:
            wandb.log({"epoch":epoch,
                       "batch":batch_idx,
                       "loss": loss.item(),
                       "accuracy": accuracy})
    # Print training progress
    print(f"Epoch {epoch+1}/{num_epochs} Completed |\
        Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")
    

# ============= EVAL ROUTINE ==============
# Set the model to evaluation mode
model.eval()

# Initialize variables to track performance metrics
total_correct = 0
total_samples = 0

# Disable gradient computation for evaluation
with torch.no_grad():
    for data, targets in test_dataloader:
        # Move data to the appropriate device
        data, targets = data.to(device), targets.to(device)
        # Forward pass
        outputs = model(data)
        # Get predictions
        _, predicted = torch.max(outputs, 1)
        # Update performance metrics
        total_samples += targets.size(0)
        total_correct += (predicted == targets).sum().item()

# Calculate accuracy
accuracy = total_correct / total_samples
print(f"Accuracy on the test set: {100 * accuracy:.2f}%")

# ========== Save Model =============
torch.save(model.state_dict(), model_path)