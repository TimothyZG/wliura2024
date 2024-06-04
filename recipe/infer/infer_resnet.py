import torch
import torchvision.models as models
import torch.nn as nn
import pandas as pd
from torch.nn.functional import softmax
import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.dataloader.dataloaders import get_dataloader, get_numclass
# from recipe.train.dataloader.dataloaders import get_dataloader, get_numclass

parser = argparse.ArgumentParser()
parser.add_argument("-r","--data_root", default = "./Data", 
                    help="Directory to which dataset should be downloaded", type=str)
parser.add_argument("-m", "--model_path", required=True, 
                    help="path to model", type=str)
parser.add_argument("-d","--dataset", required = True, help="desired dataset", type=str, 
                    choices=["Cars","DTD","MNIST","iWildCam","GTSRB","EuroSAT","Resisc45","SUN397","SVHN"])
parser.add_argument("-n","--num_workers", default=1, help="number of workers needed", type=int)
parser.add_argument("--batch_size", default=32, help="Specift batch size for dataloaders", type=int)
parser.add_argument("-a","--model_arch", required = True, choices = ["Resnet18", "Resnet50", "Resnet101"],type=str)
# parser.add_argument("-p","--pred_path", required=True, type=str)
# parser.add_argument("-t","--target_path", required=True, type=str)
args = parser.parse_args()

dataset_root = args.data_root
dataset = args.dataset
num_workers = args.num_workers
model_path = args.model_path
model_arch = args.model_arch
bs = args.batch_size
pred_path = "Pred/"+dataset+"_"+model_arch+".csv"
target_path = "Pred/"+"target_"+dataset+".csv"
num_class = get_numclass(dataset)

if model_arch=="Resnet18":
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_class)
elif model_arch=="Resnet50":
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_class)
elif model_arch=="Resnet101":
    model = models.resnet101(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_class)

model.load_state_dict(torch.load(model_path))
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
_, val_dataloader, test_dataloader = get_dataloader(ds = dataset, root=dataset_root, bs=bs, nworkers=num_workers)

columns = [f'class_{i}' for i in range(num_class)]
pred_df = pd.DataFrame(columns=columns)
targets_df = pd.DataFrame(columns=['target'])

# Disable gradient computation for evaluation
with torch.no_grad():
    for data, targets in test_dataloader:
        # Move data to the appropriate device
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        outputs = model(data)

        probs = softmax(outputs, dim=1)

        pred_df = pd.concat([pred_df,pd.DataFrame(probs.cpu().numpy(), columns=columns)], ignore_index=True)
        targets_df = pd.concat([targets_df,pd.DataFrame({'target': targets.cpu().numpy()})], ignore_index=True)

pred_df.to_csv(pred_path, index=False)
targets_df.to_csv(target_path, index=False)
print("Predictions successfully saved to CSV files.")