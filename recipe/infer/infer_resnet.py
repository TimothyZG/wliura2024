import torch
import torchvision.models as models
import torch.nn as nn
import pandas as pd
from torch.nn.functional import softmax
import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.dataloaders import get_dataloader, get_numclass

parser = argparse.ArgumentParser()
parser.add_argument("-r","--data_root", default = "./Data", 
                    help="Directory to which dataset should be downloaded", type=str)
parser.add_argument("-m", "--model_path", required=True, 
                    help="path to model", type=str)
parser.add_argument("-d","--dataset", required = True, help="desired dataset", type=str, 
                    choices=["Cars","DTD","MNIST","iWildCam","GTSRB","EuroSAT","Resisc45","SUN397","SVHN","iWildCamOOD","CIFAR10","CINIC10","Caltech256"])
parser.add_argument("-n","--num_workers", default=1, help="number of workers needed", type=int)
parser.add_argument("--batch_size", default=32, help="Specift batch size for dataloaders", type=int)
parser.add_argument("-a","--model_arch", required = True, choices = ["Resnet18", "Resnet50", "Resnet101","EffNet_S","EffNet_M","EffNet_L","ViT_B_16","ViT_L_16","ViT_H_14"],type=str)
parser.add_argument(
        "-rs",
        "--resize",
        type = int,
        default = 224
    )
args = parser.parse_args()

dataset_root = args.data_root
dataset = args.dataset
num_workers = args.num_workers
model_path = args.model_path
model_arch = args.model_arch
bs = args.batch_size
pred_path = "Pred/"+dataset+"_"+model_arch+".csv"
pred_cal_path = "Pred/cal_"+dataset+"_"+model_arch+".csv"
target_path = "Pred/"+"target_"+dataset+".csv"
num_classes = get_numclass(dataset)
if(args.model_arch=="Resnet18"):
    model = models.resnet18(weights=None)
elif(args.model_arch=="Resnet50"):
    model = models.resnet50(weights=None)
elif(args.model_arch=="Resnet101"):
    model = models.resnet101(weights=None)
elif(args.model_arch=="EffNet_S"):
    model = models.efficientnet_v2_s(weights=None)
elif(args.model_arch=="EffNet_M"):
    model = models.efficientnet_v2_m(weights=None)
elif(args.model_arch=="EffNet_L"):
    model = models.efficientnet_v2_l(weights=None)
elif(args.model_arch=="ViT_B_16"):
    model = models.vit_b_16(weights=None)
elif(args.model_arch=="ViT_L_16"):
    model = models.vit_l_16(weights=None)
elif(args.model_arch=="ViT_H_14"):
    model = models.vit_h_14(weights=None)
else:
    raise Exception(f"Model Architecture Unrecgonized: {args.model_arch}")
if 'EffNet' in args.model_arch:
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
elif 'ViT' in args.model_arch:
    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, num_classes)
# Swap fc layer for finetune dataset
elif 'Resnet' in args.model_arch:
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
else:
    raise Exception(f"Unrecgonized Model Architecture given: {args.model_arch}")


# Load the state dictionary
state_dict = torch.load(model_path)

# Remove 'module.' prefix if present
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

# model.load_state_dict(torch.load(model_path))
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
_, val_dataloader, test_dataloader = get_dataloader(ds = dataset, root=dataset_root, bs=bs, nworkers=num_workers, resize = args.resize)

columns = [f'class_{i}' for i in range(num_classes)]
pred_df = pd.DataFrame(columns=columns)
targets_df = pd.DataFrame(columns=['target'])

# Disable gradient computation for evaluation
with torch.no_grad():
    for batch_idx, labeled_batch in enumerate(test_dataloader):
        if (dataset=="iWildCam" or dataset=="iWildCamOOD"):
            data, targets, metadata = labeled_batch
        else: 
            data, targets = labeled_batch
        # Move data to the appropriate device
        data, targets = data.to(device), targets.to(device)
        # Forward pass
        outputs = model(data)

        # probs = softmax(outputs, dim=1)

        pred_df = pd.concat([pred_df,pd.DataFrame(outputs.cpu().numpy(), columns=columns)], ignore_index=True)
        targets_df = pd.concat([targets_df,pd.DataFrame({'target': targets.cpu().numpy()})], ignore_index=True)

pred_df.to_csv(pred_path, index=False)
targets_df.to_csv(target_path, index=False)
print("Predictions successfully saved to CSV files.")

pred_df_cal = pd.DataFrame(columns=columns)
# ======== Temperature Scaling ========
from infer.temperature_scaling import ModelWithTemperature
model_calibrated = ModelWithTemperature(model)

# Tune the model temperature, and save the results
if (dataset=="iWildCam" or dataset=="iWildCamOOD"):
    model_calibrated.set_temperature(val_dataloader,iwildcam=True)
else:
    model_calibrated.set_temperature(val_dataloader)
# Disable gradient computation for evaluation
with torch.no_grad():
    for batch_idx, labeled_batch in enumerate(test_dataloader):
        if (dataset=="iWildCam" or dataset=="iWildCamOOD"):
            data, targets, metadata = labeled_batch
        else: 
            data, targets = labeled_batch
        # Move data to the appropriate device
        data, targets = data.to(device), targets.to(device)
        # Forward pass
        outputs = model_calibrated(data)

        # probs = softmax(outputs/model_calibrated.temperature, dim=1)

        pred_df_cal = pd.concat([pred_df_cal,pd.DataFrame(outputs.cpu().numpy(), columns=columns)], ignore_index=True)

pred_df_cal.to_csv(pred_cal_path, index=False)
print("Calibrated Predictions successfully saved to CSV files.")