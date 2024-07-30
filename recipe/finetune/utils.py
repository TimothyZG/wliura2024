import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import wandb
import pandas as pd

def make_predictions(pred_path_id, target_path_id, pred_path_ood, target_path_ood, model, num_classes, id_test_dataloader, ood_test_dataloader, device):
    model.eval()
    columns = [f'class_{i}' for i in range(num_classes)]
    
    pred_df = pd.DataFrame(columns=columns)
    targets_df = pd.DataFrame(columns=['target'])
    with torch.no_grad():
        for labeled_batch in id_test_dataloader:
            data, targets, metadata = labeled_batch
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            pred_df = pd.concat([pred_df,pd.DataFrame(outputs.cpu().numpy(), columns=columns)], ignore_index=True)
            targets_df = pd.concat([targets_df,pd.DataFrame({'target': targets.cpu().numpy()})], ignore_index=True)

    pred_df.to_csv(pred_path_id, index=False)
    targets_df.to_csv(target_path_id, index=False)
    print("InD Predictions successfully saved to CSV files.")
    
    pred_df_ood = pd.DataFrame(columns=columns)
    targets_df_ood = pd.DataFrame(columns=['target'])
    with torch.no_grad():
        for labeled_batch in ood_test_dataloader:
            data, targets, metadata = labeled_batch
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            pred_df_ood = pd.concat([pred_df_ood,pd.DataFrame(outputs.cpu().numpy(), columns=columns)], ignore_index=True)
            targets_df_ood = pd.concat([targets_df_ood,pd.DataFrame({'target': targets.cpu().numpy()})], ignore_index=True)

    pred_df_ood.to_csv(pred_path_ood, index=False)
    targets_df_ood.to_csv(target_path_ood, index=False)
    print("OOD Predictions successfully saved to CSV files.")
    model.train()
    
def get_model(model_name):
    if(model_name=="Resnet18"):
        model = models.resnet18(weights='IMAGENET1K_V1')
    elif(model_name=="Resnet50"):
        model = models.resnet50(weights='IMAGENET1K_V2')
    elif(model_name=="Resnet101"):
        model = models.resnet101(weights='IMAGENET1K_V2')
    elif(model_name=="EffNet_S"):
        model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
    elif(model_name=="EffNet_M"):
        model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
    elif(model_name=="EffNet_L"):
        model = models.efficientnet_v2_l(weights="IMAGENET1K_V1")
    else:
        raise Exception(f"Model Architecture Unrecgonized: {model_name}")
    return model

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def swap_head(model,model_name,num_classes):
    if 'EffNet' in model_name:
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    # Swap fc layer for finetune dataset
    elif 'Resnet' in model_name:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise Exception(f"Unrecgonized Model Architecture given: {model_name}")
    return model

def get_lrs(lr_scheduler, optimizer, T_max):
    if(lr_scheduler=="cosine"):
        return CosineAnnealingLR(optimizer, T_max=T_max)
    elif(lr_scheduler=="cosine"):
        return StepLR(optimizer, int(T_max/2),0.5)
    elif(lr_scheduler=="NONE"):
        return StepLR(optimizer, T_max,1)

