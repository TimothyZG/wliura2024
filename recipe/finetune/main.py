import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import wandb
import os
import argparse
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.dataloaders import get_dataloader, get_numclass

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dr",
        "--dataset_root",
        type=str,
        default = "./Data",
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "-mr",
        "--model_root",
        type=str,
        default = "./Models",
        help="Where to download the models.",
    )
    parser.add_argument(
        "-d",
        "--dataset", 
        required = True, 
        help="desired dataset", 
        type=str, 
        choices=["iWildCam","EuroSAT","SUN397","Caltech256","CIFAR10","DTD", "MNIST"])
    parser.add_argument(
        "-e",
        "--num_epochs",
        type = int,
        default = 16
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-m",
        "--model_name", 
        required = True, 
        choices = ["Resnet18", "Resnet50", "Resnet101","EffNet_S","EffNet_M","EffNet_L","ViT_B_16","ViT_L_16","ViT_H_14"],
        type=str
    )
    parser.add_argument(
        "-nle",
        "--num_epochs_linear",
        type = int,
        default = 5
    )
    parser.add_argument(
        "-r",
        "--resize",
        type = int,
        default = 224
    )
    parser.add_argument("-pn","--wandb_project_name", required = True, type=str)
    return parser.parse_args()

def make_predictions(pred_path, pred_cal_path, target_path):
    columns = [f'class_{i}' for i in range(num_classes)]
    pred_df = pd.DataFrame(columns=columns)
    targets_df = pd.DataFrame(columns=['target'])
    with torch.no_grad():
        for batch_idx, labeled_batch in enumerate(test_dataloader):
            if (args.dataset=="iWildCam" or args.dataset=="iWildCamOOD"):
                data, targets, metadata = labeled_batch
            else: 
                data, targets = labeled_batch
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
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
    if (args.dataset=="iWildCam" or args.dataset=="iWildCamOOD"):
        model_calibrated.set_temperature(val_dataloader,iwildcam=True)
    else:
        model_calibrated.set_temperature(val_dataloader)
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch_idx, labeled_batch in enumerate(test_dataloader):
            if (args.dataset=="iWildCam" or args.dataset=="iWildCamOOD"):
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
    
if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    DEVICE = 'cuda'
    model_path = args.model_root+"/"+args.model_name+"-"+args.dataset+".pth"
    pft_model_path = args.model_root+"/"+args.model_name+"-"+args.dataset+"-pft"+".pth"
    os.makedirs(args.dataset_root, exist_ok=True)
    os.makedirs(args.model_root, exist_ok=True)
    num_classes = get_numclass(args.dataset)
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.init(
        # set the wandb project where this run will be logged
        project=args.wandb_project_name,
        name=args.model_name+"-ms",
        config={
            "learning_rate": args.lr,
            "architecture": args.model_name,
            "dataset": args.dataset,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "lr_scheduler": "cosine"
        }
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"current device is {device}")
    
    # ========== Load Data =========== 
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(ds=args.dataset,root=args.dataset_root,bs=args.batch_size,nworkers=args.workers,resize = args.resize)
    # ========== Get Pretrained Model =========== 
    if(args.model_name=="Resnet18"):
        model = models.resnet18(weights='IMAGENET1K_V1')
    elif(args.model_name=="Resnet50"):
        model = models.resnet50(weights='IMAGENET1K_V2')
    elif(args.model_name=="Resnet101"):
        model = models.resnet101(weights='IMAGENET1K_V2')
    elif(args.model_name=="EffNet_S"):
        model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
    elif(args.model_name=="EffNet_M"):
        model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
    elif(args.model_name=="EffNet_L"):
        model = models.efficientnet_v2_l(weights="IMAGENET1K_V1")
    elif(args.model_name=="ViT_B_16"):
        model = models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1")
    elif(args.model_name=="ViT_L_16"):
        model = models.vit_l_16(weights="IMAGENET1K_SWAG_E2E_V1")
    elif(args.model_name=="ViT_H_14"):
        model = models.vit_h_14(weights="IMAGENET1K_SWAG_E2E_V1")
    else:
        raise Exception(f"Model Architecture Unrecgonized: {args.model_name}")
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    if 'EffNet' in args.model_name:
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif 'ViT' in args.model_name:
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)
    # Swap fc layer for finetune dataset
    elif 'Resnet' in args.model_name:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise Exception(f"Unrecgonized Model Architecture given: {args.model_name}")
    
    pred_path_lp = "lp-ft/"+args.dataset+"_"+args.model_name+"_lp.csv"
    pred_cal_path_lp = "lp-ft/cal_"+args.dataset+"_"+args.model_name+"_lp.csv"
    pred_path_ft = "lp-ft/"+args.dataset+"_"+args.model_name+"_ft.csv"
    pred_cal_path_ft = "lp-ft/cal_"+args.dataset+"_"+args.model_name+"_ft.csv"
    target_path = "lp-ft/"+"target_"+args.dataset+".csv"
    # =========== Define Optimizer =============
    curr_lr = args.lr
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    # =========== Define Loss Function =============
    loss_function = nn.CrossEntropyLoss()
    
    # ========== Fully Fine-Tune Model =========== 
    model.train()
    model = nn.DataParallel(model)
    model = model.to(device)
    print(f"model is moved to {device}")
    max_acc = 0
    acc_threshold = 0.05
    min_loss = 10
    # ========== Training Routine ============
    for epoch in range(args.num_epochs):
        # =========== Layer by Layer Unfreezing =========
        if(epoch == args.num_epochs_linear):
            torch.save(model.state_dict(), model_path)
            print(f"partially finetuined model saved to {pft_model_path}")
            make_predictions(pred_path_lp, pred_cal_path_lp, target_path)
            for param in model.parameters():
                param.requires_grad = True
            # curr_lr = 1e-5
            curr_lr = 1e-4
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=curr_lr, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs - epoch)  # Adjust T_max to remaining epochs
            # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.num_epochs - epoch)
            print("unfreezing entire Model, decreasing lr to 1e-4")
        if(epoch == args.num_epochs-10):
            curr_lr = 1e-5
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=curr_lr, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs - epoch)  # Adjust T_max to remaining epochs
            print("decreasing lr to 1e-5")
        total = 0
        correct = 0
        for batch_idx, labeled_batch in enumerate(train_dataloader):
            # Unpack Batch for iwildcam is slightly different since there's extra metadata
            if (args.dataset=="iWildCam" or args.dataset=="rxrx1"):
                data, targets, metadata = labeled_batch
            else: 
                data, targets = labeled_batch
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
            # Calculate val_accuracy
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            train_accuracy = 100 * correct / total
            # Log to wandb every $log_interval batches
            if batch_idx % 10 == 0:
                wandb.log({"epoch":epoch,
                        "batch":batch_idx,
                        "loss": loss.item(),
                        "train_accuracy": train_accuracy})
        # ============= EVAL ROUTINE ==============
        # Set the model to evaluation mode
        model.eval()

        # Initialize variables to track performance metrics
        total_correct = 0
        total_samples = 0

        # Disable gradient computation for evaluation
        with torch.no_grad():
            for batch_idx, labeled_batch in enumerate(val_dataloader):
                if (args.dataset=="iWildCam" or args.dataset=="rxrx1" or args.dataset=="fmow"):
                    data, targets, metadata = labeled_batch
                else: 
                    data, targets = labeled_batch
                # Move data to the appropriate device
                data, targets = data.to(device), targets.to(device)
                # Forward pass
                outputs = model(data)
                val_loss = loss_function(outputs, targets)
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                # Update performance metrics
                total_samples += targets.size(0)
                total_correct += (predicted == targets).sum().item()

        # Calculate val_accuracy
        val_accuracy = total_correct / total_samples
        
        # Log val_accuracy to wandb
        wandb.log({"epoch": epoch, "val_accuracy": val_accuracy, "val_loss":val_loss})

        # ========== Save Model =============
        # if (val_accuracy > max_acc-acc_threshold and val_loss < min_loss):
        if (val_accuracy > max_acc):
            torch.save(model.state_dict(), model_path)
            print(f"Best Model Updated (val_accuracy = {val_accuracy})")
            max_acc = val_accuracy
            # min_loss = val_loss
            
        # Print training progress
        curr_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch+1}/{args.num_epochs} Completed | Loss: {loss.item():.4f} | Accuracy: {train_accuracy:.2f}% | Val Accuracy: {val_accuracy:.2f} | Val Loss: {val_loss:.2f}| LR: {curr_lr}")
        model.train()
        scheduler.step()
    make_predictions(pred_path_ft, pred_cal_path_ft, target_path)
