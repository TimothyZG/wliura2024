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
import utils
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
        choices=["iWildCam","rxrx1"]
    )
    parser.add_argument(
        "-s",
        "--suffix",
        required = True,
        help="suffix of model", 
        type=str
    )
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
        default=1e-4,
    )
    parser.add_argument(
        "-m",
        "--model_name", 
        required = True, 
        choices = ["Resnet18","Resnet34","Resnet50","Resnet101","Resnet152","EffNet_S","EffNet_M","EffNet_L","ViT_B_16","ViT_L_16","ViT_H_14"],
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
    parser.add_argument(
        "-lrs",
        "--lr_scheduler",
        required = True,
        choices = ["cosine","exp","step","NONE"],
        type=str
    )
    parser.add_argument("-pn","--wandb_project_name", required = True, type=str)
    return parser.parse_args()

    
if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    model_path = args.model_root+"/"+args.model_name+"-"+args.dataset+"-"+args.suffix+".pth"
    lp_model_path = args.model_root+"/"+args.model_name+"-"+args.dataset+"-pft"+".pth"
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
            "suffix": args.suffix,
            "lr_scheduler": args.lr_scheduler,
            "num_lp_epochs": args.num_epochs_linear
        }
    )
    torch.manual_seed(args.suffix)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"current device is {device}")
    
    # ========== Load Data =========== 
    train_dataloader, val_dataloader, id_test_dataloader, ood_test_dataloader = get_dataloader(ds=args.dataset,root=args.dataset_root,bs=args.batch_size,nworkers=args.workers,resize = args.resize)
    
    # ========== Get Pretrained Model =========== 
    model = utils.get_model(args.model_name)
    # Freeze all layers
    model = utils.freeze_model(model)
    model = utils.swap_head(model,args.model_name,num_classes)
    pred_path_lp = "ss/"+args.dataset+"_ind_"+args.model_name+"lp"+args.suffix+".csv"
    pred_path_ft = "ss/"+args.dataset+"_ind_"+args.model_name+"ft"+args.suffix+".csv"
    pred_path_lp_ood = "ss/"+args.dataset+"_ood_"+args.model_name+"lp"+args.suffix+".csv"
    pred_path_ft_ood = "ss/"+args.dataset+"_ood_"+args.model_name+"ft"+args.suffix+".csv"
    target_path = "ss/"+"target_"+args.dataset+"_ind.csv"
    target_path_ood = "ss/"+"target_"+args.dataset+"_ood.csv"
    # =========== Define Optimizer =============
    curr_lr = args.lr
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = utils.get_lrs(args.lr_scheduler, optimizer, args.num_epochs)
    # =========== Define Loss Function =============
    loss_function = nn.CrossEntropyLoss()
    
    # ========== Fully Fine-Tune Model =========== 
    model.train()
    model = nn.DataParallel(model)
    model = model.to(device)
    print(f"model is moved to {device}")
    max_acc = 0
    # ========== Training Routine ============
    for epoch in range(args.num_epochs):
        # =========== Layer by Layer Unfreezing =========
        if((args.num_epochs_linear > 0) and (epoch == args.num_epochs_linear)):
            torch.save(model.state_dict(), model_path)
            print(f"linear probed model saved to {lp_model_path}")
            utils.make_predictions(pred_path_lp, target_path, pred_path_lp_ood, target_path_ood, model, num_classes, id_test_dataloader, ood_test_dataloader, device)
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = utils.get_lrs(args.lr_scheduler, optimizer, args.num_epochs-epoch)
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
            if batch_idx % 100 == 0:
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
        
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch_idx, labeled_batch in enumerate(id_test_dataloader):
                data, targets, metadata = labeled_batch
                data, targets = data.to(device), targets.to(device)
                # Forward pass
                outputs = model(data)
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                # Update performance metrics
                total_samples += targets.size(0)
                total_correct += (predicted == targets).sum().item()
        # Calculate val_accuracy
        id_test_accuracy = total_correct / total_samples
        # Log val_accuracy to wandb
        wandb.log({"epoch": epoch, "id_test_accuracy": id_test_accuracy})
        
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch_idx, labeled_batch in enumerate(ood_test_dataloader):
                data, targets, metadata = labeled_batch
                data, targets = data.to(device), targets.to(device)
                # Forward pass
                outputs = model(data)
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                # Update performance metrics
                total_samples += targets.size(0)
                total_correct += (predicted == targets).sum().item()
        # Calculate val_accuracy
        ood_test_accuracy = total_correct / total_samples
        # Log val_accuracy to wandb
        wandb.log({"epoch": epoch, "ood_test_accuracy": ood_test_accuracy})

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
    utils.make_predictions(pred_path_ft, target_path, pred_path_ft_ood, target_path_ood, model, num_classes, id_test_dataloader, ood_test_dataloader, device)
