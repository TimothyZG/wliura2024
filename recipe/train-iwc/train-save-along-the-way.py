import torch
import torch.nn as nn
import wandb
import os
import sys
import yaml
import utils
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.dataloaders import get_dataloader, get_numclass, load_iwildcam


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config-path', type=str, required=True, help='Path to the config file')
    parser.add_argument('--suffix', type=str, required=True, help='Suffix for the run')
    parser.add_argument('--data-path', type=str, required=True, help='Path to data file, should be tmpdir')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    config_path = args.config_path
    data_path = args.data_path
    suffix = args.suffix

    config = utils.load_config(config_path)

    model_path = os.path.join(config['model_root'], f"{config['model_name']}-{config['dataset']}-{suffix}.pth")
    lp_model_path = os.path.join(config['model_root'], f"{config['model_name']}-{config['dataset']}-lp.pth")
    os.makedirs(config['dataset_root'], exist_ok=True)
    os.makedirs(config['model_root'], exist_ok=True)
    num_classes = get_numclass(config["dataset"])
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.init(
        project=config['wandb_project_name'],
        name=f"{config['model_name']}-ms",
        config=config
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"current device is {device}")
    
    train_dataloader, val_dataloader, id_test_dataloader, ood_test_dataloader = get_dataloader(
        ds=config['dataset'],
        root=data_path,
        bs=config['batch_size'],
        nworkers=config['workers'],
        resize=config['resize']
    )
    
    model = utils.get_model(config['model_name'])
    model = utils.swap_head(model, config['model_name'], num_classes)
    pred_folder = "iwcsatw"
    target_path = f"{pred_folder}/target_{config['dataset']}_ind.csv"
    target_path_ood = f"{pred_folder}/target_{config['dataset']}_ood.csv"
    target_path_val = f"{pred_folder}/target_{config['dataset']}_val.csv"

    # optimizer, scheduler = utils.init_optimizer_scheduler(model, config, config['num_epochs'])
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    loss_function = nn.CrossEntropyLoss()

    model.train()
    model = nn.DataParallel(model)
    model = model.to(device)
    print(f"model is moved to {device}")
    max_f1 = 0

    for epoch in range(config['num_epochs']):
        pred_path_ind = f"{pred_folder}/{config['dataset']}_ind_{config['model_name']}{suffix}_e{epoch}.csv"
        pred_path_val = f"{pred_folder}/{config['dataset']}_val_{config['model_name']}{suffix}_e{epoch}.csv"
        pred_path_ood = f"{pred_folder}/{config['dataset']}_ood_{config['model_name']}{suffix}_e{epoch}.csv"
        total = 0
        correct = 0
        for batch_idx, labeled_batch in enumerate(train_dataloader):
            data, targets, metadata = labeled_batch
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            training_loss = loss_function(outputs, targets)
            training_loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            train_accuracy = 100 * correct / total
            if batch_idx % 100 == 0:
                wandb.log({"epoch": epoch, "batch": batch_idx, "loss": training_loss.item(), "train_accuracy": train_accuracy})

        val_accuracy, val_loss, f1 = utils.evaluate(model, val_dataloader, loss_function, device)
        id_test_accuracy, _, _ = utils.evaluate(model, id_test_dataloader, loss_function, device)
        ood_test_accuracy, _, _ = utils.evaluate(model, ood_test_dataloader, loss_function, device)

        utils.log_metrics(epoch, batch_idx, training_loss, train_accuracy, val_accuracy, val_loss, id_test_accuracy, ood_test_accuracy)
        curr_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch+1}/{config['num_epochs']} Completed | Loss: {training_loss.item():.4f} | Accuracy: {train_accuracy:.2f}% | Val Accuracy: {val_accuracy:.2f} | Val Loss: {val_loss:.2f}| LR: {curr_lr}")

        utils.make_predictions(pred_path_val, target_path_val, pred_path_ind, target_path, pred_path_ood, target_path_ood, model, num_classes, val_dataloader, id_test_dataloader, ood_test_dataloader, device)
        
        model.train()
        scheduler.step()