import torch
import torch.nn as nn
import wandb
import os
import sys
import utils
import argparse
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.wandb import WandbLoggerCallback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.dataloaders import get_dataloader, get_numclass

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning script")
    parser.add_argument('--config-path', type=str, required=True, help='Path to the config file')
    return parser.parse_args()

def train_model(config, checkpoint_dir=None):
    num_classes = get_numclass(config["dataset"])
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.init(
        project=config['wandb_project_name'],
        name=f"{config['model_name']}-ms",
        config=config,
        reinit=True
    )
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"current device is {device}")

    train_dataloader, val_dataloader, id_test_dataloader, ood_test_dataloader = get_dataloader(
        ds=config['dataset'],
        root=config['dataset_root'],
        bs=config['batch_size'],
        nworkers=config['workers'],
        resize=config['resize']
    )
    
    model = utils.get_model(config['model_name'])
    model = utils.freeze_model(model)
    model = utils.swap_head(model, config['model_name'], num_classes)

    optimizer, scheduler = utils.init_optimizer_scheduler(model, config, config['num_epochs'])
    loss_function = nn.CrossEntropyLoss()

    model.train()
    model = nn.DataParallel(model)
    model = model.to(device)
    print(f"model is moved to {device}")

    for epoch in range(config['num_epochs']):
        if config['num_epochs_linear'] > 0 and epoch == config['num_epochs_linear']:
            for param in model.parameters():
                param.requires_grad = True
            optimizer, scheduler = utils.init_optimizer_scheduler(model, config, config['num_epochs'] - epoch)
        total = 0
        correct = 0
        for batch_idx, labeled_batch in enumerate(train_dataloader):
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
            train_accuracy = 100 * correct / total
            if batch_idx % 100 == 0:
                wandb.log({"epoch": epoch, "batch": batch_idx, "loss": loss.item(), "train_accuracy": train_accuracy})

        val_accuracy, val_loss, val_f1 = utils.evaluate(model, val_dataloader, loss_function, device)
        id_test_accuracy, _, _ = utils.evaluate(model, id_test_dataloader, loss_function, device)
        ood_test_accuracy, _, _ = utils.evaluate(model, ood_test_dataloader, loss_function, device)

        utils.log_metrics(epoch, batch_idx, loss, train_accuracy, val_accuracy, val_loss, id_test_accuracy, ood_test_accuracy, val_f1)

        tune.report(val_accuracy=val_accuracy, val_loss=val_loss, val_f1=val_f1)

        curr_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch+1}/{config['num_epochs']} Completed | Loss: {loss.item():.4f} | Accuracy: {train_accuracy:.2f}% | Val F1: {val_f1:.2f} | Val Loss: {val_loss:.2f}| LR: {curr_lr}")
        model.train()
        scheduler.step()

if __name__ == '__main__':
    args = parse_arguments()
    config_path = args.config_path

    config = utils.load_config(config_path)

    ray.init()

    tune_config = {
        "learning_rate": tune.loguniform(config['learning_rate']['min'], config['learning_rate']['max']),
        "weight_decay": tune.loguniform(config['weight_decay']['min'], config['weight_decay']['max']),
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
        callbacks=[WandbLoggerCallback(
            project=config['wandb_project_name'],
            log_config=True
        )]
    )

    print("Best hyperparameters found were: ", analysis.best_config)
