import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="test-run",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "pseudo-model",
    "dataset": "pseudo-dataset",
    "epochs": 15,
    }
)

# simulate training
epochs = 15
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})
