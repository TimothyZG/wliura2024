from caltech256loader import setup_caltech256
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import wandb

num_epochs = 80
log_interval = 100
num_classes = 257
lr=0.001
batch_size = 128
# Define a path to save the model
model_path = "models/pt-resnet101-cal256-2.pth"

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Exp3",
    name="res101-cal256",
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "Resnet101",
    "dataset": "cal256",
    "epochs": num_epochs,
    "batch_size": batch_size
    }
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"gpu availability is {torch.cuda.is_available()}, \
current device is {device}")

train_dataloader, val_dataloader, test_dataloader = setup_caltech256(data_root='Data/Caltech256', batch_size=batch_size, num_workers=4)

model = models.resnet101(weights='IMAGENET1K_V2')

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
loss_function = nn.CrossEntropyLoss()

# Swap fc layer for finetune dataset (StanfordCars in this case)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    
model.train()
model = model.to(device)
print(f"model is moved to {device}")

losses = []
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
        losses.append(loss.item())

        # Perform backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        
        if batch_idx % log_interval == 0:
            # log metrics to wandb
            wandb.log({"epoch":epoch,
                       "batch":batch_idx,
                       "loss": loss.item(),
                       "accuracy": accuracy})
    # Print training progress
    print(f"Epoch {epoch+1}/{num_epochs} Completed |\
        Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")
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


# Save the model state dictionary
torch.save(model.state_dict(), model_path)