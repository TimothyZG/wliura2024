from wildloader import load_wild_train
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import wandb

num_epochs = 20
log_interval = 100
num_classes = 182
lr=0.001
batch_size = 128
w_decay = 0.1
input_size = (256,256)
# Define a path to save the model
model_path = "models/pt-resnet18-wild.pth"
wandb.init(
    project="Exp3-attempt",
    name="res18-iwildcam",
    config={
    "learning_rate": lr,
    "architecture": "Resnet18",
    "dataset": "iWildCam",
    "epochs": num_epochs,
    "batch_size": batch_size
    }
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dataloader = load_wild_train(batch_size,input_size)
model = models.resnet18(weights='IMAGENET1K_V1')

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
loss_function = nn.CrossEntropyLoss()
# Swap fc layer for finetune dataset (StanfordCars in this case)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

model.train()
model = model.to(device)
print(f"model is moved to {device}")

for epoch in range(num_epochs):
    total = 0
    correct = 0
    for batch_idx, labeled_batch in enumerate(train_dataloader):
        data, targets, metadata = labeled_batch
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
        
        if batch_idx % log_interval == 0:
            # log metrics to wandb
            wandb.log({"epoch":epoch,
                       "batch":batch_idx,
                       "loss": loss.item(),
                       "accuracy": accuracy})
    # Print training progress
    print(f"Epoch {epoch+1}/{num_epochs} Completed |\
        Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")

# Save the model state dictionary
torch.save(model.state_dict(), model_path)