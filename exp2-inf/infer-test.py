from caltech256loader import setup_caltech256
import torch
import torchvision.models as models
import torch.nn as nn

num_class=257
res18 = models.resnet18(pretrained=False)
res18.fc = nn.Linear(res18.fc.in_features, num_class)
res50 = models.resnet50(pretrained=False)
res50.fc = nn.Linear(res50.fc.in_features, num_class)
res101 = models.resnet101(pretrained=False)
res101.fc = nn.Linear(res101.fc.in_features, num_class)

res18.load_state_dict(torch.load('./models/pt-resnet18-cal256-1.pth'))
res50.load_state_dict(torch.load('./models/pt-resnet50-cal256-1.pth'))
res101.load_state_dict(torch.load('./models/pt-resnet101-cal256-1.pth'))
res18.eval()
res50.eval()
res101.eval()
_, test_dataloader = setup_caltech256(data_root='Data/Caltech256', batch_size=128, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize variables to track performance metrics
total_correct18 = 0
total_correct50 = 0
total_correct101 = 0
total_samples = 0
# Disable gradient computation for evaluation
with torch.no_grad():
    for data, targets in test_dataloader:
        # Move data to the appropriate device
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        outputs18 = res18(data)
        outputs50 = res50(data)
        outputs101 = res101(data)

        # Get predictions
        _, predicted18 = torch.max(outputs18, 1)
        _, predicted50 = torch.max(outputs50, 1)
        _, predicted101 = torch.max(outputs101, 1)

        # Update performance metrics
        total_samples += targets.size(0)
        total_correct18 += (predicted18 == targets).sum().item()
        total_correct50 += (predicted50 == targets).sum().item()
        total_correct101 += (predicted101 == targets).sum().item()

# Calculate accuracy
accuracy18 = total_correct18 / total_samples
accuracy50 = total_correct50 / total_samples
accuracy101 = total_correct101 / total_samples
print(f"Resnet18 Accuracy on the test set: {100 * accuracy18:.2f}%")
print(f"Resnet50 Accuracy on the test set: {100 * accuracy50:.2f}%")
print(f"Resnet101 Accuracy on the test set: {100 * accuracy101:.2f}%")