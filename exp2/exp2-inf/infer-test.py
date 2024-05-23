from caltech256loader import setup_caltech256
import torch
import torchvision.models as models
import torch.nn as nn
import pandas as pd
from torch.nn.functional import softmax


num_class=257
res18 = models.resnet18(weights=None)
res18.fc = nn.Linear(res18.fc.in_features, num_class)
res50 = models.resnet50(weights=None)
res50.fc = nn.Linear(res50.fc.in_features, num_class)
res101 = models.resnet101(weights=None)
res101.fc = nn.Linear(res101.fc.in_features, num_class)

res18.load_state_dict(torch.load('exp2/models/pt-resnet18-cal256-ts.pth'))
res50.load_state_dict(torch.load('exp2/models/pt-resnet50-cal256-ts.pth'))
# Load the state dict from the file
state_dict = torch.load('exp2/models/pt-resnet101-cal256-ts.pth')
# Remove the 'module.' prefix from each key [This is due to rsnet101 being trained with parallelism]
new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
# Load the adjusted state dict into your model
res101.load_state_dict(new_state_dict)
# res101.load_state_dict(torch.load('./models/pt-resnet101-cal256-2.pth'))
res18.eval()
res50.eval()
res101.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
res18 = res18.to(device)
res50 = res50.to(device)
res101 = res101.to(device)
_, val_dataloader, test_dataloader = setup_caltech256(data_root='Data/Caltech256', batch_size=128, num_workers=4)

columns = [f'class_{i}' for i in range(num_class)]
predictions_df_18 = pd.DataFrame(columns=columns)
predictions_df_50 = pd.DataFrame(columns=columns)
predictions_df_101 = pd.DataFrame(columns=columns)
targets_df = pd.DataFrame(columns=['target'])

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
        
        probs18 = softmax(outputs18, dim=1)
        probs50 = softmax(res50(data), dim=1)
        probs101 = softmax(res101(data), dim=1)
        
        predictions_df_18 = pd.concat([predictions_df_18,pd.DataFrame(probs18.cpu().numpy(), columns=columns)], ignore_index=True)
        predictions_df_50 = pd.concat([predictions_df_50,pd.DataFrame(probs50.cpu().numpy(), columns=columns)], ignore_index=True)
        predictions_df_101 = pd.concat([predictions_df_101,pd.DataFrame(probs101.cpu().numpy(), columns=columns)], ignore_index=True)
        targets_df = pd.concat([targets_df,pd.DataFrame({'target': targets.cpu().numpy()})], ignore_index=True)

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

predictions_df_18.to_csv('exp2/exp2-inf/predictions_res18_ts.csv', index=False)
predictions_df_50.to_csv('exp2/exp2-inf/predictions_res50_ts.csv', index=False)
predictions_df_101.to_csv('exp2/exp2-inf/predictions_res101_ts.csv', index=False)
targets_df.to_csv('exp2/exp2-inf/labels_ts.csv', index=False)
print("Predictions successfully saved to CSV files.")