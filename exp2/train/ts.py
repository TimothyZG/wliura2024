from temperature_scaling import ModelWithTemperature
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

res18.load_state_dict(torch.load('exp2/models/pt-resnet18-cal256-2.pth'))
res50.load_state_dict(torch.load('exp2/models/pt-resnet50-cal256-2.pth'))
# Load the state dict from the file
state_dict = torch.load('exp2/models/pt-resnet101-cal256-2.pth')
# Remove the 'module.' prefix from each key [This is due to rsnet101 being trained with parallelism]
new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
# Load the adjusted state dict into your model
res101.load_state_dict(new_state_dict)

_, val_dataloader, test_dataloader = setup_caltech256(data_root='Data/Caltech256', batch_size=128, num_workers=4)

scaled_res18 = ModelWithTemperature(res18)
scaled_res18.set_temperature(val_dataloader)
temp18 = scaled_res18.state_dict()["temperature"]
print(f"resnet18 has {temp18 =}")

scaled_res50 = ModelWithTemperature(res50)
scaled_res50.set_temperature(val_dataloader)
temp50 =scaled_res50.state_dict()["temperature"]
print(f"resnet50 has {temp50 =}")

scaled_res101 = ModelWithTemperature(res101)
scaled_res101.set_temperature(val_dataloader)
temp101 =scaled_res101.state_dict()["temperature"]
print(f"resnet101 has {temp101 =}")

model_path18 = "exp2/models/pt-resnet18-cal256-ts.pth"
model_path50 = "exp2/models/pt-resnet50-cal256-ts.pth"
model_path101 = "exp2/models/pt-resnet101-cal256-ts.pth"
torch.save(scaled_res18.state_dict(), model_path18)
torch.save(scaled_res50.state_dict(), model_path50)
torch.save(scaled_res101.state_dict(), model_path101)