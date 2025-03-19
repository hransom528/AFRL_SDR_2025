# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 18:52:42 2025

@author: exx
"""

# Imports
import os
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np

# PyTorch Definitions
batch_size = 64 # TODO: Check batch size
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
loss_fn = nn.MSELoss()

# Defines custom dataset for simulation data
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       
        chunk = self.data[idx]
        vector1= torch.tensor(chunk[0], dtype=torch.float32).to(device)
        vector2= torch.tensor(chunk[1], dtype=torch.float32).to(device) 
        label = torch.tensor(chunk[2], dtype=torch.float32).to(device)
        return (vector1, vector2), label
    
# Defines Neural Network structure
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(370, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        vector1, vector2 = x
        vector1[0]=vector1[0]
        vector2[0]=vector2[0]
        x = torch.cat((vector1, vector2), dim=1)
        x = self.flatten(x)

        logits = self.linear_relu_stack(x)
        return logits

# Antenna spacing calculations (x-axis)
element_spacing=.16655136555555555/2
ant0 = 0-.29
ant1 = element_spacing*2-.29
ant2 = element_spacing*5-.29
ant3 = element_spacing*7-.29
d1 = ant0
d2 = ant2

# Loads trained baseband model
model = NeuralNetwork().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

# TODO: Verify correct model being loaded in for captured data
checkpoint = torch.load("modelcheckpointBB", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()

# Loads baseband training and testing data
training_data = np.rot90(np.load('training_dataset_spreadcapt.npz', allow_pickle=True)['data'])
training_dataset = CustomDataset(training_data)
train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

testing_data = np.rot90(np.load('testing_dataset_spreadcapt.npz', allow_pickle=True)['data'])
testing_dataset = CustomDataset(testing_data)
test_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True)
for X, y in test_dataloader:
    X=torch.cat((X[0],X[1]),0)
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Gets model prediction on baseband data
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    predicted_angles = np.zeros((num_batches, 2))
    model.eval()
    with torch.no_grad():
        ind = 0
        for batch, (X, y) in enumerate(dataloader):
            y = y.to(device)
            pred = model(X).cpu().numpy().ravel()
            predicted_angles[ind] = pred
            ind += 1
    return predicted_angles
predicted_angles = test(test_dataloader, model, loss_fn)

# Triangulates model location prediction
def triangulate(angleVector, d1=-0.29, d2=0.126378414):
    n = len(angleVector)
    coords = np.zeros((n, 2))
    
    for i in range(n):
        angles = angleVector[i]
        a1, a2 = angles[0], angles[1]
        a1 += 90 # Converts reference frame for angles
        a2 += 90
        m1 = np.tan(np.deg2rad(a1))
        m2 = np.tan(np.deg2rad(a2))
        x = ((m1*d1) - (m2*d2)) / (m1 - m2)
        y = m1*(x - d1)
        if (y < 0): # Keeps estimate in front of array
            x *= -1
            y *= -1
        
        coord = np.array([x, y])
        coords[i] = coord
    return coords
predicted_coords = triangulate(predicted_angles)

# Triangulates label location from testing labels
testing_labels = testing_data[:, 2]
n = len(testing_labels)
label_coords = triangulate(testing_labels)

# Truncate predicted_coords to length of labels, clean data
predicted_coords = predicted_coords[:n]
label_coords = np.nan_to_num(label_coords, copy=False, nan=0.0, posinf=0, neginf=0)

# Compare MSE loss of two triangulated positions
mse = (np.square(predicted_coords - label_coords)).mean(axis=1)
avgMSE = np.mean(mse)
print(f"Average MSE for predicted and actual coordinates: {avgMSE}")