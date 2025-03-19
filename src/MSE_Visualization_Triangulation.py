# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 15:34:34 2025

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

# Antenna spacing calculations (x-axis)
element_spacing=.16655136555555555/2
ant0 = 0-.29
ant1 = element_spacing*2-.29
ant2 = element_spacing*5-.29
ant3 = element_spacing*7-.29
d1 = ant0
d2 = ant2

# Loads baseband training and testing data
training_data = np.rot90(np.load('training_dataset_spreadcapt.npz', allow_pickle=True)['data'])
training_dataset = CustomDataset(training_data)
train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

testing_data = np.rot90(np.load('testing_dataset_spreadcapt.npz', allow_pickle=True)['data'])
testing_dataset = CustomDataset(testing_data)
test_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True)
testing_labels = testing_data[:, 2]
for X, y in test_dataloader:
    X=torch.cat((X[0],X[1]),0)
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

testing_data_baseband = np.rot90(np.load('testing_dataset_sampledBB.npz', allow_pickle=True)['data'])
testing_dataset_baseband = CustomDataset(testing_data_baseband)
test_dataloader_baseband = DataLoader(testing_dataset_baseband, batch_size=1, shuffle=True)
testing_labels_bb = testing_data_baseband[:, 2]
for X, y in test_dataloader_baseband:
    X=torch.cat((X[0],X[1]),0)
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Gets peak-power angle estimates from testing data
predicted_angles = np.zeros((len(testing_labels), 2))
for i in range(0, len(testing_data)):
    power_1 = testing_data[i][0][:, 4]
    power_2 = testing_data[i][1][:, 4]
    angle_1 = np.argmax(power_1) * 5
    angle_2 = np.argmax(power_2) * 5
    angle_estimate = np.array([angle_1, angle_2])
    predicted_angles[i] = angle_estimate
    
predicted_angles_bb = np.zeros((len(testing_labels_bb), 2))
for i in range(0, len(testing_data_baseband)):
    power_1 = testing_data_baseband[i][0][:, 4]
    power_2 = testing_data_baseband[i][1][:, 4]
    angle_1 = np.argmax(power_1) * 5
    angle_2 = np.argmax(power_2) * 5
    angle_estimate = np.array([angle_1, angle_2])
    predicted_angles_bb[i] = angle_estimate
    
# Triangulates model location prediction

def triangulate(angleVector, d1=-0.29, d2=0.126378414):
    n = len(angleVector)
    coords = np.zeros((n, 2))
    
    for i in range(n):
        angles = angleVector[i]
        a1, a2 = angles[0], angles[1]
        a1 += 90 # Converts reference frame for angles
        a2 += 90
        if (a1 == a2): # Edge case: Angles are the same
            x=0
            y=0
        else: # Normal case: Angles are different
            m1 = np.tan(np.deg2rad(a1))
            m2 = np.tan(np.deg2rad(a2))
            x = ((m1*d1) - (m2*d2)) / (m1 - m2)
            y = m1*(x - d1)
            if (y < 0): # Keeps estimate in front of array
                x *= -1
                y *= -1
        coord = np.array([x, y])
        coords[i] = coord
    vec=(coords!=[0,0])
    return coords, vec*1
predicted_coords, vec = triangulate(predicted_angles)
predicted_coords_bb,vecbb = triangulate(predicted_angles_bb)

# Triangulates label location from testing labels
testing_labels = testing_data[:, 2]
label_coords,vect = triangulate(testing_labels)
label_coords_bb,vect = triangulate(testing_labels_bb)

# Truncate predicted_coords to length of labels, clean data
# predicted_coords = predicted_coords[:n]
# label_coords = np.nan_to_num(label_coords, copy=False, nan=0.0, posinf=0, neginf=0)

# Compare MSE loss of two triangulated positions
vec = vec
vecbb = vecbb

avgMSE = sum(((np.square(predicted_coords*vec - vec*label_coords)).mean(axis=1)))/sum(vec[:, 0])
avgMSE_BB = sum(((np.square(predicted_coords_bb*vecbb - vecbb*label_coords_bb)).mean(axis=1)))/sum(vecbb[:, 0])
print(f"Average MSE for non-baseband predicted and actual coordinates: {avgMSE}")
print(f"Average MSE for baseband predicted and actual coordinates: {avgMSE_BB}")