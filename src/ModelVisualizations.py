# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 17:22:22 2025

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
import matplotlib.pyplot as plt
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
        #vector = vector.view(vector.size(0), -1)  # Reshape `vector` to be 2D with shape (batch_size, 3)
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

# Loads trained models
def load_model(checkpoint_path):
    # Loads trained baseband model
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return model, optimizer

# Loads evaluation data
testing_data = np.rot90(np.load('testing_dataset_sampled.npz', allow_pickle=True)['data'])
testing_dataset = CustomDataset(testing_data)
test_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True)
for X, y in test_dataloader:
    X=torch.cat((X[0],X[1]),0)
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

testing_data_baseband = np.rot90(np.load('testing_dataset_sampledBB.npz', allow_pickle=True)['data'])
testing_dataset_baseband = CustomDataset(testing_data_baseband)
test_dataloader_baseband = DataLoader(testing_dataset_baseband, batch_size=1, shuffle=True)
for X, y in test_dataloader_baseband:
    X=torch.cat((X[0],X[1]),0)
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Gets model predictions on evaluation data
def test(dataloader, model):
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

model, optimizer = load_model("models/norm10000")
predicted_angles_norm10 = test(test_dataloader_baseband, model)
model, optimizer = load_model("models/notdoa10000")
predicted_angles_notdoa10 = test(test_dataloader_baseband, model)
model, optimizer = load_model("models/spike10000")
predicted_angles_spike10 = test(test_dataloader, model)
model, optimizer = load_model("models/spike25000")
predicted_angles_spike25 = test(test_dataloader, model)
model, optimizer = load_model("models/spike50000")
predicted_angles_spike50 = test(test_dataloader, model)

model, optimizer = load_model("models/width10000")
predicted_angles_width10 = test(test_dataloader_baseband, model)
model, optimizer = load_model("models/width25000")
predicted_angles_width25= test(test_dataloader_baseband, model)
model, optimizer = load_model("models/width50000")
predicted_angles_width50 = test(test_dataloader_baseband, model)

# Triangulates model location predictions
def triangulate(angleVector, d1=-0.29, d2=0.126378414):
    n = len(angleVector)
    coords = np.zeros((n, 2))
    
    for i in range(n):
        angles = angleVector[i]
        a1, a2 = angles[0], angles[1]
        a1 += 90 # Converts reference frame for angles
        a2 += 90
        if (a1 == a2): # Edge case: Angles are the same
            r = 2.4
            x = 2.4 * np.cos(np.deg2rad(a1))
            y = 2.4 * np.sin(np.deg2rad(a1))
        else:
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
predicted_coords_norm10 = triangulate(predicted_angles_norm10)
predicted_coords_notdoa10 = triangulate(predicted_angles_notdoa10)
predicted_coords_spike10 = triangulate(predicted_angles_spike10)
predicted_coords_spike25 = triangulate(predicted_angles_spike25)
predicted_coords_spike50 = triangulate(predicted_angles_spike50)

predicted_coords_width10 = triangulate(predicted_angles_width10)
predicted_coords_width25 = triangulate(predicted_angles_width25)
predicted_coords_width50 = triangulate(predicted_angles_width50)

# Triangulates label location from testing labels
testing_labels = testing_data[:, 2]
label_coords = triangulate(testing_labels)

testing_labels_bb = testing_data_baseband[:, 2]
label_coords_bb = triangulate(testing_labels_bb)

# Calculate MSE for each model compared to labels
avgMSE_spike10 = np.mean((np.square(predicted_coords_spike10 - label_coords)).mean(axis=1))
avgMSE_spike25 = np.mean((np.square(predicted_coords_spike25 - label_coords)).mean(axis=1))
avgMSE_spike50 = np.mean((np.square(predicted_coords_spike50 - label_coords)).mean(axis=1))

avgMSE_width10 = np.mean((np.square(predicted_coords_width10 - label_coords_bb)).mean(axis=1))
avgMSE_width25 = np.mean((np.square(predicted_coords_width25 - label_coords_bb)).mean(axis=1))
avgMSE_width50 = np.mean((np.square(predicted_coords_width50 - label_coords_bb)).mean(axis=1))
avgMSE_norm10 = np.mean((np.square(predicted_coords_norm10 - label_coords_bb)).mean(axis=1))
avgMSE_notdoa10 = np.mean((np.square(predicted_coords_notdoa10 - label_coords_bb)).mean(axis=1))

print(f"Average MSE (spike10): {avgMSE_spike10}")
print(f"Average MSE (spike25): {avgMSE_spike25}")
print(f"Average MSE (spike50): {avgMSE_spike50}")

print(f"Average MSE (width10): {avgMSE_width10}")
print(f"Average MSE (width25): {avgMSE_width25}")
print(f"Average MSE (width50): {avgMSE_width50}")

# Plot using Matplotlib
def plotData():
    plt.figure()
    x = [10000, 25000, 50000]
    y_spike = [avgMSE_spike10, avgMSE_spike25, avgMSE_spike50]
    y_width = [avgMSE_width10, avgMSE_width25, avgMSE_width50]
    y_triangulation = [3.205975734644037] * 3 # Taken from triangulation code
    y_triangulationBB = [4.1923420979663035] * 3
    plt.plot(x, y_spike, label="ML without Baseband")
    plt.plot(x, y_width, label="ML with Baseband")
    plt.plot(x, y_triangulation, linestyle="dashed", label="Triangulation without Baseband (non-ML)")
    plt.plot(x, y_triangulationBB, linestyle="dashed", label="Triangulation with Baseband (non-ML)")
    
    plt.title("Algorithm MSE Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Testing Mean Squared Error")
    #plt.xlim([0, 55000])
    plt.ylim([0, 4.5])
    plt.legend()
    plt.show()
plotData()
def barChart():
    plt.figure()
    x = ["1","2","3","4" ,"5","6" ]
    y=[3.205975734644037,4.1923420979663035,avgMSE_width10,avgMSE_spike10,avgMSE_notdoa10,avgMSE_norm10]
    fig, ax = plt.subplots()
    bar_labels = ["1. triangulation no bandwidth","2. triangulation bandwidth","3. ML bandwidth","4. ML no bandwidth" ,"5. ML no TDoA","6. ML TDoA normalized" ]
    bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange', "tab:purple","tab:green"]
    ax.bar(x, y, label=bar_labels,color=bar_colors)
    plt.ylim([0, 10])
    ax.set_ylabel('MSE')
    ax.set_title('MSE by method for 10000 epoch')
    ax.legend(title='method')

    plt.show()
barChart()
    