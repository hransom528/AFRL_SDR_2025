# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:11:00 2025

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
load = True # Load in model checkpoint to continue training

def harshloss(output, target,e):
    loss = torch.mean((output*e - target*e)**2)
    return loss

# Define model/training parameters
batch_size = 64
test_dataloader=0
train_dataloader=0
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Define custom dataset for data
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

# Perform dataloading of data
def dataload(): 
    global train_dataloader
    global test_dataloader
    #data for training model on
    training_data = np.rot90(np.load('training_dataset_sampledBB.npz', allow_pickle=True)['data'])
    training_dataset = CustomDataset(training_data)
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    #data for testing model on
    testing_data = np.rot90(np.load('testing_dataset_sampledBB.npz', allow_pickle=True)['data'])
    testing_dataset = CustomDataset(testing_data)
    test_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True)
    for X, y in test_dataloader:
        X=torch.cat((X[0],X[1]),0)
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        return(X,y)
        break
    
dataload()

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
        )# model structure

    def forward(self, x):
        vector1, vector2 = x
        x = torch.cat((vector1, vector2), dim=1)
        x = self.flatten(x)#format data so as to make it a valid input

        logits = self.linear_relu_stack(x) 
        return logits

model = NeuralNetwork().to(device)
print(model)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
if load:
    checkpoint = torch.load('modelcheckpoint', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Train the model with training dataset
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
   
    for batch, (X, y) in enumerate(dataloader):
        
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        print(loss)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Test the model against testing dataset
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad(): #disable training on testing data
        for batch, (X, y) in enumerate(dataloader):
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.sum(abs(pred-y),1)<5).type(torch.float).sum().item()#display % of predictions within tolerance
    test_loss /= num_batches
    correct /= size
    print(test_loss)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
# Performs training and testing over many epochs
epochs = 10000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
test(test_dataloader, model, loss_fn)
print("Done!")
torch.save(model.state_dict(), 'model_sim_ongoing.pth')#model state
torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 'modelcheckpointBB')# saves model state for further training
