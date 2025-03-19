# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:33:58 2025

@author: exx
"""

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
load = True # Load in model checkpoint
notdoa= False
# Harsh loss function
# TODO: May have to redefine harsh loss for new simulation dataset
def harshloss(output, target,e):
    loss = torch.mean((output*e - target*e)**2)
    return loss

# Define model/training parameters
batch_size = 64 # TODO: Check batch size
test_dataloader=0
train_dataloader=0
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Define custom dataset for simulation data
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       
        chunk = self.data[idx]
        vector1= torch.tensor(chunk[0], dtype=torch.float32).to(device)
        vector2= torch.tensor(chunk[1], dtype=torch.float32).to(device) 
        if notdoa:
            for v1i in range(len(vector1)):
                vector1[v1i][3]=0
            for v2i in range(len(vector2)):
                vector1[v2i][3]=0
        label = torch.tensor(chunk[2], dtype=torch.float32).to(device)
        return (vector1, vector2), label

# Perform dataloading of simulation data
def dataload():
    global train_dataloader
    global test_dataloader
    
    # Data format:
        # data[0]: Channel 1
        # data[1]: Channel 2
        # data[2]: Position estimation (label)
    
    #training_data = np.load('airsim1.npz', allow_pickle=True)['data']
    #print(training_data[0]).
    #print(training_data[2])
    #training_dataset_sampledBB.npz
    
    training_data = np.rot90(np.load('training_dataset_sampled.npz', allow_pickle=True)['data'])
    training_dataset = CustomDataset(training_data)
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    
    # TODO: Why is the same dataset being used for both training and testing?
    #testing_data = np.load('airsim1.npz', allow_pickle=True)['data']
    testing_data = np.rot90(np.load('testing_dataset_sampled.npz', allow_pickle=True)['data'])
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

model = NeuralNetwork().to(device)
print(model)
#loss_fn = nn.L1Loss()
loss_fn = nn.MSELoss()
#loss_fn = harshloss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
if load:
    checkpoint = torch.load('spike25000', weights_only=True)
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
        #loss=loss_fn(pred, y,8)
        loss = loss_fn(pred, y)
        #print(pred)
        #print(y)
       
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
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            #X, y = X.to(device), y.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #correct += (torch.sum(abs(pred-y)**2,1)**.5<0.0815).type(torch.float).sum().item()
            correct += (torch.sum(abs(pred-y),1)<5).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(test_loss)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
# Performs training and testing over many epochs
epochs = 25000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
test(test_dataloader, model, loss_fn)
print("Done!")
torch.save(model.state_dict(), 'spike50000.pth')#current one training
torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 'spike50000')#doing 2 rn