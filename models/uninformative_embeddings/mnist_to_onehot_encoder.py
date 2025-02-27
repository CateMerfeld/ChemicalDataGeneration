#%%
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.decomposition import PCA
import GPUtil
import os
import itertools
from torch.utils.data import random_split
from torch.utils.data import DataLoader, TensorDataset
import random
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import plotting_functions as pf
import functions as f

# Define the transformation to convert images to tensors and flatten to 1D
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to range [-1, 1]
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the image to 1D
])

# Load the MNIST training set
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
#%%
val_size = int(len(trainset) * 0.2)
train_size = len(trainset) - val_size

# Split the dataset into training and validation sets
train_subset, val_subset = random_split(trainset, [train_size, val_size])

# Load the MNIST test set
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)

#%%
device = f.set_up_gpu()