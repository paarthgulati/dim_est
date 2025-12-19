import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import os
import json
# import matplotlib.colors as mc



class mlp(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim, layers, activation):
        """Create an mlp from the configurations."""
        super(mlp, self).__init__()
        activation_fn = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'silu': nn.SiLU,
        }[activation]
    
        # Initialize the layers list
        seq = []
    
        # Input layer
        seq.append(nn.Linear(dim, hidden_dim))
        seq.append(activation_fn())
        nn.init.xavier_uniform_(seq[0].weight)  # Xavier initialization for input layer
    
        # Hidden layers
        for _ in range(layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for hidden layers
            seq.append(layer)
            seq.append(activation_fn())
    
        # Connect all together before the output
        self.base_network = nn.Sequential(*seq)
    
        # Output layer
        self.out = nn.Linear(hidden_dim, output_dim)
        
        # Initialize the layer with Xavier initialization
        nn.init.xavier_uniform_(self.out.weight)
    
    def forward(self, x):
        x = self.base_network(x)
        
        # Get output
        out = self.out(x)
        
        return out
    
# Teacher model
class teacher(nn.Module):
    def __init__(self, dz, output_dim):
        super(teacher, self).__init__()
        self.dense1 = nn.Linear(dz, 1024)
        self.act = nn.Softplus()
        self.dense2 = nn.Linear(1024, output_dim)

    def forward(self, Z):
        x = self.act(self.dense1(Z))
        x = self.dense2(x)
        return x

# Dataset loader
class Dataset(torch.utils.data.Dataset):
  def __init__(self, X, Y):
        self.X = X
        self.Y = Y

  def __len__(self):
        return len(self.X)

  def __getitem__(self, index):
        return self.X[index], self.Y[index]

