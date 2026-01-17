import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import os
import json
import copy


class mlp(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim, layers, activation, use_norm=False, dropout=0.0):
        """
        Create an mlp from the configurations.
        Args:
            dim: Input dimension
            hidden_dim: Width of hidden layers
            output_dim: Output dimension
            layers: Number of hidden layers
            activation: Activation string (e.g. 'leaky_relu')
            use_norm: Boolean, enable LayerNorm
            dropout: Float, dropout probability
        """
        super().__init__()
        
        activation_fn = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'silu': nn.SiLU,
            'gelu': nn.GELU, 
        }[activation]
    
        # Initialize the layers list
        seq = []
    
        # --- 1. Input Layer ---
        seq.append(nn.Linear(dim, hidden_dim))
        if use_norm:
            seq.append(nn.LayerNorm(hidden_dim))
        seq.append(activation_fn())
        if dropout > 0:
            seq.append(nn.Dropout(dropout))
            
        # Init Input Layer
        nn.init.xavier_uniform_(seq[0].weight)

        # --- 2. Hidden Layers ---
        for _ in range(layers):
            seq.append(nn.Linear(hidden_dim, hidden_dim))
            if use_norm:
                seq.append(nn.LayerNorm(hidden_dim))
            seq.append(activation_fn())
            if dropout > 0:
                seq.append(nn.Dropout(dropout))
            
            # Init Hidden Layer (The Linear is the last added Linear)
            # We iterate backwards to find the last Linear
            for layer in reversed(seq):
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    break

        # Connect base
        self.base_network = nn.Sequential(*seq)
    
        # --- 3. Output Layer ---
        self.out = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.out.weight)
    
    def forward(self, x):
        # Auto-flatten for convenience (Handling >2D inputs like images)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        x = self.base_network(x)
        out = self.out(x)
        return out    
    

# Teacher model
class teacher(nn.Module):
    def __init__(self, dz, output_dim):
        super().__init__()  # FIX: Modern super syntax
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