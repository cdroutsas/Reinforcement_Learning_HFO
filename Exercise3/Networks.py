import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Define your neural networks in this class. 
# Use the __init__ method to define the architecture of the network
# and define the computations for the forward pass in the forward method.

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        input_dim = 15
        hidden_dim = 64
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.val_layer = nn.Linear(hidden_dim, 4)

    def forward(self, inputs):
        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))
        value = self.val_layer(h)
        return value
