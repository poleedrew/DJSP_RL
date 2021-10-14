import torch
import torch.nn as nn
import torch.nn.functional as F 

class ValueNetwork(nn.Module):
    def __init__(self, state_space):
        super(ValueNetwork, self).__init__()
        self.input_size = state_space
        self.network = nn.Sequential(
            nn.Linear(self.input_size, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
    
    def forward(self, state):
        x = self.network(state)
        return x