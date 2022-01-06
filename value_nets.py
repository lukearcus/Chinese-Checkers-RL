import torch
from torch import nn
import numpy as np

class NeuralNetwork_v1(nn.Module):
    def __init__(self):
        super(NeuralNetwork_v1, self).__init__()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(17*27, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 6),
                nn.Tanh()
            )
    
    def forward(self, x):
        x = torch.from_numpy(x).float().to('cuda')
        x = x.flatten()
        vals = self.linear_relu_stack(x)
        return vals

