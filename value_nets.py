import torch
from torch import nn


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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def forward(self, x):
        vals = self.linear_relu_stack(x)
        return vals

