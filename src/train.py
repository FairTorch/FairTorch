import torch
import torch.nn as nn

class Monitor(nn.Module):
    def __init__(self, model):
        super(Monitor, self).__init__()
        self.model = model

    def forward(self, x):
        """
        Forward pass of model
        """
        return self.model(x)
