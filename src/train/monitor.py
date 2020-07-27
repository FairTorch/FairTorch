"""
Monitor
=======

A wrapper class for a neural network model which computes fairness metrics with
the forward pass.

"""
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

