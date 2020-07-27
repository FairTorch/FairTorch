"""
Monitor
=======

A wrapper class for a neural network model which computes fairness metrics with
the forward pass.

"""
import torch
import torch.nn as nn


class Monitor:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Forward pass of model
        """
        return self.model(x)

