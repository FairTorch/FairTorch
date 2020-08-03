"""
map
===

Module for creating fair feature maps.
"""

import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

DEBUG = True

class FairMap(Module):
    """
    Learns a fair feature map for a feature vector.

    * :attr:`input_size` controls the expected vector input dimension.
    * :attr:`protected_dim` controls the expected dimension of the protected feature.

    Examples:
    
    .. code-block:: python
        
        X = torch.randn(20, 11)
        fm = FairMap(10, 2, 2, 15)
        fm.fit(X[:, :10], X[:, 10])

        X = fm.get_fmap(X[:, :10])
    """
    def __init__(self, input_size, protected_dim, n_hidden, layer_width, 
                activation=F.relu, inverse_loss=nn.MSELoss, adversary_loss=nn.CrossEntropyLoss,
                eta=0.5, optimizer=optim.SGD, lr=0.01):
        
        super(FairMap, self).__init__()

        self.input_size = input_size
        self.protected_dim = protected_dim

        # use same activation for feature map, adversary, and inverse function
        self.activation = activation

        if not adversary_loss is nn.CrossEntropyLoss:
            raise ValueError(f"Invalid adversary loss {adversary_loss}")
        
        self.inverse_loss = inverse_loss()
        self.adversary_loss = adversary_loss()

        self.eta = eta

        self.optimizer = optimizer
        self.lr = lr

        # layers for the feature map
        self.feature_map_layers = nn.ModuleList(
            [nn.Linear(input_size, layer_width)] + \
            [nn.Linear(layer_width, layer_width) for i in range(n_hidden)] + \
            [nn.Linear(layer_width, input_size)]
        )

        # layers for adversary
        self.adversary_layers = nn.ModuleList(
            [nn.Linear(input_size, layer_width)] + \
            [nn.Linear(layer_width, layer_width) for i in range(n_hidden)] + \
            [nn.Linear(layer_width, protected_dim)]
        )

        # layers for inverse function
        self.inverse_layers = nn.ModuleList(
            [nn.Linear(input_size, layer_width)] + \
            [nn.Linear(layer_width, layer_width) for i in range(n_hidden)] + \
            [nn.Linear(layer_width, input_size)]
        )


    def forward(self, x):
        # make the feature map
        for layer in self.feature_map_layers[:-1]:
            x = layer(x)
            x = self.activation(x)

        feature_map = self.feature_map_layers[-1](x)
        x = self.activation(x)

        # compute the adversary
        for layer in self.adversary_layers[:-1]:
            x = layer(x)
            x = self.activation(x)

        adversary_prediction = self.adversary_layers[-1](x)

        # compute the inverse of the feature map
        x = feature_map
        x = self.activation(x)
        for layer in self.inverse_layers[:-1]:
            x = layer(x)
            x = self.activation(x)

        inverse_feature_map = self.inverse_layers[-1](x)

        return feature_map, adversary_prediction, inverse_feature_map


    def fit(self, x, protected, steps=100):
        original_x = x

        optimizer = self.optimizer(self.parameters(), self.lr)
        
        for i in range(steps):
            ####### make inverse perform better and adversary perform worse #######
            
            self.zero_grad()
            feature_map, adversary_prediction, inverse_feature_map = self.forward(x)

            loss1 = self.inverse_loss(inverse_feature_map, x) - \
                self.eta * self.adversary_loss(adversary_prediction, protected)
            
            loss1.backward()
            optimizer.step()

            ##################### Make adversary perform better ###################
            
            self.zero_grad()
            feature_map, adversary_prediction, inverse_feature_map = self.forward(x)
            
            loss2 = self.adversary_loss(adversary_prediction, protected)
            loss2.backward()
            optimizer.step()
            
            if DEBUG and i%10==0:
                print(f"Loss1: {loss1} Loss 2: {loss2}")

        print(torch.sum(torch.abs(inverse_feature_map - original_x)))
        print(torch.sum(torch.abs(original_x - feature_map)))




