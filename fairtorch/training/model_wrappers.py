"""
Adversarial Training
====================

Model wrapper classes for improving and monitoring fairness during training using adversaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# import individually for documentation purposes
from torch.nn import Module
from torch.nn.functional import relu
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam


class FairModel(Module):
    """
    Wrapper class for models for adversarial training to increase fairness.

    * :attr:`model` is a pretrained model for which the fairness must increase.
    * :attr:`input_size` controls the expected vector input dimension.
    * :attr:`n_groups` is the number of unique values of the sensitive feature.
    * :attr:`n_hidden` is the number of hidden layers in the adversarial network.
    * :attr:`layer_width` is the number of units in each hidden layer.
    * :attr:`activation` is the activation function used for the hidden layers of the adversarial network.

    Examples:

    .. code-block:: python

        # input, output, and protected/sensitive feature (e.g. race, gender, etc.)
        
        X = torch.randn(20, 3)
        X[:10] = 10 * X[:10]
        y = torch.sum(torch.Tensor([[1.0, 0.5, 3.14]]) * X, axis=1, keepdims=True) + 1.41
        
        protected = torch.zeros(20, dtype=torch.long)
        protected[:10] = 0
        protected[10:] = 1

        # define and pretrain model

        model = Model()
        model.fit(X, y)

        # wrap pretrained model in FairModel adversary and train
        
        fm = FairModel(model, output_size=1, n_groups=2, n_hidden=1, layer_width=10)
        fm.fit(X, y, protected, 0.5, steps=1000)

        model_predictions, adversary_predictions = fm(X)
        m = nn.Softmax(dim=1)
        adversary_predictions = m(adversary_predictions)

    """

    def __init__(self, model, output_size, n_groups, n_hidden, layer_width, activation=relu):
        
        super(FairModel, self).__init__()

        # a neural network model for the adversary

        class Net(Module):
            def __init__(self, input_size, output_size, n_hidden, layer_width, activation=relu):
                super(Net, self).__init__()

                # use same activation for feature map, adversary, and inverse function
                self.activation = activation
                
                # layers for the feature map
                self.layers = nn.ModuleList(
                    [nn.Linear(input_size, layer_width)] + \
                    [nn.Linear(layer_width, layer_width) for i in range(n_hidden)] + \
                    [nn.Linear(layer_width, output_size)]
                )

            def forward(self, x):
                """
                Computes a forward pass of the model evaluated on x.
                Returns logits
                """
                for layer in self.layers[:-1]:
                    x = layer(x)
                    x = self.activation(x)

                return self.layers[-1](x)

        self.model = model
        self.adversary_net = Net(output_size, n_groups, n_hidden, layer_width, activation)

    def forward(self, x):
        """
        Forward pass of x through the model and adversarial network.

        * :attr:`x` is the input data with shape suitable for ``model``

        :returns: ``model_prediction, adversary_prediction``

        * ``model_prediction`` with shape ``model(x).shape``
        * ``adversary_prediction`` as logits
        """
        model_prediction = self.model(x)
        adversary_prediction = self.adversary_net(model_prediction)

        return model_prediction, adversary_prediction


    def fit(self, x, y, groups, eta, model_loss=MSELoss, adversary_loss=CrossEntropyLoss, 
            optimizer=Adam, steps=100, lr=0.001, verbose=True, grapher=None):
        """
        Pre-trains the adversarial network then simultaneously trains the wrapped model and the
        adversarial network.
        
        * :attr:`x` is the input data
        * :attr:`y` is the true label
        * :attr:`groups` is the group/protected/sensitive attribute for each input sample
        * :attr:`eta` is a weighting constant for adversarial training
        * :attr:`model_loss` is the loss function used for the non-adversarial model
        * :attr:`adversary_loss` is the loss function used for the adversarial model
        * :attr:`optimizer` is the uninitialized optimizer class
        * :attr:`steps` is the number of steps used for pre-training the adversary and for the model/adversary simultaneous training
        * :attr:`lr` is the learning rate for the training
        * :attr:`verbose` `True` if output during training desired, `False` otherwise
        * :attr:`grapher` is a function for graphing if `verbose` is `True`
        """
        
        ####### Pre-train the adversarial network #######
        self.model.eval()
        criterion = adversary_loss()
        opt = optimizer(self.adversary_net.parameters(), lr)

        for step in range(steps):
            self.adversary_net.zero_grad()
            
            y_ = self.model(x)
            y_ = self.adversary_net(y_)
            
            loss = criterion(y_, groups)
            loss.backward()
            opt.step()

            if verbose and step%100 == 0:
                print(f"Step {step}, Adversary loss: {loss}")
                if grapher is not None:
                    grapher()

        ####### Train the model and the adversarial network together #######
        self.model.train()
        model_criterion = model_loss()
        adversary_criterion = adversary_loss()
        opt = optimizer(self.parameters(), lr)

        for step in range(steps):
            ####### Optimize model, make adversary worse #######
            self.zero_grad()

            model_prediction, adversary_prediction = self.forward(x)
            
            loss1a = model_criterion(model_prediction, y)
            loss1b = eta * adversary_criterion(adversary_prediction, groups)
            loss1 = loss1a - loss1b

            loss1.backward()
            opt.step()
            
            ####### Make adversary better #######
            self.zero_grad()

            model_prediction, adversary_prediction = self.forward(x)
            
            loss2 = adversary_criterion(adversary_prediction, groups)

            loss2.backward()
            opt.step()

            if verbose and step%100 == 0:
                print(f"Step {step}, Model loss: {loss1a}, Adversary Loss: {loss2}")
                if grapher is not None:
                    grapher()

