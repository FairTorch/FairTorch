"""
Monitor
=======

A wrapper class for a neural network model which plots fairness metrics with
the forward pass.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class Monitor:
    def __init__(self, model, metrics_dict, groups): 	
        """
        Attributes:

        :param model:	the PyTorch Model Monitor is being used with
        :param metric_names:	list of the names of the metrics used
        :param metric_methods:	list of functions to evaluate metrics in the same order as metric_names
        :param groups:	list of the groups of data 
        :param fig:	Matplotlib Pyplot figure with subplots of the value of each metric over training steps
        """
        self.model = model
        self.metric_names = []
        self.metric_methods = []
        for name, method in metrics_dict.items():
            self.metric_names.append(name)
            self.metric_methods.append(method)
        self.groups = groups
        self.fig, axs = plt.subplots(len(metrics_dict))
        for m in range(len(self.metric_names)):
            axs[m].set_xlabel('Training Step')
            axs[m].set_ylabel(self.metric_names[m])
            for g in groups:
                axs[m].plot([],[], label = g)

    def update_fig(self, step_no, data, data_groups, labels):
        """
        Update figure at the end of a foward pass with values of metrics for each group and display the figure
        Assumes model outputs logits to be fed into softmax activation.

        Parameters:

            :param step_no: number of current training step (e.g. mini-batch number, batch number, epoch number)
            :param data: validation data as list of inputs to model
            :param data_groups: the group of each element in data as a 1D array
            :param labels: labels for each element in data as a 1D array
        """
        
        scores = self.model(data)
        axs = self.fig.get_axes()
        softmax = nn.Softmax(dim=1)
        for m in range(len(self.metric_names)):
            lines = axs[m].get_lines()
            new_ydata = self.metric_methods[m](torch.argmax(softmax(scores), axis=1), labels, data_groups)
            for g in range(len(self.groups)):
                if np.sum(self.groups[g] == data_groups) > 0:
                    lines[g].set_xdata(np.append(lines[g].get_xdata(), step_no))
                    lines[g].set_ydata(np.append(lines[g].get_ydata(), new_ydata[self.groups[g]]))
            axs[m].relim()
            axs[m].autoscale_view()
            axs[m].legend()
        plt.draw()
        self.fig.show()
        plt.pause(0.001)
    
    def get_fig():
        """
        Get fig attribute
        
        :return: Matplotlib Pyplot figure with subplots corresponding to each metric
        """
        return self.fig
