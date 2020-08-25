"""
Monitor
=======

A wrapper class for a neural network model which plots fairness metrics with
the forward pass.

model: PyTorch Model
metrics_dict: dictionary with name and function for each metric to be graphed
groups: the groups of data as a 1D array
prive_group: the group name of privileged
batchno: number of current batch (for x-axis)
data: validation data
labels: labels for validation data
metrics: list of names of metric plots to return
plots: matplotlib pyplot with subplots corresponding to each metric
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class Monitor:
	def __init__(self, model, metrics_dict, groups): 	
		self.model = model
		self.metric_names = []
		self.metric_methods = []
		for name, method in metrics_dict.items():
		    self.metric_names.append(name)
			self.metric_methods.append(method)
		self.groups = groups
		self.plots, axs = plt.subplots(len(metrics_dict))
		for m in range(len(self.metric_names)):
			axs[m].set_xlabel('Batch')
			axs[m].set_ylabel(metric_names[m])
			for g in groups:
				axs[m].plot([],[], label = g)

	def update(batchno, data, labels):
    	scores = self.model(data)
		axs = self.plots.get_axes()
		for m in range(len(self.metric_names)):
			lines = axs[m].get_lines()
			for g in range(len(groups)):
    			lines[g].set_xdata(np.append(lines[g].get_xdata(), batchno))
				lines[g].set_ydata(np.append(lines[g].get_ydata(), \
					self.metric_methods[m](scores, labels, self.groups))
			axs[m].relim()
			axs[m].autoscale_view()
			axs[m].legend()
		plt.draw()
		plots.show()
	
	def get_plot():
		return self.plots
