"""
metrics.py
==========

A file for metrics.

Parameter definitions:

    :param pred_labels: predicted output from model as a 1D array
    :param true_labels: the true labels as a 1D array (same size as pred_labels)
    :param classes: the group of each element

"""

import numpy as np

def pred_pos(pred_labels, true_labels, classes):
    correct = pred_labels == true_labels
    class_correct = np.zeros(np.unique(classes));

    for i, c in enumerate(correct):
        if c:
            class_correct[true_labels[i]] += 1

    return class_correct

def total_pred_pos(pred_labels, true_labels):
    """true_labels is 1D array"""
    return np.sum(pred_labels == true_labels)

def pred_neg(pred_labels, true_labels, classes):
    correct = pred_labels != true_labels
    class_correct = np.zeros(np.unique(classes));

    for i, c in enumerate(correct):
        if c:
            class_correct[true_labels[i]] += 1

    return class_correct

def pred_prevalence(pred_labels, true_labels, classes):
    class_total = np.zeros(np.unique(classes));

    for i, c in enumerate(classes):
        class_total[classes[i]] += 1 #only works if classes is numbers

    return pred_pos(pred_labels, true_labels, classes) / class_total(pred_labels);

def pred_pos_rate(pred_labels, true_labels, classes):
    return pred_pos(pred_labels, true_labels, classes) / total_pred_pos;

        
classification_scores = np.array([[ 30,   1,  10,  80],
                                    [-10,  20,   0,  -5],  
                                    [ 27,  50,   9,  30],  
                                    [ -1,   0,  84,   3],  
                                    [  5,   2,  10,   0]])
true_labels = np.array([0, 1, 1, 2, 3])
pred_labels = np.array([3, 1, 1, 2, 2])
print(pred_neg(pred_labels, true_labels, np.arange(0,4)))

