"""
metrics.py
==========

A file for metrics.

Parameter definitions:

    :param pred_labels: predicted output from model as a 1D array
    :param true_labels: the true labels as a 1D array (same size as pred_labels)
    :param classes: the groups to classify an object as a 1D array of integers

"""

import numpy as np

def pred_pos(pred_labels, true_labels, classes):
    """
    :return: integer array of total positive predictions for each group
    """

    class_correct = np.zeros(len(classes))

    for i, c in enumerate(pred_labels):
        class_correct[c] += 1

    return class_correct

def pred_neg(pred_labels, true_labels, classes):
    """
    :return: integer array of total negative predictions in each group
    """
    
    b = pred_pos(pred_labels, true_labels, classes)
    a = np.empty(len(classes))
    a.fill(len(pred_labels))    
    return a-b #don't know what the purpose of this metric is

def pred_prevalence(pred_labels, true_labels, classes):
    return pred_pos(pred_labels, true_labels, classes)

def pred_pos_rate(pred_labels, true_labels, classes):
    #this is the same as ppr since we are not having input as binary correct....
    
def total_pred_pos(pred_labels):
    """
    :return: integer of total correct predictions
    """
    #pred_labels = np.argmax(classification_scores, axis=1) #finds max along each row
    #return np.sum(pred_labels == true_labels) <-- seems like useful metric, but not this specific one

    return len(pred_labels) #this seems so wrong
    

def true_pos(pred_labels, true_labels, classes):
    """
    :return: integer array of total correct predictions in each group
    """
    correct = pred_labels == true_labels
    class_correct = np.zeros(len(classes))

    for i, c in enumerate(correct):
        if c:
            class_correct[true_labels[i]] += 1
    
    return class_correct

        
classification_scores = np.array([[ 30,   1,  10,  80],
                                    [-10,  20,   0,  -5],  
                                    [ 27,  50,   9,  30],  
                                    [ -1,   0,  84,   3],  
                                    [  5,   2,  10,   0]])
true_labels = np.array([0, 1, 1, 2, 3])
pred_labels = np.array([3, 1, 1, 2, 2])
print(pred_neg(pred_labels, true_labels, np.arange(0,4)))

