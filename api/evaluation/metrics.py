"""
metrics.py
==========

A file for metrics.

"""

import numpy as np

    
def total_pred_pos(pred_labels, true_labels):
    """true_labels is 1D array"""
    #pred_labels = np.argmax(classification_scores, axis=1) #finds max along each row
    return np.sum(pred_labels == true_labels)

def pred_pos(pred_labels, true_labels, classes):
    #pred_labels = np.argmax(classification_scores, axis=1) #finds max along each row
    correct = pred_labels == true_labels
    class_correct = np.zeros(classification_scores.shape[1])


    for i, c in enumerate(correct):
        print(i, ' ', c)
        if c:
            class_correct[true_labels[i]] += 1


    #for item in classes:
    #    class_correct = list(0 for i in range(10))

#            np.sum(num_correct, where = (pred_labels==1))
    #print(np.sum(num_correct, where = (pred_labels==i)))
    #print(np.sum(np.logical_and(pred_labels == 2, true_labels == 2)))
    print()
    return class_correct
        
classification_scores = np.array([[ 30,   1,  10,  80],
                                    [-10,  20,   0,  -5],  
                                    [ 27,  50,   9,  30],  
                                    [ -1,   0,  84,   3],  
                                    [  5,   2,  10,   0]])
true_labels = np.array([0, 1, 1, 2, 3])
pred_labels = np.array([3, 1, 1, 2, 2])
print(pred_pos(pred_labels, true_labels, np.arange(0,4)))

