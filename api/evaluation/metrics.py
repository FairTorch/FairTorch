"""
metrics.py
==========

A file for metrics.

Parameter definitions:

    :param pred_labels: predicted output from model as a 1D array (should be all 1's and 0's because binary output)
    :param true_labels: the true labels as a 1D array (same size and format as pred_labels)
    :param groups: the group of each element as a 1D array

"""

import numpy as np

def pred_pos(pred_labels, true_labels, groups):
    """
    :return: dictionary of positive predictions for each group
    """

    correct = pred_labels == true_labels
    group_correct = {k : 0 for k in np.unique(groups)}

    for i, c in enumerate(correct):
        if c:
            group_correct[groups[i]] += 1

    return group_correct

def total_pred_pos(pred_labels):
    """
    :return: integer number of positive predictions in whole sample
    """

    return np.sum(pred_labels == 1)

def pred_neg(pred_labels, true_labels, groups):
    """
    :return: dictionary of negative predictions for each group
    """
    
    correct = pred_labels == 0
    group_correct = {k : 0 for k in np.unique(groups)}

    for i, c in enumerate(correct):
        if c:
            group_correct[groups[i]] += 1

    return group_correct

def pred_prevalence(pred_labels, true_labels, groups):
    """
    :return: dictionary of the fraction of positive predictions within each group
    """

    group_total = {k : 0 for k in np.unique(groups)}
    pp = pred_pos(pred_labels, true_labels, groups)

    for item in groups:
        group_total[item] += 1

    return {k : pp[k]/group_total[k] for k in pp}
    
def pred_pos_rate(pred_labels, true_labels, groups):
    """
    :return: dictionary of the fraction positive predictions that belong to each group
    """
    
    tpp = total_pred_pos(pred_labels)
    pp = pred_pos(pred_labels, true_labels, groups)

    return {k: v / tpp for k, v in pp.items()}

def true_pos(pred_labels, true_labels, groups):
    """
    :return: dictionary of total true positive in each group
    """
    group_correct = {k : 0 for k in np.unique(groups)}

    for i in range(len(pred_labels)):
      if pred_labels[i] == 1 and true_labels[i] == 1:
        group_correct[groups[i]] += 1
        
    return group_correct

def false_neg (pred_labels, true_labels, groups):
    """
    :return: dictionary of total false negative predictions for each group
    """

    group_correct = {k : 0 for k in np.unique(groups)}
    for i in range(len(pred_labels)):
      if pred_labels[i] == 0 and true_labels[i] == 1:
        group_correct[groups[i]] += 1
        
    return group_correct

def false_pos(pred_labels, true_labels, groups):
    """
    :return: dictionary of total false positive predictions for each group
    """

    group_correct = {k : 0 for k in np.unique(groups)}
    for i in range(len(pred_labels)):
      if pred_labels[i] == 1 and true_labels[i] == 0:
        group_correct[groups[i]] += 1
        
    return group_correct


def true_neg (pred_labels, true_labels, groups):
    """
    :return: dictionary of total true negative predictions for each group
    """

    group_correct = {k : 0 for k in np.unique(groups)}
    for i in range(len(pred_labels)):
      if pred_labels[i] == 0 and true_labels[i] == 0:
        group_correct[groups[i]] += 1
        
    return group_correct

def false_disc_rate(pred_labels, true_labels, groups):
    """
    :return: dictionary of fraction of false positives within the predicted positive of the group
    """
    fp = false_pos(pred_labels, true_labels, groups)
    pp = pred_pos(pred_labels, true_labels, groups)

    return {k : fp[k]/pp[k] for k in fp}

def false_omis_rate(pred_labels, true_labels, groups):
    """
    :return: dictionary of fraction of false negatives within the predicted negatives of the group
    """

    fn = false_neg(pred_labels, true_labels, groups)
    pn = pred_neg(pred_labels, true_labels, groups)

    return {k : fp[k]/pp[k] for k in fn}


def false_pos_rate (pred_labels, true_labels, groups):
  """
  :return: dictionary of fraction of false positives within the labeled negatives of the group
  """
  unique_groups = np.unique(groups)
  labeled_neg = {k : 0 for k in unique_groups}
  group_correct = {k : 0 for k in unique_groups}
  for i in range(len(pred_labels)):
    if true_labels[i] == 0:
      labeled_neg[groups[i]] += 1
      if pred_labels[i] == 1:
        group_correct[groups[i]] += 1
  
  for i in range(len(group_correct)):
    group_correct[groups[i]] /= labeled_neg[groups[i]]  
        
  return group_correct


def false_neg_rate (pred_labels, true_labels, groups):
  """
  :return: dictionary of fraction of false negatives within the labeled positives of the group
  """
  unique_groups = np.unique(groups)
  labeled_pos = {k : 0 for k in unique_groups}
  group_correct = {k : 0 for k in unique_groups}
  for i in range(len(pred_labels)):
    if true_labels[i] == 1:
      labeled_pos[groups[i]] += 1
      if pred_labels[i] == 0:
        group_correct[groups[i]] += 1

  for i in range(len(unique_groups)):
    group_correct[unique_groups[i]] /= labeled_pos[unique_groups[i]]  
        
  return group_correct
