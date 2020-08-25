"""
Fairness Metrics
================

Parameter definitions:

    :param pred_labels:     predicted output from model as a 1D array (should be all 1's and 0's because binary output)
    :param true_labels:     the true labels as a 1D array (same size and format as pred_labels)
    :param groups:          the group of each element as a 1D array
    :param priv_groups:     the group name of privileged

"""

import numpy as np
from torch import Tensor

def pred_pos(pred_labels, true_labels, groups):
    """
    Calculates the number of predicted positive elements in a group: :math:`PP_g = (\\hat{Y}=1)`

    :return: dictionary of positive predictions for each group
    """

    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.detach().numpy()

    correct = pred_labels == true_labels
    group_correct = {k : 0 for k in np.unique(groups)}

    for i, c in enumerate(correct):
        if c:
            group_correct[groups[i]] += 1

    return group_correct

def total_pred_pos(pred_labels):
    """ 
    Calculates total predictive positives across all groups :math:`TPP=\\sum_{g_1}^{g_n} PP_g`

    :return: integer number of positive predictions in whole sample
    """
    
    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()

    return np.sum(pred_labels == 1)

def pred_neg(pred_labels, true_labels, groups):
    """
    Calculates the number of predicted negative elements in a group: :math:`PN_g = (\\hat{Y} = 0)`

    :return: dictionary of negative predictions for each group
    """
    
    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.detach().numpy()
    
    correct = pred_labels == 0
    group_correct = {k : 0 for k in np.unique(groups)}

    for i, c in enumerate(correct):
        if c:
            group_correct[groups[i]] += 1

    return group_correct

def pred_prevalence(pred_labels, true_labels, groups):
    """
    :math: :math:`\\frac{PP_g}{|g|} = P(\\hat{Y}=1 | A=g), \\forall g \\in G` 

    :return: dictionary of the fraction of positive predictions within each group
    """

    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.numpy()
    
    group_total = {k : 0 for k in np.unique(groups)}
    pp = pred_pos(pred_labels, true_labels, groups)

    for item in groups:
        group_total[item] += 1

    return {k : pp[k]/group_total[k] for k in pp}
    
    #make privelegedgroup a default parameter, just 1st one if nothing
def pred_pos_rate(pred_labels, true_labels, groups): 
    """
    Calculates predictive positive rate as :math:`PPR_g=\\frac{PP_g}{TPP}`

    :return: dictionary of the fraction positive predictions that belong to each group
    """
    
    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.detach().numpy()
    
    tpp = total_pred_pos(pred_labels)
    pp = pred_pos(pred_labels, true_labels, groups)

    return {k: v / tpp for k, v in pp.items()}

def true_pos(pred_labels, true_labels, groups):
    """
    Calculates the number of elements in a group with a positive prediction and positive true label.

    :math: :math:`\\hat{Y} = 1, Y=1`

    :return: dictionary of total true positive in each group
    """
    
    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.detach().numpy()
    
    group_correct = {k : 0 for k in np.unique(groups)}

    for i in range(len(pred_labels)):
      if pred_labels[i] == 1 and true_labels[i] == 1:
        group_correct[groups[i]] += 1
        
    return group_correct

def false_neg (pred_labels, true_labels, groups):
    """
    Calculates the number of elements in a group with a negative prediction but positive true label.

    :math: :math:`\\hat{Y} = 0, Y=1`

    :return: dictionary of total false negative predictions for each group
    """
    
    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.detach().numpy()

    group_correct = {k : 0 for k in np.unique(groups)}
    for i in range(len(pred_labels)):
      if pred_labels[i] == 0 and true_labels[i] == 1:
        group_correct[groups[i]] += 1
        
    return group_correct

def false_pos(pred_labels, true_labels, groups):
    """
    Calculates the number of elements in a group with a positive prediction but negative true label.

    :math: :math:`\\hat{Y} = 1, Y=0`

    :return: dictionary of total false positive predictions for each group
    """
    
    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.detach().numpy()

    group_correct = {k : 0 for k in np.unique(groups)}
    for i in range(len(pred_labels)):
      if pred_labels[i] == 1 and true_labels[i] == 0:
        group_correct[groups[i]] += 1
        
    return group_correct


def true_neg (pred_labels, true_labels, groups):
    """
    Calculates the number of elements in a group with a negative prediction and negative true label.

    :math: :math:`\\hat{Y} = 0, Y=0`

    :return: dictionary of total true negative predictions for each group
    """
    
    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.detach().numpy()

    group_correct = {k : 0 for k in np.unique(groups)}
    for i in range(len(pred_labels)):
      if pred_labels[i] == 0 and true_labels[i] == 0:
        group_correct[groups[i]] += 1
        
    return group_correct

def false_disc_rate(pred_labels, true_labels, groups):
    """
    :math: :math:`FDR_g = \\frac{FP_g}{PP_g} = P(Y=0 | \\hat{Y} = 1, A = g), \\forall g \\in G`

    :return: dictionary of fraction of false positives within the predicted positive of the group
    """
    
    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.detach().numpy()
    
    fp = false_pos(pred_labels, true_labels, groups)
    pp = pred_pos(pred_labels, true_labels, groups)

    return {k : (fp[k]/pp[k] if pp[k] != 0 else 0.0) for k in fp}

def false_omis_rate(pred_labels, true_labels, groups):
    """
    :math: :math:`FOR_g = \\frac{FN_g}{PN_g} = P(Y=1 | \\hat{Y} = 0, A = g), \\forall g \\in G`

    :return: dictionary of fraction of false negatives within the predicted negatives of the group
    """
    
    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.detach().numpy()

    fn = false_neg(pred_labels, true_labels, groups)
    pn = pred_neg(pred_labels, true_labels, groups)

    return {k : (fn[k]/pn[k] if pn[k] != 0 else 0.0) for k in fn}


def false_pos_rate (pred_labels, true_labels, groups):
    """
    :math: :math:`FPR_g = \\frac{FP_g}{TN_g + FP_g} = P(\\hat{Y}=1 | Y= 0, A = g), \\forall g \\in G`
    
    :return: dictionary of fraction of false positives within the labeled negatives of the group
    """
    
    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.detach().numpy()
    
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
    :math: :math:`FNR_g = \\frac{FN_g}{TP_g + FN_g} = P(\\hat{Y}=0 |  Y = 1, A = g), \\forall g \\in G`

    :return: dictionary of fraction of false negatives within the labeled positives of the group
    """
    
    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.detach().numpy()
  
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

def Demographic_Parity(pred_labels, true_labels, groups, priv_group=None):
    """
    A fair algorithm would have an equal rate of positive outcomes :math:`(\\hat{Y}=1)` in privileged group :math:`(A=0)` and unprivileged groups :math:`(A=1)`.

    :math: :math:`P(\\hat{Y} = 1|A=0) = P(\\hat{Y} = 1|A=1)` 

    :return: dictionary of predicted positive outcomes relative to the privileged group
    """
    
    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.detach().numpy()
    
    if priv_group == None: #is it better if there is no relative group
        priv_group = groups[0]

    pp = pred_pos(pred_labels, true_labels, groups)

    if pp[priv_group] != 0:
        return {k: v/pp[priv_group] for k, v in pp.items()}
    raise ZeroDivisionError("Privileged group has 0 predicted positives")

def Equality_of_Opportunity(pred_labels, true_labels, groups, priv_group=None):
    """
    A fair algorithm would have an equal rate of true positives in privileged and unprivileged groups.

    :math: :math:`P(\hat{Y}=1|A=0, Y=1) = P(\hat{Y}=1|A=1, Y=1)`

    :return: dictionary of true positives relative to the privileged group
    """
    
    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.detach().numpy()
    
    if priv_group == None:
        priv_group = groups[0]

    tp = true_pos(pred_labels, true_labels, groups)

    if tp[priv_group] != 0:
        return {k: v/tp[priv_group]  for k, v in tp.items()}
    raise ZeroDivisionError("Privileged group has 0 true positives")

def Equalized_Odds(pred_labels, true_labels, groups, priv_group=None):
    """
    A fair algorithm should have equal rates for true positives and false positives in the privileged and unprivileged groups respectively. 

    :math: :math:`P(\\hat{Y}=1|A=0,Y=y) = P(\\hat{Y} = 1 | A=1, Y=y), \\forall y \\in \\{0,1\\}`
    
    :return: dictionary of tuples consisting of true positives and false positives, relative to the privileged group
    """
    
    if isinstance(pred_labels, Tensor):
        pred_labels = pred_labels.detach().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(groups, Tensor):
        groups = groups.detach().numpy()

    if priv_group is None:        
        priv_group = groups[0]    
    tpr = Equality_of_Opportunity(pred_labels, true_labels, groups, priv_group)
    fp = false_pos(pred_labels, true_labels, groups)    
    if fp[priv_group] != 0:
        fpr = {k: (v/fp[priv_group]) for k, v in fp.items()}   
        return {k:(tp,fpr[k]) for k,tp in tpr.items()}
    raise ZeroDivisionError("Privileged group has 0 false positives")     

if __name__=="__main__":
    pred_labels = np.array([0, 1, 1, 0, 0, 1, 1, 1])
    true_labels = np.array([0, 0, 1, 0, 1, 1, 0, 0])
    groups = np.array(['a', 'b', 'c', 'd', 'd', 'd', 'c', 'b'])

    print(Equalized_Odds(pred_labels, true_labels, groups, 'c'))
    
    import torch

    pred_labels = torch.from_numpy(pred_labels)
    true_labels = torch.from_numpy(true_labels)

    print(Equalized_Odds(pred_labels, true_labels, groups, 'c'))
