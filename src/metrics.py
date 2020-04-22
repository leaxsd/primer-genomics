###############################################################
from __future__ import print_function, division
import numpy as np
###############################################################

def num_score(pred, true):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    FP = np.float(np.sum((pred == 1) & (true == 0)))
    FN = np.float(np.sum((pred == 0) & (true == 1)))
    TP = np.float(np.sum((pred == 1) & (true == 1)))
    TN = np.float(np.sum((pred == 0) & (true == 0)))

    return FP, FN, TP, TN


def accuracy_score(pred, true):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = num_score(pred, true)
    N = TP + TN
    D = FP + FN + TP + TN
    accuracy = np.divide(N, D, out=np.zeros_like(N), where=N!=0)
    return accuracy.item()

def precision_score(pred, true):
    """Getting the precison of the model"""

    FP, FN, TP, TN = num_score(pred, true)
    N = TP
    D = TP + FP
    precision = np.divide(N, D, out=np.zeros_like(N), where=N!=0)
    return precision.item()

def recall_score(pred, true):
    """Getting the recall of the model"""

    FP, FN, TP, TN = num_score(pred, true)
    N = TP
    D = TP + FN
    recall = np.divide(N, D, out=np.zeros_like(N), where=N!=0)
    return recall.item()
