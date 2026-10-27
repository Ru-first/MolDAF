import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

def compute_auc(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    prcs, rocs = [], []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0:
            try:
                prcs.append(average_precision_score(y_true[:, i], y_scores[:, i]))
            except:
                prcs.append(0)
            try:
                rocs.append(roc_auc_score(y_true[:, i], y_scores[:, i]))
            except:
                rocs.append(0)
    return np.mean(prcs), np.mean(rocs)

