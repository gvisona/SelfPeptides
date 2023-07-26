import math
import numpy as np
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
import math
from itertools import islice, zip_longest
import os
from os.path import exists, join
import torch

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def lr_schedule(t, min_frac=0.001, total_iters=100, delay=0.0, ramp_up=0.3, cool_down=0.5):
    assert isinstance(total_iters, int)
    if isinstance(ramp_up, float):
        ramp_up_steps = total_iters*ramp_up
    elif isinstance(ramp_up, int):
        assert ramp_up<total_iters
        ramp_up_steps = ramp_up
    
    if isinstance(delay, float):
        delay_steps = total_iters*delay
    elif isinstance(delay, int):
        assert delay<total_iters
        delay_steps = delay
    
    
    if isinstance(cool_down, float):
        cooldown_steps = total_iters*cool_down
    elif isinstance(cool_down, int):
        assert cool_down<total_iters
        cooldown_steps = cool_down
    assert cooldown_steps + ramp_up_steps + delay_steps <= total_iters
        
    
    if t<=delay_steps:
        return 0
    elif t<=delay_steps+ramp_up_steps:
        return min_frac + (t-delay_steps)*(1-min_frac)/ramp_up_steps
    
    elif t<=delay_steps+ramp_up_steps+cooldown_steps:
        return min_frac+0.5*(1-min_frac)*(1+np.cos(((t-ramp_up_steps-delay_steps)/cooldown_steps)*math.pi))
    return min_frac




def eval_classification_metrics(targets, predictions, is_logit=False, threshold=0.5):
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)   
        
    if is_logit:
        predictions = sigmoid(predictions)
    predicted_classes = (predictions>threshold).astype(int)
        
    tn, fp, fn, tp = confusion_matrix(targets, predicted_classes).ravel()
    metrics = {
        "MCC": matthews_corrcoef(targets, predicted_classes), 
        "FPR": fp / (fp + tn) if (fp + tn)>0 else 0.0,
        "FNR": fn / (tp + fn) if (fp + tn)>0 else 0.0, 
        "Specificity": tn / (tn + fp) if (tn + fp)>0 else 0.0,
        "NPV": tn/ (tn + fn) if (tn + fn)>0 else 0.0,
        "FDR": fp/ (tp + fp) if (tp + fp)>0 else 0.0,
        "Precision": precision_score(targets, predicted_classes, zero_division=0.0), 
        "Recall": recall_score(targets, predicted_classes, zero_division=0.0), 
        "F1": f1_score(targets, predicted_classes, zero_division=0.0), 
        "BalancedAccuracy": balanced_accuracy_score(targets, predicted_classes), 
        "AUROC": roc_auc_score(targets, predictions), 
        "AUPRC": average_precision_score(targets, predictions)
    }
    
    return metrics


def cosine_similarity_all_pairs(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt