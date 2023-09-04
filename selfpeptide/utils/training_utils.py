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
import torch.nn as nn
from scipy.stats import beta
from selfpeptide.utils.function_utils import get_alpha, get_beta, beta_distr_mean

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
        
    
    if t<delay_steps:
        return 0
    elif t<=delay_steps+ramp_up_steps:
        return min_frac + (t-delay_steps)*(1-min_frac)/ramp_up_steps
    
    elif t<=delay_steps+ramp_up_steps+cooldown_steps:
        return min_frac+0.5*(1-min_frac)*(1+np.cos(((t-ramp_up_steps-delay_steps)/cooldown_steps)*math.pi))
    return min_frac



def warmup_constant_lr_schedule(t, min_frac=0.001, total_iters=100, delay=0.0, ramp_up=0.3):
    # print("LR SCHEDULER ARGUMENT: {}".format(t))
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
            
    
    if t<delay_steps:
        return 0
    elif t<=delay_steps+ramp_up_steps:
        return min_frac + (t-delay_steps)*(1-min_frac)/ramp_up_steps
    return 1.0




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

def eval_regression_metrics(fractional_targets, predictions, target_vars=None, pred_vars=None):
    metrics = {}
    metrics["BetaMean-R"] = np.corrcoef(predictions, fractional_targets)[0,1]
    metrics["BetaMean-RMSE"] = np.sqrt(np.mean(np.square(predictions - fractional_targets)))
    
    if target_vars is not None and pred_vars is not None:
        pred_alphas = get_alpha(predictions, pred_vars)
        pred_betas = get_beta(predictions, pred_alphas)
        target_alphas = get_alpha(fractional_targets, target_vars)
        target_betas = get_beta(fractional_targets, target_alphas)
        
        kl = torch.mean(beta_kl_divergence(pred_alphas, pred_betas, target_alphas, target_betas)).item()
        metrics["BetaKL"] = kl

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



def get_effective_class_samples(n_samples_class, beta=0.99):
    return (1-beta**n_samples_class)/(1-beta)

def get_class_weights(df, target_label="Target"):
    pos_class_samples = df[target_label].sum()
    neg_class_samples = len(df) - pos_class_samples
    
    n_samples = len(df)
    rebalancing_beta = (n_samples-1)/n_samples
    effective_pos_samples = get_effective_class_samples(pos_class_samples, rebalancing_beta)
    effective_neg_samples = get_effective_class_samples(neg_class_samples, rebalancing_beta)
    
    pos_weight = n_samples/effective_pos_samples
    neg_weight = n_samples/effective_neg_samples
    
    return pos_weight, neg_weight




class CustomCosineDistanceHingeLoss(nn.Module):
    def __init__(self, margin=0.8, device="cpu", reg_weight=1e-5):
        super().__init__()
        self.margin = margin
        self.device = device
        self.hinge_loss = nn.HingeEmbeddingLoss(margin=margin)
        
        self.reg_weight = reg_weight
    
    def forward(self, embeddings, labels):
        pos_ix = (labels==1)
        neg_ix = (labels==-1)
        
        pos_embeddings = embeddings[pos_ix]
        neg_embeddings = embeddings[neg_ix]        
        
        # Similarities Pos-Pos
        pos_distance = 1 - cosine_similarity_all_pairs(pos_embeddings, pos_embeddings)
        ixs = torch.triu_indices(*pos_distance.shape, offset=1)
        pos_cos_distances = pos_distance[ixs[0], ixs[1]]
        
        
        # Similarities Pos-Neg
        neg_distance = 1 - cosine_similarity_all_pairs(pos_embeddings, neg_embeddings)
        ixs = torch.triu_indices(*neg_distance.shape, offset=0)
        neg_cos_distances = neg_distance[ixs[0], ixs[1]]
        
        cos_distances = torch.cat([neg_cos_distances, pos_cos_distances])
        hinge_labels = torch.ones(len(neg_cos_distances)+len(pos_cos_distances), device=self.device)
        hinge_labels[:len(neg_cos_distances)] = -1
        hinge_loss = self.hinge_loss(cos_distances, hinge_labels)
        
        regularization = self.reg_weight * torch.mean(embeddings.norm(dim=1))
        
        loss = hinge_loss + regularization
        logs = {"loss": loss, "hinge_loss": hinge_loss, "regularization": regularization}
        return loss, logs
    
    
    

def hypershperical_cosine_margin_similarity(emb1, emb2, s=1.0, m=0.5):
    # Embs must be normalized
    # emb1 = emb1 / emb1.norm(dim=1)[:, None]
    # emb2 = emb2 / emb2.norm(dim=1)[:, None]    
    c = torch.mm(emb1, emb2.transpose(1, 0))
    c -= m
    return torch.exp(s*c)


class CustomCMT_Loss(nn.Module):
    def __init__(self, s=1.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        
        
    def forward(self, embeddings, labels):
        emb_norm = embeddings.norm(dim=1)
        l2_weight = 0.01
        
        embeddings = embeddings / embeddings.norm(dim=1)[:, None]
        
        ix = (labels==1)
        pos_embs = embeddings[ix]
        neg_embs = embeddings[~ix]
        
        pos_sims = hypershperical_cosine_margin_similarity(pos_embs, pos_embs, s=self.s, m=self.m)
        # pos_sims -= (math.e * torch.eye(len(pos_sims), device=pos_sims.device))
        neg_sims = hypershperical_cosine_margin_similarity(pos_embs, neg_embs, s=self.s, m=0.0)       
        
        easy_pos_sims, _ = torch.max(pos_sims - (math.e * torch.eye(len(pos_sims), device=pos_sims.device)), dim=1)
        hard_neg_sims, _ = torch.max(neg_sims, dim=1)
        
        cmt_loss = torch.mean(-1* torch.log(easy_pos_sims/(easy_pos_sims+hard_neg_sims)))
        
        hypersphere_reg = torch.mean(torch.square(emb_norm-self.s))
        loss = cmt_loss #+ l2_weight*hypersphere_reg
        
        logs = {"loss": loss.item(), "cmt_loss": cmt_loss.item(), 'hypersphere_reg': hypersphere_reg}
        return loss, logs
    
    
class CustomCMT_AllTriplets_Loss(nn.Module):
    def __init__(self, s=1.0, m=0.5, reg_weight=1e-5):
        super().__init__()
        self.s = s
        self.m = m
        self.reg_weight = reg_weight
        
    def forward(self, embeddings, labels):
        emb_norm = embeddings.norm(dim=1)
        
        embeddings = embeddings / embeddings.norm(dim=1)[:, None]
        
        ix = (labels==1)
        pos_embs = embeddings[ix]
        neg_embs = embeddings[~ix]
        
        pos_sims = hypershperical_cosine_margin_similarity(pos_embs, pos_embs, s=self.s, m=self.m)
        neg_sims = hypershperical_cosine_margin_similarity(pos_embs, neg_embs, s=self.s, m=0.0)       
        
        p_ixs = torch.triu_indices(*pos_sims.shape, offset=1)
        pos_cos_sims = pos_sims[p_ixs[0], p_ixs[1]]
        
        n_ixs = torch.triu_indices(*neg_sims.shape, offset=0)
        neg_cos_sims = neg_sims[n_ixs[0], n_ixs[1]]
        
        
        
        cmt_loss = -1* torch.log(pos_cos_sims/(pos_cos_sims+neg_cos_sims.view(-1,1)))
        cmt_loss = torch.mean(cmt_loss)
        
        l2_reg = torch.mean(torch.square(emb_norm))#-self.s))
        loss = cmt_loss + self.reg_weight*l2_reg
        
        logs = {"loss": loss.item(), "cmt_loss": cmt_loss.item(), 'l2_reg': l2_reg.item()}
        return loss, logs
    
    
    
def get_class_weights(df, target_label="Target"):
    pos_class_samples = df[target_label].sum()
    neg_class_samples = len(df) - pos_class_samples
    
    n_samples = len(df)
    rebalancing_beta = (n_samples-1)/n_samples
    effective_pos_samples = get_effective_class_samples(pos_class_samples, rebalancing_beta)
    effective_neg_samples = get_effective_class_samples(neg_class_samples, rebalancing_beta)
    
    pos_weight = n_samples/effective_pos_samples
    neg_weight = n_samples/effective_neg_samples
    
    return pos_weight, neg_weight


def find_optimal_sf_quantile(df, class_threshold=0.5, target_metric="F1"):
    best_metric = 0
    best_threshold = 0.0
    for q_threshold in np.linspace(0.01, 0.51, 51):
        predicted_icdf = beta.sf(q_threshold, df["Alpha"], df["Beta"])

        metrics = eval_classification_metrics(df["Target"], predicted_icdf, is_logit=False, threshold=class_threshold)
        if metrics[target_metric]>best_metric:
            best_metric = metrics[target_metric]
            best_threshold = q_threshold
    return best_threshold




def beta_kl_divergence(alpha1, beta1, alpha2, beta2):
    if isinstance(alpha1, (list, np.ndarray)):
        alpha1 = torch.tensor(alpha1)
    if isinstance(beta1, (list, np.ndarray)):
        beta1 = torch.tensor(beta1)
    if isinstance(alpha2, (list, np.ndarray)):
        alpha2 = torch.tensor(alpha2)
    if isinstance(beta2, (list, np.ndarray)):
        beta2 = torch.tensor(beta2)
    
    
    assert (alpha1<0).sum()==0, "Invalid alpha1"
    assert (beta1<0).sum()==0, "Invalid beta1"
    assert (alpha2<0).sum()==0, "Invalid alpha2"
    assert (beta2<0).sum()==0, "Invalid beta2"
    
    kl_div = (torch.lgamma(alpha1+beta1) - torch.lgamma(alpha2+beta2)
              - (torch.lgamma(alpha1) + torch.lgamma(beta1))
              + (torch.lgamma(alpha2) + torch.lgamma(beta2))
              + (alpha1 - alpha2) * (torch.digamma(alpha1) - torch.digamma(alpha1+beta1))
              + (beta1 - beta2) * (torch.digamma(beta1) - torch.digamma(alpha1+beta1))
             )
    return kl_div


class CustomKL_Loss(nn.Module):
    def __init__(self, config, class_weights=[1.0, 1.0], device="cpu"):
        super().__init__()

        self.regression_loss = config.get("regression_loss", "none")
        self.regression_weight = config.get("regression_weight", 1.0)
        
        self.kl_weight = config.get("kl_weight", 1.0)
        self.kl_loss_type = config.get("kl_loss_type", "forward")
        self.class_weights = torch.tensor(class_weights).to(device)
        self.device = device
        

    def forward(self, predictions, alphas, betas, class_targets):
        loss = 0
        logs = {}
        
        means = predictions[:,0]
        variances = predictions[:, 1]
        
        pred_alphas = get_alpha(means, variances)
        pred_betas = get_beta(means, pred_alphas)
        
        obs_means = beta_distr_mean(alphas, betas)
        weights = torch.gather(self.class_weights, 0, class_targets.long())
        
        if self.regression_loss=="L1":
            reg_loss = torch.abs(obs_means - means)
            reg_loss = torch.mean(reg_loss * weights)
        elif self.regression_loss=="MSE":
            reg_loss = torch.square(obs_means - means)
            reg_loss = torch.mean(reg_loss * weights)

             
        # Beta KL divergence
        if self.kl_weight>0.0:
            kl_loss = 0
            if self.kl_loss_type == "forward" or self.kl_loss_type=="both":
                kl_loss += beta_kl_divergence(pred_alphas, pred_betas, alphas, betas)
            if self.kl_loss_type == "backward" or self.kl_loss_type=="both":
                kl_loss += beta_kl_divergence(alphas, betas, pred_alphas, pred_betas)
            if self.kl_loss_type=="both":
                kl_loss /= 2
            kl_loss = torch.mean(kl_loss * weights)
            loss += self.kl_weight * kl_loss 
            logs["KL"] = kl_loss.item()

        
        if self.regression_loss!="none":
            loss += self.regression_weight * reg_loss 
            logs[self.regression_loss] = reg_loss.item()
        
            
        logs["loss"] =  loss.item()
               
        return loss, logs