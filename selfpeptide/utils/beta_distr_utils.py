import numpy as np
import torch

# BETA DISTRIBUTION
def beta_distr_mean(alpha, beta):
    # assert (alpha>=0)
    # assert (beta>=0)
    return alpha/(alpha+beta)


def beta_distr_var(alpha, beta):
    # assert (alpha>=0)
    # assert (beta>=0)
    return (alpha*beta)/((alpha+beta)**2 * (alpha+beta+1))


def get_alpha(mean, variance):
    if isinstance(mean, list):
        mean = np.array(mean)
    if isinstance(variance, list):
        variance = np.array(variance)
    
    return ((1-mean)/variance - 1/mean) * (mean**2)

def get_beta(mean, alpha):
    if isinstance(mean, list):
        mean = np.array(mean)
    if isinstance(alpha, list):
        alpha = np.array(alpha)
    return alpha * (1/mean -1)


def beta_distr_params_from_alpha_beta(alpha, beta):
    if isinstance(alpha, list):
        alpha = np.array(alpha)
    if isinstance(beta, list):
        beta = np.array(beta)
    mean = alpha/(alpha+beta)
    precision = alpha+beta
    variance = (alpha*beta)/((alpha+beta)**2 * (alpha+beta+1))
    
    mode = (alpha-1)/(alpha+beta-2)
    # if alpha>1 and beta>1:
    #     mode = (alpha-1)/(alpha+beta-2)
    # elif alpha<=1 and beta>1:
    #     mode = 0
    # elif alpha>1 and beta<=1:
    #     mode = 1
        
    return mean, precision, variance, mode



def beta_distr_params_from_mean_precision(mean, precision):
    if isinstance(mean, list):
        mean = np.array(mean)
    if isinstance(precision, list):
        precision = np.array(precision)
    alpha = mean * precision
    beta = precision - alpha
    
    variance = mean*(1-mean)/(1+precision)
    
    mode = (alpha-1)/(alpha+beta-2)
    # if alpha>1 and beta>1:
    #     mode = (alpha-1)/(alpha+beta-2)
    # elif alpha<=1 and beta>1:
    #     mode = 0
    # elif alpha>1 and beta<=1:
    #     mode = 1
        
    return alpha, beta, variance, mode


def beta_chernoff_distance(alpha1, beta1, alpha2, beta2, lbd=0.5):
    assert lbd>0 and lbd<1
    if isinstance(alpha1, (int, float)):
        alpha1 = torch.tensor([alpha1])
    if isinstance(beta1, (int, float)):
        beta1 = torch.tensor([beta1])
    if isinstance(alpha2, (int, float)):
        alpha2 = torch.tensor([alpha2])
    if isinstance(beta2, (int, float)):
        beta2 = torch.tensor([beta2])
    
    
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

    jc = (torch.lgamma(lbd*alpha1+(1-lbd)*alpha2 + lbd*beta1+(1-lbd)*beta2)
          + lbd * (torch.lgamma(alpha1)+torch.lgamma(beta1)) 
          + (1-lbd) * (torch.lgamma(alpha2)+torch.lgamma(beta2))
          - torch.lgamma(lbd*alpha1 + (1-lbd)*alpha2) - torch.lgamma(lbd*beta1 + (1-lbd)*beta2)
          - lbd * torch.lgamma(alpha1+beta1)
          - (1-lbd) * torch.lgamma(alpha2 + beta2)
         )
    return jc



def beta_kl_divergence(alpha1, beta1, alpha2, beta2):
    # KL{beta(alpha1, beta1)||beta(alpha2, beta2)}
    if isinstance(alpha1, (int, float)):
        alpha1 = torch.tensor([alpha1])
    if isinstance(beta1, (int, float)):
        beta1 = torch.tensor([beta1])
    if isinstance(alpha2, (int, float)):
        alpha2 = torch.tensor([alpha2])
    if isinstance(beta2, (int, float)):
        beta2 = torch.tensor([beta2])
        
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