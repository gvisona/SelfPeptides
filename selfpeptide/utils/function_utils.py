import numpy as np

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

