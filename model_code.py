import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import multivariate_normal as mvnorm
from scipy.stats import uniform


def log_beta_prob(X, y, beta, var=1):
    """
    Variance is 1 for now, not quite sure how we want to handle it
    """
    beta = np.matrix(beta).T
    y = np.matrix(y).T
    paren = (y - X*beta)
    mat_mult = paren.T * paren
    return -(1/2*var) * mat_mult[0,0]

def MH_regression(X, y, B, tau):
    betas = np.zeros((X.shape[1], 2*B))
    reg = sm.OLS(y, X)
    res = reg.fit()
    beta = res.params
    vbeta = res.cov_params()
    X = np.matrix(X)
    
    betas[:,0] = beta
    accept = np.zeros(2*B)
    for i in range(1, 2*B):
        bstar = mvnorm(mean=beta, cov = tau*vbeta).rvs()
        log_r = log_beta_prob(X, y, bstar) - log_beta_prob(X, y, beta)
        log_u = np.log(uniform.rvs())
        if log_u < min(log_r, 0):
            beta = bstar
            accept[i] = 1
        betas[:,i] = beta
    return betas[:,B:], accept[B:]
