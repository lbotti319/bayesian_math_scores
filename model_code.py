import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import multivariate_normal as mvnorm
from scipy.stats import uniform, invgamma


def log_beta_prob(X, y, beta, var=1):
    """
    Variance is 1 for now, not quite sure how we want to handle it
    """
    beta = np.matrix(beta).T
    prediction = np.array(y - X*beta)[:,0]
    dot_product = prediction.dot(prediction)
    return -(1/2*var) * dot_product

def MH_regression(X, y, B, tau):
    betas = np.zeros((X.shape[1], 2*B))
    sigmas = np.zeros(2*B)

    reg = sm.OLS(y, X)
    res = reg.fit()

    # Initial parameters
    beta = res.params
    var = res.mse_total
    vbeta = res.cov_params()
    X = np.matrix(X)
    n = len(y)
    y = np.matrix(y).T
    
    betas[:,0] = beta
    sigmas[0] = var
    accept = np.zeros(2*B)
    for i in range(1, 2*B):
        # var and beta are from the b-1 step until they are updated
        bstar = mvnorm(mean=beta, cov = tau*vbeta, allow_singular=True).rvs()
        log_r = log_beta_prob(X, y, bstar, var) - log_beta_prob(X, y, beta, var)
        log_u = np.log(uniform.rvs())
#         print(log_u, log_r, var)
        
        # Gibbs step for sigma^2
        error = np.array(y - X*np.matrix(beta).T)[:,0]
        # sample a new variance
        var  = invgamma.rvs(n/2, scale=error.dot(error)/2)
        
        # update beta if conditions are fulfilled
        if log_u < min(log_r, 0):
            beta = bstar
            accept[i] = 1
        betas[:,i] = beta
    return betas[:,B:], accept[B:]

def Gibbs_regression(X, y, B):
    """
    Here if we want it
    """
    betas = np.zeros((X.shape[1], 2*B))
    variances = np.zeros(2*B)

#     reg = sm.OLS(y, X)
#     res = reg.fit()

    X = np.matrix(X)
    n = len(y)
    k = X.shape[1] - 1
    y = np.matrix(y).T

    beta_hat = np.array((X.T * X).I * X.T * y)[:,0]
    vbeta = (X.T * X).I
    # Initial parameters
    beta = beta_hat
    error = y - X * np.matrix(beta).T
    var =(error.T * error).sum() / (n - k)
    
    betas[:,0] = beta_hat
    variances[0] = var
    for i in range(1, 2*B):
        # var and beta are from the b-1 step until they are updated
        bstar = mvnorm(mean=beta_hat, cov = var* vbeta).rvs()
        
        # Gibbs step for sigma^2
        error = y - (X * np.matrix(beta).T)
        dot_prod = (error.T * error).sum()
        # sample a new variance
        var  = invgamma.rvs(n/2, scale=dot_prod/2)
        variances[i] = var
        
        # update beta
        beta = bstar
        betas[:,i] = beta
    return betas[:, B:] , variances[B:]

def running_mean(x):
    sums = np.cumsum(x) 
    n = np.arange(1,len(x)+1)
    return sums/n

def gelman_rubin(chains):
    M = chains.shape[0]
    N = chains.shape[1]
    W = np.sum(np.apply_along_axis(np.var, 1, chains)) / M
    means = np.apply_along_axis(np.mean, 1, chains)
    double_mean = np.mean(means)
    B = N * np.sum((means - double_mean)**2) / (M-1)
    var_hat = (1 - (1/N))* W + (1/N)*B
    R_hat = np.sqrt(var_hat/W)
    return R_hat 

