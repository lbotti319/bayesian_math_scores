import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mvnorm
from scipy.stats import uniform, invgamma, bernoulli, poisson, norm
import statsmodels.api as sm

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
        pred = np.array(y - X*np.matrix(beta).T)[:,0]
        # sample a new variance
        var  = invgamma.rvs(n/2, pred.dot(pred)/2)
        
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

    reg = sm.OLS(y, X)
    res = reg.fit()
    
    beta_hat = np.array((X.T * X).I * X.T * np.matrix(y).T)[:,0]
    vbeta = (X.T * X).I
    # Initial parameters
    beta = beta_hat
    var = res.mse_total
    X = np.matrix(X)
    n = len(y)
    y = np.matrix(y).T
    
    betas[:,0] = beta_hat
    variances[0] = var
    accept = np.zeros(2*B)
    print(var)
    for i in range(1, 2*B):
        # var and beta are from the b-1 step until they are updated
        bstar = mvnorm(mean=beta_hat, cov = var*vbeta, allow_singular=True).rvs()
        log_r = log_beta_prob(X, y, bstar, var) - log_beta_prob(X, y, beta, var)
        log_u = np.log(uniform.rvs())
        
        # Gibbs step for sigma^2
        pred = np.array(y - X*np.matrix(beta).T)[:,0]
        # sample a new variance
        var = invgamma.rvs(n/2, pred.dot(pred)/2)
        variances[i] = var
        
        # update beta
        beta = bstar
        betas[:,i] = beta
    return betas[:,B:], accept[B:]


###################################################################
######### functions for missing data part of project ##############
###################################################################


def MH_step(param, log_target, log_proposal, accepts_param, *args):
    param_star = log_proposal.rvs()
    log_u = np.log(uniform.rvs())
    log_r = log_target(param_star, *args) - log_target(param, *args)
    if log_u < log_r:
        param = param_star
        accepts_param += 1
    return param, accepts_param


def dist_log_alpha0(alpha0, alpha1, higher_yes, age_missing_higher_yes, loc, scale):
    p = np.exp(alpha0 + alpha1*age_missing_higher_yes) / (1 + np.exp(alpha0 + alpha1*age_missing_higher_yes))
    return np.sum((1 - higher_yes) * np.log(1 - p) + higher_yes * np.log(p)) - (alpha0 - loc)**2 / (2 * scale)


def dist_log_alpha1(alpha1, alpha0, higher_yes, age_missing_higher_yes, loc, scale):
    p = np.exp(alpha0 + alpha1*age_missing_higher_yes) / (1 + np.exp(alpha0 + alpha1*age_missing_higher_yes))
    return np.sum((1 - higher_yes) * np.log(1 - p) + higher_yes * np.log(p)) - (alpha1 - loc)**2 / (2 * scale)


def dist_log_gamma0(gamma0, gamma1, absences, age_missing_absences, loc, scale):
    mu = np.exp(gamma0 + gamma1*age_missing_absences)
    return np.sum(absences * (gamma0 + gamma1*age_missing_absences) - mu) - (gamma0 - loc)**2 / (2 * scale)
    # np.prod(poisson.pmf(absences, mu)) * norm.pdf(gamma0, loc=0, scale=100)


def dist_log_gamma1(gamma1, gamma0, absences, age_missing_absences, loc, scale):
    mu = np.exp(gamma0 + gamma1*age_missing_absences)
    return np.sum(absences * (gamma0 + gamma1*age_missing_absences) - mu) - (gamma1 - loc)**2 / (2 * scale)
    # np.prod(poisson.pmf(absences, mu)) * norm.pdf(gamma1, loc=0, scale=100)


def Gibbs_MH(X, y, B, n, higher_yes_col, absences_col, age_col, tau, loc=0, scale=10):
    """

    """
    ################# Initializations ################
    # regression parameters
    betas = np.zeros((X.shape[1], 2 * B))
    sigmas2 = np.zeros(2 * B)

    # parameters for missing covariates
    higher_yes_missing_idx = np.where(np.isnan(X[:, higher_yes_col]))[0]
    higher_yes_sim = np.zeros((len(higher_yes_missing_idx), 2 * B))
    alphas0, alphas1 = np.zeros(2 * B), np.zeros(2 * B)

    absences_missing_idx = np.where(np.isnan(X[:, absences_col]))[0]
    absences_sim = np.zeros((len(absences_missing_idx), 2 * B))
    gammas0, gammas1 = np.zeros(2 * B), np.zeros(2 * B)

    # getting ages for the missing values
    age_missing_higher_yes = X[higher_yes_missing_idx, age_col]
    age_missing_absences = X[absences_missing_idx, age_col]

    # Initialize parameters
    higher_yes = round(np.nanmean(X[:, higher_yes_col]))
    higher_yes_sim[:, 0] = higher_yes
    X[higher_yes_missing_idx, higher_yes_col] = higher_yes_sim[:, 0]
    alpha0, alpha1 = 0.5, 0.5
    alphas0[0], alphas1[0] = alpha0, alpha1

    absences = round(np.nanmean(X[:, absences_col]))
    absences_sim[:, 0] = absences
    X[absences_missing_idx, absences_col] = absences_sim[:, 0]
    gamma0, gamma1 = 0.05, 0.05
    gammas0[0], gammas1[0] = gamma0, gamma1

    reg = sm.OLS(y, X)
    res = reg.fit()
    beta_hat = res.params
    # same as:
    # np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    sigma2 = res.mse_total
    vbeta = np.linalg.inv(X.T.dot(X))
    betas[:, 0] = beta_hat.copy()
    sigmas2[0] = sigma2

    accepts_alpha0, accepts_alpha1, accepts_gamma0, accepts_gamma1 = 0, 0, 0, 0
    ##################################################

    ############ Sampling Gibbs + MH #################
    proposal_alpha0 = norm(loc=0, scale=tau)
    proposal_alpha1 = norm(loc=0, scale=tau)
    proposal_gamma0 = norm(loc=0, scale=tau)
    proposal_gamma1 = norm(loc=0, scale=tau)

    for i in range(1, 2 * B):
        # sample a beta
        beta = mvnorm(mean=beta_hat, cov=sigmas2[i-1] * vbeta, allow_singular=True).rvs()
        # sample a sigma2
        sigma2 = invgamma.rvs(n / 2, (y - X.dot(betas[:, i-1])).T.dot(y - X.dot(betas[:, i-1])) / 2)
        # sample the missing higher_yes
        p = np.exp(alpha0 + alpha1 * age_missing_higher_yes) / (1 + np.exp(alpha0 + alpha1 * age_missing_higher_yes))
        higher_yes = bernoulli.rvs(p)
        # sample the missing absences
        mu = np.exp(gamma0 + gamma1 * age_missing_absences)
        absences = poisson.rvs(mu)
        # sample alpha0, alpha1
        alpha0, accepts_alpha0 = MH_step(alpha0, dist_log_alpha0, proposal_alpha0, accepts_alpha0,
                                         alpha1, higher_yes_sim[:, i-1], age_missing_higher_yes, loc, scale)
        alpha1, accepts_alpha1 = MH_step(alpha1, dist_log_alpha1, proposal_alpha1, accepts_alpha1,
                                         alpha0, higher_yes_sim[:, i-1], age_missing_higher_yes, loc, scale)
        # sample gamma0, gamma1
        gamma0, accepts_alpha0 = MH_step(gamma0, dist_log_gamma0, proposal_gamma0, accepts_gamma0,
                                         gamma1, absences_sim[:, i-1], age_missing_absences, loc, scale)
        gamma1, accepts_alpha1 = MH_step(gamma1, dist_log_gamma1, proposal_gamma1, accepts_gamma1,
                                         gamma0, absences_sim[:, i-1], age_missing_absences, loc, scale)

        # updates
        betas[:, i] = beta
        sigmas2[i] = sigma2
        higher_yes_sim[:, i] = higher_yes
        absences_sim[:, i] = absences
        alphas0[i] = alpha0
        alphas1[i] = alpha1
        gammas0[i] = gamma0
        gammas1[i] = gamma1
        X[higher_yes_missing_idx, higher_yes_col] = higher_yes
        X[absences_missing_idx, absences_col] = absences
        beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        vbeta = np.linalg.inv(X.T.dot(X))

        print((y - X.dot(betas[:, i-1])).T.dot(y - X.dot(betas[:, i-1])) / 2)
    ##################################################

    return (betas, sigmas2, higher_yes_sim, absences_sim, alphas0, alphas1, gammas0, gammas1,
            accepts_alpha0, accepts_alpha1, accepts_gamma0, accepts_gamma0)

###################################################################
###################################################################

