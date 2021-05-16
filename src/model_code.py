import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mvnorm
from scipy.stats import uniform, invgamma, bernoulli, poisson, norm
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LinearRegression

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


###################################################################
######### functions for missing data part of project ##############
###################################################################





def Gibbs_MH(X, y, B, thin):
    """
    docstring here
    """
    ################# Initializations ################
    N = 2 * B * thin
    X = X.copy()
    n = X.shape[0]

    # regression parameters
    betas = np.zeros((X.shape[1], N))
    sigmas2 = np.zeros(N)

    # parameters for missing covariates
    higher_yes_missing_idx = X['higher_yes'].isna()
    higher_yes_sim = np.zeros((higher_yes_missing_idx.sum(), N))
    alphas = np.zeros((2,N))

    G2_missing_idx = X['G2'].isna()
    # G2_missing_idx.sum() gives the number of Trues 
    G2_sim = np.zeros((G2_missing_idx.sum(), N))
    gammas = np.zeros((2, N))
    etas = np.zeros(N)

    # initialize regression and missing covariates parameters
    higher_yes = round(np.nanmean(X['higher_yes']))
    higher_yes_sim[:, 0] = higher_yes
    X.loc[higher_yes_missing_idx, 'higher_yes'] = higher_yes_sim[:, 0]

    mat = np.matrix(X[['intercept', 'age']])
    base_cov = (mat.T * mat).I

    model = LogisticRegression(fit_intercept=False).fit(
        X[['intercept', 'age']][higher_yes_missing_idx == False],
        X['higher_yes'][higher_yes_missing_idx == False]
    )
    alpha = model.coef_.flatten().tolist()
    alphas[:,0] = alpha

    G2 = round(np.nanmean(X["G2"]))
    G2_sim[:, 0] = G2
    X.loc[G2_missing_idx, "G2"] = G2_sim[:, 0]

    model = LinearRegression(fit_intercept=False).fit(
        X[['intercept', 'age']][G2_missing_idx == False], 
        X['G2'][G2_missing_idx == False]
    )

    gamma = model.coef_.flatten().tolist()
    gammas[:,0] = gamma 

    reg = sm.OLS(y, X)
    res = reg.fit()
    beta_hat = res.params
    sigma2 = res.mse_total
    vbeta = np.linalg.inv(X.T.dot(X))
    betas[:, 0] = beta_hat.copy()
    sigmas2[0] = sigma2

    accepts_alpha, accepts_gamma = 0, 0
    ##################################################

    for i in tqdm(range(1, N)):
        # sample a beta
        beta = mvnorm(mean=beta_hat, cov=sigmas2[i-1] * vbeta, allow_singular=True).rvs()
        # sample a sigma2
        sigma2 = invgamma.rvs(n / 2, scale=(y - X.dot(betas[:, i-1])).T.dot(y - X.dot(betas[:, i-1])) / 2)
        # sample the missing higher_yes
        log_odds = np.array(
            X[['intercept', 'age']][higher_yes_missing_idx].values * 
            np.matrix(alphas[:, i-1]).T
        )[:,0]
        p = np.exp(log_odds) / (1 + np.exp(log_odds))
        higher_yes = bernoulli.rvs(p)
        # sample the missing G2s
        mu = X[['intercept', 'age']][G2_missing_idx].values * np.matrix(gammas[:,i-1]).T
        G2s =  mvnorm.rvs(mean=np.array(mu)[:,0], cov=etas[i-1])


        # sample alpha
        alpha_reg = sm.GLS(
             X['higher_yes'],
             X[['intercept', 'age']],
             family=sm.families.Binomial()
        )
        alpha_res =alpha_reg.fit()
        alpha_mean = alpha_res.params.values
        alpha_cov = alpha_res.cov_params().values
        alpha = mvnorm.rvs(
            mean=alpha_mean,
            cov=base_cov
        )

        gamma_reg = sm.OLS(X['G2'], X[['intercept', 'age']])
        gamma_res = gamma_reg.fit()
        gamma_mean = gamma_res.params.values
        gamma_var = etas[i-1] * base_cov
        gamma = mvnorm.rvs(
            mean=gamma_mean,
            cov=gamma_var
        )


        eta_a = (n-2) /2
        error = (
            X['G2'].values - 
            X[['intercept', 'age']].values *np.matrix(gammas[:,i-1]).T
        )
        eta_b =0.5 * (error.T * error)[0,0]
        # print(eta_a, eta_b)
        eta = invgamma.rvs(eta_a, scale=eta_b)

        # updates
        betas[:, i] = beta
        sigmas2[i] = sigma2
        higher_yes_sim[:, i] = higher_yes
        G2_sim[:, i] = G2s
        alphas[:, i] = alpha
        gammas[:, i] = gamma
        etas[i] = eta
        X.loc[higher_yes_missing_idx, 'higher_yes'] = higher_yes
        X.loc[G2_missing_idx, 'G2'] = G2s
        beta_hat = (
            np.linalg.inv(X.values.T.dot(X.values))
            .dot(X.values.T)
            .dot(y)
        )
        vbeta = np.linalg.inv(X.values.T.dot(X.values))


    return (betas[:, B*thin::thin], sigmas2[B*thin::thin], higher_yes_sim[:, B*thin::thin], G2_sim[:, B*thin::thin], 
            alphas[:, B*thin::thin], gammas[:,B*thin::thin] , etas[B*thin::thin])





