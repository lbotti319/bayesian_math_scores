import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymc3 as pm
from statsmodels.tsa.stattools import acf 


def MCMC_diagnostics(chain, param):
    plt.subplot(411)
    plt.plot(chain)
    plt.title(f'Trace Plot {param}')

    plt.subplot(412)
    plt.hist(chain, bins=60)
    plt.title(f'Histogram {param}')



    plt.subplot(413)
    acf_values = acf(chain)
    plt.scatter(range(0, len(acf_values)), acf_values)
    plt.title(f'ACF {param}')
    
    try:
        plt.subplot(414)
        gw_plot = pm.geweke(chain)
        plt.scatter(gw_plot[:,0],gw_plot[:,1])
        plt.axhline(-1.98, c='r')
        plt.axhline(1.98, c='r')

        plt.ylim(-2.5,2.5)
        plt.title(f'Geweke Plot Comparing first 10% and Slices of the Last 50% of Chain {param}')
    except AttributeError:
        pass
    
    plt.tight_layout()
    plt.show()

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
