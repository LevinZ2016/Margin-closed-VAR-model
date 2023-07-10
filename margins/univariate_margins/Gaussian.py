# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:13:35 2022

@author: Levin
"""

import scipy.stats as st
import autograd.numpy as np
from autograd import elementwise_grad

from .template import distribution

class Gaussian(distribution):
    
    def __init__(self, par=None):
        super().__init__(par=par, sup=[-np.inf, np.inf], 
                         par_dim=2, lb=[-np.inf, 1e-6], ub=[np.inf]*2)
    
    def elementwise_logpdf(self, data, mu_l, sigma_l):
        '''
        -------
        f(t) is the density function of Gaussian

        '''
        z = (data - mu_l) / sigma_l
        logM = -0.5 * np.log(2 * np.pi) - 0.5 * z**2
        return logM - np.log(sigma_l)
    
    def elementwise_cdf(self, data, mu_l, sigma_l):
        y = (data - mu_l) / sigma_l
        cdf = st.norm.cdf(y)
        return cdf
    
    def elementwise_ppf(self, prob_l, mu_l, sigma_l):
        quant = st.norm.ppf(prob_l)
        return quant * sigma_l + mu_l
    
    def elementwise_dlogpdf(self, data, mu_l, sigma_l):
        g_mu = elementwise_grad(self.elementwise_logpdf, 1)(data, mu_l, 
                                                            sigma_l)
        g_sigma = elementwise_grad(self.elementwise_logpdf, 2)(data, mu_l, 
                                                               sigma_l)
        return np.vstack([g_mu, g_sigma]).T
    
    def elementwise_dcdf(self, data, mu_l, sigma_l):
        z = (data - mu_l) / sigma_l
        g_mu = - st.norm.pdf(z) / sigma_l
        g_sigma = - st.norm.pdf(z) * z / sigma_l
        return np.vstack([g_mu, g_sigma]).T
        
    def init_par(self, data):
        mu = np.mean(data)
        sigma = np.std(data)
        return [mu, sigma]