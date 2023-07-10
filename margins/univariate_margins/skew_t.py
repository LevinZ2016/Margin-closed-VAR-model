# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:13:36 2022

@author: Levin
"""

import scipy.stats as st
import scipy.special as sp
import autograd.numpy as np
from autograd import elementwise_grad
import numpy

from .template import (break_par_dist, distribution)

class skew_t(break_par_dist):
    
    def __init__(self, par=None,
                 lb=[-np.inf, 1e-6, 1e-6, 1e-6], 
                 ub=[np.inf, np.inf, 1e3, 1e3]):
        super().__init__(par_dim=4, lb=lb, ub=ub)
        self.par = par
        self.com_dist = [skew_t_ab, skew_t_inf_a, skew_t_inf_b]
        
    def par2dist(self):
        if self.par is None:
            dist = skew_t_ab()
        elif np.isinf(self.par[2]):
            dist_par = self.par.tolist()
            dist_par.pop(2)
            dist = skew_t_inf_a(dist_par)
        elif np.isinf(self.par[3]):
            dist_par = self.par.tolist()
            dist_par.pop(3)
            dist = skew_t_inf_b(dist_par)
        else:
            dist_par = self.par.tolist()
            dist = skew_t_ab(dist_par)
        return dist
            
    def dist2par(self):
        if self.dist.par is None:
            return None
        
        par = self.dist.par.tolist()
        if self.dist.class_name == 'skew_t_inf_a':
            par.insert(2, np.inf)
        elif self.dist.class_name == 'skew_t_inf_b':
            par.insert(3, np.inf)
        
        return np.array(par)

class skew_t_ab(distribution):
    
    def __init__(self,
                 par=None,
                 lb=[-np.inf, 1e-6, 1e-6, 1e-6], 
                 ub=[np.inf, np.inf, 1e3, 1e3]):
        super().__init__(par=par, sup=[-np.inf, np.inf], par_dim=4, 
                         lb=lb, ub=ub)
    
    def elementwise_logM(self, data, mu_l, sigma_l, a_l, b_l):
        '''
        -------
        M(t) = f(t) * Beta(a, b)

        '''
        z = (data - mu_l) / sigma_l
        logM = -(a_l+b_l-1) * np.log(2) - 0.5 * np.log(a_l+b_l) \
                + (a_l + 0.5) * np.log(1 + z / (a_l + b_l + z**2)**0.5) \
                + (b_l + 0.5) * np.log(1 - z / (a_l + b_l + z**2)**0.5)
        return logM - np.log(sigma_l)
    
    def elementwise_logB(self, a_l, b_l):
        '''
        -------
        return -log(Beta(a, b))

        '''
        return -sp.betaln(a_l,b_l)
    
    def elementwise_dlogB(self, a_l, b_l):
        da = - sp.digamma(a_l) + sp.digamma(a_l+b_l)
        db = - sp.digamma(b_l) + sp.digamma(a_l+b_l)
        return da, db
    
    def elementwise_scale(self, data, mu_l, sigma_l, a_l, b_l):
        y = (data - mu_l) / sigma_l
        ytr = 0.5 * (1 + y / np.sqrt(a_l + b_l + y**2))
        return ytr
    
    def elementwise_cdf(self, data, mu_l, sigma_l, a_l, b_l):
        ytr = self.elementwise_scale(data, mu_l, sigma_l, a_l, b_l)
        cdf = st.beta.cdf(ytr, a_l, b_l)
        return cdf
    
    def elementwise_ppf(self, prob_l, mu_l, sigma_l, a_l, b_l):
        quant = st.beta.ppf(prob_l, a_l, b_l)
        quant = np.sqrt(a_l+b_l) * (2*quant - 1) / 2 / np.sqrt(quant * (1-quant))
        return quant * sigma_l + mu_l
    
    def elementwise_logpdf(self, data, mu_l, sigma_l, a_l, b_l):
        logpdf = self.elementwise_logB(a_l, b_l) \
            + self.elementwise_logM(data, mu_l, sigma_l, a_l, b_l)
        return logpdf
    
    def elementwise_dlogpdf(self, data, mu_l, sigma_l, a_l, b_l):
        da, db = self.elementwise_dlogB(a_l, b_l)
        g_mu = elementwise_grad(self.elementwise_logM, 1)(data, mu_l, 
                                                          sigma_l, a_l, b_l)
        g_sigma = elementwise_grad(self.elementwise_logM, 2)(data, mu_l, 
                                                             sigma_l, a_l, b_l)
        g_a = elementwise_grad(self.elementwise_logM, 3)(data, mu_l, 
                                                         sigma_l, a_l, b_l)
        g_b = elementwise_grad(self.elementwise_logM, 4)(data, mu_l, 
                                                         sigma_l, a_l, b_l)
        g_a += da
        g_b += db
        return np.vstack([g_mu, g_sigma, g_a, g_b]).T
    
    def elementwise_dcdf(self, data, mu_l, sigma_l, a_l, b_l):
        ytr = self.elementwise_scale(data, mu_l, sigma_l, a_l, b_l)
        pdf = st.beta.pdf(ytr, a_l, b_l)
        
        ds_mu = elementwise_grad(self.elementwise_scale, 1)(data, mu_l, 
                                                            sigma_l, a_l, b_l)
        ds_sigma = elementwise_grad(self.elementwise_scale, 2)(data, mu_l, 
                                                               sigma_l, a_l, b_l)
        ds_a = elementwise_grad(self.elementwise_scale, 3)(data, mu_l, 
                                                           sigma_l, a_l, b_l)
        ds_b = elementwise_grad(self.elementwise_scale, 4)(data, mu_l, 
                                                           sigma_l, a_l, b_l)
        g_mu, g_sigma, g_a1, g_b1 = pdf*ds_mu, pdf*ds_sigma, pdf*ds_a, pdf*ds_b
        
        dim, n = len(ytr), 100
        x1, w1 = numpy.polynomial.legendre.leggauss(n)
        x1 = np.tile(x1, [dim, 1]).ravel()
        y1 = np.repeat(ytr, n)
        a1 = np.repeat(a_l, n)
        b1 = np.repeat(b_l, n)

        z = (x1+1)/2 * y1
        pdf1 = st.beta.pdf(z, a1, b1)
        da = (np.log(z) - sp.digamma(a1) + sp.digamma(a1+b1)) * pdf1 * y1 / 2
        db = (np.log(1-z) - sp.digamma(b1) + sp.digamma(a1+b1)) * pdf1 * y1 / 2
        da.shape = [dim, n]
        db.shape = [dim, n]
        g_a2 = (da * w1[None, :]).sum(axis=1)
        g_b2 = (db * w1[None, :]).sum(axis=1)
        return np.vstack([g_mu, g_sigma, g_a1+g_a2, g_b1+g_b2]).T
    
    def init_par(self, data):
        k = st.kurtosis(data)
        if k > 0:
            a = 3/k+2
            b = 3/k+2
        else:
            a = b = 2
        mu = np.mean(data)
        sigma = np.std(data) * np.sqrt((a-1)/a)
        return [mu, sigma, a, b]


class skew_t_inf_a(distribution):
    
    def __init__(self, par=None,
                 lb=[-np.inf, 1e-6, 1e-6], ub=[np.inf, np.inf, 1e3]):
        super().__init__(par=par, sup=[0, np.inf], par_dim=3, 
                         lb=lb, ub=ub)
    
    def elementwise_logM(self, data, mu_l, sigma_l, b_l):
        '''
        -------
        M(t) = f(t) * gamma(k)

        '''
        z = (data - mu_l) / sigma_l
        logM = np.log(2) - (2 * b_l + 1) * np.log(z) - 1/z**2 - np.log(sigma_l)
        return logM
    
    def elementwise_logG(self, b_l):
        '''
        -------
        return log(gamma(k))

        '''
        return -sp.gammaln(b_l)
    
    def elementwise_dlogG(self, b_l):
        return -sp.digamma(b_l)
    
    def elementwise_cdf(self, data, mu_l, sigma_l, b_l):
        y = (data - mu_l) / sigma_l
        prob = 1 - sp.gammainc(b_l, 1/y**2)
        return prob
    
    def elementwise_ppf(self, prob_l, mu_l, sigma_l, b_l):
        y = 1 / np.sqrt(sp.gammaincinv(b_l, 1 - prob_l))
        return y * sigma_l + mu_l
    
    def elementwise_logpdf(self, data, mu_l, sigma_l, b_l):
        valid_data = data[data>mu_l]
        valid_mu = mu_l[data>mu_l]
        valid_sigma = sigma_l[data>mu_l]
        valid_b = b_l[data>mu_l]
        
        logpdf = np.array([-np.inf]*len(data))
        logpdf[data>mu_l] = self.elementwise_logG(valid_b) \
                            + self.elementwise_logM(valid_data, valid_mu, 
                                                    valid_sigma, valid_b)
        return logpdf
    
    def elementwise_dlogpdf(self, data, mu_l, sigma_l, b_l):
        db = self.elementwise_dlogG(b_l)
        g_mu = elementwise_grad(self.elementwise_logM, 1)(data, mu_l, 
                                                          sigma_l, b_l)
        g_sigma = elementwise_grad(self.elementwise_logM, 2)(data, mu_l, 
                                                             sigma_l, b_l)
        g_b = elementwise_grad(self.elementwise_logM, 3)(data, mu_l, 
                                                         sigma_l, b_l)
        g_b += db
        grad = np.vstack([g_mu, g_sigma, g_b]).T
        return grad
    
    def elementwise_dcdf(self, data, mu_l, sigma_l, b_l):
        y = (data - mu_l) / sigma_l
        inv_y = 1/y**2
        log_dcdf2y = np.log(2) + (b_l-1) * np.log(inv_y) \
                    - inv_y - sp.gammaln(b_l) - 3 * np.log(y)
        g_mu = - np.exp(log_dcdf2y - np.log(sigma_l))
        g_sigma = - np.exp(log_dcdf2y + np.log(y) - np.log(sigma_l))
        
        dim, n = len(inv_y), 100
        x1, w1 = numpy.polynomial.legendre.leggauss(n)
        x1 = np.tile(x1, [dim, 1]).ravel()
        y1 = np.repeat(inv_y, n)
        b1 = np.repeat(b_l, n)
        
        z = (x1+1)/2 * y1
        log_df = (b1-1) * np.log(z) - z + np.log(y1) - np.log(2)
        df = np.exp(log_df) * np.log(z)
        df.shape = [dim, n]
        dF = (df * w1[None, :]).sum(axis=1)
        g_b = - (dF / sp.gamma(b_l) - sp.digamma(b_l) * sp.gammainc(b_l, inv_y))
        return np.vstack([g_mu, g_sigma, g_b]).T
        
    def init_par(self, data):
        k = st.kurtosis(data)
        if k > 0:
            b = 3/k+2
        else:
            b = 2
        mu = np.min(data) - (np.max(data) - np.min(data)) / 2
        sigma = np.std(data) * np.sqrt((b-1)/b)
        return [mu, sigma, b]
    
    def fix_boundary(self, data):
        self.ub[0] = np.min(data) - (np.max(data) - np.min(data)) / 100
        
    def fix_support(self):
        if self.par is not None:
            self.sup = [self.par[0], np.inf]
        
        
class skew_t_inf_b(distribution):
    
    def __init__(self, par=None,
                 lb=[-np.inf, 1e-6, 1e-6], ub=[np.inf, np.inf, 1e3]):
        super().__init__(par=par, sup=[-np.inf, 0], par_dim=3, 
                         lb=lb, ub=ub)
    
    def elementwise_logM(self, data, mu_l, sigma_l, a_l):
        '''
        -------
        M(t) = f(t) * gamma(k)

        '''
        z = (data - mu_l) / sigma_l
        logM = np.log(2) - (2 * a_l + 1) * np.log(-z) - 1/(-z)**2 - np.log(sigma_l)
        return logM
    
    def elementwise_logG(self, a_l):
        '''
        -------
        return log(gamma(k))

        '''
        return -sp.gammaln(a_l)
    
    def elementwise_dlogG(self, a_l):
        return -sp.digamma(a_l)
    
    def elementwise_cdf(self, data, mu_l, sigma_l, a_l):
        y = (data - mu_l) / sigma_l
        prob = sp.gammainc(a_l, 1/y**2)
        return prob
    
    def elementwise_ppf(self, prob_l, mu_l, sigma_l, a_l):
        y = 1 / np.sqrt(sp.gammaincinv(a_l, prob_l))
        return -y * sigma_l + mu_l
    
    def elementwise_logpdf(self, data, mu_l, sigma_l, a_l):
        logpdf = np.array([-np.inf]*len(data))
        logpdf[data<mu_l] = self.elementwise_logG(a_l) \
                            + self.elementwise_logM(data, mu_l, sigma_l, a_l)
        return logpdf
    
    def elementwise_dlogpdf(self, data, mu_l, sigma_l, a_l):
        da = self.elementwise_dlogG(a_l)
        g_mu = elementwise_grad(self.elementwise_logM, 1)(data, mu_l, 
                                                          sigma_l, a_l)
        g_sigma = elementwise_grad(self.elementwise_logM, 2)(data, mu_l, 
                                                             sigma_l, a_l)
        g_a = elementwise_grad(self.elementwise_logM, 3)(data, mu_l, 
                                                         sigma_l, a_l)
        g_a += da
        grad = np.vstack([g_mu, g_sigma, g_a]).T
        return grad
    
    def elementwise_dcdf(self, data, mu_l, sigma_l, a_l):
        y = (data - mu_l) / sigma_l
        inv_y = 1/y**2
        log_dcdf2y = np.log(2) + (a_l-1) * np.log(inv_y) \
                    - inv_y - sp.gammaln(a_l) - 3 * np.log(-y)
        g_mu = - np.exp(log_dcdf2y - np.log(sigma_l))
        g_sigma = np.exp(log_dcdf2y + np.log(-y) - np.log(sigma_l))
        
        dim, n = len(inv_y), 100
        x1, w1 = numpy.polynomial.legendre.leggauss(n)
        x1 = np.tile(x1, [dim, 1]).ravel()
        y1 = np.repeat(inv_y, n)
        a1 = np.repeat(a_l, n)
        
        z = (x1+1)/2 * y1
        log_df = (a1-1) * np.log(z) - z + np.log(y1) - np.log(2)
        df = np.exp(log_df) * np.log(z)
        df.shape = [dim, n]
        dF = (df * w1[None, :]).sum(axis=1)
        g_a = dF / sp.gamma(a_l) - sp.digamma(a_l) * sp.gammainc(a_l, inv_y)
        return np.vstack([g_mu, g_sigma, g_a]).T
    
    def init_par(self, data):
        k = st.kurtosis(data)
        if k > 0:
            a = 3/k+2
        else:
            a = 2
        mu = np.max(data) + (np.max(data) - np.min(data)) / 2
        sigma = np.std(data) * np.sqrt((a-1)/a)
        return [mu, sigma, a]
    
    def fix_boundary(self, data):
        self.lb[0] = np.max(data) + (np.max(data) - np.min(data)) / 100
        
    def fix_support(self):
        if self.par is not None:
            self.sup = [-np.inf, self.par[0]]