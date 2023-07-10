#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:03:39 2021

@author: levin
"""

from scipy.optimize import (minimize, Bounds)
import matplotlib.pyplot as plt
import autograd.numpy as np
from copy import deepcopy


class Monitor():
    
    def __init__(self, name, n_par, sample_size, loglik, sol):
        self.dist_name = name
        self.n_par = n_par
        self.sample_size = sample_size
        self.loglik = loglik
        self.sol = sol
    
    def __repr__(self):
        return ("{},\n".format(self.dist_name) 
                + 'sample_size = {}\n'.format(self.sample_size)
                + 'loglik = {}\n'.format(self.loglik)
                + 'aic = {}\n'.format(self.aic())
                + 'bic = {}\n'.format(self.bic()))
    
    def aic(self):
        return 2 * self.n_par - 2 * self.loglik
    
    def bic(self):
        return np.log(self.sample_size) * self.n_par - 2 * self.loglik


class distribution():
    
    def __init__(self, par, sup, par_dim, lb, ub):
        self.par_dim = par_dim
        self.sup = sup
        self.lb, self.ub = np.array(lb), np.array(ub)
        self.par = par
        self.class_name = self.__class__.__name__
        
    def __repr__(self):
        return ("{},".format(self.class_name) 
                + ' parmeters = {}\n'.format(self.par))
    
    @property
    def par(self):
        return self._par
    
    @par.setter
    def par(self, par):
        self._par = self.check_par(par)
        self.fix_support()
        
    def check_par(self, par):
        if par is None:
            return None
        elif len(par) != self.par_dim:
            raise ValueError('The dimension of the parameters is incorrect!')
        elif any(par > self.ub):
            raise ValueError('The parmeters exceed the upper bound!')
        elif any(par < self.lb):
            raise ValueError('The parmeters exceed the lower bound!')
        else:
            return np.array(par)
        
    def check_data(self, data):
        return np.array(data).ravel()
    
    def check_weights(self, data, weights):
        if weights is None:
            num_sam = len(data)
            return np.array([1]*num_sam)
        else:
            return np.array(weights).ravel()
    
    def fix_boundary(self, data):
        pass
    
    def fix_support(self):
        pass
    
    def fill_output(self, data, fill):
        input_data = self.check_data(data)
        in_sup = (input_data > self.sup[0]) & (input_data < self.sup[1])
        less_sup = (input_data <= self.sup[0])
        bigger_sup = (input_data >= self.sup[1])
        output = np.ones(len(input_data)) * fill
        return input_data[in_sup], output, in_sup, less_sup, bigger_sup
    
    def trans_elementwise(self, func, data):
        input_data = self.check_data(data)
        num_obs = len(input_data)
        return func(input_data, *np.tile(self.par, [num_obs, 1]).T)
        
    def logpdf(self, data):
        valid_data, output, in_sup, _, _ = self.fill_output(data, -np.inf)
        output[in_sup] = self.trans_elementwise(self.elementwise_logpdf, 
                                                valid_data)
        return output
    
    def dlogpdf(self, data):
        valid_data, output, in_sup, _, _ = self.fill_output(data, 0.)
        output = np.array([output] * self.par_dim).T
        output[in_sup] = self.trans_elementwise(self.elementwise_dlogpdf, 
                                                valid_data)
        return self.trans_elementwise(self.elementwise_dlogpdf, data)
    
    def cdf(self, data):
        valid_data, output, in_sup, _, bigger_sup = self.fill_output(data, 0.)
        output[in_sup] = self.trans_elementwise(self.elementwise_cdf, 
                                                valid_data)
        output[bigger_sup] = 1.                                 
        return output
    
    def ppf(self, data):
        return self.trans_elementwise(self.elementwise_ppf, data)
    
    def dcdf(self, data):
        return self.trans_elementwise(self.elementwise_dcdf, data)
    
    def pdf(self, data):
        output = np.exp(self.logpdf(data))
        output[np.isneginf(output)] = 0.
        return output
    
    def nllk(self, data, par):
        cop = deepcopy(self)
        cop.par = par
        return -cop.logpdf(data)
    
    def gradnllk(self, data, par):
        cop = deepcopy(self)
        cop.par = par
        return -cop.dlogpdf(data)
    
    def fit(self, data, weights=None, init_par=None, gtol=1e-5, disp=False):
        input_data = self.check_data(data)
        w = self.check_weights(input_data, weights)
        self.fix_boundary(data)
        
        if init_par is None:
            init_par = self.init_par(input_data)
        
        nllk = lambda x: np.array(np.dot(self.nllk(input_data, x).T, w))
        gradnllk = lambda x: np.array(np.dot(self.gradnllk(input_data, x).T, w))
        
        b = Bounds(lb=self.lb, ub=self.ub, keep_feasible=True)
        sol = minimize(nllk, init_par, method='L-BFGS-B', bounds=b,
                       jac=gradnllk, options={'gtol': gtol, 'disp': disp})
        
        self.par = sol.x
        self.monitor = Monitor(self.class_name, self.par_dim, len(input_data), 
                               -np.array(np.dot(self.nllk(input_data, 
                                                          self.par).T, 
                                                w)), sol)
        self.aic = self.monitor.aic()
        self.bic = self.monitor.bic()
    
    def plot(self, data=None, interval=None, bins=None, display=True):
        if data is not None:
            if bins is None:
                plt.hist(data, density=True)
            else:
                plt.hist(data, density=True, bins=bins)
            min_x, max_x = np.min(data), np.max(data)
        else:
            min_x, max_x = self.ppf([0.05, 0.95])
        x = np.linspace(min_x - (max_x - min_x) / 6,
                        max_x + (max_x - min_x) / 6,
                        200)
        y = self.pdf(x)
        plt.plot(x, y)
        plt.grid()
        if interval is not None:
            plt.xlim(interval)
        if display:
            plt.show()
            
    def qqplot(self, data):
        import statsmodels.api as sm
        sm.qqplot(data, dist=self, line='45')
        plt.grid()
        plt.show()
            
        
class break_par_dist():
    
    def __init__(self, par_dim, lb, ub):
        self.par_dim = par_dim
        self.lb, self.ub = np.array(lb), np.array(ub)
        self.class_name = self.__class__.__name__
    
    def __repr__(self):
        return ("{},".format(self.class_name) 
                + ' parmeters = {}\n'.format(self.par))
    
    @property
    def par(self):
        return self._par
    
    @par.setter
    def par(self, par):
        self._par  = self.check_par(par)
        self.dist = self.par2dist()
        
    @property
    def dist(self):
        return self._dist
        
    @dist.setter
    def dist(self, dist):
        self._dist = dist
        self._par = self.dist2par()
        
        self.logpdf = self.dist.logpdf
        self.cdf = self.dist.cdf
        self.pdf = self.dist.pdf
        self.ppf = self.dist.ppf
        self.dlogpdf = self.dist.dlogpdf
        self.dcdf = self.dist.dcdf
        self.nllk = self.dist.nllk
        self.gradnllk = self.dist.gradnllk
        self.plot = self.dist.plot
        self.qqplot = self.dist.qqplot
        
        self.elementwise_logpdf = self.dist.elementwise_logpdf
        self.elementwise_cdf = self.dist.elementwise_cdf
        self.elementwise_ppf = self.dist.elementwise_ppf
        self.elementwise_dlogpdf = self.dist.elementwise_dlogpdf
        self.elementwise_dcdf = self.dist.elementwise_dcdf
        
        self.sup = self.dist.sup
        
    def check_par(self, par):
        if par is None:
            return None
        elif len(par) != self.par_dim:
            raise ValueError('The dimension of the parmeters is incorrect!')
        elif any(par > self.ub):
            raise ValueError('The parmeters exceed the upper bound!')
        elif any(par < self.lb):
            raise ValueError('The parmeters exceed the lower bound!')
        else:
            return np.array(par)
    
    def fit(self, data, weights=None, init_par=None, tol=1e-5, disp=False):
        fit_dist, loglik = None, -np.inf
        for dist in self.com_dist:
            d = dist()
            d.fit(data, weights=weights, init_par=init_par, 
                  gtol=tol, disp=disp)
            if d.monitor.loglik > loglik:
                fit_dist, loglik = d, d.monitor.loglik
        self.dist = fit_dist
        self.monitor = self.dist.monitor
    
    
        

    
