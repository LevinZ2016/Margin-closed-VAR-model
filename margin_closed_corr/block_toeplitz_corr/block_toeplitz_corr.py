# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:20:29 2022

@author: Levin
"""
#%%

import numpy as np
import scipy.stats as st
from scipy.optimize import (minimize, Bounds)
from copy import deepcopy

from .block_toeplitz import BlockToeplitz

#%%


class BlockToeplitzCorr(BlockToeplitz):    
    """
    A class to represent block Toeplitz correlation matrix, 
    it can be used to as a VAR model with standard Gaussian univariate margins

    ...
    
    Atrributes
    -------
    dim : int
        the dimension of square blocks
    k : int
        the number of non-diagnoal blocks
    par_dim : int
        the number of parameters of the block Toeplitz matrix
    R : array
        the block Toeplitz matrix represented as a nd-array

    Methods
    -------
    extend_corr_one: 
        adding a new block to the block Toeplitz correlation matrix by letting 
        partial correlations of order k+1 be zero
    extend_corr: 
        extend the block Toeplitz correlation matrix to a given number of blocks 
        by letting higher-order partial correlations be zero
    get_VAR_pars:
        get VAR representation coefficients and covariance of innovations by treating
        the block Toeplitz matrix as covariance of stationary distribution
    get_sub_corr:
        extract the block Toeplitz correlation matrix of a sub-vector
    fit_corr:
        fit VAR stationary distribution's  the block Toeplitz correlation matrix 
    get_start_likelihood:
        get log-likelihood of first k+1 observations
    get_main_likelihood:
        get log-likelihood of the other observations

    """

    @staticmethod
    def Gaussian_logpdf(X, R):
        d = R.shape[0]
        logf = - 1/2 * np.log(np.linalg.det(R)) \
                - 1/2 * (np.linalg.solve(R, X.T).T * X).sum(axis=1)
        return logf - d / 2 * np.log(2*np.pi)
    
    def __repr__(self):
        info_str = f"Block Toeplitz Correlation Matrix: dim = {self.dim}, k = {self.k}"
        return info_str
    
    def __init__(self, dim, k, R=None):
        super().__init__(dim, k)
        if R is not None:
            try:
                np.linalg.cholesky(R)
            except:
                raise Exception('Corr matrix is not positive definite!')
            self.R = R
        else:
            self.R = np.eye((k+1) * dim)
        
    @property
    def par(self):
        return self._par
    
    @par.setter
    def par(self, par):
        assert len(par) == self.par_dim, 'Dimension must match'
        self._par = par
        self._R = self.get_corr_fr_par(par)
    
    @property
    def R(self):
        return self._R
    
    @R.setter
    def R(self, R):
        assert R.shape[0] == (self.k+1)*self.dim, 'Dimension must match'
        self._R = R
        self._par = self.get_par_fr_corr(R)
        
    def get_corr_fr_par(self, par):
        return self.get_btm_fr_par(par)
    
    def get_par_fr_corr(self, R):
        return self.get_par_fr_btm(R)
    
    def extend_corr_one(self):
        diag_block = self.R[:self.dim, :self.dim]
        blocks_arr = self.R[:self.dim, self.dim:]
        sub_R1 = self.R[self.dim:, self.dim:]
        sub_R2 = self.R[:-self.dim, -self.dim:]
        extra_block = np.dot(blocks_arr, np.linalg.solve(sub_R1, sub_R2))
        blocks_arr = np.block([blocks_arr, extra_block])
        return BlockToeplitz(self.dim, self.k+1).get_block_toep_mat(diag_block, 
                                                                    blocks_arr,
                                                                    False)
    
    def extend_corr(self, target_k):
        btcorr = self
        for i in range(target_k - self.k):
            R = btcorr.extend_corr_one()
            btcorr = BlockToeplitzCorr(btcorr.dim, btcorr.k+1, R)
        return btcorr
    
    def get_pcorr(self):
        return self.cor2pcor(self.R)
    
    def get_sub_corr(self, ind):
        assert max(ind) < self.dim
        sub_ind = np.arange(self.k+1) * self.dim \
            + np.array(ind)[:, None]
        sub_ind = sub_ind.ravel(order='F')
        sub_R = self.R[sub_ind][:, sub_ind]
        return BlockToeplitzCorr(len(ind), self.k, sub_R)
    
    def get_VAR_pars(self):
        R12 = self.R[:self.dim, self.dim:]
        R21 = self.R[self.dim:, :self.dim]
        R11 = self.R[:self.dim, :self.dim]
        R22 = self.R[self.dim:, self.dim:]
        sigma = R11 - np.matmul(R12, np.linalg.solve(R22, R21))
        coef = np.linalg.solve(R22.T, R12.T).T
        mcoefs = [arr.T for arr in coef.T.reshape((-1, self.dim, self.dim))]
        return np.array(mcoefs), sigma
        
    def get_stack_data(self, X):
        stack_X = np.empty([len(X)-self.k, (self.k+1)*self.dim])
        for i in range(self.k+1):
            inv_X = X[::-1]
            stack_X[:, (i*self.dim):((i+1)*self.dim)] \
                = inv_X[i:(len(X)-self.k+i)]
        return stack_X
        
    def get_start_likelihood(self, X):
        start_X = X[-self.k:]
        x = start_X[::-1].ravel()
        sub_R = self.R[:(self.k * self.dim), :(self.k * self.dim)]
        try:
            return st.multivariate_normal.logpdf(x, cov=sub_R)
        except:
            print(', '.join(str(x) for x in self.get_par_fr_btm(self.R)))
    
    def get_main_likelihood(self, X):
        stack_X = self.get_stack_data(X)
        z1, z2 = stack_X[:,:self.dim], stack_X[:,self.dim:]
        R11 = self.R[:self.dim, :self.dim]
        R22 = self.R[self.dim:, self.dim:]
        R12 = self.R[:self.dim, self.dim:]
        R21 = self.R[self.dim:, :self.dim]
        R11_2 = R11 - np.dot(R12, np.linalg.solve(R22, R21))
        mu11_2 = np.dot(R12, np.linalg.solve(R22, z2.T)).T
        try:
            return  st.multivariate_normal.logpdf(z1 - mu11_2, cov=R11_2).sum()
        except:
            print(', '.join(str(x) for x in self.get_par_fr_btm(self.R)))
    
    def get_likelihood(self, X):
        n = len(X)
        if  n > self.k:
            return self.get_start_likelihood(X) + self.get_main_likelihood(X)
        else:
            sub_R = self.R[:(self.dim * n), :(self.dim * n)]
            return st.multivariate_normal.logpdf(X[::-1].ravel(), cov=sub_R)
    
    def fit_corr(self, X):
        init_R = np.corrcoef(self.get_stack_data(X).T)
        init_par = self.get_par_fr_corr(init_R)
        
        def f_corrs(par):
            sub_model = deepcopy(self)
            sub_model.R = self.get_corr_fr_par(par)
            return -sub_model.get_likelihood(X)
        
        b_corrs = Bounds(ub=np.minimum(init_par + 0.5*np.abs(init_par), 0.99), 
                         lb=np.maximum(init_par - 0.5*np.abs(init_par), -0.99), 
                         keep_feasible=True)
        sol_corrs = minimize(f_corrs, init_par, bounds=b_corrs, 
                             method='L-BFGS-B', jac='3-point',
                             options={'gtol': 1e-5, 'disp': False})
        
        self.R = self.get_corr_fr_par(sol_corrs.x)
        self.aic = 2*len(init_par) + 2*sol_corrs.fun
        self.bic = len(init_par)*np.log(len(X)) + 2*sol_corrs.fun



   
# %%
