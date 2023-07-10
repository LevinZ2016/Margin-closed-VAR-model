# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:50:43 2022

@author: Levin
"""
#%% 

import numpy as np
import scipy as sp

from .block_toeplitz_corr import (BlockToeplitzCorr, BlockToeplitz)

#%%

def vec(A):
    return A.T.ravel()

def dvec(v, n):
    return np.reshape(v, [n,-1], order='F')

def comm_mat(m,n):
    A = np.reshape(np.arange(m*n), (m,n), order='F')
    w = np.reshape(A.T, m*n, order='F')
    M = np.eye(m*n)
    M = M[w,:]
    return M

#%%

class MarginBlockCorr:
    """
    A class to compute matrix G and H for a given block Toeplitz matrix

    Methods
    -------
    get_coefs: 
        get matrices \PHI
    get_coefs_inv: 
        get matrices \PSI
    get_shift_mat:
        get matrix G
    get_shift_mat_inv:
        get matrix H

    """
    
    def __init__(self, R, dim, k):
        assert R.shape[0] == dim * (k+1), 'Wrong dimension of the input Toeplitz matrix.'
        self.R = R
        self.k = k
        self.dim = dim
        
    def get_coefs(self):
        R12 = self.R[:self.dim, self.dim:]
        R22 = self.R[self.dim:, self.dim:]
        return np.linalg.solve(R22.T, R12.T).T
    
    def get_coefs_inv(self):
        R11 = self.R[:-self.dim, :-self.dim]
        R21 = self.R[-self.dim:, :-self.dim]
        return np.linalg.solve(R11.T, R21.T).T
    
    def get_shift_mat(self):
        coefs = self.get_coefs() 
        row_mat = np.block([np.zeros([self.dim, self.dim*(self.k-1)]), 
                            -np.eye(self.dim), 
                            coefs, 
                            np.zeros([self.dim, self.dim])])
        m = np.empty([self.k*self.dim, (2*self.k+1)*self.dim])
        for i in range(self.k):
            m[(i*self.dim):((i+1)*self.dim)] = np.roll(row_mat, 
                                                       -self.dim*i, 
                                                       axis=1)
        return m
    
    def get_shift_mat_inv(self):
        coefs_inv = self.get_coefs_inv()
        row_mat = np.block([np.zeros([self.dim, self.dim*self.k]),
                            coefs_inv, 
                            -np.eye(self.dim)])
        m = np.empty([self.k*self.dim, (2*self.k+1)*self.dim])
        for i in range(self.k):
            m[(i*self.dim):((i+1)*self.dim)] = np.roll(row_mat, 
                                                       -self.dim*i, 
                                                       axis=1)
        return m

#%%

class PairClosedCorr():
    """
    A class to get the correlation between two series that makes two series margin-closed,
    given the correlation matrice of two sereis and the cross-setional correlation parameters.

    ...

    Attributes
    -------
    R1 : 
        correlation matrice of the first series
    dim1 : 
        dimension of the first series
    R2 : 
        correlation matrice of the second series
    dim2 : 
        dimension of the second series
    k : 
        Markov order of two series

    Methods
    -------
    get_link_mat: 
        get correlation matrix between two series

    """

    @staticmethod
    def get_link_block_toeplitz(blocks, dim, l):
        m = []
        for i in range(l-1, -1, -1):
            m.append(blocks[:, (i*dim):((i+l)*dim)])
        return np.vstack(m)
    
    def __init__(self, R1, R2, dim1, dim2, k):
        self.R1 = R1
        self.R2 = R2
        self.dim1 = dim1
        self.dim2 = dim2
        self.k = k
        
    def get_sigma_fr_par(self, par, partial=False):
        if partial:
            corr1 = self.R1[:self.dim1, :self.dim1]
            corr2 = self.R2[:self.dim2, :self.dim2]
            pSigma = np.reshape(par, [self.dim1, self.dim2])
            pcorr1 = BlockToeplitz.cor2pcor(corr1)
            pcorr2 = BlockToeplitz.cor2pcor(corr2)
            pcorr = np.block([[pcorr1, pSigma],
                              [pSigma.T, pcorr2]])
            corr = BlockToeplitz.pcor2cor(pcorr)
            return corr[:self.dim1, -self.dim2:]
        else:
            return np.reshape(par, [self.dim1, self.dim2])
    
    def get_link_mat(self, Sigma, type_arr):
        mg1 = MarginBlockCorr(self.R1, self.dim1, self.k)
        mg2 = MarginBlockCorr(self.R2, self.dim2, self.k)
        
        J = np.rot90(np.eye(2*self.k+1))
        J = np.kron(J, np.eye(self.dim1))
        
        if type_arr[0] != type_arr[1]:
            S = np.zeros([self.k+1, self.k+1])
            S[-1][0] = 1
            return np.kron(S, Sigma) if type_arr[0] == 0 else np.kron(S.T, Sigma)
        elif type_arr[0] == 0:
            m1 = np.dot(mg1.get_shift_mat(), J)
            m2 = mg2.get_shift_mat()
        else:
            m1 = np.dot(mg1.get_shift_mat_inv(), J)
            m2 = mg2.get_shift_mat_inv()
        
        A = np.block([m1[:, :(self.k * self.dim1)], 
                      m1[:, -(self.k * self.dim1):]])
        B = np.block([m2[:, :(self.k * self.dim2)], 
                      m2[:, -(self.k * self.dim2):]])
        
        Ha = -m1[:, (self.k * self.dim1):((self.k+1) * self.dim1)]
        Hb = -m2[:, (self.k * self.dim2):((self.k+1) * self.dim2)]
        
        K12 = comm_mat(self.dim1, self.dim2)
        M = np.block([[np.dot(np.kron(A, np.eye(self.dim2)), 
                              np.kron(np.eye(2*self.k), K12))],
                      [np.kron(B, np.eye(self.dim1))]])
        
        H = np.block([[np.dot(np.kron(Ha, np.eye(self.dim2)), K12)],
                      [np.kron(Hb, np.eye(self.dim1))]])
        
        v = np.linalg.solve(M, np.dot(H, vec(Sigma)))
        D = dvec(v, self.dim1)
        D = np.hstack([D[:, :(self.k * self.dim2)], 
                       Sigma, 
                       D[:, -(self.k * self.dim2):]])
        S = self.get_link_block_toeplitz(D, self.dim2, self.k+1)
        return S
    
    def get_link_mat_fr_par(self, par, type_arr, partial=False):
        Sigma = self.get_sigma_fr_par(par, partial=partial)
        return self.get_link_mat(Sigma, type_arr)
    
    
class MultiClosedCorr:
    """
    A class to get the correlation matirx that makes multiple series margin-closed,
    given the correlation matrice of all sereis and the cross-setional correlation 
    parameters between them.
    (The input with postfix 'dict' means data type of dictionary)
    
    ...

    Attributes
    -------
    R_arr : 
        arry of correlation matrice of all series
    dim_arr : 
        arry of dimensions of all series
    k : 
        Markov order

    Methods
    -------
    get_corr_fr_link_mat: 
        get the margin-closed correlation matrix covering all series given the 
        correlation matrix between all pairs of series.
    get_corr_fr_par:
        get the margin-closed correlation matrix covering all series given the 
        cross-sectional correlation parameters between all pairs of series.
    """

    @staticmethod
    def get_cross_par_dim(dim_arr):
        if len(dim_arr) == 2:
            return dim_arr[0] * dim_arr[1]
        else:
            return dim_arr[0] * sum(dim_arr[1:]) \
                + MultiClosedCorr.get_cross_par_dim(dim_arr[1:])
        
    def __init__(self, R_arr, dim_arr, k):
        self.R_arr = R_arr
        self.dim_arr = dim_arr 
        self.num_blocks = len(dim_arr)
        self.k = k
        self.cross_par_dim = self.get_cross_par_dim(dim_arr)
        self.uni_par_dim = sum([dim**2*k + dim*(dim-1)//2 for dim in dim_arr])
    
    def get_link_mat_fr_par_dict(self, par_dict, type_arr, partial=False):
        link_mat_dict = {}
        for i in range(self.num_blocks):
            for j in range(i+1, self.num_blocks):
                type_tuple = (type_arr[i], type_arr[j])
                pair_corr = PairClosedCorr(self.R_arr[i], self.R_arr[j], 
                                           self.dim_arr[i], self.dim_arr[j], 
                                           self.k)
                link_mat_dict[(i,j)] \
                    = pair_corr.get_link_mat_fr_par(par_dict[(i,j)], 
                                                    type_tuple,
                                                    partial=partial)
        return link_mat_dict
                
    def get_corr_fr_link_mat(self, link_mat_dict):
        rR = np.empty([sum(self.dim_arr) * (self.k+1), 
                       sum(self.dim_arr) * (self.k+1)])
        index_arr, cum_dimi_ = [], 0
        
        for i in range(self.num_blocks):
            dimi = self.dim_arr[i]
            cum_dimi = cum_dimi_ + dimi * (self.k+1)
            cum_dimj_ = cum_dimi
            rR[cum_dimi_:cum_dimi, cum_dimi_:cum_dimi] = self.R_arr[i]
            
            for j in range(i+1, self.num_blocks):
                dimj = self.dim_arr[j]
                cum_dimj = cum_dimj_ + dimj * (self.k+1)
                
                rR[cum_dimi_:cum_dimi, cum_dimj_:cum_dimj] \
                                                = link_mat_dict[(i,j)]
                rR[cum_dimj_:cum_dimj, cum_dimi_:cum_dimi] \
                                                = link_mat_dict[(i,j)].T
                cum_dimj_ = cum_dimj
            
            index_i = np.reshape(np.arange(cum_dimi_, cum_dimi), [-1, dimi])
            index_arr.append(index_i)
            cum_dimi_ = cum_dimi
        
        index = np.block(index_arr).ravel()
        R = rR[index, :][:, index]
        return rR, R
    
    def get_corr_fr_par_dict(self, par_dict, type_arr, partial=False):
        link_mat_dict = self.get_link_mat_fr_par_dict(par_dict, 
                                                      type_arr, 
                                                      partial=partial)
        rR, R = self.get_corr_fr_link_mat(link_mat_dict)
        return rR, R
    
    def get_corr_fr_par(self, par, type_arr, partial=False):
        par_dict = self.get_par_dict(par)
        return self.get_corr_fr_par_dict(par_dict, type_arr, partial=partial)
                
    def get_par_dict(self, par):
        par_dict, left_index = {}, 0
        for i in range(1, self.num_blocks):
            for j in range(self.num_blocks-i):
                ind1, ind2 = j, i+j
                dim1, dim2 = self.dim_arr[ind1], self.dim_arr[ind2]
                right_index = left_index + dim1 * dim2
                par_dict[(ind1, ind2)] = np.array(par[left_index:right_index])
                left_index = right_index
        return par_dict
    
    def get_par_fr_dict(self, par_dict):
        par = np.array([])
        for i in range(1, self.num_blocks):
            for j in range(self.num_blocks-i):
                ind1, ind2 = j, i+j
                par = np.append(par, par_dict[(ind1, ind2)])
        return par
    
#%%
    
class FullyClosedCorr(MultiClosedCorr):
    """
    A class to get the correlation matirx that makes all univariate series margin-closed 
    with Condition 2 (non-zero off-diagonal entries of coefficient matrices) in the paper ,
    given the correlation matrice of all sereis and the cross-setional correlation parameters 
    between them.
    
    ...

    Attributes
    -------
    uni_corr_arr : array of instances of BlockToeplitzCorr
        arry of correlation matrice of all univariate series
    dim : 
        the number of univariate series
    k : 
        Markov order
    _par_arr :
        a dim * dim array containing the contemporanous cross-sectional correlation
    _par :
        a 1-d array containing the contemporanous cross-sectional correlation

    Methods
    -------
    get_corr_mat: 
        get the margin-closed correlation matrix
    check_corr_mat:
        check the postive definiteness
    set_cross_par:
        set a diagonal (diag_ind) of _par_arr
        (it enabales _par_arr be fitted from main diagonal to right-up corner sequensially)
    
    """

    def __init__(self, uni_corr_arr, dim, k, par_arr=None):
        super().__init__(None, [1]*dim, k)
        self.dim = dim
        self.type_arr = [1] * dim
        if par_arr is None:
            self._par, self._par_arr = None, None
        else:
            self._par_arr = par_arr
            self._par = self.get_par_fr_arr(par_arr)
        self.uni_corr_arr = uni_corr_arr
    
    @property
    def par_arr(self):
        return self._par_arr
    
    @par_arr.setter
    def par_arr(self, arr):
        if self._par_arr is None:
            self.set_link_mat()
        self._par_arr = arr
        self._par = self.get_par_fr_arr(arr)
        
    @property
    def par(self):
        return self._par
    
    @par.setter
    def par(self, par):
        if self._par is None:
            self.set_link_mat()
        self._par = par
        self._par_arr = self.get_par_arr(par)
        
    @property
    def uni_corr_arr(self):
        return self._uni_corr_arr
    
    @uni_corr_arr.setter
    def uni_corr_arr(self, uni_corr_arr):
        self.serial_par_dim = 0
        self._R_arr = []
        for corr in uni_corr_arr:
            assert (corr.dim == 1) and (corr.k <= self.k)
            self._R_arr.append(corr.extend_corr(self.k).R)
            self.serial_par_dim += corr.k
        self._uni_corr_arr = uni_corr_arr
        if self.par_arr is not None:
            self.set_link_mat()
            
    @property
    def R_arr(self):
        return self._R_arr
    
    @R_arr.setter
    def R_arr(self, input):
        if input is not None:
            raise Exception('R_arr cannot changed directly!')
            
    def set_single_corr(self, par, index):
        self._uni_corr_arr[index].par = par
        self._R_arr[index] = \
            self.uni_corr_arr[index].extend_corr(self.k).R
        if self.par_arr is not None:
            self.set_link_mat()
            
    def set_corr_par(self, par):
        c1 = 0
        self._R_arr = []
        for corr in self._uni_corr_arr:
            c2 = corr.k + c1
            corr.par = par[c1:c2]
            self._R_arr.append(corr.extend_corr(self.k).R)
            c1 = c2
        if self.par_arr is not None:
            self.set_link_mat()
    
    def get_corr_par(self):
        par = []
        for corr in self._uni_corr_arr:
            par = np.append(par, corr.par)
        return np.array(par)
    
    def set_cross_par(self, diag_par, diag_ind):
        par_arr = self.par_arr
        for j in range(self.dim-diag_ind):
            par_arr[j, j+diag_ind] = diag_par[j]
        self.par_arr = par_arr
    
    def get_cross_par(self, diag_ind):
        start_ind = int((self.dim - diag_ind/2)*(diag_ind-1))
        return self._par[start_ind : (start_ind+self.dim-diag_ind)]
    
    def get_par_arr(self, par):
        par_arr = np.zeros([self.dim]*2)
        count = 0
        for i in range(1, self.dim):
            for j in range(self.dim-i):
                ind1, ind2 = j, i+j
                par_arr[ind1, ind2] = par[count]
                count += 1
        return par_arr
        
    def get_par_fr_arr(self, par_arr):
        par = []
        for i in range(1, self.dim):
            for j in range(self.dim-i):
                ind1, ind2 = j, i+j
                par.append(par_arr[ind1, ind2])
        return np.array(par)
        
    def set_link_mat(self):
        temp_par = np.ones(self.cross_par_dim)
        temp_par_dict = self.get_par_dict(temp_par)
        self.link_mat_dict \
            = self.get_link_mat_fr_par_dict(temp_par_dict, 
                                        self.type_arr)
            
    def get_uni_corr_mat(self):
        return self.R_arr
            
    def get_corr_mat(self):
        R = sp.linalg.block_diag(*self.R_arr)
        if self.par_arr is not None:
            for i in range(1, self.dim):
                for j in range(self.dim - i):
                    r1, r2 = j*(self.k+1), (j+1)*(self.k+1)
                    c1, c2 = (i+j)*(self.k+1), (i+j+1)*(self.k+1), 
                    R[r1:r2, c1:c2] = \
                        self.link_mat_dict[(j, i+j)] * self.par_arr[j, i+j]
                    R[c1:c2, r1:r2] = \
                        self.link_mat_dict[(j, i+j)].T * self.par_arr[j, i+j]
        
        index = np.arange(self.k+1)[:, None] \
            + np.arange(self.dim) * (self.k+1)
        index = index.ravel()
        R = R[index, :][:, index]
        return R
    
    def get_stat_corr(self):
        R = self.get_corr_mat()
        return BlockToeplitzCorr(self.dim, self.k, R)
    
    def check_corr_mat(self):
        R = self.get_corr_mat()
        try:
            np.linalg.cholesky(R)
        except:
            return False
        return True
    
        
       
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



# %%
