import numpy as np
import scipy.stats as st
from copy import deepcopy


#%%

class BlockToeplitz():
    """
    A class to represent block Toeplitz matrix

    ...
    
    Atrributes
    -------
    dim : int
        the dimension of square blocks
    k : int
        the number of non-diagnoal blocks
    par_dim : int
        the number of parameters of the block Toeplitz matrix

    Methods
    -------
    cor2pcor: 
        convert a correlation matrix to a partial correlation matrix (D-vine copula structure)
    cor2pcor: 
        convert a partial correlation matrix to a correlation matrix (D-vine copula structure)
    get_block_toep_mat:
        construct a block Toeplitz matrix given blocks
    get_btm_fr_par:
        construct a block Toeplitz matrix given parameters
    get_par_fr_btm
        get partial correlation parameters of a block Toeplitz matrix

    """
    
    @staticmethod
    def cor2pcor(R):
        d = R.shape[0]
        if d==2:
            return R
        
        P = np.eye(d)
        P[range(d-1), range(1,d)] = R[range(d-1), range(1,d)]
        P[range(1,d), range(d-1)] = R[range(d-1), range(1,d)]
        
        for i in range(2,d):
            for j in range(d-i):
                posi = (j, i+j)
                rho = R[posi]
                S11 = R[(j+1):(j+i), (j+1):(j+i)]
                S12 = R[(j+1):(j+i), (j, i+j)]

                omega = np.dot(S12.T, np.linalg.solve(S11, S12))
                p = (rho - omega[0,1]) / np.sqrt((1-omega[0,0]) * (1-omega[1,1]))
                P[posi] = p
                P[i+j, j] = p
        return P

    @staticmethod
    def pcor2cor(P):
        d = P.shape[0]
        if d==2:
            return P

        R = np.eye(d)
        R[range(d-1), range(1,d)] = P[range(d-1), range(1,d)]
        R[range(1,d), range(d-1)] = P[range(d-1), range(1,d)]
        for i in range(2,d):
            for j in range(d-i):
                posi = (j, i+j)
                p = P[posi]
                S11 = R[(j+1):(j+i), (j+1):(j+i)]
                S12 = R[(j+1):(j+i), (j, i+j)]
                
                omega = np.dot(S12.T, np.linalg.solve(S11, S12))
                rho = omega[0,1] + p * np.sqrt((1-omega[0,0]) * (1-omega[1,1]))
                
                R[posi] = rho
                R[i+j, j] = rho
        return R
    
    def __init__(self, dim, k):
        self.k = k
        self.dim = dim
        self.par_dim = self.dim*(self.dim-1) // 2 + self.dim**2 * self.k

    def get_block_toep_mat(self, diag_block, blocks_arr, partial=True):
        roll_mat = np.block([diag_block, blocks_arr])
        block_mat = np.empty([self.dim*(self.k+1), self.dim*(self.k+1)])
        for i in range(self.k+1):
            block_mat[(i*self.dim):((i+1)*self.dim), (i*self.dim):] \
                = roll_mat[:, :(self.k+1-i)*self.dim]
            block_mat[(i*self.dim):, (i*self.dim):((i+1)*self.dim)] \
                = roll_mat[:, :(self.k+1-i)*self.dim].T
        return self.pcor2cor(block_mat) if partial else block_mat
    
    def get_btm_fr_par(self, par):
        assert len(par) == self.par_dim, 'Wrong input dimension of pars.'
        diag_block = np.eye(self.dim) * 0.5
        diag_block[np.triu_indices(self.dim, 1)] = par[:(self.dim*(self.dim-1)//2)]
        diag_block = diag_block + diag_block.T
        blocks_arr = np.reshape(par[(self.dim*(self.dim-1)//2):], 
                                [self.dim,-1], 
                                order='F')
        return self.get_block_toep_mat(diag_block, blocks_arr)
    
    def get_par_fr_btm(self, R):
        pR = self.cor2pcor(R)
        return pR[np.tril_indices(n=self.dim*(self.k+1), m=self.dim, k=-1)]
    
    
# %%
