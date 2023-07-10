
#%%

import numpy as np
import scipy.stats as st
from copy import deepcopy
from scipy.optimize import (minimize, Bounds)

from margin_closed_corr import (FullyClosedCorr, BlockToeplitzCorr)
import margins.univariate_margins as umg
from margins.multi_variable_margins import MarginList


#%%

class MCVAR():
    """
    The class of marign-closed VAR model (non-zero off-diagonal entries in coefficient matrices), 
    with non-Gaussian margins incorporated.
    
    ...

    Attributes
    -------
    dim : 
        dimension of the model
    k : 
        Markov order
    mdist: a list of or a single univaraite distribution class
        uinvariate marginal distributions 

    Methods
    -------
    fit_margins: 
        fit univariate margin of each univariate component
    fit_uni_corr:
        fit stationary correlation matrix of each univariate component
    fit_cross_corr:
        fit cross-sectional correlations between univariate components
    fit_corr:
        fit stationary correlation matrix of each univariate component and 
        cross-sectional correlations simutanously
    fit:
        fit all parameters of the latent VAR process
    """

    def __init__(self, dim, k, mdist=umg.Gaussian):
        self.dim = dim
        self.k = k
        self.mglist, self.mc_corr = None, None
        if not isinstance(mdist, list):
            self.mdists = [mdist] * self.dim
        else:
            self.mdists = mdist

    def get_cloglik_fr_par(self, cpar, zdata):
        d1 = self.mc_corr.uni_par_dim
        copy_model = deepcopy(self)
        copy_model.mc_corr.set_corr_par(cpar[:d1])
        copy_model.mc_corr.par = cpar[d1:]
        return copy_model.get_copula_loglik(zdata)

    def get_cloglik_fr_cross_par(self, diag_par, diag_ind, zdata):
        copy_model = deepcopy(self)
        copy_model.mc_corr.set_cross_par(diag_par, diag_ind)
        return copy_model.get_copula_loglik(zdata)

    def get_zdata(self, data):
        zdata = st.norm.ppf(self.mglist.cdf(data))
        mg_loglik = self.mglist.logpdf(data).sum()
        return zdata, mg_loglik

    def get_copula_loglik(self, zdata):
        assert self.mc_corr.par_arr is not None, 'No cross-sectional dependence pars!'
        if self.mc_corr.check_corr_mat():
            stat_corr = self.mc_corr.get_stat_corr()
            cloglik = stat_corr.get_likelihood(zdata) \
                        - st.norm.logpdf(zdata).sum()
            return cloglik
        else:
            return -1e6

    def get_loglik(self, data):
        zdata, mg_loglik = self.get_zdata(data)
        cloglik = self.get_copula_loglik(zdata)
        return cloglik + mg_loglik

    def fit_margins(self, data):
        margin_arr = []
        for i in range(self.dim):
            margin_arr.append(self.mdists[i]())
            margin_arr[-1].fit(data[:,i])
        self.mglist = MarginList(self.dim, margin_arr)

    def fit_uni_corr(self, zdata):
        uni_corr_arr = []
        for i in range(self.dim):
            uni_corr_arr.append(BlockToeplitzCorr(dim=1, k=self.k))
            uni_corr_arr[-1].fit_corr(zdata[:,[i]])
        self.mc_corr = FullyClosedCorr(uni_corr_arr, self.dim, self.k)

    def fit_cross_corr(self, zdata):
        self.mc_corr.par_arr = np.eye(self.dim)
        for j in range(1, self.dim):
            target_f = lambda par: -self.get_cloglik_fr_cross_par(par, j, zdata)
            lb = [-0.99]*(self.dim - j)
            ub = [0.99]*(self.dim - j)
            x0 = [0.]*(self.dim - j)
            b = Bounds(lb=lb, ub=ub, keep_feasible=True)
            sol = minimize(target_f, x0, method='L-BFGS-B', jac='3-point',
                            bounds=b, options={'gtol': 1e-5, 'disp': False})
            self.mc_corr.set_cross_par(sol.x, j)

    def fit_corr(self, zdata):
        self.fit_uni_corr(zdata)
        self.fit_cross_corr(zdata)
        d1 = self.mc_corr.uni_par_dim 
        d2 = self.mc_corr.cross_par_dim
        target_f = lambda cpar: -self.get_cloglik_fr_par(cpar, zdata)
        lb, ub = [-0.99]*(d1+d2), [0.99]*(d1+d2)
        x0 = np.append(self.mc_corr.get_corr_par(), self.mc_corr.par)
        b = Bounds(lb=lb, ub=ub, keep_feasible=True)
        sol = minimize(target_f, x0, method='L-BFGS-B', jac='3-point',
                        bounds=b, options={'gtol': 1e-5, 'disp': False})
        self.mc_corr.set_corr_par(sol.x[:d1])
        self.mc_corr.par = sol.x[d1:]

    def fit(self, data):
        self.fit_margins(data)
        zdata, _ = self.get_zdata(data)
        self.fit_corr(zdata)

        self.num_par = self.mglist.par_dim \
                    + self.mc_corr.uni_par_dim \
                    + self.mc_corr.cross_par_dim
        
        self.loglik = self.get_loglik(data)
        self.aic = 2 * self.num_par - 2 * self.loglik
        self.bic = np.log(len(data)) * self.num_par - 2 * self.loglik


# %%

if __name__ == '__main__':

    from tabulate import tabulate
    import matplotlib.pyplot as plt
    import pandas as pd

    '''Import data'''
    var_names = {
        'DTCTHFNM': 'TLLO',
        'DPCERA3M086SBEA': 'PCE',
        'CPIAUCSL': 'CPI'
    }
    fred_data = pd.read_csv('fred_data.csv', index_col=0)
    mdata = fred_data[var_names.keys()]
    mdata = mdata.rename(columns=var_names).dropna()
    mdata = mdata.query('date >= \'1989-03-01\' and date <= \'2001-08-01\'') * 100

    '''Plot data'''
    plot_df = mdata.iloc[:, :3] * 100
    plot_df.columns = ['CLL', 'PCE', 'CPI']
    styles = ['--', '*-', '-']
    fig, ax = plt.subplots(figsize=(14,5))
    for col, style in zip(plot_df.columns, styles):
        plot_df[col].plot(style=style, ax=ax)
    plt.legend()
    plt.grid()
    plt.xlabel('Date')
    plt.ylabel('Percentage')
    plt.savefig('TimeSeriesPlots.pdf')

    '''Fit models of diiferent Markov orders'''
    data = mdata.values[:,:3]
    num_models = 5
    aic1, aic2 = [], []
    num_par1, num_par2 = [], []
    for k in range(1, num_models+1):
        mc_model = MCVAR(3, k, [umg.skew_t_ab, umg.Gaussian, umg.skew_t_ab])
        mc_model.fit(data)
        num_par1.append(mc_model.num_par)
        aic1.append(mc_model.aic)

        zdata, mg_loglik = mc_model.get_zdata(data)
        VAR_model = BlockToeplitzCorr(3, k)
        VAR_model.fit_corr(zdata)
        VAR_aic = VAR_model.aic + 2 * mc_model.mglist.par_dim \
                - 2*(mg_loglik - st.norm.logpdf(zdata).sum())
        num_par2.append(mc_model.mglist.par_dim + VAR_model.par_dim)
        aic2.append(VAR_aic)

    '''Organize the results in tables and wriet them into a txt file'''
    headers = ['Markov order'] + [str(k) for k in range(1, num_models+1)]
    par_table = [
        ['MC model'] + num_par1, ['Unrestricted model'] + num_par2
    ]
    aic_table = [
        ['MC model'] + aic1, ['Unrestricted model'] + aic2
    ]
    with open('results.txt', 'w') as f:
        f.write('Table of the number of parameters: \n')
        f.write(tabulate(par_table, headers, floatfmt='.1f', 
                         tablefmt='github', colalign=['center']))
        f.write('\n'*2)
        f.write('Table of AIC: \n')
        f.write(tabulate(aic_table, headers, floatfmt='.1f', 
                         tablefmt='github', colalign=['center']))
   