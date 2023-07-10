#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:55:11 2022

@author: levin
"""

import numpy as np
from copy import deepcopy

class MarginList():
    
    def __init__(self, var_dim, margin_arr=None):
        self.var_dim = var_dim
        self.margins = margin_arr
        if self.margins:
            self.set_margins(self.margins)
        
    def __repr__(self):
        if self.margins is None:
            return '[No distribution]'
        else:
            info_str = '['
            for m in self.margins:
                info_str += '{}, parameters = {}\n'.format(m.class_name,
                                                           m.par)
            return info_str + ']'

    def set_margins(self, margin_arr):
        assert self.var_dim == len(margin_arr), 'Wrong dimension of input margins.'
        self.margins = deepcopy(margin_arr)
        self.extract_par_info()
        self.extract_par()
        
    def check_data(self, data):
        if len(data.shape) == 1:
            data = data[None, :]
        assert data.shape[1] == self.var_dim
        return data
    
    def extract_par_info(self):
        count = -1
        lb, ub, par_list = [], [], []
        for m in self.margins:
            lb += m.lb.tolist()
            ub += m.ub.tolist()
            num = m.par_dim
            par_list.append((int(count) + np.arange(1,int(num)+1)).tolist())
            count += num
        self.par_dim = int(count + 1)
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.par_list = par_list
        
    def check_par(self, par):
        if len(par) != self.par_dim:
            raise ValueError('The dimension of the parameters is incorrect!')
        elif any(par > self.ub):
            raise ValueError('The parameters exceed the upper bound!')
        elif any(par < self.lb):
            raise ValueError('The parameters exceed the lower bound!')
        else:
            return np.array(par).ravel()
    
    def extract_par(self):
        par = []
        for m in self.margins:
            par += m.par.tolist()
        self._par = np.array(par)
    
    @property
    def par(self):
        return self._par
    
    @par.setter
    def par(self, input_par):
        par = self.check_par(input_par)
        count = 0
        for m in self.margins:
            m.par = par[count:(count+m.par_dim)]
            count += m.par_dim
        self.extract_par()
                    
    def copy_with_different_par(self, input_par):
        copy_copmg = MarginList(self.var_dim)
        copy_copmg.set_margins(self.margins)
        copy_copmg.par = input_par
        return copy_copmg
                    
    def logpdf(self, data):
        return self.logpdf_fr_input(data, self.par)
    
    def cdf(self, data):
        return self.cdf_fr_input(data, self.par)
    
    def ppf(self, data):
        return self.ppf_fr_input(data, self.par)
    
    def logpdf_fr_input(self, data, input_par):
        data = self.check_data(data)
        copy_margins = self.copy_with_different_par(input_par).margins
        logpdf = []
        for i, m in enumerate(copy_margins):
            logpdf.append(m.logpdf(data[:, i]))
        return np.vstack(logpdf).T
    
    def cdf_fr_input(self, data, input_par):
        data = self.check_data(data)
        copy_margins = self.copy_with_different_par(input_par).margins
        cdf = []
        for i, m in enumerate(copy_margins):
            cdf.append(m.cdf(data[:, i]))
        return np.vstack(cdf).T
    
    def ppf_fr_input(self, data, input_par):
        data = self.check_data(data)
        copy_margins = self.copy_with_different_par(input_par).margins
        ppf = []
        for i, m in enumerate(copy_margins):
            ppf.append(m.ppf(data[:, i]))
        return np.vstack(ppf).T
    