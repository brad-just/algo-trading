#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:19:26 2020

This file contains a class for running montecarlo simulations
on series of financial returns

@author: bradjust
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MonteCarlo(object):
    '''
    Run a montecarlo simulation on stock returns
    that generates geometric progressions of the 
    returns that can be aggregated and used in 
    assessing risk.
    '''
    def __init__(self, rets):
        self.rets = np.array(rets)
        self.result = None
    
    def run(self, capital_base=1.0, nsims=1, replace=True):
        '''
        Run the simulation and save the results in
        self.result.
        '''
        self.capital_base = capital_base
        
        if(replace):  
            self.result = pd.DataFrame(np.random.choice(self.rets, size=(len(self.rets), nsims)))
        else:
            result = []
            for _ in range(nsims):
                sim = np.random.choice(self.rets, size=len(self.rets), replace=False)
                result.append(sim)
            self.result = pd.DataFrame(result).T
    
    def stats(self, goal, bust):
        '''
        Generate descriptive statistics for the simulation.
        '''
        if(self.result is None):
            print('Simulation needs to be run before descriptive statistics can be generated')
            return
        
        shift_mat = np.ones(self.result.shape)
        shift_mat[0] *= self.capital_base
        geom_growth = (self.result + shift_mat).cumprod()
        progression_res =  geom_growth.iloc[-1]
        
        stats = {}
        
        maxdd = round(geom_growth.apply(self._calc_maxdd).min(),3)
        mean = round((np.mean(progression_res)-self.capital_base)/self.capital_base,3)
        
        stats['min'] = round((np.min(progression_res)-self.capital_base)/self.capital_base,3)
        stats['max'] = round((np.max(progression_res)-self.capital_base)/self.capital_base,3)
        stats['median'] = round((np.median(progression_res)-self.capital_base)/self.capital_base,3)
        stats['mean'] = mean
        stats['std'] = round(progression_res.std(),3)
        stats['maxdd'] = maxdd
        stats['goal'] = round((progression_res >= (1.0+goal)*self.capital_base).sum() / len(progression_res),3)
        stats['bust'] = round((progression_res <= (1.0+bust)*self.capital_base).sum() / len(progression_res),3)
        stats['calmar'] = round(mean/abs(maxdd),3)
        
        return(stats)
    
    def plot(self, figsize=(15,7)):
        '''
        Plot the resulting walks from the simulation.
        '''
        if(self.result is None):
            print('Simulation needs to be run before the results can be plotted')
            return
        
        shift_mat = np.ones(self.result.shape)
        shift_mat[0] *= self.capital_base
        geom_growth = (self.result + shift_mat).cumprod()
        
        plt.style.use('seaborn')
        plt.figure(figsize=figsize)
        
        plt.plot(geom_growth)
        plt.title('Results of Montecarlo Simulation')
        plt.xlabel('Observation')
        plt.ylabel('Progression')
        
        plt.show()
    
    @staticmethod
    def _calc_maxdd(ret_series):
        '''
        Calculate the maximum drawdown of a returns series
        '''
        cur_max = 0
        maxdd = 0
        for r in ret_series:
            cur_max = max(cur_max, r)
            maxdd = max(maxdd, abs((r-cur_max)/cur_max))
        return(-maxdd)
