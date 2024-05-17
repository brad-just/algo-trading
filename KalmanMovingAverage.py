#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 16:32:14 2020

Copied from Quantopian Algorithm.

@author: bradjust
"""

import pandas as pd
from pykalman import KalmanFilter


class KalmanMovingAverage(object):
    """
    Estimates the moving average of a price process 
    via Kalman Filtering. 
    
    See http://pykalman.github.io/ for docs on the 
    filtering process. 
    """
    
    def __init__(self, asset, observation_covariance=1.0, initial_value=0,
                 initial_state_covariance=1.0, transition_covariance=0.05, 
                 initial_window=20, maxlen=3000, freq='1d'):
        
        self.asset = asset
        self.freq = freq
        self.initial_window = initial_window
        self.maxlen = maxlen
        self.kf = KalmanFilter(transition_matrices=[1],
                               observation_matrices=[1],
                               initial_state_mean=initial_value,
                               initial_state_covariance=initial_state_covariance,
                               observation_covariance=observation_covariance,
                               transition_covariance=transition_covariance)
        self.state_means = pd.Series([self.kf.initial_state_mean], name=self.asset)
        self.state_vars = pd.Series([self.kf.initial_state_covariance], name=self.asset)
        
        
    def update(self, observations):
        for dt, observation in observations.iteritems():
            self._update(dt, observation)
        
    def _update(self, dt, observation):
        mu, cov = self.kf.filter_update(self.state_means.iloc[-1],
                                        self.state_vars.iloc[-1],
                                        observation)
        self.state_means[dt] = mu.flatten()[0]
        self.state_vars[dt] = cov.flatten()[0]
        if self.state_means.shape[0] > self.maxlen:
            self.state_means = self.state_means.iloc[-self.maxlen:]
        if self.state_vars.shape[0] > self.maxlen:
            self.state_vars = self.state_vars.iloc[-self.maxlen:]