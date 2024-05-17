#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 16:34:01 2020

Copied from Quantopian Algorithm.

@author: bradjust
"""

import pandas as pd
import numpy as np
from pykalman import KalmanFilter


class KalmanRegression(object):
    """
    Uses a Kalman Filter to estimate regression parameters 
    in an online fashion.
    
    Estimated model: y ~ beta * x + alpha
    """
    
    def __init__(self, initial_y, initial_x, delta=1e-5, maxlen=3000):
        self._x = initial_x.name
        self._y = initial_y.name
        self.maxlen = maxlen
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_mat = np.expand_dims(
            np.vstack([[initial_x], [np.ones(initial_x.shape[0])]]).T, axis=1)
        
        self.kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                               initial_state_mean=np.zeros(2),
                               initial_state_covariance=np.ones((2, 2)),
                               transition_matrices=np.eye(2),
                               observation_matrices=obs_mat,
                               observation_covariance=1.0,
                               transition_covariance=trans_cov)
        state_means, state_covs = self.kf.filter(initial_y.values)
        self.means = pd.DataFrame(state_means, 
                                  index=initial_y.index, 
                                  columns=['beta', 'alpha'])
        self.state_cov = state_covs[-1]
        
    def update(self, observations):
        x = observations[self._x]
        y = observations[self._y]
        mu, self.state_cov = self.kf.filter_update(
            self.state_mean, self.state_cov, y, 
            observation_matrix=np.array([[x, 1.0]]))
        mu = pd.Series(mu, index=['beta', 'alpha'], 
                       name=observations.name)
        self.means = self.means.append(mu)
        if self.means.shape[0] > self.maxlen:
            self.means = self.means.iloc[-self.maxlen:]
        
    def get_spread(self, observations):
        x = observations[self._x]
        y = observations[self._y]
        return y - (self.means.beta[-1] * x + self.means.alpha[-1])
        
    @property
    def state_mean(self):
        return self.means.iloc[-1]