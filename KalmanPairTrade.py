#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 16:36:54 2020

Copied from Quantopian Algorithm.

@author: bradjust
"""

from zipline.api import (
    get_open_orders,
    get_datetime,
    order_target,
    order_target_percent
    )
import numpy as np
import pandas as pd
from KalmanMovingAverage import KalmanMovingAverage
from KalmanRegression import KalmanRegression


class KalmanPairTrade(object):

    def __init__(self, y, x, leverage=1.0, initial_bars=10, 
                 freq='1d', delta=1e-3, maxlen=3000):
        self._y = y
        self._x = x
        self.maxlen = maxlen
        self.initial_bars = initial_bars
        self.freq = freq
        self.delta = delta
        self.leverage = leverage
        self.Y = KalmanMovingAverage(self._y, maxlen=self.maxlen)
        self.X = KalmanMovingAverage(self._x, maxlen=self.maxlen)
        self.kf = None
        self.entry_dt = pd.Timestamp('1900-01-01', tz='utc')
        
    @property
    def name(self):
        return "{}~{}".format(self._y.symbol, self._x.symbol)

    def trading_logic(self, context, data):
        #try:
        if self.kf is None:
            self.initialize_filters(context, data)
            return
        self.update(context, data)
        if get_open_orders(self._x) or get_open_orders(self._y):
            return
        spreads = self.mean_spread()

        zscore = (spreads.iloc[-1] - spreads.mean()) / spreads.std()

        reference_pos = context.portfolio.positions[self._y].amount

        now = get_datetime()
        if reference_pos:
            if (now - self.entry_dt).days > 50:
                order_target(self._y, 0.0)
                order_target(self._x, 0.0)
                return
            # Do a PNL check to make sure a reversion at least covered trading costs
            # I do this because parameter drift often causes trades to be exited
            # before the original spread has become profitable.
            pnl = self.get_pnl(context, data)
            if zscore > -0.0 and reference_pos > 0 and pnl > 0:
                order_target(self._y, 0.0)
                order_target(self._x, 0.0)

            elif zscore < 0.0 and reference_pos < 0 and pnl > 0:
                order_target(self._y, 0.0)
                order_target(self._x, 0.0)

        else:
            if zscore > 1.5:
                order_target_percent(self._y, -self.leverage / 2.)
                order_target_percent(self._x, self.leverage / 2.)
                self.entry_dt = now
            if zscore < -1.5:
                order_target_percent(self._y, self.leverage / 2.)
                order_target_percent(self._x, -self.leverage / 2.)
                self.entry_dt = now
       # except Exception as e:
        #    print("[{}] {}".format(self.name, str(e)))

    def update(self, context, data):
        prices = np.log(data.history([self._x, self._y], 'price', 1, self.freq))
        self.X.update(prices[self._x])
        self.Y.update(prices[self._y])
        self.kf.update(self.means_frame().iloc[-1])

    def mean_spread(self):
        means = self.means_frame()
        beta, alpha = self.kf.state_mean
        return means[self._y] - (beta * means[self._x] + alpha)


    def means_frame(self):
        mu_Y = self.Y.state_means
        mu_X = self.X.state_means
        return pd.DataFrame([mu_Y, mu_X]).T

            
    def initialize_filters(self, context, data):
        prices = np.log(data.history([self._x, self._y], 'price', self.initial_bars, self.freq))
        
        self.X.update(prices[self._x])
        self.Y.update(prices[self._y])

        # Drops the initial 0 mean value from the kalman filter
        self.X.state_means = self.X.state_means.iloc[-self.initial_bars:]
        self.Y.state_means = self.Y.state_means.iloc[-self.initial_bars:]
        self.kf = KalmanRegression(self.Y.state_means, self.X.state_means,
                                   delta=self.delta, maxlen=self.maxlen)
    
    def get_pnl(self, context, data):
        x = self._x
        y = self._y
        prices = data.history([x, y], 'price', 1, '1d').iloc[-1]
        positions = context.portfolio.positions
        dx = prices[x] - positions[x].cost_basis
        dy = prices[y] - positions[y].cost_basis
        return (positions[x].amount * dx +
                positions[y].amount * dy)