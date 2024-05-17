#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:27:21 2020

KalmanCapmArbitrage trading strategy class optimized for use with pylivetrader

@author: bradjust
"""

from pylivetrader_mod.api import (
    record,
    symbol,
    order_target,
    order_target_percent,
    get_datetime,
    get_open_orders
    )
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '/Users/bradjust/Desktop/trading/AlpacaAccount')
from KalmanMovingAverage import KalmanMovingAverage
from KalmanRegression import KalmanRegression

sys.path.insert(0, '/Users/bradjust/Desktop/trading/paper-strategies/StockSpecific')
from KalmanCAPMplt import KalmanCAPM

from logbook import Logger, StreamHandler
StreamHandler(sys.stdout).push_application()
log = Logger('KalmanCapmArbitrageClass')

class KalmanCAPMArbitrage(object):

    def __init__(self, y, leverage=1.0, initial_bars=10, 
                 freq='1d', delta=1e-3, maxlen=3000):
        self._y = y
        self._x = KalmanCAPM(y.symbol, initial_bars, leverage)
        self.Rf = symbol('IEF')
        self.maxlen = maxlen
        self.initial_bars = initial_bars
        self.freq = freq
        self.delta = delta
        self.leverage = leverage
        self.Y = KalmanMovingAverageMod(self._y, maxlen=self.maxlen)
        self.X = KalmanMovingAverageMod(self._x.symbol, maxlen=self.maxlen)
        self.kf = None
        self.entry_dt = pd.Timestamp('1900-01-01', tz='utc')
     
        
    @property
    def name(self):
        return "{}~{}".format(self._y.symbol, self._x.symbol)


    def trading_logic(self, context, data):
        log.debug('trading logic being executed')
  #      try:
        if self.kf is None:
            self.initialize_filters(context, data)
        else:
            self.update(context, data)
            
        if get_open_orders(self._y):
            return
        spreads = self.mean_spread()

        zscore = (spreads[-1] - spreads.mean()) / spreads.std()
        record(zscore=zscore)
        
        log.debug('zscore is: %s'%round(zscore,2))

        reference_pos = context.portfolio.positions[self._y].amount

        now = get_datetime()
        if reference_pos:
            if (now - self.entry_dt).days > 50:
                order_target(self._y, 0.0)
                order_target_percent(self.Rf, 1.0)
                return
            # Do a PNL check to make sure a reversion at least covered trading costs
            # I do this because parameter drift often causes trades to be exited
            # before the original spread has become profitable.
            pnl = self.get_pnl(context, data)
            if zscore > -0.0 and reference_pos > 0 and pnl > 0:
                order_target(self._y, 0.0)
                order_target_percent(self.Rf, 1.0)

            elif zscore < 0.0 and reference_pos < 0 and pnl > 0:
                order_target(self._y, 0.0)
                order_target_percent(self.Rf, 1.0)

        else:
            
            if zscore > 0.5:
                order_target_percent(self._y, 0.0)
                order_target_percent(self.Rf, 1.0)
                self.entry_dt = now
                
            if zscore < -0.5:
                order_target(self.Rf, 0.0)
                order_target_percent(self._y, self.leverage)
                self.entry_dt = now
                    
  #      except Exception as e:
  #          log.warn("[{}] {}".format(self.name, str(e)))


    def update(self, context, data):
        price = data.history(context.asset, 'price', 1, self.freq)
        self.Y.update(np.log(price))
        
        first_time = self._x.beta_kf is None
        if(not first_time):
            self._x.update_model(context, data)
            capm_price = self._x.get_price(context, data)
            self._x.update_price_hist(capm_price)
        else:
            capm_price = self._x.get_price(context, data)
            
        self.X.update(np.log(pd.Series(capm_price, index=price.index, name='CAPM(PG)')))
        
        self.kf.update(self.means_frame().iloc[-1])
        
        record(actual_price=np.float64(price), capm_price=capm_price)


    def mean_spread(self):
        means = self.means_frame()
        beta, alpha = self.kf.state_mean
        return means[self._y] - (beta * means[self._x.symbol] + alpha)


    def means_frame(self):
        mu_Y = self.Y.state_means
        mu_X = self.X.state_means
        return pd.DataFrame([mu_Y, mu_X]).T

            
    def initialize_filters(self, context, data):
        prices = data.history(self._y, 'price', self.initial_bars, self.freq)
        self.Y.update(np.log(prices))
        
        self._x.initialize_price(context, data)
        self.X.update(pd.Series(np.log(np.array(self._x.capm_price_hist)), index=prices.index, name='CAPM(PG)'))

        # Drops the initial 0 mean value from the kalman filter
        self.X.state_means = self.X.state_means.iloc[-self.initial_bars:]
        self.Y.state_means = self.Y.state_means.iloc[-self.initial_bars:]
        self.kf = KalmanRegression(self.Y.state_means, self.X.state_means,
                                   delta=self.delta, maxlen=self.maxlen)
    
    
    def get_pnl(self, context, data):
        y = self._y
        prices = data.history(y, 'price', 1, '1d').iloc[-1]
        positions = context.portfolio.positions
        dy = prices - positions[y].cost_basis
        return (positions[y].amount * dy)


class KalmanMovingAverageMod(KalmanMovingAverage):
    def update(self, observations):
        for dt, observation in observations.items():
            self._update(dt, observation)