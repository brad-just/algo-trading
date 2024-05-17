#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:57:46 2020

Python class with a factor model trading strategy class for zipline

@author: bradjust
"""

import os
import sys
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
from zipline.api import order_target_percent, get_datetime


class FactorModel(object):
    '''
    Class for factor model trading strategies on Zipline.
    '''
    
    def __init__(self, asset, factor_list, model_pipeline, window_size, resample_abrev, leverage=1.0, classGenerator=lambda x : (x > x.median())*1, periods=1):
        self.symbol = asset
        self.factors = factor_list
        self.model_pipe = model_pipeline
        self.window_size = window_size
        self.resample_freq = resample_abrev
        self.leverage = leverage
        self.getClasses = classGenerator
        self.periods = periods
        
    
    def trading_logic(self, context, data):
        '''
        Trading logic for the factor model.
        '''
        
        if(self.resample_freq=='QS'):
            today = get_datetime()
            
            if(today.month not in {1,4,7,10}):
                return
        
        self.train_model(context, data)
        signal = self.get_signal(context, data)
        
        if(signal==1):
            order_target_percent(self.symbol, self.leverage)
        elif(signal==0):
            order_target_percent(self.symbol, 0.0)
        elif(signal==-1):
            order_target_percent(self.symbol, -self.leverage)
        else:
            sys.exit('Unknown signal generated')
    
    
    def get_signal(self, context, data):
        
        X, _ = self.get_data(context, data, training=False)
        
        return(self.model_pipe.predict(X))
    
    
    def train_model(self, context, data):
        
        X, y = self.get_data(context, data, training=True)
        
        self.model_pipe.fit(X,y)
    
    
    def get_data(self, context, data, training=True):
        '''
        Either gets the most recent datapoint or all of the data needed for training
        depending on the value of the input "training" variable. All of the factor
        retreval logic is implemented in the 'Factor' objects stored in self.factor_list.
        '''
        
        #fetch the symbol data first to access the proper dates
        y = np.log(data.history(self.symbol, 'price', bar_count=self.window_size, frequency='1d')).diff()[1:]
        y.index = y.index.tz_localize(None)
        y.name = 'rets'
        
        start, end = y.index[0].to_pydatetime(), y.index[-1].to_pydatetime()
        
        X = None
        
        for factor in self.factors:
            if(factor.source=='zipline'):
                factor_data = factor.fetch(context, data, self.window_size, self.resample_freq)
            else:
                factor_data = factor.fetch(start, end)
                
            if(isinstance(X, pd.DataFrame)):
                X = X.join(factor_data, how='outer')
            else:
                X = factor_data
        
        X = X.resample(self.resample_freq).mean()
        y = y.resample(self.resample_freq).mean()
        
        if(self.periods > 1):
            temp_matrix = []
            for i in range(0, len(X)-self.periods+1):
                temp_matrix.append(X.iloc[i:i+self.periods].values.flatten().tolist())
                
            temp_columns = []
            for j in range(0, self.periods):
                for factor in X.columns.tolist():
                    temp_columns.append('{}_t-{}'.format(factor, self.periods-j))
                
            X = pd.DataFrame(temp_matrix, index=X.index[self.periods-1:], columns=temp_columns)
        
        y_class = self.getClasses(y)
        
        temp = X.join(y_class, how='inner').dropna()
        
        if(training):
            return(temp.drop('rets', axis=1)[:-1].values, temp['rets'][1:].values)
        else:
            return(temp.drop('rets', axis=1).iloc[-1].values.reshape(1,-1), None)    
        

class BaseFactor():
    def __init__(self, symbol, source, api_key=None, store_loc=None):
        self.symbol = symbol
        self.source = source
        self.api_key = api_key
        self.store_loc = store_loc
        self.data = None
        
    def fetch(self, start, end):
        #customly written function to return the exact factor data needed
        pass
        '''
        if(not isinatance(self.data, pd.DataFrame)):

            self.data = web.DataReader(self.symbol, data_source=self.source, api_key=self.api_key)
            
            #cleaning step
            
        data = self.data[ (start<=self.data.index) & (self.data.index<=end) ]
            
        return(data)
    '''


class Momentum(BaseFactor):
    def __init__(self, symbol):
        super().__init__(symbol, 'zipline', None, None)
    
    def fetch(self, context, data, nbars, freq):
        data = data.history(self.symbol, 'price', bar_count=nbars, frequency='1d')
        data.name = 'momentum'
        data.index = data.index.tz_localize(None)
        data = pd.DataFrame(np.log(data).diff()[1:])
        return(data)
    
    
class Volatility(BaseFactor):
    def __init__(self, symbol):
        super().__init__(symbol, 'zipline', None, None)
    
    def fetch(self, context, data, nbars, freq):
        data = data.history(self.symbol, 'price', bar_count=nbars, frequency='1d')
        data.name = 'volatility'
        data = np.log(data).diff()[1:]
        data.index = data.index.tz_localize(None)
        data = pd.DataFrame(data.resample(freq).std())
        return(data)


class FredFactor(BaseFactor):
    def __init__(self, symbol, api_key=None, storage_dir=None):
        super().__init__(symbol, 'fred', api_key, store_loc=storage_dir)
        
    def fetch(self, start, end):
        if(not isinstance(self.data, pd.DataFrame)):
            path_file = '{}-{}_{}-{}.csv'.format(self.source, self.symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            if(self.store_loc):
                path_file = self.store_loc + path_file

            if(os.path.exists(path_file)):
                self.data = pd.read_csv(path_file)
                self.data['DATE'] = pd.to_datetime(self.data['DATE'])
                self.data = self.data.set_index('DATE')
            else:
                self.data = web.DataReader(self.symbol, 
                                           data_source=self.source, 
                                           start=datetime(2000,1,1), end=datetime.now(), 
                                           api_key=self.api_key)
                self.data = self.data.interpolate(axis=1)
                self.data = np.log(self.data).diff()[1:].replace([np.inf,-np.inf], np.nan).ffill()
                self.data.to_csv(path_file)
                
        data = self.data[ (start<=self.data.index) & (self.data.index<=end) ]
        return(data)
    
    
class FamaFactor(BaseFactor):
    def __init__(self, symbol, api_key=None, storage_dir=None):
        super().__init__(symbol, 'famafrench', api_key, store_loc=storage_dir)
        
    def fetch(self, start, end):
        if(not isinstance(self.data, pd.DataFrame)):
            path_file = '{}-{}_{}-{}.csv'.format(self.source, self.symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            if(self.store_loc):
                path_file = self.store_loc + path_file
                                                 
            if(os.path.exists(path_file)):
                self.data = pd.read_csv(path_file)
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                self.data = self.data.set_index('Date')
            else:
                self.data = pd.DataFrame(web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 
                                           data_source=self.source, 
                                           start=datetime(2000,1,1), end=datetime.now())[0][self.symbol])
                self.data = self.data.interpolate(axis=1)
                self.data = np.log(self.data).diff()[1:].replace([np.inf,-np.inf], np.nan).ffill()
                self.data.to_csv(path_file)
                
        data = self.data[ (start<=self.data.index) & (self.data.index<=end) ]
        return(data)    
    

class YahooFactor(BaseFactor):
    def __init__(self, symbol, api_key=None, storage_dir=None):
        super().__init__(symbol, 'yahoo', api_key, store_loc=storage_dir)
        
    def fetch(self, start, end):
        if(not isinstance(self.data, pd.DataFrame)):
            path_file = '{}-{}_{}-{}.csv'.format(self.source, self.symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            if(self.store_loc):
                path_file = self.store_loc + path_file
                                                 
            if(os.path.exists(path_file)):
                self.data = pd.read_csv(path_file)
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                self.data = self.data.set_index('Date')
            else:
                self.data = pd.DataFrame(web.DataReader(self.symbol, 
                                           data_source=self.source, 
                                           start=datetime(2000,1,1), end=datetime.now(), 
                                           api_key=self.api_key)['Close'])
                self.data.columns = [self.symbol]
                self.data = self.data.interpolate(axis=1)
                self.data = np.log(self.data).diff()[1:].replace([np.inf,-np.inf], np.nan).ffill()
                self.data.to_csv(path_file)
                
        data = self.data[ (start<=self.data.index) & (self.data.index<=end) ]
        return(data)  
    

class QuandlFactor(BaseFactor):
    def __init__(self, symbol, api_key=None, storage_dir=None):
        super().__init__(symbol, 'quandl', api_key, store_loc=storage_dir)
        
    def fetch(self, start, end):
        if(not isinstance(self.data, pd.DataFrame)):
            s = self.symbol.split('/')[1]
            path_file = '{}-{}_{}-{}.csv'.format(self.source, s, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            if(self.store_loc):
                path_file = self.store_loc + path_file
                                                 
            if(os.path.exists(path_file)):
                self.data = pd.read_csv(path_file)
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                self.data = self.data.set_index('Date')
            else:
                self.data = web.DataReader(self.symbol, 
                                           data_source=self.source, 
                                           start=datetime(2000,1,1), end=datetime.now(), 
                                           api_key=self.api_key)
                self.data = self.data.interpolate(axis=1)
                self.data = np.log(self.data).diff()[1:].replace([np.inf,-np.inf], np.nan).ffill()
                self.data.to_csv(path_file)
                
        data = self.data[ (start<=self.data.index) & (self.data.index<=end) ]
        return(data)  
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    