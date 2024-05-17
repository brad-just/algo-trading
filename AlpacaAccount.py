#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:47:03 2020

A base class that represents an Alpaca Account with specific functionality
such as positition and order extraction.

@author: bradjust
"""


class AlpacaAccount(object):
    def __init__(self, con):
        self.api = con
        
    def get_transaction(self, order_id):
        '''
        Queries the Alpaca api for information about a specific order and
        transforms the order into a "transaction"

        Parameters
        ----------
        order_id : STR
            ALPACA GENERATED order_id.

        Returns
        -------
        transaction: DICT
            CONTAINS RELEVENT ELEMENTS FROM THE ALPACA ORDER

        '''
        order = self.api.get_order_by_client_order_id(order_id)
        transaction = {
            'order_id': order_id, 
            'filled_at': order.filled_at.to_pydatetime(),
            'amount': float(order.filled_qty), 
            'price': float(order.filled_avg_price), 
            'side': order.side, 
            'commission': 0.0,
            'symbol': order.symbol
            }
        
        return(transaction)
    
    def get_last_trade_price(self, ticker):
        return(self.api.get_last_trade(ticker).price)