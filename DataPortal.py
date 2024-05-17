#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 09:56:35 2020

Class to combine a SQL database with price information (open, high, low, close, etc)
with a NoSQL database with financial information (open, high, low, close) as well as
wrapper functions to get economic data from the quandl api. This class makes it easy
to interact with data all in one place

@author: bradjust
"""

from pymongo import MongoClient
import pymysql
import quandl
import pandas as pd
import numpy as np
from datetime import datetime
import pandas_market_calendars as mcal


class DataPortal(object):
    '''
    Class for interacting with a securities master database that is spread
    accross both SQL and NoSQL systems. Additional functionality for accessing
    economic indicator data.

    Parameters
    ----------
    prices_uri : STR
        SQL DBURI TO THE PRICES DATABASE.
    financials_uri : STR
        NoSQL DBURI TO THE FINANCIALS DATABASE.
    quandl_api_key : STR
        QUANDL API KEY.

    '''
    
    def __init__(self, prices_uri, financials_uri, quandl_api_key):

        self.p_uri = prices_uri
        self.f_uri = financials_uri 
        self.q_key = quandl_api_key
        
        self._set_econ_portal()
        self.prices_db = self._open_mysql_conn()
        self.fin_db = self._open_mongo_conn()
        
    
    def __repr__(self):
        return "DataPortal(prices_uri={}, financials_uri={}, quandl_key={})".format(self.p_uri, self.f_uri, self.q_key)
        
        
    def _open_mongo_conn(self):
        '''
        Open a new connection to MongoDB using pymongo

        Returns
        -------
        pymongo.MongoClient connection.

        '''

        return(MongoClient(self.f_uri))
    
    
    def _open_mysql_conn(self):
        '''
        Open a new connection to MySQL using SQLAlchemy

        Returns
        -------
        pymysql.Connection()

        '''
        
        s = self.p_uri
        s = s[s.find('mysql://')+8:]
        s = s.split('@')
        
        user = s[0].split(':')[0]
        password = s[0].split(':')[1]
        host = s[1].split(':')[0]
        database = s[1].split('/')[1]
        
        return(pymysql.connect(host=host, user=user, password=password, database=database))
    
    
    def _set_econ_portal(self):
        '''
        Instantiates a quandl api object and sets the api key

        Returns
        -------
        None.

        '''
        self.econ_portal = quandl
        self.econ_portal.ApiConfig.api_key = self.q_key
    
    
    def get_overview(self, tickers):
        '''
        Get the overview infomation for a ticker.

        Parameters
        ----------
        ticker : STR
            TICKER SYMBOL.

        Returns
        -------
        Pandas.DataFrame with overview information for each ticker.

        '''
        
        #convert 'tickers' and 'fields' into lists if they are strings 
        tickers = self._list_convert(tickers)
        
        sql = '''
        SELECT Symbol.sid, Symbol.ticker, Exchange.abbrev, Symbol.instrument, Symbol.name, Symbol.currency
        FROM Symbol
        INNER JOIN Exchange
        ON Symbol.exchange_id=Exchange.exchange_id
        WHERE ticker IN {}
        '''.format(self._fmt_tickers(tickers))
        
        result = self._exec_mysql_query(sql)
        
        overviews = pd.DataFrame(result, columns=['sid', 'ticker', 'abbrev', 'instrument', 'name', 'currency'])
        
        return(overviews)
    
    
    def get_sector(self, tickers, one_hot_encoded=False):
        '''
        Get the sector for a ticker

        Parameters
        ----------
        tickers : STR or LIST
            TICKER SYMBOLS.
        one_hot_encoded : BOOL
            TRUE TO RETURN AN ENCODED VECTOR, FALSE FOR THE STRING, default is False

        Returns
        -------
        DataFrame with tickers and either sector strings or one hot encoded vectors

        '''
        
        #convert 'tickers' and 'fields' into lists if they are strings 
        tickers = self._list_convert(tickers)
        
        sql = '''
        SELECT Symbol.ticker, Sectors.sector
        FROM Symbol INNER JOIN Sectors
        ON Symbol.sid=Sectors.sid
        WHERE ticker IN {}
        '''.format(self._fmt_tickers(tickers))
        
        sectors = pd.DataFrame(self._exec_mysql_query(sql), columns=['ticker', 'sector'])
        
        if(one_hot_encoded):
            
            sector_possibilities = ['Communication Services',
                                    'Consumer Discretionary',
                                    'Consumer Staples',
                                    'Energy',
                                    'Financials',
                                    'Health Care',
                                    'Industrials',
                                    'Information Technology',
                                    'Materials',
                                    'Real Estate',
                                    'Utilities']
            sector_possibilities.sort()
            
            #create an empty matrix of encoded vectors
            encoded_df = pd.DataFrame(np.zeros((len(sectors), len(sector_possibilities))), columns=sector_possibilities)
            
            sectors = sectors.join(encoded_df)
            
            #set the proper encoded vectors based on sector
            for i in range(len(sectors)):
                s = sectors.iloc[i].sector
                sectors.at[i, s] = 1
            
            sectors = sectors.drop('sector', axis=1)
            
        return(sectors)
    
    
    def get_industry(self, tickers, one_hot_encoded=False):
        '''
        Get the industry for a ticker

        Parameters
        ----------
        tickers : STR or LIST
            TICKER SYMBOLS.
        one_hot_encoded : BOOL
            TRUE TO RETURN AN ENCODED VECTOR, FALSE FOR THE STRING, default is False

        Returns
        -------
        DataFrame with tickers and either industry strings or one hot encoded vectors
        '''
        
        #convert 'tickers' and 'fields' into lists if they are strings 
        tickers = self._list_convert(tickers)
        
        sql = '''
        SELECT Symbol.ticker, Sectors.industry
        FROM Symbol INNER JOIN Sectors
        ON Symbol.sid=Sectors.sid
        WHERE ticker IN {}
        '''.format(self._fmt_tickers(tickers))
        
        industries = pd.DataFrame(self._exec_mysql_query(sql), columns=['ticker', 'industry'])
        
        if(one_hot_encoded):
            
            industry_possibilities = [x[0] for x in self._exec_mysql_query("SELECT DISTINCT industry FROM Sectors") if x[0] not in {None, '--'}]
            industry_possibilities.sort()
            
            #create an empty matrix of encoded vectors
            encoded_df = pd.DataFrame(np.zeros((len(industries), len(industry_possibilities))), columns=industry_possibilities)
            
            industries = industries.join(encoded_df)
            
            #set the proper encoded vectors based on industry
            for i in range(len(industries)):
                ind = industries.iloc[i].industry
                industries.at[i, ind] = 1
            
            industries = industries.drop('industry', axis=1)
            
        return(industries)
    
    
    def get_prices(self, tickers, start_date, end_date, fields='price'):
        '''
        Return a price series (or series for all of the input fields) for all of
        the input tickers from start_date until end_date

        Parameters
        ----------
        tickers : STR or LIST
            TICKER SYMBOL OR LIST OF TICKER SYMBOLS.
        start_date : DATETIME.DATETIME
            PYTHON DATETIME OBJECT.
        end_date : DATETIME.DATETIME
            PYTHON DATETIME OBJECT.
        fields : STR or LIST, optional
            FIELD OR FIELDS TO RETURN A TIME SERIES FOR. The default is 'price'.

        Returns
        -------
        Pandas.Multiindex with all of the daily price series for each ticker.

        '''
        
        #convert 'tickers' and 'fields' into lists if they are strings 
        tickers = self._list_convert(tickers)
        fields = self._list_convert(fields)
            
        #assertion to make sure all of the fields are valid options
        valid_fields = {'open', 'high', 'low', 'close', 'volume', 'dividend', 'split', 'price', 'volume_adj'} #need to add a field for price later which is computed as the adjusted close
        for f in fields:
            assert(f in valid_fields)
        
        sql = '''
               SELECT 
                	`ticker`,
                    `date`,
                	`open`,
                    `high`,
                    `low`,
                    `close`,
                    `volume`,
                    `dividend`,
                    `split`
                FROM (
                	SELECT * FROM securities_master.Daily_prices
                    WHERE `date` > '{}' AND `date` < '{}'
                    ) A
                INNER JOIN securities_master.Symbol
                ON A.sid=Symbol.sid
                WHERE `ticker` IN {}
              '''.format(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), self._fmt_tickers(tickers))
        
        result = self._exec_mysql_query(sql)
        
        prices_df = pd.DataFrame(result, columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'dividend', 'split'])
        
        types = {'ticker': str, 
                 'date': np.datetime64, 
                 'open': np.float64, 
                 'high': np.float64, 
                 'low': np.float64, 
                 'close': np.float64, 
                 'volume': np.float64, 
                 'dividend': np.float64, 
                 'split': np.float64}
    
        prices_df = prices_df.astype(types)
        
        if('price' in fields):
            #multiplier for adjusted close, adjusted for dividends and splits. See formula in Chapter 2 of Quantitative Trading by Ernie Chan 
            prices_df = prices_df.groupby('ticker').apply(self._calc_adj_close)
        
        if('volume_adj' in fields):
            #volume adjusted for splits
            prices_df = prices_df.groupby('ticker').apply(self._calc_adj_volume)
            
        extra_cols = set(prices_df.columns) - set(fields + ['ticker', 'date'])
    
        prices_df = prices_df.drop(extra_cols, axis=1)
        
        return(prices_df.sort_values(['ticker', 'date']).reset_index(drop=True))
            
        
    
    def get_bars(self, tickers, bar_count, fields='price'):
        '''
        Get bar_count number of daily price (fields) bars before the current timestamp. 
        Doesn't include the current day. Might return less that bar_count number of bars
        if the database isn't 100% up to date.

        Parameters
        ----------
        tickers : STR or LIST
            TICKER SYMBOL OR LIST OF TICKER SYMBOLS.
        bar_count : INT
            NUMBER OF DAILY PRICE MEASUREMENTS TO RETRIEVE.
        fields : STR or LIST, optional
            FIELD OR FIELDS TO RETURN A TIME SERIES FOR. The default is 'price'.

        Returns
        -------
        Pandas.Multiindex with all of the daily price series for each ticker.

        '''
        
        cal = mcal.get_calendar('NYSE') #NYSE and NASDAQ have roughly the same holidays 
                                       #so for now just worry about NYSE and NASDAQ calendar dates. 
                                       #Will have to change with more markets being added
    
        end_date = datetime.now()
        start_date = cal.schedule(datetime(1900,1,1), end_date).index[-1*(bar_count+1)].to_pydatetime() #get pandas timestame and convert into python datetime
        
        prices_df = self.get_prices(tickers, start_date, end_date, fields)
        
        return(prices_df)
        
    
    def get_financials(self, tickers, start_date=None, end_date=None, fields=None):
        '''
        Get timeseries of specified financial statement fields for all input 
        tickers from start_date to end_date.

        Parameters
        ----------
        tickers : STR or LIST
            TICKER SYMBOL OR LIST OF TICKER SYMBOLS.
        start_date : DATETIME.DATETIME, optional
            PYTHON DATETIME OBJECT. IF NONE RETURN THE MOST RECENT DATA. The default is None
        end_date : DATETIME.DATETIME, optional
            PYTHON DATETIME OBJECT. IF NONE DEFAULT TO START_DATE. The default is None.
        fields : STR or LIST, optional
            FIELD OR FIELDS TO RETURN A TIME SERIES FOR. The default is 'price'.

        Returns
        -------
        Pandas.Multiindex with all of the quarterly financial series for each ticker.

        '''
        
        b_sheet = self.get_balance_sheet(tickers, start_date, end_date, fields).reset_index()
        i_stmt = self.get_income_statement(tickers, start_date, end_date, fields).reset_index()
        c_flow = self.get_cash_flows(tickers, start_date, end_date, fields).reset_index().drop('netIncome', axis=1)
        
        all_financials = b_sheet.merge(i_stmt, how='outer', on=['ticker','date']).merge(c_flow, how='outer', on=['ticker','date'])
        
        all_financials = all_financials.drop(['index', 'index_x', 'index_y'], axis=1)
        
        return(all_financials.sort_values(['ticker', 'date']).reset_index(drop=True))
    
    
    def _get_fin_stmt(self, tickers, stmt, start_date=None, end_date=datetime.now(), fields=None):
        '''
        Return the quarterly time series for a specified financial statment 
        (balance sheet, income statment, cash flows)

        Parameters
        ----------
        tickers : STR or LIST
            TICKER SYMBOL OR LIST OF TICKER SYMBOLS.
        stmt : STR
            EITHER BALANCE_SHEET, INCOME_STATMENT, OR CASH_FLOWS.
        start_date : DATETIME.DATETIME, optional
            PYTHON DATETIME OBJECT. IF NONE RETURN ALL AVAILABLE DATA. The default is None
        end_date : DATETIME.DATETIME, optional
            PYTHON DATETIME OBJECT. The default is the current datetime.
        fields : STR or LIST, optional
            FIELD OR FIELDS TO RETURN A TIME SERIES FOR FROM THE BALANCE SHEET. The default is 'price'.

        Returns
        -------
        Pandas.Multiindex with all of the financial statement series for each ticker.


        '''
        
        stmt = stmt.lower()
        
        assert(stmt in {'balance_sheet', 'income_statement', 'cash_flows'})
        
        #convert 'tickers' and 'fields' into lists if they are strings 
        tickers = self._list_convert(tickers)
        fields = self._list_convert(fields)
        
        if(start_date):
            
            cursor = self.fin_db.smdb.quarterly_financial_statements.find({"ticker": 
                                                                          {"$in": tickers}, 
                                                                           "$and": [{"date": {"$gte": start_date}},{"date": {"$lt": end_date}}]}, 
                                                                          {"ticker": 1, "date": 1, stmt: 1})
                
        else:
            
            cursor = self.fin_db.smdb.quarterly_financial_statements.find({"ticker": 
                                                                          {"$in": tickers}},
                                                                          {"ticker": 1, "date": 1, stmt: 1})
        
        table = {'ticker': [], 'date': []}
        for doc in cursor:
            table['ticker'].append(doc['ticker'])
            table['date'].append(doc['date'])
            for s in doc[stmt]: #need to deal with statements that have different fields using ML
                if(s not in table):
                    table[s] = [doc[stmt][s]]
                else:
                    table[s].append(doc[stmt][s])

        df = pd.DataFrame.from_dict(table, orient='index').T.drop('last_updated_date', axis=1).replace('None', np.nan)
        
        return(df.sort_values(['ticker', 'date']).reset_index(drop=True))
    
    
    def get_balance_sheet(self, tickers, start_date=None, end_date=datetime.now(), fields=None):
        '''
        Get timeseries of quarterly balance sheets for all input tickers from start_date
        to end_date. Can specify fields from the balance sheet

        Parameters
        ----------
        tickers : STR or LIST
            TICKER SYMBOL OR LIST OF TICKER SYMBOLS.
        start_date : DATETIME.DATETIME, optional
            PYTHON DATETIME OBJECT. IF NONE RETURN ALL AVAILABLE DATA. The default is None
        end_date : DATETIME.DATETIME, optional
            PYTHON DATETIME OBJECT. The default is the current datetime.
        fields : STR or LIST, optional
            FIELD OR FIELDS TO RETURN A TIME SERIES FOR FROM THE BALANCE SHEET. The default is 'price'.

        Returns
        -------
        Pandas.Multiindex with all of the balance sheet series for each ticker.

        '''
        
        return(self._get_fin_stmt(tickers, 'balance_sheet', start_date, end_date, fields))
        
    
    def get_income_statement(self, tickers, start_date=None, end_date=None, fields=None):
        '''
        Get timeseries of quarterly income statements for all input tickers from start_date
        to end_date. Can specify fields from the balance sheet

        Parameters
        ----------
        tickers : STR or LIST
            TICKER SYMBOL OR LIST OF TICKER SYMBOLS.
        start_date : DATETIME.DATETIME, optional
            PYTHON DATETIME OBJECT. IF NONE RETURN THE MOST RECENT DATA. The default is None
        end_date : DATETIME.DATETIME, optional
            PYTHON DATETIME OBJECT. IF NONE DEFAULT TO START_DATE. The default is None.
        fields : STR or LIST, optional
            FIELD OR FIELDS TO RETURN A TIME SERIES FOR FROM THE INCOME STATEMENT. The default is 'price'.

        Returns
        -------
        Pandas.Multiindex with all of the income statement series for each ticker.

        '''
        
        return(self._get_fin_stmt(tickers, 'income_statement', start_date, end_date, fields))
    
    
    def get_cash_flows(self, tickers, start_date=None, end_date=None, fields=None):
        '''
        Get timeseries of quarterly statements of cash flows for all input tickers from start_date
        to end_date. Can specify fields from the balance sheet

        Parameters
        ----------
        tickers : STR or LIST
            TICKER SYMBOL OR LIST OF TICKER SYMBOLS.
        start_date : DATETIME.DATETIME, optional
            PYTHON DATETIME OBJECT. IF NONE RETURN THE MOST RECENT DATA. The default is None
        end_date : DATETIME.DATETIME, optional
            PYTHON DATETIME OBJECT. IF NONE DEFAULT TO START_DATE. The default is None.
        fields : STR or LIST, optional
            FIELD OR FIELDS TO RETURN A TIME SERIES FOR FROM THE STATEMENT OF CASH FLOWS. The default is 'price'.

        Returns
        -------
        Pandas.Multiindex with all of cash flows series for each ticker.

        '''
        
        return(self._get_fin_stmt(tickers, 'cash_flows', start_date, end_date, fields))
    
    
    def get_economic_indicator(self, indicators, start_date=None, end_date=None):
        '''
        Get timeseries of economic indicators from quandl

        Parameters
        ----------
        indicators : STR or LIST
            INDICATOR(S) TO GET SERIES FOR. CAN BE RAW QUANDL STRINGS OF PREDEFINED SIMPLIFIED STRINGS
        start_date : DATETIME.DATETIME, optional
            PYTHON DATETIME OBJECT. IF NONE RETURN THE MOST RECENT DATA. The default is None
        end_date : DATETIME.DATETIME, optional
            PYTHON DATETIME OBJECT. IF NONE DEFAULT TO START_DATE. The default is None.

        Returns
        -------
        Pandas.DataFrame with economic indicator series for all indicators between the time frame.

        '''
        
        indicators = self._list_convert(indicators)
        
        #short predefined map of commonly used variables to quandl series for easier access
        preset_map = {'GDP': 'FRED/GDPC1', 
                      'CPI': 'FRED/CPIAUCSL', 
                      'PPI': 'FRED/PPIACO', 
                      'EMPLOYMENT': 'FRED/PAYEMS'}
        
        new_indicators = [preset_map[x.upper()] if x.upper() in preset_map else x for x in indicators]
        
        ind_df = quandl.get(new_indicators)
        
        ind_df.columns = indicators
        
        if(start_date and end_date):
            
            ind_df = ind_df[(start_date <= ind_df.index) & (ind_df.index < end_date)]
            
        elif(start_date):
        
            ind_df = ind_df[start_date <= ind_df.index]    
        
        elif(end_date):
            
            ind_df = ind_df[ind_df.index < end_date]
        
        return(ind_df)
    
    
    @staticmethod
    def _fmt_tickers(tickers):
        '''
        Formats a list of tickers for a pymysql query

        Parameters
        ----------
        tickers : String or List
            STRING OR LIST OF TICKER SYMBOLS.

        Returns
        -------
        FORMATTED TICKER SYMBOLS.

        '''
        fmt_ticker = "('%s')"%tickers[0] if len(tickers)==1 else tuple(tickers)
        return(fmt_ticker)
    
    
    def _exec_mysql_query(self, stmt):
        '''
        Open a pymysql cursor, executes a query into the cursor, and returns
        the results

        Parameters
        ----------
        stmt : STRING
            RAW SQL QUERY TO EXECUTE.

        Returns
        -------
        PYMYSQL QUERY ITERABLE.

        '''
        with self.prices_db.cursor() as cursor:
            cursor.execute(stmt)
            result = cursor.fetchall()
        
        return(result)
    
    
    @property
    def _exec_mongo_query(self):
        '''
        Execute a query against the mongodb using pymongo. Return the results.
        '''
        cursor = self.fin_db.smdb.quarterly_financial_statements
        return(cursor)
        
    
    
    @staticmethod
    def _list_convert(variable):
        '''
        Takes in a variable (either a string or some type of iterable) and converts it to a list
        (if it isn't already). A string converted into a list will just be a list with one element

        Parameters
        ----------
        variable : STR, LIST, OR OTHER ITERABLE
            TO BE CONVERTED INTO A STRING.

        Returns
        -------
        The converted variable.

        '''
        
        if(isinstance(variable, str)): 
            variable = [variable]
            
        return(variable)
                
    
    def close_connections(self):
        '''
        Closes all open database connections. Meant to be called after you are
        finished using the DataPortal object

        Returns
        -------
        None.

        '''
        
        self.prices_db.close()
        self.fin_db.close()
        
    
    @staticmethod
    def _calc_adj_close(df):
        '''
        Static method meant to be applied during a pandas.DataFrame.groupby
        operation. Calculates the 'adjusted close column' or 'price' column
        based on the 'split', 'dividend', and 'close' columns

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame passed by groubpy.apply().

        Returns
        -------
        Copy of original DataFrame with new 'price' column added.

        '''
        mult = df['split'].cumprod() / df['split'].cumprod().iloc[-1]
        mult = mult * (1 - (df['dividend'].shift(-1) / df['close'])).iloc[::-1].cumprod().iloc[::-1].fillna(1)
        df['price'] = df['close'] * mult
        return(df)
    
    @staticmethod
    def _calc_adj_volume(df):
        '''
        Static method meant to be applied during a pandas.DataFrame.groupby
        operation. Calculates the 'adjusted volume' column
        based on the 'split' and 'volume' columns.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame passed by groubpy.apply().

        Returns
        -------
        Copy of original DataFrame with new 'volume_adj' column added.

        '''
        df['volume_adj'] = df['volume']*df['split'].iloc[::-1].cumprod().iloc[::-1]
        return(df)
    
    
    
    
    
    
    
    
    
    

