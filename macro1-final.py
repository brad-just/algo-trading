# -*- coding: utf-8 -*-
# Brad Just
# 2020-07-20
# This is a modified implementation of Brad Just's Honors Thesis Trading Algorithm.
# It uses Machine Learning to predict whether or not to buy a stock each quarter.
# It is applied to the top 15 stocks in the S&P 500 which are grouped into a portfolio
# that rebalances each quarter.
from zipline.api import (
        order_target_percent, 
        symbols,
        symbol, 
        get_datetime,
        schedule_function,
        date_rules,
        time_rules,
        set_long_only,
        set_max_leverage
        )
from zipline.errors import SymbolNotFound
from ffn import calc_mean_var_weights
import pandas as pd
import numpy as np
from datetime import datetime
import pandas_datareader.data as web
import quandl
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import time

def filter_equities(test_equities):
    '''
    A function to do a quick check on the historical data stream to make sure
    the input symbols are in the database. A filtered list of symbols in the
    database are returned, an empty list is returned otherwise.
    '''
    equities = []
    for e in test_equities:
        try:
            symbol(e)
            equities.append(e)
        except SymbolNotFound:
            print(('SymbolNotFound: %s' %e))
            pass
    return(equities)
    
def get_pricehist_chunks(data, equities, chunk_size=100):
    hist = None
    for i in range(0, len(equities), chunk_size):
        if(i < chunk_size):
            hist = data.history(symbols(*equities[i:i+min(len(equities)-i,chunk_size)]), 'price', bar_count=1000, frequency='1d')
        else:
            new_hist = data.history(symbols(*equities[i:i+min(len(equities)-i,chunk_size)]), 'price', bar_count=1000, frequency='1d')
            hist = hist.join(new_hist, how='inner')
        time.sleep(3)
    return(hist)

def train_new_model(context, data, today, gdp, emp, cpi, ppi, spy, vix):
    '''
    This function trains a new SVC model on 5 years of trailing data. 5 years worth
    of quarterly economic data and market wide data are input. 5 years of stock
    specific data are calculated and resampled into quarters. Sectors stored in context.sector_key
    and context.sector_encoding are then appended to all of the factors for each stock into
    a matrix which is then fit to a SVC model.
    '''
    #join all market and economic data into a single DataFrame
    X = gdp
    X.columns = ['gdp']
    X = X.join(emp, how='inner')
    X = X.join(cpi, how='inner')
    X = X.join(ppi, how='inner')
    X = X.join(spy, how='inner')
    X = X.join(vix, how='inner')
    X.reset_index(inplace=True)
    
    equities = filter_equities(context.trainable_equities)
    
    price_hist4y = get_pricehist_chunks(data, equities)
    shifted_ret = price_hist4y.pct_change()[1:].resample('QS').mean()*63
    ret = shifted_ret.shift(-1)
    vol = price_hist4y.pct_change()[1:].resample('QS').std()*np.sqrt(63)
    mom = price_hist4y.rolling(10).apply(lambda x: x[-1] - x[0])[10:].resample('QS').mean()*63
    
    shifted_ret = pd.DataFrame(shifted_ret.stack()).reset_index()
    ret = pd.DataFrame(ret.stack()).reset_index()
    vol = pd.DataFrame(vol.stack()).reset_index()
    mom = pd.DataFrame(mom.stack()).reset_index()
    
    shifted_ret.columns = ['Date', 'Ticker', 'shifted_ret']
    ret.columns = ['Date', 'Ticker', 'ret']
    vol.columns = ['Date', 'Ticker', 'vol']
    mom.columns = ['Date', 'Ticker', 'mom']
    
    shifted_ret['Date'] = shifted_ret['Date'].dt.tz_localize(None)
    ret['Date'] = ret['Date'].dt.tz_localize(None)
    vol['Date'] = vol['Date'].dt.tz_localize(None)
    mom['Date'] = mom['Date'].dt.tz_localize(None)
    
    X = X.merge(shifted_ret, how='right', left_on=['Date'], right_on=['Date'])
    X = X.merge(ret, how='inner', left_on=['Ticker', 'Date'], right_on=['Ticker', 'Date'])
    X = X.merge(vol, how='inner', left_on=['Ticker', 'Date'], right_on=['Ticker', 'Date'])
    X = X.merge(mom, how='inner', left_on=['Ticker', 'Date'], right_on=['Ticker', 'Date'])
    sectors = X['Ticker'].apply(lambda x: context.sector_key[context.sector_key['TICKER'] == x.symbol]['Sector'].iloc[0])
    sector_codes = []
    for i in range(len(sectors)):
        sector_codes.append(context.sector_encoding[sectors.iloc[i]])
    sectors = pd.DataFrame(sector_codes)
    
    r = X['ret']
    X = X[['cpi', 'emp', 'gdp', 'mom', 'ppi', 'shifted_ret', 'SPY', '^VIX', 'vol']]
    context.metrics['means'] = X.mean().values
    context.metrics['stds'] = X.std().values
    
    X_scale = StandardScaler()
    X = pd.DataFrame(X_scale.fit_transform(X.values))
    X = X.join(sectors, rsuffix='_sector').dropna().values
    y = np.array(r.apply(lambda x: True if x > r.median() else False))
    
    print(('Training model, %s'%today))
    model = SVC()
    model.fit(X, y)
    context.model = model

def rebalance(context, data, bounds):
    '''
    Function uses 1 quarter (63 days) of trailing returns data to calculate Markowitz mean variance weights.
    If there are less than 6 stocks, weights aren't calculated and stocks are purchased in equal weight.
    Input stock portfolio is stored in context.to_buy.
    '''
    if(len(context.to_buy) > 5):
        rets = data.history(context.to_buy, 'price', bar_count=63, frequency='1d').dropna().pct_change()[1:]
        weights = calc_mean_var_weights(rets, weight_bounds=bounds) 
        for tic in context.to_buy:
            order_target_percent(tic, round(weights[tic],2))
    elif(context.to_buy):       
        for tic in context.to_buy:
            order_target_percent(tic, 1.0/6)

def get_data(start, end):    
    '''
    A function that pulls fresh economic data from quandl/FRED
    and SPY and VIX data from yahoo finance, then converts
    the retrieved data into quarterly returns on the factor.
    '''
    #get economic data from quandl and SPY, VIX data from yahoo finance
    econ_data = quandl.get(['FRED/GDPPOT', 'FRED/PAYEMS', 'FRED/CPIAUCSL', 'FRED/PPIACO'])
    econ_data.columns = ['gdp', 'emp', 'cpi', 'ppi']
    gdp = econ_data['gdp']
    emp = econ_data['emp']
    cpi = econ_data['cpi']
    ppi = econ_data['ppi']
    yahoo_data = web.DataReader(['SPY','^VIX'], 'yahoo', start, end)
    vix = yahoo_data['Adj Close']['^VIX']
    spy = yahoo_data['Adj Close']['SPY']
    
    #get "returns" and resample into quarters
    gdp_rets = gdp.pct_change()[1:]
    emp_rets = emp.pct_change()[1:].resample('QS').mean()*(12/4)
    cpi_rets = cpi.pct_change()[1:].resample('QS').mean()*(12/4)
    ppi_rets = ppi.pct_change()[1:].resample('QS').mean()*(12/4)
    vix_rets = vix.pct_change()[1:].resample('QS').mean()*(252/4)
    spy_rets = spy.pct_change()[1:].resample('QS').mean()*(252/4)
    
    return(gdp_rets, emp_rets, cpi_rets, ppi_rets, vix_rets, spy_rets)
    
def get_signal(context, data, ticker, gdp, emp, cpi, ppi, vix, spy):
    '''
    A function that inputs the ticker, date, and economic and market factors,
    then calculates the previous quarter's returns, volatility, and momentum.
    The function also looks up the input ticker's sector then combines all of
    the factors into a vector, scales the vector, and then combines it with
    the sector. The vector is then plugged into a ML model that returns True or False.
    '''
    #Look up the stock's sector and the sector's one hot encoded vector
    sector = context.sector_key[context.sector_key['TICKER'] == ticker]['Sector'].iloc[0]
    encoding = context.sector_encoding[sector]
    
    #retrieve one quarter's worth of price history
    price_hist1q = data.history(symbol(ticker), 'price', bar_count=63, frequency='1d')
    
    #calculate the stock specific factors
    shifted_ret = price_hist1q.pct_change()[1:].mean()*63
    vol = price_hist1q.pct_change()[1:].std()*np.sqrt(63)
    mom = price_hist1q.rolling(10).apply(lambda x: x[-1] - x[0])[10:].mean()*63
    
    #put all of the factors into a vector
    factors = np.array([cpi, emp, gdp, mom, ppi, shifted_ret, spy, vix, vol])
    
    # scale the factors to fit into the training distribution
    factors = (factors - np.array(context.metrics['means'])) / np.array(context.metrics['stds'])
    
    # add the sector encoded vector to the end
    vec = np.array(list(factors) + encoding)
    
    if(False if np.isnan(vec).sum() > 0 else True):
        
        #return the output of the model: True or False
        return(context.model.predict(vec.reshape(1,-1))[0])
        
    else:
        print(('Data missing for ticker %s'%ticker))
        return(False)
    
def exec_trades(context, data):
    '''
    A function that is run at the start of each month. If the month marks the start of
    a quarter (month is 1, 4, 7, 10) then the function uses a ML model to determine whether
    to purchase the stock or not. The function then purchases securities in accordance
    with Markowitz mean variance optimization.
    '''
    today = get_datetime('US/Eastern')  
    
    #the algorithm runs quarterly
    months = [1, 4, 7, 10]
    
    if today.month in months:
        
        #get economic, spy, and vix returns
        gdp_data, emp_data, cpi_data, ppi_data, vix_data, spy_data = get_data(datetime(today.year-5,today.month,today.day), datetime(today.year,today.month,today.day))      

        #retrain the model at the start of each year
        if(today.month == 1):
            #use real gdp instead of a proxy to train
            gdp = quandl.get('FRED/GDPC1').pct_change()[1:]
            train_new_model(context, data, today, gdp, emp_data, cpi_data, ppi_data, spy_data, vix_data)
        
        context.to_buy = []
        
        #need to make sure we're using last quarter's data
        month = months[months.index(today.month)-1]
        
        if(today.month==1):
            year = today.year-1
        else:
            year = today.year
            
        #get the numbers to plug into the model from last quarter
        first_of_month = datetime(year,month,1)
        gdp_factor = gdp_data[first_of_month]
        emp_factor = emp_data[first_of_month]
        cpi_factor = cpi_data[first_of_month]
        ppi_factor = ppi_data[first_of_month]
        spy_factor = spy_data[first_of_month]
        vix_factor = vix_data[first_of_month]
        
        for ticker in context.assets:
            
            #check to make sure each equity is available at this point in time
            if(filter_equities([ticker])):
                
                #plug all of the factors into the model and determine whether to buy or not
                signal = get_signal(context, data, ticker, gdp_factor, emp_factor, cpi_factor, ppi_factor, vix_factor, spy_factor)
                
                if(signal):
                    #record the tickers to buy for later optimization
                    context.to_buy.append(symbol(ticker))
                else:
                    #sell all current shares of undesirable tickers
                    order_target_percent(symbol(ticker), 0)
        
        #get the previous quarter's returns and calculate mean variance weights
        #purchase stocks in those weights unless there are under 6 stocks in the current portfolio
        rebalance(context, data, context.weight_bounds)
            
def initialize(context):
    
    #quandl api key
    quandl.ApiConfig.api_key = "<api key>"
    
    #get sector information
    context.sector_key = pd.read_excel('/Users/bradjust/desktop/trading/research/trading-strategy-db.xlsx', sheet_name='SectorKey')
    context.sector_encoding = pd.read_excel('/Users/bradjust/desktop/trading/research/trading-strategy-db.xlsx', sheet_name='SectorEncoding').to_dict('list')
    
    #get the trained model, and the trained model's metrics
    context.model = pd.read_pickle('/Users/bradjust/desktop/trading/research/thesis-macro-strategy/models/quarterlySVM_noFin_2009-01-01_2014-01-01_py36.pickle')  
    context.metrics = pd.read_pickle('/Users/bradjust/desktop/trading/research/thesis-macro-strategy/models/quarterlySVM_noFin_metrics_2009-01-01_2014-01-01_py36.pickle')
    
    #get the top 15 constituents of the S&P 500, all 1600 stocks to optimize over, and initialize an empty list and set the constraints for mean-variance optimization
    context.assets = pd.read_excel('/Users/bradjust/desktop/trading/research/trading-strategy-db.xlsx', sheet_name='SP500')['Symbol'][0:15].tolist()
    context.trainable_equities = context.sector_key['TICKER'].unique().tolist()
    context.to_buy = []
    context.weight_bounds = (0.03,0.15)
    
    #set specifications for portfolio management and trade scheduling
    set_long_only()
    set_max_leverage(1.05)
    schedule_function(exec_trades, date_rules.month_start(0), time_rules.market_open(minutes=30))  


def handle_data(context, data):
    #trades are scheduled in the exec_trades function 
    pass
    






