
# coding: utf-8

# In[2]:


'''
1. Get data Offline Excel, Yahoo! Stock Finance
2. Calculate by ARMA/ARIMA to obtain prediction 'n next days' each of stock item
3. Calculate KDJ from output ARMA/ ARIMA
4. The No.3's output become input GA

'''

import os
import sys
import numpy as np
import pandas as pd
from itertools import izip

# get_ipython().magic(u'matplotlib inline')
import matplotlib.pylab as plt
from matplotlib.finance import candlestick
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from statsmodels.tsa.arima_model import ARMA, _arma_predict_out_of_sample
from statsmodels.tsa.arima_model import ARIMA


# Load Excel file
# basedir = os.path.abspath(os.path.dirname(__file__))
basedir = os.getcwd()
filename = os.path.join(basedir, 'stock.xlsx')
xls = pd.ExcelFile(filename)

df_train = xls.parse('Sheet3', index_col='Date') # train

class BuildModel:
    def __init__(self,
        p=2,                 # Auto Regression (AR), lag p
        d=1,                 # Differencing NaN shifting for ARIMA, lag d
        q=0,                 # Moving Average (MA), lag q
        n_days=1,            # How many days a head to be predicted
        fc=1,                # iteration for n days forecast
        kdj_short_period=3,  # to calculate D (Divergen) K
        kdj_long_period=5):  # to calculate K

        # ARMA/ ARIMA
        self.p = p
        self.d = d
        self.q = q
        
        # how many days a head to be forecasted
        self.n_days = n_days

        # KDJ
        self.kdj_short_period = kdj_short_period
        self.kdj_long_period  = kdj_long_period

        # fc
        self.fc = fc

    # -------------------------------------------------------------
    # 1st Section
    # ARIMA/ ARMA
    
    def predict_arima_next_days(self, item):
        ts = df_train[item]
        ts = ts.sort_index() # sorting index Date
        ts_last_day = ts[self.fc] # real last data
        ts = ts[0:self.fc] # index 0 until last data - 1

        model = ARIMA(ts, order=(self.p, self.d, self.q)) # build a model
        fitting = model.fit(disp=False)

        # n_days forecasting
        forecast, fcasterr, conf_int = fitting.forecast(steps=self.n_days, alpha=.05)
        # ts:          history until 1 day before self.fc
        # ts[self.fc]: last day
        # forecast:    1 day forecast (time equalto ts[self.fc])
        return ts, ts_last_day, forecast
    
    def predict_arma_next_days(self, item):
        ts = df_train[item]
        ts = ts.sort_index() # sorting index Date
        ts_last_day = ts[self.fc] # real last data
        ts = ts[0:self.fc] # index 0 until last data - 1

        model = ARMA(ts, order=(self.p,self.d self.q), freq='D') # build a model
        fitting = model.fit(disp=False)
        params = fitting.params
        residuals = fitting.resid
        p = fitting.k_ar
        q = fitting.k_ma
        k_exog = fitting.k_exog
        k_trend = fitting.k_trend

        # n_days forecasting
        forecast = _arma_predict_out_of_sample(params, self.n_days, residuals, p, q, k_trend, k_exog, endog=ts, exog=None, start=len(ts))
        # ts:          history until 1 day before self.fc
        # ts[self.fc]: last day
        # forecast:    1 day forecast (time equalto ts[self.fc])
        return ts, ts_last_day, forecast

    # -------------------------------------------------------------
    # 2nd Section
    # KDJ Indicator
    
    def rsv_indicator(self, cl, lo_prev, hi_prev):
        return 100 * (cl-np.min(lo_prev))/(np.max(hi_prev)-np.min(lo_prev))
    
    def kdj_indicator(self, rsv):
        rsv = pd.DataFrame(rsv)
        # get K by EMA
        K = rsv.ewm(ignore_na=False, span=self.kdj_long_period, min_periods=0, adjust=True).mean()
        # get D (Divergen of K) by MA
        K.dropna(inplace=True)
        D = K.rolling(window=self.kdj_short_period, center=False).mean()
        # get J
        J = 3*K - 2*D
        # print ( 'Calculate KDJ Indicator.' )
        return K, D, J
    
    def kdj_calc(self, open, high, low, close, volume):
        # print ( 'Length data before KDJ in: {}'.format(len(open)) )
        # X is total input list
        # y is output list
        rsv_list, X, y = [], [], []
        for i, op in enumerate(open):
            hi, lo, cl, vo = high[i], low[i], close[i], volume[i]
            rsv = 0.0
            # lo_prev is a list for previous low price
            # hi_prev is a list for previous high price
            lo_prev, hi_prev = [], []
            if i > kdj_long_period-1:
                for k in range(1, kdj_long_period+1):
                    _lo, _hi = low[i-k], high[i-k]
                    lo_prev.append(_lo)
                    hi_prev.append(_hi)
                rsv = self.rsv_indicator(cl, lo_prev, hi_prev)
            else: pass

            rsv_list.append(rsv)
            X.append([ op, hi, lo, cl, vo ])
            
            y.append([ cl ])


        K, D, J = self.kdj_indicator(rsv_list)
        # Convert from pandas DataFrame to a list
        K = map( float, K[0].tolist() )
        D = map( float, D[0].tolist() )
        J = map( float, J[0].tolist() )
                
        # Remove Index value in X and y before index self.kdj_long_period to get rid of NaN
        # X_new and Y_new for list input and output new X and y with index started from 'self.kdj_long_period'
        X_new, y_new = [], []
        for i, row in enumerate(X[self.kdj_long_period:], start=self.kdj_long_period):
            op, hi, lo, cl, vo = row
            _K, _D, _J = K[i], D[i], J[i]
                
            # X new for input list
            X_new.append([ op, hi, lo, cl, vo, _K, _D, _J ])
                
            # y new for output list
            y_new.append([ y[i][0] ])
            
        X, y = X_new, y_new
        # print ( 'Length data after KDJ in: {}'.format(len(X)) )
        return X, y




# -----------------------------------------------------------
# Get the forecast each of items/price & volume
# instance: is instance of class BuildModel
# item: is stock prices and volume:
#       e.g: 'Open', 'High', 'Low', 'Close', 'Volume'
# d: is from p, d, q:
#       d=0 for ARMA, and d=1 for ARIMA
def item_prediction(instance, d, item):
    ts, n_days_real, n_days_forecast = None, None, None
    if d == 0:
        ts, n_days_real, n_days_forecast = instance.predict_arma_next_days(item)
    else:
        ts, n_days_real, n_days_forecast = instance.predict_arima_next_days(item)
    
    ts_real = ts
    ts_forecast = ts
  
    # n_days_real : output format is float
    # n_days_forecast: output format is a list
    ts_real = ts_real.append(pd.DataFrame([n_days_real])) # ts + n days real
    ts_forecast = ts_forecast.append(pd.DataFrame(n_days_forecast)) #ts + n days fc
    
    ts_real = map( float, ts_real[0].tolist() ) # n days real
    ts_forecast = map( float, ts_forecast[0].tolist() ) # n days fc 

    return ts_real, ts_forecast, n_days_real, n_days_forecast[0]


# -----------------------------------------------------------
print 'Started...'
# Configuration ARMA/ARIMA, KDJ and Output target price
# NOTICE:
#    d=1 , automatically use ARIMA
#    d=0 , automatically use ARMA
p, d, q = 2, 1, 0
n_days = 1 # How many 1 days prediction/ DONOT CHANGE
kdj_short_period = 3
kdj_long_period = 5

# Start Produce Weight for GA
print 'Produce trX and trY for GA.'

trX, trY = [], []
start = 27
stop = 208
for fc in range(start, stop): # fc = forecast n days
    an = BuildModel(p, d, q, n_days, fc, kdj_short_period, kdj_long_period)
    ts_real_op, ts_fc_op, n_days_real_op, n_days_fc_op = item_prediction(an, d, 'Open')
    ts_real_hi, ts_fc_hi, n_days_real_hi, n_days_fc_hi = item_prediction(an, d, 'High')
    ts_real_lo, ts_fc_lo, n_days_real_lo, n_days_fc_lo = item_prediction(an, d, 'Low')
    ts_real_cl, ts_fc_cl, n_days_real_cl, n_days_fc_cl = item_prediction(an, d, 'Close')
    ts_real_vo, ts_fc_vo, n_days_real_vo, n_days_fc_vo = item_prediction(an, d, 'Volume')
    
    
    if fc == (stop-1):
        print 'Collecting last data in epoch: {}'.format(fc)
        X, y = an.kdj_calc(ts_fc_op, ts_fc_hi, ts_fc_lo, ts_fc_cl, ts_fc_vo)

        # get the last row of X and then extract the all items.
        op, hi, lo, cl, vo, K, D, J = X[-1]

        # trX.append([ n_days_fc_op, n_days_fc_hi, n_days_fc_lo, n_days_fc_cl, n_days_fc_vo ])
        trX.append([ n_days_fc_op, n_days_fc_hi, n_days_fc_lo, n_days_fc_cl, n_days_fc_vo, K, D, J ])
        trY.append([ n_days_fc_cl ])
    else:
        print 'Collecting data in epoch: {}'.format(fc)
        X, y = an.kdj_calc(ts_real_op, ts_real_hi, ts_real_lo, ts_real_cl, ts_real_vo)

        # get the last row of X and then extract the all items.
        op, hi, lo, cl, vo, K, D, J = X[-1]

        # trX.append([ n_days_real_op, n_days_real_hi, n_days_real_lo, n_days_real_cl, n_days_real_vo ])
        trX.append([ n_days_real_op, n_days_real_hi, n_days_real_lo, n_days_real_cl, n_days_real_vo, K, D, J ])
        trY.append([ n_days_real_cl ])



print trY[0:4]
yy = df_train['Close']
yy = yy.sort_index()
yy = yy[start:]
print yy[0:4]

