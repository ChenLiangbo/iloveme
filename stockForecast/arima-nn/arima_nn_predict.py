#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os 
import pickle
import cv2
# from myNeurolNetworkModel import MyNeurolNetworkModel
from myException import MyOwnException
from MyArimaModel import MyArimaModel
from statsmodels.tsa.arima_model import ARIMA, _arma_predict_out_of_sample



yahooData = np.load('yahoo_finance5.npy')

filename = './yahoo_finance5.xls'
xls = pd.ExcelFile(filename)
dateframe = xls.parse("Sheet1", index_col='Date') 
dateframe.index = pd.to_datetime(dateframe.index)

print "dateframe = "
print dateframe[0:10]


'''
model = ARIMA(timeSerize, order=(self.p,self.d,self.q), freq='D') # build a model
fitting = model.fit(disp=False)
forecast, fcasterr, conf_int = fitting.forecast(steps=self.next_ndays, alpha=.05)
'''
train_start = 28
train_end = 100
fit_length = 70
p = 3
d = 1
q = 2

x_sample = []
y_sample = []
itemrList = ["Open","High","Low","Close","Volume"]
for fc in xrange(train_start + fit_length,train_end):
    print "fc = ",fc
    temp = []
    for iterm in itemrList:
    	# print "iterm = ",iterm
    	selector = dateframe[iterm]
    	timeSerize = selector[fc - fit_length:fc]
    	# print "timeSerize = ",len(timeSerize)  
        model = ARIMA(timeSerize, order=(p,d,q), freq='D')
        fitting = model.fit(disp=False)
        params = fitting.params
        residuals = fitting.resid
        p = fitting.k_ar
        q = fitting.k_ma
        k_exog = fitting.k_exog
        k_trend = fitting.k_trend
        # n_days forecasting
        forecast = _arma_predict_out_of_sample(params, 1, residuals, p, q, k_trend, k_exog, endog=timeSerize, exog=None, start=len(timeSerize))

        # forecast, fcasterr, conf_int = fitting.forecast(steps=1, alpha=.05)
        real = selector[fc-1:fc]
        # print "forecast = ",(forecast,type(forecast))
        # print "real = ",(real,type(real))
        temp.append(float(real))
        temp.append(float(forecast))
        # print "temp = ",temp
    x_sample.append(temp)
    y = dateframe['Close'][fc:fc+1]
    y_sample.append(float(y))


x_sample = np.array(x_sample)
print "x_sample = ",x_sample.shape

#colunme index of x_sample 6

np.save('./npyfile/x_sample',x_sample)

y_sample = np.array(y_sample)
print "y_sample = ",y_sample.shape
np.save('./npyfile/y_sample',y_sample)
    



