#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd

csvFile = './5years_day.csv'

dataFrame = pd.read_csv(csvFile)
Date = list(dataFrame['Date'])
Open = list(dataFrame['Open'])
High = list(dataFrame['High'])
Low  = list(dataFrame['Low'])
adj_close = list(dataFrame['Adj Close'])
Volume = list(dataFrame['Volume'])
Close = list(dataFrame['Close'])

Date.reverse()
Open.reverse()
Close.reverse()
High.reverse()
Low.reverse()
adj_close.reverse()
Volume.reverse()


yahoo_date = np.asarray(Date).reshape(len(Date),1)
np.save('yahoo_date',yahoo_date)
print "yahoo_date.shape = ",yahoo_date.shape

'''yahoo_finance5 : Open,High,Low,Close,Volume*e-6'''
yahoo_finance5 = np.zeros((len(Open),5),dtype = np.float64)

yahoo_finance5[:,0] = np.asarray(Open).astype(np.float64)
yahoo_finance5[:,1] = np.asarray(High).astype(np.float64)
yahoo_finance5[:,2] = np.asarray(Low).astype(np.float64)
yahoo_finance5[:,3] = np.asarray(Close).astype(np.float64)
yahoo_finance5[:,4] = np.asarray(Volume).astype(np.float64)*(1e-6)
# yahoo_finance5[:,4] = np.asarray(adj_close).astype(np.float64)

np.save('yahoo_finance5',yahoo_finance5)

import xlwt 
book = xlwt.Workbook(encoding = 'utf-8')
sheet1 = book.add_sheet('Sheet1',cell_overwrite_ok = True)
shape = yahoo_finance5.shape
for x in xrange(shape[0]):
    for y in xrange(shape[1]+1):
        if y > shape[1]-1:
            sheet1.write(x,y,Date[x])
        else:
            sheet1.write(x,y,yahoo_finance5[x,y])

<<<<<<< HEAD:chenlb/stock/stock_1.py
book.save('yahoo_financewith_date.xls')
print "finish ..."
=======
book.save('yahoo_finance5.xls')
print "finish ..."
>>>>>>> 0af896ac212f180da2ae1dfc192a87a38b4bbe3e:chenlb/stock/dataTransform.py
