#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd

basedir = os.getcwd()
filename = os.path.join(basedir, 'stock_fanny.xlsx')
xls = pd.ExcelFile(filename)

df_train = xls.parse('Sheet4', index_col='Date') # train

close = df_train['Close']

print "Close = ",len(close)
# print close

closeArray = np.array([close]).reshape(len(close),1)

print "closeArray = ",closeArray.shape
print closeArray

print "========================================================="


shape = closeArray.shape

related = 5

x_sample = np.zeros((shape[0]-related,5))
y_sample = np.zeros((shape[0]-related,1))

for i in xrange(shape[0] - related):
    x_sample[i,:] = closeArray[i:i+related,0].reshape(1,related)
    y_sample[i,0] = closeArray[i+related,0]

print "x_sample.shape = ",x_sample.shape
print x_sample
print "-------------------------------------------------------------------"
print "y_sample.shape = ",y_sample.shape
print y_sample


outdir = './sample/'
np.save(outdir + 'x_sample',x_sample)
np.save(outdir + 'y_sample',y_sample)