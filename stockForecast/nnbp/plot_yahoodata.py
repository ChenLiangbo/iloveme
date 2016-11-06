#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import xlwt
from matplotlib import pyplot as plt

yahooData = np.load('yahooData.npy')
adj_close,high,low,close,openPrice = np.hsplit(yahooData,5)
print "openPrice.shape = ",openPrice.shape

high = high.transpose()
print "high.shape = ",high.shape
low = low.transpose()
adj_close = adj_close.transpose()
openPrice = openPrice.transpose()
close = close.transpose()
x = []
for i in range(high.shape[1]):
    x.append(i)
x = np.asarray([x])
print "x.shape = ",x.shape

plt.plot(x,high,'rx')
plt.plot(x,low,'bx')
plt.plot(x,adj_close,'gx')
plt.plot(x,close,'yx')
plt.plot(x,openPrice,'wx')

plt.grid(True)
plt.legend(['high','low','adj_close','close','openPrice'])
plt.xlabel('reversed-time')
plt.ylabel('Value')
plt.title('Yahoo Finance Data Graph')
plt.show()