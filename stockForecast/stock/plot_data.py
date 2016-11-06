#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

'''
每天的五个数据 high,low,close,open,adj_close,
将其中一个看作是另外４个的函数，使用其余四个来对其中一个进行预测
'''
order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume*e-6"}
yahooData = np.load('yahoo_finance5.npy')
shape = yahooData.shape
Open,High,Low,Close,Volume = np.hsplit(yahooData,5)
print "shape = ",shape
print yahooData[0:10,3]
print "--------------------------------------"
x_sample = yahooData[0:shape[0]-1,:]
y_sample = np.hsplit(yahooData,5)[3][1:shape[0]]
print "y_sample.shape = ",y_sample.shape
print "x_sample.shape = ",x_sample.shape
print x_sample[0:10,:]
print "--------------------------------------"
print y_sample[0:10,]

train_start = 1100
train_end = 1200
y_train = y_sample[train_start:train_end,:]
x_train = x_sample[train_start:train_end,:]
print "x_train.shape = ",x_train.shape
sample_number = x_train.shape[0]

test_start = 1000
test_end = 1250
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]

print "x_test.shape = ",x_test.shape

x = np.linspace(0,x_train.shape[0],x_train.shape[0]).reshape(x_train.shape[0],1)
from matplotlib import pyplot as plt

plt.plot(x,Open[train_start:train_end,:],'ro')
# plt.plot(x,Close[train_start:train_end,:],'bo')
plt.plot(x,y_train,'bo')
plt.plot(x,High[train_start:train_end,:],'go')

plt.plot(x,Open[train_start:train_end,:],'r-')
# plt.plot(x,Close[train_start:train_end,:],'b-')
plt.plot(x,y_train,'b-')
plt.plot(x,High[train_start:train_end,:],'g-')


plt.grid(True)
plt.legend(['Open','Close','High',])
plt.xlabel('index-time')
plt.ylabel('yahoo_finance5')
title = 'The data of yahoo_finance5'
plt.title(title)

plt.savefig('./npyfile_saver/yahoo_finance5.jpg')
plt.show()