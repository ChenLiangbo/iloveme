#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
from myNeurolNetworkModel import MyNeurolNetworkModel

'''
每天的五个数据 high,low,close,open,adj_close,
将其中一个看作是另外４个的函数，使用其余四个来对其中一个进行预测
'''
order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume*e-6"}
yahooData = np.load('yahoo_finance5.npy')
shape = yahooData.shape
print "shape = ",shape
print yahooData[0:10,3]
print "--------------------------------------"
x_sample = np.float32(yahooData[0:shape[0]-1,:])
y_sample = np.float32(np.hsplit(yahooData,5)[3][1:shape[0]])
print "y_sample.shape = ",y_sample.shape
print "x_sample.shape = ",x_sample.shape
print x_sample[0:10,:]
print "--------------------------------------"
print y_sample[0:10,]

# train_start = 700
# train_end = 1100
# y_train = y_sample[train_start:train_end,:]
# x_train = x_sample[train_start:train_end,:]

x_train = np.vstack([x_sample[0:100,:],x_sample[300:500,:],x_sample[700:900,:],x_sample[1100:1150,:]])
y_train = np.vstack([y_sample[0:100,:],y_sample[300:500,:],y_sample[700:900,:],y_sample[1100:1150,:]])

xmax = np.amax(x_train, axis=0)
xmin = np.amin(x_train, axis=0)
x_train = (x_train - xmin) / (xmax - xmin)

    
ymax = np.amax(y_train, axis=0)
ymin = np.amin(y_train, axis=0)
y_train = (y_train - ymin) / (ymax - ymin)


print "x_train.shape = ",x_train.shape
sample_number = x_train.shape[0]

test_start = 1150
test_end = 1200
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]

ann = cv2.ANN_MLP()

layer_sizes = np.int32([5,100,30,1])  
ann.create(layer_sizes)  

params = dict( term_crit = (cv2.TERM_CRITERIA_COUNT, 5000, 0.001),  
                   train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,  
                   bp_dw_scale = 0.01,  
                   bp_moment_scale = 0.0 ) 


ann.train(x_train,y_train,None,params = params)


retval, y_predict = ann.predict(x_test)

y_test_predict = y_predict*(ymax - ymin) + ymin
from matplotlib import pyplot as plt

plt.plot(y_test,'ro')
plt.plot(y_test_predict,'bo')
plt.plot(y_test,'r-')
plt.plot(y_test_predict,'b-')
plt.legend(['y_test','y_test_predict'])
plt.grid(True)
plt.xlabel('index')
plt.ylabel('value')
plt.title('MyNeurolNetworkModel Predict Close With AllOneDay')
# plt.savefig(outdir + 'close.jpg')
plt.show()

