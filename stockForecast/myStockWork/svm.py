#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os
from myNeurolNetworkModel import MyNeurolNetworkModel

order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume*e-6"}
yahooData = np.load('yahoo_finance5.npy')
Open,High,Low,Close,Volume = np.hsplit(yahooData,5)

shape = yahooData.shape
print "shape = ",shape
print "-"*80

print "Close shape = ",Close.shape

myNNmodel = MyNeurolNetworkModel()
kdj = myNNmodel.calculate_kdj(yahooData)    #(None,3)
print "kdj = ",kdj.shape

declose = myNNmodel.calculate_dclose(Close)  #(None,4)
print "declose = ",declose.shape

logfit = myNNmodel.calculate_logfit(Close)   #(None,1)
print "logfit = ",logfit.shape

closeExponent = myNNmodel.calculate_exponent(Close,exponent = 0.9) #(None,1)
print "closeExponent = ",closeExponent.shape


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
newData = pca.fit_transform(np.hstack([Open,High,Low,Volume]))  #(None,2)
print "newData = ",newData.shape

x_sample = np.hstack([newData,declose,kdj,logfit,closeExponent])
xmax = np.amax(x_sample, axis=0)
xmin = np.amin(x_sample, axis=0)
x_sample = (x_sample - xmin) / (xmax - xmin)


y_sample = np.zeros((Close.shape))
for i in xrange(0,Close.shape[0]-1):
    if Close[i+1,0] > Close[i,0]:
        y_sample[i,0] = 1


x_sample = np.float32(x_sample)
y_sample = np.float32(y_sample)

x_train = np.vstack([x_sample[0:100,:],x_sample[300:500,:],x_sample[700:900,:],x_sample[1100:1150,:]])
y_train = np.vstack([y_sample[0:100,:],y_sample[300:500,:],y_sample[700:900,:],y_sample[1100:1150,:]])

print "x_train.shape = ",x_train.shape
sample_number = x_train.shape[0]

test_start = 900
test_end = 1100
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]


import cv2

svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	               svm_type = cv2.SVM_C_SVC,
                   C=1, gamma=0.5)

svm = cv2.SVM()

svm.train(x_train,y_train,params = svm_params)

y_predict = svm.predict_all(x_test)
print "svm acurracy = ",np.mean(y_test == y_predict)
