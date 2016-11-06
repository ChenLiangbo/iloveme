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
y_sample = Close[1:]

x_train = np.vstack([x_sample[0:100,:],x_sample[300:500,:],x_sample[700:900,:],x_sample[1100:1150,:]])
y_train = np.vstack([y_sample[0:100,:],y_sample[300:500,:],y_sample[700:900,:],y_sample[1100:1150,:]])

print "x_train.shape = ",x_train.shape
sample_number = x_train.shape[0]

test_start = 1150
test_end = 1200
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]

from sklearn.linear_model import LinearRegression,LogisticRegression
# model = LogisticRegression()
model = LinearRegression()
model.fit(x_train, y_train)
print "myNNmodel train successfully ..."
y_test_predict = model.predict(x_test)

print "y_test_predict = ",y_test_predict.shape

outdir = './images/'
from matplotlib import pyplot as plt
plt.plot(y_test,'ro')
plt.plot(y_test_predict,'bo')
plt.plot(y_test,'r-')
plt.plot(y_test_predict,'b-')
plt.legend(['y_test','y_predict'])
plt.grid(True)
plt.xlabel('index')
plt.ylabel('value')
plt.title('LinearRegression')
plt.savefig(outdir + 'LinearRegression1150-1200.jpg')
plt.show()