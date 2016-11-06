#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os
from myNeurolNetworkModel import MyNeurolNetworkModel


yahooData = np.load('yahoo_finance5.npy')
shape = yahooData.shape
print "yahooData shape = ",shape
Open,High,Low,Close,Volume = np.hsplit(yahooData,5)

x_sample = np.hstack([Open,High,Low,Volume])
x_sample = x_sample[0:shape[0]-1,:]
y_sample = Close[1:shape[0],:]

x_train = np.vstack([x_sample[0:100,:],x_sample[300:500,:],x_sample[700:900,:],x_sample[1100:1250,:]])
y_train = np.vstack([y_sample[0:100,:],y_sample[300:500,:],y_sample[700:900,:],y_sample[1100:1250,:]])

print "x_train.shape = ",x_train.shape
sample_number = x_train.shape[0]

test_start = 900
test_end = 1100
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]

outdir = './images/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

myNNmodel = MyNeurolNetworkModel()
myNNmodel.inputNumber = 4

# myNNmodel.outdir = outdir
myNNmodel.errorRate = 0.01111
myNNmodel.learningRate = 0.001
# myNNmodel.train(x_train,y_train)

print "myNNmodel train successfully ..."


y_test_predict = myNNmodel.predict(x_test)
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
plt.savefig(outdir + 'close.jpg')
plt.show()

acuracy = (y_test_predict - y_test)/y_test
plt.plot(acuracy,'ro')
plt.plot(acuracy,'b-')
plt.legend('acuracy')
plt.title('MyNeurolNetworkModel Test Acuracy')
plt.xlabel('index')
plt.ylabel(['acuracy',])
plt.savefig(outdir + 'predict_close_acurracy')
plt.grid(True)
plt.show()
print "mean acuracy = ",np.mean(np.abs(acuracy))

