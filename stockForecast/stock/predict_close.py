#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os
from myNeurolNetworkModel import MyNeurolNetworkModel

'''
每天的五个数据 high,low,close,open,adj_close,用前一天的数据预测未来一天的收盘价

'''
order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume*e-6"}
yahooData = np.load('yahoo_finance5.npy')
shape = yahooData.shape
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

# train_start = 700
# train_end = 1100
# y_train = y_sample[train_start:train_end,:]
# x_train = x_sample[train_start:train_end,:]

x_train = np.vstack([x_sample[0:100,:],x_sample[300:500,:],x_sample[700:900,:],x_sample[1100:1150,:]])
y_train = np.vstack([y_sample[0:100,:],y_sample[300:500,:],y_sample[700:900,:],y_sample[1100:1150,:]])

print "x_train.shape = ",x_train.shape
sample_number = x_train.shape[0]

test_start = 1150
test_end = 1200
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]

outdir = os.path.join(os.path.dirname(__file__),'predict_close/')

myNNmodel = MyNeurolNetworkModel()
# myNNmodel.outdir = outdir
myNNmodel.errorRate = 0.0105
myNNmodel.learningRate = 0.001


print "myNNmodel outdir = ",myNNmodel.outdir

if not os.path.exists(outdir):
    os.mkdir(outdir)

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
plt.plot(acuracy,'r-')
plt.legend('acuracy')
plt.title('MyNeurolNetworkModel Test Acuracy')
plt.xlabel('index')
plt.ylabel(['acuracy'])
plt.savefig(outdir + 'predict_close_acurracy')
plt.grid(True)
plt.show()
print "mean acuracy = ",np.mean(np.abs(acuracy))

print "knn acuracy after nn = ",myNNmodel.knn_acurracy(x_test,y_test)