#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

'''
每天的五个数据 high,low,close,open,adj_close,
将其中一个看作是另外４个的函数，使用其余四个来对其中一个进行预测
'''
xSample = np.load('./npyfile_saver/xSample25.npy')
ySample = np.load('./npyfile_saver/ySample25.npy')

xSample = xSample.astype(np.float32)*(1e-2)
ySample = ySample.astype(np.float32)*(1e-2)

print "xSample.shape = ",xSample.shape
print "ySample.shape = ",ySample.shape

train_start = 0
train_end = 1000

x_train = xSample[train_start:train_end,:]
y_train = ySample[train_start:train_end,:]

test_start = 900
test_end = 1250

x_test = xSample[test_start:test_end,:]
y_test = ySample[test_start:test_end,:]
print "x_test.shape = ",x_test.shape

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
plt.savefig(outdir + 'close25predict.jpg')
plt.show()

acuracy = (y_test_predict - y_test)/y_test
plt.plot(acuracy,'ro')
plt.plot(acuracy,'r-')
plt.legend('acuracy')
plt.title('MyNeurolNetworkModel Test Acuracy')
plt.xlabel('index')
plt.ylabel(['acuracy'])
plt.savefig(outdir + 'close25Acurracy')
plt.grid(True)
plt.show()
print "mean acuracy = ",np.mean(np.abs(acuracy))