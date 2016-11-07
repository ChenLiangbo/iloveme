#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
from myNeurolNetworkModel import MyNeurolNetworkModel

dataset = np.load('./dataset/pima-indians.npy')

columns = np.hsplit(dataset,9)
xsample = np.hstack(columns[0:8])
ysample = columns[8]
shape = xsample.shape
print "xsample = ",xsample.shape
print "ysample = ",ysample.shape

# indexList = np.random.permutation(shape[0])
indexList = range(shape[0])

x_train = xsample[indexList[0:538]]
y_train = ysample[indexList[0:538]]
print "x_train.shape = ",x_train.shape
print "y_train.shape = ",y_train.shape

x_test = xsample[indexList[538:]]
y_test = ysample[indexList[538:]]
print "x_test.shape = ",x_test.shape
print "y_test.shape = ",y_test.shape

myNNmodel = MyNeurolNetworkModel()
myNNmodel.errorRate = 0.918
myNNmodel.layerOne  = 15
myNNmodel.isDropout = True
myNNmodel.learningRate = 0.001
myNNmodel.trainTimes = 7000

myNNmodel.train(x_train,y_train,x_test,y_test)

print "train model successfully!"