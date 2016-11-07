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
myNNmodel.learningRate = 0.001
myNNmodel.trainTimes = 4000
# myNNmodel.batchSize  = 20


y_predict = myNNmodel.predict(x_test)
print y_predict[0:10]

np.save('./npyfile/y_predict1',y_predict)

result = myNNmodel.f_measure(y_predict,y_test)
print "result = ",result


from sklearn.linear_model import LinearRegression,LogisticRegression
# model = LogisticRegression()
model = LinearRegression()
model.fit(x_train, y_train)
print "myNNmodel train successfully ..."
y_predict = model.predict(x_test)
for i in xrange(y_predict.shape[0]):
    if y_predict[i] >= 0.5:
        y_predict[i] = 1
    else:
        y_predict[i] = 0
result = myNNmodel.f_measure(y_predict,y_test)
print "LinearRegression result = ",result
