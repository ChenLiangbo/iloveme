#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
from myClassifier import *

xSample = np.float32(np.load('./npyfile_saver/xSample25.npy'))   #(1253,25)
ySample = np.float32(np.load('./npyfile_saver/ySample25.npy'))   #(1253,1)



x_train = xSample[400:1000,:]
y_train = ySample[400:1000,:]

x_test = xSample[1000:1250,:]
y_test = ySample[1000:1250,:]

knn = MyKNearest()
knn.train(x_train,y_train)
y_predict = knn.predict(x_test)

print "acurracy = ",knn.get_precision(x_test,y_test)
print "knn acurracy = ",np.mean(y_predict == y_test)

svm = MySVM()
# svm.train(x_train,y_train)
# y_predict = svm.predict(x_test)

# print "svm acurracy = ",svm.get_precision(x_test,y_test)
