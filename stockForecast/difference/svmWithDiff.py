#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import cv2

xSample = np.load('./npyfile/xSample20.npy').astype(np.float32)
ySample = np.load('./npyfile/ySample20.npy')
print "xSample.shape = ",(xSample.shape,xSample.dtype)
print xSample[0:3,:]
print "ySample.shape = ",(ySample.shape,ySample.dtype)
print ySample[0:10,:]
# print "ySample = ",ySample[0:10]

train_number = 1000
x_train = xSample[2:train_number,:]
y_train = ySample[2:train_number,:]
# print "x_train.shape = ",x_train.shape
# print "y_train.shape = ",y_train.shape

x_test = xSample[train_number:1258,:]
y_test = ySample[train_number:1258,:]


'''
# svm
svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	               svm_type = cv2.SVM_C_SVC,
                   C=3.67, gamma = cv2.SVM_GAMMA)

svm = cv2.SVM()
svm.train(x_train,y_train,params = svm_params)
# svm.save('svm_model.dat')
ret = svm.predict_all(x_test)
# print "ret = ",ret
acurracy = (y_test == ret)

print "svm acurracy = ",np.mean(acurracy)



'''
knn = cv2.KNearest()
knn.train(x_train,y_train)
ret, results, neighbours ,dist = knn.find_nearest(x_test, 3)
print "results = ",results[0:10]
print "ret = ",ret
acurracy = (y_test == ret)

print "knn acurracy = ",np.mean(acurracy)


'''
shape = xSample.shape

RTtree = cv2.RTrees()

var_types = np.array([cv2.CV_VAR_NUMERICAL] * shape[1] + [cv2.CV_VAR_CATEGORICAL], np.uint8)
params = dict(max_depth=50)
RTtree.train(x_train, cv2.CV_ROW_SAMPLE, y_train, varType = var_types, params = params)

results = np.zeros(y_test.shape,dtype = y_test.dtype)
for i in xrange(y_test.shape[0]):
    results[i,0] = RTtree.predict(x_test[i,:])
print "results = ",results[0:10]
acurracy = (y_test == results)

print "acurracy = ",np.mean(acurracy)
'''