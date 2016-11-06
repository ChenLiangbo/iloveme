#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from itertools import izip
import cv2


x_sample = np.float32(np.load('./npyfile/xSample.npy'))
y_sample = np.float32(np.load('./npyfile/ySample.npy'))
print "y_sample.shape = ",y_sample.shape
print "x_sample.shape = ",x_sample.shape
# print x_sample[0:10,:]
print "--------------------------------------"
# print y_sample[0:10,]
y = np.zeros(y_sample.shape,dtype = np.float32)

for i in xrange(y_sample.shape[0] - 1):
    if y_sample[i+1,0] < y_sample[i,0]:
        y[i,0] = 0
    else:
        y[i,0] = 1

y_sample = y[0:y_sample.shape[0]-1,:]
x_sample = x_sample[0:y_sample.shape[0]-1,:]
print "y_sample.shape = ",y_sample.shape
print "x_sample.shape = ",x_sample.shape
x_sample = np.float32(x_sample)
y_sample = np.float32(y_sample)


train_start = 0
train_end = 1150
y_train = y_sample[train_start:train_end,:]
x_train = x_sample[train_start:train_end,:]
sample_number = x_train.shape[0]

test_start = 1000
test_end = 1200
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]


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


knn = cv2.KNearest()
knn.train(x_train,y_train)
ret, results, neighbours ,dist = knn.find_nearest(x_test,1)

# results = results.ravel()
acurracy = (y_test == results)

print "knn acurracy = ",np.mean(acurracy)

length = 10

RTtree = cv2.RTrees()
rtree_params = dict(depth = 32)
var_type = np.array([cv2.CV_VAR_NUMERICAL]*length + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)

RTtree.train(x_train,cv2.CV_ROW_SAMPLE,y_train,
    	varType = var_type,
    	params = rtree_params)

results = np.zeros(y_test.shape,dtype = y_test.dtype)
for i in xrange(y_test.shape[0]):
    results[i,0] = RTtree.predict(x_test[i,:])

acurracy = (y_test == results)
print "RTtree acurracy = ",np.mean(acurracy)


boost_params = dict(max_depth = 1)
var_type = np.array([cv2.CV_VAR_NUMERICAL]*length + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)
Boost = cv2.Boost()
Boost.train(x_train,cv2.CV_ROW_SAMPLE,y_train,varType = var_type,params = boost_params)

results = np.zeros(y_test.shape,dtype = y_test.dtype)

for i in xrange(y_test.shape[0]):
    results[i,0] = Boost.predict(x_test[i,:],returnSum = True)

for i in xrange(y_test.shape[0]):
    if results[i,0] >0:
        results[i,0] = 1
    else:
        results[i,0] = 0

acurracy = (y_test == results)

print "Boost acurracy = ",np.mean(acurracy)