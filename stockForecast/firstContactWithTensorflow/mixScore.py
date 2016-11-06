#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import input_data
import numpy as np
import cv2


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sample = mnist.train.next_batch(5000)

xSample = sample[0]
ySample = sample[1]

yshape = ySample.shape
y = np.zeros((ySample.shape[0],1),dtype = np.float32)

for row in range(yshape[0]):
    index = np.argwhere(ySample[row] == 1)
    index = index[0,0]
    y[row,0] = index

# print "y = ",y

train_number = 4000
x_train = xSample[0:train_number,:]
y_train = y[0:train_number,:]

x_test = xSample[train_number:,:]
y_test = y[train_number:,:]



knn = cv2.KNearest()
knn.train(x_train,y_train)
ret, results, neighbours ,dist = knn.find_nearest(x_test,1)

# results = results.ravel()
acurracy = (y_test == results)

print "knn acurracy = ",np.mean(acurracy)
knn_result = results


svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	               svm_type = cv2.SVM_C_SVC,
                   C=1, gamma=0.5)

svm = cv2.SVM()

svm.train(x_train,y_train,params = svm_params)
# svm.save('svm_mnist_model.dat')


ret = svm.predict_all(x_test)
results =ret

acurracy = (y_test == results)

print "svm acurracy = ",np.mean(acurracy)
svm_result = results



# random trees
RTtree = cv2.RTrees()

rtree_params = dict(depth = 20)
var_type = np.array([cv2.CV_VAR_NUMERICAL]*xSample.shape[1] + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)


RTtree.train(x_train,cv2.CV_ROW_SAMPLE,y_train,
    	varType = var_type,
    	params = rtree_params
    	)

results = np.zeros(y_test.shape,dtype = y_test.dtype)
for i in xrange(y_test.shape[0]):
    results[i,0] = RTtree.predict(x_test[i,:])


acurracy = (y_test == results)

print "RTtree acurracy = ",np.mean(acurracy)

RTtree_result = results

'''
boost_params = dict(max_depth = 10)
var_type = np.array([cv2.CV_VAR_NUMERICAL]*xSample.shape[1] + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)
Boost = cv2.Boost()
Boost.train(x_train,cv2.CV_ROW_SAMPLE,y_train,varType = var_type,params = boost_params)

results = np.zeros(y_test.shape,dtype = y_test.dtype)

for i in xrange(y_test.shape[0]):
    results[i,0] = Boost.predict(x_test[i,:],returnSum = True)

acurracy = (y_test == results)

print "Boost acurracy = ",np.mean(acurracy)

boost_result = results
'''

score = np.zeros(results.shape,dtype = np.float32)
weights = np.array([0.915,0.908,0.849])
weights = weights/np.sum(weights)

score = knn_result*weights[0] + svm_result*weights[1] + RTtree_result*weights[2]
score = np.round(score)


acurracy = (y_test == score)

print "mix acurracy = ",np.mean(acurracy)

