#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf

xSample = np.load('./npyfile_saver/xSample25.npy')
ySample = np.load('./npyfile_saver/ySample25.npy')

xSample = xSample.astype(np.float32)
ySample = ySample.astype(np.float32)

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


'''
knn = cv2.KNearest()
knn.train(x_train,y_train)
ret, results, neighbours ,dist = knn.find_nearest(x_test, 1)

'''
svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	               svm_type = cv2.SVM_C_SVC,
                   C=2.67, gamma=3.383 )

svm = cv2.SVM()

svm.train(x_train,y_train)
svm.save('./model_saver/svm_data.dat')

ret = svm.predict_all(x_test)
print "ret = ",ret[0:10,:]
accuracy = (y_test == ret)


print "accuracy = ",np.mean(accuracy)
