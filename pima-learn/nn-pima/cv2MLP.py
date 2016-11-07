#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
from bayesClassifier import BayesClassifier
import cv2

dataset = np.load('pima-indians.npy')

columns = np.hsplit(dataset,9)
xsample = np.hstack(columns[0:8])
ysample = columns[8]
shape = xsample.shape
xsample = np.float32(xsample)
ysample = np.float32(ysample)
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

myBayes = BayesClassifier()

layers = np.array([8,15,1])

model = cv2.ANN_MLP()
model.create(layers)

params = dict( term_crit = (cv2.TERM_CRITERIA_COUNT, 3000, 0.01),  
               train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,   
               bp_dw_scale = 0.001,  
               bp_moment_scale = 0.0 )

model.train(x_train,y_train,None,params = params)

ret,resp = model.predict(x_test)

y_predict = resp.argmax(-1)
print "y_predict = ",(y_predict.shape,np.mean(y_predict == y_test))
print y_predict[0:10]
result = myBayes.f_measure(y_predict,y_test)
print "result = ",result
