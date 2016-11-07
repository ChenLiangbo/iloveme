#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from bayesClassifier import BayesClassifier
from sklearn.naive_bayes import BernoulliNB 

dataset = np.load('pima-indians.npy')


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

classifier = BayesClassifier()
classifier.saveNeeded = False
classifier.saveNeeded = 20
classifier.train(x_train,y_train)
print "classifier train succefully ..."
y_predict = classifier.predict(x_test)
result = classifier.f_measure(y_predict,y_test)
print "BayesClassifier result = ",result

