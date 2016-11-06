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

'''

from sklearn.naive_bayes import GaussianNB  
clf = GaussianNB().fit(x_train, y_train.ravel())  
predict =  clf.predict(x_test) 
count = 0
for i in xrange(y_test.shape[0]):
    if y_test[i,0] == predict[i]:
        count = count + 1

print "GaussianNB acurracy = ",float(count)/y_test.shape[0]

from sklearn.naive_bayes import MultinomialNB  
clf = MultinomialNB().fit(np.abs(x_train), y_train.ravel())  
predict =  clf.predict(np.abs(x_test)) 

count = 0
for i in xrange(y_test.shape[0]):
    if y_test[i,0] == predict[i]:
        count = count + 1

print "MultinomialNB acurracy = ",float(count)/y_test.shape[0]



from sklearn.naive_bayes import BernoulliNB  
clf = BernoulliNB()  
clf.fit(x_train, y_train.ravel())  
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)  
predict =  clf.predict(x_test) 

count = 0
for i in xrange(y_test.shape[0]):
    if y_test[i,0] == predict[i]:
        count = count + 1

print "BernoulliNB acurracy = ",float(count)/y_test.shape[0]
'''

'''
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
    n_estimators=600,learning_rate=1)

bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
    n_estimators=600,learning_rate=1.5,algorithm="SAMME")

bdt_real.fit(X_train, y_train.ravel())
bdt_discrete.fit(X_train, y_train.ravel())
real_predict = bdt_real.staged_predict(x_test)
discrete_predict = bdt_discrete.staged_predict(x_test))
count = 0
for i in xrange(y_test.shape[0]):
    if y_test[i,0] == real_predict[i]:
        count = count + 1

print "AdaBoostClassifier real acurracy = ",float(count)/y_test.shape[0]

count = 0
for i in xrange(y_test.shape[0]):
    if y_test[i,0] == discrete_predict[i]:
        count = count + 1

print "AdaBoostClassifier discrete acurracy = ",float(count)/y_test.shape[0]

'''

from sklearn import svm


clf = svm.SVC(gamma=0.001, C=100)
clf.fit(x_train, y_train.ravel())
predict = clf.predict(x_test)
count = 0
for i in xrange(y_test.shape[0]):
    if y_test[i,0] == predict[i]:
        count = count + 1
print "BernoulliNB acurracy = ",float(count)/y_test.shape[0]

