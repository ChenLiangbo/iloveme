#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from myClassifier import BayesClassifier


xSample = np.load('../dataset/trX.npy').astype(np.float32)
ySample = np.load('../dataset/trY.npy').astype(np.float32)

# xSample = np.load('../dataset/trX.npy').astype(np.float32)
# ySample = np.load('../dataset/trY.npy').astype(np.float32)

shape = xSample.shape
print "xSample.shape = ",(xSample.shape,xSample.dtype)
print "ySample.shape = ",(ySample.shape,ySample.dtype)

x = np.zeros((xSample.shape[0],xSample.shape[1]*xSample.shape[2]))

for row in xrange(shape[0]):
    array9x9 = xSample[row,:,:]
    xmax = np.amax(array9x9,axis = 0)
    array9x9 = array9x9/xmax
    x[row,:] = array9x9.reshape(1,shape[1]*shape[2])


x = x.astype(np.float32)
ySample = ySample.astype(np.float32)

x_train = x[0:300,:]
y_train = ySample[0:300,:]

x_test = x[300:500,:]
y_test = ySample[300:500,:]



classifier = BayesClassifier()
classifier.saveNeeded = False
classifier.sectionNumber = 10
classifier.localDays = 15
classifier.train(x_train,y_train)
print "="*80
print "classifier train succefully ..."
print "-"*100
y_predict = classifier.predict(x_test)
print "-"*100

# print "y_predict = ",y_predict
print "y_predict = ",y_predict.shape
accuracy = (y_test == y_predict)

print "BayesClassifier accuracy = ",np.mean(accuracy)
print "-"*100


from sklearn.naive_bayes import BernoulliNB  
clf = BernoulliNB()  
clf.fit(x_train, y_train.ravel())  
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)  
predict =  clf.predict(x_test) 

acurracy = (y_test == predict.ravel())

print "BernoulliNB acurracy = %f" % (np.mean(acurracy)) 


import cv2
svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	               svm_type = cv2.SVM_C_SVC,
                   C=3.67, gamma = cv2.SVM_GAMMA)

svm = cv2.SVM()
svm.train(x_train,y_train)
ret = svm.predict_all(x_test)
acurracy = (y_test == ret)
print "svm acurracy = ",np.mean(acurracy)

knn = cv2.KNearest()
knn.train(x_train,y_train)
ret, results, neighbours ,dist = knn.find_nearest(x_test, 1)
acurracy = (y_test == ret)
print "knn acurracy = ",np.mean(acurracy)

from sklearn.naive_bayes import MultinomialNB  
clf = MultinomialNB().fit(np.abs(x_train), y_train.ravel())  
predict =  clf.predict(np.abs(x_test)) 
acurracy = (y_test == predict.ravel())
print "MultinomialNB acurracy = %f" % (np.mean(acurracy))



from sklearn.naive_bayes import GaussianNB  
clf = GaussianNB().fit(x_train, y_train.ravel())  
predict =  clf.predict(x_test) 
acurracy = (y_test == predict.ravel())
print "GaussianNB acurracy = %f" % (np.mean(acurracy))
