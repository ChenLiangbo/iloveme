#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os
from myNeurolNetworkModel import MyNeurolNetworkModel
from myClassifier import BayesClassifier

order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume*e-6"}
yahooData = np.load('../dataset/yahoo_finance5.npy')
Open,High,Low,Close,Volume = np.hsplit(yahooData,5)
shape = yahooData.shape
print "Close shape = ",Close.shape

myNNmodel = MyNeurolNetworkModel()
kdj = myNNmodel.calculate_kdj(yahooData)    #(None,3)
print "kdj = ",kdj.shape

declose = myNNmodel.calculate_dclose(Close)  #(None,4)
print "declose = ",declose.shape

logfit = myNNmodel.calculate_logfit(Close)   #(None,1)
print "logfit = ",logfit.shape

closeExponent = myNNmodel.calculate_exponent(Close,exponent = 0.9) #(None,1)
print "closeExponent = ",closeExponent.shape

y_sample = np.zeros((Close.shape))
for i in xrange(shape[0]-1):
    if Close[i+1,:] < Close[i,:]:
        y_sample[i,:] = 0
    else:
        y_sample[i,:] = 1
# print "y_sample = ",y_sample[0:10]
# x_sample = np.hstack([yahooData,declose,kdj,logfit,closeExponent])
x_sample = np.hstack([yahooData,closeExponent])
# print "x_sample = ",x_sample[0:3]
print "x_sample.shape = ",x_sample.shape
print "y_sample.shape = ",y_sample.shape

x_sample = np.float32(x_sample)
y_sample = np.float32(y_sample)

# indexList = np.random.permutation(shape[0])
indexList = range(shape[0])

x_train = x_sample[indexList[0:881]]
y_train = y_sample[indexList[0:881]]
print "x_train.shape = ",x_train.shape
print "y_train.shape = ",y_train.shape

x_test = x_sample[indexList[881:]]
y_test = y_sample[indexList[881:]]
print "x_test.shape = ",x_test.shape
print "y_test.shape = ",y_test.shape

print "="*80


classifier = BayesClassifier()
classifier.saveNeeded = False
classifier.sectionNumber = 17
classifier.localDays = 15
# classifier.train(x_train,y_train)
classifier.load()
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
'''

result = []
start = 5
for i in range(start,50):
    classifier = BayesClassifier()
    classifier.saveNeeded = False
    classifier.sectionNumber = i
    classifier.localDays = 15
    try:
        classifier.train(x_train,y_train)
        y_predict = classifier.predict(x_test)
        accuracy = (y_test == y_predict)
        result.append(np.mean(accuracy))
    except Exception,ex:
        print "[Exception] ",str(ex)
        result.append(0.5)    
    print "i = ",i
    print "-"*100
from matplotlib import pyplot as plt
x = range(start,50)
plt.plot(x,result,'ro')
plt.plot(x,result,'r-')
plt.grid(True)
plt.legend(['sectionNumber',])
plt.xlabel('Value of sectionNumber')
plt.ylabel('acurracy')
plt.title('Distribution of Acurracy With sectionNumber')
plt.savefig('../image/sectionNumber-accuracy')
plt.show()
'''