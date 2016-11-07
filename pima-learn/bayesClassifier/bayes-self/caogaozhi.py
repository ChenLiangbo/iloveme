#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
from bayesClassifier import BayesClassifier

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
classifier.sectionNumber = 32
classifier.train(x_train,y_train)
print "classifier train succefully ..."
y_predict = classifier.predict(x_test)

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



from sklearn.naive_bayes import MultinomialNB  
clf = MultinomialNB().fit(np.abs(x_train), y_train.ravel())  
predict =  clf.predict(np.abs(x_test)) 
acurracy = (y_test == predict.ravel())
print "MultinomialNB acurracy = %f" % (np.mean(acurracy))


from sklearn.naive_bayes import GaussianNB  
clf = GaussianNB().fit(x_train, y_train.ravel())  
predict =  clf.predict(x_test) 
print "predict = ",(predict.shape,predict.dtype,y_test.shape,y_test.dtype)
acurracy = (y_test == predict.ravel())
print "GaussianNB acurracy = %f" % (np.mean(acurracy))

count = 0
for i in xrange(y_test.shape[0]):
    if int(predict[i]) == int(y_test[i]):
        count = count + 1
print "GaussianNB acurracy = ",float(count)/y_test.shape[0]
