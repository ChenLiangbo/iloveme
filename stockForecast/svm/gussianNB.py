#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import cv2

xSample = np.load('trX.npy').astype(np.float32)
ySample = np.load('trY.npy').astype(np.float32)




print "xSample.shape = ",(xSample.shape,xSample.dtype)
print "ySample.shape = ",(ySample.shape,ySample.dtype)


# x = np.zeros((xSample.shape[0],xSample.shape[1]*xSample.shape[2]))

# for row in xrange(xSample.shape[0]):
#     array9x9 = xSample[row,:,:]
#     xmax = np.amax(array9x9,axis = 0)
#     array9x9 = array9x9/xmax
#     x[row,:] = array9x9.reshape(1,81)

x = np.zeros((xSample.shape[0],xSample.shape[1]*xSample.shape[2]))

for row in xrange(xSample.shape[0]):
    x[row,:] = xSample[row,:,:].reshape(1,81)

from sklearn import preprocessing
x = preprocessing.normalize(x)
# x = preprocessing.scale(x)

x = x.astype(np.float32)
ySample = ySample.astype(np.float32)

x_train = x[100:300,:]
y_train = ySample[100:300,:]
x_test = np.vstack([x[0:100,:],x[300:400,:]])
y_test = np.vstack([ySample[0:100,:],ySample[300:400,:]])
print "x.shape = ",x.shape

'''
from sklearn.naive_bayes import GaussianNB  
clf = GaussianNB().fit(x_train, y_train.ravel())  
predict =  clf.predict(x_test) 

acurracy = (y_test == predict.ravel())
print "GaussianNB acurracy = %f" % (np.mean(acurracy))
'''
# from sklearn.naive_bayes import MultinomialNB  
# clf = MultinomialNB().fit(np.abs(x_train), y_train.ravel())  
# predict =  clf.predict(np.abs(x_test)) 

# acurracy = (y_test == predict.ravel())

# print "MultinomialNB acurracy = %f" % (np.mean(acurracy))
'''
from sklearn.naive_bayes import BernoulliNB  
clf = BernoulliNB()  
clf.fit(x_train, y_train.ravel())  
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)  
predict =  clf.predict(x_test) 

acurracy = (y_test == predict.ravel())

print "BernoulliNB acurracy = %f" % (np.mean(acurracy)) 


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

depthlist = range(1,20)
ada_acurracy = []

for depth in depthlist:
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
                         algorithm="SAMME",
                         n_estimators=200)


    bdt.fit(x_train,y_train.ravel())
    predict = bdt.predict(x_test)
    count = 0
    for i in xrange(y_test.shape[0]):
        if y_test[i,0] == predict[i]:
            count = count + 1
    ada_acurracy.append(float(count)/y_test.shape[0])
print "AdaBoostClassifier acurracy = ",float(count)/y_test.shape[0]

max_acurracy = max(ada_acurracy)
index = ada_acurracy.index(max_acurracy)
print "max_acurracy = %f,max_depth = %d" % (max_acurracy,depthlist[index])

# from matplotlib import pyplot as plt

# plt.plot(depthlist,ada_acurracy, 'r-')
# plt.plot(depthlist,ada_acurracy, 'ro')
# plt.xlabel('depthlist')
# plt.ylabel('ada_acurracy')
# plt.title('Adjust AdaBoostClassifier Acurracy with depth')
# plt.legend(['depth'])
# plt.grid(True)
# plt.savefig('./ploter/AdaBoostClassifier_depth.jpg')
# plt.show()


from sklearn import svm
max_c = 1
max_g = 0.001

clf = svm.SVC(gamma=max_g, C=max_c)
clf.fit(x_train, y_train.ravel())
predict = clf.predict(x_test)
count = 0
for i in xrange(y_test.shape[0]):
    if y_test[i,0] == predict[i]:
        count = count + 1
print "sklearn svm acurracy = " ,float(count)/y_test.shape[0]
'''
'''
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train,y_train.ravel())

predict = knn.predict(x_test)
count = 0
for i in xrange(y_test.shape[0]):
    if y_test[i,0] == predict[i]:
        count = count + 1
print "sklearn KNeighborsClassifier acurracy = " ,float(count)/y_test.shape[0]
# KNeighborsClassifier acurracy = 0.605
'''

'''
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(x_train,y_train.ravel())
predict = regr.predict(x_test)
thresh = np.linspace(0,1,20)
acurracy = []

t = 0.5
for i in xrange(predict.shape[0]):
    if predict[i] > t:
        predict[i] = 1
    else:
        predict[i] = 0
count = 0
for i in xrange(y_test.shape[0]):
    if y_test[i,0] == predict[i]:
        count = count + 1

print "sklearn LinearRegression acurracy = " ,float(count)/y_test.shape[0]
# LinearRegression acurracy = 0.6
'''

from sklearn import linear_model
regr = linear_model.Lasso()