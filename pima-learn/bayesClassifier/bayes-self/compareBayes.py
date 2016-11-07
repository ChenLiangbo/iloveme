#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from bayesClassifier import BayesClassifier
from sklearn.naive_bayes import BernoulliNB 
from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import MultinomialNB

dataset = np.load('pima-indians.npy')


columns = np.hsplit(dataset,9)
xsample = np.hstack(columns[0:8])
ysample = columns[8]
shape = xsample.shape

print "xsample = ",xsample.shape
print "ysample = ",ysample.shape


orders = ["precision",'recall','accuracy','fmeasure']
number = 20
PRAFarray = np.zeros((number,16))
for i in xrange(number):
    indexList = np.random.permutation(shape[0])
    # indexList = range(shape[0])
    print "i = ",i
    x_train = xsample[indexList[0:538]]
    y_train = ysample[indexList[0:538]]
    x_test = xsample[indexList[538:]]
    y_test = ysample[indexList[538:]]

    classifier = BayesClassifier()
    classifier.saveNeeded = False
    classifier.sectionNumber = 39
    classifier.train(x_train,y_train)
    y_predict = classifier.predict(x_test)
    result = classifier.f_measure(y_predict,y_test)
    print (np.mean(y_test == y_predict),result["accuracy"])
    for key in orders:
        index = orders.index(key)
        PRAFarray[i,index] = result[key]
    

    clf = GaussianNB().fit(x_train, y_train.ravel())  
    y_predict =  clf.predict(x_test)
    result = classifier.f_measure(y_predict,y_test)
    print (np.mean(y_test == y_predict),result["accuracy"])
    for key in orders:
        index = orders.index(key) + 4
        PRAFarray[i,index] = result[key]

    try:
        clf = BernoulliNB().fit(x_train, y_train.ravel())  
        y_predict =  clf.predict(x_test)
        result = classifier.f_measure(y_predict,y_test)
        for key in orders:
            index = orders.index(key) + 8
            PRAFarray[i,index] = result[key]
    except:
        for key in orders:
            index = orders.index(key) + 8
            PRAFarray[i,index] = 0.5

    clf = MultinomialNB().fit(x_train, y_train.ravel())  
    y_predict =  clf.predict(x_test)
    result = classifier.f_measure(y_predict,y_test)
    print (np.mean(y_test == y_predict),result["accuracy"])
    for key in orders:
        index = orders.index(key) + 12
        PRAFarray[i,index] = result[key]
    print "-"*80
    # break
    
shapes1 = ['ro','go','bo','mo']
shapes2 = ['r-','g-','b-','m-']
from matplotlib import pyplot as plt

for j in xrange(4):
    classfierOrder = ["myBayes","GaussianNB","BernoulliNB","MultinomialNB"]
    name = orders[j]
    for i in xrange(4):
        plt.plot(PRAFarray[:,j+i*4],shapes1[i])
    plt.legend(classfierOrder)
    for i in xrange(4):
        plt.plot(PRAFarray[:,j+i*4],shapes2[i])
    plt.xlabel('Experiment Number')
    plt.ylabel(name)
    plt.title(name + " of Four BayesClassifier")
    plt.grid(True)
    plt.savefig('../images/compareBayes-' + name)
    plt.show()

import xlwt
book = xlwt.Workbook(encoding = 'utf-8')
sheet1 = book.add_sheet('Sheet1',cell_overwrite_ok = False)
columns =["myBayes","GaussianNB","BernoulliNB","MultinomialNB"]

shape = PRAFarray.shape
for i in xrange(shape[0]):
    for j in xrange(shape[1]):
        sheet1.write(i,j,PRAFarray[i,j])

book.save('../result/compareBayes.xls')
