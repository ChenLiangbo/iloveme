#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from bayesClassifier import BayesClassifier


dataset = np.load('pima-indians.npy')


columns = np.hsplit(dataset,9)
xsample = np.hstack(columns[0:8])
ysample = columns[8]
shape = xsample.shape

print "xsample = ",xsample.shape
print "ysample = ",ysample.shape

from sklearn.naive_bayes import GaussianNB
orders = ["precision",'recall','accuracy','fmeasure']
number = 20
PRAFarray = np.zeros((number,12))
for i in xrange(number):
    indexList = np.random.permutation(shape[0])
    x_train = xsample[indexList[:538]]
    y_train = ysample[indexList[:538]]
    classifier = BayesClassifier()
    classifier.saveNeeded = False
    classifier.sectionNumber = 40
    classifier.train(x_train,y_train)
    x_train1 = classifier.transform_pmodel(x_train)
    x_train2 = np.hstack([x_train,x_train1])
    print "shape = ",(x_train.shape,x_train1.shape,x_train2.shape)

    x_test = xsample[indexList[538:]]
    y_test = ysample[indexList[538:]]
    x_test1 = classifier.transform_pmodel(x_test)
    x_test2 = np.hstack([x_test,x_test1])


    clf = GaussianNB().fit(x_train, y_train.ravel())  
    y_predict =  clf.predict(x_test)
    result = classifier.f_measure(y_predict,y_test)
    print (np.mean(y_test == y_predict),result["accuracy"])
    for key in orders:
        index = orders.index(key)
        PRAFarray[i,index] = result[key]

    clf = GaussianNB().fit(x_train1, y_train.ravel())  
    y_predict1 =  clf.predict(x_test1)
    result1 = classifier.f_measure(y_predict1,y_test)
    for key in orders:
        index = orders.index(key) + 4
        PRAFarray[i,index] = result1[key]


    clf = GaussianNB().fit(x_train2, y_train.ravel())  
    y_predict2 =  clf.predict(x_test2)
    result2 = classifier.f_measure(y_predict2,y_test)
    for key in orders:
        index = orders.index(key) + 8
        PRAFarray[i,index] = result2[key]


import xlwt
book = xlwt.Workbook(encoding = 'utf-8')
sheet1 = book.add_sheet('Sheet1',cell_overwrite_ok = False)
columns =["myBayes","GaussianNB","BernoulliNB","MultinomialNB"]

shape = PRAFarray.shape
for i in xrange(shape[0]):
    for j in xrange(shape[1]):
        sheet1.write(i,j,PRAFarray[i,j])

book.save('../result/p-GaussianNB.xls')


shapes1 = ['ro','go','bo','mo']
shapes2 = ['r-','g-','b-','m-']
from matplotlib import pyplot as plt

for j in xrange(4):
    classfierOrder = ["GaussianNB","GaussianNB1","GaussianNB2"]
    name = orders[j]
    for i in xrange(3):
        plt.plot(PRAFarray[:,j+i*4],shapes1[i])
    plt.legend(classfierOrder)
    for i in xrange(3):
        plt.plot(PRAFarray[:,j+i*4],shapes2[i])
    plt.xlabel('Experiment Number')
    plt.ylabel(name)
    plt.title(name + " of Prabilization")
    plt.grid(True)
    plt.savefig('../images/p-GaussianNb-' + name)
    plt.show()