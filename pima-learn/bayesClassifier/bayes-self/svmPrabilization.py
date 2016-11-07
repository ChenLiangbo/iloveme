#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from bayesClassifier import BayesClassifier


dataset = np.load('pima-indians.npy')


columns = np.hsplit(dataset,9)
xsample = np.hstack(columns[0:8])
ysample = columns[8]
xsample = np.float32(xsample)
ysample = np.float32(ysample)
shape = xsample.shape

print "xsample = ",xsample.shape
print "ysample = ",ysample.shape


orders = ["precision",'recall','accuracy','fmeasure']
number = 20
PRAFarray = np.zeros((number,12))
for i in xrange(number):
    print "i = ",i
    indexList = np.random.permutation(shape[0])
    x_train = xsample[indexList[:538]]
    y_train = ysample[indexList[:538]]
    classifier = BayesClassifier()
    classifier.saveNeeded = False
    classifier.sectionNumber = 30
    classifier.train(x_train,y_train)
    x_train1 = classifier.transform_pmodel(x_train)
    x_train2 = np.hstack([x_train,x_train1])
    print "shape = ",(x_train.shape,x_train1.shape,x_train2.shape)

    x_test = xsample[indexList[538:]]
    y_test = ysample[indexList[538:]]
    x_test1 = classifier.transform_pmodel(x_test)
    x_test2 = np.hstack([x_test,x_test1])


    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                   svm_type = cv2.SVM_C_SVC,
                   C=3.67, gamma = cv2.SVM_GAMMA)
    svm = cv2.SVM()
    svm.train(x_train,y_train,params = svm_params)
    ret = svm.predict_all(x_test)
    result = classifier.f_measure(ret,y_test)
    for key in orders:
        index = orders.index(key)
        PRAFarray[i,index] = result[key]


    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                   svm_type = cv2.SVM_C_SVC,
                   C=3.67, gamma = cv2.SVM_GAMMA)
    svm = cv2.SVM()
    svm.train(x_train1,y_train,params = svm_params)
    ret = svm.predict_all(x_test1)
    result = classifier.f_measure(ret,y_test)
    for key in orders:
        index = orders.index(key) + 4
        PRAFarray[i,index] = result[key]


    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                   svm_type = cv2.SVM_C_SVC,
                   C=3.67, gamma = cv2.SVM_GAMMA)
    svm = cv2.SVM()
    svm.train(x_train2,y_train,params = svm_params)
    ret = svm.predict_all(x_test2)
    result = classifier.f_measure(ret,y_test)
    for key in orders:
        index = orders.index(key) + 8
        PRAFarray[i,index] = result[key]



import xlwt
book = xlwt.Workbook(encoding = 'utf-8')
sheet1 = book.add_sheet('Sheet1',cell_overwrite_ok = False)

shape = PRAFarray.shape
for i in xrange(shape[0]):
    for j in xrange(shape[1]):
        sheet1.write(i,j,PRAFarray[i,j])

book.save('../result/p-svm.xls')


shapes1 = ['ro','go','bo','mo']
shapes2 = ['r-','g-','b-','m-']
from matplotlib import pyplot as plt

for j in xrange(4):
    classfierOrder = ["svm","svm1","svm2"]
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
    plt.savefig('../images/p-svm-' + name)
    plt.show()