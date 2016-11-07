#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os

npydir = './npyfile/'

dataset = np.load('./dataset/pima-indians.npy')

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

def f_measure(y_predict,y_test):
    shape = y_predict.shape
    TP,FP,FN,TN = 0,0,0,0
    for i in xrange(shape[0]):
        if int(y_predict[i]) == 1:
            if int(y_test[i]) == 1:
                TP = TP + 1
            else:
                FP = FP + 1
        else:
            if int(y_test[i]) == 1:
                FN = FN + 1
            else:
                TN = TN + 1
    # print "TP = %d,TN = %d,FP = %d,FN = %d " % (TP,TN,FP,FN)
    result = {}
    result['accuracy']  = round(float(TP + TN)/(TP + FP + FN + TN),4)
    result["precision"] = round(float(TP)/(TP + FP),4)
    result["recall"]    = round(float(TP)/(TP + FN),4)
    result["fmeasure"]  = 2*result["precision"]*result["recall"]/(result["precision"] + result["recall"])
    result["fmeasure"] = round(result["fmeasure"],4)
    return result

testShape = x_test.shape
y_vote = np.zeros((testShape[0],1))
filelist = os.listdir(npydir)

for f in filelist:
    index = filelist.index(f)
    y_predict = np.load(npydir + f)
    print "y_predict = ",y_predict.shape
    if index == 0:
        y_array = y_predict
    else:
        y_array = np.hstack([y_array,y_predict])

y_array = np.int32(y_array)
    
print "y_array = ",y_array.shape

choise = {'1':0.7826,"2":0.7739,"3":0.7826,"4":0.813,"5":0.7913}

print '-'*80
print y_array[0:10]
for i in xrange(testShape[0]):
    rowSum = 0
    for j in xrange(5):
        rowSum = rowSum + y_array[i,j]*choise[str(j+1)]
    if rowSum >= 3:
        y_vote[i] = 1
    else:
        y_vote[i] = 0


voteResult = f_measure(y_vote,y_test)
print "voteResult = ",voteResult
