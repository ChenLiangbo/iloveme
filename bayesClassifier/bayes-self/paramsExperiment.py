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

start = 5
end = 50
precision = []
accuracy = []
recall = []
fmeasure = []
for i in range(start,end):
    classifier = BayesClassifier()
    classifier.saveNeeded = False
    classifier.sectionNumber = i
    classifier.train(x_train,y_train)
    y_predict = classifier.predict(x_test)
    result = classifier.f_measure(y_predict,y_test)
    precision.append(result["precision"])
    accuracy.append(result["accuracy"])
    recall.append(result["recall"])
    fmeasure.append(result["fmeasure"])

data = np.array([accuracy,precision,recall,fmeasure])
print "data.shape = ",data.shape


names = ["accuracy","precision","recall","fmeasure"]
colors = ['r-','b-','g-','y-','m-','c-','k-']
shapes = ['ro','bo','go','yo','mo','co','ko']

shape = data.shape
from matplotlib import pyplot as plt
for i in range(shape[0]):
    plt.plot(data[i,:],colors[i])

plt.legend(names)
for i in range(shape[0]):
    plt.plot(data[i,:],shapes[i])


plt.grid(True)
plt.xlabel('section number')
plt.ylabel('value')
plt.title('Parameter Experiment Result')
plt.savefig('../images/paramExperiment')
plt.show()

classifier = BayesClassifier()
classifier.saveNeeded = False
classifier.sectionNumber = 39
classifier.train(x_train,y_train)
y_predict = classifier.predict(x_test)
result = classifier.f_measure(y_predict,y_test)
print "section number = 6,result = ",result