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

indexList = np.random.permutation(shape[0])
# indexList = range(shape[0])

x_train = xsample[indexList[0:538]]
y_train = ysample[indexList[0:538]]
print "x_train.shape = ",x_train.shape
print "y_train.shape = ",y_train.shape

x_test = xsample[indexList[538:]]
y_test = ysample[indexList[538:]]
print "x_test.shape = ",x_test.shape
print "y_test.shape = ",y_test.shape


classifier = BayesClassifier()
classifier.saveNeeded = True
classifier.sectionNumber = 12
classifier.train(x_train,y_train)
print "classifier train succefully ..."
y_predict = classifier.predict(x_test)

# print "y_predict = ",y_predict
print "y_predict = ",y_predict.shape
accuracy = (y_test == y_predict)

print "BayesClassifier accuracy = ",np.mean(accuracy)
params = classifier.params
'''
p = params['p0']
p0 = p['class0']
frequency = p0['frequency']
probability = p0["probability"]

from matplotlib import pyplot as plt
plt.plot(frequency,probability,'ro')
plt.plot(frequency,probability,'r-')
plt.title('Distribution of Attribute One Class0')
plt.xlabel('frequency')
plt.ylabel('probability')
plt.grid(True)
plt.savefig('../images/f-p-one0')
plt.show()

p = params['p0']
p1= p['class1']
frequency = p1['frequency']
probability = p1["probability"]

from matplotlib import pyplot as plt
plt.plot(frequency,probability,'ro')
plt.plot(frequency,probability,'r-')
plt.title('Distribution of Attribute One Class1')
plt.xlabel('frequency')
plt.ylabel('probability')
plt.grid(True)
plt.savefig('../images/f-p-one1')
plt.show()
'''

p = params['p1']
p0 = p['class0']
print "p0 = ",p0.keys()
xTresh = p0['xTresh']
probability = p0["probability"]

print "xTresh = ",xTresh
print "probability = ",probability

domain = []
length = len(xTresh)
for i in xrange(length -1):
    e = str(round(xTresh[i],2)) + '-' +str(round(xTresh[i+1],2))
    domain.append(e)
print "domain = ",len(domain)
print "probability = ",len(probability)


from matplotlib import pyplot as plt

plt.plot(probability,'ro')
plt.plot(probability,'r-')
plt.title('Distribution of Attribute Two Class1')
plt.xlabel('frequency')
plt.ylabel('probability')
ax=plt.gca()
ax.set_xticklabels(domain)
plt.grid(True)
plt.savefig('../images/f-p-Two0')
plt.show()



p = params['p1']
p0 = p['class1']
print "p0 = ",p0.keys()
xTresh = p0['xTresh']
probability = p0["probability"]

print "xTresh = ",xTresh
print "probability = ",probability

domain = []
length = len(xTresh)
for i in xrange(length -1):
    e = str(round(xTresh[i],2)) + '-' +str(round(xTresh[i+1],2))
    domain.append(e)
print "domain = ",len(domain)
print "probability = ",len(probability)


from matplotlib import pyplot as plt

plt.plot(probability,'ro')
plt.plot(probability,'r-')
plt.title('Distribution of Attribute Two Class1')
plt.xlabel('frequency')
plt.ylabel('probability')
ax=plt.gca()
ax.set_xticklabels(domain)
plt.grid(True)
plt.savefig('../images/f-p-Two1')
plt.show()