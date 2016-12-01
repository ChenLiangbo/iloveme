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
shape = xsample.shape
class0 = 0
class1 = 0
for i in xrange(shape[0]):
    if int(ysample[i]) == 1:
        class1 = class1 + 1
    else:
        class0 = class0 + 1
print "class0 = %d,class1 = %d" %(class0,class1)

x_train = xsample[0:538,:]
x = x_train[:,0]
outdir = '../images/'
from matplotlib import pyplot as plt
plt.plot(x,'ro')
plt.legend(['feature1',])
plt.grid(True)
plt.title('Distribution of Feature 1')
plt.xlabel('Value of Distribution')
plt.ylabel('Freqreuncy of Distribution')
plt.savefig(outdir + 'Feature1.jpg')
plt.show()
