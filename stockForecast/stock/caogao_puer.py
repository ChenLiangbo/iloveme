#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os

outdir = os.path.join(os.path.dirname(__file__),'withKDJdclose/')

order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume*e-6"}
yahooData = np.load('yahoo_finance5.npy')
shape = yahooData.shape
print "shape = ",shape
print yahooData[0:10,3]
print "--------------------------------------"
x_sample = yahooData[0:shape[0]-1,:]
y_sample = np.hsplit(yahooData,5)[3]


def reverse_array(array):
    shape = array.shape
    outArray = np.zeros(shape)
    if len(shape) < 2:
        for i in xrange(shape[0]):
            outArray[i] = array[shape[0]-1-i]
        return outArray
    
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            outArray[i,j] = array[shape[0]-1-i,j]
    return outArray

# y10 = y_sample[0:10]
# yreverse10 = reverse_array(y10)
# print y10
# print "-"*80
# print yreverse10

x = range(1,31)
x = np.array(x).reshape(30,1)

y = np.zeros(x.shape)
exponent = 0.90

for i in xrange(x.shape[0]):
    y[i,0] = np.power(exponent,x[i,0])
window = y/np.sum(y)
print "window sum = ",np.sum(window)
windowLength = 30
y_exponent = np.zeros(y_sample.shape)
for i in xrange(y_sample.shape[0]):
    if i == 0:
        y_exponent[i] = y_sample[i]
    if i < windowLength:
        pass
        
        x = range(i+1)
        print "x = ",x

        x_a = np.array(x).reshape[i,1]
        
        weights = np.zeros(x_a.shape)
        for i in xrange(x_a.shape[0]):
            weights[i,0] = np.power(exponent,x_a[i,0])
        weights = weights/np.sum(weights)

        related = y_sample[:i,0]
        related = reverse_array(related)
        y_exponent[i] = np.sum(related*weights)
        
    else:
        weights = window
        related = y_sample[i-30:i,0]
        related = reverse_array(related)
        y_exponent[i,0] = np.sum(related*weights)

from matplotlib import pyplot as plt

# plt.plot(y_sample,'ro')
# plt.plot(y_sample[200:300,0],'r-')
plt.plot(y_exponent[200:300,0],'b-')
# plt.plot(y_sample[200:300,0],'ro')
plt.plot(y_exponent[200:300,0],'bo')
plt.legend(['y_sample','y_exponent'])

plt.title('y_sample With Puer Exponent And Real Close')
plt.xlabel('index')
plt.ylabel(['Exponent'])
plt.savefig(outdir + 'close_exponent')
plt.grid(True)
plt.show()

logFitrate = np.zeros(y_sample.shape)
for i in xrange(1,shape[0]):
    logFitrate[i,0] = np.log(y_sample[i,0]/y_sample[i-1,0])
logFitrate[0,0] = logFitrate[1,0]

mean = np.mean(logFitrate)
varValue = np.var(logFitrate)
logFitrate = (logFitrate - mean)/varValue

plt.plot(logFitrate,'r-')
plt.legend(['logFitrate',])
plt.title('logFitrate With Puer Exponent And Real Close')
plt.xlabel('index')
plt.ylabel(['logFitrate'])
plt.savefig(outdir + 'close_logFitrate')
plt.grid(True)
plt.show()
