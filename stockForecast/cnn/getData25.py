#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np

order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume*e-6"}
order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume"}
yahooData = np.load('yahoo_finance5.npy')
shape = yahooData.shape
print "yahooData shape = ",shape
Open,High,Low,Close,Volume = np.hsplit(yahooData,5)

minValue = yahooData.min()
maxValue = yahooData.max()
print "minValue = %f,maxValue = %f" % (minValue,maxValue)
yahooData = 255*(yahooData - minValue)/(maxValue - minValue)
for i in xrange(shape[0]):
    for j in xrange(shape[1]):
        yahooData[i,j] = int(yahooData[i,j])
print "yahooData = "
print yahooData[0:20,:]
print "-----------------------------------"

Open,High,Low,Close,Volume = np.hsplit(yahooData,5)

relatedDays = 5

xSample = np.zeros((shape[0] - relatedDays,25))
print "xSample.shape = ",xSample.shape

for i in xrange(xSample.shape[0]):
    array5x5 = yahooData[i:i+relatedDays,:]
    xSample[i,:] = array5x5.reshape(1,25).astype(np.float32)
# np.save('./npyfile_saver/xSample25',xSample)

ySample = np.zeros((xSample.shape[0],1),dtype = np.int32)
for row in xrange(ySample.shape[0]):
    start = relatedDays -1 + row     # 4
    end = relatedDays + row          # 5
    if Close[end,:] > Close[start,:] or Close[start] == Close[end,:]:
        ySample[row,0] = 1
    else:
        ySample[row,0] = 0

# np.save('./npyfile_saver/ySample25',ySample)


