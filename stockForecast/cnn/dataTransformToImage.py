#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np

order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume*e-6"}
order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume"}
yahooData = np.load('yahoo_finance5.npy')
shape = yahooData.shape
print "yahooData shape = ",shape
Open,High,Low,Close,Volume = np.hsplit(yahooData,5)


# yahooFour = np.hstack([Open,High,Low,Close])
# yahooFive = Volume

# minFour = yahooFour.min()
# maxFour = yahooFour.max()
# print "minFour = %f,maxFour = %f" % (minFour,maxFour)
# yahooFour = 255*(yahooFour - minFour) / (maxFour - minFour)


# minFive = yahooFive.min()
# maxFive = yahooFive.max()
# print "minFive = %f,maxFive = %f" % (minFive,maxFive)
# yahooFive = 255*(yahooFive - minFive)/(maxFive - minFive)

# yahooData = np.hstack([yahooFour,yahooFive])

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

xSample = np.zeros((shape[0] - relatedDays,125),dtype = np.int32)
print "xSample.shape = ",xSample.shape


def perms(elements):
    if len(elements) <=1:
        yield elements
    else:
        for perm in perms(elements[1:]):
            for i in range(len(elements)):
                yield perm[:i] + elements[0:1] + perm[i:]

orderList = list(perms([0,1, 2, 3,4]))
totalOrders = len(orderList)
period = 6

for row in xrange(0,shape[0]-relatedDays):    # row index
    for day in xrange(0,5):                   # days,five days
        for y in range(5):                    # colunme index
            # print "yahooData row = ",yahooData[row,:]
            order = orderList[period*y]
            dayIndex = 25*y
            start = dayIndex + 5*y
            end   = dayIndex + 5*(y+1)
            xSample[row,start:end] = yahooData[row + y,order]
            
np.save('./npyfile_saver/xSample',xSample)

ySample = np.zeros((xSample.shape[0],1),dtype = np.int32)
for row in xrange(ySample.shape[0]):
    start = relatedDays -1 + row     # 4
    end = relatedDays + row          # 5
    if Close[end,:] > Close[start,:] or Close[start] == Close[end,:]:
        ySample[row,0] = 1
    else:
        ySample[row,0] = 0

np.save('./npyfile_saver/ySample',ySample)