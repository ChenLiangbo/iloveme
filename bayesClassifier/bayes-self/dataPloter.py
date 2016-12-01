#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
outdir = '../images/'
dataset = np.load('pima-indians.npy')

'''
for i in xrange(dataset.shape[0]):
    print dataset[i,:]
    print "-"*100
'''
train = dataset[0:538,:]


x = train[:,4]
xshape = x.shape
print "xshape = ",xshape
xmax = x.max()
xmin = x.min()
print 'xmin = %d,xmax = %d' % (xmin,xmax)
xaxis = range(0,18)
yaxis = np.zeros((18,))

length = float(xmax-xmin)/20
print "length = ",length

xTresh = np.arange(xmin,xmax+length+1,length)
print "xTresh = ",xTresh

domain = []
for i in xrange(xTresh.shape[0] - 1):
    domain.append([xTresh[i],xTresh[i+1]])
print "domain = ",len(domain)

def getIndex(e,item):
    for i in range(len(item) - 1):
        if (e >= item[i]) and (e < item[i+1]):
            return i


yaxis = np.zeros((len(domain),))

for xi in xrange(xshape[0]):
    index = getIndex(x[xi],xTresh)
    yaxis[index] = yaxis[index] + 1
yaxis = yaxis + 1
print "yaxis = ",yaxis/np.sum(yaxis)

from matplotlib import pyplot as plt
plt.plot(yaxis,'r-')
plt.plot(yaxis,'bo')
plt.legend(['probability',])
plt.grid(True)
plt.title('Distribution of Attribute Five')
plt.xlabel('Value of Distribution')
plt.ylabel('Freqreuncy of Distribution')
plt.savefig(outdir + 'Attribute4.jpg')
plt.show()



'''
x = train[:,1]
xshape = x.shape
xmax = x.max()
xmin = x.min()
print 'xmin = %d,xmax = %d' % (xmin,xmax)
xaxis = range(int(xmin),int(xmax+1))
print "xaxis = ",len(xaxis)
yaxis = np.zeros((int(xmax)+1,))
print "yaxis = ",yaxis.shape

for xi in xrange(xshape[0]):
    index = xaxis.index(int(x[xi]))
    yaxis[index] = yaxis[index] + 1
yaxis_1 = yaxis + 1
print "sum = ",np.sum(yaxis)
from matplotlib import pyplot as plt
plt.plot(xaxis,yaxis,'ro')
plt.plot(xaxis,yaxis_1,'bo')
plt.plot(xaxis,yaxis,'r-')
plt.plot(xaxis,yaxis_1,'b-')
plt.legend(['Freqreuncy','Freqreuncy+1'])
plt.grid(True)
plt.title('Distribution of Attribute Two')
plt.xlabel('Value of Distribution')
plt.ylabel('Freqreuncy of Distribution')
plt.savefig(outdir + 'Attribute2.jpg')
plt.show()
'''

'''
xindex = [0,1,2,3,4,7]

for i in xindex:
    x = train[:,i]
    xshape = x.shape
    xmax = x.max()
    xmin = x.min()
    print 'xmin = %d,xmax = %d' % (xmin,xmax)
    xaxis = range(int(xmin),int(xmax+1))
    print "xaxis = ",len(xaxis)
    yaxis = np.zeros((int(xmax)+1,))
    print "yaxis = ",yaxis.shape
    
    for xi in xrange(xshape[0]):
        index = xaxis.index(int(x[xi]))
        yaxis[index] = yaxis[index] + 1
    yaxis_1 = yaxis + 1
    print "sum = ",np.sum(yaxis)
    from matplotlib import pyplot as plt
    plt.plot(xaxis,yaxis,'ro')
    plt.plot(xaxis,yaxis_1,'bo')
    plt.plot(xaxis,yaxis,'r-')
    plt.plot(xaxis,yaxis_1,'b-')
    plt.legend(['Freqreuncy','Freqreuncy+1'])
    plt.grid(True)
    plt.title('Distribution of Attribute' + str(i+1))
    plt.xlabel('Value of Distribution')
    plt.ylabel('Freqreuncy of Distribution')
    plt.savefig(outdir + 'Attribute'+str(i) + '.jpg')
    plt.show()    
'''