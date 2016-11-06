#!usr/bin/env/python 
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np


yahooData = np.load('yahooData.npy')

print "yahooData.shape = ",yahooData.shape

splitList = np.hsplit(yahooData,5)

# number = 100
# y1 = splitList[0][0:number,:]
# y2 = splitList[1][0:number,:]
# print "y1.shape = ",y1.shape
# print "y2.shape = ",y2.shape

# x = np.linspace(0,y1.shape[0],y1.shape[0]).reshape(y1.shape[0],1)

# x = [1,2,3,4,5,6,7,8,9,10]
# y1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# y2 = [0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6]

x = np.linspace(0,10,100)
y1 = np.sin(x)
y2 = np.log(x)

# x = np.linspace(-3,3,100)
# y1 = np.sin(x)
# y2 = np.tanh(x)

# x = np.linspace(-10,10,100)
# y1 = np.tanh(x)
# y2 = 1/(1+np.exp(-x))

plt.plot(x,y1,'ro')
plt.plot(x,y2,'bo')
plt.plot(x,y1,'r-')
plt.plot(x,y2,'b-')

plt.grid(True)

plt.legend(['y1','y2'])
plt.xlabel('Time-Seris')
plt.ylabel('Value')
plt.title('The Graph of y1 And y2')

plt.show()
plt.savefig('Y1-Y2.jpg')


