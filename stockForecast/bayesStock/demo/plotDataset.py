#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os
# from myNeurolNetworkModel import MyNeurolNetworkModel

order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume*e-6"}
yahooData = np.load('../dataset/yahoo_finance5.npy')
Open,High,Low,Close,Volume = np.hsplit(yahooData,5)

shape = yahooData.shape
print "shape = ",shape

x = yahooData[:,0]
from matplotlib import pyplot as plt
plt.plot(x,'r-')
plt.grid(True)
plt.show()