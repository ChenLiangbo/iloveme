#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os
from myNeurolNetworkModel import MyNeurolNetworkModel

myNNmodel = MyNeurolNetworkModel()

v = myNNmodel.random_vector(10,100)
print("v = ",v.shape)
yahooData = np.load('yahoo_finance5.npy')

print("-"*100)
sample =  yahooData[v]
print("sample.shape = ",sample.shape)