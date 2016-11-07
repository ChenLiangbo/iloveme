#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np

dataset = np.load('./dataset/pima-indians.npy')

columns = np.hsplit(dataset,9)
xsample = np.hstack(columns[0:8])
ysample = columns[8]

shape = xsample.shape
y = np.zeros((shape[0],2))
for i in xrange(shape[0]):
    if ysample[i] == 0:
        y[i,0] = 1
    else:
        y[i,1] = 1
#y [1,0] 0
#y [0,1] 1
np.save('./dataset/xsample',xsample)
np.save('./dataset/ysample',y)