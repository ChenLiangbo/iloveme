#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np

filename = '../dataset/pima-indians.csv'

fp = open(filename,'rb')
lines = fp.readlines()
print "lines = ",len(lines)
data = []
for l in lines:
    l = l.split(',')
    # print "l = ",l
    for i in xrange(len(l)):
        if '\n' in l[i]:
            l[i] = l[i].strip('\r\n')
        l[i] = float(l[i])
    data.append(l)
    # print "data = ",data
    # break

dataset = np.asarray(data).astype(np.float32)
print "dataset = ",dataset.shape
np.save("pima-indians",dataset)
fp.close()

dataset = np.load('pima-indians.npy')

columns = np.hsplit(dataset,9)
xsample = np.hstack(columns[0:8])
ysample = columns[8]
print "xsample = ",xsample.shape
print "ysample = ",ysample.shape
