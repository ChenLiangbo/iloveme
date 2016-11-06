#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

xSample = np.load('./npyfile_saver/xSample625.npy')
ySample = np.load('./npyfile_saver/ySample625.npy')

xSample = xSample.astype(np.float32)
ySample = ySample.astype(np.float32)

print "xSample.shape = ",xSample.shape
print "ySample.shape = ",ySample.shape

train_start = 0
train_end = 1000

x_train = xSample[train_start:train_end,:]
y_train = ySample[train_start:train_end,:]

test_start = 900
test_end = 1250

x_test = xSample[test_start:test_end,:]
y_test = ySample[test_start:test_end,:]