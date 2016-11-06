#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from myClassifier import *

xSample = np.float32(np.load('./npyfile_saver/xSample25.npy'))   #(1253,25)
ySample = np.float32(np.load('./npyfile_saver/ySample25.npy'))   #(1253,1)



x_train = xSample[400:1000,:]
y_train = ySample[400:1000,:]

x_test = xSample[1000:1250,:]
y_test = ySample[1000:1250,:]


filename = 'knnModel.txt'
knn = cv2.KNearest()
knn.train(x_train,y_train)
# knn.load(filename)
retval, results, neigh_resp, dists = knn.find_nearest(x_test, 1)

print "knn acrracy = ",np.mean(y_test == results)

knn.save(filename)
print "save successfully ..."