#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import cv2

xSample = np.load('trX.npy').astype(np.float32)
ySample = np.load('trY.npy').astype(np.float32)

print "xSample.shape = ",(xSample.shape,xSample.dtype)
print "ySample.shape = ",(ySample.shape,ySample.dtype)

x = np.zeros((xSample.shape[0],xSample.shape[1]*xSample.shape[2]))

for row in xrange(xSample.shape[0]):
    x[row,:] = xSample[row,:,:].reshape(1,81)

x = x.astype(np.float32)
ySample = ySample.astype(np.float32)

x_train = x[100:300,:]
y_train = ySample[100:300,:]
x_test = np.vstack([x[0:100,:],x[300:400,:]])
y_test = np.vstack([ySample[0:100,:],ySample[300:400,:]])
print "x.shape = ",x.shape

# xSample = np.load('./npyfile_saver/xSample25.npy').astype(np.float32)
# ySample = np.load('./npyfile_saver/ySample25.npy').astype(np.float32)
# print "xSample.shape = ",(xSample.shape,xSample.dtype)
# print "ySample.shape = ",(ySample.shape,ySample.dtype)
# # print "ySample = ",ySample[0:10]

# train_number = 1000
# x_train = xSample[0:train_number,:]
# y_train = ySample[0:train_number,:]
# # print "x_train.shape = ",x_train.shape
# # print "y_train.shape = ",y_train.shape

# x_test = xSample[train_number:,:]
# y_test = ySample[train_number:,:]

svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	               svm_type = cv2.SVM_C_SVC,
                   C=3.67, gamma = cv2.SVM_GAMMA)

svm = cv2.SVM()
svm.train(x_train,y_train,params = svm_params)
# svm.save('svm_model.dat')
ret = svm.predict_all(x_test)
# print "ret = ",ret
acurracy = (y_test == ret)

print "svm acurracy = ",np.mean(acurracy)



'''
knn = cv2.KNearest()
knn.train(x_train,y_train)
ret, results, neighbours ,dist = knn.find_nearest(x_test, 3)
print "results = ",results[0:10]
print "ret = ",ret
acurracy = (y_test == ret)

print "knn acurracy = ",np.mean(acurracy)
'''
'''
ann = cv2.ANN_MLP()

layer_sizes = np.int32([81,1000,30,1])  
ann.create(layer_sizes)  

params = dict( term_crit = (cv2.TERM_CRITERIA_COUNT, 10000, 0.001),  
                   train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,  
                   bp_dw_scale = 0.001,  
                   bp_moment_scale = 0.0 ) 


ann.train(x_train,y_train,None,params = params)
ret, results = ann.predict(x_test)

# print "ret = ",ret[0:10]
#print "result = ",results[0:10]

ret = results
#print "ret.shape = ",ret.shape
for i in range(ret.shape[0]):
    if ret[i,0] > 0.5:
        ret[i,0] = 1
    else:
        ret[i,0] = 0

acurracy = (y_train == ret)
# print "results = ",results[0:10,:]

print "acurracy = %f " % (np.mean(acurracy),)
'''


'''
length = 20

for i in range(x_train.shape[0]/20):
    x_batch = x_train[i*20:20*(i+1),:]
    y_batch = y_train[i*20:20*(i+1),:]
    ann.train(x_batch,y_batch,None,params = params)
    ret, results = ann.predict(x_train)
    
    # print "ret = ",ret[0:10]
    print "result = ",results[0:10]

    ret = results
    acurracy = (y_train == ret)
    # print "results = ",results[0:10,:]

    print "step = %d,acurracy = %f " % (i,np.mean(acurracy))
'''



'''
RTtree = cv2.RTrees()

var_types = np.array([cv2.CV_VAR_NUMERICAL] * x.shape[1] + [cv2.CV_VAR_CATEGORICAL], np.uint8)
params = dict(max_depth=20)
RTtree.train(x_train, cv2.CV_ROW_SAMPLE, y_train, varType = var_types, params = params)

results = np.zeros(y_test.shape,dtype = y_test.dtype)
for i in xrange(y_test.shape[0]):
    results[i,0] = RTtree.predict(x_test[i,:])

acurracy = (y_test == results)

print "acurracy = ",np.mean(acurracy)
'''



'''
boost_params = dict(max_depth = 30)
var_type = np.array([cv2.CV_VAR_NUMERICAL]*xSample.shape[1] + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)
Boost = cv2.Boost()
Boost.train(x_train,cv2.CV_ROW_SAMPLE,y_train)

results = np.zeros(y_test.shape,dtype = y_test.dtype)

for i in xrange(y_test.shape[0]):
    results[i,0] = Boost.predict(x_test[i,:],returnSum = False)

print "results = ",results[0:10]
print "y_test = ",y_test[0:10]
acurracy = (y_test == results)

print "Boost acurracy = ",np.mean(acurracy)
'''


#k means

# kmeansSample = xSample[0:200,:]
# y = ySample[0:200]

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
# flags = cv2.KMEANS_RANDOM_CENTERS

# compatches,labels,centers = cv2.kmeans(kmeansSample,2,criteria,10,flags)
# print "labels = ",labels[0:10]

# acurracy = (y == labels)
# print "kmeans acurracy = ",np.mean(acurracy)
# from matplotlib import pyplot as plt
# x_asix = np.linspace(0,kmeansSample.shape[0],kmeansSample.shape[0]).reshape(kmeansSample.shape[0],1)

# print "x_asix.shape = ",x_asix.shape
# print "kmeansSample.shape = ",kmeansSample.shape
# # plt.plot(x,kmeansSample[:,0,3].reshape(kmeansSample.shape[0],1),'r-')

