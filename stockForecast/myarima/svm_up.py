#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import cv2

xSample = np.float32(np.load('./npyfile/xSample.npy'))
ySample = np.float32(np.load('./npyfile/ySample.npy'))

shape = xSample.shape
length = shape[1]

y = np.zeros((ySample.shape))

for i in xrange(shape[0]-1):
    if ySample[i+1,0] < ySample[i,0]:
        y[i,0] = 0
    else:
        y[i,0] = 1

ySample = np.float32(y)

x_train = xSample[300:900,:]
y_train = ySample[300:900,:]

x_test = xSample[900:1258,:]
y_test = ySample[900:1258,:]


knn = cv2.KNearest()
knn.train(x_train,y_train)
ret, results, neighbours ,dist = knn.find_nearest(x_test,1)

# results = results.ravel()
acurracy = (y_test == results)

print "knn acurracy = ",np.mean(acurracy)

knn_acurracy = np.mean(acurracy)
knn_result = results

ret, results, neighbours ,dist = knn.find_nearest(x_train,1)
knn_trainX = results





svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	               svm_type = cv2.SVM_C_SVC,
                   C=1, gamma=0.5)

svm = cv2.SVM()

svm.train(x_train,y_train,params = svm_params)
# svm.save('svm_mnist_model.dat')


ret = svm.predict_all(x_test)
results =ret

acurracy = (y_test == results)

print "svm acurracy = ",np.mean(acurracy)
svm_result = results
svm_acurracy = np.mean(acurracy)
svm_trainX = svm.predict_all(x_train)



# random trees
RTtree = cv2.RTrees()

rtree_params = dict(depth = 32)
var_type = np.array([cv2.CV_VAR_NUMERICAL]*length + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)


RTtree.train(x_train,cv2.CV_ROW_SAMPLE,y_train,
    	varType = var_type,
    	params = rtree_params)

results = np.zeros(y_test.shape,dtype = y_test.dtype)
for i in xrange(y_test.shape[0]):
    results[i,0] = RTtree.predict(x_test[i,:])


acurracy = (y_test == results)

print "RTtree acurracy = ",np.mean(acurracy)

RTtree_result = results
RTtree_acurracy = np.mean(acurracy)

results = np.zeros(x_train.shape,dtype = x_train.dtype)
for i in xrange(x_train.shape[0]):
    results[i,0] = RTtree.predict(x_train[i,:])

RTtree_trainX = results






boost_params = dict(max_depth = 1)
var_type = np.array([cv2.CV_VAR_NUMERICAL]*length + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)
Boost = cv2.Boost()
Boost.train(x_train,cv2.CV_ROW_SAMPLE,y_train,varType = var_type,params = boost_params)

results = np.zeros(y_test.shape,dtype = y_test.dtype)

for i in xrange(y_test.shape[0]):
    results[i,0] = Boost.predict(x_test[i,:],returnSum = True)

for i in xrange(y_test.shape[0]):
    if results[i,0] >0:
        results[i,0] = 1
    else:
        results[i,0] = 0


acurracy = (y_test == results)

print "Boost acurracy = ",np.mean(acurracy)

boost_result = results
boost_acurracy = np.mean(acurracy)

for i in xrange(x_train.shape[0]):
    results[i,0] = Boost.predict(x_train[i,:],returnSum = True)

for i in xrange(x_train.shape[0]):
    if results[i,0] >0:
        results[i,0] = 1
    else:
        results[i,0] = 0
boost_trainX = results


weights = np.array([knn_acurracy,svm_acurracy,RTtree_acurracy,boost_acurracy])
gradArray = svm_result + knn_result + RTtree_result + boost_result
# print "gradArray = ",gradArray
for i in xrange(gradArray.shape[0]):
    # print "i = ",(i,gradArray[i,0],y_test[i,0])
    if gradArray[i,0] > 2:   
        gradArray[i,0] = 1
    elif gradArray[i,0] < 2:  
        gradArray[i,0] = 0     
    else:
        gradArray[i,0] = 1 - knn_result[i,0]

acurracy = (y_test == gradArray)
print "gradArray acurracy = ",np.mean(acurracy)

svm_result + knn_result + RTtree_result + boost_result

py_train = y_train
px_train = np.zeros((x_train.shape[0],8),dtype = np.float32)

for i in xrange(x_train.shape[0]):
    knn_temp = {}
    knn_temp[knn_trainX[i,0]] = knn_acurracy
    knn_temp[1-knn_trainX[i,0]] = 1 - knn_acurracy
    px_train[i,0] = knn_temp[0]
    px_train[i,1] = knn_temp[1]

    svm_temp = {}
    svm_temp[svm_trainX[i,0]] = svm_acurracy
    svm_temp[1 - svm_trainX[i,0]] = 1 - svm_acurracy
    px_train[i,2] = svm_temp[0]
    px_train[i,3] = svm_temp[1]

    rt_temp = {}
    rt_temp[RTtree_trainX[i,0]] = RTtree_acurracy
    rt_temp[1 - RTtree_trainX[i,0]] = 1 - RTtree_acurracy
    px_train[i,4] = rt_temp[0]
    px_train[i,5] = rt_temp[1]

    boost_temp = {}
    boost_temp[boost_trainX[i,0]] = boost_acurracy
    boost_temp[1 - boost_trainX[i,0]] = 1 - boost_acurracy
    px_train[i,6] = boost_temp[0]
    px_train[i,7] = boost_temp[1]

print "px_train.shape = ",px_train.shape

px_test = np.zeros((x_test.shape[0],8),dtype = np.float32)
py_test = y_test

for i in xrange(x_test.shape[0]):
    knn_temp = {}
    knn_temp[knn_result[i,0]] = knn_acurracy
    knn_temp[1-knn_result[i,0]] = 1 - knn_acurracy
    px_test[i,0] = knn_temp[0]
    px_test[i,1] = knn_temp[1]

    svm_temp = {}
    svm_temp[svm_result[i,0]] = svm_acurracy
    svm_temp[1 - svm_result[i,0]] = 1 - svm_acurracy
    px_test[i,2] = svm_temp[0]
    px_test[i,3] = svm_temp[1]

    rt_temp = {}
    rt_temp[RTtree_result[i,0]] = RTtree_acurracy
    rt_temp[1 - RTtree_result[i,0]] = 1 - RTtree_acurracy
    px_test[i,4] = rt_temp[0]
    px_test[i,5] = rt_temp[1]

    boost_temp = {}
    boost_temp[boost_result[i,0]] = boost_acurracy
    boost_temp[1 - boost_result[i,0]] = 1 - boost_acurracy
    px_test[i,6] = boost_temp[0]
    px_test[i,7] = boost_temp[1]



px_train = np.float32(px_train)
px_test = np.float32(px_test)
print "px_test.shape = ",(px_test.shape,px_test.dtype)

knn = cv2.KNearest()
knn.train(px_train,py_train)
ret, results, neighbours ,dist = knn.find_nearest(px_test,1)

# results = results.ravel()
acurracy = (py_test == results)

print "knn px_test acurracy = ",np.mean(acurracy)


# svm_params = dict( kernel_type = cv2.SVM_LINEAR,
#                    svm_type = cv2.SVM_C_SVC,
#                    C=1, gamma=0.5)

# svm = cv2.SVM()

# svm.train(px_train,py_train,params = svm_params)
# # svm.save('svm_mnist_model.dat')


# ret = svm.predict_all(px_test)
# results =ret

# acurracy = (py_test == results)

# print "svm px_train acurracy = ",np.mean(acurracy)

boost_params = dict(max_depth = 1)
var_type = np.array([cv2.CV_VAR_NUMERICAL]*8 + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)
Boost = cv2.Boost()
Boost.train(px_train,cv2.CV_ROW_SAMPLE,py_train,varType = var_type,params = boost_params)

results = np.zeros(py_test.shape,dtype = py_test.dtype)

for i in xrange(py_test.shape[0]):
    results[i,0] = Boost.predict(px_test[i,:],returnSum = True)

for i in xrange(py_test.shape[0]):
    if results[i,0] >0:
        results[i,0] = 1
    else:
        results[i,0] = 0


acurracy = (py_test == results)

print "Boost px_train acurracy = ",np.mean(acurracy)
