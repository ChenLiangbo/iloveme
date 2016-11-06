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
    array9x9 = xSample[row,:,:]
    xmax = np.amax(array9x9,axis = 0)
    array9x9 = array9x9/xmax
    x[row,:] = array9x9.reshape(1,81)

# for row in xrange(xSample.shape[0]):
#     array9x9 = xSample[row,:,:]
#     x_max = np.max(array9x9)
#     x_min = np.min(array9x9)
#     array9x9 = 255*(array9x9 - x_min)/(x_max - x_min)
#     # array9x9 = cv2.GaussianBlur(array9x9,(5,5),0)
#     x[row,:] = array9x9.reshape(1,81)

x = x.astype(np.float32)
ySample = ySample.astype(np.float32)

x_train = x[100:300,:]
y_train = ySample[100:300,:]
x_test = np.vstack([x[0:100,:],x[300:400,:]])
y_test = np.vstack([ySample[0:100,:],ySample[300:400,:]])


knn = cv2.KNearest()
knn.train(x_train,y_train)
ret, results, neighbours ,dist = knn.find_nearest(x_test,1)

# results = results.ravel()
acurracy = (y_test == results)

print "knn acurracy = ",np.mean(acurracy)
knn_result = results


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



# random trees
RTtree = cv2.RTrees()

rtree_params = dict(depth = 32)
var_type = np.array([cv2.CV_VAR_NUMERICAL]*81 + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)


RTtree.train(x_train,cv2.CV_ROW_SAMPLE,y_train,
    	varType = var_type,
    	params = rtree_params)

results = np.zeros(y_test.shape,dtype = y_test.dtype)
for i in xrange(y_test.shape[0]):
    results[i,0] = RTtree.predict(x_test[i,:])


acurracy = (y_test == results)

print "RTtree acurracy = ",np.mean(acurracy)

RTtree_result = results


boost_params = dict(max_depth = 1)
var_type = np.array([cv2.CV_VAR_NUMERICAL]*81 + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)
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


score = np.zeros(results.shape,dtype = np.float32)
weights = np.array([0.56,0.595,0.575,0.625])
weights = weights/np.sum(weights)


score = knn_result*weights[0] + svm_result*weights[1] + RTtree_result*weights[2] + boost_result*weights[3]
score = np.round(score)
acurracy = (y_test == score)
print "mix acurracy = ",np.mean(acurracy)



gradArray = svm_result + knn_result + RTtree_result + boost_result
# print "gradArray = ",gradArray
for i in xrange(gradArray.shape[0]):
    # print "i = ",(i,gradArray[i,0],y_test[i,0])
    if gradArray[i,0] > 2:   
        gradArray[i,0] = 1
    elif gradArray[i,0] < 2:  
        gradArray[i,0] = 0     
    else:
        # gradArray[i,0] = knn_result[i,0]*weights[0] + svm_result[i,0]*weights[1] + RTtree_result[i,0]*weights[2] + boost_result[i,0]*weights[3]
        # gradArray[i,0] = np.round(gradArray[i,0])
        gradArray[i,0] = 1 - knn_result[i,0]


acurracy = (y_test == gradArray)
print "gradArray acurracy = ",np.mean(acurracy)


'''
threshlist = np.linspace(0,1,20).tolist()
mix_acurracy = []

scoreShape = score.shape

for t in threshlist:
    for i in range(scoreShape[0]):
        if score[i,0] > t:
            score[i,0] = 1
        else:
            score[i,0] = 0


    acurracy = (y_test == score)
    mix_acurracy.append(np.mean(acurracy))
    print "mix acurracy = ",np.mean(acurracy)

max_acurracy = max(mix_acurracy)
index = mix_acurracy.index(max_acurracy)
print "max_acurracy = %f,depth = %d" % (max_acurracy,threshlist[index])

from matplotlib import pyplot as plt

plt.plot(threshlist,mix_acurracy, 'r-')
plt.plot(threshlist,mix_acurracy, 'ro')
plt.xlabel('thresh')
plt.ylabel('mix acurracy')
plt.title('mix acurracy with thresh')
plt.legend(['thresh'])
plt.grid(True)
plt.savefig('./ploter/min_thresh.jpg')
plt.show()

'''