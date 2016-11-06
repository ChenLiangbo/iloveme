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

'''
klist = range(1,9)

shape = len(klist)

knn_acurracy = []
knn = cv2.KNearest()
knn.train(x_train,y_train)

for k in klist: 
    ret, results, neighbours ,dist = knn.find_nearest(x_test, k)
    acurracy = (y_test == ret)
    knn_acurracy.append(np.mean(acurracy))
    print "k = %d,acurracy = %f" % (k,np.mean(acurracy))

from matplotlib import pyplot as plt

plt.plot(klist,knn_acurracy, 'r-')
plt.plot(klist,knn_acurracy, 'ro')
plt.xlabel('klist')
plt.ylabel('knn_acurracy')
plt.title('Adjust knn Acurracy with k')
plt.legend(['k'])
plt.grid(True)
plt.savefig('./ploter/knn_k.jpg')
plt.show()
'''


'''
depthlist = range(1,60)
# random trees
tree_acurracy = []


var_type = np.array([cv2.CV_VAR_NUMERICAL]*81 + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)

for depth in depthlist:
    RTtree = cv2.RTrees()
    rtree_params = dict(depth = depth)
    
    RTtree.train(x_train,cv2.CV_ROW_SAMPLE,y_train,
    	varType = var_type,
    	params = rtree_params)

    results = np.zeros(y_test.shape,dtype = y_test.dtype)
    for i in xrange(y_test.shape[0]):
        results[i,0] = RTtree.predict(x_test[i,:])


    acurracy = (y_test == results)

    tree_acurracy.append(np.mean(acurracy))

    print "depth =%d , acurracy = %f" % (depth,np.mean(acurracy))

max_acurracy = max(tree_acurracy)
index = tree_acurracy.index(max_acurracy)
print "max_acurracy = %f,depth = %d" % (max_acurracy,depthlist[index])

from matplotlib import pyplot as plt

plt.plot(depthlist,tree_acurracy, 'r-')
plt.plot(depthlist,tree_acurracy, 'ro')
plt.xlabel('depthlist')
plt.ylabel('RTtree acurracy')
plt.title('RTtree acurracy with depth')
plt.legend(['depth'])
plt.grid(True)
plt.savefig('./ploter/RTtree_depth.jpg')
plt.show()

RTtree = cv2.RTrees()
rtree_params = dict(depth = 32)
    
RTtree.train(x_train,cv2.CV_ROW_SAMPLE,y_train,
    	varType = var_type,
    	params = rtree_params)

results = np.zeros(y_test.shape,dtype = y_test.dtype)
for i in xrange(y_test.shape[0]):
    results[i,0] = RTtree.predict(x_test[i,:])


acurracy = (y_test == results)

tree_acurracy.append(np.mean(acurracy))

print "depth =%d , acurracy = %f" % (depth,np.mean(acurracy))
'''

depthlist = range(1,50)
boost_acurracy = []

for depth in depthlist:

    boost_params = dict(max_depth = depth)

    var_type = np.array([cv2.CV_VAR_NUMERICAL]*81 + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)
    Boost = cv2.Boost()
    Boost.train(x_train,cv2.CV_ROW_SAMPLE,y_train,varType = var_type,params = boost_params)

    results = np.zeros(y_test.shape,dtype = y_test.dtype)

    for i in xrange(y_test.shape[0]):
        results[i,0] = Boost.predict(x_test[i,:],returnSum = True)
    
    # print results[0:10]
    # print "-"*80

    for i in xrange(y_test.shape[0]):
        if results[i,0] >0:
            results[i,0] = 1
        else:
            results[i,0] = 0

    acurracy = (y_test == results)
    

    print "Boost acurracy = ",np.mean(acurracy)

    boost_acurracy.append(np.mean(acurracy))

max_acurracy = max(boost_acurracy)
index = boost_acurracy.index(max_acurracy)
print "max_acurracy = %f,depth = %d" % (max_acurracy,depthlist[index])

from matplotlib import pyplot as plt

plt.plot(depthlist,boost_acurracy, 'r-')
plt.plot(depthlist,boost_acurracy, 'ro')
plt.xlabel('depthlist')
plt.ylabel('Boost acurracy')
plt.title('Boost acurracy with depth')
plt.legend(['depth'])
plt.grid(True)
plt.savefig('./ploter/Boost_depth.jpg')
plt.show()
