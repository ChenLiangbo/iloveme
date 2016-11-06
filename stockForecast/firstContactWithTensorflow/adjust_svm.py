#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import input_data
import numpy as np
import cv2


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sample = mnist.train.next_batch(5000)

xSample = sample[0]
ySample = sample[1]

yshape = ySample.shape
y = np.zeros((ySample.shape[0],1),dtype = np.float32)

for row in range(yshape[0]):
    index = np.argwhere(ySample[row] == 1)
    index = index[0,0]
    y[row,0] = index

# print "y = ",y

train_number = 4000
x_train = xSample[0:train_number,:]
y_train = y[0:train_number,:]

x_test = xSample[train_number:,:]
y_test = y[train_number:,:]

'''
Cs = np.logspace(0, 10, 15, base=2)
gammas = np.logspace(-7, 4, 15, base=2)

shape = Cs.shape

svm_acurracy = np.zeros(shape,dtype = np.float32)

for c in xrange(shape[0]):
    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	               svm_type = cv2.SVM_C_SVC,
                   C=Cs[c], gamma = cv2.SVM_GAMMA)
    svm = cv2.SVM()
    svm.train(x_train,y_train,params = svm_params)
    # svm.save('svm_model.dat')
    ret = svm.predict_all(x_test)
    # print "ret = ",ret
    acurracy = (y_test == ret)

    svm_acurracy[c] = np.mean(acurracy)
    print "c = %d,acurracy = %f" % (c,np.mean(acurracy))
from matplotlib import pyplot as plt

plt.plot(Cs,svm_acurracy, 'r-')
plt.plot(Cs,svm_acurracy, 'ro')
plt.xlabel('Cs')
plt.ylabel('svm_acurracy')
plt.title('Adjust SVM Acurracy with C')
plt.legend(['Cs'])
plt.grid(True)
plt.savefig('./ploter/svm_C.jpg')
plt.show()

svm_acurracy = np.zeros(shape,dtype = np.float32)

for c in xrange(shape[0]):
    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	               svm_type = cv2.SVM_C_SVC,
                   C=3.75, gamma = gammas[c])
    svm = cv2.SVM()
    svm.train(x_train,y_train,params = svm_params)
    # svm.save('svm_model.dat')
    ret = svm.predict_all(x_test)
    # print "ret = ",ret
    acurracy = (y_test == ret)

    svm_acurracy[c] = np.mean(acurracy)
    print "c = %d,acurracy = %f" % (c,np.mean(acurracy))
from matplotlib import pyplot as plt

plt.plot(gammas,svm_acurracy, 'r-')
plt.plot(gammas,svm_acurracy, 'ro')
plt.xlabel('gammas')
plt.ylabel('svm_acurracy')
plt.title('Adjust SVM Acurracy with gammas')
plt.legend(['gammas'])
plt.grid(True)
plt.savefig('./ploter/svm_gammas.jpg')
plt.show()
'''
klist = range(1,9)

shape = len(klist)

knn_acurracy = []
knn = cv2.KNearest()
knn.train(x_train,y_train)

for k in klist: 
    ret, results, neighbours ,dist = knn.find_nearest(x_test, k)
    acurracy = (y_test == results)
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