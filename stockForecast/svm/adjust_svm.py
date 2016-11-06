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