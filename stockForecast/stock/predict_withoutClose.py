#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os
from myNeurolNetworkModel import MyNeurolNetworkModel

'''
每天的五个数据 high,low,close,open,adj_close,
'''
order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume*e-6"}
yahooData = np.load('yahoo_finance5.npy')
Open,High,Low,Close,Volume = np.hsplit(yahooData,5)

shape = yahooData.shape
print "shape = ",shape
print "-"*80

print "Close shape = ",Close.shape

myNNmodel = MyNeurolNetworkModel()
kdj = myNNmodel.calculate_kdj(yahooData)
print "kdj = ",kdj.shape

declose = myNNmodel.calculate_dclose(Close)
print "declose = ",declose.shape


logfit = myNNmodel.calculate_logfit(Close)
print "logfit = ",logfit.shape

closeExponent = myNNmodel.calculate_exponent(Close)
print "closeExponent = ",closeExponent.shape

# x_sample = np.hstack([Open,High,Low,Volume,kdj,logfit,closeExponent])
x_sample = np.hstack([Open,kdj,logfit,closeExponent])
y_sample = Close[1:]

x_train = np.vstack([x_sample[0:100,:],x_sample[300:500,:],x_sample[700:900,:],x_sample[1100:1150,:]])
y_train = np.vstack([y_sample[0:100,:],y_sample[300:500,:],y_sample[700:900,:],y_sample[1100:1150,:]])

print "x_train.shape = ",x_train.shape
sample_number = x_train.shape[0]

test_start = 1150
test_end = 1200
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]

outdir = os.path.join(os.path.dirname(__file__),'predict_withoutClose/')


myNNmodel.inputNumber = 6
myNNmodel.errorRate = 0.0105
myNNmodel.learningRate = 0.001

# myNNmodel.train(x_train,y_train)

print "myNNmodel train successfully ..."

y_test_predict = myNNmodel.predict(x_test)
from matplotlib import pyplot as plt

plt.plot(y_test,'ro')
plt.plot(y_test_predict,'bo')
plt.plot(y_test,'r-')
plt.plot(y_test_predict,'b-')
plt.legend(['y_test','y_test_predict'])
plt.grid(True)
plt.xlabel('index')
plt.ylabel('value')
plt.title('MyNeurolNetworkModel Predict Close With AllOneDay')
plt.savefig(outdir + 'close6_1150-1200.jpg')
plt.show()



acuracy = (y_test_predict - y_test)/y_test
plt.plot(acuracy,'ro')
plt.plot(acuracy,'r-')
plt.legend('acuracy')
plt.title('MyNeurolNetworkModel Test Acuracy')
plt.xlabel('index')
plt.ylabel(['acuracy'])
plt.savefig(outdir + 'acurracy6_1150-1200')
plt.grid(True)
plt.show()
print "mean acuracy = ",np.mean(np.abs(acuracy))
'''



import cv2
x_sample = np.hstack([Open,High,Low,Volume,kdj,logfit,closeExponent])
y_sample = np.zeros(y_sample.shape)

for i in xrange(shape[0]-1):
    if Close[i+1,0] < Close[i,0]:
        y_sample[i,0] = 0
    else:
        y_sample[i,0] = 1

x_sample = myNNmodel.normalize_xtest(x_sample)

x_sample = np.float32(x_sample)
y_sample = np.float32(y_sample)

x_train = np.vstack([x_sample[0:100,:],x_sample[300:500,:],x_sample[700:900,:],x_sample[1100:1150,:]])
y_train = np.vstack([y_sample[0:100,:],y_sample[300:500,:],y_sample[700:900,:],y_sample[1100:1150,:]])

test_start = 100
test_end = 300
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]

knn = cv2.KNearest()
knn.train(x_train,y_train)
ret, results, neighbours ,dist = knn.find_nearest(np.float32(x_test),1)

print "knn acuracy with nn's output = ",np.mean(results == y_test)


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