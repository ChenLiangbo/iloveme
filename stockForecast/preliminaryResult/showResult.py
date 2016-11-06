#!usr/bin/env/python 
# -*- coding: utf-8 -*-
'''
目的　　：　本程序的目的是将nn4.py文件的预测值去归一化之后同真实数据做比较
预测值：　归一化之后的预测值存放在　npyfile　文件夹中
真实值：　真实的数据存放在　dataset　文件夹中，真实的数据来自 getDataWithArima.py 归一化之前
'''
import numpy as np
import os

outdir = './finalfile/'


x_real = np.load(outdir + 'x_train_real.npy')
y_real = np.load(outdir + 'y_train_real.npy')

ymax = np.amax(y_real, axis=0)
ymin = np.amin(y_real, axis=0)
print "ymax = %f,ymin = %f " % (ymax,ymin)

x_normalized = np.load(outdir+ 'x_train_normalized.npy')
y_normalized = np.load(outdir + 'y_train_normalized.npy')

train_start = 600
train_end = 1150

test_start = 1000
test_end = 1200

y_train_real = y_real[train_start:train_end,:]
y_test_real = y_real[test_start:test_end,:]


# y_train = (y_train - ymin) / (ymax - ymin)
y_train_predict = np.load( outdir + 'y_train_predict.npy')

y_train_denormalized = y_train_predict*(ymax - ymin) + ymin

y_test_predict = np.load(outdir + 'y_test_pridict.npy')
y_test_denormalized = y_test_predict*(ymax - ymin) + ymin

from matplotlib import pyplot as plt
outdir = './images/'
if not os.path.exists(outdir):
    os.mkdir(outdir)
# plt.plot(y_train_real,'ro')
# plt.plot(y_train_predict,'bo')
plt.plot(y_train_real,'r-')
plt.plot(y_train_denormalized,'b-')
plt.xlabel('index')
plt.ylabel('value')
plt.title('arima-nn denormalized training predict')
plt.legend(['y_train_real','y_train_denormalized'])
plt.grid(True)
plt.savefig( outdir + 'y_train_denormalized.jpg')
plt.show()





plt.plot(y_test_real,'ro')
plt.plot(y_test_denormalized,'bo')
plt.plot(y_test_real,'r-')
plt.plot(y_test_denormalized,'b-')
plt.xlabel('index')
plt.ylabel('value')
plt.title('arima-nn denormalized test predict')
plt.legend(['y_test_real','y_test_denormalized'])
plt.grid(True)
plt.savefig(outdir + 'y_test_denormalized.jpg')
plt.show()


train_acurracy = y_train_denormalized - y_train_real
shape = train_acurracy.shape
for i in xrange(shape[0]):
    train_acurracy[i,0] = (y_train_denormalized[i,0] - y_train_real[i,0])/y_train_real[i,0]
plt.plot(train_acurracy,'b-')
plt.xlabel('index')
plt.ylabel('train_acurracy')
plt.title('arima-nn train predict acurracy ')
plt.legend(['train_acurracy',])
plt.grid(True)
plt.savefig(outdir + 'y_train_acurracy.jpg')
plt.show()

print "mean train acurracy = ",np.mean(train_acurracy)



test_acrracy = y_test_denormalized - y_test_real
shape = test_acrracy.shape
for i in xrange(shape[0]):
    test_acrracy[i,0] = (y_test_denormalized[i,0] - y_test_real[i,0])/y_test_real[i,0]


plt.plot(test_acrracy,'b-')
plt.xlabel('index')
plt.ylabel('test_acrracy')
plt.title('arima-nn test predict acurracy ')
plt.legend(['test_acrracy',])
plt.grid(True)
plt.savefig(outdir + 'y_test_acurracy.jpg')
plt.show()

print "mean train acurracy = ",np.mean(test_acrracy)