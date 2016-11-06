#!usr/bin/env/python 
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import os
outdir = './npyfile/'
imagedir = './ploter/'
if not os.path.exists(imagedir):
    os.mkdir(imagedir)

x_sample = np.load(outdir + 'x_asix.npy')
y_train = np.load(outdir + 'y_train.npy')
y_pridict = np.load(outdir + 'y_pridict.npy')
print "y_train.shape = ",y_train.shape
print "y_pridict.shape = ",y_pridict.shape
x_axis = np.linspace(0,x_sample.shape[0],x_sample.shape[0]).reshape(x_sample.shape[0],1)
start = 700
end = 900
plt.plot(x_axis[start:end,:],y_train[start:end,:],'ro')
plt.plot(x_axis[start:end,:],y_train[start:end,:],'r-')
plt.plot(x_axis[start:end,:],y_pridict[start:end,:],'bo')
plt.plot(x_axis[start:end,:],y_pridict[start:end,:],'b-')
plt.grid(True)
plt.legend(['y_train','y_pridict'])
plt.xlabel('reversed-time')
plt.ylabel('Value')
title = 'The Pridiction on ' + str(start) +'--' + str(end) + ' Train Dataset'
plt.title(title)
imgname = imagedir + "pridict" + str(start) +'--' + str(end) + '.jpg'
plt.savefig(imgname)
plt.show()

'''Plot Test Dataset,y_test,new_pridict'''

xnew_asix = np.load(outdir + 'xnew_asix.npy')
y_test = np.load(outdir + 'y_test.npy')
test_pridict = np.load(outdir + 'test_pridict.npy')

print "y_test.shape = ",y_test.shape
print "test_pridict.shape = ",test_pridict.shape

plt.plot(xnew_asix,y_test,'ro')
plt.plot(xnew_asix,y_test,'r-')
plt.plot(xnew_asix,test_pridict,'bo')
plt.plot(xnew_asix,test_pridict,'b-')
plt.grid(True)
plt.legend(['y_test','test_pridict'])
plt.xlabel('Time-Seris')
plt.ylabel('Value')
plt.title('The Pridiction on Test Dataset')
plt.show('y_test.jpg')
plt.savefig('test_pridict.jpg')
