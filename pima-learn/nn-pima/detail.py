#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np

trainDetail = np.load('trainDetail.npy')
print "shape = ",trainDetail.shape

from matplotlib import pyplot as plt
plt.plot(trainDetail[0,:],trainDetail[1,:],'r-')
plt.plot(trainDetail[0,:],trainDetail[2,:],'b-')
plt.title('Accuracy As Traing Times')
plt.xlabel('train times')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.grid(True)
plt.savefig('./result/train-accuracy')
plt.show()