#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np

x = range(0,31)
x = np.array(x).reshape(31,1)

y = np.zeros(x.shape)
exponent = 0.87

for i in xrange(x.shape[0]):
    y[i,0] = np.power(exponent,x[i,0])
# y = y/np.sum(y)

from matplotlib import pyplot as plt

# plt.plot(y_sample,'ro')

plt.plot(x,y,'r-')
plt.plot(x,y,'ro')
plt.legend(['y_sample',])
plt.title('y_sample With Puer Exponent And Real Close')
plt.xlabel('index')
plt.ylabel(['Exponent'])
# plt.savefig(outdir + 'close_exponent')
plt.grid(True)
plt.show()