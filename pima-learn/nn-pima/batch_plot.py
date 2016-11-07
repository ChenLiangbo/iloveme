#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_excel("./result/pima-nn-result.xlsx")

df = df[0:16]
batch = df['batch']
print "batch = "
print batch[0:17]

keys = ['batch','recall','fmeasure','precision','accuracy']
shapes1 = ['ro','go','bo','mo']
shapes2 = ['r-','g-','b-','m-']

from matplotlib import pyplot as plt
for i in range(1,5):
    plt.plot(df[keys[0]],df[keys[i]],shapes1[i-1])
plt.legend(keys[1:])
for i in range(1,5):
    plt.plot(df[keys[0]],df[keys[i]],shapes2[i-1])
plt.grid(True)
plt.title('accuracy with batch size ')
plt.xlabel('batch size')
plt.ylabel('value')
plt.savefig('./result/batchsize')
plt.show()