#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os
from myNeurolNetworkModel import MyNeurolNetworkModel

order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume*e-6"}
yahooData = np.load('yahooData.npy')
Adj_Close,High,Low,Close,Open = np.hsplit(yahooData,5)
Volume = Adj_Close

yahooData = np.hstack([Open,High,Low,Close,Volume])

shape = yahooData.shape
print "shape = ",shape
print "-"*80
# print yahooData[0:10,:]


print "Close shape = ",Close.shape

myNNmodel = MyNeurolNetworkModel()
kdj = myNNmodel.calculate_kdj(yahooData)    #(None,3)
print "kdj = ",kdj.shape

declose = myNNmodel.calculate_dclose(Close)  #(None,4)
print "declose = ",declose.shape

logfit = myNNmodel.calculate_logfit(Close)   #(None,1)
print "logfit = ",logfit.shape

closeExponent = myNNmodel.calculate_exponent(Close,exponent = 0.9) #(None,1)
print "closeExponent = ",closeExponent.shape


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
newData = pca.fit_transform(np.hstack([Open,High,Low,Volume]))  #(None,2)
print "newData = ",newData.shape

x_sample = np.hstack([newData,declose,kdj,logfit,closeExponent])
y_sample = Close[1:]

x_train = np.vstack([x_sample[0:100,:],x_sample[300:500,:],x_sample[700:900,:],x_sample[1100:1150,:]])
y_train = np.vstack([y_sample[0:100,:],y_sample[300:500,:],y_sample[700:900,:],y_sample[1100:1150,:]])

print "x_train.shape = ",x_train.shape
sample_number = x_train.shape[0]

test_start = 800
test_end = 900
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]

outdir = os.path.join(os.path.dirname(__file__),'images/')
if len(outdir) >0 and (not os.path.exists(outdir)):
    os.mkdir(outdir)

myNNmodel.inputNumber = 11
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
plt.savefig(outdir + 'close.jpg')
plt.show()



acuracy = (y_test_predict - y_test)/y_test
plt.plot(acuracy,'ro')
plt.plot(acuracy,'r-')
plt.legend('acuracy')
plt.title('MyNeurolNetworkModel Test Acuracy')
plt.xlabel('index')
plt.ylabel(['acuracy'])
plt.grid(True)
plt.savefig(outdir + 'acurracy')
plt.show()
print "mean acuracy = ",np.mean(np.abs(acuracy))
