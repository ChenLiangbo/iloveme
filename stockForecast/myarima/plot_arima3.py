import numpy as np
from itertools import izip


'''
y_train = np.load('./npyfile/y_train.npy')
predict = np.load('./npyfile/predict.npy')

yt = []
pr = []
for i, j in izip(y_train, predict):
    yt.append(i[0])
    pr.append(j[0])

# matplotlib inline
import matplotlib.pylab as plt
from matplotlib.finance import candlestick
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

plt.plot(yt, label='Original', color='green')
plt.plot(pr, label='NN', color='red')
plt.xlabel('index')
plt.ylabel('value')
plt.title('arima-nn predict')
plt.legend(loc='best')
plt.grid(True)
plt.plot(yt,'go')
plt.plot(pr,'ro')
plt.savefig('./ploter/arima3.jpg')
plt.show()
'''

'''
import matplotlib.pylab as plt
y_test = np.load('./npyfile/y_test.npy')
y_test_predict = np.load('./npyfile/y_test_predict.npy')

plt.plot(y_test[50:150], 'ro')
plt.plot(y_test_predict[50:150],'bo')
plt.plot(y_test[50:150], 'r-')
plt.plot(y_test_predict[50:150],'b-')
plt.xlabel('index')
plt.ylabel('value')
plt.title('arima-nn valiation predict')
plt.legend(['y_test','y_test_predict'])
plt.grid(True)
plt.savefig('./ploter/arima3_valiation50-150.jpg')
plt.show()
'''


from matplotlib import pyplot as plt
outdir = './npyfile/'
imagedir = './ploter/'




title = "plot_nn4"
y_test = np.load(outdir + 'y_test_close.npy')
y_test_pridict = np.load(outdir + 'y_test_pridict_close.npy')
print "y_test.shape = ",y_test.shape
print "y_test_pridict.shape = ",y_test_pridict.shape

x_axis = np.linspace(0,y_test.shape[0],y_test.shape[0]).reshape(y_test.shape[0],1)
start = 0
end = 200

plt.plot(x_axis[start:end,:],y_test[start:end,:],'ro')
plt.plot(x_axis[start:end,:],y_test_pridict[start:end,:],'bo')
plt.plot(x_axis[start:end,:],y_test[start:end,:],'r-')
plt.plot(x_axis[start:end,:],y_test_pridict[start:end,:],'b-')
plt.grid(True)
plt.legend(['y_test','y_test_pridict'])
plt.xlabel('reversed-time')
plt.ylabel('Value')
title = 'The Pridiction on ' + str(start) +'--' + str(end) + ' Train Dataset'
plt.title(title)
imgname = imagedir + "pridict" + str(start) +'--' + str(end) + '.jpg'
plt.savefig(imgname)
plt.show()
'''

title = 'plot nn4 y_train and valiation'
outdir = './npyfile/'
from matplotlib import pyplot as plt
y_train = np.load(outdir + 'nn4_y_train.npy')
y_predict = np.load(outdir + 'nn4_y_train_predict.npy')
plt.plot(y_train,'ro')
plt.plot(y_predict,'bo')
plt.plot(y_train,'r-')
plt.plot(y_predict,'b-')
plt.grid(True)
plt.legend(['nn4_y_train','nn4_y_train_predict'])
plt.xlabel('index')
plt.ylabel('Value')
title = 'nn4 predict with arima Dataset'
plt.title(title)
imgname = 'nn4_y_train_predict'
plt.savefig(imgname)
plt.show()

'''