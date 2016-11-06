#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from itertools import izip


x_sample = np.float32(np.load('./npyfile/xSample.npy'))
y_sample = np.float32(np.load('./npyfile/ySample.npy'))
print "y_sample.shape = ",y_sample.shape
print "x_sample.shape = ",x_sample.shape
# print x_sample[0:10,:]
# print "--------------------------------------"
# print y_sample[0:10,]

train_start = 600
train_end = 1150
y_train = y_sample[train_start:train_end,:]
x_train = x_sample[train_start:train_end,:]
sample_number = x_train.shape[0]

test_start = 1000
test_end = 1200
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]


tf.InteractiveSession()


def init_weight(shape, name):
    init = tf.random_uniform(shape, -1.0, 1.0)
    return tf.Variable(init, name=name)

def init_bias(shape, name):
    init = tf.zeros(shape)
    return tf.Variable(init, name=name)

def NN_model(X, W, B):
    m = tf.matmul(X, W) + B
    return tf.nn.relu(m)

X = tf.placeholder('float32', [None, 10], name='Input')
print ('X input.')
Y = tf.placeholder('float32', [None, 1], name='Output')
print ('Y output.')

W1 = init_weight([10, 6], 'W1')
B1 = init_bias([6], 'B1')

W2 = init_weight([6, 6], 'W2')
B2 = init_weight([6], 'B2')

W3 = init_weight([6, 1], 'W3')
B3 = init_bias([1], 'B2')

L2 = NN_model(X, W1, B2)
L3 = NN_model(L2, W2, B2)

hypothesis = tf.sigmoid(tf.matmul(L3, W3) + B3)
print ( 'Calculate hypothesis.' )

with tf.name_scope('loss') as scope:
    loss = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))

with tf.name_scope('train') as scope:
    lr = tf.Variable(1.0) # Learning rate
    optimizer = tf.train.GradientDescentOptimizer(lr)
    batch_train = optimizer.minimize(loss)


init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(10000):
        _, loss_value = sess.run([batch_train, loss], feed_dict={ X: x_train, Y: y_train })
        
        if epoch % 10 == 0:
            print ( 'Epoch: {0} Loss: {1}'.format(epoch, loss_value) )
    
    predict = sess.run(hypothesis, feed_dict={ X: x_train })

    yt, pr = [], []
    for i, j in izip(y_train, predict):
        yt.append(i[0])
        pr.append(j[0])
    
    print "training acurracy = ",np.mean(y_train - predict)
    np.save('./npyfile/y_train1',y_train)
    np.save('./npyfile/y_train_predict1',predict)
    

    y_test_predict = sess.run(hypothesis, feed_dict={ X: x_test})
    print "valiation acurracy = ",np.mean(y_test - y_test_predict)

    np.save('./npyfile/y_test1',y_test)
    np.save('./npyfile/y_test_predict1',y_test_predict)
    
    save_path = './model_saver/nn4Model.ckpt'
    saver = tf.train.Saver()
    saver.save(sess,save_path)

y_train = np.load('./npyfile/y_train1.npy')
predict = np.load('./npyfile/y_train_predict1.npy')

save_path = './model_saver/nn4Model.ckpt'
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
print "ok"
saver.restore(sess,save_path)

# matplotlib inline
import matplotlib.pylab as plt
from matplotlib.finance import candlestick
from matplotlib.pylab import rcParams
# rcParams['figure.figsize'] = 15, 6

# plt.plot(yt, 'ro')
# plt.plot(pr, 'bo')
plt.plot(yt, 'r-')
plt.plot(pr, 'b-')
plt.xlabel('index')
plt.ylabel('value')
plt.title('arima-nn training predict')
plt.legend(['y_train','y_train_predict'])
plt.grid(True)
plt.savefig('./ploter/arima3_training.jpg')
plt.show()


y_test = np.load('./npyfile/y_test1.npy')
y_test_predict = np.load('./npyfile/y_test_predict1.npy')

# plt.plot(y_test, 'ro')
# plt.plot(y_test_predict,'bo')
plt.plot(y_test, 'r-')
plt.plot(y_test_predict,'b-')
plt.xlabel('index')
plt.ylabel('value')
plt.title('arima-nn valiation predict')
plt.legend(['y_test','y_test_predict'])
plt.grid(True)
plt.savefig('./ploter/arima3_valiation.jpg')
plt.show()
