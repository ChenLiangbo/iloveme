#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

'''
使用
'''

yahooData = np.load('yahooData.npy')
adj_close,high,low,close,openPrice = np.hsplit(yahooData,5)
# print "openPrice.shape = ",openPrice.shape

sample_number = openPrice.shape[0]
x_sample = np.hstack([adj_close[0:sample_number],high[0:sample_number],low[0:sample_number],close[0:sample_number]])
y_sample = openPrice[0:sample_number]
print "y_sample.shape = ",y_sample.shape
print "x_sample.shape = ",x_sample.shape

train_start = 0
train_end = 100
y_train = y_sample[train_start:train_end,:]
x_train = x_sample[train_start:train_end,:]


test_start = 100
test_end = 110
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]

# print "y_test.shape = ",y_test.shape

# 初始化权重
def init_weight(shape,name = None):
    return tf.Variable(tf.random_normal(shape, stddev=0.01),name = name)

def init_bias(shape,name = None):
	init = tf.zeros(shape)
	return tf.Variable(init, name=name)


def model(X, W, B):
	m = tf.matmul(X, W) + B
	# RELU for instead sigmoid, Sigmoid only for Final
	L = tf.nn.softmax(m)
	return L

# 定义占位符+ 初始化变量
X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 1])


W1 = init_weight([4, 4], 'W1')
B1 = init_bias([4], 'B1')

W2 = init_weight([4, 20], 'W2')
B2 = init_bias([20], 'B2')


W3 = init_weight([20, 1], 'W3')
B3 = init_bias([1], 'B3')
# -------------------------------------
L2 = model(X,  W1, B1)
L3 = model(L2, W2, B2)

y_out = tf.nn.relu(tf.matmul(L3, W3) + B3)
print "y_out = ",type(y_out)

# 损失函数
cost = tf.reduce_mean(tf.square((Y - y_out)))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
pridict_op = tf.nn.relu(tf.matmul(L3, W3) + B3)


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

print "running ..."
sess.run(train_op, feed_dict={X: x_train, Y: y_train})
y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
erro_pridict = y_train - y_pridict
old_erro = np.abs(erro_pridict).mean()

erro_rate = 3.07

run_times = 0
while(old_erro > erro_rate):
    sess.run(train_op, feed_dict={X:x_train, Y:y_train})
    if run_times % 300 == 0:
    	y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
        erro_pridict = y_train - y_pridict
        old_erro = np.abs(erro_pridict).mean()
        print "old_erro = ",old_erro
    run_times = run_times + 1

print "I have trianed %d times !!!!" % (run_times)

'''plot y_train;y_pridict'''
from matplotlib import pyplot as plt
x_axis = np.linspace(0,y_train.shape[0],y_train.shape[0]).reshape(y_train.shape[0],1)
plt.plot(x_axis,y_train,'ro')
plt.plot(x_axis,y_train,'r-')
plt.plot(x_axis,y_pridict,'bo')
plt.plot(x_axis,y_pridict,'b-')
plt.grid(True)
plt.legend(['y_train','y_pridict'])
plt.xlabel('reversed-time')
plt.ylabel('Value')
plt.title('The Pridiction on Train Dataset')
plt.show()


'''Plot Test Dataset,y_test,new_pridict'''
new_pridict = sess.run(pridict_op,feed_dict = {X:x_test})
print "new_pridict.shape = ",new_pridict.shape
xnew_asix = np.linspace(0,new_pridict.shape[0],new_pridict.shape[0]).reshape(new_pridict.shape[0],1)
erro_pridict = y_test - new_pridict
print "erro_mean = ",np.abs(erro_pridict).mean()
plt.plot(xnew_asix,y_test,'ro')
plt.plot(xnew_asix,y_test,'r-')
plt.plot(xnew_asix,new_pridict,'bo')
plt.plot(xnew_asix,new_pridict,'b-')
plt.grid(True)
plt.legend(['y_test','new_pridict'])
plt.xlabel('Time-Seris')
plt.ylabel('Value')
plt.title('The Pridiction on Test Dataset')
# plt.show()
plt.show('y_test')
sess.close()

