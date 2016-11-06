#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

'''使用前面几天的数据预测接下来一天的数据是升高还是降低'''


yahooData = np.load('yahooData.npy')
print "yahooData.shape = ",yahooData.shape
yahooData_length = yahooData.shape[0]   #总共数据
yahooData_colums = yahooData.shape[1]   #样本的维数　５维

related_number = 5

x_sample = np.zeros((yahooData_length - related_number,yahooData_colums*related_number))

max_index = yahooData_length - related_number


for i in xrange(yahooData_length - related_number):
    if i + related_number < yahooData_length:
        x_sample[i,:] = yahooData[i:i + related_number,:].reshape(1,yahooData_colums*related_number)

y_1 = yahooData[related_number-1:yahooData_length-1,:]
y_2 = yahooData[related_number:yahooData_length,:]

def sigmoidArray(array):
	shape = array.shape
	ret = np.zeros(shape,dtype = array.dtype)
	for i in xrange(shape[0]):
		for j in xrange(shape[1]):
			if array[i,j] < 0:
				ret[i,j] = -1
			elif array[i,j] == 0:
				ret[i,j] = 0
			else:
				ret[i,j] = 1
	return ret

print "y_1.shape = ",y_1.shape
print "y_2.shape = ",y_2.shape
y_sample = sigmoidArray(y_2-y_1)

output_number = 5

print "x_sample.shape = ",x_sample.shape
print "y_sample.shape = ",y_sample.shape

train_start = 0
train_end = 1000
x_train = x_sample[train_start:train_end,:]
y_train = y_sample[train_start:train_end,:]
print "x_train = ",x_train.shape
print "y_train = ",y_train.shape
train_length = x_train.shape[0]

test_start = 900
test_end = 1200
x_test = x_sample[test_start:test_end,:]
y_test = y_sample[test_start:test_end,:]
print "x_test = ",x_test.shape
print "y_test = ",y_test.shape


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
X = tf.placeholder("float", [None, yahooData_colums*related_number])
Y = tf.placeholder("float", [None, output_number])


W1 = init_weight([yahooData_colums*related_number, train_length], 'W1')
B1 = init_bias([train_length], 'B1')


W2 = init_weight([train_length, train_length/2], 'W2')
B2 = init_bias([train_length/2], 'B2')


W3 = init_weight([train_length/2, output_number], 'W3')
B3 = init_bias([output_number], 'B3')
# -------------------------------------
L2 = model(X,  W1, B1)
L3 = model(L2, W2, B2)


y_out = tf.nn.tanh(tf.matmul(L3, W3) + B3)


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

erro_rate = 0.02

run_times = 0
while(old_erro > erro_rate):
    start = np.random.randint(y_train.shape[0])
    end = np.random.randint(y_train.shape[0])
    
    if end < start:
        start,end = end,start

    x_batch = x_train[start:end,:]
    y_batch = y_train[start:end]
    sess.run(train_op, feed_dict={X:x_batch, Y:y_batch})

    y_train_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
    erro_pridict = y_train - y_train_pridict
    old_erro = np.abs(erro_pridict).mean()

    if run_times % 300 == 0:

        print "old_erro = ",old_erro
    run_times = run_times + 1

    print "run_times = %d,old_erro = %f,start = %d,end = %d" % (run_times,old_erro,start,end)

print "I have trianed %d times !!!!" % (run_times)

y_test_pridict = sess.run(pridict_op,feed_dict = {X:x_test})

sess.close()

outdir = './caogao/'
import os
if not os.path.exists(outdir):
    os.mkdir(outdir)

np.save(outdir + "y_test_01",y_test)
np.save(outdir + 'y_test_pridict_01',y_test_pridict)

'''
outdir = './caogao/'
y_test = np.load(outdir + "y_test_01.npy")
y_test_pridict = np.load(outdir + 'y_test_pridict_01.npy')
print "y_test.shape = ",y_test.shape
from matplotlib import pyplot as plt
x_asix = np.linspace(0,y_test.shape[0],y_test.shape[0]).reshape(y_test.shape[0],1)

high = y_test
high_pridict = y_test_pridict

plt.plot(x_asix,high,'ro')
plt.plot(x_asix,high,'r-')
plt.plot(x_asix,high_pridict,'bo')
plt.plot(x_asix,high_pridict,'b-')

plt.grid(True)
plt.legend(['real_value','pridict_value'])
plt.xlabel('time siries')
plt.ylabel('Value')
plt.title('Back Protagation Neurol Network To Pridict')
plt.savefig('time-one.jpg')
plt.show()
'''