#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

'''
对一列数据按照时间序列进行拟合
'''
related_number = 5

yahooData = np.load('yahooData.npy')
print "yahooData.shape = ",yahooData.shape

sample = np.hsplit(yahooData,5)[0]
print "sample.shape = ",sample.shape

sample_length = sample.shape[0]
sample_rows = sample.shape[1]
print "sample_length = ",sample_length
print "sample_rows = ",sample_rows


x_sample = np.zeros((sample_length - related_number,related_number*sample_rows))
print "x_sample.shape = ",x_sample.shape

for i in xrange(sample_length):
    if i+related_number < sample.shape[0]:
        x_sample[i,:] = sample[i:i+related_number,:].reshape(1,related_number*sample_rows)


y_sample = sample[related_number:]
print "y_sample.shape = ",y_sample.shape
print "x_sample.shape = ",x_sample.shape

train_start = 0
train_end = 1000
x_train = x_sample[train_start:train_end,:]
y_train = y_sample[train_start:train_end,:]
train_length = x_train.shape[0]
print "x_train.shape = ",x_train.shape
print "y_train.shape = ",y_train.shape

test_start = 70
test_end = 120
x_test = x_sample[test_start:test_end,:]
y_test = y_sample[test_start:test_end,:]

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
X = tf.placeholder("float", [None, related_number])
Y = tf.placeholder("float", [None, 1])

'''hidden layer 1'''
W1 = init_weight([related_number, train_length], 'W1')
B1 = init_bias([train_length], 'B1')

'''hidden layer 2'''
W2 = init_weight([train_length, train_length/2], 'W2')
B2 = init_bias([train_length/2], 'B2')

'''ouput layer'''
W3 = init_weight([train_length/2, 1], 'W3')
B3 = init_bias([1], 'B3')
# -------------------------------------
L2 = model(X,  W1, B1)
L3 = model(L2, W2, B2)

'''output'''
y_out = tf.nn.relu(tf.matmul(L3, W3) + B3)

'''损失函数'''
cost = tf.reduce_mean(tf.square((Y - y_out)))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
pridict_op = tf.nn.relu(tf.matmul(L3, W3) + B3)


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

test_number = 100
x_test = x_sample[0:test_number,:]
y_test = y_sample[0:test_number,:]



sess.run(train_op,feed_dict = {X:x_train,Y:y_train})
y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
erro_pridict = y_train - y_pridict
erro_mean = np.abs(erro_pridict).mean()


erro_rate = 2

run_times = 0
while(erro_mean > erro_rate):
    start = np.random.randint(y_train.shape[0])
    end = np.random.randint(y_train.shape[0])
	
    if start < end:
        start,end = end,start
    # print "start = %f,end = %f" % (start,end)
    # print "x_train.shape = ",x_train.shape
    # print "y_train.shape = ",y_train.shape

    x_batch = x_train[start:end,:]
    y_batch = y_train[start:end,:]
    
    # print "x_batch.shape = ",x_batch.shape
    # print "y_batch.shape = ",y_batch.shape

    sess.run(train_op,feed_dict = {X:x_batch,Y:y_batch})

    y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
    erro_pridict = y_train - y_pridict
    erro_mean = np.abs(erro_pridict).mean()
    # print "erro_mean = ",erro_mean
    
    print "run_times = %d,erro_mean = %f,start = %d,end = %d " % (run_times,erro_mean,start,end)
    
    run_times = run_times + 1

y_test_pridict = sess.run(pridict_op,feed_dict = {X:x_test})
outdir = "./nn_times/"
import os 
if not os.path.exist(outdir):
    os.mkdir(outdir)

np.save(outdir + 'y_test',y_test)
np.save(outdir + 'y_test_pridict',y_test_pridict)


from matplotlib import pyplot as plt
outdir = "./nn_times/"
y_test = np.load(outdir + 'y_test.npy')
y_test_pridict = np.load(outdir + 'y_test_pridict.npy')

x_axis = np.linspace(0,y_test.shape[0],y_test.shape[0]).reshape(y_test.shape[0],1)
plt.plot(x_axis,y_test,'ro')
plt.plot(x_axis,y_test,'r-')
plt.plot(x_axis,y_test_pridict,'bo')
plt.plot(x_axis,y_test_pridict,'b-')
plt.grid(True)
plt.legend(['y_test','y_test_pridict'])
plt.xlabel('time-serize')
plt.ylabel('Value')
plt.title('Back Protagation Neurol Network To Pridict With Time-serize')
plt.savefig(outdir + 'plot.jpg')
plt.show()
