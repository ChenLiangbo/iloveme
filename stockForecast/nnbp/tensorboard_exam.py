#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

'''
每天的五个数据 high,low,close,open,adj_close,
将其中一个看作是另外４个的函数，使用其余四个来对其中一个进行预测
'''

yahooData = np.load('yahooData.npy')
adj_close,high,low,close,openPrice = np.hsplit(yahooData,5)
# print "openPrice.shape = ",openPrice.shape
sample_number = yahooData.shape[0]

x_sample = np.hstack([adj_close[0:sample_number],high[0:sample_number],low[0:sample_number],close[0:sample_number]])
y_sample = openPrice[0:sample_number]
print "y_sample.shape = ",y_sample.shape
print "x_sample.shape = ",x_sample.shape

train_start = 0
train_end = 1000
y_train = y_sample[train_start:train_end,:]
x_train = x_sample[train_start:train_end,:]
sample_number = x_train.shape[0]

test_start = 900
test_end = 1100
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
	L = tf.nn.sigmoid(m)
	return L

# 定义占位符+ 初始化变量
X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 1])


W1 = init_weight([4, sample_number], 'W1')
B1 = init_bias([sample_number], 'B1')

W2 = init_weight([sample_number, sample_number/2], 'W2')
B2 = init_bias([sample_number/2], 'B2')


W3 = init_weight([sample_number/2, 1], 'W3')
B3 = init_bias([1], 'B3')
# -------------------------------------
L2 = model(X,  W1, B1)
L3 = model(L2, W2, B2)

y_out = tf.nn.relu(tf.matmul(L3, W3) + B3)


# with tf.name_scope('loss') as scope:
#     # 1st, cross_entropy, on backpropagation
#     cost = -tf.reduce_mean(Y * tf.log(y_out) + (1-Y) * tf.log(1 - y_out))
# 损失函数

cost = tf.reduce_mean(tf.square((Y - y_out)))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_out, Y)) 
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

merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('./logdir', GraphDef = sess.graph)

erro_rate = 0.2

run_times = 0
while(old_erro > erro_rate):
    start = np.random.randint(y_train.shape[0])
    end = np.random.randint(y_train.shape[0])
    if end < start:
        start,end = end,start

    x_batch = x_train[start:end,:]
    y_batch = y_train[start:end,:]
    sess.run(train_op, feed_dict={X:x_batch, Y:y_batch})

    y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
    erro_pridict = y_train - y_pridict
    old_erro = np.abs(erro_pridict).mean()

    if run_times % 200 == 0:
        print "old_erro = ",old_erro
        summary_str = session.run(merged_summary_op)
        summary_writer.add_summary(summary_str, run_times)

    run_times = run_times + 1

    print "start = %d, end = %d, run_times = %d,old_erro = %f" % (start,end,run_times,old_erro)


print "I have trianed %d times !!!!" % (run_times)


import os 
outdir = './npyfile/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

y_test_pridict = sess.run(pridict_op,feed_dict = {X:x_test})

erro_pridict = y_test - y_test_pridict

np.save(outdir + 'y_test_board',y_test)
np.save(outdir + 'y_test_pridict_board',y_test_pridict)

sess.close()
