#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

'''
每天的五个数据 high,low,close,open,adj_close,
将其中一个看作是另外４个的函数，使用其余四个来对其中一个进行预测
'''
order = {"0":"adj_close","1":"high","2":"low","3":"close","4":"open"}
yahooData = np.load('5years_day.npy')
shape = yahooData.shape
# print "shape = ",shape
# print yahooData[0:10,3]
# print "--------------------------------------"
x_sample = yahooData[0:shape[0]-1,:]
y_sample = np.hsplit(yahooData,5)[3][1:shape[0]]
# print "y_sample.shape = ",y_sample.shape
# print "x_sample.shape = ",x_sample.shape
# print x_sample[0:10,:]
# print "--------------------------------------"
# print y_sample[0:10,]

train_start = 0
train_end = 1500
y_train = y_sample[train_start:train_end,:]
x_train = x_sample[train_start:train_end,:]
sample_number = x_train.shape[0]

test_start = 1000
test_end = 2000
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]

# print "y_test.shape = ",y_test.shape

nn_input = 5
layer_one = 2000
layer_two = 20
nn_output = 1
learn_rate = 0.001

erro_rate = 10
# 初始化权重
def init_weight(shape,name = None):
    return tf.Variable(tf.random_normal(shape, stddev=learn_rate),name = name)

def init_bias(shape,name = None):
    init = tf.zeros(shape)
    return tf.Variable(init, name=name)


def model(X, W, B):
    m = tf.matmul(X, W) + B
    # RELU for instead sigmoid, Sigmoid only for Final
    L = tf.nn.sigmoid(m)
    return L

# 定义占位符+ 初始化变量
X = tf.placeholder("float", [None, nn_input])
Y = tf.placeholder("float", [None, nn_output])


W1 = init_weight([nn_input, layer_one], 'W1')
B1 = init_bias([layer_one], 'B1')

W2 = init_weight([layer_one, layer_two], 'W2')
B2 = init_bias([layer_two], 'B2')


W3 = init_weight([layer_two, nn_output], 'W3')
B3 = init_bias([nn_output], 'B3')
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
train_op = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
pridict_op = tf.nn.relu(tf.matmul(L3, W3) + B3)


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

print "running ..."
sess.run(train_op, feed_dict={X: x_train, Y: y_train})
y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
erro_pridict = y_train - y_pridict
old_erro = np.abs(erro_pridict).mean()




itertimes = 200
batch_length = 20

for i in xrange(itertimes):
    
    for j in xrange(y_train.shape[0] - batch_length):

        x_batch = x_train[j:j+batch_length,:]
        y_batch = y_train[j:j+batch_length,:]
        sess.run(train_op, feed_dict={X:x_batch, Y:y_batch})

        y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
        erro_pridict = y_train - y_pridict
        old_erro = np.abs(erro_pridict).mean()

        print "itertimes = %d,old_erro = %f " % (itertimes,old_erro)



import os 
outdir = './npyfile/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

y_test_pridict = sess.run(pridict_op,feed_dict = {X:x_test})

erro_pridict = y_test - y_test_pridict

np.save(outdir + 'y_test_stock',y_test)
np.save(outdir + 'y_test_pridict_stock',y_test_pridict)

sess.close()


'''


from matplotlib import pyplot as plt
import os
outdir = './npyfile/'
imagedir = './ploter/'
if not os.path.exists(imagedir):
    os.mkdir(imagedir)


y_test = np.load(outdir + 'y_test_stock.npy')
y_test_pridict = np.load(outdir + 'y_test_pridict_stock.npy')
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

