#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import input_data

# 初始化权重
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# 定义模型，2层的隐藏层+ 3层的dropout
def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden): 
    X = tf.nn.dropout(X, p_drop_input) # 输入就开始用dropout
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_drop_hidden) # dropout
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_drop_hidden) # dropout

    return tf.matmul(h2, w_o)


# 加载数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# 定义占位符+ 初始化变量
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])

# dropout 的概率
p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# 模型
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)
print "py_x type = ",type(py_x)
# 损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                      p_keep_input: 0.8, p_keep_hidden: 0.5})
    
    print i, np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teY,
                                                     p_keep_input: 1.0,
                                                     p_keep_hidden: 1.0}))

