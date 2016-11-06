#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import input_data

# 初始化权重 w
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# 定义网络模型，只是基本的mlp模型，堆叠两层的逻辑回归
def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) 
    return tf.matmul(h, w_o) #这里没有用softmax

# 加载数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# 定义占位符
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])


# 初始化模型参数
w_h = init_weights([784, 625]) 
w_o = init_weights([625, 10])

# 定义模型
py_x = model(X, w_h, w_o)

# 定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) 
# 定义训练操作
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
# 定义测试操作
predict_op = tf.argmax(py_x, 1)

# 定义并初始化会话
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


# 训练测试
for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    print i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX, Y: teY}))