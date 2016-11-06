#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

x_sample = np.load('./sample/x_sample.npy')
y_sample = np.load('./sample/y_sample.npy')
print "x_sample = ",x_sample.shape
print "y_sample = ",y_sample.shape


Xmax = np.amax(x_sample, axis=0)
Xmin = np.amin(x_sample, axis=0)
x_sample = (x_sample - Xmin) / (Xmax - Xmin)

ymax = np.amax(y_sample, axis=0)
ymin = np.amin(y_sample, axis=0)
y_sample = (y_sample - ymin) / (ymax - ymin)

print x_sample
print "-"*80
print y_sample

parameters = {"ymax":ymax,"ymin":ymin,"Xmax":Xmax,"Xmin":Xmin}

train_start = 600
train_end = 1150
y_train = y_sample[train_start:train_end,:]
x_train = x_sample[train_start:train_end,:]
sample_number = x_train.shape[0]

nn_input = 5
layer_one = 100
layer_two = 50
nn_output = 1
learn_rate = 0.001

erro_rate = 0.0128
# 初始化权重
def init_weight(shape,name = None):
    return tf.Variable(tf.random_normal(shape, stddev=learn_rate),name = name)

def init_bias(shape,name = None):
    init = tf.zeros(shape)
    return tf.Variable(init, name=name)


def model(X, W, B):
    m = tf.matmul(X, W) + B
    # RELU for instead sigmoid, Sigmoid only for Final
    L = tf.nn.tanh(m)
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


cost = tf.reduce_mean(tf.square((Y - y_out)))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_out, Y)) 
# train_op = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
pridict_op = tf.nn.relu(tf.matmul(L3, W3) + B3)

'''
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

print "running ..."
sess.run(train_op, feed_dict={X: x_train, Y: y_train})
y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
erro_pridict = y_train - y_pridict
old_erro = np.abs(erro_pridict).mean()


run_times = 0
while(old_erro > erro_rate):

    try:
        sess.run(train_op, feed_dict={X:x_train, Y:y_train})
    except Exception,ex:
        print "[WARMING]exception happens when run train model"
        continue
    y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
    erro_pridict = y_train - y_pridict
    old_erro = np.abs(erro_pridict).mean()

    if run_times % 300 == 0:
        print "old_erro = ",old_erro
    run_times = run_times + 1
    print "run_times = %d ,old_erro = %f" % (run_times,old_erro)


print "I have trianed %d times !!!!" % (run_times)

import os 

outdir = './npyfile/'
if not os.path.exists(outdir):
    os.mkdir(outdir)
np.save(outdir + 'y_train',y_train)
np.save(outdir + 'y_train_predict',y_pridict)

y_test_pridict = sess.run(pridict_op,feed_dict = {X:x_test})
np.save(outdir + 'y_test',y_test)
np.save(outdir + 'y_test_pridict',y_test_pridict)

save_path = './model_saver/OnlyCloseNNModel.ckpt'
saver = tf.train.Saver()
saver.save(sess,save_path)

'''
save_path = './model_saver/OnlyCloseNNModel.ckpt'
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()

saver.restore(sess,save_path)
print "tensorflow session restore successfully ----------------------"

outdir = './npyfile/'


import os
outdir = './npyfile/'
imagedir = './trainModelploter/'
if not os.path.exists(imagedir):
    os.mkdir(imagedir)

import matplotlib.pylab as plt

y_train = np.load(outdir + 'y_train.npy')
y_train_predict = np.load(outdir + 'y_train_predict.npy')

# plt.plot(nn4_y_train, 'ro')
# plt.plot(nn4_y_train_predict, 'bo')
plt.plot(y_train, 'r-')
plt.plot(y_train_predict, 'b-')
plt.xlabel('index')
plt.ylabel('value')
plt.title('nn4 Only Close Normalized Training Predict')
plt.legend(['y_train','y_train_predict'])
plt.grid(True)
plt.savefig(imagedir + 'training.jpg')
plt.show()


test_start = 1150
test_end = 1200
y_test = y_sample[test_start:test_end,:]
x_test = x_sample[test_start:test_end,:]

y_test_predict = sess.run(pridict_op,feed_dict = {X:x_test})

plt.plot(y_test, 'ro')
plt.plot(y_test_predict,'bo')
plt.plot(y_test, 'r-')
plt.plot(y_test_predict,'b-')
plt.xlabel('index')
plt.ylabel('value')
plt.title('nn4 Only Close Normalized Test Predict')
plt.legend(['y_test','y_test_predict'])
plt.grid(True)
plt.savefig(imagedir + 'valiation.jpg')
plt.show()
