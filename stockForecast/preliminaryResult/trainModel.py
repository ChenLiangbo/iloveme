#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from itertools import izip

# outdir = './dataset/'
outdir = './finalfile/'
x_sample = np.load(outdir + 'x_train_normalized.npy')
y_sample = np.load(outdir + 'y_train_normalized.npy')
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


nn_input = 10
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
    start = np.random.randint(y_train.shape[0])
    end = np.random.randint(y_train.shape[0])
    if end < start:
        start,end = end,start

    x_batch = x_train[start:end,:]
    y_batch = y_train[start:end,:]
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

    print "run_times = %d ,old_erro = %f,start = %d, end = %d" % (run_times,old_erro,start,end)
import os 
outdir = './npyfile/'
if not os.path.exists(outdir):
    os.mkdir(outdir)
np.save(outdir + 'nn4_y_train',y_train)
np.save(outdir + 'nn4_y_train_predict',y_pridict)

print "I have trianed %d times !!!!" % (run_times)

save_path = './model_saver/Arimann4Model.ckpt'
saver = tf.train.Saver()
saver.save(sess,save_path)


y_test_pridict = sess.run(pridict_op,feed_dict = {X:x_test})

erro_pridict = y_test - y_test_pridict

np.save(outdir + 'y_test_close',y_test)
np.save(outdir + 'y_test_pridict_close',y_test_pridict)

parameters = {
              "W1":sess.run(W1),"B1":sess.run(B1),
              "W2":sess.run(W2),"B2":sess.run(B2),
              "W3":sess.run(W3),"B3":sess.run(B3)
              }


sess.close()
import pickle

f = open('./model_saver/nn4Model.txt','wb')
pickle.dump(parameters,f)
'''
save_path = './finalfile' + '/Arimann4Model.ckpt'
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()

saver.restore(sess,save_path)
y_test_pridict = sess.run(pridict_op,feed_dict = {X:x_test})

print "session restore ok"




import os
outdir = './npyfile/'
imagedir = './trainModelploter/'
if not os.path.exists(imagedir):
    os.mkdir(imagedir)

# import matplotlib.pylab as plt

# nn4_y_train = np.load(outdir + 'nn4_y_train.npy')
# nn4_y_train_predict = np.load(outdir + 'nn4_y_train_predict.npy')

# # plt.plot(nn4_y_train, 'ro')
# # plt.plot(nn4_y_train_predict, 'bo')
# plt.plot(nn4_y_train, 'r-')
# plt.plot(nn4_y_train_predict, 'b-')
# plt.xlabel('index')
# plt.ylabel('value')
# plt.title('arima-nn training predict')
# plt.legend(['nn4_y_train','nn4_y_train_predict'])
# plt.grid(True)
# plt.savefig(imagedir + 'nn4_training.jpg')
# plt.show()


# y_test = np.load('./npyfile/y_test1.npy')
# y_test_predict = np.load('./npyfile/y_test_predict1.npy')

# # plt.plot(y_test, 'ro')
# # plt.plot(y_test_predict,'bo')
# plt.plot(y_test, 'r-')
# plt.plot(y_test_predict,'b-')
# plt.xlabel('index')
# plt.ylabel('value')
# plt.title('arima-nn valiation predict')
# plt.legend(['y_test','y_test_predict'])
# plt.grid(True)
# plt.savefig(imagedir + '/nn4_valiation.jpg')
# plt.show()


x_test_1 = x_sample[400:600,:]
y_test_1 = y_sample[400:600,:]

y_test_1_predict = sess.run(pridict_op,feed_dict = {X:x_test_1})
from matplotlib import pyplot as plt
plt.plot(y_test_1, 'r-')
plt.plot(y_test_1_predict,'b-')
plt.xlabel('index')
plt.ylabel('value')
plt.title('arima-nn y_test_1 predict')
plt.legend(['y_test_1','y_test_1_predict'])
plt.grid(True)
plt.savefig(imagedir + '/y_test_1_predict.jpg')
plt.show()