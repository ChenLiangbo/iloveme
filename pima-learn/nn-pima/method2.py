#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

xsample = np.load('./dataset/xsample.npy')
ysample = np.load('./dataset/ysample.npy')
print "xsample = ",xsample.shape
print "ysample = ",ysample.shape
shape = xsample.shape
# indexList = np.random.permutation(shape[0])
indexList = range(shape[0])

x_train = xsample[indexList[0:538]]
y_train = ysample[indexList[0:538]]
print "x_train.shape = ",x_train.shape
print "y_train.shape = ",y_train.shape

x_test = xsample[indexList[538:]]
y_test = ysample[indexList[538:]]
print "x_test.shape = ",x_test.shape
print "y_test.shape = ",y_test.shape

y = np.zeros((y_test.shape[0],))
for i in xrange(y_test.shape[0]):
    if y_test[i,0] == 1:
        y[i] = 0
    else:
        y[i] = 1

def f_measure(y_predict,y_test):
    shape = y_predict.shape
    TP,FP,FN,TN = 0,0,0,0
    for i in xrange(shape[0]):
        if int(y_predict[i]) == 1:
            if int(y_test[i]) == 1:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if int(y_test[i]) == 1:
                FP = FP + 1
            else:
                TN = TN + 1
    # print "TP = %d,TN = %d,FP = %d,FN = %d " % (TP,TN,FP,FN)
    result = {}
    result['accuracy']  = round(float(TP + TN)/(TP + FP + FN + TN),4)
    result["precision"] = round(float(TP)/(TP + FP),4)
    result["recall"]    = round(float(TP)/(TP + FN),4)
    result["fmeasure"]  = 2*result["precision"]*result["recall"]/(result["precision"] + result["recall"])
    result["fmeasure"] = round(result["fmeasure"],4)
    return result


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

layerOne = 7
layerTwo = 3
learnRate = 0.001
# 定义占位符+ 初始化变量
X = tf.placeholder("float", [None, 8])
Y = tf.placeholder("float", [None, 2])

w_h = init_weights([8, layerOne])
w_h2 = init_weights([layerOne, layerTwo])
w_o = init_weights([layerTwo, 2])

# dropout 的概率
p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# 模型
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

# 损失函数

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(learnRate, 0.1).minimize(cost)
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(100):
    sess.run(train_op, feed_dict={X: x_train, Y: y_train,
                                  p_keep_input: 0.8, p_keep_hidden: 0.5})
    y_predict = sess.run(predict_op, feed_dict={X: x_test, Y: y_test,
                                                     p_keep_input: 1.0,
                                                     p_keep_hidden: 1.0})
    try:
        print "i = ",i
        print "result = ",f_measure(y_predict,y)
    except:
        continue
                     

