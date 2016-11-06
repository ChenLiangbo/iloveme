#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

'''
使用前几天五列数据来预测后一天的五个数值，一部分训练，一部分预测，先生成样本，然后选择训练集和测试
'''

yahooData = np.load('yahooData.npy')
print "yahooData.shape = ",yahooData.shape
'''yahoo data:adj_close,high,low,close,open'''
related_number = 5

yahooData_length = yahooData.shape[0]   #总共数据

'''kdj_array : k,d,j'''
kdj_array = np.zeros((yahooData_length,3),dtype = np.float32)

for hang in xrange(yahooData_length):
    if hang < related_number:
        for lie in range(3):
            kdj_array[hang,lie] = 0
    else:
        RSV = (yahooData[hang,3] - yahooData[hang-related_number:hang,2].min()) / (yahooData[hang-related_number:hang,1].max()-yahooData[hang-related_number:hang,2].min() ) *100

        if hang == related_number:
            kdj_array[hang-1,0] = 50
            kdj_array[hang-1,1]= 50
        kdj_array[hang,0] = 2*kdj_array[hang-1,0]/3 + RSV/3                # Kn = 2*Kn_1/3 + RSV/3
        kdj_array[hang,1] = 2*kdj_array[hang-1,1]/3 + kdj_array[hang,0]/8  # Dn = 2*Dn_1/3 + Kn/3
        kdj_array[hang,2] = 3*kdj_array[hang,1] - 2*kdj_array[hang,0]      # Jn = 3*Dn - 2*Kn


print "lowest = ",yahooData[hang-related_number:hang,2].min()

y_sample = yahooData[related_number*2:,2:4]

#y_sample = np.asarray(yahooData[related_number*2:,3]).astype(np.float32)

yahooData = np.hstack([yahooData,kdj_array])
'''yahooData.shape = (2080,8)'''
print "yahooData.shape -------hstack = ",yahooData.shape

yahooData_length = yahooData.shape[0]   #总共数据
yahooData_colums = yahooData.shape[1]   #样本的维数　５维


x_sample = np.zeros((yahooData_length - related_number,yahooData.shape[1]*related_number))
for i in xrange(yahooData_length - related_number):
    if i + related_number < yahooData_length:
        x_sample[i,:] = yahooData[i:i + related_number,:].reshape(1,yahooData.shape[1]*related_number)

    else:
        break



#y_sample = yahooData[related_number:,:]

x_sample = x_sample[related_number:,:]
#y_sample = y_sample[related_number:,:]
output_number = 2
print "x_sample.shape = ",x_sample.shape
print "y_sample.shape = ",y_sample.shape
# print "-------------------------------------------------"
# print "y_sample - yahooData = "
# print y_sample[0:related_number,:] - yahooData[related_number:related_number+related_number,:]
# print "--------------------------------------------------"



train_start = 0
train_end = 1500
x_train = x_sample[train_start:train_end,:]
y_train = y_sample[train_start:train_end,:]
print "x_train = ",x_train.shape
print "y_train = ",y_train.shape
train_length = x_train.shape[0]

test_start = 1500
test_end = 2000
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


W1 = init_weight([yahooData_colums*related_number, 1024], 'W1')
B1 = init_bias([1024], 'B1')


W2 = init_weight([1024, 256], 'W2')
B2 = init_bias([256], 'B2')


W3 = init_weight([256, output_number], 'W3')
B3 = init_bias([output_number], 'B3')
# -------------------------------------
L2 = model(X,  W1, B1)
L3 = model(L2, W2, B2)


y_out = tf.nn.relu(tf.matmul(L3, W3) + B3)


cost = tf.reduce_mean(tf.square((Y - y_out)))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
pridict_op = tf.nn.relu(tf.matmul(L3, W3) + B3)


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

print "running ..."
sess.run(train_op, feed_dict={X: x_train, Y: y_train})
y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
erro_pridict = y_train - y_pridict
old_erro = np.abs(erro_pridict).mean()
'''
erro_rate = 0.5

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

np.save(outdir + "y_test_caogaozhi",y_test)
np.save(outdir + 'y_pridict_caogaozhi',y_test_pridict)


outdir = './caogao/'
y_test = np.load(outdir + "y_test_caogaozhi.npy")
y_test_pridict = np.load(outdir + 'y_pridict_caogaozhi.npy')
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
