#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os 
import pickle
import cv2

class MyNeurolNetworkModel(object):

    def __init__(self,):
        super(MyNeurolNetworkModel,self).__init__()
        self.inputNumber  = 5
        self.layerOne     = 100
        self.layerTwo     = 50
        self.outputNumber = 1
        self.learningRate = 1e-4
        self.errorRate    = 0.0128
        self.parameter = {}
        self.x_train = None
        self.y_train = None
        self.outdir = os.path.join(os.path.dirname(__file__),'modelSaver/')
        self.savePath      = self.outdir + 'MyNeurolNetworkModel.ckpt'
        self.parameterPath = self.outdir + 'MyNeurolNetworkParameter.txt'

        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
       

    def normalize(self,x_train,y_train):
    	xmax = np.amax(x_train, axis=0)
        xmin = np.amin(x_train, axis=0)
        x_train = (x_train - xmin) / (xmax - xmin)
        self.x_train = x_train
        parameterX = {"xmax":xmax,"xmin":xmin,"x_train":x_train}
    
        ymax = np.amax(y_train, axis=0)
        ymin = np.amin(y_train, axis=0)
        y_train = (y_train - ymin) / (ymax - ymin)
        self.y_train = y_train
        parameterY = {"ymin":ymin,"ymax":ymax,"y_train":y_train}

        parameter = {"parameterX":parameterX,"parameterY":parameterY}
        f = open(self.parameterPath,'wb')
        pickle.dump(parameter,f)



    def normalize_xtest(self,x_test):
        # parameter = {"parameterX":parameterX,"parameterY":parameterY}
        f = open(self.parameterPath,'rb')
        parameter = pickle.load(f)

        xmax = parameter['parameterX']["xmax"]
        xmin = parameter['parameterX']["xmin"]
        x_test = (x_test - xmin) / (xmax - xmin)
        return x_test


    def denormalize_ypredict(self,y_predict):
        f = open(self.parameterPath,'rb')
        parameter = pickle.load(f)

    	ymin = parameter['parameterY']['ymin']
    	ymax = parameter['parameterY']["ymax"]
    	y_predict = y_predict*(ymax - ymin) + ymin
        return y_predict


    def init_weight(self,shape,name = None):
    	init = tf.random_normal(shape, stddev = 0.1)
        return tf.Variable(init,name = name)

    def init_bias(self,shape,name = None):
        init = tf.zeros(shape)
        return tf.Variable(init, name=name)

    def model(self,X, W, B):
        m = tf.matmul(X, W) + B
        L = tf.nn.tanh(m)
        return L

    def train(self,x_train,y_train):
        self.normalize(x_train,y_train)
        x_train = self.x_train
        y_train = self.y_train

        X = tf.placeholder("float", [None, self.inputNumber])
        Y = tf.placeholder("float", [None, self.outputNumber])

        W1 = self.init_weight([self.inputNumber, self.layerOne], 'W1')
        B1 = self.init_bias([self.layerOne], 'B1')

        W2 = self.init_weight([self.layerOne, self.layerTwo], 'W2')
        B2 = self.init_bias([self.layerTwo], 'B2')

        W3 = self.init_weight([self.layerTwo,self.outputNumber], 'W3')
        B3 = self.init_bias([self.outputNumber], 'B3')
  
        L2 = self.model(X,  W1, B1)
        L3 = self.model(L2, W2, B2)

        y_out = tf.nn.relu(tf.matmul(L3, W3) + B3)
        cost = tf.reduce_mean(tf.square((Y - y_out)))
        train_op = tf.train.AdamOptimizer(self.learningRate).minimize(cost)
        pridict_op = tf.nn.relu(tf.matmul(L3, W3) + B3)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        sess.run(train_op, feed_dict={X: x_train, Y: y_train})
        y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
        erro_pridict = y_train - y_pridict
        error = np.abs(erro_pridict).mean()

        run_times = 0
        while(error > self.errorRate):
            try:
                sess.run(train_op, feed_dict={X:x_train, Y:y_train})
            except Exception,ex:
                print "[WARMING]exception happens when run train model"
                continue
            y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
            erro_pridict = y_train - y_pridict
            error = np.abs(erro_pridict).mean()

            if run_times % 300 == 0:
                print "run_times = %d error = %f " % (run_times,error)
            run_times = run_times + 1
        print "I have trianed %d times !!!!" % (run_times)

        saver = tf.train.Saver()
        saver.save(sess,self.savePath)
        sess.close()


        # saver = tf.train.Saver(tf.all_variables())
        # saver.restore(sess,self.savePath)

        # y_predict = sess.run(pridict_op, feed_dict={X:x_test})
        # return self.denormalize_ypredict(y_predict)




    def predict(self,x_test):
        x_test = self.normalize_xtest(x_test)

        X = tf.placeholder("float", [None, self.inputNumber])
        Y = tf.placeholder("float", [None, self.outputNumber])

        W1 = self.init_weight([self.inputNumber, self.layerOne], 'W1')
        B1 = self.init_bias([self.layerOne], 'B1')

        W2 = self.init_weight([self.layerOne, self.layerTwo], 'W2')
        B2 = self.init_bias([self.layerTwo], 'B2')

        W3 = self.init_weight([self.layerTwo,self.outputNumber], 'W3')
        B3 = self.init_bias([self.outputNumber], 'B3')
  
        L2 = self.model(X,  W1, B1)
        L3 = self.model(L2, W2, B2)

        y_out = tf.nn.relu(tf.matmul(L3, W3) + B3)
        cost = tf.reduce_mean(tf.square((Y - y_out)))
        train_op = tf.train.AdamOptimizer(self.learningRate).minimize(cost)
        pridict_op = tf.nn.relu(tf.matmul(L3, W3) + B3)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess,self.savePath)

        y_predict = sess.run(pridict_op, feed_dict={X:x_test})
        return self.denormalize_ypredict(y_predict)


    def knn_predict(self,x_test):
        pass

        


# myNNmodel = MyNeurolNetworkModel()

if __name__ == '__main__':
    import pandas as pd
    myNNmodel = MyNeurolNetworkModel()
    myNNmodel.errorRate = 0.045
    basedir = os.getcwd()
    filename = os.path.join(basedir, 'stock_fanny.xlsx')
    xls = pd.ExcelFile(filename)

    df_train = xls.parse('Sheet4', index_col='Date') # train

    close = df_train['Close']

    print "Close = ",len(close)
    # print close
    
    closeArray = np.array([close]).reshape(len(close),1)

    print "closeArray = ",closeArray.shape
    print closeArray

    print "========================================================="

    shape = closeArray.shape
    related = 5

    x_sample = np.zeros((shape[0]-related,5))
    y_sample = np.zeros((shape[0]-related,1))

    for i in xrange(shape[0] - related):
        x_sample[i,:] = closeArray[i:i+related,0].reshape(1,related)
        y_sample[i,0] = closeArray[i+related,0]

    print "x_sample.shape = ",x_sample.shape
    print x_sample
    print "-------------------------------------------------------------------"
    print "y_sample.shape = ",y_sample.shape
    print y_sample

    train_start = 600
    train_end = 1150
    y_train = y_sample[train_start:train_end,:]
    x_train = x_sample[train_start:train_end,:]

    test_start = 1000
    test_end = 1200
    y_test = y_sample[test_start:test_end,:]
    x_test = x_sample[test_start:test_end,:]

    test_start_1 = 400
    test_end_1 = 600
    x_test_1 = x_sample[test_start:test_end,:]
    y_test_1 = y_sample[test_start:test_end,:]

    print "training model ..."
    # myNNmodel.train(x_train,y_train)
    # print "train myNNmodel successfully -------------------------------"

    y_test_predict = myNNmodel.predict(x_test)
    # y_test_predict = myNNmodel.train_predict(x_test)
    print "------------------------------ok---------------------------------"


    from matplotlib import pyplot as plt

    plt.plot(y_test,'ro')
    plt.plot(y_test_predict,'bo')
    plt.plot(y_test,'r-')
    plt.plot(y_test_predict,'b-')
    plt.legend(['y_test','y_test_predict'])
    plt.grid(True)
    plt.show()

    # myNNmodel1 = MyNeurolNetworkModel()
    # myNNmodel1.errorRate = 0.045

    y_test_1_predict = myNNmodel.predict(x_test_1)

    plt.plot(y_test_1,'r-')
    plt.plot(y_test_1_predict,'b-')
    plt.legend(['y_test_1','y_test_1_predict'])
    plt.grid(True)
    plt.show()
