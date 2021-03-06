#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os 
import pickle
import cv2

class MyNeurolNetworkModel(object):
    '''
    此类使用神经网络对股票数据经行预测，训练之后将模型以文件名的形式保存，之后使用的时候加载模型
    训练好模型训练和预测函数在一次程序运行中之可以调用一次，不然会有莫名的错误,训练和预测的时候参数需一致
    对于传入的数据训练和预测的时候对输入有要求，必须是Open,High,Low,Close,Volume*e-6的顺序
    训练和预测的时候可以加入K,D,J数值以及close的一阶差分，默认都加入，凡是训练模型时加入了，则预测的时候也需要加入
    '''
    def __init__(self,):
        super(MyNeurolNetworkModel,self).__init__()
        self.inputNumber  = 5        #层入曾数目
        self.layerOne     = 100      #第一隐含层数目
        self.layerTwo     = 30       #第二隐含层数目
        self.outputNumber = 1        #输出曾数目
        self.learningRate = 1e-4     #学习率
        self.errorRate    = 0.0128   #训练网络需要达到的最小均方差值
        self.keep_prob    = 0.5      #弃权的参数
        self.trainTimes   = 20000    #训练神经网络需要达到的最大训练次数
        self.parameter = {}          #归一化时候的变量以及训练样本存储
        self.x_train = None          #训练样本输入 Open,High,Low,Close,Volume*e-6顺序
        self.y_train = None          #训练样本输出　Close
        self.kdjPeriod   = 5         #计算k,d,j值的周期
        self.outdir        = os.path.join(os.path.dirname(__file__),'modelSaver/')             #模型存储路径
        self.savePath      = self.outdir + 'MyNeurolNetworkModel.ckpt'        #模型稳健
        self.parameterPath = self.outdir + 'MyNeurolNetworkParameter.txt'     #参数文件

        if len(self.outdir) >0 and (not os.path.exists(self.outdir)):
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

    def random_vector(self,length,limit):
        vector = []
        for i in xrange(length):
            vector.append(np.random.randint(limit))
        return np.asarray(vector)

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
  
        # L2 = self.model(X,  W1, B1)

        # keep_prob = tf.placeholder("float")
        # L2_drop = tf.nn.dropout(L2, keep_prob)

        # L3 = self.model(L2_drop, W2, B2)

        L2 = self.model(X,  W1, B1)
        L3 = self.model(L2, W2, B2)

        y_out = tf.nn.relu(tf.matmul(L3, W3) + B3)
        cost = tf.reduce_mean(tf.square((Y - y_out)))
        train_op = tf.train.AdamOptimizer(self.learningRate).minimize(cost)
        pridict_op = tf.nn.relu(tf.matmul(L3, W3) + B3)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        # sess.run(train_op, feed_dict={X: x_train, Y: y_train,keep_prob:1.0})
        # y_pridict = sess.run(pridict_op,feed_dict = {X:x_train,keep_prob:0.5})
        sess.run(train_op, feed_dict={X: x_train, Y: y_train})
        y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})

        erro_pridict = y_train - y_pridict
        error = np.abs(erro_pridict).mean()

        train_length = x_train.shape[0]/10

        run_times = 0
        while(error > self.errorRate):
            # vector = self.random_vector(x_train.shape[0]/10,x_train.shape[0])
            # x_batch = x_train[vector]
            # y_batch = y_train[vector]
            try:
                sess.run(train_op, feed_dict={X:x_train, Y:y_train})
                # sess.run(train_op, feed_dict={X:x_train, Y:y_train,keep_prob:1.0})
            except Exception,ex:
                print "[WARMING]exception happens when run train model",str(ex)
                continue
            y_pridict = sess.run(pridict_op,feed_dict = {X:x_train})
            # y_pridict = sess.run(pridict_op,feed_dict = {X:x_train,keep_prob:0.5})
            erro_pridict = y_train - y_pridict
            error = np.abs(erro_pridict).mean()

            if run_times % 300 == 0:
                print "run_times = %d error = %f " % (run_times,error)
            if run_times > self.trainTimes:
                break
            run_times = run_times + 1
        print "I have trianed %d times !!!!" % (run_times)

        saver = tf.train.Saver()
        saver.save(sess,self.savePath)
        sess.close()


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

        # keep_prob = tf.placeholder("float")
        # L2_drop = tf.nn.dropout(L2, keep_prob)
        # L3 = self.model(L2_drop, W2, B2)
 
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

        # y_predict = sess.run(pridict_op, feed_dict={X:x_test,keep_prob:0.5})
        y_predict = sess.run(pridict_op, feed_dict={X:x_test})
        return self.denormalize_ypredict(y_predict)

    # input : x_train,order Open,High,Low,Close,Volume*e-6 five columns
    # output : numpy array [K,D,J]
    def calculate_kdj(self,x_train):
        shape = x_train.shape
        kdj = np.zeros((shape[0],3))
        Kn_1,Dn_1 = 50,50
        for i in xrange(self.kdjPeriod,shape[0]):
            rsv = 100*(x_train[i,3]-min(x_train[i-self.kdjPeriod:i,2]))/(max(x_train[i-self.kdjPeriod:i,1]) - min(x_train[i-self.kdjPeriod:i,2]))
            Kn = 2*Kn_1/3 + rsv/3
            Dn = 2*Dn_1/3 + Kn/3
            Jn = 3*Kn - 2*Dn
            kdj[i] = np.array([Kn,Dn,Jn])
            Kn_1,Dn_1 = Kn,Dn
        kmax,kmin = np.max(kdj[:,0]),np.min(kdj[self.kdjPeriod:,0])
        dmax,dmin = np.max(kdj[:,1]),np.min(kdj[self.kdjPeriod:,1])
        jmax,jmin = np.max(kdj[:,2]),np.min(kdj[self.kdjPeriod:,2])
        kdj[0:self.kdjPeriod,0] = ((kmax-kmin)*np.random.rand(self.kdjPeriod,1) + kmin).ravel()
        kdj[0:self.kdjPeriod,1] = ((dmax-dmin)*np.random.rand(self.kdjPeriod,1) + dmin).ravel()
        kdj[0:self.kdjPeriod,2] = ((jmax-jmin)*np.random.rand(self.kdjPeriod,1) + jmin).ravel()
        return kdj

    #input:numpy array shape=[None,1]
    #output:numpy array shape = [None,4],[dclose,eclose,d2close,e2close]
    def calculate_dclose(self,closeArray):
        shape = closeArray.shape
        dclose = np.zeros((shape[0],1))
        for i in xrange(1,shape[0]):
            dclose[i] = closeArray[i,0] - closeArray[i-1,0]
        dmax = np.max(dclose[1:,0])
        dmin = np.max(dclose[1:,0])
        dclose[0,0] = (dmax-dmin)*np.random.random(1) + dmin
        eclose = np.exp(dclose)
        d2close = np.zeros((shape[0],1))
        for i in xrange(1,shape[0]):
            d2close[i,0] = dclose[i,0] - dclose[i-1,0]
        d2max = np.max(d2close[1:,:])
        d2min = np.min(d2close[1:,:])
        d2close[0,0] = (d2max - d2min)*np.random.random(1) + d2min
        e2close = np.exp(d2close)
        declose =  np.hstack([dclose,eclose,d2close,e2close])
        for j in xrange(declose.shape[1]):
            declose[:,j] = declose[:,j]/np.mean(declose[:,j])
        return declose

    #input : close price,numpy shape = [None,1]
    #output : logfit,numpy shape = [None,1]
    def calculate_logfit(self,closeArray):
        shape = closeArray.shape
        logFitrate = np.zeros(shape)
        for i in xrange(1,shape[0]):
            logFitrate[i,0] = np.log(closeArray[i,0]/closeArray[i-1,0])
        logFitrate[0,0] = logFitrate[1,0]
        return logFitrate - np.mean(logFitrate)
    
    #input  :array
    #output : reverse in y function
    def reverse_array(self,array):
        shape = array.shape
        outArray = np.zeros(shape)
        if len(shape) < 2:
            for i in xrange(shape[0]):
                outArray[i] = array[shape[0]-1-i]
            return outArray
    
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                outArray[i,j] = array[shape[0]-1-i,j]
        return outArray

    #input :close numpy array,shape = [None,1],0<exponent<1,length = int
    #output:exponent blur,numpy array shape = [None,1]
    def calculate_exponent(self,closeArray,exponent = 0.9,length = 31):
        shape = closeArray.shape
        x = range(0,length)
        x = np.array(x).reshape(length,1)
        y = np.zeros(x.shape)
        for i in xrange(x.shape[0]):
            y[i,0] = np.power(exponent,x[i,0])
        # y = y/np.sum(y)

        retExponent = np.zeros((shape[0],1))
        for i in xrange(shape[0]):
            if i < length:
                related = closeArray[0:i+1,:]
                related = self.reverse_array(related)
                weights = y[0:i+1,:]
                # weights =s weights/np.sum(weights)
                retExponent[i,0] = np.sum(related*weights)
            else:
                start = i + 1 - length
                end = i + 1
                related = closeArray[start:end,:]
                related = self.reverse_array(related)
                retExponent[i,0] = np.sum(related*weights)

        return retExponent



# myNNmodel = MyNeurolNetworkModel()

if __name__ == '__main__':
    import numpy as np
    import os

    '''
    每天的五个数据 high,low,close,open,adj_close,
    '''
    order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume*e-6"}
    yahooData = np.load('yahoo_finance5.npy')
    Open,High,Low,Close,Volume = np.hsplit(yahooData,5)

    shape = yahooData.shape
    print "shape = ",shape
    print "-"*80
    
    print "Close shape = ",Close.shape
    
    myNNmodel = MyNeurolNetworkModel()
    kdj = myNNmodel.calculate_kdj(yahooData)    #(None,3)
    print "kdj = ",kdj.shape
    
    declose = myNNmodel.calculate_dclose(Close)  #(None,4)
    print "declose = ",declose.shape

    logfit = myNNmodel.calculate_logfit(Close)   #(None,1)
    print "logfit = ",logfit.shape
    
    closeExponent = myNNmodel.calculate_exponent(Close,exponent = 0.87) #(None,1)
    print "closeExponent = ",closeExponent.shape


    from sklearn.decomposition import PCA
    pca = PCA(n_components = 2)
    newData = pca.fit_transform(np.hstack([Open,High,Low,Volume]))  #(None,2)
    print "newData = ",newData.shape
    
    x_sample = np.hstack([newData,declose,kdj,logfit,closeExponent])
    y_sample = Close[1:]
    
    x_train = np.vstack([x_sample[0:100,:],x_sample[300:500,:],x_sample[700:900,:],x_sample[1100:1150,:]])
    y_train = np.vstack([y_sample[0:100,:],y_sample[300:500,:],y_sample[700:900,:],y_sample[1100:1150,:]])

    print "x_train.shape = ",x_train.shape
    sample_number = x_train.shape[0]
    
    test_start = 1150
    test_end = 1200
    y_test = y_sample[test_start:test_end,:]
    x_test = x_sample[test_start:test_end,:]
    
    outdir = os.path.join(os.path.dirname(__file__),'images/')
    if len(outdir) >0 and (not os.path.exists(outdir)):
        os.mkdir(self.outdir)
    

    myNNmodel.inputNumber = 11
    myNNmodel.errorRate = 0.0105
    myNNmodel.learningRate = 0.001
    
    myNNmodel.train(x_train,y_train)

    print "myNNmodel train successfully ..."
    
    y_test_predict = myNNmodel.predict(x_test)
    from matplotlib import pyplot as plt
        
    plt.plot(y_test,'ro')
    plt.plot(y_test_predict,'bo')
    plt.plot(y_test,'r-')
    plt.plot(y_test_predict,'b-')
    plt.legend(['y_test','y_test_predict'])
    plt.grid(True)
    plt.xlabel('index')
    plt.ylabel('value')
    plt.title('MyNeurolNetworkModel Predict Close With AllOneDay')
    plt.savefig(outdir + 'close.jpg')
    plt.show()
    
    
    acuracy = (y_test_predict - y_test)/y_test
    plt.plot(acuracy,'ro')
    plt.plot(acuracy,'r-')
    plt.legend('acuracy')
    plt.title('MyNeurolNetworkModel Test Acuracy')
    plt.xlabel('index')
    plt.ylabel(['acuracy'])
    plt.grid(True)
    plt.savefig(outdir + 'acurracy')
    plt.show()
    print "mean acuracy = ",np.mean(np.abs(acuracy))    