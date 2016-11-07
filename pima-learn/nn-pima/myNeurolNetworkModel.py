#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os 
import pickle

class MyNeurolNetworkModel(object):
    '''
    此类使用神经网络对股票数据经行预测，训练之后将模型以文件名的形式保存，之后使用的时候加载模型
    训练好模型训练和预测函数在一次程序运行中之可以调用一次，不然会有莫名的错误,训练和预测的时候参数需一致
    对于传入的数据训练和预测的时候对输入有要求，必须是Open,High,Low,Close,Volume*e-6的顺序
    训练和预测的时候可以加入K,D,J数值以及close的一阶差分，默认都加入，凡是训练模型时加入了，则预测的时候也需要加入
    '''
    def __init__(self,):
        super(MyNeurolNetworkModel,self).__init__()
        self.inputNumber  = 8        #层入曾数目
        self.layerOne     = 15       #第一隐含层数目
        self.outputNumber = 1        #输出曾数目
        self.learningRate = 1e-3     #学习率
        self.accuracy     = 0.9128   #训练网络需要达到的最小均方差值
        self.isDroupout   = False
        self.keep_prob    = 0.5      #弃权的参数
        self.trainTimes   = 20000    #训练神经网络需要达到的最大训练次数
        self.batchSize    = 20
        self.parameter = {}          #归一化时候的变量以及训练样本存储
        self.x_train = None          #训练样本输入 Open,High,Low,Close,Volume*e-6顺序
        self.y_train = None          #训练样本输出　Close
        self.kdjPeriod   = 5         #计算k,d,j值的周期
        self.outdir        = os.path.join(os.path.dirname(__file__),'NNmodel/')             #模型存储路径
        self.savePath      = self.outdir + 'MyNeurolNetworkModel.ckpt'        #模型稳健
        self.parameterPath = self.outdir + 'MyNeurolNetworkParameter.txt'     #参数文件

        if len(self.outdir) >0 and (not os.path.exists(self.outdir)):
            os.mkdir(self.outdir)
       

    def normalize(self,x_train):
    	xmax = np.amax(x_train, axis=0)
        xmin = np.amin(x_train, axis=0)
        x_train = (x_train - xmin) / (xmax - xmin)
        self.x_train = x_train
        parameterX = {"xmax":xmax,"xmin":xmin,"x_train":x_train}
        f = open(self.parameterPath,'wb')
        pickle.dump(parameterX,f)
        f.close()
        return x_train

    def normalize_xtest(self,x_test):
        # parameter = {"parameterX":parameterX,"parameterY":parameterY}
        f = open(self.parameterPath,'rb')
        parameter = pickle.load(f)
        xmax = parameter["xmax"]
        xmin = parameter["xmin"]
        x_test = (x_test - xmin) / (xmax - xmin)
        f.close()
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

    def train(self,x_train,y_train,x_test,y_test):
        x_train = self.normalize(x_train)
        shape = x_train.shape
        X = tf.placeholder("float", [None, self.inputNumber])
        Y = tf.placeholder("float", [None, self.outputNumber])

        W1 = self.init_weight([self.inputNumber, self.layerOne], 'W1')
        B1 = self.init_bias([self.layerOne], 'B1')

        W2 = self.init_weight([self.layerOne, self.outputNumber], 'W2')
        B2 = self.init_bias([self.outputNumber], 'B2')

        L2 = self.model(X,  W1, B1)

        if self.isDroupout: 
            L2 = h = tf.nn.dropout(L2, keep_prob)
            y_out = tf.nn.relu(tf.matmul(L2, W2) + B2)
            cost = tf.reduce_mean(tf.square((Y - y_out)))
            train_op = tf.train.AdamOptimizer(self.learningRate).minimize(cost)
            pridict_op = tf.nn.relu(tf.matmul(L2, W2) + B2)
        else:
            y_out = tf.nn.relu(tf.matmul(L2, W2) + B2)
            cost = tf.reduce_mean(tf.square((Y - y_out)))
            train_op = tf.train.AdamOptimizer(self.learningRate).minimize(cost)
            pridict_op = tf.nn.relu(tf.matmul(L2, W2) + B2)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        trainList    = []
        testList     = []
        timesList    = []
        if self.isDroupout:
            sess.run(train_op, feed_dict={X: x_train, Y: y_train,keep_prob:self.keep_prob})
            y_predict = sess.run(pridict_op,feed_dict = {X:x_train,keep_prob:2*self.keep_prob})
            y_predict = self.get_ypredict(y_predict)
            accuracy = np.mean(y_predict == y_train)

            run_times = 0
            while(accuracy < self.accuracy):
                try:                   
                    for start, end in zip(range(0, shape[0], self.batchSize), range(self.batchSize , shape[0], self.batchSize )):
                        x_batch = x_train[start:end]
                        y_batch = y_train[start:end]
                        sess.run(train_op, feed_dict={X: x_batch, Y: y_batch,keep_prob:self.keep_prob})
                except Exception,ex:
                    print "[WARMING]exception happens when run train model",str(ex)
                    continue
                y_predict = sess.run(pridict_op,feed_dict = {X:x_train,keep_prob:2*self.keep_prob})
                y_predict = self.get_ypredict(y_predict)
                accuracy = np.mean(y_predict == y_train)

                yt = sess.run(pridict_op,feed_dict = {X:x_test,keep_prob:2*self.keep_prob})
                yt = self.get_ypredict(yt)
                ya = np.mean(yt == y_test)

                trainList.append(accuracy)
                timesList.append(run_times)
                testList.append(ya)

                if run_times % 300 == 0:
                    print "run_times = %d accuracy = %f " % (run_times,accuracy)
                if run_times > self.trainTimes:
                    break
                run_times = run_times + 1

        else:
            sess.run(train_op, feed_dict={X: x_train, Y: y_train,})
            y_predict = sess.run(pridict_op,feed_dict = {X:x_train,})
            y_predict = self.get_ypredict(y_predict)
            accuracy = np.mean(y_predict == y_train)
            print "accuracy = ", accuracy

            run_times = 0
            while(accuracy < self.accuracy):
                try:                   
                    sess.run(train_op, feed_dict={X:x_train, Y:y_train})
                except Exception,ex:
                    print "[WARMING]exception happens when run train model",str(ex)
                    continue
                y_predict = sess.run(pridict_op,feed_dict = {X:x_train})
                y_predict = self.get_ypredict(y_predict)
                accuracy = np.mean(y_predict == y_train)

                yt = sess.run(pridict_op,feed_dict = {X:x_test})
                yt = self.get_ypredict(yt)
                ya = np.mean(yt == y_test)

                trainList.append(accuracy)
                timesList.append(run_times)
                testList.append(ya)

                if run_times % 300 == 0:
                    print "run_times = %d accuracy = %f,ya = %f " % (run_times,accuracy,ya)
                if run_times > self.trainTimes:
                    break
                run_times = run_times + 1
            
        print "I have trianed %d times !!!!" % (run_times)
        trainDetail = np.array([timesList,trainList,testList])
        np.save('trainDetail',trainDetail)
        saver = tf.train.Saver()
        saver.save(sess,self.savePath)
        sess.close()


    def predict(self,x_test):
        x_test = self.normalize_xtest(x_test)
        X = tf.placeholder("float", [None, self.inputNumber])
        Y = tf.placeholder("float", [None, self.outputNumber])

        W1 = self.init_weight([self.inputNumber, self.layerOne], 'W1')
        B1 = self.init_bias([self.layerOne], 'B1')

        W2 = self.init_weight([self.layerOne, self.outputNumber], 'W2')
        B2 = self.init_bias([self.outputNumber], 'B2')  

        L2 = self.model(X,  W1, B1)
        if not self.isDroupout:
            hypothesis = tf.nn.relu(tf.matmul(L2, W2) + B2)
            cost = tf.reduce_mean(tf.square((Y - hypothesis)))
            train_op = tf.train.AdamOptimizer(self.learningRate).minimize(cost)
            pridict_op = tf.nn.relu(tf.matmul(L2, W2) + B2)
        else:
            L2 = h = tf.nn.dropout(L2, keep_prob)
            y_out = tf.nn.relu(tf.matmul(L2, W2) + B2)
            cost = tf.reduce_mean(tf.square((Y - y_out)))
            train_op = tf.train.AdamOptimizer(self.learningRate).minimize(cost)
            pridict_op = tf.nn.relu(tf.matmul(L2, W2) + B2)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess,self.savePath)
        if not self.isDroupout:
            y_predict = sess.run(pridict_op, feed_dict={X:x_test})
        else:
            y_predict = sess.run(pridict_op, feed_dict={X:x_test,keep_prob:2*self.keep_prob})
        return self.get_ypredict(y_predict)


    def f_measure(self,y_predict,y_test):
        shape = y_predict.shape
        TP,FP,FN,TN = 0,0,0,0
        for i in xrange(shape[0]):
            if int(y_predict[i]) == 1:
                if int(y_test[i]) == 1:
                    TP = TP + 1
                else:
                    FP = FP + 1
            else:
                if int(y_test[i]) == 1:
                    FN = FN + 1
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

    def get_ypredict(self,y_predict,t = 0.5):
        shape = y_predict.shape
        y_predict = (y_predict >= t)
        return y_predict
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
