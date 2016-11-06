#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle


class BayesClassifier(object):
    
    def __init__(self,):
        super(BayesClassifier,self).__init__()
        self.params       = {}
        self.outdir       = os.path.join(os.path.dirname(__file__),'classifierFile/')             #模型存储路径
        self.modeSaveFile = self.outdir  + 'BayesModel.pkl'
        self.saveNeeded   = True
        self.sectionNumber= 20
        self.localDays    = 21

        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

    def seperateDataset(self,x_train,y_train):
        shape = x_train.shape
        class0,class1 = 0,1
        xtrain_c0,xtrain_c1 = np.zeros((1,shape[1])),np.zeros((1,shape[1]))
        count_c0,count_c1 = 0,0

        for i in xrange(shape[0]):
            if int(y_train[i,:]) == class0:
                count_c0 = count_c0 + 1
                if not xtrain_c0.any():
                    xtrain_c0 = x_train[i,:]
                else:
                    xtrain_c0 = np.vstack([xtrain_c0,x_train[i]])
            else:
                count_c1 = count_c1 + 1
                if not xtrain_c1.any():
                    xtrain_c1 = x_train[i,:]
                else:
                    xtrain_c1 = np.vstack([xtrain_c1,x_train[i]])

        return xtrain_c0,xtrain_c1,count_c0,count_c1


    def local_normalize(self,x_train):
        days = self.localDays
        shape = x_train.shape
        i = 0
        data = []
        while(True):
            start = days*i
            end = (i+1)*days
            if end > shape[0]:
                end = shape[0]
            local = x_train[start:end,:]
            xmax = np.amax(local, axis=0)
            xmin = np.amin(local, axis=0)           
            local = (local - xmin) / (xmax - xmin)
            data.append(local)
   
            if end == shape[0]:
                break
            i = i + 1
        normalize_train = np.vstack(data)
        # print "normalize_train.shape = ",normalize_train.shape
        return normalize_train*20


    def train(self,x_train,y_train):
        # print "train ..."
        shape = x_train.shape
        x_train = self.local_normalize(x_train)
        xtrain_c0,xtrain_c1,count_c0,count_c1 = self.seperateDataset(x_train,y_train)
        # print "count_c0 = %d,count_c1 = %d " % (count_c0,count_c1)
        # print "xtrain_c1 = ",xtrain_c1.shape
        self.params["Pci"] = [float(count_c0)/(count_c0 + count_c1),float(count_c1)/(count_c0 + count_c1)]
        class0 = 'class0'
        class1 = 'class1'
        for j in range(shape[1]):
            p_num = 'p' + str(j)
            self.params[p_num] = {class0:self.train_section(xtrain_c0[:,j]),class1:self.train_section(xtrain_c1[:,j])}
        # print "self.params[p0] = ",self.params
        if self.saveNeeded:
            self.save()

    def predict(self,x_test):
        x_test = self.local_normalize(x_test)
        shape = x_test.shape
        Pci = self.params["Pci"]
        # print "Pci[0] = ",Pci[0]
        # print "Pci[1] = ",Pci[1]

        y_predict = []
        for i in xrange(shape[0]):
            
            estimator = []
            xnum = len(self.params.keys()) - 1
            class0 = "class0"
            class1 = "class1"
            for j in range(xnum):
                p_num = 'p' + str(j)
                test0 = x_test[i,j]
                # print "test0 = ",test0
                params0 = self.params[p_num][class0]
                # print "params = ",self.params[p_num]
                p_c0 = self.predict_section(test0,params0)
                
                test1 = x_test[i,j]
                params1 = self.params[p_num][class1]
                p_c1 = self.predict_section(test1,params1)

                estimator.append([p_c0,p_c1])             

            p0 = []
            p1 = []            
            # print "estimator = ",estimator
            for e in estimator:
                # print "e = ",e
                p0.append(e[0])
                p1.append(e[1])
           

            p0.append(float(Pci[0]))
            p1.append(float(Pci[1]))
            # print "p0 = ",p0
            # print "p1 = ",p1

            try:
                pc0 = reduce(self.multiply,p0)*(1e14)   #or too small
                pc1 = reduce(self.multiply,p1)*(1e14)
                # print "pc0 = %f,pc1 = %f " % (pc0,pc1)
                # print "i = ",(i,x_test[i,:].shape)
            except Exception,ex:
                print "-"*80
                print "[Exception]: ",str(ex)
                # print "p0 = ",p0
                # print "p1 = ",p1
                print "i = ",(i,x_test[i].shape)
                print "[Exception]: ",str(ex)
                print "-"*80
                os.sys.exit()
            try:
                if pc0 > pc1:
            	    result = 0
                else:
                    result = 1
            except Exception,ex:
                print "-"*80
                print "[Exception]: ",str(ex)
                # print "pc0 = ",pc0
                # print "pc1 = ",pc1
                print "-"*80
                
                os.sys.exit()
            y_predict.append([result])

        return np.array(y_predict).reshape(shape[0],1)

    def save(self,filename = ''):
        if not filename:
            filename = self.modeSaveFile
        fp = open(filename,'wb')
        pickle.dump(self.params,fp)
        fp.close()


    def load(self,filename = ''):
        if not filename:
            filename = self.modeSaveFile
        fp = open(filename,'rb')
        self.params = pickle.load(fp)
        fp.close()



    def multiply(self,a,b):
        return a*b*(10)

    def train_frequency(self,x):
        xshape = x.shape
        xmax = x.max()
        xmin = x.min()
        xaxis = range(int(xmin),int(xmax+1))
        num = xmax - xmin + 1
        yaxis = np.zeros((int(num),))
    
        for xi in xrange(xshape[0]):
            index = xaxis.index(int(x[xi]))
            yaxis[index] = yaxis[index] + 1
        yaxis = yaxis + 1
        return {"value":xaxis,"probability":yaxis/np.sum(yaxis),"frequency":yaxis}

    def predict_frequnecy(self,x0,params):
        class0,class1 = params["class0"],params["class1"]
        if int(x0) in class0["value"]:
            value = class0['value']
            index = value.index(x0)
            p0_c0 = class0["probability"][index]
        else:
            p0_c0 = float(np.min(class0["probability"])/2)


        if int(x0) in class1["value"]:
            value = class1['value']
            index = value.index(x0)
            p0_c1 = class1["probability"][index]
        else:
            p0_c1 = float(np.min(class1["probability"])/2)
        # print "[p0_c0,p0_c1] = ",[p0_c0,p0_c1]
        return [p0_c0,p0_c1]


    def train_gussian(self,x1):
        retDict = self.train_frequency(x1)
        frequency = retDict["frequency"]
        value = retDict["value"]
        umean = np.mean(frequency)
        sigama = frequency.var()
        return {"value":value,"umean":umean,"sigama":sigama}

    def gussian(self,x,umean,sigama):
    	c = 1.0/(sigama*np.sqrt(2*np.pi))
    	e = -1.0*(x-umean)*(x-umean)/(2*sigama*sigama)
    	print "c = %f,e = %f,gussian = %f" % (c,e,c*np.exp(e))
    	return c*np.exp(e)


    def predict_gussian(self,x,params):
        sigama = params["sigama"]
        umean = params['umean']
        # print "predict_gussian = " ,self.gussian(x,umean,sigama)
        return self.gussian(x,umean,sigama)


    def getThresh(self,x,num):
        xmax = np.max(x)
        xmin = np.min(x)
        length = float(xmax-xmin)/num
        xThresh = np.arange(xmin,xmax+length+1,length)
        return xThresh

    def getIndex(self,e,thresh):
        for i in range(len(thresh) - 1):
            if (e >= thresh[i]) and (e < thresh[i+1]):
                return i

    def countFrequency(self,x,num):
        xshape = x.shape
        xTresh = self.getThresh(x,num)

        domain = []
        for i in xrange(xTresh.shape[0] - 1):
            domain.append([xTresh[i],xTresh[i+1]])

        yaxis = np.zeros((len(domain),))

        for xi in xrange(xshape[0]):
            index = self.getIndex(x[xi],xTresh)
            yaxis[index] = yaxis[index] + 1
        yaxis = yaxis + 1
        return yaxis/np.sum(yaxis)


    def train_section(self,x):
        num = self.sectionNumber
        xTresh = self.getThresh(x,num)
        yaxis = self.countFrequency(x,num)
        # print "yaxis = ",yaxis
        return {"xTresh":xTresh,"probability":yaxis}

    def predict_section(self,x5,params):
        xTresh = params["xTresh"]
        probability = params["probability"]
        index = self.getIndex(x5,xTresh)
        if not index:
            return float(np.min(probability)/2)

        probability = probability.tolist()
        if index < len(probability):
            # print "probability[index] = ",probability[index]
            return probability[index]
        else:
            # print "float(np.min(probability)/2) = ",float(np.min(probability)/2)
            return float(np.min(probability)/2)