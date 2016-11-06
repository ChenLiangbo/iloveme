#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os 
import pickle


outdir = os.path.join(os.path.dirname(__file__),'myClassifierModel/')

if not os.path.exists(outdir):
    os.mkdir(outdir)



class MyStatModel(object):
 
    def __init__(self,):
        super(MyStatModel,self).__init__()
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.outdir    = os.path.join(os.path.dirname(__file__),'myClassifierModel/')
        self.modeldir  = ''
        self.modelpath = ''
        self.parameterpath = ''

        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)


    def save_model(self,):
        f = open(self.modelpath,'wb')
        pickle.dump({"model":self.model},f)
        f.close()

    def load_model(self, ):
        f = open(self.modelpath,'rb')
        params =  pickle.load(f)
        self.model = params["model"]
        f.close()

    def normalize_train(self,x_train,y_train):
        xmax = np.amax(x_train, axis=0)
        xmin = np.amin(x_train, axis=0)
        x_train = (x_train - xmin) / (xmax - xmin)
        x_parameter = {"xmin":xmin,"xmax":xmax}
        X = {"x_train":x_train,"x_parameter":x_parameter}
        self.x_train = x_train

        ymax = np.amax(y_train, axis=0)
        ymin = np.amin(y_train, axis=0)
        y_train = (y_train - ymin) / (ymax - ymin)
        y_parameter = {"ymax":ymax,"y_train":y_train}
        Y = {"y_train":y_train,"y_parameter":y_parameter}
        self.y_train = y_train

        f = open(self.parameterpath,'wb')
        pickle.dump({"X":X,"Y":Y},f)
        f.close()

    def normalize_xtrain(self,x_train):
        xmax = np.amax(x_train, axis=0)
        xmin = np.amin(x_train, axis=0)
        x_train = (x_train - xmin) / (xmax - xmin)
        x_parameter = {"xmin":xmin,"xmax":xmax}
        X = {"x_train":x_train,"x_parameter":x_parameter}
        self.x_train = x_train

        f = open(self.parameterpath,'wb')
        pickle.dump(X,f)
        f.close()
        return x_train

    def normalize_xtest(self,x_test):
        f = open(self.parameterpath,'rb')
        X =  pickle.load(f)    #X = {"x_train":x_train,"x_parameter":x_parameter}
        f.close()
        x_parameter = X["x_parameter"]
        xmax,xmin = x_parameter["xmax"],x_parameter["xmin"]
        return (x_test - xmin) / (xmax - xmin)

    def denormalize_ypredict(self,y_predict,y_parameter):
        f = open(self.parameterpath,'rb')
        params =  pickle.load(f)
        f.close()
        y_parameter = params["Y"]["y_parameter"]
        ymax,ymin = y_parameter["ymax"],y_parameter["ymin"]
        return y_predict*(ymax - ymin) + ymin

    def get_precision(self,x_test,y_test):
        y_test_predict = self.predict(x_test)
        return np.mean(y_test == y_test_predict)


class MyKNearest(MyStatModel):
    def __init__(self, k = 1):
        super(MyKNearest,self).__init__()
        self.k     = k
        self.model = cv2.KNearest()
        self.modeldir      = self.outdir + 'MyKNearestModel/'
        self.modelpath     = self.modeldir + 'MyKNearestModel.txt'
        self.parameterpath = self.modeldir + 'MyKNearestParams.txt'

        if not os.path.exists(self.modeldir):
            os.mkdir(self.modeldir)

    def train(self,x_train,y_train):
        x_train = self.normalize_xtrain(x_train)
        self.model.train(x_train, y_train)
        # self.save_model()

    def predict(self, x_test):
        # self.load_model()
        x_test = self.normalize_xtest(x_test)
        retval, results, neigh_resp, dists = self.model.find_nearest(x_test, self.k)
        return results




class MySVM(MyStatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv2.SVM()
        self.outdir        = outdir
        self.modeldir      = self.outdir + 'MySVMModel/'
        self.modelpath     = self.modeldir + 'MySVMModel.txt'
        self.parameterpath = self.modeldir + 'MySVMParams.txt'

        if not os.path.exists(self.modeldir):
            os.mkdir(self.modeldir)


    def train(self,x_train,y_train):
        x_train = self.normalize_xtrain(x_train)        
        self.model.train(x_train, y_train, params = self.params)

    def predict(self, x_test):
        x_test = self.normalize_xtest(x_test)
        return self.model.predict_all(x_test)




class MyBoost(MyStatModel):
    def __init__(self,inLength,outLength):
        super(MyBoost,self).__init__()
        self.in_length = inLength
        self.out_length = outLength
        self.model = cv2.Boost()
        self.params = dict(max_depth = 1)
        self.var_type = np.array([cv2.CV_VAR_NUMERICAL]*self.in_length + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)

    def train(self,x_train,y_train):
        x_train = self.normalize_xtrain(x_train)
        x_train = np.float32(x_train)
        y_train = np.float32(y_train)
        self.model.train(x_train,cv2.CV_ROW_SAMPLE,y_train,varType = self.var_type,params = self.params)


    def predict(self,x_test):
        x_test = self.normalize_xtest(x_test)
        results = np.zeros((x_test.shape[0],self.out_length),dtype = x_test.dtype)
        for i in xrange(x_test.shape[0]):
            results[i,0] = self.model.predict(x_test[i,:],returnSum = False)
        return results

    def pricision_test(self,x_test,y_test):
        results = np.zeros(y_test.shape,dtype = y_test.dtype)
        for i in xrange(y_test.shape[0]):
            results[i,0] = self.model.predict(x_test[i,:],returnSum = False)
        acurracy = (y_test == results)
        return np.mean(acurracy)



class MyRondomForest(MyStatModel):
    def __init__(self,inLength,outLength):
        super(MyRondomForest,self).__init__()
        self.in_length = inLength
        self.out_length = outLength
        self.model = cv2.RTrees()
        self.var_type = np.array([cv2.CV_VAR_NUMERICAL] * self.in_length + [cv2.CV_VAR_CATEGORICAL], np.uint8)
        self.params = dict(depth = 32)

    def train(self,x_train,y_train):
        x_train = np.float32(x_train)
        y_train = np.float32(y_train)
        self.model.train(x_train,cv2.CV_ROW_SAMPLE,y_train,
        varType = self.var_type,
        params = self.params)


    def predict(self,x_test):
        x_test = np.float32(x_test)
        results = np.zeros((x_test.shape[0],self.out_length),dtype = np.float32)
        for i in xrange(x_test.shape[0]):
            results[i,0] = self.model.predict(x_test[i,:])
        return results


    def precision_test(self,x_test,y_test):
        x_test = np.float32(x_test)
        results = np.zeros(y_test.shape,dtype = y_test.dtype)
        for i in xrange(y_test.shape[0]):
            results[i,0] = self.model.predict(x_test[i,:])

        acurracy = (y_test == results)
        return np.mean(acurracy)

if __name__ == '__main__':
    pass