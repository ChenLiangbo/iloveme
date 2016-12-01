#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle


class BayesClassifier(object):
    
    def __init__(self,):
        super(BayesClassifier,self).__init__()
        self.params = {"p0":None,    #["class0"] {"value":,"probability":,"frequency":} |["class1"]
                       "p1":None,    #["class0"] {"value":,"umean":,"sigama":}          |["class1"]
                       "p2":None,    #["class0"] {"value":,"umean":,"sigama":}          |["class1"]
                       "p3":None,    #["class0"] {"value":,"probability":,"frequency":} |["class1"]
                       "p4":None,    #["class0"] {"value":,"probability":,"frequency":} |["class1"]
                       "p5":None,    #["class0"] {"xTresh":,"probability":}
                       "p6":None,
                       "p7":None,     
                       "Pci":[0,0]}    #[Pc0,Pc1]
        self.modeSaveFile = 'BayesModel.pkl'
        self.saveNeeded   = True
        self.sectionNumber= 20


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


    def train(self,x_train,y_train):
        shape = x_train.shape
        
        xtrain_c0,xtrain_c1,count_c0,count_c1 = self.seperateDataset(x_train,y_train)

        # print "count_c0 = %d,count_c1 = %d " % (count_c0,count_c1)

        self.params["Pci"] = [float(count_c0)/(count_c0 + count_c1),float(count_c1)/(count_c0 + count_c1)]
        self.params["p0"] = {"class0":self.train_frequency(xtrain_c0[:,0]),"class1":self.train_frequency(xtrain_c1[:,0])}
        
        # self.params["p1"] = {"class0":self.train_gussian(xtrain_c0[:,1]),"class1":self.train_gussian(xtrain_c1[:,1])}  
        # self.params["p2"] = {"class0":self.train_gussian(xtrain_c0[:,2]),"class1":self.train_gussian(xtrain_c1[:,2])}  

        self.params["p1"] = {"class0":self.train_section(xtrain_c0[:,1]),"class1":self.train_section(xtrain_c1[:,1])}
        self.params["p2"] = {"class0":self.train_section(xtrain_c0[:,2]),"class1":self.train_section(xtrain_c1[:,2])}

        self.params["p3"] = {"class0":self.train_frequency(xtrain_c0[:,3]),"class1":self.train_frequency(xtrain_c1[:,3])}
        self.params["p4"] = {"class0":self.train_frequency(xtrain_c0[:,4]),"class1":self.train_frequency(xtrain_c1[:,4])}
        self.params["p5"] = {"class0":self.train_section(xtrain_c0[:,5]),"class1":self.train_section(xtrain_c1[:,5])}
        self.params["p6"] = {"class0":self.train_section(xtrain_c0[:,6]),"class1":self.train_section(xtrain_c1[:,6])}
        self.params["p7"] = {"class0":self.train_section(xtrain_c0[:,7]),"class1":self.train_section(xtrain_c1[:,7])}
        # print "p1 = ",self.params["p0"]["class0"]
        if self.saveNeeded:
            self.save()

    def predict(self,x_test):
        if self.saveNeeded:
            self.load()

        shape = x_test.shape
        Pci = self.params["Pci"]
        # print "Pci[0] = ",Pci[0]
        # print "Pci[1] = ",Pci[1]

        y_predict = []
        for i in xrange(shape[0]):
            
            estimator = []
            estimator.append(self.predict_frequnecy(x_test[i,0],self.params["p0"])) #p0

            '''
            p1_c0 = self.predict_gussian(x_test[i,1],self.params["p1"]["class0"])
            p1_c1 = self.predict_gussian(x_test[i,1],self.params["p1"]["class1"])
            estimator.append([p1_c0,p1_c1])                               #p1
            print "p1 = ",[p1_c0,p1_c1]

            p2_c0 = self.predict_gussian(x_test[i,2],self.params["p1"]["class0"])
            p2_c1 = self.predict_gussian(x_test[i,2],self.params["p1"]["class1"])
            estimator.append([p2_c0,p2_c1])                                #p2
            print "p2 = ",[p2_c0,p2_c1]
            '''

            p1_c0 = self.predict_section(x_test[i,1],self.params["p1"]["class0"])
            p1_c1 = self.predict_section(x_test[i,1],self.params["p1"]["class1"])
            estimator.append([p1_c0,p1_c1]) 
            # print "p1 = ",[p1_c0,p1_c1]

            p2_c0 = self.predict_section(x_test[i,2],self.params["p2"]["class0"])
            p2_c1 = self.predict_section(x_test[i,2],self.params["p2"]["class1"])
            estimator.append([p2_c0,p2_c1]) 
            # print "p2 = ",[p2_c0,p2_c1]

            estimator.append(self.predict_frequnecy(x_test[i,3],self.params["p3"])) #p3
            # print "p3 = ",self.predict_frequnecy(x_test[i,3],self.params["p3"])


            estimator.append(self.predict_frequnecy(x_test[i,4],self.params["p4"])) #p4
            # print 'p4 = ',self.predict_frequnecy(x_test[i,4],self.params["p4"])

            p5_c0 = self.predict_section(x_test[i,5],self.params["p5"]["class0"])
            p5_c1 = self.predict_section(x_test[i,5],self.params["p5"]["class1"])
            estimator.append([p5_c0,p5_c1])                                #p5
            # print "p5 = ",[p5_c0,p5_c1]

            p6_c0 = self.predict_section(x_test[i,6],self.params["p6"]["class0"])
            p6_c1 = self.predict_section(x_test[i,6],self.params["p6"]["class1"])
            estimator.append([p6_c0,p6_c1])                                #p6
            # print "p6 = ",[p6_c0,p6_c1]


            p7_c0 = self.predict_section(x_test[i,7],self.params["p7"]["class0"])
            p7_c1 = self.predict_section(x_test[i,7],self.params['p7']["class1"])
            estimator.append([p7_c0,p7_c1])                                #p7
            # print "p7 = ",[p7_c0,p7_c1]
            # print "="*100

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
        return a*b

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

    def transform_pmodel(self,xsample):
        shape = xsample.shape
        Pci = self.params["Pci"]
        plist = []
        for i in xrange(shape[0]):
            
            estimator = []
            p0_0,p0_1 = self.predict_frequnecy(xsample[i,0],self.params["p0"])
            estimator.append(p0_0) #p0
            estimator.append(p0_1) #p0

            p1_c0 = self.predict_section(xsample[i,1],self.params["p1"]["class0"])
            p1_c1 = self.predict_section(xsample[i,1],self.params["p1"]["class1"])
            estimator.append(p1_c0)
            estimator.append(p1_c1)  
            # print "p1 = ",[p1_c0,p1_c1]

            p2_c0 = self.predict_section(xsample[i,2],self.params["p2"]["class0"])
            p2_c1 = self.predict_section(xsample[i,2],self.params["p2"]["class1"])
            estimator.append(p2_c0) 
            estimator.append(p2_c1) 
            # print "p2 = ",[p2_c0,p2_c1]
            
            p3_0,p3_1 = self.predict_frequnecy(xsample[i,3],self.params["p3"])
            estimator.append(p3_0) #p3
            estimator.append(p3_1)
            # print "p3 = ",self.predict_frequnecy(x_test[i,3],self.params["p3"])

            p4_0,p4_1 = self.predict_frequnecy(xsample[i,4],self.params["p4"])
            estimator.append(p4_0) #p4
            estimator.append(p4_1) #p4

            # print 'p4 = ',self.predict_frequnecy(x_test[i,4],self.params["p4"])

            p5_c0 = self.predict_section(xsample[i,5],self.params["p5"]["class0"])
            p5_c1 = self.predict_section(xsample[i,5],self.params["p5"]["class1"])
            estimator.append(p5_c0)                                #p5
            estimator.append(p5_c1)  
            # print "p5 = ",[p5_c0,p5_c1]

            p6_c0 = self.predict_section(xsample[i,6],self.params["p6"]["class0"])
            p6_c1 = self.predict_section(xsample[i,6],self.params["p6"]["class1"])
            estimator.append(p6_c0)                                #p6
            estimator.append(p6_c1)  
            # print "p6 = ",[p6_c0,p6_c1]


            p7_c0 = self.predict_section(xsample[i,7],self.params["p7"]["class0"])
            p7_c1 = self.predict_section(xsample[i,7],self.params['p7']["class1"])
            estimator.append(p7_c0)                                #p7
            estimator.append(p7_c1)
            # print "p7 = ",[p7_c0,p7_c1]

            estimator.append(Pci[0])
            estimator.append(Pci[1])

            plist.append(estimator)
            # print "plist = ",plist
            # break
        parray = np.array(plist)
        return np.float32(parray)

    def f_measure(self,y_predict,y_test):
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


    def normalize(self,x):
        shape = x.shape
        xlist = np.hsplit(x,shape[1])
        for i in xrange(shape[1]):
            xmean = np.mean(xlist[i])
            xvar = xlist[i].var()
            xlist[i] = (xlist[i] - xmean) / xvar
        return np.hstack(xlist)