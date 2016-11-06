# coding: utf-8
import numpy as np
import pandas as pd
import os,xlwt,xlrd
from statsmodels.tsa.arima_model import ARIMA, _arma_predict_out_of_sample
from myException import MyOwnException

class MyArimaModel(object):
    def __init__(self,p=2,d=1,q=0,start_train=0,end_train=100,next_ndays=1):
        super(MyArimaModel,self).__init__()
        self.p = p
        self.d = d
        self.q = q
        self.start_train = start_train
        self.end_train = end_train
        self.next_ndays = next_ndays
        self.selector = {1:'Open',2:"High",3:"Low",4:"Close",5:"Volume"}
        self.selected = 'Close'
        self.train = None
        self.excelname = os.path.join(os.path.dirname(__file__),'arima.xls')



    def getTrainFromExcel(self,sheet='Sheet1'):
        if not os.path.isfile(self.excelname):
            raise TypeError('excelname must be an excel file')
        xls = pd.ExcelFile(self.excelname)
        train = xls.parse(sheet, index_col='Date') 
        train.index = pd.to_datetime(train.index)
        self.train = train
        return train

    def getTrainFromList(dateList,seriesList):
        book = xlwt.Workbook(encoding = 'utf-8')
        sheet1 = book.add_sheet('Sheet 1',cell_overwrite_ok = True)
        
        if len(dateList) != len(seriesList):
            raise MyOwnException('The length of dateList and seriesList must be the same')

        sheet1.write(0,0,'Date')
        sheet1.write(0,1,'Value')
        length = len(dateList)
        for x in xrange(length):
            sheet1.write(x,0,dateList[x])
            sheet1.write(x,1,seriesList[x])
        book.save(self.excelname)

        train =  self.getTrainFromExcel(self.excelname)
        
        os.remove(self.excelname)
        return train


    def getTrianFromArray(dateArray,serieseArray):
        if dateArray.shape != serieseArray.shape or dateArray.shape[0] < dateArray.shape[1]:
            raise MyOwnException('dateArray and serieseArray must have same shape(n,m),n>m')

        shape = dateArray.shape
        dateArray = dateArray.reshape(shape[1],shape[0])
        serieseArray = serieseArray.reshape(shape[1],shape[0])
        return self.getTrainFromList(dateArray.tolist(),serieseArray.tolist())

    def chooseOneValue(self,number):
        if number in self.selector.keys():
            self.selected = self.selector[int(number)]
        else:
            self.selected = self.selector[4]


    def pridictNextNdays(self,train):
        timeSerize = train[self.selected]
        timeSerize = timeSerize[self.start_train:self.end_train]
        model = ARIMA(timeSerize, order=(self.p,self.d,self.q), freq='D') # build a model
        fitting = model.fit(disp=False)
        forecast, fcasterr, conf_int = fitting.forecast(steps=self.next_ndays, alpha=.05)

        # params = fitting.params
        # residuals = fitting.resid
        # p = fitting.k_ar
        # q = fitting.k_ma
        # k_exog = fitting.k_exog
        # k_trend = fitting.k_trend
        # forecast = _arma_predict_out_of_sample(params,self.next_ndays,residuals, p, q, k_trend, k_exog, endog=timeSerize, exog=None, start=len(timeSerize))
        return  forecast
        

    def testArima(self,train):
        realSerize = train[self.selected]
        timeSerize = realSerize[self.start_train:self.end_train]
        realData = train[self.selected][self.end_train:self.next_ndays]
        model = ARIMA(timeSerize, order=(self.p,self.d, self.q)) # build a model
        fitting = model.fit(disp=False)
        forecast, fcasterr, conf_int = fitting.forecast(steps=self.next_ndays, alpha=.05)
        # params = fitting.params
        # residuals = fitting.resid
        # p = fitting.k_ar
        # q = fitting.k_ma
        # k_exog = fitting.k_exog
        # k_trend = fitting.k_trend
        # forecast = _arma_predict_out_of_sample(params,self.next_ndays,residuals, p, q, k_trend, k_exog, endog=timeSerize, exog=None, start=len(timeSerize))
        return  {'real':list(realSerize)[self.end_train:self.end_train+self.next_ndays],'pridiction':forecast}

arimaObject = MyArimaModel()
if __name__ == '__main__':

    basedir = os.getcwd()
    filename = os.path.join(basedir,'yahoo_finance5.xls')
    arimaObject. excelname= filename

    arimaObject.start_train = 0
    arimaObject.end_train = 100

    train = arimaObject.getTrainFromExcel('Sheet1')
    arimaObject.chooseOneValue(4)

    arimaObject.next_ndays = 5

    result = arimaObject.testArima(train)
    print "real = ",result['real']
    print "pridiction = ",result['pridiction']

