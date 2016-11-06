# -*- coding: utf-8 -*-
from django.contrib.auth import authenticate,login, logout
from django.db.models import Q
from django.conf import settings
from django.http import FileResponse,StreamingHttpResponse
from rest_framework.response import Response
from rest_framework import permissions, renderers, viewsets
from rest_framework.decorators import permission_classes,detail_route,api_view,list_route
# from rest_framework_jwt.settings import api_settings
from models import *
# from serializer import *
import os,json,hashlib,datetime
from myNeurolNetworkModel import MyNeurolNetworkModel
import pandas as pd
import numpy as np

def crossDomainResponse(data):
    response = Response(data)
    response['Access-Control-Allow-Origin']= "*"
    response['Access-Control-Allow-Methods'] = ['GET','POST']
    response['Access-Control-Allow-Headers'] = "x-requested-with,content-type"
    return response


'''将股票数据写入数据库'''
@api_view(['POST','GET'])
def predict_onlyclose(request):
    print "----------------------predict_onlyclose---------------------------"
    try:
        token      = str(request.GET.get('token')) 
        is_train   = int(request.GET.get('is_train')) # 1 train;0 predict
    except Exception,ex:
        print "Exception:",ex
        return crossDomainResponse({"code":201,"msg":"Bad arguments:"+str(ex)})

    filename = './stock_fanny.xlsx'
    
    filename = os.path.join(os.path.dirname(__file__),filename)
    xls = pd.ExcelFile(filename)
    
    df_train = xls.parse('Sheet4', index_col='Date') # train

    close = df_train['Close']
    
    closeArray = np.array([close]).reshape(len(close),1)

    shape = closeArray.shape
    related = 5

    x_sample = np.zeros((shape[0]-related,5))
    y_sample = np.zeros((shape[0]-related,1))

    for i in xrange(shape[0] - related):
        x_sample[i,:] = closeArray[i:i+related,0].reshape(1,related)
        y_sample[i,0] = closeArray[i+related,0]

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

    mYnnModel = MyNeurolNetworkModel()
    mYnnModel.errorRate = 0.040

    if is_train == 1:
        mYnnModel.train(x_train,y_train)
        return crossDomainResponse({"code":200,"msg":"ok"})

    else:
        y_predict = mYnnModel.predict(x_test)
        return crossDomainResponse({"code":200,"msg":"ok","y_test":y_test.tolist(),"y_predict":y_predict.tolist()})