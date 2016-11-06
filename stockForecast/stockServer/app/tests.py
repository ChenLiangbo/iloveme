# -*- coding: utf-8 -*-

import os
import pandas
import urllib2
import urllib
import json
import sys,os,time


print "start ..."
def http_post(url,data):
    data = json.dumps(data)
    req = urllib2.Request(url)                 #返回一个请求
    try:
        res = urllib2.urlopen(req,data)        #返回一个响应实例
    except urllib2.URLError, e:                #以下几句是异常捕捉
        if hasattr(e, 'reason'):
            print 'Reason: ', e.reason
        elif hasattr(e, 'code'):
            print 'Error code: ', e.code
        else:
            print "other erro"
        return {"errmsg":"erro happens"}

    ret_json = res.read()  #可以把返回的实例当做文件对象操作 可读可写
    ret_python = json.loads(ret_json)
    return ret_python

def http_get(url,data):
    urldata = urllib.urlencode(data)
    url = url + '?' + urldata
    response = urllib2.urlopen(url)
    content = response.read()
    content = json.loads(content) 
    return content

if __name__ == '__main__':
    '''登录验证测试'''

    # someurl = 'http://127.0.0.1:2000/predict_onlyclose/'
    # get_data = {"token":"adminhk1688ilove","is_train":1}
    # ret = http_get(someurl,get_data)
    # print "predict_onlyclose ret = ",ret
    # print "training successfully ..."
    
    someurl = 'http://127.0.0.1:2000/predict_onlyclose/'
    get_data = {"token":"adminhk1688ilove","is_train":0}
    ret = http_get(someurl,get_data)
    y_test = ret["y_test"]
    y_predict = ret["y_predict"]

    from matplotlib import pyplot as plt
    plt.plot(y_test,'ro')
    plt.plot(y_predict,'bo')
    plt.plot(y_test,'r-')
    plt.plot(y_predict,'b-')
    plt.legend(['y_test','y_predict'])
    plt.grid(True)
    plt.show()