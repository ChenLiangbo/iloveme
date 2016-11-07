# -*- coding: utf-8 -*-
import urllib
import urllib2
import json
import sys

apikey = 'ac9ebaf03dbf7014b03844847d957362'

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

#baudi turing robot by means of apikey
def BaiduTuringRobot(apikey,content):
    key = '879a6cb3afb84dbf4fc84a1df2ab7319'
    userid = 'eb2edb736'
    url = 'http://apis.baidu.com/turing/turing/turing?key=%s&info=%s&userid=%s' % (key,info,userid)
    req = urllib2.Request(url)
    req.add_header("apikey", apikey)
    resp = urllib2.urlopen(req)
    ret_json = resp.read()
    ret_dict = json.loads(ret_json)  
    return ret_dict["text"]   

#get city welther from baidu welther by means of apikey,but not perfact
def BaiduCityWelther(apikey,cityname):
    url = 'http://apis.baidu.com/apistore/weatherservice/citylist?cityname=%s' % (cityname,)
    req = urllib2.Request(url)
    req.add_header("apikey", apikey)
    resp = urllib2.urlopen(req)
    content = resp.read()
    content = json.loads(content)  
    return content['retData'] 

#get beauty picture from baiduMeinv by means of apikey
def BaiduMeitu(apikey,num,outdir):
    url = 'http://apis.baidu.com/txapi/mvtp/meinv?num=%d' % (num,)
    req = urllib2.Request(url)
    req.add_header("apikey", apikey)
    resp = urllib2.urlopen(req)
    content = resp.read()
    content = json.loads(content)
    newslist = content["newslist"] 
    for d in newslist: #d.keys()= url,picUrl,description,ctime,title  
        picUrl = d['picUrl']
        response = urllib2.urlopen(picUrl)
        picfile = response.read()
        imageName = outdir +  '/' + d['title'] + '.jpg'
        fp = open(imageName,'w')
        fp.write(picfile)
        fp.close()            





if __name__ =='__main__':
    import time
    #http post    
    
    '''测试登录验证'''
    # someurl = "http://127.0.0.1:5656/login/"  #请求URL
    # someurl = 'http://polarwin.cn:3821/login/'
    # data = {"username":"admin","password":"admin"}
    # ret = http_post(someurl,data)
    # print " login ret = ",ret   

    # '''测试站点查询'''    
    # # station = {'name':"长宁站"}
    # station = {'name':"欧威来"}  
    # URL = "http://127.0.0.1:3821/stations/"
    # # URL = 'http://polarwin.cn:3821/login/'
    # result = http_get(URL,station)
    # print "get result:",result
    # time.sleep(1)

    '''测试用户反馈'''
    # feedbackURL= "http://127.0.0.1:3821/feedback/"
    # data = {"content":"Test information from chanliangbo"}
    # ret = http_post(feedbackURL,data)
    # print "ret = ",ret
    # time.sleep(1)

    '''测试上传地理位置信息'''
    # data = {"latitude":31,"longitude":123,"device_id":"TESTFQQ1"}
    # # positionURL = "http://polarwin.cn:3821/position/"
    # positionURL = "http://127.0.0.1:3821/position/"
    # ret = http_post(positionURL,data)
    # print "ret = ",ret
    # time.sleep(1)

    # '''测试报警/上报信息'''
    # alertURL = "http://127.0.0.1:3821/alert/"
    # data = {"name":"长宁站","locat":"长宁","flag":1,"number":0}
    # ret = http_get(alertURL,data)
    # print "ret = ",ret
    # time.sleep(1)

    # '''测试设备信息查询'''
    # deviceURL = "http://127.0.0.1:3821/device/"
    # data = {"name":"长宁站","locat":"长宁","flag":1,"number":0}
    # ret = http_get(deviceURL,data)
    # print "ret = ",ret
    # time.sleep(1)

    # '''画图数据查询'''
    # '''折线图多天查询，参数有三个'''
    # chartURL = "http://127.0.0.1:3821/chart/"
    # device_id = '030W0627'
    # start_date =  '2015-12-31T16:00:00.000Z'
    # end_date  = '2016-01-02T16:00:00.000Z'
    # data = {"device_id":device_id,"start_date":start_date,"end_date":end_date}
    # ret = http_get(chartURL,data)
    # print "ret = ",ret
    # time.sleep(1)

    # '''圆图查询，返回一天数据'''
    # start_date  = '2016-04-02T16:00:00.000Z'
    # end_date  = '0'
    # device_id = '030W0627'
    # data = {"device_id":device_id,"start_date":start_date,"end_date":end_date}
    # ret = http_get(chartURL,data)
    # print "ret = ",ret
    # time.sleep(1)

    # '''默认情况，返回昨天零点现在数据，参数只有一个'''
    # device_id  = '030W0617'
    # data = {"device_id":device_id}
    # ret = http_get(chartURL,data)
    # print "ret = ",ret
    # time.sleep(1)

    # info = 'hello'
    # content = BaiduTuringRobot(apikey,info)
    # print "content = ",content
    # cityname = '宝山'
    # # cityname = '101020100'
    # welther = BaiduCityWelther(apikey,cityname)  # list of dict
    # for city in welther:
    #     for key in city:
    #         print key,city[key]
    #     print '----------------------------------'

    # outdir = '../images'
    # BaiduMeitu(apikey,10,outdir)
    url1 = "http://127.0.0.1:5656/alert_search/"   
    data = {"flag":2,"type":0,"content":"030W0616"}
    ret = http_get(url1,data)   
    print "alert_search ret = ",ret


 
