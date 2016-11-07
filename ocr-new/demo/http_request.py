# -*- coding: utf-8 -*-
from django.test import TestCase
import urllib
import urllib2
import json
import sys,os,time
from myRecognizer import recognizerObject

media_dir = os.path.join(os.path.dirname(__file__),'./media/data_example/')

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


if __name__ =='__main__':
    import time
    #http post    
    '''测试登录验证'''
    someurl = "http://127.0.0.1:7788/user_login/"  #请求URL
    # someurl = "http://192.168.1.40:7788/user_login/"  #请求URL
    #data = {"username":"admin","password":'admin1688'}
    #ret = http_post(someurl,data)
    #print "user_login ret = ",ret
    
    '''退出测试'''
    #token =ret['token']
    #someurl = "http://127.0.0.1:7788/user_logout/"  #请求URL
    # someurl =  "http://192.168.1.40:7788/user_logout/"
    #logout_ret = http_get(someurl,{"token":token})
    #print "logout_ret = ",logout_ret

    '''测试按照时间查询表格数据'''
    # someurl = "http://127.0.0.1:7788/table_time/"  #请求URL
    someurl = "http://192.168.1.40:7788/table_time/"
    # data = {"token":'',"table_number":2,"time":"2016-4","locat_name":'all'}
    # ret = http_get(someurl,data)
    # print "table_time ret = ",ret  
    '''读文件中的数据发送到服务器table_write 接口，每张６次，时间不同'''
    # for t in range(1,6):
    #     for myTime in range(4,10):
    #         file = media_dir + '/txtfile/table' + str(t) +'_1.txt'
    #         fp = open(file)
    #         data = json.loads(fp.read())
    #         fp.close()
    #         for key in data:
    #             data[key]['time'] = '2016-' + str(myTime) 
    #         print "data = ",data['y1'].keys()
    #         Request_data = {"token":"hk16888","table_number":t,"data":data}
    #         someurl = "http://127.0.0.1:7788/table_write/"  #请求URL
    #         ret = http_post(someurl,Request_data)
    #         print "table_write ret = ",ret
    #         time.sleep(2)　　
    '''根据时间来查询某张表格中只属于杨浦区的数据'''
    someurl = "http://127.0.0.1:7788/table_locat/"  #请求URL
    # request_data = {"token":token,"table_number":5,"start_time":"2016-3","end_time":'2016-9',"locat_name":"杨浦区"}
    # ret = http_get(someurl,request_data)
    # print "table_yangpu ret = ",ret  
    print "it is okay ..."

    '''识别测试'''
    someurl = "http://127.0.0.1:7788/recognize_five/"  #请求URL
    # request_data = {"token":token,'table_number':5}
    # ret = http_post(someurl,request_data)
    # print "table_yangpu ret = ",ret  

    '''测试数据库数据导出'''
    someurl = "http://127.0.0.1:7788/database_out/"
    # request_data = {'token':"hk1688"}
    # ret = http_get(someurl,request_data)
    # print "database_out ret = ",ret.keys()
    # for key in ret['data']:
    #     pass
    # print "key = ",key
    # table5 = ret['data'][key]
    # print "table5 type = ",table5[0]
    
    '''导出数据表　　一张表的所有数据'''
    someurl = "http://127.0.0.1:7788/table_out/"
    # request_data = {'token':"hk1688",'table_number':5}
    # ret = http_get(someurl,request_data)
    # print "database_out ret = ",ret['data'][0].keys()


    '''测试将整个数据库到处为excel表格将路径传到前端'''
    someurl = "http://127.0.0.1:7788/excel_out/"
    # request_data = {'token':token}
    # ret = http_get(someurl,request_data)
    # print "excel_out ret = ",ret

    # file_path = media_dir + 'recognize_five-table5' + '.txt' 
    # fp = open(file_path,'wb')
    # data = json.dumps(ret,separators = (',',':'))
    # fp.write(data)
    # fp.close()
    ''' 
    from method2Recognizer import tableOneRecognizer
    image = './media/table/table1/Img_28.jpg'
    ret1 = tableOneRecognizer(image)
    post_data = {'token':'token','time':'2016-11','table_number':1,'data':ret1}
    someurl = 'http://127.0.0.1:7788/table_write/'
    ret = http_post(someurl,post_data)
    print "write ret = ",ret
    import time
    for i in range(1,7):
        time1 = '2016-' + str(i)
        post_data = {'token':'token','time':time1,'table_number':1,'data':ret1}
        someurl = 'http://127.0.0.1:7788/table_write/'
        ret = http_post(someurl,post_data)
        time.sleep(2)

    '''

    imageName = '../image/img3.jpg'
    recognizerObject.isExcel = True
    retDict = recognizerObject.tableRecognizer(imageName)

    recognizedExcelName = retDict['excel']

    ret = retDict["data"]
    for i in range(1,13):
        t = '2015-' + str(i)
        post_data = {'token':'token','time':t,'table_number':7,'data':ret}
        someurl = 'http://127.0.0.1:7700/table_write/'
        ret1 = http_post(someurl,post_data)
        time.sleep(2)
