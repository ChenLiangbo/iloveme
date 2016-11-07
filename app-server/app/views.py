# -*- coding:utf-8 -*-
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render_to_response
from django.db import connection
from models import *
from django.db.models import Avg, Max, Min, Count, Q
from django.contrib.admin.models import LogEntry
from django.views.decorators.csrf import csrf_exempt
import time, datetime, re, os
from django.contrib.auth.decorators import login_required
try:
    import json
except:
    import simplejson as json  
from django.http import HttpResponse  
from mychat import mychat
# import  MySQLdb


'''
全局常量定义
为了使得每次刷新都可以加载十个数据，此处定义全局变量
'''
limitNum = 10
form = '%Y-%m-%d %H:%M:%S'        #This fromat used frequently as datetime format

#Transform python data to json data,and response to method with data for login
def response_to_app(data,method):
    response = HttpResponse(data)
    response['Access-Control-Allow-Origin']= "*"
    response['Access-Control-Allow-Methods'] = method 
    response['Access-Control-Allow-Headers'] = "x-requested-with,content-type" 
    return response

#Transform data to json and response to app request
def response_with_json(data,method):
    jsondata  = json.dumps(data,separators = (',',':'))
    response  = HttpResponse('%s' % (jsondata))
    response['Access-Control-Allow-Origin']= "*"
    response['Access-Control-Allow-Methods'] = method 
    response['Access-Control-Allow-Headers'] = "x-requested-with,content-type" 
    return response

def get_station_id_list(username):
    '''
    import request
    return a list of station.id im privilige of request.user
    '''
    station_id_list = []
    users = User.objects.filter(username = username) #will get a  list with one item
    auth_user_stations_list = Auth_user_stations.objects.filter(user_id = users[0].id)
    for auth_user_station in auth_user_stations_list:
        station_list = Station.objects.filter(id = auth_user_station.station_id) 
        station_id_list.append(station_list[0].id)
    return station_id_list
 
    
@csrf_exempt
def testLogin(request):
    print "============running in login============"
    if request.method == 'GET':    
        data = "It's ok!"
        response = response_to_app(data,request.method)
        return response     

    elif request.method == 'POST': 
        data = {}    
        try:
            body_str = request.body
            body_dict = json.loads(body_str)
            user = body_dict['username']
            pswd = body_dict['password']
            # print user,pswd
        except:
            response  = response_to_app(data,request.method)
            return response        
        users  = User.objects.filter(username = user)
        if len(users) > 0:         
            flag = users[0].check_password(pswd)
            if flag ==  True:        
                ret = "success"
                user0 = authenticate(username=user, password=pswd)  #nessary for check user
                login(request, user0)
                token = users[0].id
                message = u"欢迎登录系统！"
                user_type = users[0].username
                if user_type == 'gongchengdui':
                    data["usertype"] = 0
                else:
                    data["usertype"] = 1

            else:                
                ret = "fail"
                token = ''
                message =  u"对不起，你输入的密码有错，请重新输入。。。"
        else:
            ret = "fail"
            token = ''
            message = u"对不起，你输入的用户名有错，请重新输入。。。"
        station_names_list = []
        station_id_list    = []
        if ret == "success":
            if user_type == 'gongchengdui': 
                users = User.objects.filter(username = 'justin')
            else:         
                users = User.objects.filter(username = request.user) #will get a  list with one item
            auth_user_stations_list = Auth_user_stations.objects.filter(user_id = users[0].id)
            for auth_user_station in auth_user_stations_list:
                station_name_dict = {}
                station_list = Station.objects.filter(id = auth_user_station.station_id.id)
                station_name = station_list[0].name
                station_id   = station_list[0].id
                station_name_dict["name"] = station_name
                station_names_list.append(station_name_dict)
                station_id_list.append(station_id)
            data["stations"] = station_names_list
            #print "station_name:",station_names_list
        else:
            data["station"] = "" 

        if len(station_id_list) > 0:
            station_id = station_id_list[0]
            stations_1 = Station.objects.filter(id = station_id)
            company_id = stations_1[0].company_id
            company = Company.objects.filter(id = company_id.id)
            company_name = company[0].name
            data["company"] = company_name
        else:
            data["company"] = ""

        data['token']    = token         
        data["message"]  = message
        if ret == "success":
            print "---------------------"
            loggers = AlertappLogin.objects.filter(user_id = users[0].id)
            if len(loggers) < 1:
                logger = AlertappLogin()
                logger.user_id = user0
                logger.first_login = datetime.datetime.now()
                logger.last_login  = datetime.datetime.now()
                logger.is_paid = False
                logger.save()                
            else:
                logger = loggers[0]
                if logger.is_paid is False:
                    first_login = datetime.datetime.strptime(logger.first_login.strftime('%Y-%m-%d %H:%M:%S'),'%Y-%m-%d %H:%M:%S')
                    login_time = (datetime.datetime.now() - first_login).days
                    if login_time > 60:
                        ret = 'fail'
                    else:
                        logger.last_login = datetime.datetime.strptime(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'%Y-%m-%d %H:%M:%S')
                        logger.save()   
            print "----------------------"                             
        data["mystatus"] = ret
        # print "data = ",data
        response = response_with_json(data,request.method)          
        return response
    else: 
        data = "other method"
        response  = response_to_app(data,request.method)
        return response
'''
#query data from database named psrvdb by mysql language
def query_from_db(mysql):
    db       = MySQLdb.connect(host= 'localhost',user='root',passwd='hk1688',db='psrvdb',charset="utf8")
    cursor   = db.cursor()    
    cursor.execute(mysql)
    results  = cursor.fetchall()   
    cursor.close()
    db.close()
    return results
'''

def check_user(request):
    user = User.objects.filter(username = request.user)
    if user:
        flag = True
    else:
        flag = False
    return flag


def get_locat(request):
    print "============running in get_locat============"  
    if request.method == 'GET': 
        for key in request.GET:
            print key,request.GET[key]
        '''         
        user_flag = check_user(request)
        if user_flag == False:
            return response_to_app([],request.method)
        '''
        try:        
            stationName = request.GET.get("name") 
        except:
            response  = response_with_json({},request.method)
            return response #response nothing but without any server bug
        stations    = Station.objects.filter(name = stationName)  #only one item
        data = []           #response data
        if stations:
            t0_results  = T0.objects.filter(station_id = stations[0].id)
            if t0_results:
                locatDict   = {}
                locatList   = []
                for result in t0_results:
                    if result.locat not in locatList:
                        locatDict[result.locat] = 1         #number of the same
                        locatList.append(result.locat)   
                    else:
                        locatDict[result.locat] = locatDict[result.locat] + 1  
                for key in locatList:
                    tmpDict = {}
                    tmpDict["name"] = key
                    tmpDict["num"]  = locatDict[key]
                    data.append(tmpDict)
        response    = response_with_json(data,request.method)
        # print "data = ",data
        return response
    elif request.method == 'POST': 
        data = "post method"       
        response    = response_with_json(data,request.method)
        return response
    else:
        data = "other method"
        response    = response_with_json(data,request.method)
        return response

@csrf_exempt
def feedbackbody(request):
    if request.method == 'POST':
        '''
        user_flag = check_user(request)
        if user_flag == False:
            return response_to_app([],request.method) 
        '''    
        body_dict = json.loads(request.body)
        content = body_dict.get("content")
        access_token = mychat.getTokenIntime()
        head = "AlertApp Feedback Information From "+str(request.user)+":"+'\r\n'
        ret_json  = mychat.sendTxtMsg(access_token,head+content,0,"go2newera0006|go2newera0007")
        ret_dict = json.loads(ret_json)
        errmsg = ret_dict.get("errmsg")
        if errmsg == "ok":
            data = "success"
        else:
            data = "fail"
        response = response_with_json(data,request.method)
        return response
    else:
        data = "other method"
        response    = response_with_json(data,request.method)
        return response



def get_alert_information(request):
    '''
    报警数据和上报信息查询 数据
    T0.device_id,T0.name,T0.address,Alert.time,Alert.message 
    应该收到的get参数是{"station":"","locat":"","normal":""} normal = 1 正常 normal = 0 报警
    '''
    print "--------runing in get_alert_information()--------"
    if request.method == 'GET':
        for key in request.GET:
            print key,request.GET[key]
        print "---------------------------------------------------"
        import time
        t0 = time.time()  
        '''
        user_flag = check_user(request)
        if user_flag == False:
            return response_to_app([],request.method)
        '''  
        try:     #in case of getting no elements
           stationName  = request.GET.get("name")
           locatName    = request.GET.get("locat")
           flag         = request.GET.get("flag")
           number       = request.GET.get("number")
        except:
            response  = response_with_json({},request.method)
            return response #response nothing but without any server bug
        if (not stationName) or (not locatName) or (not flag):
            return response_with_json([],request.method)
        # locatName = u"外高桥"     #data for test
        # flag             = 1                  #data for test
        # t1 = time.time()
        # print "t1 - t0 = ",t1 -t0
        
        stations   = Station.objects.filter(name = stationName) #only one item

        t0_results = T0.objects.filter(station_id = stations[0].id,locat = locatName)      
        t2 = time.time()
        # print "t2 -t1 = ",t2 - t1 

        t0_len = len(t0_results)       
        N = 6
        number = int(number)
        start_index = number*N
        end_index = (number+1)*N
        if end_index < t0_len:
            t0_results = t0_results[start_index:end_index]
        elif end_index < (t0_len + N):
            t0_results = t0_results[start_index:]
        else:
            return response_with_json([],request.method)

        data = [] 
        flag = int(flag)
        for t0_obj in t0_results:
            if flag == 1:
                sql1 = "select * from A1 where t0_id = %d and normal = 1 order by time desc limit 1" % t0_obj.id
            else:
                sql1 = "select * from A1 where t0_id = %d and normal = 0 order by time desc limit 1" % t0_obj.id

            alert_obj = Alert.objects.raw(sql1)
            alert_obj = list(alert_obj)

            # alert_obj = Alert.objects.filter(t0_id = t0_obj.id,normal = flag)     
            if len(alert_obj) > 0:        
                data_dict = {}
                data_dict["device_id"]   = t0_obj.device_id
                data_dict["name"]        = t0_obj.name
                data_dict["address"]     = t0_obj.address
                data_dict["time"]        = alert_obj[0].time.strftime('%Y-%m-%d %H:%M:%S')             
                message_str              = alert_obj[0].message
                message_list  = message_str.split('|')
                data_dict["message"]     = message_list[-2]              
                data.append(data_dict)           
        t3 = time.time()
        # print "t3 - t2 = ",t3 - t2

        d_len = len(data)
        for i in range(0,d_len-1):
            for j in range(i,d_len): 
                if data[i]["time"] < data[j]["time"]: 
                    data[i],data[j] = data[j],data[i]

        t4 = time.time()
        # print "t4 - t3 = ",t4 - t3
        # print "t4 - t0 = ",t4 - t0
        # t5 = time.time()
    
        response  = response_with_json(data,request.method)
        return response
    elif request.method == 'POST':
        data     = "Your method is post"
        response = response_with_json(data,request.method)
        return response
    else:
        data      = "other method"
        response  = response_with_json(data,request.method)
        return response


def ZMD_countToName(zmd_count):
    PRESSURE_ORDER = {
        1:    str('出口压力'),
        2:    str('进口压力'),
        4:    str('差压'),
        8:    str('差压2'),
        16:   str('气体泄漏1'),
        32:   str('气体泄漏2'),
        64:   str('气体泄漏3'),
        128:  str('气体泄漏4'),
        256:  str('流量1'),
        512:  str('流量2'),
        1024: str('进口压力2')
        }
    names = []
    for key in PRESSURE_ORDER:
        if (key & zmd_count) != 0:
            name = PRESSURE_ORDER[key].encode('utf-8')
            names.append(PRESSURE_ORDER[key])
    ret = '+'.join(names)
    return ret

#@login_required
def get_device_infomation(request):
    '''
    获取设备信息 data  包括
    【设备编号(device_id)       名称(name)      地址(address)     设备安装状态(states)     厂商(product)     结构(structure)   运行气质(gas)  类型(type)     调压器型号(modelnum)
    出口压力(outpre)     进口压力(inpre)        出口下限(outdownlimit)       出口上限1(outuplimit1)      出口上限2(outlimit2)     出口上限3(outuplimit3)  进口下限(indownlimit)
    进口上限(inuplimit)    差压报警设定(predif)       安装时间(settime)      出口2压力      出口2下限     出口2上限】
    应该收到的get参数是 {"station":"","locat":""}

    '''
    print "--------runing in get_device_infomation()--------"   
    if request.method == 'GET':
        for key in request.GET:
            print key,request.GET[key]
        print "---------------------------------------------"
        '''
        user_flag = check_user(request)
        if user_flag == False:
            return response_to_app([],request.method) 
        '''
        import time 
        t0 = time.time()
        try:                  
            stationName  = request.GET.get("name")
            locatName    = request.GET.get("locat")
            number       = request.GET.get("number")
            #usertype     = int(request.GET.get("usertype"))
        except:
            return response_with_json([],request.method)
        if (not stationName) or (not locatName):
            return response_with_json([],request.method)
        # stationName  = u"虹口站"     #data for test
        # locatName    = u"宝山"
        data = []
        t1 = time.time()
        station_objs = Station.objects.filter(name = stationName)
        #privilige limit             
        t0_objs = T0.objects.filter(station_id = station_objs[0].id,locat = locatName)
      
        t0_len = len(t0_objs)       
        print "len(t0_objs) = ",t0_len
        N = 10
        number = int(number)
        start_index = number*N
        end_index = (number+1)*N
        if end_index < t0_len:
            t0_objs = t0_objs[start_index:end_index]
        elif end_index < (t0_len + N):
            t0_objs = t0_objs[start_index:]
        else:
            return response_with_json([],request.method)
        data = []
        for t in t0_objs:            
            data_dict = {}                   
            #print "latitude = ",t.latitude
            #print "longitude = ",t.longitude
            data_dict["device_id"]      = t.device_id             #设备编号  Y
            data_dict["name"]           = t.name                  #名称      Y
            data_dict["address"]        = t.address               #设备地址  Y
            data_dict["latitude"]       = t.latitude              #latitude  Y
            data_dict["longitude"]      = t.longitude             #longitude Y
            #data_dict["outpre"]         = t.exit_pmeter           #出口压力  Y
            data_dict["inpre"]          = t.inlet_pressure        #进口压力  Y
            if t.install_time:
                data_dict["time"]       = t.install_time.strftime('%Y-%m-%d %H:%M:%S')             #安装时间  Y
            else:
                data_dict["time"]       = t.install_time      #此时安装时间字段时空值
            data_dict["structure"]      = t.structure         #结构       Y
            data_dict["modelnum"]       = t.modem             #modem 种类 Y
            #data_dict["gas"]            = t.is_artifacial_gas #运行气质
            data_dict["product"]        = t.manufactory       #厂商         Y
            zmd_count = t.ZMD_count          
            data_dict["states"]         = ZMD_countToName(zmd_count)         #变送器安装情况 Y
            data_dict["outpre"]         = t.exit_presure     #出口压力       Y
            data_dict["type"]           = t.tiaoyaqi_model    #类型     
            data_dict["outdownlimit"]   = t.up_value1  #出口下限    Y
            data_dict["outuplimit1"]    = t.up_value2  #出口上限1   Y
            data_dict["outuplimit2"]    = t.up_value3  #出口上限2   Y
            data_dict["outuplimit3"]    = t.up_value4  #出口上限3   Y
            data_dict["inuplimit"]      = t.h_high     #进口上限    Y
            data_dict["indownlimit"]    = t.h_low      #进口下限    Y
            data_dict["predif"]         = t.d_range    #差压量程表
            data_dict["other"]          = t.other
            #if usertype == 0:
            #    try:
            #        sql1 = "select * from A1 where t0_id = %d and normal = 1 order by time desc limit 1" % t.id
            #        alert_obj = Alert.objects.raw(sql1)
            #        alert_obj = list(alert_obj)
            #        message_str              = alert_obj[0].message
            #        message_list  = message_str.split('|')
            #        data_dict["message"]     = message_list[-2]     #最新上报   Y        
            #    except Exception,ex:
            #        print "Exception happens:",ex
            #        data_dict["message"]         = '' 
            #else:
            #    data_dict["message"] = ''
            data.append(data_dict)       
            t.save()   
        # print "this is before response num of datalist",len(data)
        # t2 = time.time()
        # print "t1 - t0 = ",t1 - t0
        # print "t2 - t1 = ",t2 - t1
        # print "data is :",len(data)
        response  = response_with_json(data,request.method)
        return response
    elif request.method =='POST':
        data      = "it is post request"
        response  = response_with_json(data,request.method)
        return response
    else:
        data      = "it is other request"
        response  = response_with_json(data,request.method)
        return response

def string_to_datetime(string):
    '''
    Transform time string received from app to python datetime.datetime object 
    input string like "2014-12-31T16:00:00.000Z"
    output datetime.datetime object
    '''
    index1 = string.find('T')
    index2 = string.find('Z')
    str1 = string[0:index1]
    str2 = string[index1+1:index2-4]
    form = '%Y-%m-%d %H:%M:%S'
    str3 = str1 + ' ' + str2
    time_tmp = datetime.datetime.strptime(str3,form)
    t = datetime.timedelta(hours = 8)
    result = time_tmp + t
    return result

#@login_required
def get_for_chart(request):
    '''
    响应图像数据查询，查询P0表的内容
    应该受到的get参数{"device_id":""}
    '''
    print "================runing in get_for_chart()================"  
    if request.method == 'GET':
        for key in request.GET:
            print key,request.GET[key]
        print "------------------------------------------------------"
        '''
        user_flag = check_user(request)
        if user_flag == False:
            return response_to_app([],request.method)    
        ''' 
        device_id = request.GET.get("device_id")
        start_time_string= str(request.GET.get("start_date"))
        end_time_string = str(request.GET.get("end_date"))
        try:
            start_time = string_to_datetime(start_time_string)
            delta = datetime.timedelta(hours = 24)
            end_time  = string_to_datetime(end_time_string) #接收到的截止时间是0点
            end_time = end_time + delta                     #实际上截止时间应该是24点才对
        except: 
            if end_time_string != '0':
                f = "%Y-%m-%d %H:%M:%S"                      
                end_time = datetime.datetime.now()                                #while running on polarwin 此处需要特别注意
                utc_time = datetime.timedelta(hours = 8)        #在django里面此时间需要加上八个小时才是当地时间
                end_time = end_time + utc_time
                delta = datetime.timedelta(days = 1)                                    #default:return one day's data 24 hours
                t = end_time - delta
                s1 = t.strftime('%Y-%m-%d')
                s2 = '00:00:00'
                s = s1 +' ' + s2
                start_time = datetime.datetime.strptime(s,f) 
            else:
                start_time = string_to_datetime(start_time_string)
                delta = datetime.timedelta(hours = 24)
                end_time = start_time + delta  #before exception happenned on try
                                                #start_time get 
        if start_time > datetime.datetime.now():
            response_with_json({},request.method)       
        if not device_id:          
            device_id = '00019131' 
        need_time = end_time - start_time
        need_days = need_time.days         
        data = {} 
        t0_objs = T0.objects.filter(device_id = device_id)        #一定是只有一个元素的列表  
        report_interval = t0_objs[0].report_interval    
        if t0_objs:
            s_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
            e_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
            p0_objs = P0.objects.filter(t0_id = t0_objs[0].id).filter(time__gt = s_time).filter(time__lt = e_time)
            p0_nums = len(p0_objs)  
            if p0_nums == 0:
                data["date"]          = [] 
                data["time"]          = []
                data["outpre"]        = []   #出口压力
                data["inpre"]         = []       #进口压力           
                data["address"]       = t0_objs[0].address
                data["device_id"]     = device_id
                # print "data = ",data
                return response_with_json(data,request.method)
            address = t0_objs[0].address

            nums_len = 144
            n = p0_nums/nums_len
            # nums_len = p0_nums/n     #数据数量           
            p0_list_tmp = []
            for i in range(p0_nums):
                if n*i < p0_nums:
                    p0_tmp = p0_objs[n*i]
                    p0_list_tmp.append(p0_tmp)
                else:
                    break
            p0_objs = p0_list_tmp          
            #返回数据类型封装处理，根据前端的需要返回适当地数据类型          
            times = []
            dates = []
            values_outpre = []
            values_inpre = []      
            for p in p0_objs:               
                tmp1 = p.time.strftime('%m-%d %H:%M')
                tmp_date = p.time.strftime('%m-%d')
                tmp2 = p.valve_pressure1      #出口压力
                tmp3 = p.high_pressure          #进口压力
                tmp4 = p.diff_pressure            #差压
                times.append(tmp1)
                values_outpre.append(tmp2)
                values_inpre.append(tmp3)
                dates.append(tmp_date)
                # values_predif.append(tmp4)
            date1 = start_time.strftime('%Y-%m-%d')
            date2 = end_time.strftime('%Y-%m-%d')             
            times.reverse()       #将数据倒叙
            dates.reverse()
            values_outpre.reverse()
            values_inpre.reverse()  
            
            #返回的时间是日期加时间，分天封装为数组
            times_result = []
            times_list   = []            
            for i in range(1,len(times)):
                if dates[i] != dates[i-1]:
                    times_result.append(times_list)
                    times_list = []
                    times_list.append(times[i])
                else:
                    times_list.append(times[i])
            times_result.append(times_list)  
            
            #返回数据
            up_value1 = []
            up_value2 = []
            value_length = len(values_outpre)          
            for i in range(value_length):
                up_value1.append(t0_objs[0].up_value1)     #压力过低预警
                up_value2.append(t0_objs[0].up_value2)       #放散压力预警                
            data["date"]          = date1 + '~' + date2  
            data["time"]          = times_result
            data["outpre"]        = [values_outpre,up_value1,up_value2]   #出口压力
            data["inpre"]         = [values_inpre]       #进口压力           
            data["address"]       = address
            data["device_id"]     = device_id 

        # print "data['address'] = ",data['address']
        # print "data['device_id'] = ",data["device_id"]
        response  = response_with_json(data,request.method)
        return response
    elif request.GET =='POST':
        data      = "it is post request"
        response  = response_with_json(data,request.method)
        return response
    else:
        data      = "it is other request"
        response  = response_with_json(data,request.method)
        return response

#@login_required
def alert_search(request):
    '''
    满足报警信息全局查询 get参数{"flag","type":"","content"}   
    flag命令{"0":报警信息,"1","上报信息","2","设备信息"}
    type命令{"0:":"设备编号","1":"设备名称","2":"设备地址"}
    content上传内容
    '''
    print "--------runing in alert_search()--------"
    if request.method == 'GET':
        '''
        user_flag = check_user(request)
        if user_flag == False:
            return response_to_app([],request.method)
        '''       
        try:
            cmd1 = int(request.GET.get("flag"))      #cmd1 = flag
            cmd2 = int(request.GET.get("type"))      #cmd2 = type
            content = request.GET.get("content").encode("utf-8")
            station = request.GET.get('station').encode('utf-8')
            locat = request.GET.get("locat").encode("utf-8")  
            print "cmd1 = %d,cmd2 = %d,content = %s,station = %s,locat = %s" % (cmd1,cmd2,content,station,locat)
        except:
            return response_with_json([],request.method)
        #cmd1 = 1       #saved for test
        #cmd2 =  1
        #content = "共和二村64号"  
        stations    = Station.objects.filter(name = station)  
        data = []
        if (cmd1 ==0) or (cmd1== 1):           #查询报警信息cmd1=0和上报信息cmd=1
            if cmd2 == 0:   #已知设备编号
                t0_objs = T0.objects.filter(station_id = stations[0].id).filter(locat = locat).filter(Q(device_id__contains = content))
                print "len(t0_objs) --------1",len(t0_objs)
            elif cmd2 == 1: #已知设备名称           
                t0_objs = T0.objects.filter(station_id = stations[0].id).filter(locat = locat).filter(Q(name__contains = content))
            else:                #cmd2 = 2已知设备地址
                t0_objs = T0.objects.filter(station_id = stations[0].id).filter(locat = locat).filter(Q(address__contains = content))
            print "t0_objs is --------",len(t0_objs)
            if len(t0_objs) < 1:
                return response_with_json([],request.method)
            alerts = []
            for t0_obj in t0_objs:
                alerts = Alert.objects.filter(t0_id = t0_obj.id,normal = cmd1)
            print "alerts are --------",len(alerts)
            # if len(alerts) >= limitNum:
            #     alerts = alerts[0:limitNum]
            
            for alert_obj in alerts:
                data_dict = {}
                data_dict["device_id"]    = t0_objs[0].device_id
                data_dict["name"]         = t0_objs[0].name
                data_dict["address"]      = t0_objs[0].address
                data_dict["time"]         = alert_obj.time.strftime(form)
                # print "alert time is --------",alert_obj.time
                message_str               = alert_obj.message
                message_list  = message_str.split('|')
                data_dict["message"]      = message_list[-2]
                # print "message is --------",message_list[-2]
                data.append(data_dict)
        else:    #查询设备信息cmd1 =2
            if cmd2 == 0:                  #已知设备编号
                t0_objs = T0.objects.filter(station_id = stations[0].id).filter(locat = locat).filter(Q(device_id__contains = content))
            elif cmd2 ==  1:              #已知设备名称
                t0_objs = T0.objects.filter(station_id = stations[0].id).filter(locat = locat).filter(Q(name__contains = content))
            else:                                #cmd2 = 2已知设备地址
                t0_objs = T0.objects.filter(station_id = stations[0].id).filter(locat = locat).filter(Q(address__contains = content))
            print "len(t0_objs) -----------",len(t0_objs)
            if t0_objs < 1:
                return response_with_json([],request.method)
            for t in t0_objs:
                data_dict = {}                   
                data_dict["device_id"]      = t.device_id             #设备编号  Y
                data_dict["name"]           = t.name                  #名称      Y
                data_dict["address"]        = t.address               #设备地址  Y
                data_dict["latitude"]       = t.latitude              #latitude  Y
                data_dict["longitude"]      = t.longitude             #longitude Y
                data_dict["outpre"]         = t.exit_pmeter           #出口压力  Y
                data_dict["inpre"]          = t.inlet_pressure        #进口压力  Y
                if t.install_time:
                    data_dict["time"]       = t.install_time.strftime('%Y-%m-%d %H:%M:%S')             #安装时间  Y
                else:
                    data_dict["time"]       = t.install_time      #此时安装时间字段时空值
                data_dict["structure"]      = t.structure         #结构       Y
                data_dict["modelnum"]       = t.modem             #modem 种类 Y
                #data_dict["gas"]            = t.is_artifacial_gas #运行气质
                data_dict["product"]        = t.manufactory       #厂商         Y
                zmd_count = t.ZMD_count           
                data_dict["states"]         = ZMD_countToName(zmd_count)         #变送器安装情况 Y
                data_dict["outper"]         = t.exit_presure     #出口压力       Y
                #data_dict["type"]           = t.tiaoyaqi_model    #类型     
                data_dict["outdownlimit"]   = t.up_value1  #出口下限    Y
                data_dict["outuplimit1"]    = t.up_value2  #出口上限1   Y
                data_dict["outuplimit2"]    = t.up_value3  #出口上限2   Y
                data_dict["outuplimit3"]    = t.up_value4  #出口上限3   Y
                data_dict["inuplimit"]      = t.h_high     #进口上限    Y
                data_dict["indownlimit"]    = t.h_low      #进口下限    Y
                data_dict["predif"]         = t.d_range    #差压报警   差压量程表
                data_dict["other"]          = t.other
                sql1 = "select * from A1 where t0_id = %d and normal = 1 order by time desc limit 1" % t.id
                alert_obj = Alert.objects.raw(sql1)
                alert_obj = list(alert_obj)
                message_str              = alert_obj[0].message
                message_list  = message_str.split('|')
                data_dict["message"]     = message_list[-2]     #最新上报   Y         
                data.append(data_dict)                    
        print "len(data) = ",len(data)                
        # print "data :",data
        response  = response_with_json(data,request.method)
        return response
    elif request.method =='POST':
        data = "it is post request"
        response  = response_with_json(data,request.method)
        return response
    else:
        data = "it is other request"
        response  = response_with_json(data,request.method)
        return response


# get position(latitude,longitude) from app
@csrf_exempt
def get_position(request):
    print "===================running in update_position==========================="
    if request.method == 'POST':
        import torndb
        for key in request.POST:
            print key,request.POST[key]
        print '--------------------------------------------------------------'
        # print "request.body:",request.body
        '''
        user_flag = check_user(request)
        if user_flag == False:
            return response_to_app([],request.method)
        '''
        try:
            body_dict = json.loads(request.body)
            device_id = str(body_dict.get("device_id"))
            latitude  = float(body_dict.get("latitude"))
            longitude = float(body_dict.get("longitude"))
        except Exception,ex:
            print "Exception happens:",ex
            return resopnse_with_json('fail',request.method)

        # print "latitude = ",latitude
        # print "longitude = ",longitude
        # print "device_id = ",device_id
        '''
        try:            
            if latitude != 0:          
                hostadderss = '120.26.105.20:3306'
                database = 'T0tables'
                #database = 'psrvdb'
                user = 'justin'
                password = 'hk1688'
                db = torndb.Connection(hostadderss,database,user,password)
                sql = "update T0 set latitude = %d,longitude = %d where device_id = %s" % (latitude,longitude,device_id)
                db.execute(sql)
                # T0tablesOperator.db.execute(sql)
                db.close()
                data = "success"
            else:
                data = "fail"
        except Exception,ex:
            print "Exception happens:",ex 
            data = "fail"
        '''
        try:
            if latitude != 0:
                t0_objs = T0.objects.filter(device_id = device_id)
                t0_obj = t0_objs[0]
                t0_obj.latitude = latitude
                t0_obj.longitude = longitude
                t0_obj.save()
                data = "success"
        except Exception,ex:
            print "Exception:",ex
            data = "fail"
        return response_with_json(data,request.method)

    else:
        return response_with_json("It is ok!",request.method)

