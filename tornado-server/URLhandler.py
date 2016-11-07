#!/user/bin/env/python
#coding:utf-8

import json
import datetime
import tornado.web
from tornado import gen
import dbOperator
import toolFunction
from mychat import mychat

class BaseHandler(tornado.web.RequestHandler):

    db_psrvdb = dbOperator.psrvdbOperatorDB
    # db_T0tables = dbOperator.T0tablesOperator
    def setGetRequestHeader(self):
    	'''deal with cross domain trouble '''
        self.set_header('Access-Control-Allow-Origin','*')
        self.set_header('Access-Control-Allow-Methods',self.request.method)
        self.set_header('Access-Control-Allow-Headers',"x-requested-with,content-type")    

    def setPostRequestHeader(self):
    	'''deal with cross domain trouble '''
        self.set_header('Access-Control-Allow-Origin','*')
        self.set_header('Access-Control-Allow-Methods',self.request.method)
        self.set_header('Access-Control-Allow-Headers',"x-requested-with,content-type")

    def get_current_user(self):
    	if self.get_secure_cookie('user'):
            return self.get_secure_cookie('user')
        else:
            return 'NoUser'

    
    def check_user(self):
        username = self.get_current_user('user')
        sqlAuthUser = "select * from auth_user where username ='%s'" % (username)
        userList = self.db_psrvdb.select(sqlAuthUser)
        flag = False
        for U in userList:
            if username == U["username"]:
                flag = True
                break
        return flag

'''登录验证  使用post方法，接受username和password参数'''
class LoginHandler(BaseHandler):
    def get(self):
        self.setGetRequestHeader()                	 
     	data = {"name":"chenlb","password":"chenliangbohk1688"}
    	data = json.dumps(data,separators = (',',':'))  	
        print "----------------------------------------"
        self.write(data)

    def post(self):
        print "---------------------------running in login post--------------------------------"
        data = {}
        self.setPostRequestHeader() 
        try:
            bodyString = self.request.body
            bodyDict = json.loads(bodyString)
            username = bodyDict["username"]
            password = bodyDict['password']
            print "username = '%s',password = '%s'" % (username,password)
        except Exception,ex:
            print "Exception happens when get data in login post:",ex
            self.write(data)
            return        
        sqlAuthUser = "select id,username,password from auth_user where username = '%s'" % (username,)
        auth_user_list = self.db_psrvdb.select(sqlAuthUser)
        auth_user = auth_user_list[0]     
        # print "auth_user = ",auth_user
        if username == auth_user["username"]:
            message = "欢迎你!"
            data["mystatus"] = 'success'
            data["token"]    = auth_user["id"]
            sqlUserStation = "select * from auth_user_stations where user_id = '%s'" % (auth_user['id'],)
            station_ids = self.db_psrvdb.select(sqlUserStation)    # list dict
            # print "station_ids = ",station_ids
            data["stations"] = []         
            for d in station_ids:
                dataTmpDict = {}                
                sqlStation = "select * from Station where id = %d" % (int(d["station_id"]),)
                stationList = self.db_psrvdb.select(sqlStation)
                station = stationList[0]
                dataTmpDict["name"] = station['name']
                data["stations"].append(dataTmpDict) 
            # print "stqtion = ",station
            sqlCompany = 'select * from Company where id = %d' % (int(station['Company ID']))        
            companyList = self.db_psrvdb.select(sqlCompany)
            # print "companyList = ",companyList[0]['name']
            data['company'] = companyList[0]['name']
            self.set_secure_cookie('user',username)
            if username == 'justin':
                data["usertype"] = 0
            else:
                data["usertype"] = 1
        else:
            data['stations'] = ''
            data["usertype"] = 1
            data['company'] = ''          
            data["mystatus"] = 'fail'
            data['token'] = ''
            message = '对不起，你输入的用户名不存在，请重新输入 ...'        
        data["message"] = message
        # print "data = ",data
        data = json.dumps(data,separators = (',',':'))  
        self.write(data)

'''站点查询 接收站点名称name参数'''
class StationsHandler(BaseHandler):
    def get(self):
        user = self.get_current_user()
        print "user ---------------",user
        self.setGetRequestHeader()
        data = {}
        print "------------------------running in StationsHandler get -----------------------"
        try:   
            stationName = self.get_argument('name')
            print "stationName = ",stationName
        except Exception,ex:
            print "Exception happens in StationsHandler get:",ex
            self.write(data)
            return
        sqlStation = "select * from Station where name = '%s'" % (stationName,)
        stationList = self.db_psrvdb.select(sqlStation)
        # print "stationList = ",stationList
        if len(stationList) < 1:
            self.write(data)
            return
        sqlT0 = "select locat as name,count(*) AS num from T0 where station_id = %d group by locat" % (int(stationList[0]["id"]),)     
        T0List = self.db_psrvdb.select(sqlT0)     #T0List = [{"name":,"num":,},{},{}]
        for T0 in T0List:     
            T0['num'] = int(T0["num"])  
            T0["name"].encode('UTF-8')      
        data = T0List
        print "data = ",data
        data = json.dumps(data,separators = (',',':'))
        self.setGetRequestHeader()
        self.write(data)

    def post(self):
        pass

class AlertHandler(BaseHandler):
    def post(self):
        pass
    def get(self):
        user = self.get_current_user()
        print "user ---------------",user
        self.setGetRequestHeader()
        try:
            stationName = str(self.get_argument('name'))
            locatName   = str(self.get_argument('locat'))
            flag        = int(self.get_argument('flag'))
            number      = int(self.get_argument('number'))
        except Exception,ex:
            print "Exception happens in AlertHandler get:",ex
            self.write({})
            return 
        data = []
        print "stationName = '%s',locatName = '%s',flag = %d,number = %d" % (stationName,locatName,flag,number)
        sqlStation = "select * from Station where name = '%s'" % (stationName,)
        stationList = self.db_psrvdb.select(sqlStation)
        # print "stationList = ",statiT0onList
        stationDict = dict(stationList[0])
        # print "stationDict = ",stationDict
        sqlT0 = "select id,device_id,name,locat,address from T0 where station_id = %d and locat = '%s' order by id " % (int(stationDict["id"]),locatName)
        T0List = self.db_psrvdb.select(sqlT0)
        # print "T0List = ",T0List
        N = 6                    #limit number of data 
        t0_len = len(T0List)
        start_index = number*N
        end_index = (number+1)*N
        if end_index < t0_len:
            T0List = T0List[start_index:end_index]
        elif end_index < (t0_len + N):
            T0List = T0List[start_index:]
        else:
            self.write({})
            return
        print "len(T0List) = ",len(T0List)
        for T0 in T0List:
            T0 = dict(T0)
            if flag == 1:
                sqlAlert = "select * from A1 where t0_id = %d and normal = 1 order by time desc limit 1" % (int(T0["id"]),)
            else:
                sqlAlert = "select * from A1 where t0_id = %d and normal = 0 order by time desc limit 1" % (int(T0["id"]),)
            alertList = self.db_psrvdb.select(sqlAlert)     
            if len(alertList) > 0:
                alertDict = dict(alertList[0])
                dataDict = {}
                dataDict["device_id"] = T0["device_id"]
                dataDict["name"]      = T0["name"]
                dataDict["address"]   = T0["address"]
                dataDict["message"]   = alertDict["message"].split('|')[-2]
                dataDict["time"]      = alertDict["time"].strftime('%Y-%m-%d %H:%M:%S')
                # print "dataDict['time'] = ",type(dataDict["time"])
                data.append(dataDict)
        d_len = len(data)              #order by time desc
        for i in range(0,d_len-1):
            for j in range(i,d_len): 
                if data[i]["time"] < data[j]["time"]: 
                    data[i],data[j] = data[j],data[i]
        # print "data = ",data
        data = json.dumps(data,separators = (',',':'))
        ret = {"data":data} 
        # print "ret = ",ret       
        self.write(ret)

class DeviceHandler(BaseHandler):
    def get(self):
        user = self.get_current_user()
        print "user ---------------",user
        self.setGetRequestHeader()
        try:
            stationName = self.get_argument("name")
            locatName   = self.get_argument("locat")
            number      = int(self.get_argument("number"))
        except Exception,ex:
            print "Exception happens in DeviceHandler get:",ex
            self.write({})
            return
        print "stationName = '%s',locatName = '%s',number = %d" % (stationName,locatName,number)
        sqlStation = "select * from Station where name = '%s'" % (stationName,)
        stationList = self.db_psrvdb.select(sqlStation)
        stationDict = dict(stationList[0])        
        sqlT0 = "select device_id,name,address,latitude,longitude,inlet_pressure as inpre,install_time as time,structure,\
        modem as modelnum,manufactory as product,ZMD_count as states,exit_presure as outpre,tiaoyaqi_model as type,\
        up_value1 as outdownlimit,up_value2 as outuplimit1,up_value3 as outuplimit2,up_value4 as outuplimit3,h_high as inuplimit,\
        h_low as indownlimit,d_range as predif,other from T0 where station_id = %d and locat = '%s'" % (stationDict["id"],locatName)
        
        # sqlT0 = "select * from T0 where station_id = %d and locat = '%s'" % (stationDict["id"],locatName)
        T0List = self.db_psrvdb.select(sqlT0)
        # print "len(T0List) = ",len(T0List)  

        N = 10    # number limit 
        t0_len = len(T0List)   
        start_index = number*N
        end_index = (number+1)*N
        if end_index < t0_len:
            T0List = T0List[start_index:end_index]
        elif end_index < (t0_len + N):
            T0List = T0List[start_index:]
        else:
            self.write({})
            return
        print "len(T0List) = ",len(T0List)
        for T0 in T0List:                       
            if T0['time']:
                time = T0['time'].strftime('%Y-%m-%d %H:%M:%S')
                T0["time"] = time      
            T0["states"] = toolFunction.ZMD_countToName(T0["states"])
        # for d in T0List[0]:
        #     print (d,T0List[0][d],type(T0List[0][d]))
        ret = {"data":T0List}
        ret = json.dumps(ret,separators = (',',':')) 
        # print "ret = ",ret       
        self.write(ret)


class ChartHandler(BaseHandler):    
    '''返回的数据格式{"errmsg":'ok'/'erro','data':data},data is dict
       data.keys() = ['time', 'address', 'date', 'outpre', 'inpre', 'device_id']
    '''
    def get(self):
        self.setGetRequestHeader()
        flag = 0
        try:
            device_id = self.get_argument("device_id")
            flag = 1
            start_time_string= str(self.get_argument("start_date"))
            end_time_string = str(self.get_argument("end_date"))
            flag = 2        
        except Exception,ex:
            print "Exception happens in ChartHandler  get:",ex
            if flag == 1: 
                print "flag = 1"      
                '''只收到device_id，说明是默认情况下，返回昨天0点到现在数据'''       
                end_time = datetime.datetime.now()
                delta = datetime.timedelta(days = 1)                                    #default:return one day's data 24 hours
                t = end_time - delta
                s = t.strftime('%Y-%m-%d') +' ' + '00:00:00'             
                start_time = datetime.datetime.strptime(s,"%Y-%m-%d %H:%M:%S")
                print "start_time = ",start_time
                print "end_time = ",end_time
            elif flag == 0:
                print "flag = 0"
                '''连device_id都没收到直接是错误，返回错误信息'''
                ret = {'errmsg':'erro',"data":[]}
                ret = json.dumps(ret,separators = (',',':'))
                self.write(ret)
                return 
            else:
                '''理论上说此处程序不可能执行flag=2就不是异常了'''
                pass
        if flag == 2:
            print "flag = 2"
            '''是按照时间的查询，三个参数都收到，有两种情况'''
            if end_time_string == '0':
                '''圆图查询，只查询一天的数据24小时，返回起始日期所在的当天数据'''
                print "start_time_string = ",start_time_string
                start_time = toolFunction.string_to_datetime(start_time_string)
                delta = datetime.timedelta(hours = 24)
                end_time = start_time + delta  #before exception happenned on try 
            else:
                '''折线图多天查询，返回多天数据'''
                start_time = toolFunction.string_to_datetime(start_time_string)
                delta = datetime.timedelta(hours = 24)
                end_time  = toolFunction.string_to_datetime(end_time_string) #接收到的截止时间是0点
                end_time = end_time + delta                     #实际上截止时间应该是24点才对    
        s_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
        e_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
        print "--------------------------------------"
        if start_time > datetime.datetime.now():
            ret = {'errmsg':'erro',"data":[]}
            ret = json.dumps(ret,separators = (',',':'))
            self.write({})
            return
        if not device_id:          
            device_id = '00019131'
        sqlT0 = "select * from T0 where device_id = '%s'" % (device_id,)
        T0List = self.db_psrvdb.select(sqlT0)
        if len(T0List) < 1:
            ret = {'errmsg':'erro',"data":[]}
            ret = json.dumps(ret,separators = (',',':'))
            self.write(ret)
            return
        T0Dict = T0List[0]
        data = {}
        data["report_interval"] = T0Dict["report_interval"]
        data["address"]         = T0Dict["address"]
        data["device_id"]       = device_id     
        sqlP0 = "select * from P0 where t0_id = %d and time > '%s' and time < '%s'" % (T0Dict['id'],s_time,e_time)
        P0List = self.db_psrvdb.select(sqlP0)

        p0_nums = len(P0List)
        nums_len = 144
        n = p0_nums/nums_len                
        p0_list_tmp = []
        for i in range(p0_nums):
            if n*i < p0_nums:
                p0_tmp = P0List[n*i]
                p0_list_tmp.append(p0_tmp)
            else:
                break
        P0List = p0_list_tmp
        times = []
        dates = []
        values_outpre = []
        values_inpre = []
        for p in P0List:
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
            up_value1.append(T0Dict["up_value1"])     #压力过低预警
            up_value2.append(T0Dict["up_value2"])     #放散压力预警  
        data["date"]          = date1 + '~' + date2 
        data["time"]          = times_result
        data["outpre"]        = [values_outpre,up_value1,up_value2]   #出口压力
        data["inpre"]         = [values_inpre]       #进口压力          
        # ret = {"errmsg":"ok","data":data}
        data = json.dumps(data,separators = (',',':'))
        self.write(data)



'''update latitude and longtitude into T0tables database's T0 table'''
class UpdatePositionHandler(BaseHandler):
    def get(self):
        self.setGetRequestHeader()
        data = {"errmsg":'ok',"data":[]}
        data = json.dumps(data,separators = (',',':'))
        self.write(data)
    def post(self):        
        self.setPostRequestHeader()
        try:
            bodyString = self.request.body
            print "bodyString = ",bodyString
            bodyDict = json.loads(bodyString)
            device_id = str(bodyDict["device_id"])
            latitude = float(bodyDict["latitude"])
            longitude = float(bodyDict["longitude"])
        except Exception,ex:
            print "Exception happens in UpdatePositionHandler post:",ex
            self.write({})
            return     
        sqlT0 = "update T0 set latitude = %d,longitude = %d where device_id = '%s'" % (latitude,longitude,device_id)
        flag = self.db_psrvdb.update(sqlT0)
        if flag:
            data = 'success'
            print "update successfully!"
        else:
            data = 'fail'
        data = json.dumps(data,separators = (',',':'))
        self.write(data)

class FeedbackHandler(BaseHandler):
    def get(self):
        self.setGetRequestHeader()
        data = {"errmsg":'ok',"data":[]}
        data = json.dumps(data,separators = (',',':'))
        self.write(data)
    def post(self):        
        self.setPostRequestHeader()
        try:
            bodyString = self.request.body
            bodyDict = json.loads(bodyString)
            content = bodyDict["content"]
        except:
            print "Exception happens in FeedbackHandler post:",ex
            self.write(data)
            return
        access_token = mychat.getTokenIntime()
        head = "AlertWechatWeb Feedback Information From  "+str(self.get_current_user())+":"+'\r\n'
        txtJson = mychat.sendTxtMsg(access_token,head+content,'go2newera0006')
        txtDict = json.loads(txtJson)     #{u'errcode': 0, u'errmsg': u'ok'}
        print "txt_dict = ",txtDict        
        if txtDict["errmsg"] == "ok":
            data = "success"
        else:
            data = "fail"
        self.write(data)


class AlertSearchHandler(BaseHandler):
    '''
    满足报警信息全局查询 get参数{"flag","type":"","content"}    content上传内容   
    flag命令{"0":报警信息,"1","上报信息","2","设备信息"}
    type命令{"0:":"设备编号","1":"设备名称","2":"设备地址"}   
   '''
    def get(self):
        self.setGetRequestHeader()
        try:
            cmd1 = int(self.get_argument("flag"))
            cmd2 = int(self.get_argument("type"))
            content = self.get_argument("content")
            content = content.encode("utf-8")
        except Exception,ex:
            print "Exception:",ex
            data = {"errmsg":str(ex)}
            data = json.dumps(data,separators = (',',':'))
            self.write(data)
            return
        data = []
        if (cmd1 ==0) or (cmd1== 1):           #查询报警信息cmd1=0和上报信息cmd=1
            if cmd2 == 0:   #已知设备编号
                sqlT0 = "select * from T0 where device_id = '%s'" % (content,)          
            elif cmd2 == 1: #已知设备名称           
                t0_objs = T0.objects.filter(name = content)
                sqlT0 = "select * from T0 where name = '%s'" % (content,)
            else:                #cmd2 = 2已知设备地址
                t0_objs = T0.objects.filter(address = content)
                sql = "select * from T0 where address = '%s'" % (content,)
            # print "t0_objs is --------",t0_objs
            T0List = self.db_psrvdb.select(sqlT0)
            print "len(T0List) = ",len(T0List)
            if len(T0List) < 1:
                data = {"errmsg":"No Alert Information Found"}
                data = json.dumps(data,separators = (',',':'))
                self.write(data)
                return
            for T0 in T0List:
                T0 = dict(T0)
                if cmd1 == 1:
                    sqlAlert = "select * from A1 where t0_id = %d and normal = 1 order by time desc limit 1" % (int(T0["id"]),)
                else:
                    sqlAlert = "select * from A1 where t0_id = %d and normal = 0 order by time desc limit 1" % (int(T0["id"]),)
                alertList = self.db_psrvdb.select(sqlAlert)
                print "len(alertList) = ",len(alertList)       
                if len(alertList) > 0:
                    alertDict = dict(alertList[0])
                    dataDict = {}
                    dataDict["device_id"] = T0["device_id"]
                    dataDict["name"]      = T0["name"]
                    dataDict["address"]   = T0["address"]
                    dataDict["message"]   = alertDict["message"].split('|')[-2]
                    dataDict["time"]      = alertDict["time"].strftime('%Y-%m-%d %H:%M:%S')
                    # print "dataDict['time'] = ",type(dataDict["time"])
                    data.append(dataDict)
            ret = {"data":data}
            ret = json.dumps(ret,separators = (',',':'))
            print "ret = ",ret
            self.write(ret)
            return
        else:
            if cmd2 == 0:#已知设备编号                
                sqlT0 = "select device_id,name,address,latitude,longitude,inlet_pressure as inpre,install_time as time,structure,\
                modem as modelnum,manufactory as product,ZMD_count as states,exit_presure as outpre,tiaoyaqi_model as type,\
                up_value1 as outdownlimit,up_value2 as outuplimit1,up_value3 as outuplimit2,up_value4 as outuplimit3,h_high as inuplimit,\
                h_low as indownlimit,d_range as predif,other from T0 where device_id = '%s'" % (content,)
            elif cmd2 ==  1:#已知设备名称             
                sqlT0 = "select device_id,name,address,latitude,longitude,inlet_pressure as inpre,install_time as time,structure,\
                modem as modelnum,manufactory as product,ZMD_count as states,exit_presure as outpre,tiaoyaqi_model as type,\
                up_value1 as outdownlimit,up_value2 as outuplimit1,up_value3 as outuplimit2,up_value4 as outuplimit3,h_high as inuplimit,\
                h_low as indownlimit,d_range as predif,other from T0 where name = '%s'" % (content,)
            else:#cmd2 = 2已知设备地址            
                sqlT0 = "select device_id,name,address,latitude,longitude,inlet_pressure as inpre,install_time as time,structure,\
                modem as modelnum,manufactory as product,ZMD_count as states,exit_presure as outpre,tiaoyaqi_model as type,\
                up_value1 as outdownlimit,up_value2 as outuplimit1,up_value3 as outuplimit2,up_value4 as outuplimit3,h_high as inuplimit,\
                h_low as indownlimit,d_range as predif,other from T0 where address = '%s'" % (content,)
            
            T0List = self.db_psrvdb.select(sqlT0)
            if len(T0List) < 0:
                data = {"errmsg":"No Device Found"}
                data = json.dumps(data,separators = (',',':'))
                self.write(data)
                return     
            for T0 in T0List:                       
                if T0['time']:
                    time = T0['time'].strftime('%Y-%m-%d %H:%M:%S')
                    T0["time"] = time      
                T0["states"] = toolFunction.ZMD_countToName(T0["states"])
            ret = {"data":T0List}
            ret = json.dumps(ret,separators = (',',':')) 
            # print "ret = ",ret       
            self.write(ret)
            return