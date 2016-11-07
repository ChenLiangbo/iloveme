#!usr/bin/env/python

# import MySQLdb


# def query_from_db(mysql):
#     db       = MySQLdb.connect(host= 'localhost',user='root',passwd='hk1688',db='psrvdb',charset="utf8")
#     cursor = db.cursor()
#     cursor.execute(mysql)
#     results = cursor.fetchall()
#     cursor.close()
#     db.close()
#     return results

# device_id = '00019643'
# sql1 = "SELECT * FROM T0 WHERE device_id = %s" % device_id
# T0_result = query_from_db(sql1)
# id = T0_result[0][0]
# sql2 = "SELECT time,valve_pressure1,t0_id FROM P0 WHERE t0_id = %d order by time desc limit 20" % id
# P0_result = query_from_db(sql2)
# time = []
# valve_pressure1 = []
# for p in P0_result:
#     time.append(p[0])
#     valve_pressure1.append(p[1])
# print "result is--------:",len(P0_result)
# print "time is========",time
# print "valve_pressure1 is ========",valve_pressure1
# # re1 = result[0]
# # for i in re1:
# #     print "your re1 is:",i
                                 

'''
#!/bin/bash

#VIRTUALENV="/var/data/projects/alarm_server/"

#cd $VIRTUALENV
#source ./bin/activate

# Replace these three settings.
#PROJDIR="/var/data/projects/alarm_server/AlertSystem/website/"
#PIDFILE="/var/data/projects/alarm_server/log/new_AlertSystem_site.pid"

#cd $PROJDIR
#if [ -f $PIDFILE ]; then
#    kill `cat -- $PIDFILE`
#    rm -f -- $PIDFILE
#fi
#kill -9 `cat /var/data/projects/go2newera_erp/go2newera/minierp.pid`
python  /var/data/projects/go2newera_erp/go2newera/manage.py runfcgi method=prefork host=127.0.0.1 port=9100 pidfile=/var/data/projects/go2newera_erp/go2newera/minierp.pid

'''
import datetime
def str_to_datetime(string):
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
    result = datetime.datetime.strptime(str3,form)
    return result
   

# form = '%Y-%m-%d %H:%M:%S'                                         #This fromat used frequently
# test_end_time = '2015-09-01 00:01:00'                               #Test datetime string
# end_time = datetime.datetime.strptime(test_end_time,form)    #Test time 
# # end_time = datetime.datetime.now()                                #while running on polarwin
# delta = datetime.timedelta(days = 1)
# start_time = end_time - delta
# flag = (start_time < end_time)
# if flag == True:
#     print flag
# l1 = [1,2,4,6,7,3,2,9,0]
# print "l1----1:",l1
# length = len(l1)
# for i in range(0,length-1):
#     for j in range(i,length):
#         if l1[i] < l1[j]:
#             l1[i],l1[j] = l1[j],l1[i]

# print "l1----2:",l1

def ZMD_countToName(zmd_count):
    PRESSURE_ORDER = {
        1:    u'出口压力',
        2:    u'进口压力',
        4:    u'差压',
        8:    u'差压2',
        16:   u'气体泄漏1',
        32:   u'气体泄漏2',
        64:   u'气体泄漏3',
        128:  u'气体泄漏4',
        256:  u'流量1',
        512:  u'流量2',
        1024: u'进口压力2'
        }
    for key in PRESSURE_ORDER:
        names = []
        if (key & zmd_count) != 0:
            name = str(PRESSURE_ORDER[key])
            print "key = ",key
            names.append(PRESSURE_ORDER[key])
    names = ' + '.join(names)
    return names

zmd_count = 5
names = ZMD_countToName(zmd_count)
print "names = ",names

print "Your test is okay!"