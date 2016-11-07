#!/user/bin/env/python
#coding:utf-8

import os
from dbOperator import DBoperator

try:
    username = os.sys.argv[1]
    print "username = ",username
    if len(os.sys.argv) < 3:
    	days = 60
    else:
        days = int(os.sys.argv[2])
except Exception,ex:
    print "Exception:",ex
    os.sys.exit()

hostadderss = '120.26.105.20:3306'
# hostadderss = '192.168.1.40:3306'
database = 'psrvdb'
user = 'justin'
password = 'WGTo0dz9'
psrvdbOperatorDB = DBoperator(hostadderss,database,user,password)  

sqlUser = "select * from auth_user where username = '%s'" % (username,)
user = psrvdbOperatorDB.select(sqlUser)
user = user[0]
print "user = ",user

sqlUpdate = 'update alertapp_login set is_paid = %d,time_length = %d where user_id = %d' % (False,days,int(user.id))
psrvdbOperatorDB.update(sqlUpdate)

sql = 'select * from alertapp_login where user_id = %d' % (int(user.id),)
ret = psrvdbOperatorDB.select(sql)
print "ret = ",ret[0]
psrvdbOperatorDB.disconnect()
