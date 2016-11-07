#!/user/bin/env/python
#coding:utf-8
import torndb


hostadderss = '120.26.105.20:3306'
# hostadderss = '192.168.1.40:3306'
database = 'psrvdb'
user = 'justin'
password = 'hk1688'


class DBoperator(object):
    def __init__(self,hostadderss,database,user,password):
        super(DBoperator,self).__init__()
        self.host     = hostadderss
        self.database = database
        self.user     = user
        self.password = password
        self.db       = self.getConnect()

    def getConnect(self):
    	try:
    	    db = torndb.Connection(hostadderss,database,user,password)
    	    return db
    	except Exception,ex:
            print "Exception happens in execute sql:",ex
            return None

    def select(self,sql):
    	if self.db is None:
    	    print "Exception happens when connect database"
    	    return None
    	try:
            ret = self.db.query(sql)
    	    return ret
        except Exception,ex:
            print "Exception happens in execute sql:",ex
            return None
    	

    def update(self,sql):
    	if self.db is None:
    	    return False
    	try:
            self.db.execute(sql)
            return True
        except Exception,ex:
            print "Exception happens in execute sql:",ex
            return False
    def disconnect(self):
    	if self.db:
    	    self.db.close()
psrvdbOperatorDB = DBoperator(hostadderss,database,user,password)    
T0tablesOperator = DBoperator(hostadderss,'T0tables',user,password)
#test and example
if __name__ == '__main__':
    # hostadderss = 'localhost:3306'
    # database = 'T0tables'
    # user = 'root'
    # password = 'iloveme1314'     
    sql = 'select id,device_id,name,locat,address,latitude,longitude from T0 limit 1'
    result = OperatorDB.select(sql)
    print "result = ",result
    print dir(result[0])
    print len(result)

    latitude = 32
    longitude = 23
    device_id = 'TESTFQQ1'
    sql1 = "update T0 set latitude = %d,longitude = %d where device_id = '%s'" % (latitude,longitude,device_id)
    updateFlag = OperatorDB.update(sql1)
    print "updateFlag = ",updateFlag

    sql2 = "select id,username,password from auth_user where username = 'justin'"
    chenlb = OperatorDB.select(sql2)
    print "chenlb = ",chenlb
    print "test ok ..."    
