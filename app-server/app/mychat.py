# -*- coding: utf-8 -*-
import urllib2
from urllib import urlencode
try:
    import simplejson
except:
    import json as simplejson
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

WECHAT_TOKEN = "wechattoken"
CORPID       = "wx4b4b6d394cf22d38"
SECRET       = "ZRDYpDqg3X1wu1Y4bCZH7rADlGFpl-zEBAOOXi5avbXhumtRu-d4cwyrHJNpaEfY"

class myChat(object):
    CorpId = ""
    Secret = ""
    token = ""
    
    def __init__(self,CorpId,Secret,token):
        self.CorpId = CorpId
        self.Secret = Secret
        self.token  = token 

    def getTokenIntime(self):
        res = urllib2.urlopen('https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid=%s&corpsecret=%s'%(self.CorpId,self.Secret))
        res_dict = simplejson.loads(res.read())  
        res_json = simplejson.dumps(res_dict) 
        return res_dict["access_token"]

    def sendTxtMsg(self,access_token,content,application_id,to_user="@all",to_party="",to_tag="",safe=0):
        try:
            data = {
               "touser": to_user,
               "toparty": to_party,
               "totag": to_tag,
               "agentid": application_id,            
               "msgtype": "text",
               "text": {
                   "content": content,
               },
               "safe":safe
               }

            data = simplejson.dumps(data,ensure_ascii=False).encode("utf-8")
            req = urllib2.Request('https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=%s'%(access_token,))        
            resp = urllib2.urlopen(req,data)
            # msg  = u'sendTextMsg return value:' + resp.read()
            return resp.read()
        except Exception,ex:
            print "exception this"
            print 'exception %s' % str(ex)

    def media_upload(self,access_token,media_name,media_type):
    	from poster.encode import multipart_encode  
        from poster.streaminghttp import register_openers 
        register_openers()
        datagen, headers = multipart_encode({"file1": open(media_name, "rb")})
        URL= 'https://qyapi.weixin.qq.com/cgi-bin/media/upload?access_token=%s&type=%s' %(access_token,media_type)
        request = urllib2.Request(URL, datagen,headers) 
        ret_json = urllib2.urlopen(request).read()
        ret_dict = simplejson.loads(ret_json)
        return ret_dict  

    def getFileCount(self,access_token,application_id):
        #get the number of file you have uploaded
        url = "https://qyapi.weixin.qq.com/cgi-bin/material/get_count?access_token=%s&agentid=%d" %(access_token,application_id)
        resp = urllib2.urlopen(url)
        ret_dict = resp.read()
        return ret_dict

    def getFileList(self,access_token,TYPE,application_id,offset=0,count=1):
        url = "https://qyapi.weixin.qq.com/cgi-bin/material/batchget?access_token=%s" %(access_token,)
        req = urllib2.Request(url)       
        data = {
                "type":TYPE,
                "agentid":application_id,
                "offset":offset,
                "count":count
                }
        data = simplejson.dumps(data,ensure_ascii=False).encode("utf-8")
        resp = urllib2.urlopen(req,data)
        ret_dict = simplejson.loads(resp.read())
        print "ret_dict:",ret_dict
        itemlist = ret_dict.get("itemlist")
        return itemlist

    #send video message by weixin qiyehao
    def sendVideoMsg(self,access_token,MEDIA_ID,application_id,to_user="@all",to_party="",to_tag=""):
        try:
            data = {
                 "touser": to_user,
                 "toparty":to_party,
                 "totag": to_tag,
                 "msgtype": "video",
                 "agentid": application_id,
                 "video": {
                     "media_id": MEDIA_ID,     #media_id is on server
                     "title": "Video Message", #video message title
                     "description": "This is a mp4 message!" #video discription
                 },
                 "safe":"0"
                 }
            data = simplejson.dumps(data,ensure_ascii=False).encode("utf-8")
            req = urllib2.Request('https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=%s'%(access_token,))        
            resp = urllib2.urlopen(req,data)
            msg  = u'send_video return value:' + resp.read()
            print msg
        except Exception,ex:
            print "exception this"
            print 'exception %s' % str(ex)


    def sendImageMsg(self,access_token,MEDIA_ID,application_id,to_user="@all",to_party="",to_tag=""):
        try:
            data={
              "touser": to_user,
              "toparty": to_party,
              "totag": to_tag,
              "msgtype": "image",
              "agentid": application_id,
              "image": {
                  "media_id": MEDIA_ID
              },
              "safe":"0"
              }
            data = simplejson.dumps(data,ensure_ascii=False).encode("utf-8")
            req = urllib2.Request('https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=%s'%(access_token,))        
            resp = urllib2.urlopen(req,data)
            msg  = u'send_image return value:' + resp.read()
            print msg
        except exception,ex:
            print "exception this"
            print 'exception %s' % str(ex)


mychat = myChat(CORPID,SECRET,WECHAT_TOKEN)
'''
access_token = mychat.getTokenIntime()
 
media_id1='1eoSJcZ-yrwoOKc8tcTq6WGiMahE5F4nW-5IVtu46KbLaQVgmIOKgEApGQqsDJQj86vYTD848pJZSt4nlmCt1eA'
media_type = "image/jpg"
mychat.sendImageMsg(access_token,media_id1,1,"go2newera0006")
message = "Test information from chenlb!"
mychat.sendTxtMsg(access_token,message,1)
'''
