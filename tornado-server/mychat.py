# -*- coding: utf-8 -*-
import urllib2
import sys
import os
import datetime
from urllib import urlencode
try:
    import simplejson
except:
    import json as simplejson

reload(sys)
sys.setdefaultencoding('utf-8')

WECHAT_TOKEN = "wechattoken"
CORPID       = "wx4b4b6d394cf22d38"
SECRET       = "ZRDYpDqg3X1wu1Y4bCZH7rADlGFpl-zEBAOOXi5avbXhumtRu-d4cwyrHJNpaEfY"
EncodingAESKey = "D7Mm9PtgPuz8SDIkKAzxEfvOL2aUgSQ4uKWeL4JK0Xc"    


class myChat(object):
    CorpId = ""
    Secret = ""
    token = ""
    
    def __init__(self,CorpId,Secret,token,AppID = 0):
        self.CorpId = CorpId
        self.Secret = Secret
        self.token  = token
        self.AppId  = AppID
        self.accessTokenFile = os.path.join(os.path.dirname(__file__),'./notes/wx_access_token.txt')

    def getTokenIntime(self):        
        fp = open(self.accessTokenFile,'w+')
        wx_token = fp.read()
        fp.close()
        now = datetime.datetime.now()
        if len(wx_token) > 0:          
            wx_token_dict = simplejson.loads(wx_token)            
            getTime = datetime.datetime.strptime(wx_token_dict["time"],'%Y-%m-%d:%H:%M:%S')
            if int((now - getTime).seconds) < int(wx_token_dict["expires_in"]):
                access_token = wx_token_dict["access_token"]
                return access_token
        res = urllib2.urlopen('https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid=%s&corpsecret=%s'%(self.CorpId,self.Secret))
        res_dict = simplejson.loads(res.read())        
        res_dict["time"] = now.strftime('%Y-%m-%d:%H:%M:%S')
        res_json = simplejson.dumps(res_dict)
        fp1 = open(self.accessTokenFile,'w')
        fp1.write(res_json)
        fp1.close()
        return res_dict["access_token"]

    def sendTxtMsg(self,access_token,content,to_user="@all",to_party="",to_tag="",safe=0):
        try:
            data = {
               "touser": to_user,
               "toparty": to_party,
               "totag": to_tag,
               "agentid": self.AppId,            
               "msgtype": "text",
               "text": {
                   "content": content,
               },
               "safe":safe
               }

            data = simplejson.dumps(data,ensure_ascii=False).encode("utf-8")
            req = urllib2.Request('https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=%s'%(access_token,))        
            resp = urllib2.urlopen(req,data)          
            ret_json = resp.read() 
            return ret_json          
        except Exception,ex:
            print "exception this"
            print 'exception %s' % str(ex)
            ret_dict = {"errmsg":"send txtmsg error"}
            return simplejson.dumps(res_dict)


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

    def getFileCount(self,access_token):
        #get the number of file you have uploaded
        url = "https://qyapi.weixin.qq.com/cgi-bin/material/get_count?access_token=%s&agentid=%d" %(access_token,self.AppId)
        resp = urllib2.urlopen(url)
        ret_dict = resp.read()
        return ret_dict

    def getFileList(self,access_token,TYPE,offset=0,count=1):
        url = "https://qyapi.weixin.qq.com/cgi-bin/material/batchget?access_token=%s" %(access_token,)
        req = urllib2.Request(url)       
        data = {
                "type":TYPE,
                "agentid":self.AppId,
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
    def sendVideoMsg(self,access_token,MEDIA_ID,to_user="@all",to_party="",to_tag=""):
        try:
            data = {
                 "touser": to_user,
                 "toparty":to_party,
                 "totag": to_tag,
                 "msgtype": "video",
                 "agentid": self.AppId,
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
            ret_json = resp.read()
            return ret_json    
        except Exception,ex:
            print "exception this"
            print 'exception %s' % str(ex)
            ret_dict = {"errmsg":"video send error"}
            return simplejson.dumps(res_dict)
      


    def sendImageMsg(self,access_token,MEDIA_ID,to_user="@all",to_party="",to_tag=""):
        try:
            data={
              "touser": to_user,
              "toparty": to_party,
              "totag": to_tag,
              "msgtype": "image",
              "agentid": self.AppId,
              "image": {
                  "media_id": MEDIA_ID
              },
              "safe":"0"
              }
            data = simplejson.dumps(data,ensure_ascii=False).encode("utf-8")
            req = urllib2.Request('https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=%s'%(access_token,))        
            resp = urllib2.urlopen(req,data)            
            ret_json = resp.read()  
            return ret_json                  
        except Exception,ex:
            print "exception this"
            print 'exception %s' % str(ex)
            ret_dict = {"errmsg":"image send error"}
            return simplejson.dumps(res_dict)        

mychat = myChat(CORPID,SECRET,WECHAT_TOKEN)

if __name__ == "__main__":
    import time
    from imageFunction import videoTransform
    # get access_token 
    access_token = mychat.getTokenIntime()
    
    # access_token = '24GBhfGFtk9TQ5eyoJVg9uIXc_Fp0jH-KQ-D3Fo8szXPE7m-FDh6aCaP6yDeN_QT'
    print "access_token = ",access_token
 
    '''
    # uploads  video media
    media_name = '../video/VIDMODEL420160414-224543'   # leave .avi
    media_name = videoTransform(media_name)
    media_type = 'video/mp4'
    up_load_dict = mychat.media_upload(access_token,media_name,media_type)  # type(up_load_dict) = dict
    video_id = up_load_dict["media_id"]
    
    
    #send video message
    # video_id = '18EgCA3DQOygHxfSuaBZrPfQG2I2O5r2wOVqweycOzJb0P9vdgrDJJL3X9-fJ45z_ThJelyvrX_o5vo93i7SKTA'
    video_dict = mychat.sendVideoMsg(access_token,video_id,"go2newera0006")
    video_dict = simplejson.loads(video_dict)
    print "video_dict = ",video_dict               #{u'errcode': 0, u'errmsg': u'ok'} 
  
    #upload image media image message
    # image_id1='1JhD9rf5k7fNYRLTzM5jhLQPubbJl8KS3jFrvpCs1vm9H4CO_mCqPuYjG6I5228FtnuokAv8RlkC7amNpY1fwjQ'
    #upload image media
    image_type = "image/jpg"
    image_name = '../images/tmp_image.jpg'
    image_upload_dict = mychat.media_upload(access_token,image_name,image_type)
    image_id = image_upload_dict["media_id"]
    print "image_id = ",image_id    

    image_id = '1LKtPTjuR5AJLnRIn-B-nK8EENadjsAfQwuBMupujjHcQcOt7l0aBNae7EbHV7A_mHhUNDJw__2lCPlvNVZ-kQQ'
    image_dict = mychat.sendImageMsg(access_token,image_id,"go2newera0006")
    image_dict = simplejson.loads(image_dict)    #{u'errcode': 0, u'errmsg': u'ok'}
    print 'image_dict = ',image_dict
    '''
    # send txt message
    message = "Test information from chenlb!"
    txt_dict = mychat.sendTxtMsg(access_token,message,'go2newera0006')
    txt_dict = simplejson.loads(txt_dict)     #{u'errcode': 0, u'errmsg': u'ok'}
    print "txt_dict = ",txt_dict 
    
    print "It is ok!"
