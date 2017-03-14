#!usr/bin/env/python 
# -*- coding: utf-8 -*-
from __future__ import print_function
from scrapy import Spider
from scrapy import Request
from scrapy.selector import HtmlXPathSelector
from scrapy.selector import Selector
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import os
import json
from bs4 import BeautifulSoup
from mytuils import ColorPrint,OutWriter
import urllib
import csv
import urllib2
import sys
import re 



reload(sys)
sys.setdefaultencoding('utf-8')


color = ColorPrint()




 
class LianjiaSpider(CrawlSpider):
    name = "lianjia"
 
    allowed_domains = ["lianjia.com"]
 
    start_urls = [
        "http://bj.lianjia.com/ershoufang/"
    ]
 
    rules = [
        # 匹配正则表达式,处理下一页
        Rule(LinkExtractor(allow=(r'http://bj.lianjia.com/ershoufang/pg\s+$',)), callback='parse_item'),
 
        # 匹配正则表达式,结果加到url列表中,设置请求预处理函数
        # Rule(FangLinkExtractor(allow=('http://www.lianjia.com/client/', )), follow=True, process_request='add_cookie')
    ]
 
    def parse_item(self, response):
        # 这里与之前的parse方法一样，处理
        # hxs = HtmlXPathSelector(response)
        pass




class TianqiSpider(CrawlSpider):

    '''This spider use python3 on windows
       there maybe some errors on other platform
    '''

    name = "tianqi"
    

    allowed_domains = ["lishi.tianqi.com"]
 
    start_urls = [
        "http://lishi.tianqi.com/"
    ]
    
    # rules = [
    #     # 匹配正则表达式,处理下一页
    #     Rule(LinkExtractor(allow=(r'http://bj.lianjia.com/ershoufang/pg\s+$',)), callback='parse_item'),
 
    #     # 匹配正则表达式,结果加到url列表中,设置请求预处理函数
    #     # Rule(FangLinkExtractor(allow=('http://www.lianjia.com/client/', )), follow=True, process_request='add_cookie')
    # ]
 
    def parse(self, response):
        print("-"*80)
        # print("response = ",dir(response))
        # fp = open('first.txt','wb')
        # fp.write(response.body)
        # fp.close()
        hxs = HtmlXPathSelector(response)
        uls = response.xpath('//ul')
        # print("uls = ",type(uls),dir(uls))
        icount = 0
        filedir = os.getcwd() + '/' + str(icount)
        if not os.path.exists(filedir):
            os.mkdir(filedir)

        for ul in uls:
            jcount = 0
            filename = filedir + '/' + str(jcount) + '.txt'
            fp = open(filename,'w')
            # title = site.select('a/text()').extract()
            # link = site.select('a/@href').extract()
            # desc = site.select('text()').extract()
            # print("title = ",title, link, desc)
            # print("ul = ",type(ul),dir(ul))
            lis = ul.select('./li')
            print("lis = ",type(lis),len(lis))        
            for li in lis:
                title = li.select('//a/@title').extract()
                link = li.select('//a/@href').extract()
                # print("li = %s" % (li))
                print("title = %s" % (len(title),))
                print("link = %s" % (len(link),))
                cdict = {"title":title[0],"link":link[0]}
                cjson = json.dumps(cdict,separators = (',',':'))

                print(title[0],type(title[0]))
                print(link[0],type(link[0]))
                fp.write(cjson)
                fp.write('/r/n')
                fp.close()
                # print(dir(li))
                # print()
                # html = li.extract()
                # link = Selector(text=html).xpath('//a/@href').extract()
                # title = Selector(text=html).xpath('//a/@title').extract()
                # break
            icount = icount + 1
            # if icount > 5:
            #     break
        print("-"*80)


# This spider get Jiading Tianqi information from website
# Test finished with python3 on windows
class JiadingTianqiSpider(CrawlSpider):
    name = "JiadingTianqi"
    
    allowed_domains = ["lishi.tianqi.com"]
 
    start_urls = [
        "http://lishi.tianqi.com/jiading/index.html"
    ]
    
    # rules = [
    #     # 匹配正则表达式,处理下一页
    #     Rule(LinkExtractor(allow=(r'http://bj.lianjia.com/ershoufang/pg\s+$',)), callback='parse_item'),
 
    #     # 匹配正则表达式,结果加到url列表中,设置请求预处理函数
    #     # Rule(FangLinkExtractor(allow=('http://www.lianjia.com/client/', )), follow=True, process_request='add_cookie')
    # ]
 
    def parse(self, response):
        print('-'*80)
        html = response.body
        # print(html)
        soup = BeautifulSoup(html,'html.parser')
        links = []
        for link in soup.find_all('a'):
            link = link.get('href')
            if 'lishi.tianqi.com/jiading' in link:
                # print("link = ",link)
                links.append(link)
        spiderdir = './jiadiangTianqi/'
        if not os.path.exists(spiderdir):
            os.mkdir(spiderdir)
        
        for url in links[1:]:
            # print("url = ",url)
            filename = spiderdir + url.split('/')[-1].split('.')[0] + '.txt'
            fp = open(filename,'a')
            res = urllib2.urlopen(url)         # python2
            #res = urllib.request.urlopen(url)  # python3
            html = res.read()
            soup = BeautifulSoup(html,'html.parser')
            #//div class = 'tqtongji2' 
            div = soup.find_all("div", class_="tqtongji2")
            try:
                div = div[0]
            except Exception as ex:
                print("Exception happens that:",str(ex))
                continue
            # print("div = ",div,dir(div))
            uls = div.find_all('ul')
            # print("uls = ",uls)
            for ul in uls:
                alist = []
                lis = ul.find_all('li')
                # print("lis = ",lis)
                # print('-'*20)
                for li in lis:
                    c = li.string
                    alist.append(c)
                # print("alist = ",alist)
                try:
                    aline = ','.join(alist)
                except Exception as ex:
                    print("Exception happens that:",str(ex))
                    aline = "excption happens"
                fp.write(aline )
                fp.write('\n')
            fp.close()


            # break 
        print('-'*80)


# This spider gets Shanghai Tianqi from websitei: lishi,tianqi.com
# Test finished with python2 on ubuntu,especially server 40 in our company

class ShanghaiTianqiSpider(CrawlSpider):
    name = "ShanghaiTianqi"
    
    allowed_domains = ["lishi.tianqi.com"]
 
    start_urls = [
        "http://lishi.tianqi.com/"
    ]
    
    # rules = [
    #     # 匹配正则表达式,处理下一页
    #     Rule(LinkExtractor(allow=(r'http://bj.lianjia.com/ershoufang/pg\s+$',)), callback='parse_item'),
 
    #     # 匹配正则表达式,结果加到url列表中,设置请求预处理函数
    #     # Rule(FangLinkExtractor(allow=('http://www.lianjia.com/client/', )), follow=True, process_request='add_cookie')
    # ]


    def get_one_city(self,link):

        outer = OutWriter()
        city = link.split('/')[3]
        csvfile = file(outer.outdir + city + '_tianqi.csv', 'wb')
        writer = csv.writer(csvfile)
        
        html = urllib2.urlopen(link).read()  
        #print("html = ",html)
        soup = BeautifulSoup(html,'html.parser')  


        links = []
        for alink in soup.find_all('a'):
            alink = alink.get('href')
            if ('lishi.tianqi.com/'+ city) in alink:
                # print("link = ",link)
                links.append(alink)
        
        for url in links[1:]:
            # print("url = ",url)
            res = urllib2.urlopen(url)
            html = res.read()
            soup = BeautifulSoup(html,'html.parser')
            #//div class = 'tqtongji2' 
            div = soup.find_all("div", class_="tqtongji2")
            try:
                div = div[0]
            except Exception as ex:
                print("Exception happens that:",str(ex))
                continue
            # print("div = ",div,dir(div))
            uls = div.find_all('ul')
            # print("uls = ",uls)
            for ul in uls:
                alist = []
                lis = ul.find_all('li')
                # print("lis = ",lis)
                # print('-'*20)
                for li in lis:
                    try:
                        c = li.string.encode('utf-8')
                        alist.append(c)
                    except Exception as ex:
                        print("Exception happens : ",str(ex))
                        alist.append(' ')
                #print("alist = ",alist,type(alist))

                writer.writerow(alist)
           
 
    def parse(self, response):
        print("-"*80)

        citylist = ['huangpu1','xujiahui','changning','jingan','putuo','zhabei','hongkou1','yangpu',\
                    'fengxian','qingpu','songjiang','jiading','jinshan','baoshan']

        path = os.getcwd() + '/'
        outdir = path + 'ShanghaiTianqi/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        html = response.body
        # print(html)
        soup = BeautifulSoup(html,'html.parser')
        uls = soup.find_all('ul',id = re.compile('city_'))
        links = []
        citys = []
        for ul in uls:
            #print("ul = %s" % (ul,))
            lis = ul.find_all('li')
            for li in lis:
                try:
                    #print("li = ",li,type(li))
                    link = str(li.a['href'])
                    #print("link = ",link)
                    city = str(link.split('/')[3])
                    #print("city = %s,link = %s" %(city,link))
                except Exception as ex:
                    print('Exception happens :',str(ex),"link = ",link)
                    city = ''
                    link = ''
                if city in citylist:
                    #print("city = ",city)
                    links.append(link)
                    citys.append(city)
        print("links = ",links)  
                #break
            #break
        for link in links:
            self.get_one_city(link) 
          
        print('-'*80)

