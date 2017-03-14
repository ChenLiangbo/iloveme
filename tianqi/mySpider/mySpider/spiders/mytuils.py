#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import os


class TianQiConfig(object):
    def __init__(self,):
        super(TianQiConfig,self).__init__()
        self.years    = list(range(2014,2016))
        self.months   = list(range(1,13)) 
        self.cityName = 'jiading'

    def getYearAndMonth(self,yearList,monthList):
        time = []
        for y in yearList:
            for m in monthList:
                if int(m) < 10:
                    t = str(y) + '0' + str(m)
                else:
                    t = str(y) + str(m)
                time.append(t)
        return time

    def getUrls(self,):
        # http://lishi.tianqi.com/jiading/201611.html
        domainUrl = 'http://lishi.tianqi.com'
        cityUrl = domainUrl + '/' + self.cityName + '/'
        times = self.getYearAndMonth(self.years,self.months)
        urls = []
        for t in times:
            timeUrl = cityUrl + t + '.html'
            urls.append(timeUrl)
        return urls

class ColorPrint(object):
    # 显示格式: \033[显示方式;前景色;背景色m
    # 只写一个字段表示前景色,背景色默认
    RED = '\033[31m'       # 红色
    GREEN = '\033[32m'     # 绿色
    YELLOW = '\033[33m'    # 黄色
    BLUE = '\033[34m'      # 蓝色
    FUCHSIA = '\033[35m'   # 紫红色
    CYAN = '\033[36m'      # 青蓝色
    WHITE = '\033[37m'     # 白色

    #: no color
    RESET = '\033[0m'      # 终端默认颜色

    def color_str(self, color, s):
        return '{}{}{}'.format(
            getattr(self, color),
            s,
            self.RESET
        )

    def red(self, s):
        return self.color_str('RED', s)

    def green(self, s):
        return self.color_str('GREEN', s)

    def yellow(self, s):
        return self.color_str('YELLOW', s)

    def blue(self, s):
        return self.color_str('BLUE', s)

    def fuchsia(self, s):
        return self.color_str('FUCHSIA', s)

    def cyan(self, s):
        return self.color_str('CYAN', s)

    def white(self, s):
        return self.color_str('WHITE', s)

class OutWriter(object):
    
    def __init__(self, ):
        super(OutWriter,self).__init__()
        self.name = 'ShanghaiTianqi'
        self.path = os.getcwd()
        self.outdir = self.path + '/' + self.name + '/'
        
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)




if __name__ == '__main__':
     tianqiObject = TianQiConfig()
     urls = tianqiObject.getUrls()
     for url in urls:
         print(url) 
