#!usr/bin/env/python 
# -*- coding: utf-8 -*-
from scrapy.selector import Selector
from scrapy.http import HtmlResponse

html = '<li><a href="http://www.baidu.com" class="one" title="C">C</a>li-text-li</li>'
link = Selector(text=html).xpath('//a/@href').extract()
title = Selector(text=html).xpath('//a/@title').extract()
print("text = ",link)
print("title = ",title)
