#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import os
from myRecognizer import recognizerObject

#以下是表格是别主要程序的使用例子，也可以使用该程序寻找最合适的y_min_offset
print "recognize A3 paper like example"
imageName = '../image/A3_2.jpg'
#默认情况下不输出excel
recognizerObject.isExcel = True  
#传递合适的y_min_offset可以明显加快程序的速度，不准确也没关系
retDict = recognizerObject.tableRecognizer(imageName,y_min_offset = 480)
recognizedExcelName = retDict['excel']
#识别之后的数据，由字典组成的列表，每一行为一个字典，共16个
retData = retDict["data"]        
print "data = ",retData[0]
print "-"*80

print "recognize A4 paper like example"
imageName = '../image/A4_3.jpg'
recognizerObject.isExcel = True   
#默认情况下识别A3纸张
recognizerObject.isA4 = True    
retDict = recognizerObject.tableRecognizer(imageName,y_min_offset = 150)
recognizedExcelName = retDict['excel']
retData = retDict["data"]
print "data = ",retData[0]
print "-"*80