#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import os
from myRecognizer import recognizerObject

print "start ..."
imageName = '../image/img2.jpg'
recognizerObject.isExcel = True
retDict = recognizerObject.tableRecognizer(imageName)

recognizedExcelName = retDict['excel']

ret = retDict["data"]

print "data = ",ret[0]
