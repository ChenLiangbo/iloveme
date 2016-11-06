# coding: utf-8
import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

excelname = './yahoo_finance5.xls'
xls = pd.ExcelFile(excelname)

excelParser = xls.parse('Sheet1') 
print "excelParser = ",excelParser
'''Open,High,Low,Close,Volume,Date'''
Open = excelParser['Open']

myOpen = Open[0:10]
print myOpen
print "------------------------"
print myOpen.sort_index()
