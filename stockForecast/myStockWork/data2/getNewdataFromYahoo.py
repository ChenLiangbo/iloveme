#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import xlwt
from yahoo_finance import Share
yahoo = Share('YHOO')


# print yahoo.get_open()
# print yahoo.get_price()
# print yahoo.get_trade_datetime()

print ('Wait...')
# pprint(yahoo.get_historical('2014-04-25', '2014-04-29'))
yahooData = yahoo.get_historical('2008-04-01', '2016-07-01')
n = len(yahooData)
print "data[0] ",yahooData[0]

volume = []
symbol = []
adj_close = []
high = []
low = []
date = []
close = []
myopen = []

book = xlwt.Workbook(encoding = 'utf-8')
sheet1 = book.add_sheet('Sheet 1',cell_overwrite_ok = True)
sheet1.write(0,0,'Adj_Close')
sheet1.write(0,1,'High')
sheet1.write(0,2,'Low')
sheet1.write(0,3,'Close')
sheet1.write(0,4,'Open')
sheet1.write(0,5,'Date')
sheet1.write(0,6,'Volume')
sheet1.write(0,7,'Symbol')

i = 1
for d in yahooData:
    # print "d = ",d
    try:
        adj_close.append(float(d['Adj_Close']))
        high.append(float(d['High']))
        low.append(float(d['Low']))
        date.append(d['Date'])
        close.append(float(d['Close']))
        myopen.append(float(d['Open']))

        sheet1.write(n+1-i,0,float(d['Adj_Close']))
        sheet1.write(n+1-i,1,float(d['High']))
        sheet1.write(n+1-i,2,float(d['Low']))
        sheet1.write(n+1-i,3,float(d['Close']))
        sheet1.write(n+1-i,4,float(d['Open']))
        sheet1.write(n+1-i,5,d['Date'])
        sheet1.write(n+1-i,6,d['Volume'])
        sheet1.write(n+1-i,7,d['Symbol'])
        i = i + 1
    except Exception,ex:
        print "[WRMNING]Exception ",str(ex)

book.save('yahooData.xls')

print "saving excel ..."
# print "myopen = ",len(myopen)
# print "myopen[0]",myopen[0]

outputArray = np.zeros((len(myopen),5),dtype = np.float32)
print "output.shape = ",outputArray.shape

adj_close.reverse()
high.reverse()
low.reverse()
close.reverse()
myopen.reverse()

'''adj_close 0'''
outputArray[:,0] = np.asarray(adj_close).astype(np.float32).transpose()
'''high 1'''
outputArray[:,1] = np.asarray(high).astype(np.float32).transpose()
'''low  2'''
outputArray[:,2] = np.asarray(low).astype(np.float32).transpose()
'''close 3'''
outputArray[:,3] = np.asarray(close).astype(np.float32).transpose()
'''open 4'''
outputArray[:,4] = np.asarray(myopen).astype(np.float32).transpose()


np.save('yahooData',outputArray)

print "finished ..."