#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import findContour
import math
import time

t1 = time.time()
path = '../result/'


imageName = '../image/img1.jpg'

if not os.path.isfile(imageName):
    raise TypeError,'function tableTreeRecognizer need a file name as input'

if not os.path.exists(path):
    os.mkdir(path)


image = cv2.imread(imageName)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

myGray = gray.copy()
GaussianBlur = cv2.GaussianBlur(gray,(5,5),0)
adaptiveBinary = cv2.adaptiveThreshold(GaussianBlur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
retVal,threshBinary = cv2.threshold(GaussianBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  
contours, hierarchy = cv2.findContours(adaptiveBinary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


cv2.imwrite(path + 'adaptiveBinary.jpg',adaptiveBinary)
cv2.imwrite(path + 'threshBinary.jpg',threshBinary)


img1 = image.copy()
contoursAreaList = []


areaList = []
wideList = []
hightList = []
x_asix = []
y_asix = []
max_three_w = []
minArea = 1200
for cnt in contours:
    contourArea = cv2.contourArea(cnt)
    [x,y,w,h] = cv2.boundingRect(cnt)
    areaList.append(contourArea)
    wideList.append(w)
    hightList.append(h)
    x_asix.append(x)
    y_asix.append(y)
    cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),1)
    if contourArea > minArea:
        contoursAreaList.append([x,y,w,h])

print "contoursAreaList length = ",len(contoursAreaList)
cv2.imwrite(path + 'contours.jpg',img1)


# from matplotlib import pyplot as plt
# plt.plot(wideList,hightList,'ro')
# plt.plot(wideList,hightList,'r-')
# plt.grid(True)
# plt.xlabel('number')
# plt.ylabel('w')
# plt.title('w-h distribution graph')
# plt.savefig(path + 'w-h.jpg')
# plt.show()


img2 = image.copy()
for cnt in contoursAreaList:
    [x,y,w,h] = cnt
    cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),1)
cv2.imwrite(path + 'contoursAreaList.jpg',img2)

max_three_w = []
for w in wideList:
    if w > 3000:
        max_three_w.append(w)

max_three_w.sort()

'''
max_three_w = wideList
max_three_w.sort()
num = len(max_three_w)
max_three_w = max_three_w[num-3:num]
'''

max_index = wideList.index(max_three_w[0]) 
[x,y,w,h] = [x_asix[max_index],y_asix[max_index],wideList[max_index],hightList[max_index]]

print "[x,y,w,h] = ",[x,y,w,h]
x_min = x-10
x_max = x + w+10
y_min = y-10
y_max = y + h+10
y_min_offset = 480
print "x_min = %d,y_min = %d" % (x_min,y_min)

img5 = image.copy()
# cv2.line(img5,(x_min,y_min + 500),(x_min + 1000,y_min + 500),(255,0,0),2)
cv2.rectangle(img5,(x_min,y_min + y_min_offset),(x_max,y_max ),(0,0,255),2)
cv2.imwrite(path + 'line.jpg',img5)

# myGray = gray[y_min+500:y_max+10,x_min-10:x_max+10]
# cv2.imwrite(path + 'myGray.jpg',myGray)

img6 = image.copy()
myContours = []
for [x,y,w,h] in contoursAreaList:
    if x > x_min-10 and x < x_max+10:
        if y > y_min+y_min_offset and y < y_max+10:
            if h<200 and h>50 and w<300:
                myContours.append([x,y,w,h])
                cv2.rectangle(img6,(x,y),(x+w,y+h),(0,0,255),2)

print "myContours length = ",len(myContours)
cv2.imwrite(path + 'myContours.jpg',img6)



wList = []
hList = []
xList = []
yList = []
for [x,y,w,h] in myContours:
    xList.append(x)
    yList.append(y)
    wList.append(w)
    hList.append(h)

def maopaoSort(oldList):
    length = len(oldList)
    for i in xrange(0,length-1):
        for j in xrange(i,length):
            if oldList[i] > oldList[j]:
                oldList[i],oldList[j] = oldList[j],oldList[i]
    return oldList

'''将排序好的横坐标和纵坐标作为输入，根据横纵坐标差值的剧烈变化，找到合适的
分割阈值，将每一行和每一列分割开来
'''
def getLadder(eList):
    retList = []
    for i in xrange(len(eList)-1):
        retList.append(eList[i+1]-eList[i])
    retList.sort()
    for i in xrange(len(retList) - 1):
        if retList[i] >0 and retList[i+1] > 5*retList[i]:
            return (retList[i+1]+retList[i])/3

xList = maopaoSort(xList)
xLadder = getLadder(xList)
print "xLadder = ",xLadder
xThresh = []
for i in xrange(len(xList)-1):
    if xList[i+1]-xList[i]>xLadder:
        xThresh.append((xList[i]+xList[i+1])/2)

print "xThresh = ",len(xThresh)

yList = maopaoSort(yList)
yLadder = getLadder(yList)
print "yLadder = ",yLadder
yThresh = []
for i in xrange(len(yList)-1):
    if yList[i+1]-yList[i]>yLadder:
        yThresh.append((yList[i]+yList[i+1])/2)

print "yThresh = ",len(yThresh)


def getLabel(e,eThresh):
    '''根据给定值和阈值区间找到对应的标签'''  
    if e < min(eThresh):
        return 1
    elif e > max(eThresh):
        return len(eThresh) + 1
    else:
        for i in xrange(1,len(eThresh)):
            if e > eThresh[i-1] and e < eThresh[i]:
                return i+1

img7 = image.copy()
for xline in xThresh:
    for yline in yThresh:
        cv2.line(img7,(xline,yline),(xline + 1000,yline),(0,0,255),2)
        cv2.line(img7,(xline,yline),(xline,yline+400),(255,0,0),2)

cv2.imwrite(path +'xLine-yLine.jpg',img7)

'''给定矩形分割存储位置'''
picedir = './pice/'
if  not os.path.exists(picedir):
    os.mkdir(picedir)

'''分割原始图像得到小单元格，需要保证不重复，不遗漏'''
for [x,y,w,h] in myContours:
    xlabel = str(getLabel(x,xThresh))
    ylabel = str(getLabel(y,yThresh))
    xoffset = 2
    yoffset = 2
    pice = threshBinary[y+yoffset:y+h-yoffset,x+xoffset:x+w-xoffset]
    picename = picedir +'x' +xlabel+'y'+ylabel+'.jpg'
    cv2.imwrite(picename,pice) 


'''识别分割好的小单元格内的内容并将结果保存未表格'''
excelname = path + 'result' 
# findContour.recAndSaveAsExcel(picedir,excelname)


# from matplotlib import pyplot as plt
# h_number = len(xList)
# x_asix = range(h_number)
# # plt.plot(x_asix,xList,'-')
# # plt.plot(x_asix,xList,'ro')
# plt.plot(x_asix[0:50],yList[0:50],'ro')
# plt.grid(True)
# plt.xlabel('number')
# plt.ylabel('xList')
# plt.title('xList distribution graph')
# plt.savefig(path + 'xList.jpg')
# plt.show()


# from matplotlib import pyplot as plt
# areaList.sort()
# # plt.plot(wList,hList,'-')
# plt.plot(xList,yList,'ro')

# plt.grid(True)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('x-y graph')
# plt.savefig(path + 'x-y.jpg')
# plt.show()

# print "time = ",time.time() - t1
