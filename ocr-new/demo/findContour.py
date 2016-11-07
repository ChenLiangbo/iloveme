# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
import os
import xlrd
import xlwt
from constant import domain

# https://github.com/goncalopp/simple-ocr-opencv
media_dir = os.path.join(os.path.dirname(__file__),'./media')

def findContour(BGRimage):
    '''给定一张彩色图片，返回所有轮廓组成的列表'''
    gray = cv2.cvtColor(BGRimage,cv2.COLOR_BGR2GRAY)
    GaussianBlur = cv2.GaussianBlur(gray,(5,5),0)
    retVal,binary = cv2.threshold(GaussianBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # kernel = np.ones((5,5),dtype = np.uint8)
    # opening = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)    
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def drawRectangle(img,contours,limit = [1500,14400,50,100]):
    '''输入图片，轮廓列表，以及轮廓限制，将限制之后的轮廓画在另一张图上并返回，原始图片不变'''
    print "len(contours) = ",len(contours)
    [minArea,maxArea,minHight,maxHight] = limit
    for cnt in contours:
        contourArea = cv2.contourArea(cnt)
        if contourArea > minArea and contourArea < maxArea:
    	    [x,y,w,h] = cv2.boundingRect(cnt)
    	    if (h > minHight) and (h < maxHight):
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
    return img

def getMaxAreaRectangle(contours):
    area = 0
    for cnt in contours:
        contourArea = cv2.contourArea(cnt)
        area.append(contourArea)



def contoursLargerFilter(contours,limit = [1500,14400,50,100]):
    '''根据矩形的面积最小值，最大值，高度最小值和最大值将轮廓列表初步过滤'''
    newContours = []
    minArea,maxArea,minHight,maxHight = limit
    for cnt in contours:
        contourArea = cv2.contourArea(cnt)
        if contourArea > minArea and contourArea < maxArea:
    	    [x,y,w,h] = cv2.boundingRect(cnt)
    	    rectangle = [x,y,w,h]
    	    if (h > minHight) and (h < maxHight):
    	    	newContours.append(rectangle)
    return newContours

def axisYfilter(contours,yLimit = 1000):
    '''过滤掉矩形列表中纵坐标小于给定值的矩形'''
    newContours = []
    for cnt in contours:
        if cnt[1] > yLimit:
            newContours.append(cnt)
    return newContours

def getDistance(pointA,pointB):
    '''计算两个元祖或者列表组成的点之间的距离
    	pointA = (x1,y1),pointB = (x2,y2),return float'''
    a = pointA[0] - pointB[0]
    b = pointA[1] - pointB[1]
    return math.sqrt(a**2 + b**2)


def in_it(item,l,T = 30):
    '''判断亮点的距离是否小于给定阈值Ｔ'''
    for i in xrange(len(l)):
        if getDistance(item,l[i][0:2]) < T:
            return True
    return False


def contourSmallerFilter(contours,T=30,smallerContour = []):
    '''去除矩形列表中左上角点距离小于Ｔ的重复矩形'''
    smallerContour.append(contours[0])
    for p in contours:
        if not in_it((p[0],p[1]),smallerContour,T):
            smallerContour.append(p)          
    return smallerContour


def autoFilterN(contours,T,N):
    '''使用随机加减法将给定矩形列表依据距离过滤得到Ｎ个矩形组成的列表，阈值随意给定自动调整'''
    smallerContour = []
    smallerContour.append(contours[0])
    for cnt in contours:
        if not in_it((cnt[0],cnt[1]),smallerContour,T):
            smallerContour.append(cnt)
    print "T = ",T
    print "length contours in  autoFilter442 -------- ",len(smallerContour)
    flag = False
    if np.abs(N - len(smallerContour)) < 5:
        flag = True
    if len(smallerContour) == N:
        ret = smallerContour
        return ret
    elif len(smallerContour) > N:
        if flag:
            T = T + np.random.rand(1)[0]
        else:
            T = T + np.random.randint(5)
        return autoFilterN(contours,T,N)
    else:
        if  flag:
            T = T - np.random.rand(1)[0]
        else:
            T = T - np.random.randint(5)
        return autoFilterN(contours,T,N)

def dichotomyAutoFilter(contours,N,Tmin = 0,Tmax = 400):
    '''使用二分法自动调整阈值，将给定矩形列表过滤剩下Ｎ个举行组成的列表'''
    T = float(Tmin + Tmax)/2
    if np.abs(Tmax - Tmin) < 0.00001:
        raise IndexError,"contours list can not be divided in precision of %f in dichotomyAutoFilter" % (np.abs(Tmax - Tmin),)
    smallerContour = []
    smallerContour.append(contours[0])
    for cnt in contours:
        if not in_it((cnt[0],cnt[1]),smallerContour,T):
            smallerContour.append(cnt)
    if len(smallerContour) == N:
        return smallerContour
    elif len(smallerContour) > N:
        return dichotomyAutoFilter(contours,N,T,Tmax)
    else:
        return dichotomyAutoFilter(contours,N,Tmin,T)
    

def myDraw(image,contours):
    '''给定图片矩阵和矩形列表，使用红色线条将矩形绘制在图片上'''
    for [x,y,w,h] in contours:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
    return image

def maopaoSortX(contourList):
    '''使用冒泡排序算法将给定的矩形列表依据左上角点横坐标值由小到大排序'''
    length = len(contourList)
    for i in range(0,length-1):
        for j in range(i,length):
            if contourList[i][0] > contourList[j][0]:
                contourList[i],contourList[j] = contourList[j],contourList[i]
    return contourList


def maopaoSortY(contourList,tableSize = (26,17)):
    '''给定矩形列表和表格大小，将横坐标由小到大排序的矩形列表每一列按照纵坐标由小到大排序，26列，17行'''
    new = []
    xSize,ySize = tableSize[0] + 1,tableSize[1]
    for k in range(1,xSize):
        startIndex = (k-1)*ySize
        endIndex = k*ySize
        pice17 = contourList[startIndex:endIndex]
        length = len(pice17)
        for i in range(0,length-1):
            for j in range(i,length):
                if pice17[i][1] > pice17[j][1]:
                    pice17[i],pice17[j] = pice17[j],pice17[i]
        new = new + pice17
    return new


def cutIntoPices(image,contours,outdir,tableSize = (26,17)):
    '''size = (26,17),根据表格有效单元格的（列数，行数），在给定图片和举行列表的情况下将图片分割'''
    xSize,ySize = tableSize[0] + 1,tableSize[1] + 1
    if len(contours) != ((xSize-1)*(ySize-1)):
        print "contours is not %d ,can not cutIntoPices" % ((xSize-1)*(ySize-1),)
        return
    if outdir.endswith('/'):
        pass
    else:
        outdir = outdir + '/'
    for i in range(1,xSize):
        for j in range(1,ySize):
            index = (i-1)*(ySize-1) + j -1
            [x,y,w,h] = contours[index]
            xOffset = int(w*0.02)
            yOffset = int(h*0.05)
            pice = image[y+yOffset:y+h-yOffset,x+xOffset:x+w-xOffset]
            saveFilename = outdir + '/' + 'x' + str(i) + 'y' + str(j) + '.jpg'
            cv2.imwrite(saveFilename,pice)

def wordRecognize(imageName):
    '''给定一张图片的路径，返回经过识别之后图片中数字的内容'''
    if not os.path.isfile(imageName):
        print "gvie image name is not file "
        return None
    filedir,filename = os.path.split(imageName)
    name,jpg = os.path.splitext(filename)
    outfile = filedir + '/' + name
    if jpg in ['.jpg','.jpeg','.png','.tif']:
        conmand = 'tesseract ' + imageName + ' '+ outfile + ' -psm 6 t_config'
        os.system(conmand)
    fp = open(outfile + '.txt')
    content = fp.readline().strip()
    fp.close()
    os.remove(outfile + '.txt')
    return content

def segment(filepath,outdir,tableSize = (26,17),limit = [1500,14400,50,100],T=30):
    '''分割，给定图片路径，输出路径，表格大小，矩形限制条件，去重距离将表格分割
    tableSize:size of table,(columns,rows),used in maopaoSortY,cutIntoPices
    limit:tangle limit,(minArea,maxArea,minHight,maxHight),used in contoursLargerFilter
    T:point distance,when distance is small than T,combine them,used in contourSmallerFilter
    '''
    image = cv2.imread(filepath)
    try:    
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        GaussianBlur = cv2.GaussianBlur(gray,(5,5),0)
        retVal,binary = cv2.threshold(GaussianBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    except Exception,ex:
        print "Exception:",ex
        return
    contours = findContour(image)
    largerContours = contoursLargerFilter(contours,limit)
    smallerContours = contourSmallerFilter(largerContours,T)    
    smallerContours = maopaoSortX(smallerContours)
    smallerContours = maopaoSortY(smallerContours,tableSize)
    cutIntoPices(binary,smallerContours,outdir,tableSize)

def recAndSaveAsExcel(picedir,excelname):
    '''picedir = ./pice001,excelname = 'result',(string) 
    给定单元格分割的切片文件夹位置，识别其中的内容并且以excel表格形式返回'''
    if not picedir.endswith('/'):
        picedir = picedir + '/'
    book = xlwt.Workbook(encoding = 'utf-8')
    sheet1 = book.add_sheet('Sheet 1',cell_overwrite_ok = True)
    piceList = os.listdir(picedir)
    for pice in piceList:
        xIndex = pice.index('x')
        yIndex = pice.index('y')
        dotIndex = pice.index('.')
        x = pice[xIndex+1:yIndex]
        y = pice[yIndex+1:dotIndex]
        if 'x1y' in pice:
            ret = y
        elif 'x2y' in pice:
            ret = domain[y]
        else:       
            ret = wordRecognize(picedir + pice)
            if ret.startswith('%') or ret.startswith('.'):
                ret = ret[1:]
        sheet1.write(int(y),int(x)-1,str(ret))
        os.remove(picedir + pice)
    if ('.xls' not in excelname) or ('.xlsx' not in excelname):
        excelname = excelname + '.xls'
    book.save(excelname)

def recAndReturnAsDict(picedir):
    '''picedir = ./pice001 pice path，给定分割后矩形的文件夹，将识别的结果以字典返回，每一行一个小字典，列号为小字典的键
    return data = {"y1":{"x1":"12","x2":"1","x3":"0", ... "x26":"100%"},"y2":{},"y3":{}, ... ,}'''
    data = {}
    if not picedir.endswith('/'):
        picedir = picedir + '/'
    piceList = os.listdir(picedir)
    for pice in piceList:
        xIndex = pice.index('x')
        yIndex = pice.index('y')
        dotIndex = pice.index('.')
        x = pice[xIndex:yIndex]
        y = pice[yIndex:dotIndex]
        dy = pice[yIndex+1:dotIndex]
        if 'x1y' in pice:
            ret = dy
        elif 'x2y' in pice:
            ret = domain[dy]
        else:            
            ret = wordRecognize(picedir + pice)
            if ret.startswith('%') or ret.startswith('.'):
                ret = ret[1:]
        if y in data:
            data[y][x] = ret
        else:
            data[y] = {}
            data[y][x] = ret
        os.remove(picedir + pice)
    return data 

def rotate90DegreesToRight(gray):
    if len(gray.shape) != 2:
        raise TypeError,'function rotate90DegreesToRight need image as gray'
    cols,rows = gray.shape
    dst = np.zeros((rows,cols))
    for i in range(cols):
        dst[:,cols - 1 -i] = np.transpose(gray[i,:])
    return dst

if __name__ == '__main__':
    import time
    print "start ... "
    t1 = time.time()
    path = media_dir + '/table/table1/'
    sourceImage = 'img_0003.jpg'
    imageName = path + sourceImage

    image = cv2.imread(imageName)
    contours = findContour(image)
    print "contours = ",len(contours)
    
    largerContours = contoursLargerFilter(contours,limit = [1000,9600,30,100])
    print "length largerContours = ",len(largerContours)

    smallerContours = contourSmallerFilter(largerContours,T = 35)

    print "length smallerContours = ",len(smallerContours)

    # autoContour = autoFilter(largerContours,442)
    # print "length autoContour = ",len(autoContour)

    outdir = media_dir + '/pice'
    cutIntoPices(image,smallerContours,outdir,tableSize = (26,17))
    
    # recAndSaveAsExcel(outdir,media_dir + '/result')
    # table = recAndReturnAsDict(outdir)
    # print "table[1] = ",table['1']
    t2 = time.time()
    print "finishee,it took %f seconds ..." % (t2-t1,)
  
