#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import xlwt
import datetime
import platform



class MyRecognizer(object):

    def __init__(self,):
        super(MyRecognizer,self).__init__()
        self.path    = os.path.join(os.path.dirname(__file__),'output/')
        self.picedir = os.path.join(os.path.dirname(__file__),'pice/')
        self.tesseractConfig = os.path.join(os.path.dirname(__file__),'t_config')

        self.iterTimes  = 0
        self.maxIters   = 200
        self.yMinOffset = 480
        self.isExcel    = False
        self.isTable345 = False        

        self.tableSettings12  = {"minHigh":50,"maxHigh":200,"minArea":1200,"yRows":15,
                                 "myWeight":3000,"maxWeight":300,"offset":20,"yMinOffset":480}
        self.tableSettings345 = {"minHigh":50,"maxHigh":140,"minArea":1200,"yRows":16,
                                 "myWeight":1500,"maxWeight":200,"offset":10,"yMinOffset":150}
        self.domain = {"1":"浦东新区","2":"徐汇区","3":"长宁区","4":"普陀区","5":"虹口区","6":"杨浦区","7":"黄浦区","8":"静安区",
                       "9":"宝山区","10":"闵行区","11":"嘉定区","12":"金山区","13":"松江区","14":"青浦区", "15":"奉贤区","16":"崇明县"}
        
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        if not os.path.exists(self.picedir):
            os.mkdir(self.picedir)
            
        if (platform.system() == 'Windows') and (not os.path.isfile(self.tesseractConfig)):
            fp = open(self.tesseractConfig,'wb')
            fp.write('tessedit_char_whitelist 0123456789-.%')
            fp.close() 
    

    def reset(self,):
        if self.isTable345:
            self.isTable34 = False
        if self.iterTimes > 0:
            self.iterTimes = 0
      

    def tableRecognizer(self,imageName,y_min_offset = 480):
        if not os.path.isfile(imageName):
            self.reset()
            raise TypeError,'function tableTreeRecognizer need a file name as input'
        self.iterTimes = self.iterTimes + 1
        if self.iterTimes > self.maxIters:
            self.reset()
            raise ValueError,'This image cannot be recognized'

        image = cv2.imread(imageName)

        if self.isTable345:
            tableSettings = self.tableSettings345
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            gray = self.rotate90DegreesToRight(gray)
            # gray = self.rotate90DegreesToRight(gray)
            # gray = self.rotate90DegreesToRight(gray)
            cv2.imwrite(self.path +'gray.jpg',gray)
            image = cv2.imread(self.path + 'gray.jpg')
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            os.remove(self.path + 'gray.jpg')
        else:
            tableSettings = self.tableSettings12
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        myGray = gray.copy()
        GaussianBlur = cv2.GaussianBlur(gray,(5,5),0)
        adaptiveBinary = cv2.adaptiveThreshold(GaussianBlur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        retVal,threshBinary = cv2.threshold(GaussianBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(adaptiveBinary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        contoursAreaList = []
        areaList = []
        wideList = []
        hightList = []
        x_asix = []
        y_asix = []
        max_three_w = []
        for cnt in contours:
            contourArea = cv2.contourArea(cnt)
            [x,y,w,h] = cv2.boundingRect(cnt)
            areaList.append(contourArea)
            wideList.append(w)
            hightList.append(h)
            x_asix.append(x)
            y_asix.append(y)
            if contourArea > tableSettings["minArea"]:
                contoursAreaList.append([x,y,w,h])
            if w > tableSettings["myWeight"]:
                max_three_w.append(w)

        if len(max_three_w) < 3:
            raise IndexError('max_three_w is not three')

        max_three_w.sort()
        max_index = wideList.index(max_three_w[0]) 
        [x,y,w,h] = [x_asix[max_index],y_asix[max_index],wideList[max_index],hightList[max_index]]

        x_min = x - tableSettings["offset"]
        x_max = x + w + tableSettings["offset"]
        y_min = y - tableSettings["offset"]
        y_max = y + h + tableSettings["offset"]
    

        myContours = []
        for [x,y,w,h] in contoursAreaList:
            if x > x_min and x < x_max:
                if y > y_min+y_min_offset and y < y_max:
                    if h<tableSettings["maxHigh"] and h>tableSettings["minHigh"] and w<self.tableSettings12["maxWeight"]:
                        myContours.append([x,y,w,h])
        #print "myContours = ",len(myContours)
        wList = []
        hList = []
        xList = []
        yList = []
        for [x,y,w,h] in myContours:
            xList.append(x)
            yList.append(y)
            wList.append(w)
            hList.append(h)

        xList = self.maopaoSort(xList)
        xLadder = self.getLadder(xList)
        xThresh = []
        for i in xrange(len(xList)-1):
            if xList[i+1]-xList[i]>xLadder:
                xThresh.append((xList[i]+xList[i+1])/2)

        yList = self.maopaoSort(yList)
        yLadder = self.getLadder(yList)
        yThresh = []
        for i in xrange(len(yList)-1):
            if yList[i+1]-yList[i]>yLadder:
                yThresh.append((yList[i]+yList[i+1])/2)

        if len(yThresh) < tableSettings["yRows"]:
            y_min_offset = y_min_offset - 10
            return self.tableRecognizer(imageName,y_min_offset)

        elif len(yThresh) > tableSettings["yRows"]:
            y_min_offset = y_min_offset + 10
            return self.tableRecognizer(imageName,y_min_offset)

        else:
            pass

        xoffset = 2
        yoffset = 2
        for [x,y,w,h] in myContours:
            xlabel = str(self.getLabel(x,xThresh))
            ylabel = str(self.getLabel(y,yThresh))
            pice = threshBinary[y+yoffset:y+h-yoffset,x+xoffset:x+w-xoffset]
            picename = self.picedir +'x' +xlabel+'y'+ylabel+'.jpg'
            cv2.imwrite(picename,pice)
        piceList = os.listdir(self.picedir)
        #print "pice number = ",len(piceList)
        retDict =  self.recAndReturnAsList(self.picedir)  # ret = {"excel":excelname,"data":list}
        piceList = os.listdir(self.picedir)
        for p in piceList:
            os.remove(self.picedir + p)
        retDict["data"] = self.polishRetlist(retDict["data"])
        self.reset()
        return retDict


    def getLabel(self,e,eThresh):
        '''根据给定值和阈值区间找到对应的标签'''  
        if e < min(eThresh):
            return 1
        elif e > max(eThresh):
            return len(eThresh) + 1
        else:
            for i in xrange(1,len(eThresh)):
                if e > eThresh[i-1] and e < eThresh[i]:
                    return i+1

    def maopaoSort(self,oldList):
        length = len(oldList)
        for i in xrange(0,length-1):
            for j in xrange(i,length):
                if oldList[i] > oldList[j]:
                    oldList[i],oldList[j] = oldList[j],oldList[i]
        return oldList

    def getLadder(self,eList):
        retList = []
        for i in xrange(len(eList)-1):
            retList.append(eList[i+1]-eList[i])
        retList.sort()
        for i in xrange(len(retList) - 1):
            if retList[i] >0 and retList[i+1] > 5*retList[i]:
                return (retList[i+1]+retList[i])/2 

    def rotate90DegreesToRight(self,gray):
        if len(gray.shape) != 2:
            raise TypeError,'function rotate90DegreesToRight need image as gray'
        cols,rows = gray.shape
        dst = np.zeros((rows,cols))
        for i in xrange(cols):
            dst[:,cols - 1 -i] = np.transpose(gray[i,:])
        return dst

    def wordRecognize(self,imageName):
        if not os.path.isfile(imageName):
            print "gvie image name is not file "
            return None
        filedir,filename = os.path.split(imageName)
        name,jpg = os.path.splitext(filename)
        outfile = filedir + '/' + name
        if jpg in ['.jpg','.jpeg','.png','.tif']:
            if platform.system() == 'Linux':
                conmand = 'tesseract ' + imageName + ' '+ outfile + ' -psm 6 t_config'
            elif platform.system() == 'Windows':
                conmand = 'tesseract ' + imageName + ' '+ outfile + ' -psm 6 ' + self.tesseractConfig
            else:
                raise TypeError("your system is not Windows nor Linux")
            os.system(conmand)
        fp = open(outfile + '.txt')
        content = fp.readline().strip()
        fp.close()
        os.remove(outfile + '.txt')
        return content

    def recAndReturnAsList(self,picedir):
        '''return retDict = {"excel":excelname,"data":list}'''
        data = {}
        if not picedir.endswith('/'):
            picedir = picedir + '/'
        piceList = os.listdir(picedir)
        if self.isExcel:
            book = xlwt.Workbook(encoding = 'utf-8')
            sheet1 = book.add_sheet('Sheet 1',cell_overwrite_ok = True)
      
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
                ret = self.domain[dy]
            else:            
                ret = self.wordRecognize(picedir + pice)
            if ret.startswith('%') or ret.startswith('.'):
                ret = ret[1:]
            #print "ret in findcontour = ",ret
            if y in data:
                data[y][x] = ret
            else:
                data[y] = {}
                data[y][x] = ret
            if self.isExcel:
                sheet1.write(int(y[1:]),int(x[1:])-1,str(ret))
            os.remove(picedir + pice)
            '''按照序号排序之后返回列表，每个列表元素为一行'''
        retDict = {}
        if self.isExcel:
            now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            excelname = self.path + 'REC' + now + '.xls'
            book.save(excelname)
            print "[INFO] save result in excel as:",excelname
            self.isExcel = False
            retDict["excel"] = excelname
        else:
            retDict["excel"] = ''
        myData = []
        for key in data:
            myData.append(data[key])
        retDict["data"] = myData
        return retDict

    def polishRetlist(self,retList):
        keysList = []
        for l in retList:
            if len(l.keys()) > len(keysList):
                keysList = l.keys()
        for y_dict in retList:
            for x_key in keysList:
                if x_key not in y_dict.keys():
                    y_dict[x_key] = 'error'
        return retList


recognizerObject = MyRecognizer()
