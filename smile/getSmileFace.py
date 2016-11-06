#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time

personName = "simle"
outdir = './images'
# storeDir = '../paperImage/getFace/'
iCount = 0
classfier = cv2.CascadeClassifier('./haarcascade_smile.xml')


cap = cv2.VideoCapture(0)

print "start taking face photo..."

while(iCount < 20):
    flag = cap.isOpened()
    if not flag:
    	cap.open()
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # print "gray shape = ",gray.shape
    cv2.equalizeHist(gray)     #灰度图像进行直方图等距化
    gray = cv2.GaussianBlur(gray,(5,5),0)
    faces = classfier.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        face = faces[0]
    	# print "get a good face image ----------",iCount
     #    if iCount < 10:
    	#     imageName = outdir + '/' + personName + '0' + str(iCount) + '.jpg'
     #    else:
     #        imageName = outdir + '/' + personName + str(iCount) + '.jpg'
    	# cv2.imwrite(imageName,gray) 
        (x,y,w,h) = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)   	 
    	# iCount = iCount + 1
        cv2.imshow('frame',frame)
        # if iCount < 10:
        #     imageName1 = storeDir + '/' + personName + '0' + str(iCount) + '.jpg'
        # else:
        #     imageName1 = storeDir + '/' + personName + str(iCount) + '.jpg'
        # cv2.imwrite(imageName1,frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        # time.sleep(1)
	
# Release everything if job is finished
cap.release()

cv2.destroyAllWindows()

print "It is okay!"