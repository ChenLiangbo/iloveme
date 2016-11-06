#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np

order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume*e-6"}
order ={"0":"Open","1":"High","2":"Low","3":"Close","4":"Volume"}
yahooData = np.load('yahoo_finance5.npy')
shape = yahooData.shape
# print "yahooData shape = ",shape
# print yahooData[0:10]
# print "----------------------------------------------------------------"
Open,High,Low,Close,Volume = np.hsplit(yahooData,5)

related = 5

xSample = np.zeros((shape[0],20),np.float32)


# 0-5:open,high,low,close,volume
for x in xrange(shape[0]):
    for y in xrange(shape[1]):
        xSample[x,y]   = yahooData[x,y]     #origin

# 6 -10: X(n) - X(n-1),diff open,high,low,close,volume
diff0 = np.zeros((shape[0],5),np.float32)
for x in xrange(shape[0]):
    for y in xrange(shape[1]):
        if x == 0:
            # xSample[x,2*y] =  yahooData[1,y] - yahooData[0,y]
            diff0[x,y] = yahooData[1,y] - yahooData[0,y]
        else:
            # xSample[x,2*y] = yahooData[x,y] - yahooData[x-1,y]
            diff0[x,y] = yahooData[x,y] - yahooData[x-1,y]

# 10-15:X(n) - X(n-1) // open,high,low,close,volume    
diff2 = np.zeros((shape[0],5),np.float32)
for x in xrange(shape[0]):
    for y in xrange(shape[1]):
        if x == 0:
            # xSample[x,3*y] =  xSample[1,2*y] - xSample[0,2*y]
            diff2[x,y] = diff0[1,y] - diff0[0,y]
        else:
            # xSample[x,3*y] = xSample[x,2*y] - xSample[x-1,2*y]
            diff2[x,y] = diff0[x,y] - diff0[x-1,y]


# 16-20 X(n)-X(n-2) //open,high,low,close,volume    
diff3 = np.zeros((shape[0],5),np.float32)
for x in xrange(shape[0]):
    for y in xrange(shape[1]):
        if x < 2:
            # xSample[x,4*y] = xSample[2,y] - xSample[0,y]
            diff3[x,y] = yahooData[2,y] - yahooData[0,y]
        else:
            # xSample[x,4*y] = xSample[x,y] - xSample[x-2,y]
            diff3[x,y] = yahooData[x,y] - yahooData[x-1,y]



'''
 d - d_min
--------------  * (r_max - r_min)  + r_min
d_max - d_min
'''

#normalize for xsample
r_min = np.min(yahooData[:,3])
r_max = np.max(yahooData[:,3])

print "r_min = %f,r_max = %f" % (r_min,r_max)

d_min = np.min(diff0)
d_max = np.max(diff0)
diff0 = (diff0 - d_min)*(r_max - r_min) / (d_max - d_min) + d_min


d_min = np.min(diff3)
d_max = np.max(diff3)
diff3 = (diff3 - d_min)*(r_max - r_min) / (d_max - d_min) + d_min

d_min = np.min(diff2)
d_max = np.max(diff2)
diff2 = (diff2 - d_min)*(r_max - r_min) / (d_max - d_min) + d_min


xSample = np.hstack([yahooData,diff0,diff2,diff3])
print "xSample.shape = ",xSample.shape
# print xSample[0:2]
print "------------------------------------------------------------------------"
np.save('./npyfile_saver/xSample20',xSample)

# ySample = np.zeros((shape[0],1),dtype = np.float32)
# # close up 1,down 0
# for i in xrange(shape[0] - 1):
#     if yahooData[i +1,3] - yahooData[i,3] > 0:
#         ySample[i,0] = 1

ySample = Close
print "ySample.shape = ",ySample.shape
np.save('./npyfile_saver/ySample20',ySample)
# #

x = np.zeros((shape[0]-related,related*20),dtype = np.float32)

for i in xrange(shape[0]-related):
    x[i] = xSample[i:i+related,:].reshape(1,related*20) 

np.save('./npyfile_saver/xSample100',x)
print "x.shape = ",x.shape
# print x[0:2]

# ySample = np.zeros((shape[0] - related,1),dtype = np.float32)

# for i in xrange(shape[0] - related):
#     if yahooData[i+related,3] - yahooData[i+related-1,3] < 0:
#         ySample[i,0] = 0
#     else:
#         ySample[i,0] = 1
# print "ySample.shape = ",ySample.shape
# np.save('./npyfile/ySample100',ySample)

ySample = Close[related:,:]
print "ySample.shape = ",ySample.shape
np.save('./npyfile_saver/ySample20',ySample)