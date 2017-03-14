#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import os
from matplotlib import pyplot as plt
import pandas as pd

csvdir  = './ShanghaiTianqi' + '_clear' + '/'
csvlist = os.listdir(csvdir)
imgdir = './image/'
if not os.path.exists(imgdir):
    os.mkdir(imgdir)

start = 0      # plot start index
end   = 300    # plot end  index

for f in csvlist:
    csvfile = csvdir + f
    print("csvfile = ",csvfile)
    area = (f.split('.')[0]).split('_')[0]
    df = pd.read_csv(csvfile)
    highest = df['最高气温'][start:end]
    date    = df['日期'][start:end]
    lowest  = df['最低气温'] [start:end]


    plt.plot(highest, 'ro')
    plt.plot(lowest,'bo')
    xindex = range(len(highest))
    gate = 20
    xlist =[gate*i for i in range(len(highest)/gate)]
    xlist.append(xindex[-1])
    # print("xlist = ",xlist)
    xticks  = []
    xlabels = []
    for i in xlist:
        xticks.append(xindex[i])
        xlabels.append(date[i])
    plt.plot(highest, 'r-')
    plt.plot(lowest,'b-')
    plt.xlabel('date')
    plt.ylabel('Temperature')
    plt.title('Shanghai Temperature of ' + area)
    plt.legend(['highest','lowest'])
    plt.grid(True)
    plt.xticks(xticks,xlabels,rotation = 90)
    plt.savefig(imgdir + area + '.jpg')
    plt.show()

    
    # break