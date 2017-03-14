#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import csv
import os

csvdir  = './ShanghaiTianqi'
outdir  = csvdir + '_clear' + '/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

csvlist = os.listdir(csvdir)

for f in csvlist:
    csvname = csvdir + '/' + f
    outname = outdir + f
    reader = csv.reader(open(csvname,'rb'))
    fp = open(outname,'wb')
    writer = csv.writer(fp)

    
    i = 0
    rows = []
    for line in reader:
        if i == 0:
        	writer.writerow(line)
        else:
            if line[0].startswith('2'):
            	rows.append(line)
        i = i + 1

    rows.sort()
    writer.writerows(rows)

    fp.close()
    # break 