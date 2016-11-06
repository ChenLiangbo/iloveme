#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np

def perms(elements):
    if len(elements) <=1:
        yield elements
    else:
        for perm in perms(elements[1:]):
            for i in range(len(elements)):
                yield perm[:i] + elements[0:1] + perm[i:]

orderList = list(perms([0,1, 2, 3,4]))
print "orderList = ",len(orderList)
for item in list(perms([0,1, 2, 3,4])):
    print item