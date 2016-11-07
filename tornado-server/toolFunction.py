#!/user/bin/env/python
#coding:utf-8


'''
Functions will be used in main demo and other demo in projects
You should not run this script alone
Every function has its own statements
'''
import json
import datetime

def ZMD_countToName(zmd_count):
    PRESSURE_ORDER = {
        1:    str('出口压力'),
        2:    str('进口压力'),
        4:    str('差压'),
        8:    str('差压2'),
        16:   str('气体泄漏1'),
        32:   str('气体泄漏2'),
        64:   str('气体泄漏3'),
        128:  str('气体泄漏4'),
        256:  str('流量1'),
        512:  str('流量2'),
        1024: str('进口压力2')
        }
    names = []
    for key in PRESSURE_ORDER:
        if (key & zmd_count) != 0:
            name = PRESSURE_ORDER[key].encode('utf-8')
            names.append(PRESSURE_ORDER[key])
    ret = '+'.join(names)
    return ret


def string_to_datetime(string):
    '''
    Transform time string received from app to python datetime.datetime object 
    input string like "2014-12-31T16:00:00.000Z"
    output datetime.datetime object
    '''
    index1 = string.find('T')
    index2 = string.find('Z')
    str1 = string[0:index1]
    str2 = string[index1+1:index2-4]
    form = '%Y-%m-%d %H:%M:%S'
    str3 = str1 + ' ' + str2
    time_tmp = datetime.datetime.strptime(str3,form)
    t = datetime.timedelta(hours = 8)
    result = time_tmp + t
    return result
