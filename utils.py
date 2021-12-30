import numpy as np
from collections import OrderedDict

def convert2Bool(x):
    if x != x:
        return False
    elif x == 'Y' or x == 'Yes' or x == True:
        return True
    else:
        return False

def convert2Gender(x):
    if x != x:
        return 'M'
    elif x == '男' or x == 'M':
        return 'M'
    else:
        return 'F'

def convert2Diagnosises(x):
    strings = x.split(';')
    d = list(map(lambda str: str.split(':')[1].upper().split('.')[0], strings)) # return the ID of diagnosis
    return list(OrderedDict.fromkeys(d))


