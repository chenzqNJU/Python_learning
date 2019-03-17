#coding=utf-8




len('ifdgsdg')
str(134)
a=[]
a.append(10)
print a
a.append(4)
print a
a.insert(2,3)
print a
a.index(4)
a.remove(3)
print a

a=[1,2,'ff']
a[2]

a='abcdef'
print a[:-1]

import random
print random.random()

[python]
<span style="font-size:18px">
from datetime import datetime
now=datetime.now()
print now.month

<span style="font-size:18px">from datetime import datetime
now = datetime.now()
print str(now.month)+"/"+str(now.day)+"/"+str(now.year)</span>

def using():
    if 1>0:
        return "dfdf"
def using1():
    if 1>0:
        return "dfd"
print using()
print using1()

def square(n):
    squared=n**2
    print "%d squared is %d."%(n,squared)
    return squared
square(10)

from math import *
print sqrt(25)

import math # Imports the math module
everything = dir(math) # Sets everything to a list of things from math
print everything

