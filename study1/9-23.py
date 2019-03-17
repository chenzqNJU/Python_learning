__author__ = 'chenzq'

s = 'This is a string. \
This continues the string.'
print s

i = 5
print 'Value is', i # Error! Notice a single space at
the start of the line
print 'I repeat, the value is', i

a=pd.Series([1,2,'3.0得到'])
a[0];a[2]
a=a.astype(str)


a.str.extract('([\u4E00-\u9FA5]+)')

a.str.isnumeric()

'1.2'.isdecimal()

help(pd.Series.str.extract)
import pandas as pd
b=pd.read_sas('e:\\stock\\final.sas7bdat',encoding='gbk')
a.iloc[:,1]=a.iloc[:,1].str.decode('gbk')
a.iloc[0,0].decode('gbk')


help(pd.read_sas)
b=a[:3]
#!/usr/bin/python
# Filename: expression.py
length = 5
breadth = 2
area = length * breadth
print 'Area is', area
print 'Perimeter is', 2 * (length + breadth)

#!/usr/bin/python
# Filename: if.py
number = 23;guess=33;
guess = int(raw_input('Enter an integer : '))
if guess == number:
    print 'Congratulations, you guessed it.' # New block starts here
    print "(but you do not win any prizes!)" # New block ends here
elif guess < number:
    print 'No, it is a little higher than that' #Another block
# You can do whatever you want in a block ...
else:
    print 'No, it is a little lower than that'
# you must have guess > number to reach here
    print 'Done'
# This last statement is always executed, after the if statement is executed

if True:
    print 'Yes, it is true'


#!/usr/bin/python
# Filename: while.py
number = 23
running = True
while running:
    guess = int(raw_input('Enter an integer : '))
    if guess == number:
        print 'Congratulations, you guessed it.'
        running = False #'''this causes the while loop to stop'''
    elif guess < number:
        print 'No, it is a little higher than that'
    else:
        print 'No, it is a little lower than that'
else:
    print 'The while loop is over.'
# Do anything else you want to do here
print 'Done'


for i in range(0, 4):
    print i
else:
    print 'The for loop is over'

print range(1,2)

while True:
    s=raw_input('enter something:')
    if s=='quit':
        break
    print 'length of the string is',len(s)
print 'done'

def sayhello():
    print 'hello world'

sayhello()

def printMax(a,b):
    if a>b:
        print a
    else:
        print b
printMax(3,4)

x=5;y=7;
printMax(x,y)

def func():
    global x
    print x
    x=2
    print x
x=50
func()
print x

print 'df'*3

def func(a,b=4,c=10):
    print a,b,c

func(1)
func(1,2)
func(1,2,2)
func(c=10,a=1)

func.__doc__

import sys
for i in sys.argv:
    print i

print '\n\nThe PYTHONPATH is', sys.path, '\n'

a=5

shoplist=['apple','mango','carrot','banana']
print len(shoplist)
for i in shoplist:
    print i

shoplist.append('risce')
print shoplist

shoplist.sort()
print shoplist

print shoplist[0]
del shoplist[0]
print shoplist

zoo=('wolf','elephant','penguin')
print len(zoo)
new_zoo=('monkey','dolphin',zoo)
print len(new_zoo)
print new_zoo
new_zoo[2][2][2]

new_zoo[2][2].__class__
age=22;name='swaroop';
print name,'is',age,'years old'
print '%s is %d years old' %(name,age)
ab={'swaroop':'swaroopch@byteofpython.info',
    'larry':'larry@wall.org',
    'matsumoto':'matz@ruby-lang.org',
    'spammer':'spammer@hotmail.com'}
print ab
ab['swaroop']
ab['guido']='guido@oython.org'
ab['guido']
del ab['guido']
len(ab)
print '\nThere are %d contacts in the address-book\n' % len(ab)
ab.items()
for name, address in ab.items():
    print 'Contact %s at %s' % (name, address)

if ab.has_key('spammer'):
    print "spammer's address is %s"%ab['spammer']
help(dict)

shoplist[1:3]
shoplist[:-1]

name='Swaroop'
print name.startswith('Sw')
print  name.find('S')

import os
import time
source = ['/home/swaroop/byte', '/home/swaroop/bin']
target_dir = '/mnt/e/backup/'

time.strftime('%Y%m%d%H%M%S')
target = target_dir + time.strftime('%Y%m%d%H%M%S') +'.zip'

zip_command = "zip -qr '%s' %s" % (target, ''.join(source))
print  zip_command

import numpy
import scipy
import pandas

from urllib import urlopen
data = urlopen('http://peak.telecommunity.com/dist/ez_setup.py')
with open('ez_setup.py', 'wb') as f:
f.write(data.read())


help('modules sys')
import numpy
import numpy as np
a = np.arange(10)
print(a)
e=np.zeros((2,3))
print e
import pandas as pd
s = pd.Series([1,3,5,np.nan,6,8])
print s

import scipy
import platform
platform.architecture()
import numpy
numpy.__version__
$ pip list
import matplotlib
sys.modules.keys()

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats as stats
import scipy.stats as stats
from scipy import signal as t
timeStep=0.02
sp.__version__
sp.fft


import math
math.sqrt(9)

np.version.full_version
a=np.arange(20)
print a
np.zeros((4,5),dtype=int)
a=np.random.rand(5)
np.exp(a)
np.max(a)
c=np.linspace(0,2,9)
loc=numpy.where(c==2)
print loc
b=a.copy()
print c
import scipy.stats as stats
from pandas import Series, DataFrame
s = Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
print s
s.index
d = {'a': 0., 'b': 1, 'c': 2}
print "d is a dict:"
print d
s = Series(d)
print s
Series(d, index=['b', 'c', 'd', 'a'])


s = Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'], name='my_series')
print s
print s.name
d = {'one': Series([1., 2., 3.], index=['a', 'b', 'c']), 'two': Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df = DataFrame(d)
print df
df = DataFrame(d, index=['r', 'd', 'a'], columns=['two', 'three'])
print df
a = Series(range(5))
b = Series(np.linspace(4, 20, 5),dtype=int)
df = pd.concat([a, b], axis=0)
print df



from urllib import urlopen
data = urlopen('http://peak.telecommunity.com/dist/ez_setup.py')
with open('ez_setup.py', 'wb') as f:
f.write(data.read())
import os
file_name = 'ez_setup.py'
from urllib import urlopen
data = urlopen('http://peak.telecommunity.com/dist/ez_setup.py')
with open(file_name, 'wb') as f:
f.write(data.read())
os.system('python %s' % (os.path.join(os.getcwd(),file_name)))


for i in range(0,5):
    for j in range(0,5):
        print i,j

import numpy as np
import pandas as pd
x = pd.read_csv("E:\\break.csv")
print x

import os
mxdPath=r"E:\\break.csv"
fpa=open(mxdPath)
indexx=0
for linea in fpa.readlines():
    indexx=indexx+1
    linea=linea.replace("\n","")
    print linea
    if indexx ==5:
        break
fpa.close()

print fpa

import xlrd
data = xlrd.open_workbook("e:\\python\\ETF_menu.xlsx")
table = data.sheets()[0]             #通过索引顺序获取
table = data.sheet_by_index(0)       #通过索引顺序获取
table = data.sheet_by_name(u'Sheet') #通过名称获取

a=table.row_values(2)
table.col_values(2)

sh = data.sheet_by_name("Wind")
print data
print data.nrows
print table


import pandas as pd
data = pd.read_csv("e:\\python\\break.csv")
data #查看表格

data.columns #查看表格有哪些列，可以看到有x, y, z列
Index([u'name'], dtype='object')
>>> data['x'] #查看x列

Name: x, dtype: int64
>>> data['y'] #查看y列
0 2
1 4
2 8
3 16
4 32
Name: y, dtype: int64
data['z'] #查看z列

Name: z, dtype: object
>>> import matplotlib.pyplot as plt
>>> plt.bar(data['x'], data['y']) #画柱状图
<Container object of 5 artists>
>>> plt.title('example') #设置标题
>>> plt.xlabel('x') #横坐标加说明文字'x'
<matplotlib.text.Text object at 0x110e06110>
>>> plt.ylabel('y') #纵坐标加说明文字'y'
<matplotlib.text.Text object at 0x10e80d890>
>>> plt.show() #显示图形


