# -*- coding: utf-8 -*-

import csv
import numpy as np
import pandas as pd
import xlrd
a=np.mat([1,2,3,4,5,6,6,7,8])
b=a.reshape(3,3)
b=np.mat('1 2 3;4 5 3')
b.I
b1=np.array(b)
np.linalg.inv(b)
b.I

z=pd.read_csv('e:/python/stockprice1.csv')
z.index[1:3]
z1=z.iloc[0:3,0:3]
z5=z['002505.SZ'].values
z2=z5.copy()
int(z2[1000]+1)
z2_=np.array(z2,dtype=int)
z2[np.isones(z2)]=None
t=z.iloc[:,1:].values
t1=t.flatten(1)
date=z[z.columns[0]].values
date1=np.tile(date,3349)
code=z.columns.tolist()
code1=np.array(code[1:])
code2=code1.repeat(2573)

final=pd.DataFrame({'date':date1,'price':t1,'code':code2})
final.to_csv('e:/python/stockprice_c.csv')


final.head(12)
date1.shape
t1.shape
code2.shape

final.dtypes

code2[:3]
code2.shape
type(date[2])
z1=date[:3]
z1.repeat(2)
np.tile(z1,2)

date1.shape
z=t1[:3]
z[1]
np.float(z)
round(z[1],3)

t1[:3]
type(t1[1])
type(z2_[1])
zz=np.array([None,None,1,1])
int(np.NaN)
type(np.NaN)
a=[1,2,"dd"]
b=[1,2]
a1=np.mat(a)
t=np.array(b)[1]
np.array(b)
np.dtype(z2)
np.dtype
z4=a1[0,0]
a1.ndim

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5],dtype=float)
type(xarr[1])


book = ["Python", "Development", 8]

t=z3.tolist()
z.shape
z.columns[1:5]
z.index
z[:2]
z.iloc[:,1]
f=0
z1=z.iloc[:,1].values
t=z1.astype(int)
z2=np.mat(z1)
z3=np.asmatrix(z1)

z=pd.read_csv('e:/python/break.csv')
t=pd.read_excel('e:/python/ETF_menu目录.xlsx')
t.columns
t["证券代码"]

xlsx = pd.ExcelFile('e:/python/ETF_menu.xlsx')
df = pd.read_excel(xlsx, 'Sheet1')














help(pd.read_csv)
pd.read_csv
csv_reader = csv.reader(open('e:/python/break.csv', encoding='utf-8'))
for row in csv_reader:
    f+= 1
    if f > 3:
        break
    print(row)
print(csv_reader)

csv_file = csv.file(r'e:/python/break.csv', 'rb')
reader = csv.reader(csv_file)
for line in reader:
    # the type of line is list
    print
    line
csvfile.close()

import csv

with open(r'e:/python/data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    f=1
    for row in readCSV:
        f+=1
        if f>3:
            break
        print(row)

fr = open('e:\\python\\data.csv','r')   #读写模式：r-只读；r+读写；w-新建（会覆盖原有文件）;更多信息见参考文章2
all_utf8 = fr.read()
print (all_utf8)
all_uni = all_utf8.decode("utf-8")
all_uni = unicode(all_utf8, 'utf-8')

from itertools import islice
input_file = open("e:/python/data.csv")
for line in islice(input_file, 1, None):
    print(line)


import csv
with open('e:/python/data.csv','rb') as csvfile:
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
    print (rows)


with open('e:/python/data.csv', 'rb') as csvfile1:
    reader = csv.reader(csvfile1)
    column = [row[2] for row in reader]
print (column)

reader = csv.reader(open("e:/python/data.csv"))
for title, year in reader:
   print (year)
for line in open("e:/python/data.csv"):
   title, year= line.split(",")
   print (year)


with open("e:/python/data.csv", 'r') as f_in:
    with open("e:/python/data_.csv", 'w') as f_out:
        for line in f_in:
            f_out.write(line.split(',')[1] + '\n')

'dfdfdf,dfdf'.split(',')[1]

# -*- coding: utf-8 -*-
from numpy import *

a1 = [[1, 2, 3], [4, 5, 6]]  # 列表
print('a1 :', a1)
# ('a1 :', [[1, 2, 3], [4, 5, 6]])

a2 = array(a1)  # 列表 -----> 数组
print('a2 :', a2)
# ('a2 :', array([[1, 2, 3],[4, 5, 6]]))

a3 = mat(a1)  # 列表 ----> 矩阵
print('a3 :', a3)
# ('a3 :', matrix([[1, 2, 3],[4, 5, 6]]))

a4 = a3.tolist()  # 矩阵 ---> 列表
print('a4 :', a4)
# ('a4 :', [[1, 2, 3], [4, 5, 6]])

print(a1 == a4)
# True

a5 = a2.tolist()  # 数组 ---> 列表

print('a5 :', a5)
# ('a5 :', [[1, 2, 3], [4, 5, 6]])
print(a5 == a1)
# True

a6 = mat(a2)  # 数组 ---> 矩阵
print('a6 :', a6)
# ('a6 :', matrix([[1, 2, 3],[4, 5, 6]]))

print(a6 == a3)
# [[ True  True  True][ True  True  True]]

a7 = array(a3)  # 矩阵 ---> 数组
print('a7 :', a7)
# ('a7 :', array([[1, 2, 3],[4, 5, 6]]))
print(a7 == a2)
# [[ True  True  True][ True  True  True]]

###################################################################
a1 = [1, 2, 3, 4, 5, 6]  # 列表
print('a1 :', a1)
# ('a1 :', [1, 2, 3, 4, 5, 6])

a2 = array(a1)  # 列表 -----> 数组
print('a2 :', a2)
# ('a2 :', array([1, 2, 3, 4, 5, 6]))

a3 = mat(a1)  # 列表 ----> 矩阵
print('a3 :', a3)
# ('a3 :', matrix([[1, 2, 3, 4, 5, 6]]))

a4 = a3.tolist()  # 矩阵 ---> 列表
print('a4 :', a4)
# ('a4 :', [[1, 2, 3, 4, 5, 6]])   #注意！！有不同
print(a1 == a4)
# False

a8 = a3.tolist()[0]  # 矩阵 ---> 列表
print('a8 :', a8)
# ('a8 :', [1, 2, 3, 4, 5, 6])  #注意！！有不同
print(a1 == a8)
# True

a5 = a2.tolist()  # 数组 ---> 列表
print('a5 :', a5)
# ('a5 :', [1, 2, 3, 4, 5, 6])
print(a5 == a1)
# True

a6 = mat(a2)  # 数组 ---> 矩阵
print('a6 :', a6)
# ('a6 :', matrix([[1, 2, 3, 4, 5, 6]]))

print(a6 == a3)
# [[ True  True  True  True  True  True]]

a7 = array(a3)  # 矩阵 ---> 数组
print('a7 :', a7)
# ('a7 :', array([[1, 2, 3, 4, 5, 6]]))
print(a7 == a2)
# [[ True  True  True  True  True  True]]

