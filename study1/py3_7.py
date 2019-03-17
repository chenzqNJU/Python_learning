import pandas as pd
import numpy as np
import xlwt

a=pd.Series(['a', 'b', 'c']).str.cat(['A', 'B', 'C'], sep=',')

df2 = pd.DataFrame({'A':1.,
'各个':pd.Timestamp('20130102'),
'的':pd.Series(1, index=list(range(4)),dtype='float32'),
'D':np.array([3] * 4, dtype='int32'),
'E':pd.Categorical(['test','train', 'test','train']),
'F':'foo',
'打':list('顶顶顶顶')
})
df2.loc[:2,['D']]
df2[1]
df2[np.where(df2=='顶')]
df2[[1,2,3]]
df2.ix[[1,2,3]]

[1,2,3,4].__dir__()
'的' in ['的大丰','的']

test = [1,2,3,4,2,2,3,1,4,4,4]

################set key用法
print(max(set(test),key=test.count))

a = [1,4,2,3,2,3,4,2]
from collections import Counter
Counter(a).most_common(2)

import functools
product = functools.reduce((lambda x, y: x * y),  [1, 2, 3, 4])
import operator
operator.xor(60,13)

functools.reduce(operator.xor, [1,2,5,2,1,5,9,2])

t1 = [1,2,3]
t2 =[10,20,30]
dict(zip(t1,t2))[3]

la = [1,2]
lb = [4,5,6]
lc = [7,8,9,10]
list(zip(la,lb,lc))[1]

lt1 = {1,2,3}
l2 = ['a','b','c']
import itertools
for i in itertools.chain(lt1,l2):
    print(i)

test.count(3)
max('ah', 'bh', key=lambda x: x[1])

students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
aa=students[1][2]
set('2355')
list('顶,顶,顶,顶')
a=df2.打
a=df2['打']
a=df2.loc[:,['打']]

s = pd.Series(['的辅导费','大方地色'])
s.isin(['大方地色'])
s[s.str.contains('的')]

'大,范,甘,迪'.split(',')

name = 'HelloWord'
reault = name.center(20,'*')
name = '大富大贵个'
name.count('大')
name.endswith('贵发个')
if name.find('大',3):
    1
'****的辅导费***'.strip('*')
#指定字符分隔
name = 'swht'
li = 'hhsslswhtolljm'
li.partition(name)

z1='dfd\'gdf'
"dfd'fd"
name.__contains__('个')
name.index('个')

''.join(list['123'])
''.join(['1','2'])
''.join([1,2])
'1'.isdigit()

[str(i) for i in [1,2,3]]

s = '<html><body><h1>hello world<h1></body></html>'
start_index = s.find('<h1>')

import  itertools
from itertools import groupby
lst=[2,8,11,25,43,6,9,29,51,66]

def gb(num):
    if num <= 10:
        return 'less'
    elif num >=30:
        return 'great'
    else:
        return 'middle'


print([(k, list(g)) for k, g in groupby(sorted(lst), key=gb)])
print([(k, list(g)) for k, g in groupby(lst, key=gb)])

list['1234']

[i for i in itertools.chain(str(1234),'fefg')]


s = '3a4b5cdd7e'
print([''.join(list(g)) for k, g in groupby(s, key=lambda x: x.isdigit())])

df = pd.DataFrame()
index = ['alpha', 'beta', 'gamma', 'delta', 'eta']
for i in range(5):
    a = pd.DataFrame([np.linspace(i, 5*i, 5)], index=[index[i]])
    df = pd.concat([df, a], axis=0)
df[1]


tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
'foo', 'foo', 'qux', 'qux'],
['one', 'two', 'one', 'two',
'one', 'two', 'one', 'two']]))
tuples[1]
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
index.values
a=np.array([1,2])
t=[[a,b] for a,b in tuples]
a,b=tuples[1]
z=[a,y]
list(a)
a.split('a')
''.join(a.split('a'))

df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])

import openpyxl
df.to_excel('temp.xls')
df.to_excel('temp.xls',sheet_name='Random Data')
df.to_excel('foo.xlsx', sheet_name='Sheet1')
a=pd.read_excel('foo.xlsx')
z=a.index.values
t=[[a,b] for a,b in z]
t.fillna()



z1=list(zip(*z))
list(map(lambda x:'a' if x!=x else x,list(z1[0])))

x1=list(z1[0])[0];x2=[];
for x in list(z1[0]):
    if x!=x:
        x=x1
    x1=x
    x2.append(x)



x!=x  ####用来判断是不是nan
np.isnan('df')
t1=list(z1[0])[1]
t2=np.NaN
t1==t2
t1.isnan()

import math
math.isnan(t2)
np.isnan(t1)
t2.isnan



from collections import Counter

lst = [1, np.nan, 3, 4, np.nan, 5]
lstnan = np.isnan(lst)
lstcounts = Counter(lstnan)
print(lstcounts)
lstnan.sum()

######### if else
a=1;b=2;
a if a>b else b
[b,a][a>b]

a = [1,2,3]
b = [4,5,6]
[(i,j) for i in a for j in b if i%2==0 and j%2==0]

############## 透析表
df = pd.DataFrame({
'A': ['one', 'one', 'two', 'three'] * 3,
'B': np.array(['A', 'B', 'C']).repeat(4),
'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
'D': np.random.randn(12),
'E': np.random.randn(12)
})
pd.pivot_table(df, values=['D','E'], index=['A', 'B'], columns=['C'])
pd.pivot_table(df, values=['D','E'], index=['A', 'B'], columns=['C'])

df=pd.DataFrame(np.arange(6).reshape((2,3)),index=pd.Index(['street1','street2']),columns=pd.Index(['one','two','three']))
df1=df.stack()
df1.unstack()
####### 删除行列 drop默认删除行 pop默认删除列
df.pop('A')
df.drop(range(3,5,1))
df.drop('B',axis=1)
#修改当前工作目录
import os
os.chdir("e:\\bank")
os.getcwd()

#################array
np.array((1,2,3,4,5))# 参数是元组
np.array([6,7,8,9,0])# 参数是list
np.array(list(zip([1,2,3],[4,5,6])))# 参数二维数组

c = np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]],dtype=np.int8)
c.shape # (3L, 4L)
c.shape=4,-1   #c.reshape((2,-1))
c
c[1]
c.reshape((2,-1))

def fun(i):
    return i%4
np.fromfunction(fun,(10,2))

b=np.array([1,2,3]*2+['df','gf']*3)
list(b)

[2,3,4,5,6]>2
c>2

c.flatten(1)

np.repeat(c,2)
np.repeat(c,2,axis=0)
np.repeat(3,4)
np.array([3, 3, 3, 3])
x = np.array([[1, 2], [3, 4]])
np.repeat(x, 2)

np.repeat(x, 2, axis=0)
np.tile(x, [2,2])
np.repeat(x, [1, 2], axis=0)


# append 将不同维度列表变成array
t=[1,2,3]
[t]
np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
np.append([[1, 2, 3], [4, 5, 6]], [t], axis=0)

c+1
c+[1,2,3,4]
c+np.array([1,2,3,4])
t=np.array([t])
c+t.T
# 合并
A = np.array([1, 1, 1])
B = np.array([2, 2, 2])
np.vstack((A, B))
np.hstack((A, B))
np.concatenate((A, B), axis = 1)

t=[1,2]
a = np.array([[1, 2], [3, 4]])
b = np.array([[1,2]])
b = np.array([t])
np.concatenate((a, b), axis=0)
np.concatenate((a, b.T), axis=1)
b.T
np.vstack((a, b))
np.hstack((a, b.T))

####################  zip zip* 行列转化
m = [[1, 2, 3],  [4, 5, 6],  [7, 8, 9]]
n = [[2, 2, 2],  [3, 3, 3],  [4, 4, 4]]
# 矩阵点乘
print('=*'*10 + "矩阵点乘" + '=*'*10)
print([x*y for a, b in zip(m, n) for x, y in zip(a, b)])
z=list(zip(m, n))
list(zip(*m))
list(zip(*[[1,2,3],[2,3,4],[2,3,4]]))
list(zip([1,2,3],[2,3,4]))

m=[1,2,3,4];n=[3,4,5,6];y='abcd';
list(zip(m,n,y))

#####################################正则表达式
import re

key = r"<html><body><h4>hello world<h1></body></html>"#这段是你要匹配的文本
p1 = r"(?<=<h[1-6]>).+?(?=<h1>)"#这是我们写的正则表达式规则，你现在可以不理解啥意思
pattern1 = re.compile(p1)#我们在编译这段正则表达式
matcher1 = re.search(pattern1,key)#在源文本中搜索符合正则表达式的部分
print(matcher1.group(0))

key = r"<html><body><h4>hello world<h4></body></html>"
p1 = r"<h([1-6])>.*?<h\1>"#这是我们写的正则表达式规则，你现在可以不理解啥意思
pattern1 = re.compile(p1)#我们在编译这段正则表达式
matcher1 = re.search(pattern1,key)#在源文本中搜索符合正则表达式的部分
print(matcher1.group(0))


key = r"<h1>hello world</h1>"
p1 = r"<h([1-6])>.*?</h\1>"
pattern1 = re.compile(p1)
m1 = re.search(pattern1,key)
print(m1.group(0))

key = r"<h1>hello 得到world的</h1><h1>hello 得到worl的d的</h1>"
p1 = r"<h([1-6])>.*?的(?=</h\1>)"
pattern1 = re.compile(p1)
m1 = re.search(pattern1,key)
print(m1.group(0))

s='3a4b5cd-的d7e'
z2=re.findall(r'[0-9]+|\w+',s)
z2=re.findall(r'\w+',s)

['3', 'a', '4', 'b', '5', 'cdd', '7', 'e']

key=r"放的发的的的发额的的额他"
p1 = r"的+.?的+"
pattern1 = re.compile(p1)
m1 = re.search(pattern1,key)
print(m1.group(0))

key = r"s的的s and sas and saaas"
p1 = r"s的{1,2}s|sa+s"
pattern1 = re.compile(p1)
print(pattern1.findall(key))

key = r"的发的的"
p1 = r"的.*?的的"
pattern1 = re.compile(p1)
m1 = re.search(pattern1,key)
print(m1.group(0))

######提取数字

totalCount = '100abc'
re.sub("\D","d", totalCount)


inputStr='hello 123 nihao 2324'
re.sub("\d+", "222", inputStr)

inputStr = "hello 的, nihao 的";
re.sub(r"hello (\w+), nihao \1", "\g<1>", inputStr);



#递推

f = lambda x,y,n:x if not n else f(y,x+y,n-1)
list(map(lambda n:f(1,1,n),range(10)))


[x[0] for x in [ (a[i][0], a.append((a[i][1], a[i][0]+a[i][1]))) for a in ([[1,1]], ) for i in range(10) ]]

print ([x[0] for x in [ (a[i][0], a.append((a[i][1], a[i][0]+a[i][1]))) for a in ([[1,1]], ) for i in xrange(100) ]])

f=[1,1]
[f.append(f[-1]+f[-2]) or f.pop(0) for i in range(10)]
[i-1 or i*2 for i in range(10)]
from functools import reduce
reduce(lambda f,i:f.append((f[-2]+f[-1])) or f,range(10),[1,1])

a=lambda f,i:f.append((f[-2]+f[-1])) or 1
a([1,1,3],1)
f=[1,2,3]
if (not f.append((f[-2]+f[-1]))): print('dd')

type(f.append(1))

type(f.pop(0))

if f:
    print('dd')

b if a else c
((a and b) or c)

1 if 2 else 3
2 and 1 or 3