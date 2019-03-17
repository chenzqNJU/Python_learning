import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
import tushare as ts
import os
import gc
os.chdir("e:\\temp")

import myfunc
myfunc.search('filter',)

#以读入文件为例：
f = open("shiti.txt")#二进制格式读文件
a=[]
lines = f.readlines()

for line in lines:
    a.append(line)


b=pd.DataFrame({'a':a})

c=b.a.str.contains('考试')
c1=np.cumsum(c)
t1=np.where(c)

b['c']=c1


c=b.a.str.contains('(参考答案)|(真题（卷二）答案)')
t2=np.where(c)

t3=np.hstack((t1[0],t2[0]))
t3.sort()
t3=np.unique(t3)

t4=np.hstack((np.arange(t3[1],t3[2]),np.arange(t3[4],t3[5]),np.arange(t3[6],len(b))))
t5=t4.tolist()

daan=b.iloc[t5,:]
z=daan[daan.c<4].a.str.split('\d+')
z1=daan[daan.c==4].a.str.split('（\d+）')
z2=z.append(z1)

daan['d']=z2

c=daan.a.str.contains('题')
t=daan[c].index[1:].tolist()+[daan.index[-1]]


import operator
from functools import reduce
temp=pd.DataFrame()
for x in range(len(t)-1):
    a=daan.loc[t[x]+1:t[x+1]-1,:].d.values.tolist()
    a=list(filter(lambda x:len(x)>1,a))
    a1=reduce(operator.add, a)
    a2=list(filter(lambda x:x!='',a1))
    a3=pd.DataFrame({'e':a2})
    a3.e=a3.e.str.replace('(\.)|(\\n)|( )|(．)|(;)','')
    a3['tihao']=range(1,1+len(a2))
    a3['tixing']=np.mod(x,3)+1
    a3['juanzi']=int(np.floor(x/3))+1
    temp=temp.append(a3)

b=b[b.c!=2]

z=list(set(b.index)-set(t5))
b1=b.loc[z]

c=b1.a.str.contains('题')

c1=b1[c][b1[c].a.str.len()<24]
c1['flag']=1
c1=c1.flag
b2=pd.concat([b1,c1],axis=1)
b2=b2.fillna(0)

b2.flag=np.cumsum(b2.flag.values)
b2['tixing'] = np.mod(b2.flag-1, 3) + 1
b2['juanzi'] = (np.floor((b2.flag-1) / 3))+1

b2['tihao'] = b2.a.str.lstrip().str.slice(0,2).str.extract('(\d+)')
b2=b2.fillna(0)
b2.tihao=b2.tihao.astype(int)

c=pd.merge(b2,temp,how='left',on=['juanzi','tixing','tihao'])
c=c[['a','e']]
c=c.fillna('')
c['Value'] = c.apply(lambda row: row['a']+row['e'], axis=1)
c=c.Value.str.replace('\\n','')

c=c.to_frame()
c.to_csv('final.csv',index=False)

help(c.to_csv)