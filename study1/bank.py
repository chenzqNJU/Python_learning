import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## 目录下所有文件
def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result

test='F:\download\南大调数据研\行业改革发展调研基层法人数据表_9农商行'
result=all_path(os.path.dirname(test))

## 解压文件 zip rar
import zipfile
def un_zip(file_name):
    """unzip zip file"""
    zip_file = zipfile.ZipFile(file_name)

    path=os.path.splitext(file_name)[0]
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
    for names in zip_file.namelist():
        name = names.encode('cp437')
        name = name.decode("gbk")
        zip_file.extract(names,path)
        os.rename(path + os.sep + names, path + os.sep + name)
    zip_file.close()
yasuo1=[]
for each in result:
    if '.zip' in each:
        yasuo1.append(each)
        un_zip(each)

from unrar import rarfile

yasuo2=[]
for each in result:
    if '.rar' in each:
        yasuo2.append(each)
        file = rarfile.RarFile(each)
        path = os.path.splitext(each)[0]
        file.extractall(path)

###########################################################
result1=all_path(os.path.dirname(test))
len(result)
z=[]
for each in result:
    if 'S44' in each:
        z.append(each)
length=len(z)

z1=[x[19:] for x in z]
z2=[x[15:19] for x in z1]
z4=[]
for i in range(len(z2)):
    z3=list(filter(lambda x:x in '0123456789',z2[i]))
    t=int("".join([str(i) for i in z3]))
    z4.append(t)

z5_1=[x.count('2014') for x in z1]
z5_2=[x.count('2015') for x in z1]
z5_3=[x.count('2016') for x in z1]
z5_4=[x.count('2017') for x in z1]
z5=z5_1+z5_2+z5_3+z5_4
z6=np.array(z5)
z7=z6.reshape(4,length)
z8=np.zeros(length,dtype=np.int)
for i in range(length):
    z8[i]=np.argmax(z7[:,i])
    if z7[1,i]>0:
        z8[i]=1
    if z7[2,i]>0:
        z8[i]=2

shape1=[]
for i in range(length):
    if z[i][-3:] != 'xls':
        continue
    try:
        data= pd.read_excel(z[i])
    except:
        continue
    shape=list(data.shape)
    shape1.append(shape)
len(shape1)

final=pd.DataFrame()
for i in range(length):
    if z[i][-3:] != 'xls':
        continue
    try:
        data= pd.read_excel(z[i])
    except:
        continue
    data = pd.read_excel(z[i])
    data1=data.iloc[:,:4]
    data1['n']=z4[i]
    data1['year']=z8[i]
    final=final.append(data1,ignore_index=True)

################################################################
z=[]
for each in result:
    if 'S41' in each:
        z.append(each)
length=len(z)

z1=[x[19:] for x in z]
z2=[x[15:19] for x in z1]
z4=[]
for i in range(len(z2)):
    z3=list(filter(lambda x:x in '0123456789',z2[i]))
    t=int("".join([str(i) for i in z3]))
    z4.append(t)

z5_1=[x.count('2014') for x in z1]
z5_2=[x.count('2015') for x in z1]
z5_3=[x.count('2016') for x in z1]
z5_4=[x.count('2017') for x in z1]
z5=z5_1+z5_2+z5_3+z5_4
z6=np.array(z5)
z7=z6.reshape(4,length)
z8=np.zeros(length,dtype=np.int)
for i in range(length):
    z8[i]=np.argmax(z7[:,i])
    if z7[1,i]>0:
        z8[i]=1
    if z7[2,i]>0:
        z8[i]=2

shape1=[]
for i in range(length):
    if z[i][-3:] != 'xls':
        continue
    try:
        data= pd.read_excel(z[i])
    except:
        continue
    shape=list(data.shape)
    shape1.append(shape)
len(shape1)

final=pd.DataFrame()
for i in range(length):
    if z[i][-3:] != 'xls':
        continue
    try:
        data= pd.read_excel(z[i])
    except:
        continue
    data = pd.read_excel(z[i])
    data1=data.iloc[:,:9]
    data1['n']=z4[i]
    data1['year']=z8[i]
    final=final.append(data1,ignore_index=True)

##############################################################
result1=all_path(os.path.dirname(test))
len(result)
z=[]
for each in result:
    if 'G0102' in each or '贷款质量五级分类' in each:
        z.append(each)
length=len(z)

z1=[x[19:] for x in z]
z2=[x[15:19] for x in z1]
z4=[]
for i in range(len(z2)):
    z3=list(filter(lambda x:x in '0123456789',z2[i]))
    t=int("".join([str(i) for i in z3]))
    z4.append(t)

z5_1=[x.count('2014') for x in z1]
z5_2=[x.count('2015') for x in z1]
z5_3=[x.count('2016') for x in z1]
z5_4=[x.count('2017') for x in z1]
z5=z5_1+z5_2+z5_3+z5_4
z6=np.array(z5)
z7=z6.reshape(4,length)
z8=np.zeros(length,dtype=np.int)
for i in range(length):
    z8[i]=np.argmax(z7[:,i])
    if z7[1,i]>0:
        z8[i]=1
    if z7[2,i]>0:
        z8[i]=2

shape1=[]
for i in range(length):
    if z[i][-3:] != 'xls':
        continue
    try:
        data= pd.read_excel(z[i])
    except:
        continue
    shape=list(data.shape)
    shape1.append(shape)

len(shape1)
d= pd.read_excel(z[7])
d1= pd.read_excel(z[50])
d2= pd.read_excel(z[80])
int(np.where(d2=='本外币合计')[1])

#由观察可知，有的excel仅本外币合计一项，有的包括本币、外币，这里仅提取本外币列
y=[]
for i in range(length):
    if z[i][-3:] != 'xls':
        continue
    try:
        data= pd.read_excel(z[i])
    except:
        continue
    #if np.where(data.iloc[:,:5] == '本外币合计')
    y.append(int(np.where(data.iloc[:,:5] == '本外币合计')[1]))


final=pd.DataFrame()
for i in range(length):
    if z[i][-3:] != 'xls':
        continue
    try:
        data= pd.read_excel(z[i])
    except:
        continue
    y_=int(np.where(data.iloc[:,:5] == '本外币合计')[1])
    data1=data.iloc[:,[0,1,y_]]
    data1.columns = list('abc')
    data1['n']=z4[i]
    data1['year']=z8[i]
    final=final.append(data1,ignore_index=True)

del data1

final.to_csv('daikuan.csv',encoding='GBK')

final.to_excel('data.xlsx')
os.getcwd()

###################根据行业改革发展文档确定文件夹对应的银行
z=[]
for each in result:
    if '改革发展' in os.path.basename(each):
        z.append(each)
length=len(z)

z1=[x[19:] for x in z]
z2=[x[15:19] for x in z1]
z4=[]
for i in range(len(z2)):
    z3=list(filter(lambda x:x in '0123456789',z2[i]))
    t=int("".join([str(i) for i in z3]))
    z4.append(t)

final=pd.DataFrame()
for i in range(length):
    try:
        data= pd.read_excel(z[50])
    except:
        continue
    data1=data.iloc[:10,[0]]
    data1.columns = ['a']
    data1['n']=z4[i]
    final=final.append(data1,ignore_index=True)
final.to_csv('mingcheng.csv',encoding='GBK')

del data
len(final)


import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)

data=pd.read_excel("E:\\bank\\qipao.xlsx")

data1=data[np.where(data.a2014>3)]
data1=data[list(data.a2014!='.') and list(data.a2015!='.') and \
           list(data.a2016 != '.') and list(data.a2017 != '.')]
data1=data[list(data.a2014!='.')]
data2=data1[list(data1.a2016!='.')]
data=data2[list(data2.a2017!='.')]

a=list(data.a2014=='.') and list(data.a2015=='.')
data[a]
z=data.a2014
type(z[1])
float('.')
data.loc[data.columns[0]]
data.columns[0]

final=pd.DataFrame()
for i in range(1,data.shape[1]):
    t=pd.DataFrame()
    t=data.iloc[:,[0,i]]
    t.columns=list('ab')
    t['year']=np.array([10*i*1] * data.shape[0])
    t['x']=range(1,data.shape[0]+1)
    final = final.append(t, ignore_index=True)
a=final.head()
a1=a.values
a1[:1]
a.x
type(a.x)
a[['x','year']]
a.iloc[:2,:2]
a.ix[:2,['x','year']]
a['df']=np.array(range(5))
a2=a.ix[1,2]
a.ix[1,3]='f'
a.ix[2,2]=np.nan
a.ix[1,0]=34
type(a.ix[1,3])
a3=a.year.reindex(a.index, fill_value = 0)


final1=final[:]
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
plt.axis([0,60,0,40])
plt.grid(True)
ax.scatter(final1['x'], final1['year'], s=list(final1['b']))  # 第三个变量表明根据收入气泡的大小
s = ax.set_xticks(range(data.shape[0]))
x_labels =list(data.银行.values)
s = ax.set_xticklabels(x_labels, minor=False,fontproperties=font_set, fontsize=10, rotation=45)
plt.show()
plt.close()



s=final['b']
type(s[0])
df = s[1].astype('float32')
np.array(s[1]).sqrt()

[1,2,3].sqrt()
int('None')

np.sqrt([1,2,3,None])