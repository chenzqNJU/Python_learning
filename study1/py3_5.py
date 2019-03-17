# -*- coding: UTF-8 –*-
import os
import numpy as np
import pandas as pd

print(os.path.dirname(os.path.realpath('F:\download\南大调数据研\行业改革发展调研基层法人数据表_9农商行')))

print(os.listdir('F:\download\南大调数据研\行业改革发展调研基层法人数据表_9农商行'))
a=os.walk('F:\download\南大调数据研\行业改革发展调研基层法人数据表_9农商行')

def file_path(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root, end=' ')    # 当前目录路径
        print(dirs, end=' ')    # 当前路径下的所有子目录
        print(files)            # 当前目录下的所有非目录子文件

test='F:\download\南大调数据研\行业改革发展调研基层法人数据表_9农商行'
file_path('F:\download\南大调数据研\行业改革发展调研基层法人数据表_9农商行')

sum([len(x) for _, _, x in os.walk(os.path.dirname(test))])

sum([len(x) for _, _, x in os.walk(test)])



def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result

result=all_path(os.path.dirname(test))

for each in result:
    if 'zip' in each:
        a.append(each)

file_name=a[0]
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
        # win下一般使用的是gbk编码
        name = name.decode("gbk")
      #  print(name)
        zip_file.extract(names,path)
        os.rename(path + os.sep + names, path + os.sep + name)
    zip_file.close()
un_zip(a[2])

import rarfile
def un_rar(file_name):
    """unrar zip file"""
    rar = rarfile.RarFile(file_name)
    path = os.path.splitext(file_name)[0]
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
    os.chdir(path)
    for names in rar.namelist():
        #name = names.encode('cp437')
        # win下一般使用的是gbk编码
        #name = name.decode("gbk")
      #  print(name)
        names=rar.namelist()[1]
        rar.extract(names,path)
        os.rename(path + os.sep + names, path + os.sep + name)
    rar.extractall(path)
    rar.close()
from unrar import rarfile
file = rarfile.RarFile(file_name)  #这里写入的是需要解压的文件，别忘了加路径
file.extractall(path)
a2=[]
for each in result:
    if 'rar' in each:
        a2.append(each)
file_name=a2[0]

print(os.path.realpath(file_name))    # 当前文件的路径
print(os.path.dirname(os.path.realpath(file_name)))  # 从当前文件路径中获取目录
print(os.path.basename(os.path.realpath(file_name))) # 从当前文件路径中获取文件名
name, ext = os.path.splitext(os.path.basename(file_name))[0]
os.path.splitext(file_name)[0]

def listzipfilesinfo(path):
  z=zipfile.ZipFile(path,'r')
  try:
    for filename in z.namelist():
      bytes=z.read(filename)
      print('File:%s Size:%s'%(unicode(filename, 'cp936').decode('utf-8'),len(bytes)))
  finally:
    z.close()

for filename in zip_file.namelist():
    bytes=zip_file.read(filename)
    print('File:%s Size:%s'%(unicode(filename, 'cp936').decode('utf-8'),len(bytes)))
for filename in zip_file.namelist():
    name = filename.encode('cp437')
    #win下一般使用的是gbk编码
    name = name.decode("gbk")
    print(name)

z=[]
for each in result:
    if 'S44' in each:
        z.append(each)
len(z)
a=[]

len(z)


z1[10][15:19]
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
z7=z6.reshape(4,221)
z8=np.zeros(221,dtype=np.int)
for i in range(221):
    z8[i]=np.argmax(z7[:,i])
    if z7[1,i]>0:
        z8[i]=1
    if z7[2,i]>0:
        z8[i]=2

shape1=[]
for i in range(221):
    if z[i][-3:] != 'xls':
        continue
    try:
        data= pd.read_excel(z[i])
    except:
        continue
    shape=list(data.shape)
    shape1.append(shape)

z[192]
data = pd.read_excel(z[104])
try:
    data = pd.read_excel(z[192])
except:
    a1=1

final_=final
final_.tail
i=1
final=pd.DataFrame()
for i in range(3):
    if z[i][-3:] != 'xls':
        continue
    try:
        data= pd.read_excel(z[i])
    except:
        continue
    data = pd.read_excel(z[i])
    data['n']=z4[i]
    data['year']=z8[i]
    final=final.append(data,ignore_index=True)


final.shape
np.argmax(z7[:,1])

z6_1=np.array(z5_1)
z6_2=np.array(z5_2)
z6_3=np.array(z5_3)
z6_4=np.array(z5_4)
z7=np.vstack(z6_1,z6_2,z6_3,z6_4)
z7=np.concatenate((z6_1,z6_2,z6_3,z6_4),axis=0)


z7=np.vstack((z6_1,z6_2),axis=0)




t=z1[1].split('\\')
'2015' in t
z1[1].count("2014")








str = 'a1b2c3-)'
print(filter(lambda x: x not in '0123456789', str))

N=range(10)
print(filter(lambda x:x>5,N))
zz=list(filter(lambda x:x>5,N))

string = '127 米'
print (filter(str.isdigit, string))
string.split()[0]

a = "32 个答案"
b = a.split()[0]
print(b)